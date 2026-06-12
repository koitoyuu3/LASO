from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/truthfinder_mplconfig")

from truth_discovery.core.zk_proof import digest_object, verify_proof_package
from truth_discovery.experiment.five_methods_comparison import (
    DEFAULT_NUMERIC_INPUT,
    DEFAULT_TEXT_INPUT,
    NUMERIC_METHOD_ORDER,
    TEXT_METHOD_ORDER,
    generate_numeric_proof_bundles,
    generate_text_proof_bundles,
)

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "proof_scalability"
DEFAULT_BATCH_COUNTS = (5, 25, 50, 100, 150, 300)
DEFAULT_REPEATS = 30
DEFAULT_WARMUP = 1

@dataclass(frozen=True)
class WorkerVerificationResult:
    verified: bool
    proof_group_count: int
    result_digest_ok: bool
    experiment_digest_ok: bool
    verify_elapsed_sec: float
    baseline_rss_mb: float
    peak_rss_mb: float
    peak_delta_rss_mb: float

def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"Expected integer >= 1, got {value!r}")
    return parsed

def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected float > 0, got {value!r}")
    return parsed

def _parse_batch_counts(raw_value: str) -> List[int]:
    counts = sorted({_positive_int(token.strip()) for token in raw_value.split(",") if token.strip()})
    if not counts:
        raise ValueError("batch_counts cannot be empty")
    return counts

def _current_rss_mb() -> float:
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return max(float(rss) / (1024.0 * 1024.0), 0.0)
        return max(float(rss) / 1024.0, 0.0)
    except Exception:
        return float("nan")

class _PeakRssSampler:
    def __init__(self, interval_sec: float):
        self.interval_sec = max(float(interval_sec), 0.01)
        self.baseline_rss_mb = float("nan")
        self.peak_rss_mb = float("nan")
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.baseline_rss_mb = _current_rss_mb()
        self.peak_rss_mb = self.baseline_rss_mb
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            current_rss_mb = _current_rss_mb()
            if not pd.isna(current_rss_mb):
                if pd.isna(self.peak_rss_mb):
                    self.peak_rss_mb = current_rss_mb
                else:
                    self.peak_rss_mb = max(self.peak_rss_mb, current_rss_mb)
            time.sleep(self.interval_sec)

    def stop(self) -> tuple[float, float, float]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_sec * 2.0, 0.1))
        final_rss_mb = _current_rss_mb()
        if not pd.isna(final_rss_mb):
            if pd.isna(self.peak_rss_mb):
                self.peak_rss_mb = final_rss_mb
            else:
                self.peak_rss_mb = max(self.peak_rss_mb, final_rss_mb)
        peak_delta_rss_mb = float("nan")
        if not pd.isna(self.baseline_rss_mb) and not pd.isna(self.peak_rss_mb):
            peak_delta_rss_mb = max(float(self.peak_rss_mb) - float(self.baseline_rss_mb), 0.0)
        return self.baseline_rss_mb, self.peak_rss_mb, peak_delta_rss_mb

def _json_size_bytes(value: Any) -> int:
    if value is None:
        return 0
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))

def _recompute_experiment_digest(bundle_payload: Mapping[str, Any]) -> str:
    proof_groups = bundle_payload.get("proof_groups")
    if not isinstance(proof_groups, list):
        raise ValueError("bundle_payload.proof_groups must be a list")
    group_digest_entries = [
        {
            "proof_id": group.get("proof_id"),
            "group_digest": group.get("group_digest"),
            "result_digest": group.get("result_digest"),
        }
        for group in proof_groups
    ]
    return digest_object(
        {
            "experiment_id": str(bundle_payload.get("experiment_id")),
            "method": str(bundle_payload.get("method")),
            "result_digest": bundle_payload.get("result_digest"),
            "proof_groups": group_digest_entries,
        }
    )

def _verify_bundle_payload(bundle_payload: Mapping[str, Any]) -> Dict[str, Any]:
    proof_groups = bundle_payload.get("proof_groups")
    if not isinstance(proof_groups, list) or not proof_groups:
        raise ValueError("bundle_payload.proof_groups must be a non-empty list")

    top_rows = bundle_payload.get("rows")
    result_digest_ok = True
    if isinstance(top_rows, list):
        result_digest_ok = digest_object(top_rows) == str(bundle_payload.get("result_digest"))

    group_results: List[bool] = []
    for group in proof_groups:
        proof_payload = group.get("proof_payload")
        proof_ok = verify_proof_package(proof_payload) if isinstance(proof_payload, Mapping) else False

        group_rows = group.get("rows")
        rows_ok = True
        if isinstance(group_rows, list):
            rows_ok = digest_object(group_rows) == str(group.get("result_digest"))
        group_results.append(bool(proof_ok and rows_ok))

    experiment_digest_ok = _recompute_experiment_digest(bundle_payload) == str(bundle_payload.get("experiment_digest"))
    return {
        "verified": bool(all(group_results) and result_digest_ok and experiment_digest_ok),
        "proof_group_count": int(len(proof_groups)),
        "result_digest_ok": bool(result_digest_ok),
        "experiment_digest_ok": bool(experiment_digest_ok),
    }

def _worker_verify_bundle(bundle_path: Path, memory_sample_interval: float) -> WorkerVerificationResult:

    with bundle_path.open("rb") as file_obj:
        file_obj.read(1)

    sampler = _PeakRssSampler(memory_sample_interval)
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    sampler.start()
    started_at = time.perf_counter()
    try:
        with bundle_path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        verification = _verify_bundle_payload(payload)
        verify_elapsed_sec = time.perf_counter() - started_at
    finally:
        baseline_rss_mb, peak_rss_mb, peak_delta_rss_mb = sampler.stop()
        if gc_was_enabled:
            gc.enable()
    return WorkerVerificationResult(
        verified=bool(verification["verified"]),
        proof_group_count=int(verification["proof_group_count"]),
        result_digest_ok=bool(verification["result_digest_ok"]),
        experiment_digest_ok=bool(verification["experiment_digest_ok"]),
        verify_elapsed_sec=float(verify_elapsed_sec),
        baseline_rss_mb=float(baseline_rss_mb),
        peak_rss_mb=float(peak_rss_mb),
        peak_delta_rss_mb=float(peak_delta_rss_mb),
    )

def _invoke_verify_worker(
    *,
    script_path: Path,
    bundle_path: Path,
    memory_sample_interval: float,
) -> WorkerVerificationResult:
    command = [
        sys.executable,
        str(script_path),
        "--worker-verify-bundle",
        str(bundle_path),
        "--memory-sample-interval",
        str(memory_sample_interval),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    stdout_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or not stdout_lines:
        raise RuntimeError(
            f"Verify worker failed for {bundle_path} with code {completed.returncode}. "
            f"stdout_tail={stdout_lines[-1] if stdout_lines else ''!r}; stderr_tail={completed.stderr.strip()[-500:]!r}"
        )
    payload = json.loads(stdout_lines[-1])
    return WorkerVerificationResult(**payload)

def _measure_bundle(
    *,
    script_path: Path,
    domain: str,
    batch_count: int,
    method_name: str,
    bundle_path: Path,
    repeats: int,
    memory_sample_interval: float,
    warmup: int = 0,
) -> List[Dict[str, Any]]:
    with bundle_path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    proof_groups = payload.get("proof_groups", [])
    first_group = proof_groups[0] if proof_groups else {}
    proof_payload_bytes = _json_size_bytes(first_group.get("proof_payload"))
    bundle_size_bytes = bundle_path.stat().st_size
    row_count = int(payload.get("row_count", 0))
    proof_group_count = int(len(proof_groups))

    for _ in range(max(int(warmup), 0)):
        _invoke_verify_worker(
            script_path=script_path,
            bundle_path=bundle_path,
            memory_sample_interval=memory_sample_interval,
        )

    rows: List[Dict[str, Any]] = []
    for repeat_index in range(1, repeats + 1):
        worker_result = _invoke_verify_worker(
            script_path=script_path,
            bundle_path=bundle_path,
            memory_sample_interval=memory_sample_interval,
        )
        rows.append(
            {
                "domain": domain,
                "batch_count": int(batch_count),
                "method": method_name,
                "bundle_path": str(bundle_path.resolve()),
                "row_count": row_count,
                "proof_group_count": proof_group_count,
                "proof_scheme": first_group.get("proof_scheme"),
                "proof_backend": first_group.get("proof_backend"),
                "proof_size_bytes": int(bundle_size_bytes),
                "proof_payload_bytes": int(proof_payload_bytes),
                "repeat_index": int(repeat_index),
                **asdict(worker_result),
            }
        )
    return rows

def _build_summary_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        raise RuntimeError("No benchmark rows were produced.")

    return (
        raw_df.groupby(["domain", "method", "batch_count"], sort=False)
        .agg(
            proof_bundle_path=("bundle_path", "first"),
            row_count=("row_count", "first"),
            proof_group_count=("proof_group_count", "first"),
            proof_scheme=("proof_scheme", "first"),
            proof_backend=("proof_backend", "first"),
            proof_size_bytes=("proof_size_bytes", "mean"),
            proof_payload_bytes=("proof_payload_bytes", "mean"),
            verify_elapsed_sec_mean=("verify_elapsed_sec", "mean"),
            verify_elapsed_sec_std=("verify_elapsed_sec", "std"),
            verify_peak_rss_mb_mean=("peak_rss_mb", "mean"),
            verify_peak_delta_rss_mb_mean=("peak_delta_rss_mb", "mean"),
            verify_peak_delta_rss_mb_std=("peak_delta_rss_mb", "std"),
            verified_all=("verified", "all"),
            result_digest_ok_all=("result_digest_ok", "all"),
            experiment_digest_ok_all=("experiment_digest_ok", "all"),
        )
        .reset_index()
    )

def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark proof scalability by sweeping batch counts and measuring "
            "proof bundle size, verification memory, and verification time."
        )
    )
    parser.add_argument("--numeric-input", type=Path, default=DEFAULT_NUMERIC_INPUT)
    parser.add_argument("--text-input", type=Path, default=DEFAULT_TEXT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--batch-counts",
        default=",".join(str(value) for value in DEFAULT_BATCH_COUNTS),
        help="Comma-separated batch counts to benchmark.",
    )
    parser.add_argument(
        "--domains",
        default="numeric,text",
        help="Comma-separated domains to benchmark. Supported: numeric,text",
    )
    parser.add_argument("--repeats", type=_positive_int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Number of warmup verify runs to discard per bundle (must be >= 0).",
    )
    parser.add_argument("--memory-sample-interval", type=_positive_float, default=0.01)
    parser.add_argument("--worker-verify-bundle", type=Path, default=None)
    return parser

def _run_worker_mode(args: argparse.Namespace) -> int:
    result = _worker_verify_bundle(args.worker_verify_bundle, args.memory_sample_interval)
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0 if result.verified else 1

def _run_benchmark_mode(args: argparse.Namespace) -> int:
    batch_counts = _parse_batch_counts(args.batch_counts)
    requested_domains = [token.strip() for token in args.domains.split(",") if token.strip()]
    unsupported_domains = [domain for domain in requested_domains if domain not in {"numeric", "text"}]
    if unsupported_domains:
        raise ValueError(f"Unsupported domains: {unsupported_domains}")
    warmup = max(int(args.warmup), 0)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()

    raw_rows: List[Dict[str, Any]] = []

    print("=" * 108)
    print("Proof scalability benchmark")
    print("=" * 108)
    print(f"Numeric input: {Path(args.numeric_input).resolve()}")
    print(f"Text input:    {Path(args.text_input).resolve()}")
    print(f"Output root:   {output_root}")
    print(f"Batch counts:  {batch_counts}")
    print(f"Domains:       {requested_domains}")
    print(f"Repeats:       {args.repeats}")
    print(f"Warmup:        {warmup}")

    for batch_count in batch_counts:
        bundle_output_dir = output_root / f"bundles_batch_{batch_count}"
        print(f"\n[batch_count={batch_count}] generating proof bundles...")

        if "numeric" in requested_domains:
            numeric_artifacts = generate_numeric_proof_bundles(
                input_path=Path(args.numeric_input),
                output_dir=bundle_output_dir,
                max_batches=batch_count,
                methods=NUMERIC_METHOD_ORDER,
            )
            for artifact in numeric_artifacts:
                print(f"  measuring {artifact.method_name} [numeric]")
                raw_rows.extend(
                    _measure_bundle(
                        script_path=script_path,
                        domain="numeric",
                        batch_count=batch_count,
                        method_name=artifact.method_name,
                        bundle_path=Path(artifact.output_path),
                        repeats=args.repeats,
                        memory_sample_interval=args.memory_sample_interval,
                        warmup=warmup,
                    )
                )

        if "text" in requested_domains:
            text_artifacts = generate_text_proof_bundles(
                input_path=Path(args.text_input),
                output_dir=bundle_output_dir,
                max_batches=batch_count,
                methods=TEXT_METHOD_ORDER,
            )
            for artifact in text_artifacts:
                print(f"  measuring {artifact.method_name} [text]")
                raw_rows.extend(
                    _measure_bundle(
                        script_path=script_path,
                        domain="text",
                        batch_count=batch_count,
                        method_name=artifact.method_name,
                        bundle_path=Path(artifact.output_path),
                        repeats=args.repeats,
                        memory_sample_interval=args.memory_sample_interval,
                        warmup=warmup,
                    )
                )

    raw_df = pd.DataFrame(raw_rows)
    summary_df = _build_summary_df(raw_df)

    raw_csv_path = output_root / "raw_results.csv"
    summary_csv_path = output_root / "summary.csv"
    manifest_json_path = output_root / "manifest.json"

    raw_df.to_csv(raw_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    with manifest_json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "numeric_input": str(Path(args.numeric_input).resolve()),
                "text_input": str(Path(args.text_input).resolve()),
                "output_root": str(output_root),
                "batch_counts": batch_counts,
                "domains": requested_domains,
                "repeats": int(args.repeats),
                "warmup": int(warmup),
                "memory_sample_interval": float(args.memory_sample_interval),
                "artifacts": {
                    "raw_results_csv": str(raw_csv_path),
                    "summary_csv": str(summary_csv_path),
                },
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSummary sample:")
    display_columns = [
        "domain",
        "method",
        "batch_count",
        "proof_size_bytes",
        "verify_peak_delta_rss_mb_mean",
        "verify_elapsed_sec_mean",
        "verified_all",
    ]
    print(summary_df[display_columns].to_string(index=False))
    print(f"\nSaved raw results: {raw_csv_path}")
    print(f"Saved summary:     {summary_csv_path}")
    return 0

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)
    if args.worker_verify_bundle is not None:
        return _run_worker_mode(args)
    return _run_benchmark_mode(args)

if __name__ == "__main__":
    raise SystemExit(main())
