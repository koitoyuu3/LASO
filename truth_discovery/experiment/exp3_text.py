from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truth_discovery.experiment.exp_utils import (
    TEXT_METHODS,
    build_phase_switched_df,
    eval_text_metrics_per_batch,
    load_batches,
    plot_temporal_metric,
    to_text_df,
)

DEFAULT_DATA_DIR = ROOT / "truth_discovery" / "data" / "data_agent-50_news-300"
DEFAULT_SCHEDULE_TAG = "10bad_at_50__15bad_at_100__20bad_at_200"
OUTPUTS_BASE = (
    ROOT / "truth_discovery" / "examples" / "outputs" / "exp3_text"
)
DEFAULT_OUT_DIR = OUTPUTS_BASE / DEFAULT_DATA_DIR.name / DEFAULT_SCHEDULE_TAG

def _default_out_dir(data_dir: Path) -> Path:
    return OUTPUTS_BASE / data_dir.name / DEFAULT_SCHEDULE_TAG
TITLE = "E3-Text: Progressive Byzantine Source Mutiny"
N_AGENTS_50 = 50

def save_results(
    accuracy_results: Dict[str, Dict[int, float]],
    hybrid_score_results: Dict[str, Dict[int, float]],
    out_dir: Path,
    dataset_name: str,
    phase1_start: int,
    phase2_start: int,
    phase3_start: int,
    phase1_bad_count: int,
    phase2_bad_count: int,
    phase3_bad_count: int,
    accuracy_threshold: float,
) -> Path:
    rows = []
    for method in TEXT_METHODS:
        acc_series = accuracy_results.get(method, {})
        score_series = hybrid_score_results.get(method, {})
        all_batches = sorted(set(acc_series) | set(score_series))
        for batch_idx in all_batches:
            rows.append(
                {
                    "batch_idx": batch_idx,
                    "method": method,
                    "data_accuracy": acc_series.get(batch_idx, float("nan")),
                    "mean_hybrid_score": score_series.get(batch_idx, float("nan")),
                    "dataset": dataset_name,
                    "attack_profile": "collude",
                    "phase1_start_batch": phase1_start,
                    "phase2_start_batch": phase2_start,
                    "phase3_start_batch": phase3_start,
                    "phase1_bad_count": phase1_bad_count,
                    "phase2_bad_count": phase2_bad_count,
                    "phase3_bad_count": phase3_bad_count,
                    "accuracy_threshold": accuracy_threshold,
                }
            )
    path = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path

def _validate_inputs(
    phase1_start: int,
    phase2_start: int,
    phase3_start: int,
    phase1_bad_count: int,
    phase2_bad_count: int,
    phase3_bad_count: int,
    n_batches: int,
) -> None:
    if phase1_start < 1:
        raise ValueError(f"phase1_start must be >= 1, got {phase1_start}")
    if phase2_start <= phase1_start:
        raise ValueError(
            f"phase2_start must be > phase1_start, got phase1_start={phase1_start}, phase2_start={phase2_start}"
        )
    if phase3_start <= phase2_start:
        raise ValueError(
            f"phase3_start must be > phase2_start, got phase2_start={phase2_start}, phase3_start={phase3_start}"
        )
    if phase3_start > n_batches:
        raise ValueError(f"phase3_start={phase3_start} exceeds available batches ({n_batches})")
    if phase1_bad_count < 0 or phase2_bad_count < 0:
        raise ValueError("bad_count must be non-negative")
    if phase3_bad_count < 0:
        raise ValueError("phase3_bad_count must be non-negative")
    if phase2_bad_count < phase1_bad_count:
        raise ValueError("phase2_bad_count must be >= phase1_bad_count")
    if phase3_bad_count < phase2_bad_count:
        raise ValueError("phase3_bad_count must be >= phase2_bad_count")
    if phase3_bad_count >= N_AGENTS_50:
        raise ValueError(f"phase3_bad_count must be < {N_AGENTS_50}")

def build_file_map(
    data_dir: Path,
    phase1_bad_count: int,
    phase2_bad_count: int,
    phase3_bad_count: int,
) -> Dict[int, Path]:
    files = {
        0: data_dir / "demo.json",
        phase1_bad_count: data_dir / f"demo_{phase1_bad_count}bad_collude.json",
        phase2_bad_count: data_dir / f"demo_{phase2_bad_count}bad_collude.json",
        phase3_bad_count: data_dir / f"demo_{phase3_bad_count}bad_collude.json",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing text experiment files:\n" + "\n".join(missing))
    return files

def run_experiment(
    data_dir: Path,
    out_dir: Path,
    phase1_start: int,
    phase2_start: int,
    phase3_start: int,
    phase1_bad_count: int,
    phase2_bad_count: int,
    phase3_bad_count: int,
    accuracy_threshold: float = 0.90,
) -> None:
    files = build_file_map(data_dir, phase1_bad_count, phase2_bad_count, phase3_bad_count)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_txt_batches = load_batches(files[0])
    phase1_txt_batches = load_batches(files[phase1_bad_count])
    phase2_txt_batches = load_batches(files[phase2_bad_count])
    phase3_txt_batches = load_batches(files[phase3_bad_count])

    n_batches = min(
        len(clean_txt_batches),
        len(phase1_txt_batches),
        len(phase2_txt_batches),
        len(phase3_txt_batches),
    )
    _validate_inputs(
        phase1_start,
        phase2_start,
        phase3_start,
        phase1_bad_count,
        phase2_bad_count,
        phase3_bad_count,
        n_batches,
    )

    print("\n" + "=" * 60)
    print(TITLE)
    print("=" * 60)
    print(
        "dataset = text:data_agent-50_news-300/demo_*bad_collude.json, "
        f"schedule = 0 bad -> {phase1_bad_count} bad -> {phase2_bad_count} bad -> {phase3_bad_count} bad"
    )

    clean_txt_df = to_text_df(clean_txt_batches)
    phased_txt_df = build_phase_switched_df(
        [
            (1, clean_txt_batches),
            (phase1_start, phase1_txt_batches),
            (phase2_start, phase2_txt_batches),
            (phase3_start, phase3_txt_batches),
        ],
        n_total=n_batches,
        mode="text",
    )
    accuracy_results, hybrid_score_results = eval_text_metrics_per_batch(
        clean_txt_df,
        phased_txt_df,
        accuracy_threshold=accuracy_threshold,
    )

    event_lines = [
        (phase1_start, f"{phase1_bad_count} bad ({phase1_bad_count / N_AGENTS_50:.0%})"),
        (phase2_start, f"{phase2_bad_count} bad ({phase2_bad_count / N_AGENTS_50:.0%})"),
        (phase3_start, f"{phase3_bad_count} bad ({phase3_bad_count / N_AGENTS_50:.0%})"),
    ]
    mixed_path = out_dir / "mixed_text.png"
    accuracy_path = out_dir / "mixed_text_accuracy.png"
    plot_temporal_metric(
        hybrid_score_results,
        TEXT_METHODS,
        ylabel="Mean Hybrid Score",
        title=TITLE,
        out_path=mixed_path,
        event_lines=event_lines,
        y_lim=(0.0, 1.05),
    )
    plot_temporal_metric(
        accuracy_results,
        TEXT_METHODS,
        ylabel=f"Data Accuracy (hybrid ≥ {accuracy_threshold:g})",
        title=f"{TITLE} — Thresholded Accuracy (≥{accuracy_threshold:g})",
        out_path=accuracy_path,
        event_lines=event_lines,
        y_lim=(0.0, 1.05),
    )
    results_path = save_results(
        accuracy_results,
        hybrid_score_results,
        out_dir,
        dataset_name=data_dir.name,
        phase1_start=phase1_start,
        phase2_start=phase2_start,
        phase3_start=phase3_start,
        phase1_bad_count=phase1_bad_count,
        phase2_bad_count=phase2_bad_count,
        phase3_bad_count=phase3_bad_count,
        accuracy_threshold=accuracy_threshold,
    )

    print(f"saved {mixed_path}")
    print(f"saved {accuracy_path}")
    print(f"saved {results_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run 50-agent text exp3 with staged malicious activation.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/exp3_text/<data-dir-name>/<schedule>, derived from --data-dir.",
    )
    parser.add_argument("--phase1-start", type=int, default=50)
    parser.add_argument("--phase2-start", type=int, default=100)
    parser.add_argument("--phase3-start", type=int, default=200)
    parser.add_argument("--phase1-bad-count", type=int, default=10)
    parser.add_argument("--phase2-bad-count", type=int, default=15)
    parser.add_argument("--phase3-bad-count", type=int, default=20)
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.90,
        help="Hybrid score threshold for DataAccuracy (default 0.90).",
    )
    args = parser.parse_args()

    if args.out_dir is None:
        out_dir = _default_out_dir(args.data_dir)
        if args.accuracy_threshold != 0.90:
            out_dir = out_dir.parent / f"{out_dir.name}__thr{args.accuracy_threshold:.2f}"
    else:
        out_dir = args.out_dir

    run_experiment(
        data_dir=args.data_dir,
        out_dir=out_dir,
        phase1_start=args.phase1_start,
        phase2_start=args.phase2_start,
        phase3_start=args.phase3_start,
        phase1_bad_count=args.phase1_bad_count,
        phase2_bad_count=args.phase2_bad_count,
        phase3_bad_count=args.phase3_bad_count,
        accuracy_threshold=args.accuracy_threshold,
    )

if __name__ == "__main__":
    main()
