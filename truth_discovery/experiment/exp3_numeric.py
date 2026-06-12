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
    NUM_METHODS,
    build_gt_numeric,
    build_phase_switched_df,
    eval_numeric_per_batch,
    load_batches,
    plot_temporal_metric,
)

DEFAULT_DATA_DIR = ROOT / "truth_discovery" / "data" / "data_agent-50_news-300"
DEFAULT_OUT_DIR = (
    ROOT
    / "truth_discovery"
    / "examples"
    / "outputs"
    / "exp3_numeric"
    / "data_agent-50_news-300"
    / "10bad_at_50__15bad_at_100__20bad_at_200"
)
TITLE = "E3-Numeric: Progressive Byzantine Source Mutiny"
N_AGENTS_50 = 50

def save_results(
    per_batch_results: Dict[str, Dict[int, float]],
    out_dir: Path,
    phase1_start: int,
    phase2_start: int,
    phase3_start: int,
    phase1_bad_count: int,
    phase2_bad_count: int,
    phase3_bad_count: int,
) -> Path:
    rows = []
    for method in NUM_METHODS:
        series = per_batch_results.get(method, {})
        for batch_idx, relative_rmse in sorted(series.items()):
            rows.append(
                {
                    "batch_idx": batch_idx,
                    "method": method,
                    "relative_rmse": relative_rmse,
                    "dataset": "data_agent-50_news-300",
                    "attack_profile": "collude",
                    "phase1_start_batch": phase1_start,
                    "phase2_start_batch": phase2_start,
                    "phase3_start_batch": phase3_start,
                    "phase1_bad_count": phase1_bad_count,
                    "phase2_bad_count": phase2_bad_count,
                    "phase3_bad_count": phase3_bad_count,
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
    if phase1_bad_count < 0 or phase2_bad_count < 0 or phase3_bad_count < 0:
        raise ValueError("bad_count must be non-negative")
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
        0: data_dir / "num_demo.json",
        phase1_bad_count: data_dir / f"num_demo_{phase1_bad_count}bad_collude.json",
        phase2_bad_count: data_dir / f"num_demo_{phase2_bad_count}bad_collude.json",
        phase3_bad_count: data_dir / f"num_demo_{phase3_bad_count}bad_collude.json",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing numeric experiment files:\n" + "\n".join(missing))
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
) -> None:
    files = build_file_map(data_dir, phase1_bad_count, phase2_bad_count, phase3_bad_count)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_num_batches = load_batches(files[0])
    phase1_num_batches = load_batches(files[phase1_bad_count])
    phase2_num_batches = load_batches(files[phase2_bad_count])
    phase3_num_batches = load_batches(files[phase3_bad_count])

    n_batches = min(
        len(clean_num_batches),
        len(phase1_num_batches),
        len(phase2_num_batches),
        len(phase3_num_batches),
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
        "dataset = numeric:data_agent-50_news-300/num_demo_*bad_collude.json, "
        f"schedule = 0 bad -> {phase1_bad_count} bad -> {phase2_bad_count} bad -> {phase3_bad_count} bad"
    )

    gt_num = build_gt_numeric(clean_num_batches)
    phased_num_df = build_phase_switched_df(
        [
            (1, clean_num_batches),
            (phase1_start, phase1_num_batches),
            (phase2_start, phase2_num_batches),
            (phase3_start, phase3_num_batches),
        ],
        n_total=n_batches,
        mode="numeric",
    )
    per_batch_results = eval_numeric_per_batch(phased_num_df, gt_num, normalize=True)

    event_lines = [
        (phase1_start, f"{phase1_bad_count} bad ({phase1_bad_count / N_AGENTS_50:.0%})"),
        (phase2_start, f"{phase2_bad_count} bad ({phase2_bad_count / N_AGENTS_50:.0%})"),
        (phase3_start, f"{phase3_bad_count} bad ({phase3_bad_count / N_AGENTS_50:.0%})"),
    ]
    numeric_path = out_dir / "numeric.png"
    plot_temporal_metric(
        per_batch_results,
        NUM_METHODS,
        ylabel="Relative RMSE",
        title=TITLE,
        out_path=numeric_path,
        event_lines=event_lines,
    )
    results_path = save_results(
        per_batch_results,
        out_dir,
        phase1_start=phase1_start,
        phase2_start=phase2_start,
        phase3_start=phase3_start,
        phase1_bad_count=phase1_bad_count,
        phase2_bad_count=phase2_bad_count,
        phase3_bad_count=phase3_bad_count,
    )

    print(f"saved {numeric_path}")
    print(f"saved {results_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run 50-agent numeric exp3 with staged malicious activation.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--phase1-start", type=int, default=50)
    parser.add_argument("--phase2-start", type=int, default=100)
    parser.add_argument("--phase3-start", type=int, default=200)
    parser.add_argument("--phase1-bad-count", type=int, default=10)
    parser.add_argument("--phase2-bad-count", type=int, default=15)
    parser.add_argument("--phase3-bad-count", type=int, default=20)
    args = parser.parse_args()
    run_experiment(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        phase1_start=args.phase1_start,
        phase2_start=args.phase2_start,
        phase3_start=args.phase3_start,
        phase1_bad_count=args.phase1_bad_count,
        phase2_bad_count=args.phase2_bad_count,
        phase3_bad_count=args.phase3_bad_count,
    )

if __name__ == "__main__":
    main()
