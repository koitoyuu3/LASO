from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truth_discovery.experiment.exp_utils import (
    NUM_METHODS,
    build_gt_numeric,
    eval_numeric,
    load_batches,
    plot_numeric,
    to_numeric_df,
)

DEFAULT_DATA_DIR = ROOT / "truth_discovery" / "data" / "data_agent-50_news-300"
DEFAULT_OUT_DIR = (
    ROOT
    / "truth_discovery"
    / "examples"
    / "outputs"
    / "exp4_numeric"
    / "data_agent-50_news-300"
    / "bad_5"
    / "metric_relative"
)
BAD_COUNT = 5
AGENT_RANGE = [15, 20, 25, 30, 35, 40, 45, 50]
TITLE = "E4-Numeric: Varying agent count"
X_LABEL = "Total number of agents"
N_AGENTS_50 = 50

def save_results(
    agent_range: List[int],
    num_results: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
    metric_scale: str,
) -> Path:
    rows = []
    for i, agent_count in enumerate(agent_range):
        for method in NUM_METHODS:
            rows.append(
                {
                    "agent_count": agent_count,
                    "method": method,
                    "rmse": num_results[method]["rmse"][i] if i < len(num_results[method]["rmse"]) else float("nan"),
                    "mae": num_results[method]["mae"][i] if i < len(num_results[method]["mae"]) else float("nan"),
                    "metric_scale": metric_scale,
                    "dataset": "data_agent-50_news-300",
                    "bad_count": BAD_COUNT,
                }
            )
    path = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path

def run_experiment(data_dir: Path, out_dir: Path, metric_scale: str) -> None:
    clean_path = data_dir / "num_demo.json"
    noisy_path = data_dir / f"num_demo_{BAD_COUNT}bad_collude.json"
    for path in [clean_path, noisy_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    out_dir.mkdir(parents=True, exist_ok=True)
    normalize_metric = metric_scale == "relative"

    clean_num_batches = load_batches(clean_path)
    mal_num_batches = load_batches(noisy_path)
    gt_num = build_gt_numeric(clean_num_batches)

    all_agents = [f"agent-{i}" for i in range(1, N_AGENTS_50 + 1)]
    num_results = {method: {"rmse": [], "mae": []} for method in NUM_METHODS}

    print("\n" + "=" * 60)
    print(TITLE)
    print("=" * 60)
    print(
        "dataset = numeric:data_agent-50_news-300/num_demo_*bad_collude.json, "
        f"bad_count = {BAD_COUNT}, metric_scale = {metric_scale}"
    )

    for n_agents in AGENT_RANGE:
        subset = all_agents[:n_agents]
        ratio = BAD_COUNT / n_agents * 100
        print(f"\n  n_agents={n_agents}  malicious ratio≈{ratio:.1f}%")

        noisy_num_df = to_numeric_df(mal_num_batches, agent_subset=subset)
        res_num = eval_numeric(noisy_num_df, gt_num, normalize=normalize_metric)
        for method in NUM_METHODS:
            num_results[method]["rmse"].append(res_num[method]["rmse"])
            num_results[method]["mae"].append(res_num[method]["mae"])

    plot_numeric(
        AGENT_RANGE,
        num_results,
        X_LABEL,
        TITLE,
        out_dir,
        metric_scope_label=(
            "last 10% batches, relative error normalized by |ground truth|"
            if normalize_metric
            else "last 10% batches"
        ),
        metric_name_prefix="Normalized " if normalize_metric else "",
    )
    results_path = save_results(AGENT_RANGE, num_results, out_dir, metric_scale=metric_scale)
    print(f"saved {out_dir / 'numeric.png'}")
    print(f"saved {results_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run 50-agent numeric exp4 with fixed 5 malicious agents.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metric-scale", choices=["raw", "relative"], default="relative")
    args = parser.parse_args()
    run_experiment(args.data_dir, args.out_dir, metric_scale=args.metric_scale)

if __name__ == "__main__":
    main()
