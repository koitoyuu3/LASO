from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truth_discovery.experiment.exp_utils import (
    COLORS,
    NUM_METHODS,
    build_gt_numeric,
    eval_numeric_per_batch,
    load_batches,
    to_numeric_df,
)

DEFAULT_DATA_DIR = ROOT / "truth_discovery" / "data" / "data_agent-50_news-300"
DEFAULT_OUT_DIR = (
    ROOT
    / "truth_discovery"
    / "examples"
    / "outputs"
    / "exp1_numeric"
    / "data_agent-50_news-300_collude_relative"
)
BAD_COUNTS = [0, 5, 10, 15, 20, 25]
N_AGENTS_50 = 50

def build_file_map(data_dir: Path, bad_counts: Iterable[int]) -> Dict[int, Path]:
    files = {}
    for bad_k in bad_counts:
        if bad_k == 0:
            files[bad_k] = data_dir / "num_demo.json"
        else:
            files[bad_k] = data_dir / f"num_demo_{bad_k}bad_collude.json"
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing numeric experiment files:\n" + "\n".join(missing))
    return files

def _format_bar_label(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3f}"

def plot_bar_by_ratio(
    results_df: pd.DataFrame,
    bad_counts: list[int],
    out_dir: Path,
    metric_scale: str,
) -> Path:
    import numpy as np

    is_relative = metric_scale == "relative"
    metric_label = "Normalized RMSE" if is_relative else "RMSE"

    mean_df = (
        results_df.groupby(["bad_count", "method"])["rmse"]
        .mean()
        .reset_index()
        .rename(columns={"rmse": "mean_rmse"})
    )

    ratios = [f"{int(k / N_AGENTS_50 * 100)}%" for k in bad_counts]
    n_groups = len(bad_counts)
    n_methods = len(NUM_METHODS)
    bar_width = 0.13
    group_gap = 0.18
    group_width = n_methods * bar_width + group_gap
    x = np.arange(n_groups) * group_width

    plt.rcParams["font.family"] = ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(max(16, n_groups * 2.2), 6))

    y_max = float(mean_df["mean_rmse"].max()) if not mean_df.empty else 0.0
    y_pad = y_max * 0.02 if y_max > 0 else 0.02
    label_h = y_max * 0.04 if y_max > 0 else 0.04
    placed: list[tuple[float, float, float]] = []

    for i, method in enumerate(NUM_METHODS):
        method_data = mean_df[mean_df["method"] == method]
        vals = []
        for bad_k in bad_counts:
            row = method_data[method_data["bad_count"] == bad_k]
            vals.append(float(row["mean_rmse"].iloc[0]) if not row.empty else 0.0)

        offset = (i - (n_methods - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            vals,
            width=bar_width * 0.90,
            color=COLORS[method],
            label=method,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        for bar, val in zip(bars, vals):
            if val <= 0:
                continue
            x_c = bar.get_x() + bar.get_width() / 2
            y_try = bar.get_height() + y_pad
            for _, py_bot, py_top in placed:
                if py_bot < y_try + label_h and y_try < py_top:
                    y_try = py_top + y_pad * 0.5
            placed.append((x_c, y_try, y_try + label_h))
            ax.text(
                x_c,
                y_try,
                _format_bar_label(val),
                ha="center",
                va="bottom",
                fontsize=6,
                color="#333333",
                rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(ratios, fontsize=11)
    ax.set_xlabel("Malicious agent ratio (bad_count / 50 agents)", fontsize=12, labelpad=8)
    ax.set_ylabel(metric_label, fontsize=12, labelpad=8)
    ax.set_title(f"E1-Numeric: Mean {metric_label} of each method vs. malicious agent ratio", fontsize=13, pad=14)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, y_max * 1.40 if y_max > 0 else 1.0)
    ax.legend(
        title="Method",
        title_fontsize=10,
        fontsize=9.5,
        loc="upper left",
        framealpha=0.85,
        edgecolor="#cccccc",
        ncol=1,
    )

    fig.tight_layout()
    path = out_dir / "numeric_per_batch_rmse.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path

def run_experiment(data_dir: Path, out_dir: Path, metric_scale: str) -> pd.DataFrame:
    if metric_scale not in {"raw", "relative"}:
        raise ValueError(f"Unsupported metric_scale: {metric_scale}")

    files = build_file_map(data_dir, BAD_COUNTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalize_metric = metric_scale == "relative"

    clean_num_batches = load_batches(files[0])
    gt_num = build_gt_numeric(clean_num_batches)

    rows = []
    for bad_k in BAD_COUNTS:
        ratio = bad_k / N_AGENTS_50 * 100
        print(f"\n  bad_count={bad_k}  ({ratio:.0f}% malicious)")
        num_batches = load_batches(files[bad_k])
        noisy_num_df = to_numeric_df(num_batches)
        per_batch_results = eval_numeric_per_batch(noisy_num_df, gt_num, normalize=normalize_metric)
        for method in NUM_METHODS:
            for batch_idx, rmse in sorted(per_batch_results[method].items()):
                rows.append(
                    {
                        "bad_count": bad_k,
                        "method": method,
                        "batch_idx": batch_idx,
                        "rmse": rmse,
                        "metric_scale": metric_scale,
                    }
                )

    results_df = pd.DataFrame(rows, columns=["bad_count", "method", "batch_idx", "rmse", "metric_scale"])
    csv_path = out_dir / "numeric_per_batch_rmse.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    png_path = plot_bar_by_ratio(results_df, BAD_COUNTS, out_dir, metric_scale=metric_scale)

    print("\nmean RMSE by bad_count/method")
    print(results_df.groupby(["bad_count", "method"])["rmse"].mean().reset_index().to_string(index=False))
    print(f"saved {csv_path}")
    print(f"saved {png_path}")
    return results_df

def main() -> None:
    parser = argparse.ArgumentParser(description="Run 50-agent numeric exp1 over 0-50% malicious ratios.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metric-scale", choices=["raw", "relative"], default="relative")
    args = parser.parse_args()
    run_experiment(args.data_dir, args.out_dir, metric_scale=args.metric_scale)

if __name__ == "__main__":
    main()
