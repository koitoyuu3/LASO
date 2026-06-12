
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truth_discovery.core import LASOTruthFinder
from truth_discovery.experiment import exp_utils
from truth_discovery.experiment.exp_utils import (
    load_batches,
    metric_data_accuracy,
    to_text_df,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_DATA_DIR = ROOT / "truth_discovery" / "data" / "data_agent-50_news-300"
DEFAULT_OUT_DIR = (
    ROOT
    / "truth_discovery"
    / "examples"
    / "outputs"
    / "exp1_text"
    / "1.2-agent-50-news-300-50%-0.9-numonly"
)
BAD_COUNTS = [5, 10, 15, 20, 25]
THRESHOLD = 0.90

def build_file_map(data_dir: Path, bad_counts: Iterable[int], numeric_only: bool = False) -> Dict[int, Path]:
    files = {0: data_dir / "demo_div.json"}
    for bad_k in bad_counts:
        if numeric_only:
            files[bad_k] = data_dir / f"demo_{bad_k}bad_numonly.json"
        else:
            files[bad_k] = data_dir / f"demo_{bad_k}bad_collude.json"
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing experiment files:\n" + "\n".join(missing))
    return files

def run_LASO_single(df: pd.DataFrame) -> Dict[Tuple[int, str], str]:
    if df.empty:
        return {}
    model = LASOTruthFinder(enable_zk_proof=False, use_sbert=True, local_files_only=True)
    preds: Dict[Tuple[int, str], str] = {}
    empty_numeric_df = pd.DataFrame(columns=["website", "object", "fact", "batch_index"])
    for bi in sorted(df["batch_index"].unique()):
        grp = df[df["batch_index"] == bi][["website", "fact", "object", "batch_index"]].copy()
        _, out_txt = model.process_batch(empty_numeric_df.copy(), grp)
        for _, row in out_txt[["batch_index", "object", "global_truth"]].drop_duplicates().iterrows():
            preds[(int(row["batch_index"]), str(row["object"]))] = str(row["global_truth"])
    return preds

def _plot_text(x_vals, results, xlabel, title_prefix, out_dir):

    plt.rcParams["font.family"] = ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"SenteTruth": "#2ecc71", "BasicTruth": "#e67e22", "LASOTruth": "#9b59b6"}
    markers = {"SenteTruth": "s", "BasicTruth": "D", "LASOTruth": "*"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in ["SenteTruth", "BasicTruth", "LASOTruth"]:
        ax.plot(x_vals, results[m]["data_accuracy"],
                color=colors[m], marker=markers[m], label=m,
                linewidth=1.8, markersize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Data Accuracy (hybrid score ≥ {THRESHOLD})")
    ax.set_title(f"{title_prefix} — Text data accuracy (mixed scenario)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "mixed_text.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved {path}")

def run_experiment(data_dir: Path, out_dir: Path, numeric_only: bool = False) -> pd.DataFrame:
    files = build_file_map(data_dir, BAD_COUNTS, numeric_only=numeric_only)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_df = to_text_df(load_batches(files[0]))
    noisy_dfs = {bad_k: to_text_df(load_batches(path)) for bad_k, path in files.items() if bad_k != 0}

    print("clean: SenteTruth")
    clean_sente = exp_utils._run_sente_txt(clean_df)
    print("clean: BasicTruth")
    clean_basic = exp_utils._run_basic_txt(clean_df)
    print("clean: LASOTruth")
    clean_LASO = run_LASO_single(clean_df)

    methods = ["SenteTruth", "BasicTruth", "LASOTruth"]
    results = {m: {"data_accuracy": []} for m in methods}
    x_vals = [0.0]
    for m, preds in [("SenteTruth", clean_sente), ("BasicTruth", clean_basic), ("LASOTruth", clean_LASO)]:
        results[m]["data_accuracy"].append(metric_data_accuracy(preds, preds, threshold=THRESHOLD))

    for bad_k in BAD_COUNTS:
        ratio = bad_k / 50 * 100
        noisy_df = noisy_dfs[bad_k]
        print(f"{bad_k} bad ({ratio:.0f}%): SenteTruth")
        noisy_sente = exp_utils._run_sente_txt(noisy_df)
        print(f"{bad_k} bad ({ratio:.0f}%): BasicTruth")
        noisy_basic = exp_utils._run_basic_txt(noisy_df)
        print(f"{bad_k} bad ({ratio:.0f}%): LASOTruth")
        noisy_LASO = run_LASO_single(noisy_df)

        x_vals.append(ratio)
        for m, (c, n) in [
            ("SenteTruth", (clean_sente, noisy_sente)),
            ("BasicTruth", (clean_basic, noisy_basic)),
            ("LASOTruth", (clean_LASO, noisy_LASO)),
        ]:
            results[m]["data_accuracy"].append(metric_data_accuracy(c, n, threshold=THRESHOLD))

    _plot_text(x_vals, results, "Malicious agent ratio (%)", "E1-Text: Malicious agent ratio", out_dir)

    rows = []
    for idx, ratio in enumerate(x_vals):
        rows.append(
            {
                "malicious_ratio_pct": ratio,
                "SenteTruth": results["SenteTruth"]["data_accuracy"][idx],
                "BasicTruth": results["BasicTruth"]["data_accuracy"][idx],
                "LASOTruth": results["LASOTruth"]["data_accuracy"][idx],
            }
        )
    df = pd.DataFrame(rows)
    csv_path = out_dir / "text_data_accuracy.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"saved {csv_path}")
    print(f"saved {out_dir / 'mixed_text.png'}")
    return df

def main() -> None:
    parser = argparse.ArgumentParser(description="Run 50-agent text exp1 over 0-50% malicious ratios.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--numeric-only",
        action="store_true",
        help="Use numeric-only attack data (num_demo_* files). "
             "Output dir gets _numonly suffix.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    if args.numeric_only:
        out_dir = out_dir.parent / (out_dir.name + "_numonly")
    run_experiment(args.data_dir, out_dir, numeric_only=args.numeric_only)

if __name__ == "__main__":
    main()
