from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truth_discovery.experiment.exp_utils import (
    TEXT_METHODS,
    COLORS,
    MARKERS,
    load_batches,
    to_text_df,
    _run_sente_txt,
    _run_basic_txt,
    _run_LASO_txt,
    _get_metric_sbert_encoder,
    _numeric_text_score_for_accuracy,
)
from truth_discovery.core.hybrid_text_alignment import (
    DEFAULT_NUMERIC_MATCH_TOL,
    blend_LASO_numeric_scores,
)

DEFAULT_DATA_DIR = ROOT / "truth_discovery" / "data" / "data_agent-50_news-300"

N_AGENTS = 50
BAD_COUNTS = [0, 5, 10, 15, 20, 25]
ALPHA_VALUES = [round(x * 0.1, 1) for x in range(11)]
ACCURACY_THRESHOLD = 0.9

DEFAULT_OUT_DIR = (
    ROOT
    / "truth_discovery"
    / "examples"
    / "outputs"
    / f"exp7_alpha_sensitivity_th{ACCURACY_THRESHOLD:.2f}"
)

TITLE = "E5: Alpha Sensitivity Analysis"
X_LABEL = r"$\alpha$ (LASO weight)"

def _precompute_scores(
    x_preds: Dict[Tuple[int, str], str],
    y_preds: Dict[Tuple[int, str], str],
    num_tol: float = DEFAULT_NUMERIC_MATCH_TOL,
) -> Tuple[np.ndarray, List[Optional[float]]]:

    keys = [k for k in x_preds if k in y_preds]
    if not keys:
        return np.array([], dtype=float), []

    x_txts = [x_preds[k] for k in keys]
    y_txts = [y_preds[k] for k in keys]

    encoder = _get_metric_sbert_encoder()
    if encoder is not None:
        x_emb = encoder.encode(x_txts, convert_to_numpy=True, show_progress_bar=False)
        y_emb = encoder.encode(y_txts, convert_to_numpy=True, show_progress_bar=False)
        x_n = x_emb / (np.linalg.norm(x_emb, axis=1, keepdims=True) + 1e-12)
        y_n = y_emb / (np.linalg.norm(y_emb, axis=1, keepdims=True) + 1e-12)
        sem = np.clip((x_n * y_n).sum(axis=1), -1.0, 1.0)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

        vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        mat = vect.fit_transform(x_txts + y_txts)
        n = len(keys)
        sem = np.clip(sk_cosine(mat[:n], mat[n:]).diagonal(), 0.0, 1.0)

    num = [
        _numeric_text_score_for_accuracy(x, y, tol=num_tol)
        for x, y in zip(x_txts, y_txts)
    ]
    return sem, num

def _sweep_alpha(
    sem_scores: np.ndarray,
    num_scores: List[Optional[float]],
    alpha_values: List[float],
    threshold: float = ACCURACY_THRESHOLD,
) -> Dict[float, float]:

    if sem_scores.size == 0:
        return {a: float("nan") for a in alpha_values}
    out: Dict[float, float] = {}
    for alpha in alpha_values:
        combined = np.array(
            [
                blend_LASO_numeric_scores(float(s), n, alpha=alpha)
                for s, n in zip(sem_scores, num_scores)
            ],
            dtype=float,
        )
        out[alpha] = float(np.mean(combined >= threshold))
    return out

def _setup_font() -> None:
    plt.rcParams.update(
        {
            "font.family": [
                "Arial Unicode MS",
                "Hiragino Sans GB",
                "Songti SC",
                "DejaVu Sans",
            ],
            "font.sans-serif": [
                "Arial Unicode MS",
                "Hiragino Sans GB",
                "Songti SC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )

def _plot_by_method(
    alpha_values: List[float],
    results: Dict[str, Dict[int, Dict[float, float]]],
    bad_counts: List[int],
    out_dir: Path,
) -> None:

    _setup_font()
    methods = TEXT_METHODS
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    cmap = plt.cm.YlOrRd
    bad_max = max(max(bad_counts), 1)
    norm = plt.Normalize(vmin=-bad_max * 0.15, vmax=bad_max * 1.1)

    for ax, method in zip(axes, methods):
        for bad_k in bad_counts:
            pct = bad_k / N_AGENTS * 100
            color = cmap(norm(bad_k))
            y_vals = [results[method][bad_k].get(a, float("nan")) for a in alpha_values]
            ax.plot(
                alpha_values,
                y_vals,
                color=color,
                marker="o",
                markersize=3.5,
                linewidth=1.5,
                label=f"{pct:.0f}%",
            )
        ax.axvline(0.5, color="#aaaaaa", linestyle="--", linewidth=0.9)
        ax.set_xlabel(X_LABEL)
        ax.set_title(method)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f"Data Accuracy (threshold={ACCURACY_THRESHOLD})")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Malicious %",
        loc="center right",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        title_fontsize=9,
    )
    fig.suptitle(TITLE, fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.91, 0.94))
    path = out_dir / "alpha_sensitivity_by_method.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {path}")

def _plot_summary(
    alpha_values: List[float],
    results: Dict[str, Dict[int, Dict[float, float]]],
    bad_counts: List[int],
    out_dir: Path,
) -> None:

    _setup_font()
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in TEXT_METHODS:
        mean_vals: List[float] = []
        for alpha in alpha_values:
            accs = [
                results[method][bad_k].get(alpha, float("nan"))
                for bad_k in bad_counts
            ]
            accs_valid = [a for a in accs if not np.isnan(a)]
            mean_vals.append(float(np.mean(accs_valid)) if accs_valid else float("nan"))
        ax.plot(
            alpha_values,
            mean_vals,
            color=COLORS[method],
            marker=MARKERS[method],
            label=method,
            linewidth=1.8,
            markersize=6,
        )

    ax.axvline(0.5, color="#aaaaaa", linestyle="--", linewidth=0.9, label=r"$\alpha$=0.5")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(f"Mean Data Accuracy (threshold={ACCURACY_THRESHOLD})")
    ax.set_title(f"{TITLE} — Average across malicious ratios")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "alpha_sensitivity_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {path}")

def _save_csv(
    alpha_values: List[float],
    results: Dict[str, Dict[int, Dict[float, float]]],
    bad_counts: List[int],
    out_dir: Path,
) -> None:
    rows: List[Dict] = []
    for method in TEXT_METHODS:
        for bad_k in bad_counts:
            pct = bad_k / N_AGENTS * 100
            for alpha in alpha_values:
                rows.append(
                    {
                        "method": method,
                        "bad_count": bad_k,
                        "malicious_pct": pct,
                        "alpha": alpha,
                        "data_accuracy": results[method][bad_k].get(
                            alpha, float("nan")
                        ),
                    }
                )
    path = out_dir / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  -> saved {path}")

def run_experiment(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    file_map: Dict[int, Path] = {}
    for bad_k in BAD_COUNTS:
        if bad_k == 0:
            p = data_dir / "demo.json"
        else:
            p = data_dir / f"demo_{bad_k}bad_collude.json"
        if p.exists():
            file_map[bad_k] = p
        else:
            print(f"  [skip] {p.name} not found")

    clean_path = data_dir / "demo.json"
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean data not found: {clean_path}")

    available_bad_counts = sorted(file_map)
    clean_batches = load_batches(clean_path)
    clean_txt_df = to_text_df(clean_batches)

    results: Dict[str, Dict[int, Dict[float, float]]] = {
        m: {} for m in TEXT_METHODS
    }

    print("\n" + "=" * 60)
    print(TITLE)
    print("=" * 60)

    for bad_k in available_bad_counts:
        pct = bad_k / N_AGENTS * 100
        print(f"\n  bad_count={bad_k}  ({pct:.0f}% malicious)")

        noisy_batches = load_batches(file_map[bad_k])
        noisy_txt_df = to_text_df(noisy_batches)

        print("    [SenteTruth] running ...", end=" ", flush=True)
        x_sente = _run_sente_txt(clean_txt_df)
        y_sente = _run_sente_txt(noisy_txt_df)
        print(f"{len(x_sente)}/{len(y_sente)} preds")

        print("    [BasicTruth] running ...", end=" ", flush=True)
        x_basic = _run_basic_txt(clean_txt_df)
        y_basic = _run_basic_txt(noisy_txt_df)
        print(f"{len(x_basic)}/{len(y_basic)} preds")

        print("    [LASOTruth] running ...", end=" ", flush=True)
        x_sem, y_sem = _run_LASO_txt(clean_txt_df, noisy_txt_df)
        print(f"{len(x_sem)}/{len(y_sem)} preds")

        method_preds = [
            ("SenteTruth", x_sente, y_sente),
            ("BasicTruth", x_basic, y_basic),
            ("LASOTruth", x_sem, y_sem),
        ]
        for method, xp, yp in method_preds:
            sem, num = _precompute_scores(xp, yp)
            results[method][bad_k] = _sweep_alpha(sem, num, ALPHA_VALUES)
            acc05 = results[method][bad_k].get(0.5, float("nan"))
            print(f"    {method} alpha sweep done  (alpha=0.5 -> {acc05:.4f})")

    _plot_by_method(ALPHA_VALUES, results, available_bad_counts, out_dir)
    _plot_summary(ALPHA_VALUES, results, available_bad_counts, out_dir)
    _save_csv(ALPHA_VALUES, results, available_bad_counts, out_dir)

    print("\n  Optimal alpha per method (averaged across bad ratios):")
    for method in TEXT_METHODS:
        best_alpha, best_acc = 0.5, 0.0
        for alpha in ALPHA_VALUES:
            accs = [
                results[method][bk].get(alpha, float("nan"))
                for bk in available_bad_counts
            ]
            mean_acc = float(np.nanmean(accs))
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_alpha = alpha
        print(f"    {method}: alpha={best_alpha:.1f}  mean_DataAcc={best_acc:.4f}")

    print(f"\n[E5] Done. Results saved to {out_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description="E5: Alpha sensitivity analysis")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    run_experiment(args.data_dir, args.out_dir)

if __name__ == "__main__":
    main()
