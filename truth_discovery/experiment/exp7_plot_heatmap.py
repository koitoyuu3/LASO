
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = (
    ROOT
    / "truth_discovery"
    / "examples"
    / "outputs"
    / "exp7_alpha_sensitivity_th0.90"
    / "results.csv"
)

def plot_heatmap(csv_path: Path, out_path: Path, threshold: float = 0.9) -> None:
    df = pd.read_csv(csv_path)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]

    methods = ["SenteTruth", "BasicTruth", "LASO(Ours)"]
    alphas = sorted(df["alpha"].unique())
    pcts = sorted(df["malicious_pct"].unique())

    fig, axes = plt.subplots(
        1, len(methods), figsize=(5.2 * len(methods), 4.6), sharey=True
    )
    if len(methods) == 1:
        axes = [axes]

    cmap = plt.get_cmap("viridis")
    vmin, vmax = 0.0, 1.0

    im = None
    for ax, method in zip(axes, methods):
        sub = df[df["method"] == method]

        mat = (
            sub.pivot(index="malicious_pct", columns="alpha", values="data_accuracy")
            .reindex(index=pcts, columns=alphas)
            .values
        )

        mat_disp = mat[::-1, :]
        pcts_disp = pcts[::-1]

        im = ax.imshow(
            mat_disp,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )

        for i in range(mat_disp.shape[0]):
            for j in range(mat_disp.shape[1]):
                val = mat_disp[i, j]
                if np.isnan(val):
                    continue

                txt_color = "white" if val < 0.55 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=txt_color,
                )

        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:.1f}" for a in alphas], fontsize=8)
        ax.set_yticks(range(len(pcts_disp)))
        ax.set_yticklabels([f"{p:.0f}%" for p in pcts_disp], fontsize=8)
        ax.set_xlabel(r"$\alpha$ (LASO weight)")
        ax.set_title(method, fontsize=11)

        if 0.5 in alphas:
            j_mid = alphas.index(0.5)
            ax.axvline(j_mid, color="white", linewidth=0.8, alpha=0.55, linestyle="--")

    axes[0].set_ylabel("Malicious %")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
    cbar.set_label(f"Data Accuracy (threshold={threshold})", fontsize=9)

    fig.suptitle("E5: Alpha Sensitivity Analysis (heatmap)", fontsize=13)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {out_path}")
    print(f"  -> saved {out_path.with_suffix('.pdf')}")

def main() -> None:
    parser = argparse.ArgumentParser(description="E5 alpha sensitivity heatmap plot")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()

    out_path = args.out or args.csv.parent / "alpha_sensitivity_heatmap.png"
    plot_heatmap(args.csv, out_path, threshold=args.threshold)

if __name__ == "__main__":
    main()
