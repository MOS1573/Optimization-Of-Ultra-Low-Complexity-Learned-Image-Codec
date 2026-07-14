#!/usr/bin/env python3
"""
Generate per-image PSNR box plot from results/per_image_psnr_kodak.json.
Saves 19_psnr_boxplot.{pdf,png} to results/plots/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_JSON   = REPO_ROOT / "results" / "per_image_psnr_kodak.json"
OUT_DIR   = REPO_ROOT / "results" / "plots"

LABELS_ORDER = [
    "FP32 Baseline",
    "INT8 GPTQ",
    "INT8 PTQ",
    "Mixed QAT Phase 3",
]

DISPLAY_LABELS = [
    "FP32",
    "INT8 GPTQ",
    "INT8 PTQ",
    "Mixed QAT\nPhase 3",
]

COLORS = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]


def main():
    with open(IN_JSON) as f:
        data = json.load(f)

    # Build ordered list; skip any missing models
    psnr_data = []
    labels     = []
    colors     = []
    for key, disp, col in zip(LABELS_ORDER, DISPLAY_LABELS, COLORS):
        if key in data:
            psnr_data.append(data[key])
            labels.append(disp)
            colors.append(col)

    fig, ax = plt.subplots(figsize=(7, 4))

    bp = ax.boxplot(
        psnr_data,
        patch_artist=True,
        notch=False,
        widths=0.5,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, linestyle="none",
                        markeredgecolor="grey", alpha=0.7),
    )

    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("RGB-PSNR (dB)", fontsize=10)
    ax.set_title("Per-image RGB-PSNR distribution on Kodak (24 images, λ = 0.0018)",
                 fontsize=10, pad=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Annotate medians
    for i, vals in enumerate(psnr_data):
        med = float(np.median(vals))
        ax.text(i + 1, med + 0.05, f"{med:.2f}", ha="center", va="bottom",
                fontsize=7.5, color="black")

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"19_psnr_boxplot.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
