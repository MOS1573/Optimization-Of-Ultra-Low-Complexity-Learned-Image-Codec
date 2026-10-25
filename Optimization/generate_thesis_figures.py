"""
Generate new thesis figures:
  08_sensitivity_distribution.pdf  – layer sensitivity scores by module group
  09_qualitative_comparison.pdf    – side-by-side FP32 / INT8 GPTQ / Mixed QAT crop
  10_training_convergence.pdf      – QAT convergence (val_loss proxy) across phases
"""

import json
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── paths ────────────────────────────────────────────────────────────────────
REPO = os.environ.get("LOCLIC_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLOTS = os.path.join(REPO, "results", "plots")
os.makedirs(PLOTS, exist_ok=True)

LAYER_BITS_JSON = os.path.join(
    REPO, "Optimization", "4-bit", "4-bit Mixed-Precision", "layer_bits.json"
)
FP32_IMG_DIR = os.path.join(
    REPO, "results", "legacy", "new_tests", "LoC_LIC_fp32", "images", "kodak", "0.0018"
)
GPTQ_IMG_DIR = os.path.join(
    REPO, "results", "legacy", "new_tests", "LoC_LIC_int8_gptq", "images", "kodak", "0.0018"
)
MIXED_IMG_DIR = os.path.join(
    REPO, "results", "4-bit", "phase3-mixed-finetuned", "lambda-0018",
    "LoC_LIC_mixed_qat", "images", "kodak", "0.0018"
)

# Training log paths (job outputs, most complete data available)
EXPERIMENTS_DIR = os.path.join(REPO, "experiments")

VAL_LOG_DIRS = {
    "Phase 1 – Uniform INT4 QAT": os.path.join(EXPERIMENTS_DIR, "4-bit QAT LOCLIC"),
    "Phase 2 – Mixed-precision init": os.path.join(EXPERIMENTS_DIR, "mixed-precision QAT LOCLIC"),
    "Phase 3 – Mixed fine-tune": os.path.join(EXPERIMENTS_DIR, "mixed-precision QAT LOCLIC Phase3"),
    "Phase 4b – KD v2": os.path.join(EXPERIMENTS_DIR, "knowledge-distillation LOCLIC v2"),
}

FP32_PSNR = 28.415  # dB, from sensitivity analysis log (4 Kodak images, 256×256)

# matplotlib style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
})


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Sensitivity distribution
# ─────────────────────────────────────────────────────────────────────────────

def module_group(name: str) -> str:
    """Map a layer name to a human-readable module group."""
    if name.startswith("g_a"):
        return "g_a (analysis)"
    if name.startswith("g_s"):
        return "g_s (synthesis)"
    if name.startswith("h_a"):
        return "h_a (hyper-enc)"
    if name.startswith("h_s"):
        return "h_s (hyper-dec)"
    if name.startswith("channel_context"):
        return "channel context"
    if name.startswith("local_context"):
        return "local context"
    if name.startswith("global_inter_context"):
        return "global inter-ctx"
    if name.startswith("global_intra_context"):
        return "global intra-ctx"
    if "entropy_param" in name:
        return "entropy params"
    if name.startswith("lrp"):
        return "LRP transform"
    return "other"


GROUP_ORDER = [
    "g_a (analysis)",
    "h_a (hyper-enc)",
    "h_s (hyper-dec)",
    "g_s (synthesis)",
    "entropy params",
    "channel context",
    "LRP transform",
    "local context",
    "global inter-ctx",
    "global intra-ctx",
]

GROUP_COLORS = {
    "g_a (analysis)":    "#d62728",
    "h_a (hyper-enc)":   "#e05f5f",
    "h_s (hyper-dec)":   "#e08080",
    "g_s (synthesis)":   "#ff7f0e",
    "entropy params":    "#9467bd",
    "channel context":   "#8c564b",
    "LRP transform":     "#c49c94",
    "local context":     "#2ca02c",
    "global inter-ctx":  "#17becf",
    "global intra-ctx":  "#aec7e8",
}

def plot_sensitivity():
    with open(LAYER_BITS_JSON) as f:
        data = json.load(f)

    ranking = data["sensitivity_ranking"]  # sorted by drop_db desc

    # Aggregate by module group
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for entry in ranking:
        g = module_group(entry["name"])
        groups[g].append(entry["drop_db"])

    # Build ordered lists
    ordered_groups = [g for g in GROUP_ORDER if g in groups]
    means = [np.mean(groups[g]) for g in ordered_groups]
    maxs  = [np.max(groups[g])  for g in ordered_groups]
    mins  = [np.min(groups[g])  for g in ordered_groups]
    colors = [GROUP_COLORS.get(g, "#aaaaaa") for g in ordered_groups]

    # Shorter display labels
    display = [g.split(" (")[0].replace(" ", "\n") if len(g) > 14 else g
               for g in ordered_groups]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(ordered_groups))
    bars = ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.5, width=0.65, zorder=3)
    # Error bars: min/max
    yerr_low  = np.array(means) - np.array(mins)
    yerr_high = np.array(maxs)  - np.array(means)
    ax.errorbar(x, means, yerr=[yerr_low, yerr_high],
                fmt="none", ecolor="#444", elinewidth=1.2, capsize=3, zorder=4)

    # INT8 threshold line (0.05 dB becomes irrelevant at this scale; show the
    # actual decision boundary implied by the data)
    ax.axhline(0.05, color="#555", linestyle="--", linewidth=1.0, label="INT8 threshold (0.05 dB)")
    ax.set_xticks(x)
    ax.set_xticklabels([g for g in ordered_groups], rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("PSNR drop when layer set to INT4 (dB)")
    ax.set_title("Layer sensitivity to INT4 quantization by module group (λ = 0.0018)")
    ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    # Annotate mean values
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.5, f"{m:.1f}", ha="center", va="bottom", fontsize=7.5)

    fig.tight_layout()
    out_pdf = os.path.join(PLOTS, "08_sensitivity_distribution.pdf")
    out_png = os.path.join(PLOTS, "08_sensitivity_distribution.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_pdf}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Qualitative comparison
# ─────────────────────────────────────────────────────────────────────────────

def load_crop(img_dir: str, img_name: str, crop_box):
    path = os.path.join(img_dir, img_name)
    img = Image.open(path).convert("RGB")
    return img.crop(crop_box)

def compute_psnr(ref: np.ndarray, cmp: np.ndarray) -> float:
    mse = np.mean((ref.astype(float) - cmp.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255 ** 2 / mse)


def plot_qualitative():
    # Use Kodak_10.png – urban scene with structure/text artefacts visible
    img_name = "Kodak_10.png"

    # Open FP32 to determine crop region
    fp32_full = Image.open(os.path.join(FP32_IMG_DIR, img_name)).convert("RGB")
    w, h = fp32_full.size

    # Choose a 256×256 crop from centre-right (often has detail)
    cx, cy = int(w * 0.55), int(h * 0.45)
    half = 128
    crop_box = (cx - half, cy - half, cx + half, cy + half)

    fp32_crop  = load_crop(FP32_IMG_DIR,  img_name, crop_box)
    gptq_crop  = load_crop(GPTQ_IMG_DIR,  img_name, crop_box)
    mixed_crop = load_crop(MIXED_IMG_DIR, img_name, crop_box)

    fp32_arr  = np.array(fp32_crop)
    gptq_arr  = np.array(gptq_crop)
    mixed_arr = np.array(mixed_crop)

    psnr_gptq  = compute_psnr(fp32_arr, gptq_arr)
    psnr_mixed = compute_psnr(fp32_arr, mixed_arr)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))

    titles = [
        "FP32 (reference)\n268 MB",
        "INT8 GPTQ  (67 MB)\nΔPSNR = −0.011 dB",
        "Mixed QAT Phase 3  (48 MB)\nΔPSNR = −0.455 dB",
    ]
    crops = [fp32_crop, gptq_crop, mixed_crop]

    for ax, crop, title in zip(axes, crops, titles):
        ax.imshow(crop)
        ax.set_title(title, fontsize=8.5)
        ax.axis("off")

    fig.tight_layout(pad=0.4, rect=[0, 0, 1, 0.88])
    fig.suptitle(
        f"Kodak image 10 — 256×256 centre crop (λ = 0.0018)",
        fontsize=9.5
    )
    out_pdf = os.path.join(PLOTS, "09_qualitative_comparison.pdf")
    out_png = os.path.join(PLOTS, "09_qualitative_comparison.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_pdf}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Training convergence (PSNR per epoch across all phases)
# ─────────────────────────────────────────────────────────────────────────────

PHASE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def parse_val_logs(directory: str):
    """Parse all val_*.log files in directory. Returns sorted list of (epoch, psnr)."""
    if not os.path.isdir(directory):
        return []
    data = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.startswith("val_"):
            continue
        fpath = os.path.join(directory, fname)
        with open(fpath) as fh:
            for line in fh:
                m = re.search(r"Test epoch\s+(\d+).*PSNR:\s+([\d.]+)", line)
                if m:
                    epoch = int(m.group(1))
                    psnr  = float(m.group(2))
                    if epoch not in data:
                        data[epoch] = psnr
    return sorted(data.items())


def plot_convergence():
    """Plot PSNR vs cumulative training epoch for all QAT phases."""
    fig, ax = plt.subplots(figsize=(8, 4.0))

    cumulative_offset = 0
    legend_handles = []

    for (label, val_dir), color in zip(VAL_LOG_DIRS.items(), PHASE_COLORS):
        data = parse_val_logs(val_dir)
        if not data:
            print(f"  No val log data in: {val_dir}")
            continue
        epochs = [e for e, _ in data]
        psnr   = [p for _, p in data]
        base   = epochs[0]
        x      = [cumulative_offset + (e - base) for e in epochs]
        line,  = ax.plot(x, psnr, color=color, linewidth=1.5, label=label)
        legend_handles.append(line)
        if cumulative_offset > 0:
            ax.axvline(cumulative_offset, color="#bbbbbb", linestyle=":", linewidth=0.8)
        cumulative_offset += (epochs[-1] - base + 1)

    # FP32 reference line
    fp32_line = ax.axhline(FP32_PSNR, color="#333333", linestyle="--", linewidth=1.2,
                           label=f"FP32 baseline ({FP32_PSNR:.2f} dB)")
    legend_handles.append(fp32_line)

    ax.set_xlabel("Cumulative training epoch")
    ax.set_ylabel("Validation PSNR (dB)")
    ax.set_title("QAT training convergence across all phases (λ = 0.0018, Kodak validation)")
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9, fontsize=8.5)
    fig.tight_layout()

    out_pdf = os.path.join(PLOTS, "10_training_convergence.pdf")
    out_png = os.path.join(PLOTS, "10_training_convergence.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    print("Generating thesis figures …")
    plot_sensitivity()
    plot_qualitative()
    plot_convergence()
    print("Done.")
