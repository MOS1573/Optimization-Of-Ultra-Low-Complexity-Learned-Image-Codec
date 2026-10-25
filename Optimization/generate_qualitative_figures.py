"""
Generate all qualitative comparison figures for the thesis.
Every figure uses a UNIQUE image — no image appears in more than one figure.

Image assignment
───────────────
Fig 09 (existing)  Kodak_10   : FP32 | GPTQ | Mixed P3 — average-case summary (Results)
Fig 11 (existing)  Kodak_8    : All 6 methods 2×3 grid — worst PTQ on Kodak  (Results)
Fig 12 (existing)  pro_2020_38: All 6 methods 2×3 grid — worst PTQ on CLIC   (Results)

NEW figures (this script)
Fig 13  Kodak_13    : Zoom strip, 2nd-worst PTQ Kodak         (Results §PTQ failure)
Fig 14  pro_2020_25 : Zoom strip, largest GPTQ/PTQ gap CLIC   (Results §qualitative)
Fig 15  pro_2020_11 : 3-panel FP32|PTQ|GPTQ motivation        (Methodology §PTQ)
Fig 16  Kodak_5     : Lambda progression 2×4 FP32 vs Mixed P3 (Results §full R-D)
Fig 17  Kodak_6     : QAT recovery stages INT4G→P1→P2→P3      (Results §mixed QAT)
Fig 18  Kodak_2     : Clean 3-panel conclusion summary         (Conclusion)
"""

import os, re, numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PLOTS = os.path.join(REPO, "plots")
os.makedirs(PLOTS, exist_ok=True)

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 8.5, "pdf.fonttype": 42,
})

# ── model path helpers ────────────────────────────────────────────────────

def dirs_0018(dataset):
    return {
        "FP32\n(268 MB)":          f"{REPO}/legacy/new_tests/LoC_LIC_fp32/images/{dataset}/0.0018",
        "INT8 PTQ\n(67 MB)":       f"{REPO}/legacy/new_tests/LoC_LIC_int8_ptq/images/{dataset}/0.0018",
        "INT8 GPTQ\n(67 MB)":      f"{REPO}/legacy/new_tests/LoC_LIC_int8_gptq/images/{dataset}/0.0018",
        "INT4 GPTQ\n(34 MB)":      f"{REPO}/legacy/new_tests/LoC_LIC_int4_gptq/images/{dataset}/0.0018",
        "INT4 QAT Ph.1\n(34 MB)":  f"{REPO}/4-bit/phase1-uniform-int4/lambda-0018/LoC_LIC_int4_qat/images/{dataset}/0.0018",
        "Mixed QAT Ph.3\n(48 MB)": f"{REPO}/4-bit/phase3-mixed-finetuned/lambda-0018/LoC_LIC_mixed_qat/images/{dataset}/0.0018",
    }

# Dataset-average ΔPSNR vs FP32 (from thesis Table 4.11)
AVG_DELTA = {
    "FP32\n(268 MB)":          None,
    "INT8 PTQ\n(67 MB)":       "−2.586 dB",
    "INT8 GPTQ\n(67 MB)":      "−0.011 dB",
    "INT4 GPTQ\n(34 MB)":      "−3.088 dB",
    "INT4 QAT Ph.1\n(34 MB)":  "−1.342 dB",
    "Mixed QAT Ph.3\n(48 MB)": "−0.455 dB",
}
FAIL_COLOR    = "#d62728"
SUCCESS_COLOR = "#2ca02c"
PARTIAL_COLOR = "#ff7f0e"
LABEL_COLORS  = {
    "FP32\n(268 MB)":          "#333333",
    "INT8 PTQ\n(67 MB)":       FAIL_COLOR,
    "INT8 GPTQ\n(67 MB)":      SUCCESS_COLOR,
    "INT4 GPTQ\n(34 MB)":      FAIL_COLOR,
    "INT4 QAT Ph.1\n(34 MB)":  PARTIAL_COLOR,
    "Mixed QAT Ph.3\n(48 MB)": SUCCESS_COLOR,
}

# ── helpers ───────────────────────────────────────────────────────────────

def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)


def find_worst_crop(fp32_arr, ptq_arr, crop=256, stride=32):
    """Find crop×crop window with highest MSE between fp32 and ptq."""
    h, w = fp32_arr.shape[:2]
    best, best_box = -1, (0, 0, crop, crop)
    for y in range(0, h - crop + 1, stride):
        for x in range(0, w - crop + 1, stride):
            mse = np.mean((fp32_arr[y:y+crop, x:x+crop].astype(float)
                          - ptq_arr [y:y+crop, x:x+crop].astype(float)) ** 2)
            if mse > best:
                best, best_box = mse, (x, y, x+crop, y+crop)
    return best_box


def crop_arr(arr, box):
    l, t, r, b = box
    return arr[t:b, l:r]


def load_arr(path):
    return np.array(Image.open(path).convert("RGB"))


def title_with_delta(label, delta=None, color=None):
    base = label.replace("\n", " ")
    if delta:
        return f"{base}\n{delta}", color or LABEL_COLORS.get(label, "#333")
    return base, LABEL_COLORS.get(label, "#333")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PLOTS, f"{stem}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {stem}.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 13 — zoom strip, Kodak_13  (2nd-worst PTQ on Kodak, unused image)
# ═══════════════════════════════════════════════════════════════════════════

def fig13_zoom_kodak13():
    img = "Kodak_13.png"
    paths = dirs_0018("kodak")
    labels = list(paths.keys())

    fp32_arr = load_arr(os.path.join(paths["FP32\n(268 MB)"],     img))
    ptq_arr  = load_arr(os.path.join(paths["INT8 PTQ\n(67 MB)"],  img))

    outer_box = find_worst_crop(fp32_arr, ptq_arr, crop=192, stride=32)
    ox, oy = outer_box[0], outer_box[1]
    print(f"  Kodak_13 outer_box={outer_box}")

    # Find tightest zoom inside outer crop
    fp32_outer = crop_arr(fp32_arr, outer_box)
    ptq_outer  = crop_arr(ptq_arr,  outer_box)
    zoom, best_mse, zx, zy = 96, -1, 0, 0
    for ty in range(0, 192 - zoom + 1, 16):
        for tx in range(0, 192 - zoom + 1, 16):
            mse = np.mean((fp32_outer[ty:ty+zoom, tx:tx+zoom].astype(float)
                         - ptq_outer [ty:ty+zoom, tx:tx+zoom].astype(float))**2)
            if mse > best_mse:
                best_mse, zx, zy = mse, tx, ty
    zoom_box = (ox+zx, oy+zy, ox+zx+zoom, oy+zy+zoom)

    outer_crops = {l: crop_arr(load_arr(os.path.join(d, img)), outer_box) for l,d in paths.items()}
    zoom_crops  = {l: crop_arr(load_arr(os.path.join(d, img)), zoom_box)  for l,d in paths.items()}

    fig, axes = plt.subplots(2, 6, figsize=(13.5, 5.2))
    plt.subplots_adjust(hspace=0.10, wspace=0.04)

    for col, label in enumerate(labels):
        t, c = title_with_delta(label, AVG_DELTA.get(label))

        ax0, ax1 = axes[0, col], axes[1, col]
        ax0.imshow(outer_crops[label])
        rect = patches.Rectangle((zx, zy), zoom, zoom,
                                  linewidth=1.4, edgecolor="yellow", facecolor="none")
        ax0.add_patch(rect)
        ax0.axis("off")
        ax0.set_title(t, fontsize=7.5, color=c, pad=3)

        ax1.imshow(zoom_crops[label])
        ax1.axis("off")

    axes[0, 0].text(-0.18, 0.5, "192×192 crop", transform=axes[0,0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=8.5)
    axes[1, 0].text(-0.18, 0.5, "96×96 zoom",   transform=axes[1,0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=8.5)

    fig.suptitle("Kodak image 13 — worst-PTQ region detail (λ = 0.0018). "
                 "Yellow box marks the zoomed patch.", fontsize=9.5, y=1.01)
    save(fig, "13_qualitative_zoom_kodak13")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 14 — zoom strip, pro_2020_25  (largest GPTQ/PTQ gap on CLIC, unused)
# ═══════════════════════════════════════════════════════════════════════════

def fig14_zoom_pro2020_25():
    img = "pro_2020_25.png"
    paths = dirs_0018("pro_2020")
    labels = list(paths.keys())

    fp32_arr = load_arr(os.path.join(paths["FP32\n(268 MB)"],    img))
    ptq_arr  = load_arr(os.path.join(paths["INT8 PTQ\n(67 MB)"], img))

    outer_box = find_worst_crop(fp32_arr, ptq_arr, crop=192, stride=32)
    ox, oy = outer_box[0], outer_box[1]

    fp32_outer = crop_arr(fp32_arr, outer_box)
    ptq_outer  = crop_arr(ptq_arr,  outer_box)
    zoom, best_mse, zx, zy = 96, -1, 0, 0
    for ty in range(0, 192 - zoom + 1, 16):
        for tx in range(0, 192 - zoom + 1, 16):
            mse = np.mean((fp32_outer[ty:ty+zoom, tx:tx+zoom].astype(float)
                         - ptq_outer [ty:ty+zoom, tx:tx+zoom].astype(float))**2)
            if mse > best_mse:
                best_mse, zx, zy = mse, tx, ty
    zoom_box = (ox+zx, oy+zy, ox+zx+zoom, oy+zy+zoom)

    outer_crops = {l: crop_arr(load_arr(os.path.join(d, img)), outer_box) for l,d in paths.items()}
    zoom_crops  = {l: crop_arr(load_arr(os.path.join(d, img)), zoom_box)  for l,d in paths.items()}

    # Per-image PSNR for captions (vs FP32)
    fp32_ref = outer_crops["FP32\n(268 MB)"]
    per_img_delta = {l: psnr(fp32_ref, outer_crops[l]) for l in labels if l != "FP32\n(268 MB)"}
    print(f"  pro_2020_25 per-crop PSNR vs FP32: " +
          ", ".join(f"{l.split(chr(10))[0]}={v:.1f}" for l,v in per_img_delta.items()))

    fig, axes = plt.subplots(2, 6, figsize=(13.5, 5.2))
    plt.subplots_adjust(hspace=0.10, wspace=0.04)

    for col, label in enumerate(labels):
        t, c = title_with_delta(label, AVG_DELTA.get(label))
        ax0, ax1 = axes[0, col], axes[1, col]
        ax0.imshow(outer_crops[label]); ax0.axis("off")
        ax0.set_title(t, fontsize=7.5, color=c, pad=3)
        rect = patches.Rectangle((zx, zy), zoom, zoom,
                                  linewidth=1.4, edgecolor="yellow", facecolor="none")
        ax0.add_patch(rect)
        ax1.imshow(zoom_crops[label]); ax1.axis("off")

    axes[0, 0].text(-0.18, 0.5, "192×192 crop", transform=axes[0,0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=8.5)
    axes[1, 0].text(-0.18, 0.5, "96×96 zoom",   transform=axes[1,0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=8.5)

    fig.suptitle(
        r"CLIC 2020 image 25 — worst-PTQ region detail ($\lambda = 0.0018$, PTQ/GPTQ contrast: 13 dB gap). "
        "Yellow box marks the zoomed patch.", fontsize=9, y=1.01)
    save(fig, "14_qualitative_zoom_pro2020_25")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 15 — methodology motivation: FP32 | PTQ | GPTQ on pro_2020_11
#           (3-panel, clean; for the INT8 PTQ subsection)
# ═══════════════════════════════════════════════════════════════════════════

def fig15_methodology_motivation():
    img = "pro_2020_3.png"
    d = dirs_0018("pro_2020")
    fp32_arr = load_arr(os.path.join(d["FP32\n(268 MB)"],    img))
    ptq_arr  = load_arr(os.path.join(d["INT8 PTQ\n(67 MB)"], img))
    gptq_arr = load_arr(os.path.join(d["INT8 GPTQ\n(67 MB)"],img))

    box = find_worst_crop(fp32_arr, ptq_arr, crop=256, stride=32)
    print(f"  pro_2020_11 box={box}")

    fp32_c = crop_arr(fp32_arr, box)
    ptq_c  = crop_arr(ptq_arr,  box)
    gptq_c = crop_arr(gptq_arr, box)

    dp = psnr(fp32_c, ptq_c)
    dg = psnr(fp32_c, gptq_c)
    print(f"  pro_2020_11 per-crop: PTQ={dp:.2f} dB, GPTQ={dg:.2f} dB")

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
    plt.subplots_adjust(wspace=0.04)

    # Use dataset-level ΔPSNR (vs original image) from thesis — not per-crop similarity scores
    panels = [
        (fp32_c, "FP32 (reference)\n268 MB",                            "#333333"),
        (ptq_c,  "INT8 PTQ (naive)\n67 MB     ΔPSNR = −2.586 dB",      FAIL_COLOR),
        (gptq_c, "INT8 GPTQ\n67 MB     ΔPSNR = −0.011 dB",             SUCCESS_COLOR),
    ]

    for ax, (crop, title, color) in zip(axes, panels):
        ax.imshow(crop); ax.axis("off")
        ax.set_title(title, fontsize=8.5, color=color)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(color); sp.set_linewidth(1.5)

    fig.suptitle(
        "CLIC 2020 image 3 — INT8 PTQ vs INT8 GPTQ at same bit-width (λ = 0.0018, 256×256 crop).\n"
        "Both use 8-bit weights; the algorithm determines the outcome.",
        fontsize=9, y=1.03)
    fig.tight_layout(pad=0.4)
    save(fig, "15_methodology_ptq_motivation")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 16 — lambda progression: Kodak_5, FP32 vs Mixed QAT P3 at all 4 λ
# ═══════════════════════════════════════════════════════════════════════════

LAMBDA_DIRS = {
    "0.0018": {
        "FP32":   f"{REPO}/legacy/new_tests/LoC_LIC_fp32/images/kodak/0.0018",
        "Mixed":  f"{REPO}/4-bit/phase3-mixed-finetuned/lambda-0018/LoC_LIC_mixed_qat/images/kodak/0.0018",
    },
    "0.0035": {
        "FP32":   f"{REPO}/8-bit/gptq-per-lambda/lambda-0035/LoC_LIC_fp32/images/Kodak/0.0035",
        "Mixed":  f"{REPO}/4-bit/phase3-mixed-finetuned/lambda-0035/LoC_LIC_mixed_qat/images/kodak/0.0035",
    },
    "0.0067": {
        "FP32":   f"{REPO}/8-bit/gptq-per-lambda/lambda-0067/LoC_LIC_fp32/images/Kodak/0.0067",
        "Mixed":  f"{REPO}/4-bit/phase3-mixed-finetuned/lambda-0067/LoC_LIC_mixed_qat/images/kodak/0.0067",
    },
    "0.025": {
        "FP32":   f"{REPO}/8-bit/gptq-per-lambda/lambda-025/LoC_LIC_fp32/images/Kodak/0.025",
        "Mixed":  f"{REPO}/4-bit/phase3-mixed-finetuned/lambda-025/LoC_LIC_mixed_qat/images/kodak/0.025",
    },
}
# Dataset-average ΔPSNR per lambda from thesis Table 4.4
LAMBDA_DELTA = {
    "0.0018": "−0.455 dB",
    "0.0035": "−0.888 dB",
    "0.0067": "−1.017 dB",
    "0.025":  "−2.431 dB",
}
LAMBDA_BPP_FP32  = {"0.0018": "0.121", "0.0035": "0.203", "0.0067": "0.339", "0.025": "0.751"}
LAMBDA_BPP_MIXED = {"0.0018": "0.121", "0.0035": "0.197", "0.0067": "0.327", "0.025": "0.665"}


def fig16_lambda_progression():
    img = "Kodak_5.png"
    lambdas = ["0.0018", "0.0035", "0.0067", "0.025"]

    # Use fixed crop based on the worst PTQ region at λ=0.025
    fp32_025 = load_arr(os.path.join(LAMBDA_DIRS["0.025"]["FP32"], img))
    mix_025  = load_arr(os.path.join(LAMBDA_DIRS["0.025"]["Mixed"], img))
    box = find_worst_crop(fp32_025, mix_025, crop=256, stride=32)
    print(f"  Kodak_5 lambda-prog box={box}")

    # Load all crops
    fp32_crops = {}
    mix_crops  = {}
    deltas_crop = {}
    for lam in lambdas:
        fp32_arr = load_arr(os.path.join(LAMBDA_DIRS[lam]["FP32"], img))
        mix_arr  = load_arr(os.path.join(LAMBDA_DIRS[lam]["Mixed"], img))
        fp32_crops[lam] = crop_arr(fp32_arr, box)
        mix_crops[lam]  = crop_arr(mix_arr, box)
        deltas_crop[lam] = psnr(fp32_crops[lam], mix_crops[lam])
        print(f"  λ={lam}: crop PSNR FP32 vs Mixed = {deltas_crop[lam]:.2f} dB")

    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
    plt.subplots_adjust(wspace=0.04, hspace=0.25)

    for col, lam in enumerate(lambdas):
        # Row 0: FP32
        ax0 = axes[0, col]
        ax0.imshow(fp32_crops[lam]); ax0.axis("off")
        ax0.set_title(f"FP32  (BPP = {LAMBDA_BPP_FP32[lam]})", fontsize=8, color="#333333")

        # Row 1: Mixed QAT P3
        ax1 = axes[1, col]
        ax1.imshow(mix_crops[lam]); ax1.axis("off")
        color = FAIL_COLOR if float(lam) >= 0.0067 else (PARTIAL_COLOR if float(lam)>=0.0035 else SUCCESS_COLOR)
        ax1.set_title(f"Mixed QAT Ph.3  (BPP = {LAMBDA_BPP_MIXED[lam]})\nΔPSNR = {LAMBDA_DELTA[lam]}",
                      fontsize=7.5, color=color)

        # Column header: lambda value
        ax0.text(0.5, 1.15, f"λ = {lam}", transform=ax0.transAxes,
                 ha="center", fontsize=9.5, fontweight="bold", color="#333333")

    axes[0, 0].text(-0.14, 0.5, "FP32", transform=axes[0,0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=10, fontweight="bold")
    axes[1, 0].text(-0.14, 0.5, "Mixed QAT Ph.3", transform=axes[1,0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=9, fontweight="bold")

    fig.suptitle(
        "Kodak image 5 — FP32 vs Mixed QAT Phase 3 across all four operating points "
        "(256×256 crop, same region for all λ).\n"
        r"The quality gap widens from −0.455 dB at $\lambda=0.0018$ to −2.431 dB at $\lambda=0.025$.",
        fontsize=9, y=1.02)
    save(fig, "16_lambda_progression_kodak5")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 17 — QAT recovery stages: Kodak_6, INT4G → Phase1 → Phase2 → Phase3
# ═══════════════════════════════════════════════════════════════════════════

QAT_DIRS = {
    "INT4 GPTQ\n(34 MB, no train)": f"{REPO}/legacy/new_tests/LoC_LIC_int4_gptq/images/kodak/0.0018",
    "INT4 QAT Ph.1\n(34 MB, 100 ep)": f"{REPO}/4-bit/phase1-uniform-int4/lambda-0018/LoC_LIC_int4_qat/images/kodak/0.0018",
    "Mixed QAT Ph.2\n(48 MB, +100 ep)": f"{REPO}/4-bit/phase2-mixed/lambda-0018/LoC_LIC_mixed_qat/images/kodak/0.0018",
    "Mixed QAT Ph.3\n(48 MB, +50 ep)":  f"{REPO}/4-bit/phase3-mixed-finetuned/lambda-0018/LoC_LIC_mixed_qat/images/kodak/0.0018",
}
QAT_COLORS = {
    "INT4 GPTQ\n(34 MB, no train)":     FAIL_COLOR,
    "INT4 QAT Ph.1\n(34 MB, 100 ep)":  PARTIAL_COLOR,
    "Mixed QAT Ph.2\n(48 MB, +100 ep)": "#1f77b4",
    "Mixed QAT Ph.3\n(48 MB, +50 ep)":  SUCCESS_COLOR,
}
QAT_DATASET_PSNR = {
    "INT4 GPTQ\n(34 MB, no train)":     "ΔPSNR = −3.088 dB",
    "INT4 QAT Ph.1\n(34 MB, 100 ep)":  "ΔPSNR = −1.342 dB",
    "Mixed QAT Ph.2\n(48 MB, +100 ep)": "ΔPSNR = −0.535 dB",
    "Mixed QAT Ph.3\n(48 MB, +50 ep)":  "ΔPSNR = −0.455 dB",
}


def fig17_qat_recovery():
    img = "Kodak_6.png"
    labels = list(QAT_DIRS.keys())

    fp32_arr = load_arr(os.path.join(f"{REPO}/legacy/new_tests/LoC_LIC_fp32/images/kodak/0.0018", img))
    int4g_arr = load_arr(os.path.join(QAT_DIRS[labels[0]], img))

    box = find_worst_crop(fp32_arr, int4g_arr, crop=256, stride=32)
    print(f"  Kodak_6 QAT-recovery box={box}")

    crops = {}
    for label in labels:
        crops[label] = crop_arr(load_arr(os.path.join(QAT_DIRS[label], img)), box)
    fp32_crop = crop_arr(fp32_arr, box)

    # Compute per-crop PSNR
    per_crop = {l: psnr(fp32_crop, crops[l]) for l in labels}
    print("  Per-crop PSNR vs FP32: " + ", ".join(f"{l.split(chr(10))[0]}={v:.1f}" for l,v in per_crop.items()))

    fig, axes = plt.subplots(1, 5, figsize=(13, 3.5))
    plt.subplots_adjust(wspace=0.04)

    # Left panel: FP32 reference
    axes[0].imshow(fp32_crop); axes[0].axis("off")
    axes[0].set_title("FP32 (reference)\n268 MB", fontsize=8, color="#333333")
    for sp in axes[0].spines.values():
        sp.set_visible(True); sp.set_edgecolor("#333333"); sp.set_linewidth(1.0)

    for ax, label in zip(axes[1:], labels):
        ax.imshow(crops[label]); ax.axis("off")
        color = QAT_COLORS[label]
        title = label.replace("\n", " ") + f"\n{QAT_DATASET_PSNR[label]}"
        ax.set_title(title, fontsize=7.5, color=color)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(color); sp.set_linewidth(1.5)

    fig.suptitle(
        r"Kodak image 6 — QAT recovery stages from INT4 GPTQ to Mixed QAT Phase 3 ($\lambda = 0.0018$, 256×256 crop)."
        "\nΔPSNR values are dataset averages. Each stage adds training compute and recovers quality.",
        fontsize=9, y=1.04)
    save(fig, "17_qat_recovery_stages_kodak6")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 18 — conclusion summary: Kodak_2, FP32 | INT8 GPTQ | Mixed QAT P3
#           (Kodak_2 has GPTQ=38.62 dB vs FP32 — near-perfect, good for conclusion)
# ═══════════════════════════════════════════════════════════════════════════

def fig18_conclusion_summary():
    img = "Kodak_2.png"
    d = dirs_0018("kodak")

    fp32_arr = load_arr(os.path.join(d["FP32\n(268 MB)"],          img))
    gptq_arr = load_arr(os.path.join(d["INT8 GPTQ\n(67 MB)"],      img))
    mix_arr  = load_arr(os.path.join(d["Mixed QAT Ph.3\n(48 MB)"], img))

    # Use centre crop (Kodak_2 is a garden/outdoor scene — centre is most informative)
    h, w = fp32_arr.shape[:2]
    cx, cy, half = int(w * 0.5), int(h * 0.5), 128
    box = (cx-half, cy-half, cx+half, cy+half)

    fp32_c = crop_arr(fp32_arr, box)
    gptq_c = crop_arr(gptq_arr, box)
    mix_c  = crop_arr(mix_arr,  box)

    dg = psnr(fp32_c, gptq_c)
    dm = psnr(fp32_c, mix_c)
    print(f"  Kodak_2 centre crop: GPTQ={dg:.2f} dB, Mixed={dm:.2f} dB")

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
    plt.subplots_adjust(wspace=0.04)

    panels = [
        (fp32_c, "FP32 (reference)\n268 MB  |  0.121 BPP",                          "#333333"),
        (gptq_c, "INT8 GPTQ\n67 MB  |  0.121 BPP  |  ΔPSNR = −0.011 dB",           SUCCESS_COLOR),
        (mix_c,  "Mixed QAT Ph.3\n48 MB  |  0.121 BPP  |  ΔPSNR = −0.455 dB",      PARTIAL_COLOR),
    ]
    for ax, (crop, title, color) in zip(axes, panels):
        ax.imshow(crop); ax.axis("off")
        ax.set_title(title, fontsize=8, color=color)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(color); sp.set_linewidth(1.4)

    fig.suptitle(
        r"Kodak image 2 — deployment-ready model comparison at $\lambda = 0.0018$ (256×256 centre crop)."
        "\nINT8 GPTQ achieves 4× weight compression with identical bitstream size at near-zero quality cost.",
        fontsize=9, y=1.03)
    fig.tight_layout(pad=0.4)
    save(fig, "18_conclusion_summary_kodak2")


# ── main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating unique qualitative figures …\n")
    print("Fig 13 — zoom strip Kodak_13:")
    fig13_zoom_kodak13()

    print("\nFig 14 — zoom strip pro_2020_25:")
    fig14_zoom_pro2020_25()

    print("\nFig 15 — methodology motivation pro_2020_11:")
    fig15_methodology_motivation()

    print("\nFig 16 — lambda progression Kodak_5:")
    fig16_lambda_progression()

    print("\nFig 17 — QAT recovery stages Kodak_6:")
    fig17_qat_recovery()

    print("\nFig 18 — conclusion summary Kodak_2:")
    fig18_conclusion_summary()

    print("\nDone — all 6 new figures generated with unique images.")
