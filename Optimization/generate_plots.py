#!/usr/bin/env python3
"""
Generate all 7 thesis plots for LoC-LIC quantization results.
Outputs saved to results/plots/ as 300 DPI PNG + PDF pairs.

Usage:
    python Optimization/generate_plots.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO      = Path(__file__).parent.parent
_eval_root = REPO / "results/eval_harness"
_subdirs   = sorted(_eval_root.glob("eval_*"))
EVAL_DIR   = _subdirs[-1] if _subdirs else _eval_root   # auto-detect latest run
OUT_DIR   = REPO / "results/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _actual_bpp(df: pd.DataFrame, keys: list) -> list:
    """Read actual_bpp column (per-image avg from codec timing, exact for each dataset)."""
    return [df.loc[k, "actual_bpp"] for k in keys]

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     13,
    "legend.fontsize":    9.5,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "axes.axisbelow":     True,
})

C_FP32   = "#1565C0"   # deep blue
C_GPTQ   = "#E65100"   # deep orange
C_QAT    = "#2E7D32"   # deep green
C_QAT1   = "#66BB6A"   # light green (phase 1)
C_COMP   = "#6A1B9A"   # purple (torch.compile)
C_GREY   = "#90A4AE"   # grey  (unfair / context rows)
C_RED    = "#EF5350"   # red   (slower)

# ── Load eval CSVs ────────────────────────────────────────────────────────────
kodak = pd.read_csv(EVAL_DIR / "evaluation_kodak.csv",    index_col="key")
pro   = pd.read_csv(EVAL_DIR / "evaluation_pro_2020.csv", index_col="key")

LAMBDAS       = [0.0018, 0.0035, 0.0067, 0.025]
FP32_KEYS     = ["fp32",      "fp32_0035",      "fp32_0067",      "fp32_025"]
GPTQ_KEYS     = ["int8_gptq", "int8_gptq_0035", "int8_gptq_0067", "int8_gptq_025"]
QAT_KEYS      = ["mixed_p3",  "qat_mixed_0035", "qat_mixed_0067", "qat_mixed_025"]


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", format=ext)
    print(f"  Saved  results/plots/{name}.{{png,pdf}}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 1 — R-D Curve  (PSNR vs BPP, 2 panels)
# ══════════════════════════════════════════════════════════════════════════════
def plot_rd_curve() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    specs = [
        (FP32_KEYS,  "FP32 Baseline",            C_FP32, "-",  "o"),
        (GPTQ_KEYS,  "INT8 GPTQ",                C_GPTQ, "--", "s"),
        (QAT_KEYS,   "Mixed QAT Phase 3 (best)", C_QAT,  "-.", "^"),
    ]
    for ax, (ds_name, df) in zip(axes, [
            ("Kodak (24 images)",    kodak),
            ("pro_2020 (41 images)", pro),
    ]):
        for keys, label, color, ls, mk in specs:
            bpp   = _actual_bpp(df, keys)
            psnr  = [df.loc[k, "psnr"] for k in keys]
            ax.plot(bpp, psnr, color=color, linestyle=ls, marker=mk,
                    linewidth=2, markersize=7, label=label)
            for b, p, lm in zip(bpp, psnr, LAMBDAS):
                ax.annotate(f"λ={lm}", (b, p), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, color=color, alpha=0.75)

        ax.set_xlabel("Bits per pixel (BPP — actual bitstream)")
        ax.set_ylabel("RGB-PSNR (dB)")
        ax.set_title(ds_name)
        ax.legend(loc="lower right")

    fig.suptitle("Rate–Distortion Performance — LoC-LIC Quantization", y=1.01)
    fig.tight_layout()
    _save(fig, "01_rd_curve")


# ══════════════════════════════════════════════════════════════════════════════
# 2 — ΔPSNR bar chart  (quality gap vs FP32, per λ)
# ══════════════════════════════════════════════════════════════════════════════
def plot_delta_psnr() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    x, w = np.arange(len(LAMBDAS)), 0.35

    for ax, (ds_name, df) in zip(axes, [("Kodak", kodak), ("pro_2020", pro)]):
        fp32_p = np.array([df.loc[k, "psnr"] for k in FP32_KEYS])
        gptq_d = np.array([df.loc[k, "psnr"] for k in GPTQ_KEYS]) - fp32_p
        qat_d  = np.array([df.loc[k, "psnr"] for k in QAT_KEYS])  - fp32_p

        b1 = ax.bar(x - w/2, gptq_d, w, label="INT8 GPTQ",                color=C_GPTQ, alpha=0.85)
        b2 = ax.bar(x + w/2, qat_d,  w, label="Mixed QAT Phase 3 (best)", color=C_QAT,  alpha=0.85)

        for bar, val in zip(b1, gptq_d):
            ax.text(bar.get_x() + bar.get_width()/2, val - 0.03,
                    f"{val:.3f}", ha="center", va="top", fontsize=8)
        for bar, val in zip(b2, qat_d):
            ax.text(bar.get_x() + bar.get_width()/2, val - 0.03,
                    f"{val:.2f}", ha="center", va="top", fontsize=8)

        ax.axhline(0,    color="black", linewidth=0.9)
        ax.axhline(-1.0, color="grey",  linewidth=0.9, linestyle=":", alpha=0.7)
        ax.text(len(LAMBDAS) - 0.55, -1.05, "−1 dB target",
                ha="right", va="top", fontsize=8, color="grey")

        ax.set_xticks(x)
        ax.set_xticklabels([f"λ={l}" for l in LAMBDAS])
        ax.set_xlabel("Operating point (λ)")
        ax.set_ylabel("ΔRGB-PSNR vs FP32 baseline (dB)")
        ax.set_title(f"Quality Gap — {ds_name}")
        ax.legend()

    fig.suptitle("RGB-PSNR Degradation Relative to FP32 (same λ)", y=1.01)
    fig.tight_layout()
    _save(fig, "02_delta_psnr")


# ══════════════════════════════════════════════════════════════════════════════
# 3 — Full pipeline runtime  (encode + decode stacked bars)
# ══════════════════════════════════════════════════════════════════════════════
def plot_pipeline_runtime() -> None:
    # enc/dec times from evaluation_kodak.csv, lambda=0.0018, mean over N=15 passes
    labels = ["FP32\nBaseline", "INT8\nGPTQ", "INT4 QAT\nPhase 1", "Mixed QAT\nPhase 3\n(best)"]
    enc    = [427, 203, 320, 202]
    dec    = [287, 261, 357, 245]
    clrs   = [C_FP32, C_GPTQ, C_QAT1, C_QAT]
    base   = enc[0] + dec[0]
    x, w   = np.arange(len(labels)), 0.55

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # encode (lighter shade via alpha on a white-blended color)
    enc_clrs = [c + "99" for c in clrs]   # 60% alpha hex
    ax.bar(x, enc, w, color=enc_clrs, label="Encode")
    ax.bar(x, dec, w, bottom=enc, color=clrs, alpha=0.90, label="Decode")

    for i, (e, d) in enumerate(zip(enc, dec)):
        total = e + d
        spd   = base / total
        ax.text(i, total + 12, f"{spd:.2f}×\n({total} ms)",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        ax.text(i, e / 2,     f"Enc\n{e} ms",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.text(i, e + d / 2, f"Dec\n{d} ms",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(0, 900)
    ax.set_title("Full Pipeline Runtime — compress() + decompress()\n"
                 "Kodak 24 images (avg), λ=0.0018, NVIDIA RTX 3080")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "03_pipeline_runtime")


# ══════════════════════════════════════════════════════════════════════════════
# 4 — Complexity: BOPs + weight storage  (two sub-panels)
# ══════════════════════════════════════════════════════════════════════════════
def plot_complexity() -> None:
    models   = ["FP32\nBaseline", "INT8\nGPTQ", "INT4 QAT\nPhase 1", "Mixed QAT\nPhase 3\n(best)"]
    clrs     = [C_FP32, C_GPTQ, C_QAT1, C_QAT]
    x, w     = np.arange(4), 0.50

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # BOPs — activations stay FP32 (b_a=32) for all variants
    bops     = [37.4, 9.35, 4.68, 6.78]
    labels_b = ["37.4 T\n(1×)", "9.35 T\n(4×)", "4.68 T\n(8×)", "6.78 T\n(5.5×)"]
    ax = axes[0]
    bars = ax.bar(x, bops, w, color=clrs, alpha=0.88)
    for bar, val, lbl in zip(bars, bops, labels_b):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.4,
                lbl, ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Bit Operations (T BOPs)")
    ax.set_title("Computational Complexity (BOPs)\nBackbone 512×768, b_a=32 (FP32 activations)")
    ax.set_ylim(0, 46)

    # Weight storage
    sizes    = [268, 67, 34, 48]
    labels_s = ["268 MB\n(1×)", "67 MB\n(4×)", "34 MB\n(8×)", "48 MB\n(5.5×)"]
    ax = axes[1]
    bars = ax.bar(x, sizes, w, color=clrs, alpha=0.88)
    for bar, val, lbl in zip(bars, sizes, labels_s):
        ax.text(bar.get_x() + bar.get_width()/2, val + 3,
                lbl, ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Deployment weight size (MB)")
    ax.set_title("Model Weight Storage\n(True INT8/INT4 deployment format)")
    ax.set_ylim(0, 330)

    fig.suptitle("Complexity & Storage Reduction vs FP32 Baseline", y=1.01)
    fig.tight_layout()
    _save(fig, "04_complexity")


# ══════════════════════════════════════════════════════════════════════════════
# 5 — ONNX runtime  (horizontal bar, backbone only)
# ══════════════════════════════════════════════════════════════════════════════
def plot_onnx_runtime() -> None:
    entries = [
        ("PyTorch FP32\n(full model — unfair baseline)",  86.3, C_GREY),
        ("PyTorch FP32\n(backbone only — fair baseline)",  6.9, C_FP32),
        ("PyTorch FP32\n+ torch.compile",                  4.8, C_COMP),
        ("PyTorch Mixed QAT\n(fake-quant, backbone)",     62.8, C_GREY),
        ("ONNX Runtime FP32\n(CUDA EP)",                   7.0, C_FP32),
        ("ONNX Runtime INT8 PTQ\n(CUDA EP)",              13.7, C_RED),
        ("ONNX Runtime Mixed QAT INT8\n(CUDA EP)",         14.3, C_RED),
    ]
    labels, times, clrs = zip(*entries)
    baseline = 6.9
    speedups = [baseline / t for t in times]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(labels))
    ax.barh(y, times, 0.55, color=clrs, alpha=0.88)
    ax.axvline(baseline, color=C_FP32, linewidth=1.5, linestyle="--",
               alpha=0.6, label=f"Backbone baseline ({baseline} ms)")

    for i, (t, sp) in enumerate(zip(times, speedups)):
        ax.text(t + 0.7, i, f"{t:.1f} ms   ({sp:.2f}×)",
                va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Inference latency (ms) — 512×768 image, 100 runs")
    ax.set_title("ONNX Runtime Backbone Benchmark\nNVIDIA RTX 3080, CUDA Execution Provider")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.legend(fontsize=9)

    ax.annotate(
        "ONNX INT8 is ~2× SLOWER than FP32\n"
        "(QDQ ops fall back to CPU → 42 memory transfers)\n"
        "GPU INT8 speedup requires TensorRT.",
        xy=(13.7, 5.5), xytext=(45, 5.7),
        fontsize=8.5, color="#B71C1C",
        arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFCDD2", alpha=0.8),
    )

    fig.tight_layout()
    _save(fig, "05_onnx_runtime")


# ══════════════════════════════════════════════════════════════════════════════
# 6 — BPP reduction at each λ  (% savings vs FP32)
# ══════════════════════════════════════════════════════════════════════════════
def plot_bpp_reduction() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    x, w = np.arange(len(LAMBDAS)), 0.35

    for ax, (ds_name, df) in zip(axes, [("Kodak", kodak), ("pro_2020", pro)]):
        fp32_b = np.array(_actual_bpp(df, FP32_KEYS))
        gptq_r = (1 - np.array(_actual_bpp(df, GPTQ_KEYS)) / fp32_b) * 100
        qat_r  = (1 - np.array(_actual_bpp(df, QAT_KEYS))  / fp32_b) * 100

        b1 = ax.bar(x - w/2, gptq_r, w, label="INT8 GPTQ",                color=C_GPTQ, alpha=0.85)
        b2 = ax.bar(x + w/2, qat_r,  w, label="Mixed QAT Phase 3 (best)", color=C_QAT,  alpha=0.85)

        for bar, val in zip(b1, gptq_r):
            ypos = val + 0.5 if val >= 0 else val - 1.0
            va   = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f"{val:+.1f}%", ha="center", va=va, fontsize=9)
        for bar, val in zip(b2, qat_r):
            ypos = val + 0.5 if val >= 0 else val - 1.0
            va   = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f"{val:+.1f}%", ha="center", va=va, fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([f"λ={l}" for l in LAMBDAS])
        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"λ={l}" for l in LAMBDAS])
        ax.set_xlabel("Operating point (λ)")
        ax.set_ylabel("BPP difference vs FP32 (%) — negative = fewer bits")
        ax.set_title(f"Bitrate Difference — {ds_name}")
        ax.set_ylim(-15, 15)
        ax.legend()

    fig.suptitle("Bitrate Difference vs FP32 Baseline (corrected evaluation)", y=1.01)
    fig.tight_layout()
    _save(fig, "06_bpp_reduction")


# ══════════════════════════════════════════════════════════════════════════════
# 7 — Quality vs model size scatter  (λ=0.0018, Kodak)
# ══════════════════════════════════════════════════════════════════════════════
def plot_quality_vs_size() -> None:
    # All points from merged CSV (λ=0.0018, Kodak) — no hardcoded values
    points = [
        ("FP32 Baseline",            "fp32",      C_FP32, "o",  13, ( 7,  6)),
        ("INT8 GPTQ",                "int8_gptq", C_GPTQ, "s",  11, ( 7, -9)),
        ("INT8 PTQ",                 "int8_ptq",  C_GREY, "D",   9, ( 7,  5)),
        ("INT4 GPTQ",                "int4_gptq", C_GREY, "v",   9, ( 7,  5)),
        ("INT4 QAT Phase 1",         "int4_qat",  C_QAT1, "P",  10, ( 7,  5)),
        ("Mixed QAT Phase 2",        "mixed_p2",  C_QAT1, "^",  10, ( 7, -9)),
        ("Mixed QAT Phase 3 (best)", "mixed_p3",  C_QAT,  "^",  13, ( 7,  6)),
        ("KD QAT Phase 4 (v1)",       "kd_p4",     C_GREY, "X",   9, ( 7,  5)),
        ("KD QAT Phase 4b (v2)",      "kd_v2",     C_QAT1, "X",  10, ( 7, -9)),
    ]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for label, key, color, mk, ms, (offx, offy) in points:
        rgb_psnr = kodak.loc[key, "psnr"]
        size_mb  = kodak.loc[key, "deploy_size_mb"]
        ax.scatter(size_mb, rgb_psnr, s=ms**2, color=color, marker=mk,
                   zorder=5, edgecolors="white", linewidths=0.9,
                   alpha=1.0 if color != C_GREY else 0.80)
        ax.annotate(label, (size_mb, rgb_psnr),
                    xytext=(offx, offy), textcoords="offset points",
                    fontsize=8.5,
                    fontweight="bold" if "best" in label else "normal",
                    color=color)

    # Arrow: FP32 -> Mixed QAT Phase 3
    fp32_p  = kodak.loc["fp32",     "psnr"]
    mixed_p = kodak.loc["mixed_p3", "psnr"]
    fp32_s  = kodak.loc["fp32",     "deploy_size_mb"]
    mix_s   = kodak.loc["mixed_p3", "deploy_size_mb"]
    ax.annotate("",
                xy=(mix_s, mixed_p), xytext=(fp32_s, fp32_p),
                arrowprops=dict(arrowstyle="->", color="grey", lw=1.3, linestyle="dashed"))
    ratio = fp32_s / mix_s
    delta = mixed_p - fp32_p
    ax.text((fp32_s + mix_s) / 2, (fp32_p + mixed_p) / 2 + 0.08,
            f"{ratio:.1f}× smaller / {delta:.3f} dB", ha="center", fontsize=8.5, color="grey")

    ax.set_xlabel("Deployment weight size (MB)")
    ax.set_ylabel("Kodak RGB-PSNR (dB)  [lambda = 0.0018]")
    ax.set_title("Quality vs. Model Size Trade-off\nAll quantization methods — measured codec eval")
    ax.set_xlim(-10, 310)
    ax.set_ylim(25.0, 30.0)
    fig.tight_layout()
    _save(fig, "07_quality_vs_size")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nGenerating 7 thesis plots …\n")
    plot_rd_curve();         print("  [1/7] R-D curve")
    plot_delta_psnr();       print("  [2/7] ΔPSNR bar chart")
    plot_pipeline_runtime(); print("  [3/7] Pipeline runtime")
    plot_complexity();       print("  [4/7] Complexity & storage")
    plot_onnx_runtime();     print("  [5/7] ONNX runtime")
    plot_bpp_reduction();    print("  [6/7] BPP reduction")
    plot_quality_vs_size();  print("  [7/7] Quality vs size scatter")
    print(f"\nDone — all plots in results/plots/\n")
