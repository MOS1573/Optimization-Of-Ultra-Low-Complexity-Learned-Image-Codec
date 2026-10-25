#!/usr/bin/env python3
"""
Full end-to-end pipeline benchmark for LoC-LIC.

Measures the COMPLETE workflow for each model:
  encode: model.compress(image)      → produces bitstream
  decode: model.decompress(strings)  → reconstructs image

This includes EVERYTHING: NN transforms + entropy probability computation
+ arithmetic range coder. This is what the user actually experiences.

Models compared at λ=0.0018:
  1. FP32 baseline
  2. INT8 GPTQ (PTQ, weight-only)
  3. Mixed QAT Phase 1 (INT4/INT8, original LR)
  4. Mixed QAT Phase 3 ★ (INT4/INT8, best model)

Usage:
  python benchmark_pipeline.py --images /path/to/Kodak --n_images 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torchvision import transforms as T

REPO_ROOT = Path(__file__).parents[3]
for _p in [str(REPO_ROOT), str(REPO_ROOT / "Optimization" / "mixed-precision QAT")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.get_model import get_model       # noqa: E402
from utils.config import model_config        # noqa: E402
from utils.func import update_registered_buffers  # noqa: E402

_CFG   = REPO_ROOT / "configs" / "models" / "loclic.yaml"
_LMBDA = 0.0018

MODELS = [
    ("FP32 Baseline",
     REPO_ROOT / "checkpoints/loc-lic/0.0018.pth.tar"),
    ("INT8 GPTQ (λ=0.0018)",
     REPO_ROOT / "results/legacy/old_kodak_test/gptq_second_try/LoC_LIC_int8_gptq_8bit.pt"),
    ("Mixed QAT Phase 1 (λ=0.0018)",
     REPO_ROOT / "results/4-bit/phase2-mixed/lambda-0018/LoC_LIC_mixed_qat.pt"),
    ("Mixed QAT Phase 3 ★ (λ=0.0018)",
     REPO_ROOT / "results/4-bit/phase3-mixed-finetuned/lambda-0018/LoC_LIC_mixed_qat.pt"),
]


def _load(ckpt: Path, device: torch.device) -> nn.Module:
    cfg = model_config(str(_CFG))
    net, _, _ = get_model(cfg, argparse.Namespace(lmbda=_LMBDA), device)
    raw   = torch.load(ckpt, map_location=device, weights_only=False)
    state = raw.get("state_dict", raw.get("model", raw)) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        return state.eval().to(device)
    for attr, keys in [
        ("entropy_bottleneck",   ["_quantized_cdf", "_offset", "_cdf_length"]),
        ("gaussian_conditional", ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"]),
    ]:
        if hasattr(net, attr):
            update_registered_buffers(getattr(net, attr), attr, keys, state)
    nn.Module.load_state_dict(net, state, strict=False)   # bypass LoC_LIC override
    net.update(force=True)   # rebuild CDF tables for arithmetic coder
    return net.eval().to(device)


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    img = PILImage.open(path).convert("RGB")
    t   = T.ToTensor()(img).unsqueeze(0).to(device)
    # pad to multiple of 64
    _, _, h, w = t.shape
    pH = (64 - h % 64) % 64
    pW = (64 - w % 64) % 64
    if pH or pW:
        import torch.nn.functional as F
        t = F.pad(t, (0, pW, 0, pH), mode="reflect")
    return t


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a - b) ** 2).mean().item()
    return 10 * np.log10(1.0 / mse) if mse > 0 else 100.0


def benchmark_model(model: nn.Module, images: list[torch.Tensor],
                    device: torch.device, n_warmup: int = 2) -> dict:
    enc_times, dec_times, psnrs, bpps = [], [], [], []

    for i, x in enumerate(images):
        _, _, h, w = x.shape

        # warmup on first image
        if i < n_warmup:
            with torch.no_grad():
                out = model.compress(x)
                model.decompress(out["strings"], out["shape"])
            if device.type == "cuda":
                torch.cuda.synchronize()

        # encode
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            compressed = model.compress(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        enc_times.append((time.perf_counter() - t0) * 1000)

        # decode
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.decompress(compressed["strings"], compressed["shape"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        dec_times.append((time.perf_counter() - t0) * 1000)

        # quality metrics
        x_hat = out["x_hat"].clamp(0, 1)
        x_hat = x_hat[:, :, :h-((h%64 or 64)%64) or h, :w-((w%64 or 64)%64) or w]
        x_orig = x[:, :, :x_hat.shape[2], :x_hat.shape[3]]
        psnrs.append(psnr(x_orig, x_hat))

        n_bits = sum(len(s[0]) for s in compressed["strings"]) * 8
        bpps.append(n_bits / (h * w))

    return {
        "enc_ms":  float(np.mean(enc_times)),
        "enc_std": float(np.std(enc_times)),
        "dec_ms":  float(np.mean(dec_times)),
        "dec_std": float(np.std(dec_times)),
        "total_ms": float(np.mean(enc_times) + np.mean(dec_times)),
        "psnr_db": float(np.mean(psnrs)),
        "bpp":     float(np.mean(bpps)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images",   default="path/to/dataset/Kodak")
    p.add_argument("--n_images", type=int, default=6, help="number of Kodak images to use")
    p.add_argument("--device",   choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--output_dir", type=Path,
                   default=REPO_ROOT / "results" / "pipeline_benchmark")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load images once
    img_paths = sorted(Path(args.images).glob("*.png"))[:args.n_images]
    if not img_paths:
        img_paths = sorted(Path(args.images).glob("*.jpg"))[:args.n_images]
    print(f"\nLoaded {len(img_paths)} images from {args.images}")
    images = [load_image(p, device) for p in img_paths]
    h, w = images[0].shape[2], images[0].shape[3]

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device: {gpu_name}  |  Image size: {h}×{w}\n")

    results = {}
    for label, ckpt in MODELS:
        if ckpt is None or not Path(ckpt).exists():
            print(f"[SKIP] {label}  (checkpoint not found: {ckpt})\n")
            continue
        print(f"Loading: {label} …")
        try:
            model = _load(Path(ckpt), device)
        except Exception as e:
            print(f"  [ERROR] Could not load: {e}\n")
            continue

        print(f"Benchmarking {len(images)} images …")
        r = benchmark_model(model, images, device)
        results[label] = r
        print(f"  Encode: {r['enc_ms']:.0f} ± {r['enc_std']:.0f} ms")
        print(f"  Decode: {r['dec_ms']:.0f} ± {r['dec_std']:.0f} ms")
        print(f"  Total:  {r['total_ms']:.0f} ms")
        print(f"  PSNR:   {r['psnr_db']:.2f} dB   BPP: {r['bpp']:.4f}\n")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Report ────────────────────────────────────────────────────────────────
    if not results:
        print("No models benchmarked.")
        return

    baseline_total = next(iter(results.values()))["total_ms"]
    baseline_psnr  = next(iter(results.values()))["psnr_db"]

    hdr = (f"\n{'='*80}\n"
           f"  FULL PIPELINE BENCHMARK  —  {h}×{w} image, {len(images)} Kodak images\n"
           f"  Device: {gpu_name}\n"
           f"{'='*80}\n"
           f"{'Model':<28} {'Enc(ms)':>8} {'Dec(ms)':>8} {'Total':>8} "
           f"{'Speedup':>8} {'PSNR(dB)':>9} {'BPP':>7}\n"
           f"{'-'*80}")
    print(hdr)
    table = hdr + "\n"

    for label, r in results.items():
        sx = f"{baseline_total/r['total_ms']:.2f}×"
        dp = f"{r['psnr_db']-baseline_psnr:+.2f}"
        row = (f"{label:<28} {r['enc_ms']:>8.0f} {r['dec_ms']:>8.0f} "
               f"{r['total_ms']:>8.0f} {sx:>8} {r['psnr_db']:>8.2f} "
               f"({'Δ'+dp+' dB':>9}) {r['bpp']:>7.4f}")
        print(row)
        table += row + "\n"

    print("=" * 80)
    table += "=" * 80

    out_path = args.output_dir / "pipeline_results.txt"
    with open(out_path, "w") as f:
        import datetime
        f.write(f"LoC-LIC Full Pipeline Benchmark\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"GPU: {gpu_name}  |  Image: {h}×{w}  |  N={len(images)} images\n\n")
        f.write(table)
    print(f"\nSaved → {out_path}\n")


if __name__ == "__main__":
    main()
