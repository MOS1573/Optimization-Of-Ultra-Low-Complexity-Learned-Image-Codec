#!/usr/bin/env python3
"""
Layer sensitivity analysis for mixed-precision QAT.

Tests each quantizable Conv2d / ConvTranspose2d / Linear in isolation:
applies INT4 fake-quantization to only that layer, runs a forward pass on a
small image sample, and measures the PSNR drop vs the FP32 baseline.

Layers are then ranked by sensitivity and split into two tiers:
  - PSNR drop > --int8_threshold  → INT8  (preserve quality)
  - PSNR drop ≤ --int8_threshold  → INT4  (save memory)

Output: a JSON file consumable by train_qat_mixed.py via --layer_bits_file.

Usage:
  cd /path/to/repo
  python "Optimization/4-bit/4-bit Mixed-Precision/sensitivity_analysis.py" \\
      --checkpoint checkpoints/loc-lic/0.0018.pth.tar \\
      --images /path/to/kodak \\
      --output "Optimization/4-bit/4-bit Mixed-Precision/layer_bits.json" \\
      --n_images 8 \\
      --int8_threshold 0.05

  SLURM:
      sbatch "Optimization/4-bit/4-bit Mixed-Precision/run_sensitivity.sh"
"""

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import transforms

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from models.get_model import get_model
from utils.config import model_config
from utils.utils import patch_compressai_buffer_loading

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qat_modules_mixed import fake_quant_ste, _should_skip, _QUANTIZABLE, _collect_conv_names

patch_compressai_buffer_loading()
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a.clamp(0, 1), b.clamp(0, 1)).item()
    if mse == 0:
        return float("inf")
    return -10.0 * torch.log10(torch.tensor(mse)).item()


@contextmanager
def _quantize_layer(module: nn.Module, bits: int):
    """Temporarily replace module.weight with its fake-quantized version."""
    maxv = float(2 ** (bits - 1) - 1)
    orig = module.weight.data.clone()
    with torch.no_grad():
        module.weight.data = fake_quant_ste(orig.float(), maxv).to(orig.dtype)
    try:
        yield
    finally:
        module.weight.data = orig


def load_images(images_dir: str, n_images: int, patch_size: int, device: torch.device):
    tf = transforms.Compose([transforms.CenterCrop(patch_size), transforms.ToTensor()])
    paths = sorted([
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ])[:n_images]
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if img.size[0] < patch_size or img.size[1] < patch_size:
            continue
        tensors.append(tf(img).unsqueeze(0).to(device))
    print(f"  Loaded {len(tensors)} images ({patch_size}×{patch_size}) from {images_dir}")
    return tensors


def fp32_baseline(net: nn.Module, images: list) -> list:
    net.eval()
    results = []
    with torch.no_grad():
        for img in images:
            out = net(img)
            x_hat = out["x_hat"] if isinstance(out, dict) else out[0]
            results.append(x_hat.detach())
    return results


def measure_psnr_with_layer_quantized(
    net: nn.Module,
    module: nn.Module,
    images: list,
    baselines: list,
    bits: int,
) -> float:
    net.eval()
    drops = []
    with _quantize_layer(module, bits):
        with torch.no_grad():
            for img, base in zip(images, baselines):
                out = net(img)
                x_hat = out["x_hat"] if isinstance(out, dict) else out[0]
                p_fp32 = psnr(base, base)          # infinite (baseline vs itself)
                p_q    = psnr(base, x_hat.detach())
                # psnr drop = how much worse vs FP32
                # compare quantized output vs FP32 output (not vs original image)
                drops.append(psnr(base, base) - psnr(base, x_hat.detach()))
    # Use median to be robust to outlier images
    drops.sort()
    return drops[len(drops) // 2]  # median drop


def measure_absolute_psnr_drop(
    net: nn.Module,
    module: nn.Module,
    images: list,
    fp32_outputs: list,
    bits: int,
) -> float:
    """PSNR(fp32_output, quantized_output) — measures degradation from quantizing this layer."""
    net.eval()
    drops = []
    with _quantize_layer(module, bits):
        with torch.no_grad():
            for img, fp32_out in zip(images, fp32_outputs):
                out = net(img)
                x_hat = out["x_hat"] if isinstance(out, dict) else out[0]
                # Drop: how many dB did we lose by quantizing this layer?
                p = psnr(fp32_out, x_hat.detach())
                drop = 100.0 - p if p != float("inf") else 0.0
                # Actually just compute psnr drop directly
                drops.append(p)
    # Return mean PSNR when this layer is quantized (lower = more sensitive)
    return sum(drops) / len(drops)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True,
                   help="FP32 checkpoint path (e.g. checkpoints/loc-lic/0.0018.pth.tar)")
    p.add_argument("--images",        required=True,
                   help="Directory of images to test on (e.g. Kodak)")
    p.add_argument("--output",        default="Optimization/4-bit/4-bit Mixed-Precision/layer_bits.json",
                   help="Output JSON path")
    p.add_argument("--config",        default="loclic",
                   help="Model config name in configs/models/")
    p.add_argument("--n_images",      type=int, default=8,
                   help="Number of images to use for analysis (default: 8)")
    p.add_argument("--patch_size",    type=int, default=512,
                   help="Image crop size for analysis (default: 512)")
    p.add_argument("--bits",          type=int, default=4,
                   help="Bit-width to test for sensitivity (default: 4 = INT4)")
    p.add_argument("--int8_threshold",type=float, default=0.05,
                   help="PSNR drop threshold (dB): layers with drop > this go to INT8 (default: 0.05)")
    p.add_argument("--lmbda",         type=float, default=0.0018)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f" LoC-LIC Sensitivity Analysis")
    print(f" Checkpoint: {args.checkpoint}")
    print(f" Images:     {args.images}  (n={args.n_images}, crop={args.patch_size})")
    print(f" INT8 threshold: {args.int8_threshold} dB")
    print(f"{'='*60}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    config_path = _REPO / "configs" / "models" / f"{args.config}.yaml"
    model_cfg   = model_config(str(config_path))

    class _FakeArgs:
        lmbda = args.lmbda
    net, _, _ = get_model(model_cfg, _FakeArgs(), device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    # Strip 'module.' prefix if saved from DDP
    state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval().to(device)
    print(f"FP32 checkpoint loaded: {args.checkpoint}")

    # ── Images ────────────────────────────────────────────────────────────────
    images = load_images(args.images, args.n_images, args.patch_size, device)

    # ── FP32 baseline outputs ─────────────────────────────────────────────────
    print("\nComputing FP32 baseline...")
    fp32_outputs = []
    with torch.no_grad():
        for img in images:
            out = net(img)
            x_hat = out["x_hat"] if isinstance(out, dict) else out[0]
            fp32_outputs.append(x_hat.detach().clone())

    fp32_psnr = sum(psnr(img, out) for img, out in zip(images, fp32_outputs)) / len(images)
    print(f"FP32 reconstruction PSNR (vs input): {fp32_psnr:.3f} dB")

    # ── Collect quantizable layers ────────────────────────────────────────────
    conv_names = _collect_conv_names(net)
    first_conv = conv_names[0] if conv_names else None
    last_conv  = conv_names[-1] if conv_names else None

    candidates = []
    for name, module in net.named_modules():
        if not isinstance(module, _QUANTIZABLE):
            continue
        if _should_skip(name, module):
            continue
        candidates.append((name, module))

    print(f"\nFound {len(candidates)} quantizable layers. Testing INT{args.bits} sensitivity...\n")

    # ── Per-layer sensitivity ─────────────────────────────────────────────────
    results = []
    for i, (name, module) in enumerate(candidates):
        is_first_last = name in (first_conv, last_conv)
        tag = " [first/last conv]" if is_first_last else ""

        psnr_vals = []
        with _quantize_layer(module, args.bits):
            with torch.no_grad():
                for img, fp32_out in zip(images, fp32_outputs):
                    out = net(img)
                    x_hat = out["x_hat"] if isinstance(out, dict) else out[0]
                    psnr_vals.append(psnr(fp32_out, x_hat.detach()))

        mean_psnr_q = sum(psnr_vals) / len(psnr_vals)
        # Drop = how much worse than FP32 when only this layer is quantized
        # FP32 output vs quantized output; FP32 output psnr with itself = inf,
        # so we measure degradation of x_hat_q relative to x_hat_fp32
        # A large drop = this layer is highly sensitive.
        # Since psnr(fp32_out, fp32_out) = inf, we define:
        #   drop ≈ 40 - mean_psnr_q  (40 dB is effectively "no change" for this model)
        # Better: compare against a very-high-quality reference.
        # We use the actual FP32 psnr computed against original input, and
        # track how much psnr(fp32_out, x_hat_q) deviates from "infinity".
        # Practically: drop = max_possible - mean_psnr_q, where max = ~60 dB for near-identical.
        # Simplest meaningful metric: lower mean_psnr_q → higher sensitivity.
        drop = -mean_psnr_q  # rank: most sensitive = most negative psnr (lower psnr_q)

        results.append({
            "name":           name,
            "psnr_q":         mean_psnr_q,
            "is_first_last":  is_first_last,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(candidates):
            print(f"  [{i+1:3d}/{len(candidates)}] {name[:60]:<60} psnr_q={mean_psnr_q:6.2f} dB{tag}")

    # ── Rank by sensitivity (lower psnr_q = more sensitive) ──────────────────
    results.sort(key=lambda r: r["psnr_q"])

    # ── Assign bits ───────────────────────────────────────────────────────────
    # Find the FP32 baseline psnr of the output against itself (effectively inf),
    # but for a practical threshold we use:
    #   If psnr(fp32_out, quantized_out) < (fp32_psnr - threshold) → INT8
    # where fp32_psnr is the FP32 model PSNR against the original image.
    # Any layer whose quantization costs more than `int8_threshold` dB → INT8.

    # For simplicity: layers where psnr_q (fp32_out vs quantized_out) drops below
    # a large value (e.g. 45 dB) are considered sensitive.
    # A more principled threshold: layers where psnr_q < high_psnr_cutoff.
    # We use the distribution: top quartile by sensitivity → INT8.

    # Count how many layers to assign INT8 based on the threshold.
    # psnr_q measures: if only this one layer is INT4, how similar is the output to FP32?
    # A psnr_q < 45 dB indicates the layer contributes > 0.1% signal change.
    # We use the user-specified threshold relative to the max observed psnr_q.

    max_psnr_q = max(r["psnr_q"] for r in results)

    layer_bits = {}
    sensitivity_log = []
    n_int8 = n_int4 = n_fp32 = 0

    for r in results:
        name = r["name"]
        psnr_q_val = r["psnr_q"]

        if r["is_first_last"]:
            bits_assigned = 32  # always FP32 for first/last conv
            n_fp32 += 1
        elif (max_psnr_q - psnr_q_val) > args.int8_threshold:
            bits_assigned = 8
            n_int8 += 1
        else:
            bits_assigned = 4
            n_int4 += 1

        layer_bits[name] = bits_assigned
        sensitivity_log.append({
            "name":    name,
            "psnr_q":  round(psnr_q_val, 4),
            "drop_db": round(max_psnr_q - psnr_q_val, 4),
            "bits":    bits_assigned,
        })

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" Sensitivity Analysis Results")
    print(f"{'='*60}")
    print(f" Total layers: {len(results)}")
    print(f"   INT8 (sensitive):  {n_int8}  (drop > {args.int8_threshold} dB)")
    print(f"   INT4 (standard):   {n_int4}")
    print(f"   FP32 (first/last): {n_fp32}")
    print(f"\n Top-20 most sensitive layers (→ INT8):")
    for r in sensitivity_log[:20]:
        print(f"   drop={r['drop_db']:5.3f} dB  bits={r['bits']}  {r['name']}")
    print(f"\n Bottom-10 least sensitive layers (→ INT4):")
    for r in sensitivity_log[-10:]:
        print(f"   drop={r['drop_db']:5.3f} dB  bits={r['bits']}  {r['name']}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "config": {
            "checkpoint":      args.checkpoint,
            "images_dir":      args.images,
            "n_images":        len(images),
            "patch_size":      args.patch_size,
            "bits_tested":     args.bits,
            "int8_threshold":  args.int8_threshold,
            "fp32_psnr":       round(fp32_psnr, 4),
        },
        "summary": {
            "n_int8":  n_int8,
            "n_int4":  n_int4,
            "n_fp32":  n_fp32,
        },
        "layer_bits": layer_bits,
        "sensitivity_ranking": sensitivity_log,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n Layer bits saved to: {out_path}")
    print(f" Run mixed-precision QAT with:")
    print(f"   --layer_bits_file {out_path}")


if __name__ == "__main__":
    main()
