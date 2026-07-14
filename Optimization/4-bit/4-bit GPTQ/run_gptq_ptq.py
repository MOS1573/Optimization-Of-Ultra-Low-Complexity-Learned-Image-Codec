#!/usr/bin/env python3
"""
GPTQ 4-bit PTQ Pipeline for LoC-LIC.

Pipeline:
  1. Load FP32 model from checkpoint
  2. Build calibration dataloader from real images (Kodak PNG crops)
  3. Run GPTQ: accumulate per-layer Hessians, then apply column-wise weight update
  4. Evaluate with fast_eval (forward pass + likelihoods-based BPP)
  5. Save results and print FP32 vs INT4 comparison

4-bit differs from 8-bit in:
  - bits=4 (maxval=7 instead of 127)
  - percdamp=0.02 (slightly higher for Hessian stability)
  - num_calibration_samples=256 (more samples for a better Hessian estimate)
  - Expected PSNR degradation: -1 to -5 dB depending on model sensitivity

Usage (GPU node):
  python run_gptq_ptq.py \\
      --checkpoint ../../checkpoints/loc-lic/0.0018.pth.tar \\
      --calibration_data ../../datasets/kodak \\
      --test_dataset ../../datasets/kodak \\
      --output_dir ../../results/gptq_4bit \\
      --lmbda 0.0018
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import transforms

# ── Repo root on sys.path ────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent.parent.parent  # loc-lic-new/
sys.path.insert(0, str(_REPO))

from models.get_model import get_model
from utils.config import model_config, get_device
from utils.args import test_options, get_args
from utils.metrics import ImageMetric, write_metrics_to_csv
from utils.utils import torch2img, write_uints, write_body, read_uints, read_body, filesize
from utils.image import read_image

from quantize_gptq import GPTQPipeline, QuantConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(net, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    # LoC_LIC.load_state_dict() does not accept a 'strict' kwarg
    net.load_state_dict(state)
    logger.info("Checkpoint loaded.")
    return net


def save_results(model, scales, summary, output_dir: str, model_name: str):
    os.makedirs(output_dir, exist_ok=True)
    pt_path = os.path.join(output_dir, f"{model_name}.pt")
    torch.save({"state_dict": model.state_dict(), "quant_summary": summary}, pt_path)
    logger.info(f"Saved quantized model: {pt_path}")

    meta = {**summary, "scales": {k: v.tolist() for k, v in scales.items()}}
    json_path = os.path.join(output_dir, f"{model_name}_gptq_config.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved GPTQ config:     {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Calibration data
# ─────────────────────────────────────────────────────────────────────────────

class PatchDataset(torch.utils.data.Dataset):
    """Random 256×256 crops from PNG/JPG images for Hessian calibration."""

    def __init__(self, images_dir: str, num_patches: int = 256, patch_size: int = 256,
                 seed: int = 42):
        self.patch_size = patch_size
        rng = torch.Generator()
        rng.manual_seed(seed)

        img_paths = sorted(
            p for p in Path(images_dir).iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if not img_paths:
            raise FileNotFoundError(f"No images found in {images_dir}")

        self.patches: list = []
        to_tensor = transforms.ToTensor()

        patches_per_image = max(1, (num_patches + len(img_paths) - 1) // len(img_paths))
        for img_path in img_paths:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            W, H = img.size
            if W < patch_size or H < patch_size:
                img = img.resize((max(W, patch_size), max(H, patch_size)), Image.BILINEAR)
                W, H = img.size
            t = to_tensor(img)
            for _ in range(patches_per_image):
                x0 = torch.randint(0, W - patch_size + 1, (1,), generator=rng).item()
                y0 = torch.randint(0, H - patch_size + 1, (1,), generator=rng).item()
                self.patches.append(t[:, y0:y0 + patch_size, x0:x0 + patch_size])
                if len(self.patches) >= num_patches:
                    break
            if len(self.patches) >= num_patches:
                break

        logger.info(f"Calibration dataset: {len(self.patches)} patches from {images_dir}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]


def build_calibration_loader(images_dir: str, num_samples: int = 256,
                             batch_size: int = 4, patch_size: int = 256):
    dataset = PatchDataset(images_dir, num_patches=num_samples, patch_size=patch_size)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=True, drop_last=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (likelihoods-based BPP — no RANS coding needed)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_fast(images_dir: str, net: torch.nn.Module, output_dir: str,
                  dataset_name: str, quality: str, model_name: str) -> dict:
    """
    Fast evaluation: one forward pass per image.
    BPP = sum(-log2(likelihoods)) / (H × W) — same as the training objective.
    """
    img_out_dir = os.path.join(output_dir, model_name, "images", dataset_name, quality)
    os.makedirs(img_out_dir, exist_ok=True)

    per_img_csv = os.path.join(img_out_dir, "results.csv")
    avg_csv     = os.path.join(img_out_dir, "results_avg.csv")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    net.eval()
    device = next(net.parameters()).device
    image_metric = ImageMetric(device="cpu")

    img_paths = sorted(
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    logger.info(f"Evaluating {len(img_paths)} images  [{model_name}]")

    metrics_dict: dict = {}

    with torch.inference_mode():
        for i, img_path in enumerate(img_paths):
            image_name = img_path.stem

            img = read_image(str(img_path), device).to(device)
            _, _, H, W = img.shape
            pad_h = (64 - H % 64) % 64
            pad_w = (64 - W % 64) % 64
            img_pad = F.pad(img, (0, pad_w, 0, pad_h))

            out = net(img_pad)
            x_hat = out["x_hat"][:, :, :H, :W].clamp(0.0, 1.0)

            num_pixels = H * W
            bpp = sum(
                (-torch.log2(like).sum()) / num_pixels
                for like in out["likelihoods"].values()
            ).item()

            size_bytes = max(int(bpp * num_pixels / 8), 1)
            metrics = image_metric.metric_image(output=x_hat.cpu(), target=img.cpu(),
                                                size_of_image=size_bytes)
            metrics["bpp"] = bpp
            metrics_dict = image_metric.append_to_metrics_dict(metrics, metrics_dict)

            metrics.update(model_name=model_name, dataset_name=dataset_name,
                           quality=quality)

            torch2img(x_hat).save(os.path.join(img_out_dir, image_name + ".png"))
            write_metrics_to_csv(metrics, per_img_csv, overwrite=False)

            if (i + 1) % max(1, len(img_paths) // 5) == 0 or (i + 1) == len(img_paths):
                logger.info(f"  [{i+1:2d}/{len(img_paths)}]  {image_name:<10s} "
                            f"PSNR={metrics.get('psnr', 0):.2f}  BPP={bpp:.4f}")

    avg = image_metric.get_avg_metrics(metrics_dict)
    avg.update(model_name=model_name, dataset_name=dataset_name, quality=quality)
    write_metrics_to_csv(avg, avg_csv, overwrite=True)
    write_metrics_to_csv(avg, os.path.join(metrics_dir, f"average_{model_name}.csv"),
                         overwrite=True)

    logger.info(f"Results → {img_out_dir}")
    logger.info(f"  PSNR   = {avg.get('psnr',   0):.4f} dB")
    logger.info(f"  SSIM   = {avg.get('ssim',   0):.4f}")
    logger.info(f"  MS-SSIM= {avg.get('msssim', 0):.4f}")
    logger.info(f"  LPIPS  = {avg.get('lpips',  0):.4f}")
    logger.info(f"  BPP    = {avg.get('bpp',    0):.4f}")
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(fp32: dict, ptq: dict, fp32_name: str = "FP32",
                     ptq_name: str = "INT4-GPTQ"):
    metrics = ["psnr", "ssim", "msssim", "lpips", "bpp"]
    labels  = ["PSNR (dB)", "SSIM", "MS-SSIM", "LPIPS ↓", "BPP"]
    fmts    = [".4f", ".4f", ".4f", ".4f", ".4f"]

    line = "=" * 68
    logger.info("\n" + line)
    logger.info("  COMPARISON: FP32 vs INT4 PTQ (GPTQ) — Kodak")
    logger.info(line)
    logger.info(f"  {'Metric':<12}  {fp32_name:>12}  {ptq_name:>14}  {'Δ':>10}  {'Δ%':>8}")
    logger.info("-" * 68)
    for key, label, fmt in zip(metrics, labels, fmts):
        v_fp32 = fp32.get(key, float("nan"))
        v_ptq  = ptq.get(key, float("nan"))
        delta  = v_ptq - v_fp32
        pct    = (delta / abs(v_fp32) * 100) if v_fp32 != 0 else float("nan")
        logger.info(f"  {label:<12}  {v_fp32:>{12}{fmt}}  "
                    f"{v_ptq:>{14}{fmt}}  {delta:>+10.4f}  {pct:>+7.2f}%")
    logger.info(line)


def save_comparison_csv(fp32: dict, ptq: dict, output_dir: str):
    path = os.path.join(output_dir, "comparison_fp32_vs_gptq.csv")
    metrics = ["psnr", "ssim", "msssim", "lpips", "bpp"]
    rows = []
    for key in metrics:
        v_fp32 = fp32.get(key, float("nan"))
        v_ptq  = ptq.get(key, float("nan"))
        delta  = v_ptq - v_fp32
        pct    = (delta / abs(v_fp32) * 100) if v_fp32 != 0 else float("nan")
        rows.append({"metric": key, "fp32": v_fp32, "gptq_int4": v_ptq,
                     "delta": delta, "delta_pct": f"{pct:+.2f}%"})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "fp32", "gptq_int4",
                                               "delta", "delta_pct"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Comparison CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GPTQ 4-bit PTQ for LoC-LIC"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pth.tar checkpoint")
    p.add_argument("--calibration_data", required=True,
                   help="Directory with calibration images (PNG/JPG)")
    p.add_argument("--test_dataset", required=True,
                   help="Directory with test images (Kodak PNG)")
    p.add_argument("--output_dir", default="results/gptq_4bit",
                   help="Where to write results (default: results/gptq_4bit)")
    p.add_argument("--config", default="loclic",
                   help="Model config YAML name in configs/models/ (default: loclic)")
    p.add_argument("--lmbda", type=float, default=0.0018,
                   help="Lambda quality value used during training (default: 0.0018)")
    p.add_argument("--num_calibration_samples", type=int, default=256,
                   help="Number of 256×256 patches for calibration (default: 256)")
    p.add_argument("--calibration_batch_size", type=int, default=4,
                   help="Batch size during calibration (default: 4)")
    p.add_argument("--bits", type=int, default=4,
                   help="Quantization bit-width (default: 4)")
    p.add_argument("--percdamp", type=float, default=0.02,
                   help="GPTQ Hessian damping factor (default: 0.02)")
    p.add_argument("--no_skip_first_last", action="store_true",
                   help="Also quantize first and last conv layers (default: keep FP32)")
    p.add_argument("--fp32_eval_only", action="store_true",
                   help="Skip quantization, evaluate FP32 model only")
    return p.parse_args()


def main():
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None
    torch.backends.cudnn.deterministic = True

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    logger.info(f"Device:           {device}")
    logger.info(f"Checkpoint:       {args.checkpoint}")
    logger.info(f"Calibration data: {args.calibration_data}")
    logger.info(f"Test dataset:     {args.test_dataset}")
    logger.info(f"Output dir:       {args.output_dir}")
    logger.info(f"Bits:             {args.bits}")

    for path, label in [(args.checkpoint, "checkpoint"),
                        (args.calibration_data, "calibration_data"),
                        (args.test_dataset, "test_dataset")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    config_path = _REPO / "configs" / "models" / f"{args.config}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    model_cfg = model_config(str(config_path))
    args_dict = get_args("test")
    args_dict["lmbda"] = args.lmbda
    model_args = argparse.Namespace(**args_dict)
    model_args.checkpoint = args.checkpoint

    net, _, _ = get_model(model_cfg, model_args, device)
    net = load_checkpoint(net, args.checkpoint, device)
    net = net.to(device)
    net.eval()

    quality = Path(Path(args.checkpoint).stem).stem
    dataset_name = Path(args.test_dataset).name
    logger.info(f"Model: {net.__class__.__name__}  quality={quality}  dataset={dataset_name}")

    # ── FP32 baseline evaluation ──────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1 / 3 — FP32 BASELINE EVALUATION")
    logger.info("=" * 60)
    fp32_metrics = evaluate_fast(
        images_dir=args.test_dataset,
        net=net,
        output_dir=args.output_dir,
        dataset_name=dataset_name,
        quality=quality,
        model_name="LoC_LIC_fp32",
    )

    if args.fp32_eval_only:
        logger.info("--fp32_eval_only set: stopping after FP32 evaluation.")
        return

    # ── GPTQ quantization ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(f"PHASE 2 / 3 — GPTQ INT{args.bits} QUANTIZATION")
    logger.info("=" * 60)

    quant_cfg = QuantConfig(
        bits=args.bits,
        percdamp=args.percdamp,
        skip_first_last_conv=not args.no_skip_first_last,
        output_dir=args.output_dir,
    )

    pipeline = GPTQPipeline(net, quant_cfg)

    calib_loader = build_calibration_loader(
        images_dir=args.calibration_data,
        num_samples=args.num_calibration_samples,
        batch_size=args.calibration_batch_size,
    )

    pipeline.calibrate(calib_loader)
    scales = pipeline.quantize_all()
    summary = pipeline.get_quantization_summary()

    logger.info(f"Quantization summary: {summary}")
    model_name_ptq = f"LoC_LIC_int{args.bits}_gptq_{args.bits}bit"
    save_results(net, scales, summary, args.output_dir, model_name_ptq)

    # ── INT4 PTQ evaluation ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(f"PHASE 3 / 3 — INT{args.bits} GPTQ EVALUATION")
    logger.info("=" * 60)

    ptq_metrics = evaluate_fast(
        images_dir=args.test_dataset,
        net=net,
        output_dir=args.output_dir,
        dataset_name=dataset_name,
        quality=quality,
        model_name=model_name_ptq,
    )

    # ── Comparison ────────────────────────────────────────────────────────────
    print_comparison(fp32_metrics, ptq_metrics,
                     fp32_name="FP32", ptq_name=f"INT{args.bits}-GPTQ")
    save_comparison_csv(fp32_metrics, ptq_metrics, args.output_dir)

    logger.info("\nDone. All results in: " + args.output_dir)


if __name__ == "__main__":
    main()
