#!/usr/bin/env python3
"""
Unified evaluation script for all LoC-LIC quantized models.

Evaluates FP32, INT8-GPTQ, INT4-GPTQ, and INT8-PTQ on Kodak and pro_2020.
Results are written to --output_dir (default: results/legacy/new_tests).
INT4-QAT is excluded because it was already evaluated with the new dataset paths.

Usage (interactive GPU):
    python Optimization/eval_all_models.py

Usage (SLURM batch):
    sbatch Optimization/run_eval_all.sh
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from models.get_model import get_model
from utils.config import model_config
from utils.image import read_image
from utils.metrics import ImageMetric, write_metrics_to_csv
from utils.utils import torch2img, patch_compressai_buffer_loading

patch_compressai_buffer_loading()
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model catalogue: (display_name, model_name_slug, path_relative_to_repo)
# ─────────────────────────────────────────────────────────────────────────────
MODELS = [
    (
        "FP32 Baseline",
        "LoC_LIC_fp32",
        "checkpoints/loc-lic/0.0018.pth.tar",
    ),
    (
        "INT8 GPTQ",
        "LoC_LIC_int8_gptq",
        "results/legacy/old_kodak_test/gptq_second_try/LoC_LIC_int8_gptq_8bit.pt",
    ),
    (
        "INT4 GPTQ",
        "LoC_LIC_int4_gptq",
        "results/legacy/old_kodak_test/gptq_4bit/LoC_LIC_int4_gptq_4bit.pt",
    ),
    (
        "INT8 PTQ",
        "LoC_LIC_int8_ptq",
        "results/legacy/old_kodak_test/ptq_fast_test/LoC_LIC_int8_ptq.pt",
    ),
]

QUALITY = "0.0018"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    config_path = _REPO / "configs" / "models" / "loclic.yaml"
    model_cfg = model_config(str(config_path))
    args = argparse.Namespace(lmbda=float(QUALITY))
    net, _, _ = get_model(model_cfg, args, device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    net.load_state_dict(state)
    return net.eval().to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(net: torch.nn.Module, images_dir: str, output_dir: str,
             dataset_name: str, model_name: str, device: torch.device) -> dict:
    img_out_dir = os.path.join(output_dir, model_name, "images", dataset_name, QUALITY)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

    per_img_csv = os.path.join(img_out_dir, "results.csv")
    avg_csv     = os.path.join(img_out_dir, "results_avg.csv")

    image_metric = ImageMetric(device="cpu")
    img_paths = sorted(
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    logger.info(f"  {len(img_paths)} images  [{dataset_name}]")

    metrics_dict: dict = {}
    net.eval()
    with torch.inference_mode():
        for i, img_path in enumerate(img_paths):
            img = read_image(str(img_path), device).to(device)
            _, _, H, W = img.shape
            pad_h = (64 - H % 64) % 64
            pad_w = (64 - W % 64) % 64
            img_pad = F.pad(img, (0, pad_w, 0, pad_h))

            out   = net(img_pad)
            x_hat = out["x_hat"][:, :, :H, :W].clamp(0.0, 1.0)

            num_pixels = H * W
            bpp = sum(
                (-torch.log2(like).sum()) / num_pixels
                for like in out["likelihoods"].values()
            ).item()

            size_bytes = max(int(bpp * num_pixels / 8), 1)
            m = image_metric.metric_image(
                output=x_hat.cpu(), target=img.cpu(), size_of_image=size_bytes
            )
            m["bpp"] = bpp
            metrics_dict = image_metric.append_to_metrics_dict(m, metrics_dict)
            m.update(model_name=model_name, dataset_name=dataset_name, quality=QUALITY)
            torch2img(x_hat).save(os.path.join(img_out_dir, img_path.stem + ".png"))
            write_metrics_to_csv(m, per_img_csv, overwrite=False)

            logger.info(f"    [{i+1:3d}/{len(img_paths)}] {img_path.stem:<12s} "
                        f"PSNR={m.get('psnr', 0):.2f} dB  BPP={bpp:.4f}")

    avg = image_metric.get_avg_metrics(metrics_dict)
    avg.update(model_name=model_name, dataset_name=dataset_name, quality=QUALITY)
    write_metrics_to_csv(avg, avg_csv, overwrite=True)
    write_metrics_to_csv(
        avg,
        os.path.join(output_dir, "metrics", f"average_{model_name}_{dataset_name}.csv"),
        overwrite=True,
    )

    logger.info(f"    PSNR   = {avg.get('psnr',   0):.4f} dB")
    logger.info(f"    SSIM   = {avg.get('ssim',   0):.4f}")
    logger.info(f"    MS-SSIM= {avg.get('msssim', 0):.4f}")
    logger.info(f"    LPIPS  = {avg.get('lpips',  0):.4f}")
    logger.info(f"    BPP    = {avg.get('bpp',    0):.4f}")
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate all LoC-LIC models on both test datasets")
    p.add_argument("--output_dir", default=str(_REPO / "results" / "new_tests"))
    p.add_argument("--test_dataset_kodak",
                   default="path/to/dataset/Kodak")
    p.add_argument("--test_dataset_pro2020",
                   default="path/to/dataset/pro_2020")
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    test_datasets = [
        (args.test_dataset_kodak,    "kodak"),
        (args.test_dataset_pro2020,  "pro_2020"),
    ]
    for ds_path, ds_name in test_datasets:
        if not os.path.isdir(ds_path):
            logger.error(f"Test dataset not found: {ds_path}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all results for final summary
    all_results: dict = {}   # {model_slug: {dataset_name: avg_metrics}}

    for display_name, model_slug, rel_path in MODELS:
        ckpt_path = str(_REPO / rel_path)
        logger.info("=" * 60)
        logger.info(f"Model: {display_name}")
        logger.info(f"  Checkpoint: {ckpt_path}")

        if not os.path.exists(ckpt_path):
            logger.error(f"  NOT FOUND — skipping: {ckpt_path}")
            continue

        logger.info("  Loading model...")
        net = load_model(ckpt_path, device)
        all_results[model_slug] = {}

        for ds_path, ds_name in test_datasets:
            logger.info(f"\n  Evaluating on {ds_name} ...")
            avg = evaluate(net, ds_path, args.output_dir, ds_name, model_slug, device)
            all_results[model_slug][ds_name] = avg

        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Print summary table ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY — Kodak")
    logger.info("=" * 70)
    logger.info(f"{'Model':<25} {'PSNR':>8} {'MS-SSIM':>9} {'LPIPS':>8} {'BPP':>8}")
    logger.info("-" * 70)
    for display_name, model_slug, _ in MODELS:
        r = all_results.get(model_slug, {}).get("kodak", {})
        logger.info(f"{display_name:<25} {r.get('psnr',0):>8.4f} "
                    f"{r.get('msssim',0):>9.4f} {r.get('lpips',0):>8.4f} "
                    f"{r.get('bpp',0):>8.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY — pro_2020")
    logger.info("=" * 70)
    logger.info(f"{'Model':<25} {'PSNR':>8} {'MS-SSIM':>9} {'LPIPS':>8} {'BPP':>8}")
    logger.info("-" * 70)
    for display_name, model_slug, _ in MODELS:
        r = all_results.get(model_slug, {}).get("pro_2020", {})
        logger.info(f"{display_name:<25} {r.get('psnr',0):>8.4f} "
                    f"{r.get('msssim',0):>9.4f} {r.get('lpips',0):>8.4f} "
                    f"{r.get('bpp',0):>8.4f}")

    logger.info(f"\nAll results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
