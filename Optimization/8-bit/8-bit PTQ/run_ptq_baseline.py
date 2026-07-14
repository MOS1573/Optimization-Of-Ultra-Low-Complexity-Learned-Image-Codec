#!/usr/bin/env python3
"""
PTQ 8-bit Baseline Test Runner

Runs LoC-LIC baseline test with INT8 PTQ quantization applied.
Compares quantized model metrics with baseline FP32 metrics.

Usage:
    python run_ptq_baseline.py \\
        --checkpoint checkpoints/loc-lic/0.0018.pth.tar \\
        --main_path . \\
        --test_dataset datasets/kodak \\
        --output_dir results/ptq_test

Comparison:
    - Quantized model loaded from checkpoint
    - INT8 PTQ applied during inference
    - Metrics: PSNR, SSIM, LPIPS, BPP
    - Results saved as JSON + CSV
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add parent directories to path
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageFile

# Import from main repo
from models.get_model import get_model
from utils.config import model_config, get_device
from utils.args import test_options, get_args
from utils.metrics import ImageMetric, write_metrics_to_csv
from utils.utils import *
from utils.image import read_image

# Import from quantization module
from quantize_8bit_ptq import (
    QuantizationConfig,
    Int8PTQQuantizer,
    ModelEvaluator,
    load_checkpoint,
    create_calibration_loader,
    save_quantized_model
)
from dataclasses import asdict
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compress_one_image(model, x, stream_path, H, W, img_name):
    """Compress single image (same as test.py)."""
    with torch.no_grad():
        out = model.compress(x)

    shape = out["shape"]
    output = os.path.join(stream_path, img_name)
    with Path(output).open("wb") as f:
        write_uints(f, (H, W))
        write_body(f, shape, out["strings"])

    size = filesize(output)
    bpp = float(size) * 8 / (H * W)
    return bpp, out["cost_time"]


def decompress_one_image(model, stream_path, img_name):
    """Decompress single image (same as test.py)."""
    output = os.path.join(stream_path, img_name)
    with Path(output).open("rb") as f:
        original_size = read_uints(f, 2)
        strings, shape = read_body(f)

    with torch.no_grad():
        out = model.decompress(strings, shape)

    x_hat = out["x_hat"]
    x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
    cost_time = out["cost_time"]
    return x_hat, cost_time


def test_quantized_dataset_fast(images_dir, net, save_dir, dataset_name, quality, model_name):
    """
    Fast evaluation using forward pass + likelihoods-based BPP.
    Skips RANS compress/decompress (impractically slow on CPU).
    BPP = sum(-log2(likelihoods)) / (H*W), same metric used in training.
    """
    save_dir_metrics = os.path.join(os.path.dirname(save_dir), 'metrics')
    os.makedirs(save_dir_metrics, exist_ok=True)
    metrics_csv_path = os.path.join(save_dir_metrics, model_name + '.csv')
    metrics_avg_csv_path = os.path.join(save_dir_metrics, 'average_' + model_name + '.csv')

    save_dir = os.path.join(save_dir, model_name)
    save_dir = os.path.join(save_dir, "images", dataset_name, quality)
    os.makedirs(save_dir, exist_ok=True)

    save_log_dir = os.path.join(save_dir, 'results.csv')
    save_log_avg_dir = os.path.join(save_dir, 'results_avg.csv')

    net.eval()
    device = next(net.parameters()).device
    image_metric = ImageMetric(device=device)

    images = os.listdir(images_dir)
    images = [os.path.join(images_dir, img) for img in images
              if img.endswith('.png') or img.endswith('.jpg')]
    images.sort()

    logger.info(f"Testing {len(images)} images from {dataset_name} (fast_eval mode)")
    metrics_dict = {}

    with torch.no_grad():
        for i, img_path in enumerate(images):
            image_name = img_path.split('/')[-1].split('.')[0]
            img = read_image(img_path, device)
            img = img.to(device)
            B, C, H, W = img.shape

            pad_h = (64 - H % 64) % 64
            pad_w = (64 - W % 64) % 64
            img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)

            out = net(img_pad)
            x_hat = out["x_hat"][:, :, :H, :W].clamp(0, 1)

            # BPP from likelihoods (matches training objective)
            num_pixels = H * W
            bpp = sum(
                (-torch.log2(like).sum()) / num_pixels
                for like in out["likelihoods"].values()
            ).item()

            # size_bytes is a lossless round-trip: metric_image recomputes bpp = size*8/(H*W) = bpp
            size_bytes = int(bpp * num_pixels / 8)
            metrics = image_metric.metric_image(output=x_hat, target=img, size_of_image=size_bytes)
            metrics['bpp'] = bpp  # use float bpp directly (avoids int truncation error)
            metrics_dict = image_metric.append_to_metrics_dict(metrics, metrics_dict)

            metrics['model_name'] = model_name
            metrics['dataset_name'] = dataset_name
            metrics['quality'] = quality

            rec = torch2img(x_hat)
            rec.save(os.path.join(save_dir, image_name + '.png'))
            write_metrics_to_csv(metrics, metrics_csv_path, overwrite=False)
            write_metrics_to_csv(metrics, save_log_dir, overwrite=False)

            if (i + 1) % max(1, len(images) // 5) == 0 or (i + 1) == len(images):
                logger.info(f"  Processed {i + 1}/{len(images)} images — "
                            f"PSNR={metrics.get('psnr', 0):.2f} BPP={bpp:.4f}")

    avg_metrics = image_metric.get_avg_metrics(metrics_dict)
    avg_metrics['model_name'] = model_name
    avg_metrics['dataset_name'] = dataset_name
    avg_metrics['quality'] = quality
    write_metrics_to_csv(avg_metrics, metrics_avg_csv_path, overwrite=False)
    write_metrics_to_csv(avg_metrics, save_log_avg_dir, overwrite=False)

    logger.info(f"Results saved to {save_dir}")
    logger.info(f"Metrics: PSNR={avg_metrics.get('psnr', 0):.2f}, "
                f"SSIM={avg_metrics.get('ssim', 0):.4f}, "
                f"BPP={avg_metrics.get('bpp', 0):.4f}")

    return avg_metrics


def test_quantized_dataset(images_dir, net, save_dir, dataset_name, quality, model_name):
    """
    Test quantized model on a dataset (same flow as test_single_dataset in test.py).

    Args:
        images_dir: Directory with test images
        net: Quantized model
        save_dir: Output directory for results
        dataset_name: Name of dataset (e.g., "kodak")
        quality: Lambda/quality parameter (e.g., "0.0018")
        model_name: Model name for logging
    """
    save_dir_metrics = os.path.join(os.path.dirname(save_dir), 'metrics')
    os.makedirs(save_dir_metrics, exist_ok=True)
    metrics_csv_path = os.path.join(save_dir_metrics, model_name + '.csv')
    metrics_avg_csv_path = os.path.join(save_dir_metrics, 'average_' + model_name + '.csv')

    save_dir = os.path.join(save_dir, model_name)
    save_dir = os.path.join(save_dir, "images", dataset_name, quality)
    os.makedirs(save_dir, exist_ok=True)

    save_log_dir = os.path.join(save_dir, 'results.csv')
    save_log_avg_dir = os.path.join(save_dir, 'results_avg.csv')

    net.eval()
    device = next(net.parameters()).device
    image_metric = ImageMetric(device=device)

    # Get all images
    images = os.listdir(images_dir)
    images = [os.path.join(images_dir, img) for img in images
              if img.endswith('.png') or img.endswith('.jpg')]

    logger.info(f"Testing {len(images)} images from {dataset_name}")
    metrics_dict = {}

    with torch.no_grad():
        for i, img_path in enumerate(images):
            image_name = img_path.split('/')[-1].split('.')[0]
            img = read_image(img_path, device)
            img = img.to(device)
            B, C, H, W = img.shape

            # Padding to multiple of 64
            pad_h = 0
            pad_w = 0
            if H % 64 != 0:
                pad_h = 64 * (H // 64 + 1) - H
            if W % 64 != 0:
                pad_w = 64 * (W // 64 + 1) - W
            img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)

            # Warmup and actual test
            bitstream_path = os.path.join(save_dir, image_name)
            if i == 0:
                bpp, enc_time = compress_one_image(model=net, x=img_pad, stream_path=save_dir,
                                                   H=H, W=W, img_name=image_name)
            try:
                net.update_resolutions(16, 16)
            except:
                pass

            bpp, enc_time = compress_one_image(model=net, x=img_pad, stream_path=save_dir,
                                              H=H, W=W, img_name=image_name)
            try:
                net.update_resolutions(16, 16)
            except:
                pass

            x_hat, dec_time = decompress_one_image(model=net, stream_path=save_dir,
                                                   img_name=image_name)
            bitstream_size = os.path.getsize(bitstream_path)

            metrics = image_metric.metric_image(output=x_hat, target=img, size_of_image=bitstream_size)
            metrics_dict = image_metric.append_to_metrics_dict(metrics, metrics_dict)

            metrics['model_name'] = model_name
            metrics['dataset_name'] = dataset_name
            metrics['quality'] = quality

            rec = torch2img(x_hat)
            img_vis = torch2img(img)
            rec.save(os.path.join(save_dir, image_name + '.png'))
            write_metrics_to_csv(metrics, metrics_csv_path, overwrite=False)
            write_metrics_to_csv(metrics, save_log_dir, overwrite=False)

            if (i + 1) % max(1, len(images) // 5) == 0:
                logger.info(f"  Processed {i + 1}/{len(images)} images")

        avg_metrics = image_metric.get_avg_metrics(metrics_dict)
        avg_metrics['model_name'] = model_name
        avg_metrics['dataset_name'] = dataset_name
        avg_metrics['quality'] = quality
        write_metrics_to_csv(avg_metrics, metrics_avg_csv_path, overwrite=False)
        write_metrics_to_csv(avg_metrics, save_log_avg_dir, overwrite=False)

    logger.info(f"Results saved to {save_dir}")
    logger.info(f"Metrics: PSNR={avg_metrics.get('psnr', 'N/A'):.2f}, "
               f"SSIM={avg_metrics.get('ssim', 'N/A'):.4f}, "
               f"BPP={avg_metrics.get('bpp', 'N/A'):.4f}")

    return avg_metrics


def main():
    """Main PTQ baseline test pipeline."""
    
    parser = argparse.ArgumentParser(
        description="PTQ 8-bit Baseline Test - Compare FP32 vs INT8 metrics"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pretrained checkpoint (e.g., checkpoints/loc-lic/0.0018.pth.tar)"
    )
    parser.add_argument(
        "--main_path",
        type=str,
        default=".",
        help="Main repository path"
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="datasets/kodak",
        help="Test dataset path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ptq_test",
        help="Output directory for PTQ results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="loclic",
        help="Model config name"
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=100,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="qnnpack",
        choices=["fbgemm", "qnnpack"],
        help="Quantization backend (qnnpack for CPU, fbgemm for CUDA/x86 CPU)"
    )
    parser.add_argument(
        "--skip_quantization",
        action="store_true",
        help="Skip quantization and just run inference (for debugging)"
    )
    parser.add_argument(
        "--fast_eval",
        action="store_true",
        help="Use forward pass + likelihoods BPP instead of RANS compress/decompress (much faster on CPU)"
    )
    
    args = parser.parse_args()
    
    # Setup
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None
    torch.backends.cudnn.deterministic = True
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = get_device()
    logger.info(f"Device: {device}")
    
    # Check checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    # Check dataset
    if not os.path.exists(args.test_dataset):
        raise FileNotFoundError(f"Test dataset not found: {args.test_dataset}")
    logger.info(f"Test dataset: {args.test_dataset}")
    
    # Load config and model
    repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(repo_path, 'configs', 'models', args.config + '.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    config = model_config(config_path)
    logger.info(f"Model config: {args.config}")
    
    # Build and load model
    args_dict = get_args("test")
    args_dict["lmbda"] = 0.0018
    model_args = argparse.Namespace(**args_dict)
    model_args.checkpoint = args.checkpoint  # Keep the parsed checkpoint

    net, _, _ = get_model(config, model_args, device)
    net = load_checkpoint(net, model_args.checkpoint, device)
    net = net.to(device)
    net.eval()

    logger.info(f"Baseline model loaded: {net.__class__.__name__}")
    
    # Extract model info from checkpoint path
    checkpoint_path = args.checkpoint
    model_description = checkpoint_path.split('/')[-2] if '/' in checkpoint_path else checkpoint_path.split('\\')[-2]
    model_name = '_'.join(model_description.split('_')[0:-1]) if '_' in model_description else 'LoC_LIC'
    quality = checkpoint_path.split('/')[-1].replace('.pth.tar', '')
    dataset_name = args.test_dataset.split('/')[-1]
    
    logger.info(f"Model: {model_name}, Quality: {quality}, Dataset: {dataset_name}")
    
    # ===== QUANTIZATION =====
    if not args.skip_quantization:
        logger.info("\n" + "="*80)
        logger.info("QUANTIZATION PHASE")
        logger.info("="*80)
        
        # Create quantization config
        quant_config = QuantizationConfig(
            backend=args.backend,
            num_calibration_samples=args.num_calibration_samples,
            output_dir=args.output_dir
        )
        
        # Create calibration dataloader (simple: use dummy data if training set not available)
        try:
            calibration_loader = create_calibration_loader(
                asdict(config) if hasattr(config, '__dict__') else vars(config),
                vars(args_dict) if hasattr(args_dict, '__dict__') else vars(args_dict),
                batch_size=8,
                num_samples=args.num_calibration_samples
            )
        except Exception as e:
            logger.warning(f"Could not create real calibration loader: {e}")
            logger.warning("Creating dummy calibration data...")
            # Fallback: dummy data
            class DummyDataset:
                def __init__(self, num_samples=100):
                    self.num_samples = num_samples
                def __len__(self):
                    return self.num_samples
                def __getitem__(self, idx):
                    return torch.rand(3, 256, 256)
            
            calibration_loader = DataLoader(
                DummyDataset(args.num_calibration_samples),
                batch_size=8,
                shuffle=True
            )
        
        logger.info(f"Calibration loader created with {args.num_calibration_samples} samples")
        
        # Apply quantization
        quantizer = Int8PTQQuantizer(net, quant_config, device)
        net_quantized, scale_dict, zero_point_dict = quantizer.quantize(calibration_loader)
        
        logger.info("Quantization completed successfully")
        logger.info(f"Quantized model parameters: {sum(p.numel() for p in net_quantized.parameters()):,}")
        
        # Save quantized model
        model_name_ptq = f"{model_name}_int8_ptq"
        save_quantized_model(net_quantized, scale_dict, zero_point_dict, args.output_dir, model_name_ptq)

        net = net_quantized
        model_name = model_name_ptq  # Fix: use ptq name for results directory
    else:
        model_name += "_fp32"
        logger.info("Skipping quantization (FP32 inference mode)")

    # ===== TESTING =====
    logger.info("\n" + "="*80)
    logger.info("TEST PHASE")
    logger.info("="*80)

    if args.fast_eval:
        metrics = test_quantized_dataset_fast(
            images_dir=args.test_dataset,
            net=net,
            save_dir=args.output_dir,
            dataset_name=dataset_name,
            quality=quality,
            model_name=model_name
        )
    else:
        metrics = test_quantized_dataset(
            images_dir=args.test_dataset,
            net=net,
            save_dir=args.output_dir,
            dataset_name=dataset_name,
            quality=quality,
            model_name=model_name
        )
    
    logger.info("\n" + "="*80)
    logger.info("PTQ BASELINE TEST COMPLETED")
    logger.info("="*80)
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Quality: {quality}")
    logger.info(f"Final Metrics:")
    for key, val in metrics.items():
        if isinstance(val, float):
            logger.info(f"  {key}: {val:.6f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
