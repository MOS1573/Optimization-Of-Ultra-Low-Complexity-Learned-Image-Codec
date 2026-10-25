#!/usr/bin/env python3
"""
4-bit Quantization-Aware Training (QAT) for LoC-LIC.
Experiment name: 4-bit QAT LOCLIC

Pipeline:
  1. Load the FP32 checkpoint (e.g. checkpoints/loc-lic/0.0018.pth.tar)
  2. Register STE-based INT4 fake-quantization parametrizations on all eligible
     Conv2d / ConvTranspose2d / Linear layers (entropy models always skipped)
  3. Fine-tune with the standard RD loss for --epochs epochs (default: 50)
  4. Save per-epoch checkpoints + a final deployment state dict
  5. Run fast evaluation (likelihoods-based BPP) on Kodak at the end

Key differences from GPTQ PTQ:
  - Weights are updated by the optimizer — the model learns to tolerate INT4 noise
  - The entropy model re-aligns to the quantized transform statistics during training
  - Uses Straight-Through Estimator so gradients flow through the rounding operation

Key differences from training from scratch:
  - Starts from a trained FP32 checkpoint (fast convergence, 50 epochs instead of 200)
  - Learning rate is 10x lower (1e-5 instead of 1e-4)
  - The first and last convolutional layers are kept at FP32

Usage:
  cd path/to/loc-lic-new
  python "Optimization/4-bit QAT/train_qat_4bit.py" \\
      --checkpoint checkpoints/loc-lic/0.0018.pth.tar \\
      --lmbda 0.0018
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# ── Repo root ────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from datasets.datasets import ImageFolder, ImageFolderCSV
from loss.rd_loss import RateDistortionLoss
from models.get_model import get_model
from utils.config import model_config, get_device
from utils.logger import setup_logger
from utils.metrics import ImageMetric, write_metrics_to_csv
from utils.optimizers import configure_optimizers
from utils.schedules import get_cosine_schedule_with_warmup
from utils.testing import test_one_epoch
from utils.utils import patch_compressai_buffer_loading, setup_environment, save_checkpoint
from utils.image import read_image
from utils.utils import torch2img

from qat_modules import (
    register_qat_parametrizations,
    extract_deployment_state_dict,
    fake_quant_ste,
)

patch_compressai_buffer_loading()
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_fp32_checkpoint(net, checkpoint_path: str, device: torch.device):
    """Load an FP32 checkpoint — handles both raw and dict-wrapped formats."""
    logger.info(f"Loading FP32 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    # Strip DataParallel prefix
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    # LoC_LIC.load_state_dict() does not accept strict kwarg
    net.load_state_dict(state)
    logger.info("FP32 checkpoint loaded successfully.")
    return net


def load_qat_checkpoint(net, checkpoint_path: str, device: torch.device):
    """
    Resume from a QAT checkpoint (state dict has parametrize keys).
    Call AFTER register_qat_parametrizations() so the model has the right structure.
    """
    logger.info(f"Resuming QAT from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    # LoC_LIC custom load_state_dict: calls update_registered_buffers + super()
    # With parametrize registered, the keys match (parametrizations.weight.original)
    net.load_state_dict(state)
    epoch = ckpt.get("epoch", 0)
    loss  = ckpt.get("loss", float("inf"))
    logger.info(f"Resumed from epoch {epoch}, loss={loss:.4f}")
    return net, epoch, loss


# ─────────────────────────────────────────────────────────────────────────────
# Fast evaluation (likelihoods BPP — no RANS needed)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset_fast(net, images_dir: str, output_dir: str,
                          quality: str, model_name: str, dataset_name: str,
                          device: torch.device):
    """Forward-pass evaluation on all images in images_dir. BPP = -log2(likelihoods)/pixels."""
    img_out_dir = os.path.join(output_dir, model_name, "images", dataset_name, quality)
    os.makedirs(img_out_dir, exist_ok=True)

    per_img_csv = os.path.join(img_out_dir, "results.csv")
    avg_csv     = os.path.join(img_out_dir, "results_avg.csv")
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

    net.eval()
    image_metric = ImageMetric(device="cpu")

    img_paths = sorted(
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    logger.info(f"Evaluating {len(img_paths)} images on {dataset_name} [{model_name}]")

    metrics_dict: dict = {}
    with torch.inference_mode():
        for i, img_path in enumerate(img_paths):
            img = read_image(str(img_path), device).to(device)
            _, _, H, W = img.shape
            pad_h = (64 - H % 64) % 64
            pad_w = (64 - W % 64) % 64
            img_pad = F.pad(img, (0, pad_w, 0, pad_h))

            out  = net(img_pad)
            x_hat = out["x_hat"][:, :, :H, :W].clamp(0.0, 1.0)

            num_pixels = H * W
            bpp = sum(
                (-torch.log2(like).sum()) / num_pixels
                for like in out["likelihoods"].values()
            ).item()

            size_bytes = max(int(bpp * num_pixels / 8), 1)
            m = image_metric.metric_image(output=x_hat.cpu(), target=img.cpu(),
                                          size_of_image=size_bytes)
            m["bpp"] = bpp
            metrics_dict = image_metric.append_to_metrics_dict(m, metrics_dict)
            m.update(model_name=model_name, dataset_name=dataset_name, quality=quality)
            torch2img(x_hat).save(os.path.join(img_out_dir, img_path.stem + ".png"))
            write_metrics_to_csv(m, per_img_csv, overwrite=False)

            logger.info(f"  [{i+1:2d}/{len(img_paths)}] {img_path.stem:<10s} "
                        f"PSNR={m.get('psnr', 0):.2f}  BPP={bpp:.4f}")

    avg = image_metric.get_avg_metrics(metrics_dict)
    avg.update(model_name=model_name, dataset_name=dataset_name, quality=quality)
    write_metrics_to_csv(avg, avg_csv, overwrite=True)
    write_metrics_to_csv(avg,
        os.path.join(output_dir, "metrics", f"average_{model_name}_{dataset_name}.csv"),
        overwrite=True)

    logger.info(f"  [{dataset_name}] PSNR   = {avg.get('psnr',   0):.4f} dB")
    logger.info(f"  [{dataset_name}] SSIM   = {avg.get('ssim',   0):.4f}")
    logger.info(f"  [{dataset_name}] MS-SSIM= {avg.get('msssim', 0):.4f}")
    logger.info(f"  [{dataset_name}] LPIPS  = {avg.get('lpips',  0):.4f}")
    logger.info(f"  [{dataset_name}] BPP    = {avg.get('bpp',    0):.4f}")
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="4-bit QAT fine-tuning for LoC-LIC")

    # Paths
    p.add_argument("--checkpoint", required=True,
                   help="FP32 checkpoint to start from, e.g. checkpoints/loc-lic/0.0018.pth.tar")
    p.add_argument("--resume_qat", default=None,
                   help="Path to a QAT checkpoint to resume training (optional)")
    p.add_argument("--dataset", default="path/to/dataset/images",
                   help="Root directory of the training dataset")
    p.add_argument("--dataset_csv", default="train_256",
                   help="CSV filename (without .csv) in datasets/csv/ (default: train_256)")
    p.add_argument("--test_dataset", default=None,
                   help="Path to Kodak test images (default: <repo>/datasets/kodak)")
    p.add_argument("--test_dataset_pro2020", default=None,
                   help="Path to pro_2020 test images for additional evaluation (optional)")
    p.add_argument("--output_dir", default=None,
                   help="Where to write results (default: <repo>/results/4-bit/phase1-uniform-int4/lambda-0018)")
    p.add_argument("--config", default="loclic",
                   help="Model config name in configs/models/ (default: loclic)")

    # Loss
    p.add_argument("--lmbda", type=float, default=0.0018,
                   help="Rate-distortion lambda — MUST match the checkpoint (default: 0.0018)")

    # Training
    p.add_argument("--epochs", type=int, default=50,
                   help="Number of QAT fine-tuning epochs (default: 50)")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Training batch size (default: 16)")
    p.add_argument("--test_batch_size", type=int, default=16,
                   help="Validation batch size (default: 16)")
    p.add_argument("--learning_rate", type=float, default=1e-5,
                   help="Main optimizer LR (default: 1e-5, 10x lower than base training)")
    p.add_argument("--aux_learning_rate", type=float, default=1e-4,
                   help="Aux optimizer LR for entropy quantiles (default: 1e-4)")
    p.add_argument("--clip_max_norm", type=float, default=1.0,
                   help="Gradient clip norm (default: 1.0)")
    p.add_argument("--patch_size", type=int, default=256,
                   help="Training crop size (default: 256)")
    p.add_argument("--num_workers", type=int, default=16,
                   help="DataLoader worker count (default: 16)")
    p.add_argument("--save_every", type=int, default=5,
                   help="Save checkpoint every N epochs (default: 5)")
    p.add_argument("--mixed_precision", type=str, default="no",
                   choices=["no", "fp16", "bf16"],
                   help="Mixed precision mode (default: no — QAT+fp16 can be unstable)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="Gradient accumulation steps (default: 1; use 5 for eff. batch=40 on 2×GPU with --batch_size 4)")
    p.add_argument("--max_steps_per_epoch", type=int, default=None,
                   help="Cap the number of training batches per epoch. "
                        "Use this instead of running the full 1.93M-image dataset per epoch — "
                        "e.g. 4000 keeps each epoch to ~50 min, 50 epochs fits in 48h.")
    p.add_argument("--resume_accelerate", default=None,
                   help="Path to an accelerate checkpoint directory to resume from "
                        "(e.g. experiments/4-bit QAT LOCLIC/checkpoints/checkpoint_035). "
                        "Restores model weights, optimizer, scheduler, and epoch counter.")
    p.add_argument("--reset_lr", action="store_true",
                   help="After loading a resume checkpoint, override the saved optimizer LR "
                        "with --learning_rate and --aux_learning_rate. Use when continuing "
                        "a run with a different LR than what was checkpointed (e.g. 2B fine-tuning).")

    # Quantization
    p.add_argument("--bits", type=int, default=4,
                   help="Weight quantization bit-width (default: 4)")
    p.add_argument("--no_skip_first_last", action="store_true",
                   help="Also quantize first and last conv (default: keep them FP32)")

    # Experiment
    p.add_argument("--experiment", default="4-bit QAT LOCLIC",
                   help="Experiment name used for log/checkpoint directories")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop with gradient accumulation
# ─────────────────────────────────────────────────────────────────────────────

def _train_one_epoch(net, criterion, train_loader, optimizer, aux_optimizer,
                     epoch, clip_max_norm, logger_train, tb_logger, current_step, accelerator,
                     max_steps_per_epoch=None):
    """
    QAT training loop using accelerate.accumulate() for gradient accumulation.
    accelerate handles:
      - DDP no_sync() on micro-steps (no all-reduce until sync step)
      - accelerator.backward() scales loss by 1/gradient_accumulation_steps
      - wrapped optimizer.step() / zero_grad() are no-ops on micro-steps
    """
    net.train()
    last_aux_loss = torch.tensor(0.0)
    for i, image in enumerate(train_loader):
        if max_steps_per_epoch is not None and i >= max_steps_per_epoch:
            break
        with accelerator.accumulate(net):
            with accelerator.autocast():
                out_net = net(image)
                out_criterion = criterion(out_net, image)
                loss = out_criterion["loss"]

            accelerator.backward(loss)
            if accelerator.sync_gradients and clip_max_norm > 0:
                accelerator.clip_grad_norm_(net.parameters(), clip_max_norm)
            optimizer.step()
            optimizer.zero_grad()

            # Only update aux params at gradient-sync steps (same cadence as main optimizer).
            # Avoids a second full backward pass on every micro-step.
            if accelerator.sync_gradients:
                aux_optimizer.zero_grad()
                last_aux_loss = accelerator.unwrap_model(net).get_aux_loss()
                last_aux_loss.backward()
                aux_optimizer.step()

        if current_step % 100 == 0 and accelerator.is_main_process:
            if tb_logger is not None:
                tb_logger.add_scalar('[train]: loss', out_criterion["loss"].item(), current_step)
                tb_logger.add_scalar('[train]: bpp_loss', out_criterion["bpp_loss"].item(), current_step)
                tb_logger.add_scalar('[train]: lr', optimizer.param_groups[0]['lr'], current_step)
                tb_logger.add_scalar('[train]: aux_loss', last_aux_loss.item(), current_step)
                tb_logger.add_scalar('[train]: mse_loss', out_criterion["mse_loss"].item(), current_step)
            logger_train.info(
                f"Train epoch {epoch}: ["
                f"{i * len(image):5d}/{len(train_loader.dataset)}"
                f" ({100. * i / len(train_loader):.0f}%)] "
                f'Loss: {out_criterion["loss"].item():.4f} | '
                f'MSE loss: {out_criterion["mse_loss"].item():.4f} | '
                f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                f"Aux loss: {last_aux_loss.item():.2f}"
            )
        current_step += 1
    return current_step


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    setup_environment(args.seed)

    # Resolve paths relative to repo root
    repo = _REPO
    if args.test_dataset is None:
        args.test_dataset = str(repo / "datasets" / "kodak")
    if args.output_dir is None:
        args.output_dir = str(repo / "results" / "4-bit" / "phase1-uniform-int4" / "lambda-0018")

    experiment_path = repo / "experiments" / args.experiment
    checkpoint_path = experiment_path / "checkpoints"

    # ── Accelerate setup ──────────────────────────────────────────────────────
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device
    torch.backends.cudnn.benchmark = True

    if accelerator.is_main_process:
        experiment_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # ── Logging ───────────────────────────────────────────────────────────────
    setup_logger("train", str(experiment_path), "train_" + args.experiment.replace(" ", "_"),
                 level=logging.INFO, screen=True, tofile=True)
    setup_logger("val",   str(experiment_path), "val_"   + args.experiment.replace(" ", "_"),
                 level=logging.INFO, screen=True, tofile=True)
    logger_train = logging.getLogger("train")
    logger_val   = logging.getLogger("val")
    tb_logger = SummaryWriter(log_dir=str(experiment_path)) if accelerator.is_main_process else None

    # ── Model ─────────────────────────────────────────────────────────────────
    config_path = repo / "configs" / "models" / f"{args.config}.yaml"
    model_cfg = model_config(str(config_path))
    net, _, _ = get_model(model_cfg, args, device)

    if args.resume_qat:
        # Register parametrizations FIRST, then load QAT weights (old .tar format)
        register_qat_parametrizations(net, bits=args.bits,
                                      skip_first_last_conv=not args.no_skip_first_last)
        net, start_epoch, best_loss = load_qat_checkpoint(net, args.resume_qat, device)
    elif args.resume_accelerate:
        # Load FP32 to establish model architecture, register QAT parametrizations to
        # match the saved state-dict keys, then let accelerator.load_state() overwrite
        # the weights after prepare().
        net = load_fp32_checkpoint(net, args.checkpoint, device)
        n_q, n_skip = register_qat_parametrizations(
            net, bits=args.bits, skip_first_last_conv=not args.no_skip_first_last)
        if accelerator.is_main_process:
            logger_train.info(f"QAT parametrizations registered ({n_q} layers). "
                              f"Weights will be loaded from: {args.resume_accelerate}")
        start_epoch = 0        # overridden after accelerator.load_state()
        best_loss   = float("inf")
    else:
        # Fresh start: load FP32 weights, then register parametrizations
        net = load_fp32_checkpoint(net, args.checkpoint, device)
        n_q, n_skip = register_qat_parametrizations(
            net, bits=args.bits, skip_first_last_conv=not args.no_skip_first_last)
        if accelerator.is_main_process:
            logger_train.info(f"QAT: {n_q} layers quantized, {n_skip} skipped (FP32).")
        start_epoch = 0
        best_loss   = float("inf")

    net = net.to(device)

    # Loss and optimizers
    criterion   = RateDistortionLoss(lmbda=args.lmbda)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, args.epochs // 10),
        num_training_steps=args.epochs,
        base_lr=args.learning_rate,
        end_lr=1e-6,
    )
    lr_scheduler_aux = get_cosine_schedule_with_warmup(
        aux_optimizer,
        num_warmup_steps=max(1, args.epochs // 10),
        num_training_steps=args.epochs,
        base_lr=args.aux_learning_rate,
        end_lr=1e-6,
    )
    image_metric = ImageMetric()

    # ── Datasets ──────────────────────────────────────────────────────────────
    csv_path = repo / "datasets" / "csv" / f"{args.dataset_csv}.csv"
    train_transforms = transforms.Compose([
        transforms.RandomCrop(args.patch_size), transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(), transforms.CenterCrop(args.patch_size)
    ])

    train_dataset = ImageFolderCSV(args.dataset, csv_file=str(csv_path),
                                   transform=train_transforms)
    test_dataset  = ImageFolder(os.path.join(args.dataset, "test"),
                                transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=True,
                              drop_last=True)
    test_loader  = DataLoader(test_dataset, batch_size=args.test_batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=True)

    # ── Accelerate prepare ────────────────────────────────────────────────────
    (net, criterion, optimizer, aux_optimizer,
     train_loader, test_loader,
     lr_scheduler, lr_scheduler_aux,
     image_metric) = accelerator.prepare(
        net, criterion, optimizer, aux_optimizer,
        train_loader, test_loader,
        lr_scheduler, lr_scheduler_aux,
        image_metric,
    )

    # ── Resume from accelerate checkpoint (after prepare so DDP is ready) ─────
    if args.resume_accelerate:
        ckpt_dir = Path(args.resume_accelerate)
        accelerator.load_state(str(ckpt_dir))
        meta_path = ckpt_dir / "metadata.pth"
        if meta_path.exists():
            meta = torch.load(str(meta_path), map_location="cpu")
            start_epoch = meta.get("epoch", 0)
            best_loss   = meta.get("loss", float("inf"))
        else:
            # Fall back: parse epoch from directory name (e.g. "checkpoint_035" → 35)
            try:
                start_epoch = int(ckpt_dir.name.split("_")[-1])
            except ValueError:
                start_epoch = 0
            best_loss = float("inf")
        if args.reset_lr:
            # Override LR in all param groups — the saved optimizer state carries
            # the old LR (typically at the end of a cosine schedule, ~1e-6).
            for pg in optimizer.param_groups:
                pg["lr"] = args.learning_rate
            for pg in aux_optimizer.param_groups:
                pg["lr"] = args.aux_learning_rate
            # Also reset the scheduler internal state. load_state() restores
            # base_lrs from Phase 0 (1e-5), so scheduler.step() overrides the
            # optimizer reset above (1e-5 × cosine_multiplier ≠ 5e-6).
            # Fix: set base_lrs to the new LR and jump to end of warmup so the
            # LR starts at new_lr immediately and decays from there.
            warmup_steps = max(1, args.epochs // 10)
            for sched, new_lr in [(lr_scheduler, args.learning_rate),
                                   (lr_scheduler_aux, args.aux_learning_rate)]:
                underlying = sched.scheduler if hasattr(sched, 'scheduler') else sched
                underlying.base_lrs = [new_lr] * len(underlying.base_lrs)
                underlying.last_epoch = warmup_steps
                if hasattr(underlying, '_last_lr'):
                    underlying._last_lr = [new_lr] * len(underlying._last_lr)
                if hasattr(underlying, '_step_count'):
                    underlying._step_count = warmup_steps + 1
            if accelerator.is_main_process:
                logger_train.info(f"LR reset:        lr={args.learning_rate}  "
                                  f"aux_lr={args.aux_learning_rate}  "
                                  f"(scheduler reset to step {warmup_steps}, LR={args.learning_rate})")
        if accelerator.is_main_process:
            logger_train.info(f"Resumed from:    {args.resume_accelerate}  "
                              f"(epoch={start_epoch}, best_loss={best_loss:.4f})")

    if accelerator.is_main_process:
        logger_train.info(f"Experiment:      {args.experiment}")
        logger_train.info(f"FP32 checkpoint: {args.checkpoint}")
        logger_train.info(f"Bits:            {args.bits}")
        logger_train.info(f"Lambda:          {args.lmbda}")
        logger_train.info(f"Epochs:          {args.epochs}  (start_epoch={start_epoch})")
        logger_train.info(f"LR:              {args.learning_rate}")
        logger_train.info(f"Batch size:      {args.batch_size}  (per process)")
        eff = args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes
        logger_train.info(f"Eff. batch size: {eff}  ({args.batch_size}/GPU × {accelerator.num_processes} GPUs × {args.gradient_accumulation_steps} accum)")
        steps_msg = f"{args.max_steps_per_epoch:,}" if args.max_steps_per_epoch else "full dataset"
        logger_train.info(f"Steps/epoch:     {steps_msg}")

    # ── Training loop ─────────────────────────────────────────────────────────
    quality = Path(Path(args.checkpoint).stem).stem  # "0.0018.pth.tar" → "0.0018"
    steps_per_epoch = args.max_steps_per_epoch or math.ceil(len(train_loader.dataset) / args.batch_size)
    current_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, args.epochs):
        current_step = _train_one_epoch(
            net, criterion, train_loader,
            optimizer, aux_optimizer,
            epoch, args.clip_max_norm,
            logger_train, tb_logger,
            current_step, accelerator,
            max_steps_per_epoch=args.max_steps_per_epoch,
        )

        val_loss = test_one_epoch(
            epoch, test_loader, net, None, criterion,
            logger_val, tb_logger, image_metric, accelerator,
        )

        lr_scheduler.step()
        lr_scheduler_aux.step()

        if accelerator.is_main_process:
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            logger_train.info(
                f"Epoch {epoch+1}/{args.epochs}  val_loss={val_loss:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
                + ("  [BEST]" if is_best else "")
            )

            if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
                save_checkpoint(
                    accelerator,
                    {"epoch": epoch + 1, "loss": val_loss},
                    is_best,
                    str(checkpoint_path),
                    model=net,
                )

        accelerator.wait_for_everyone()

    # ── Final evaluation ──────────────────────────────────────────────────────
    # Sync all processes before the main process runs standalone evaluation.
    # Non-main processes will exit cleanly after this barrier.
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger_train.info("\n" + "=" * 60)
        logger_train.info("FINAL EVALUATION")
        logger_train.info("=" * 60)

        # Unwrap from accelerate / DDP
        raw_net = accelerator.unwrap_model(net)
        raw_net.eval()

        # Save deployment state dict (permanent QDQ applied, standard keys)
        deploy_sd = extract_deployment_state_dict(raw_net, bits=args.bits)
        deploy_path = os.path.join(args.output_dir, f"LoC_LIC_int{args.bits}_qat.pt")
        torch.save({"state_dict": deploy_sd, "bits": args.bits, "lmbda": args.lmbda},
                   deploy_path)
        logger_train.info(f"Deployment model saved: {deploy_path}")

        eval_device = torch.device(device.type if hasattr(device, "type") else str(device))
        eval_datasets = [(args.test_dataset, "kodak")]
        if args.test_dataset_pro2020:
            eval_datasets.append((args.test_dataset_pro2020, "pro_2020"))

        for ds_path, ds_name in eval_datasets:
            if not os.path.isdir(ds_path):
                logger_train.warning(f"Test dataset not found, skipping: {ds_path}")
                continue
            logger_train.info(f"\n{'='*60}")
            logger_train.info(f"Evaluating on {ds_name}")
            logger_train.info(f"{'='*60}")
            evaluate_dataset_fast(
                raw_net,
                images_dir=ds_path,
                output_dir=args.output_dir,
                quality=quality,
                model_name=f"LoC_LIC_int{args.bits}_qat",
                dataset_name=ds_name,
                device=eval_device,
            )

        logger_train.info("\nQAT training complete.")
        logger_train.info(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
