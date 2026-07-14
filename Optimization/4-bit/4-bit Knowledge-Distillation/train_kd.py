#!/usr/bin/env python3
"""
Knowledge Distillation QAT for LoC-LIC (Phase 4).

Adds a feature-level distillation loss on top of the standard RD objective:
    L_total = L_RD + alpha * mean_i( MSE(student_feat_i, teacher_feat_i) )

The FP32 model acts as teacher; the mixed-precision quantized model (Phase 3
checkpoint) is the student. Hooks capture intermediate features at g_a (analysis
transform), h_s (hyper decoder), and g_s (synthesis transform) by default.

Usage:
  cd /path/to/repo
  python "Optimization/4-bit/4-bit Knowledge-Distillation/train_kd.py" \\
      --checkpoint checkpoints/loc-lic/0.0018.pth.tar \\
      --teacher_checkpoint checkpoints/loc-lic/0.0018.pth.tar \\
      --layer_bits_file "Optimization/4-bit/4-bit Mixed-Precision/layer_bits.json" \\
      --lmbda 0.0018

  SLURM:
      sbatch "Optimization/4-bit/4-bit Knowledge-Distillation/run_kd.sh"
"""

import argparse
import json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))                          # knowledge-distillation/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mixed-precision QAT"))  # mixed-precision QAT/
from kd_modules import (
    DEFAULT_KD_LAYERS,
    register_kd_hooks,
    remove_kd_hooks,
    compute_kd_loss,
)
from qat_modules_mixed import (
    register_mixed_qat_parametrizations,
    extract_deployment_state_dict_mixed,
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
    logger.info(f"Loading FP32 checkpoint: {checkpoint_path}")
    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    net.load_state_dict(state)
    logger.info("FP32 checkpoint loaded successfully.")
    return net


# ─────────────────────────────────────────────────────────────────────────────
# Fast evaluation (likelihoods BPP)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset_fast(net, images_dir, output_dir, quality,
                          model_name, dataset_name, device):
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

    metrics_dict = {}
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
    p = argparse.ArgumentParser(description="Knowledge Distillation QAT for LoC-LIC")

    # Paths
    p.add_argument("--checkpoint",          required=True,
                   help="FP32 checkpoint used to initialise student weights "
                        "(e.g. checkpoints/loc-lic/0.0018.pth.tar)")
    p.add_argument("--teacher_checkpoint",  required=True,
                   help="Path to FP32 teacher checkpoint (e.g. checkpoints/loc-lic/0.0018.pth.tar)")
    p.add_argument("--layer_bits_file",     default=None,
                   help="JSON from sensitivity_analysis.py with per-layer bit assignments. "
                        "If omitted, all layers use --default_bits (uniform quantization).")
    p.add_argument("--dataset",             default="/home/woody/iwnt/iwnt153h/data/images")
    p.add_argument("--dataset_csv",         default="train_256")
    p.add_argument("--test_dataset",        default=None)
    p.add_argument("--test_dataset_pro2020",default=None)
    p.add_argument("--output_dir",          default=None)
    p.add_argument("--config",              default="loclic")

    # Loss
    p.add_argument("--lmbda",               type=float, default=0.0018)

    # Training
    p.add_argument("--epochs",              type=int,   default=100)
    p.add_argument("--batch_size",          type=int,   default=8)
    p.add_argument("--test_batch_size",     type=int,   default=16)
    p.add_argument("--learning_rate",       type=float, default=1e-5)
    p.add_argument("--aux_learning_rate",   type=float, default=1e-4)
    p.add_argument("--clip_max_norm",       type=float, default=1.0)
    p.add_argument("--patch_size",          type=int,   default=256)
    p.add_argument("--num_workers",         type=int,   default=16)
    p.add_argument("--save_every",          type=int,   default=5)
    p.add_argument("--mixed_precision",     type=str,   default="no",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_steps_per_epoch", type=int,   default=4000)
    p.add_argument("--resume_accelerate",   default=None,
                   help="Accelerate checkpoint dir to resume from (for chaining 24h jobs).")
    p.add_argument("--reset_lr",            action="store_true",
                   help="After loading resume checkpoint, override saved LR with "
                        "--learning_rate and --aux_learning_rate, and reset scheduler.")

    # Quantization
    p.add_argument("--default_bits",        type=int,   default=4,
                   help="Bit-width for layers not listed in --layer_bits_file (default: 4)")
    p.add_argument("--no_skip_first_last",  action="store_true")

    # Knowledge distillation
    p.add_argument("--kd_alpha",            type=float, default=0.1,
                   help="Weight for KD distillation loss (default: 0.1)")
    p.add_argument("--kd_layers",           default=",".join(DEFAULT_KD_LAYERS),
                   help="Comma-separated list of module names to distill "
                        "(default: g_a,h_s,g_s)")

    # Experiment
    p.add_argument("--experiment",          default="knowledge-distillation LOCLIC")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def _train_one_epoch(net, criterion, train_loader, optimizer, aux_optimizer,
                     epoch, clip_max_norm, logger_train, tb_logger,
                     current_step, accelerator, max_steps_per_epoch=None,
                     teacher=None, teacher_hooks=None, student_hooks=None,
                     kd_alpha=0.1):
    net.train()
    last_aux_loss = torch.tensor(0.0)
    for i, image in enumerate(train_loader):
        if max_steps_per_epoch is not None and i >= max_steps_per_epoch:
            break
        with accelerator.accumulate(net):
            with accelerator.autocast():
                out_net       = net(image)
                out_criterion = criterion(out_net, image)
                rd_loss       = out_criterion["loss"]

                # Knowledge distillation
                kd = torch.tensor(0.0, device=rd_loss.device)
                if teacher is not None and teacher_hooks and student_hooks:
                    with torch.no_grad():
                        _ = teacher(image)
                    kd = compute_kd_loss(teacher_hooks, student_hooks, kd_alpha)

                loss = rd_loss + kd

            accelerator.backward(loss)
            if accelerator.sync_gradients and clip_max_norm > 0:
                accelerator.clip_grad_norm_(net.parameters(), clip_max_norm)
            optimizer.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                aux_optimizer.zero_grad()
                last_aux_loss = accelerator.unwrap_model(net).get_aux_loss()
                last_aux_loss.backward()
                aux_optimizer.step()

        if current_step % 100 == 0 and accelerator.is_main_process:
            if tb_logger is not None:
                tb_logger.add_scalar('[train]: loss',     loss.item(),                      current_step)
                tb_logger.add_scalar('[train]: bpp_loss', out_criterion["bpp_loss"].item(), current_step)
                tb_logger.add_scalar('[train]: lr',       optimizer.param_groups[0]['lr'],  current_step)
                tb_logger.add_scalar('[train]: aux_loss', last_aux_loss.item(),             current_step)
                tb_logger.add_scalar('[train]: mse_loss', out_criterion["mse_loss"].item(), current_step)
                tb_logger.add_scalar('[train]: kd_loss',  kd.item(),                        current_step)
                tb_logger.add_scalar('[train]: rd_loss',  rd_loss.item(),                   current_step)
            logger_train.info(
                f"Train epoch {epoch}: [{i * len(image):5d}/{len(train_loader.dataset)} "
                f"({100. * i / len(train_loader):.0f}%)] "
                f'Loss: {loss.item():.4f} | '
                f'RD: {rd_loss.item():.4f} | '
                f'KD: {kd.item():.4f} | '
                f'Bpp: {out_criterion["bpp_loss"].item():.2f} | '
                f"Aux: {last_aux_loss.item():.2f}"
            )
        current_step += 1
    return current_step


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    setup_environment(args.seed)

    repo = _REPO
    if args.test_dataset is None:
        args.test_dataset = str(repo / "datasets" / "kodak")
    if args.output_dir is None:
        args.output_dir = str(repo / "results" / "4-bit" / "phase4-kd-v1")

    experiment_path = repo / "experiments" / args.experiment
    checkpoint_path = experiment_path / "checkpoints"

    # ── Accelerate ────────────────────────────────────────────────────────────
    ddp_kwargs  = DistributedDataParallelKwargs(find_unused_parameters=False)
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
    exp_tag = args.experiment.replace(" ", "_")
    setup_logger("train", str(experiment_path), f"train_{exp_tag}",
                 level=logging.INFO, screen=True, tofile=True)
    setup_logger("val",   str(experiment_path), f"val_{exp_tag}",
                 level=logging.INFO, screen=True, tofile=True)
    logger_train = logging.getLogger("train")
    logger_val   = logging.getLogger("val")
    tb_logger = SummaryWriter(log_dir=str(experiment_path)) if accelerator.is_main_process else None

    # ── Load layer_bits dict ──────────────────────────────────────────────────
    layer_bits: dict = {}
    if args.layer_bits_file:
        with open(args.layer_bits_file) as f:
            data = json.load(f)
        layer_bits = data.get("layer_bits", data)
        if accelerator.is_main_process:
            summary = data.get("summary", {})
            logger_train.info(
                f"Layer bits loaded: {summary.get('n_int8', '?')} INT8, "
                f"{summary.get('n_int4', '?')} INT4, "
                f"{summary.get('n_fp32', '?')} FP32"
            )

    # ── Model ─────────────────────────────────────────────────────────────────
    config_path = repo / "configs" / "models" / f"{args.config}.yaml"
    model_cfg   = model_config(str(config_path))
    net, _, _   = get_model(model_cfg, args, device)

    if args.resume_accelerate:
        net = load_fp32_checkpoint(net, args.checkpoint, device)
        n_int4, n_int8, n_skip = register_mixed_qat_parametrizations(
            net, layer_bits=layer_bits, default_bits=args.default_bits,
            skip_first_last_conv=not args.no_skip_first_last,
        )
        if accelerator.is_main_process:
            logger_train.info(
                f"Mixed-QAT parametrizations registered ({n_int4} INT4, {n_int8} INT8, "
                f"{n_skip} FP32). Weights will be loaded from: {args.resume_accelerate}"
            )
        start_epoch = 0
        best_loss   = float("inf")
    else:
        net = load_fp32_checkpoint(net, args.checkpoint, device)
        n_int4, n_int8, n_skip = register_mixed_qat_parametrizations(
            net, layer_bits=layer_bits, default_bits=args.default_bits,
            skip_first_last_conv=not args.no_skip_first_last,
        )
        if accelerator.is_main_process:
            logger_train.info(f"Mixed-QAT: {n_int4} INT4, {n_int8} INT8, {n_skip} FP32 (skipped).")
        start_epoch = 0
        best_loss   = float("inf")

    net = net.to(device)

    # ── Teacher model (FP32, frozen) ──────────────────────────────────────────
    teacher = None
    teacher_hooks: dict = {}
    student_hooks: dict = {}
    kd_layer_names = [s.strip() for s in args.kd_layers.split(",") if s.strip()]

    if accelerator.is_main_process:
        logger_train.info(f"KD alpha:        {args.kd_alpha}")
        logger_train.info(f"KD layers:       {kd_layer_names}")

    teacher_net, _, _ = get_model(model_cfg, args, device)
    teacher_net = load_fp32_checkpoint(teacher_net, args.teacher_checkpoint, device)
    teacher_net.eval()
    for p in teacher_net.parameters():
        p.requires_grad_(False)
    teacher = teacher_net.to(device)

    # Register KD hooks before DDP wrapping — hooks survive DDP wrapping
    teacher_hooks = register_kd_hooks(teacher, kd_layer_names)
    student_hooks = register_kd_hooks(net, kd_layer_names)

    # ── Optimizers & schedulers ───────────────────────────────────────────────
    criterion       = RateDistortionLoss(lmbda=args.lmbda)
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
    train_dataset = ImageFolderCSV(
        args.dataset, csv_file=str(csv_path),
        transform=transforms.Compose([
            transforms.RandomCrop(args.patch_size), transforms.ToTensor()
        ])
    )
    test_dataset = ImageFolder(
        os.path.join(args.dataset, "test"),
        transform=transforms.Compose([
            transforms.ToTensor(), transforms.CenterCrop(args.patch_size)
        ])
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=True, drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.test_batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=True)

    # ── Accelerate prepare ────────────────────────────────────────────────────
    # NOTE: teacher is NOT passed to accelerator.prepare() — it stays as a
    # plain PyTorch model on each process's device.
    (net, criterion, optimizer, aux_optimizer,
     train_loader, test_loader,
     lr_scheduler, lr_scheduler_aux,
     image_metric) = accelerator.prepare(
        net, criterion, optimizer, aux_optimizer,
        train_loader, test_loader,
        lr_scheduler, lr_scheduler_aux,
        image_metric,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume_accelerate:
        ckpt_dir = Path(args.resume_accelerate)
        accelerator.load_state(str(ckpt_dir))
        meta_path = ckpt_dir / "metadata.pth"
        if meta_path.exists():
            meta        = torch.load(str(meta_path), map_location="cpu")
            start_epoch = meta.get("epoch", 0)
            best_loss   = meta.get("loss", float("inf"))
        else:
            try:
                start_epoch = int(ckpt_dir.name.split("_")[-1])
            except ValueError:
                start_epoch = 0
            best_loss = float("inf")

        if args.reset_lr:
            for pg in optimizer.param_groups:
                pg["lr"] = args.learning_rate
            for pg in aux_optimizer.param_groups:
                pg["lr"] = args.aux_learning_rate
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
                logger_train.info(
                    f"LR reset:        lr={args.learning_rate}  "
                    f"aux_lr={args.aux_learning_rate}  (scheduler reset to step {warmup_steps})"
                )
        if accelerator.is_main_process:
            logger_train.info(
                f"Resumed from:    {args.resume_accelerate}  "
                f"(epoch={start_epoch}, best_loss={best_loss:.4f})"
            )

    if accelerator.is_main_process:
        logger_train.info(f"Experiment:      {args.experiment}")
        logger_train.info(f"FP32 checkpoint: {args.checkpoint}")
        logger_train.info(f"Teacher ckpt:    {args.teacher_checkpoint}")
        logger_train.info(f"Layer bits file: {args.layer_bits_file or 'none (uniform INT' + str(args.default_bits) + ')'}")
        logger_train.info(f"Lambda:          {args.lmbda}")
        logger_train.info(f"Epochs:          {args.epochs}  (start_epoch={start_epoch})")
        logger_train.info(f"LR:              {args.learning_rate}")
        logger_train.info(f"Batch size:      {args.batch_size}  (per process)")
        eff = args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes
        logger_train.info(
            f"Eff. batch size: {eff}  "
            f"({args.batch_size}/GPU × {accelerator.num_processes} GPUs × "
            f"{args.gradient_accumulation_steps} accum)"
        )
        steps_msg = f"{args.max_steps_per_epoch:,}" if args.max_steps_per_epoch else "full dataset"
        logger_train.info(f"Steps/epoch:     {steps_msg}")

    # ── Training loop ─────────────────────────────────────────────────────────
    quality = Path(Path(args.checkpoint).stem).stem
    steps_per_epoch = args.max_steps_per_epoch or math.ceil(
        len(train_loader.dataset) / args.batch_size
    )
    current_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, args.epochs):
        current_step = _train_one_epoch(
            net, criterion, train_loader,
            optimizer, aux_optimizer,
            epoch, args.clip_max_norm,
            logger_train, tb_logger,
            current_step, accelerator,
            max_steps_per_epoch=args.max_steps_per_epoch,
            teacher=teacher,
            teacher_hooks=teacher_hooks,
            student_hooks=student_hooks,
            kd_alpha=args.kd_alpha,
        )

        val_loss = test_one_epoch(
            epoch, test_loader, net, None, criterion,
            logger_val, tb_logger, image_metric, accelerator,
        )

        lr_scheduler.step()
        lr_scheduler_aux.step()

        if accelerator.is_main_process:
            is_best   = val_loss < best_loss
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

    # ── Clean up KD hooks ─────────────────────────────────────────────────────
    remove_kd_hooks(teacher_hooks)
    remove_kd_hooks(student_hooks)

    # ── Final evaluation ──────────────────────────────────────────────────────
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger_train.info("\n" + "=" * 60)
        logger_train.info("FINAL EVALUATION")
        logger_train.info("=" * 60)

        raw_net = accelerator.unwrap_model(net)
        raw_net.eval()

        deploy_sd   = extract_deployment_state_dict_mixed(raw_net)
        deploy_path = os.path.join(args.output_dir, "LoC_LIC_kd.pt")
        torch.save({"state_dict": deploy_sd, "layer_bits": layer_bits, "lmbda": args.lmbda},
                   deploy_path)
        logger_train.info(f"Deployment model saved: {deploy_path}")

        eval_device  = torch.device(device.type if hasattr(device, "type") else str(device))
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
                model_name="LoC_LIC_kd",
                dataset_name=ds_name,
                device=eval_device,
            )

        logger_train.info("\nKD QAT training complete.")
        logger_train.info(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
