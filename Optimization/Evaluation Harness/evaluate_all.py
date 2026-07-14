#!/usr/bin/env python3
"""
LoC-LIC Full Evaluation Harness
================================
Evaluates all 9 model variants (FP32, GPTQ, PTQ, QAT, Mixed QAT, KD)
on Kodak and pro_2020 datasets.

Metrics collected per model:
  - PSNR (RGB), Y-PSNR (luma), SSIM, MS-SSIM, LPIPS  (image quality)
  - BPP                            (rate, from likelihoods)
  - Enc time mean ± std (ms/image) (FULL encode: NN + arithmetic range coding)
  - Dec time mean ± std (ms/image) (FULL decode: arithmetic coding + NN)
  - Bitstream size (KB)            (average compressed bytes per image)
  - Latency mean ± std (ms/image)  (NN forward-only, GPU only)
  - Peak VRAM (MB)                 (GPU memory during inference)
  - GMACs / kMAC-per-pixel         (neural-net compute, Conv2d + Linear)
  - Deployment weight size (MB)    (theoretical compressed weights)
  - Checkpoint size on disk (MB)

Outputs written to  results/eval_harness/eval_<timestamp>/
  evaluation_<dataset>.csv
  evaluation_<dataset>.json
  rd_psnr_bpp_<dataset>.png
  quality_vs_size_<dataset>.png

Submit via SLURM:
  sbatch --partition=rtx3080 --gres=gpu:1 --cpus-per-task=8 --time=6:00:00 \\
         "Optimization/Evaluation Harness/run_eval.sh"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# ── Repository root ───────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [
    str(REPO_ROOT),
    str(REPO_ROOT / "Optimization" / "4-bit" / "4-bit Mixed-Precision"),
    str(REPO_ROOT / "Optimization" / "4-bit" / "4-bit QAT"),
    str(REPO_ROOT / "Optimization" / "Lic Container Format"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets.datasets import ImageFolder                         # noqa: E402
from models.get_model import get_model                           # noqa: E402
from utils.config import model_config                            # noqa: E402
from utils.func import update_registered_buffers                 # noqa: E402
from utils.metrics import ImageMetric                            # noqa: E402
from lic_format import write_lic, read_lic                       # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
_CFG_PATH      = REPO_ROOT / "configs" / "models" / "loclic.yaml"
_LAYER_BITS    = REPO_ROOT / "Optimization" / "4-bit" / "4-bit Mixed-Precision" / "layer_bits.json"

_DATASET_PATHS = {
    "kodak":    "/home/atuin/b290dc/b290dc10/TestDataset/images/Kodak",
    "pro_2020": "/home/atuin/b290dc/b290dc10/TestDataset/images/pro_2020",
}

# ── Model registry ────────────────────────────────────────────────────────────
# deploy_size_mb = theoretical compressed weights (not the checkpoint on disk):
#   FP32  = 69.9M × 4 B / 1024² =  268 MB
#   INT8  = 69.9M × 1 B / 1024² =   67 MB
#   INT4  = 69.9M × 0.5 B / 1024² = 34 MB
#   Mixed = 69.9M × (0.45×1 + 0.55×0.5) B / 1024² = 48 MB  (45% INT8, 55% INT4)

@dataclass
class ModelEntry:
    key:             str
    name:            str    # display name
    checkpoint:      Path
    loader:          str    # 'fp32' | 'gptq_ptq' | 'int4_qat' | 'mixed_qat'
    bits:            str    # '32' | '8' | '4' | '4/8'
    method:          str
    deploy_size_mb:  float
    lic_lambda:      float          = 0.0018  # lambda stored in .lic header
    lic_quant:       Optional[str]  = None    # None = skip .lic round-trip test


MODEL_REGISTRY: List[ModelEntry] = [
    # ── λ = 0.0018 (primary operating point, fully evaluated) ─────────────────
    ModelEntry(
        "fp32",       "FP32 Baseline",
        REPO_ROOT / "checkpoints/loc-lic/0.0018.pth.tar",
        "fp32", "32", "—", 268.0,
        lic_lambda=0.0018, lic_quant='fp32',
    ),
    ModelEntry(
        "int8_gptq",  "INT8 GPTQ",
        REPO_ROOT / "results/legacy/old_kodak_test/gptq_second_try/LoC_LIC_int8_gptq_8bit.pt",
        "gptq_ptq", "8", "GPTQ, no fine-tune", 67.0,
        lic_lambda=0.0018, lic_quant='int8_gptq',
    ),
    ModelEntry(
        "int8_ptq",   "INT8 PTQ",
        REPO_ROOT / "results/legacy/old_kodak_test/ptq_fast_test/LoC_LIC_int8_ptq.pt",
        "gptq_ptq", "8", "Weight-only QDQ", 67.0,
        lic_lambda=0.0018, lic_quant='int8_gptq',   # closest type in lic format
    ),
    ModelEntry(
        "int4_gptq",  "INT4 GPTQ",
        REPO_ROOT / "results/legacy/old_kodak_test/gptq_4bit/LoC_LIC_int4_gptq_4bit.pt",
        "gptq_ptq", "4", "GPTQ, no fine-tune", 34.0,
        lic_lambda=0.0018, lic_quant=None,           # not in production set, skip
    ),
    ModelEntry(
        "int4_qat",   "INT4 QAT (Phase 1)",
        REPO_ROOT / "results/4-bit/phase1-uniform-int4/lambda-0018/LoC_LIC_int4_qat.pt",
        "int4_qat", "4", "QAT, 100 epochs", 34.0,
        lic_lambda=0.0018, lic_quant='qat_best',
    ),
    ModelEntry(
        "mixed_p2",   "Mixed QAT Phase 2",
        REPO_ROOT / "results/4-bit/phase2-mixed/lambda-0018/LoC_LIC_mixed_qat.pt",
        "mixed_qat", "4/8", "Mixed INT8/INT4 QAT, 100 epochs", 48.0,
        lic_lambda=0.0018, lic_quant='qat_best',
    ),
    ModelEntry(
        "mixed_p3",   "Mixed QAT Phase 3 ★",
        REPO_ROOT / "results/4-bit/phase3-mixed-finetuned/lambda-0018/LoC_LIC_mixed_qat.pt",
        "mixed_qat", "4/8", "Mixed QAT, lower LR, 50 epochs", 48.0,
        lic_lambda=0.0018, lic_quant='qat_best',
    ),
    ModelEntry(
        "kd_p4",      "KD QAT (Phase 4)",
        REPO_ROOT / "results/4-bit/phase4-kd-v1/LoC_LIC_kd.pt",
        "mixed_qat", "4/8", "QAT + feature distillation, 100 epochs", 48.0,
        lic_lambda=0.0018, lic_quant='qat_best',
    ),

    # ── Additional λ values for R-D curves (Phase 1a/1b of project roadmap) ───
    ModelEntry(
        "fp32_0035",  "FP32 λ=0.0035",
        REPO_ROOT / "checkpoints/loc-lic/0.0035.pth.tar",
        "fp32", "32", "—", 268.0,
        lic_lambda=0.0035, lic_quant='fp32',
    ),
    ModelEntry(
        "fp32_0067",  "FP32 λ=0.0067",
        REPO_ROOT / "checkpoints/loc-lic/0.0067.pth.tar",
        "fp32", "32", "—", 268.0,
        lic_lambda=0.0067, lic_quant='fp32',
    ),
    ModelEntry(
        "fp32_025",   "FP32 λ=0.025",
        REPO_ROOT / "checkpoints/loc-lic/0.025.pth.tar",
        "fp32", "32", "—", 268.0,
        lic_lambda=0.025, lic_quant='fp32',
    ),
    # ── Phase 1b: INT8 GPTQ at remaining λ values (created by run_gptq_all_lambdas.sh) ──
    ModelEntry(
        "int8_gptq_0035", "INT8 GPTQ λ=0.0035",
        REPO_ROOT / "results/8-bit/gptq-per-lambda/lambda-0035/LoC_LIC_int8_gptq_8bit.pt",
        "gptq_ptq", "8", "GPTQ, no fine-tune", 67.0,
        lic_lambda=0.0035, lic_quant='int8_gptq',
    ),
    ModelEntry(
        "int8_gptq_0067", "INT8 GPTQ λ=0.0067",
        REPO_ROOT / "results/8-bit/gptq-per-lambda/lambda-0067/LoC_LIC_int8_gptq_8bit.pt",
        "gptq_ptq", "8", "GPTQ, no fine-tune", 67.0,
        lic_lambda=0.0067, lic_quant='int8_gptq',
    ),
    ModelEntry(
        "int8_gptq_025",  "INT8 GPTQ λ=0.025",
        REPO_ROOT / "results/8-bit/gptq-per-lambda/lambda-025/LoC_LIC_int8_gptq_8bit.pt",
        "gptq_ptq", "8", "GPTQ, no fine-tune", 67.0,
        lic_lambda=0.025, lic_quant='int8_gptq',
    ),
    # ── Phase 1c: Mixed QAT at remaining lambdas (50 epochs each) ────────────
    ModelEntry(
        "qat_mixed_0035", "Mixed QAT λ=0.0035",
        REPO_ROOT / "results/4-bit/phase3-mixed-finetuned/lambda-0035/LoC_LIC_mixed_qat.pt",
        "mixed_qat", "4/8", "Mixed INT4/INT8 QAT, 50 epochs", 48.0,
        lic_lambda=0.0035, lic_quant='qat_best',
    ),
    ModelEntry(
        "qat_mixed_0067", "Mixed QAT λ=0.0067",
        REPO_ROOT / "results/4-bit/phase3-mixed-finetuned/lambda-0067/LoC_LIC_mixed_qat.pt",
        "mixed_qat", "4/8", "Mixed INT4/INT8 QAT, 50 epochs", 48.0,
        lic_lambda=0.0067, lic_quant='qat_best',
    ),
    ModelEntry(
        "qat_mixed_025",  "Mixed QAT λ=0.025",
        REPO_ROOT / "results/4-bit/phase3-mixed-finetuned/lambda-025/LoC_LIC_mixed_qat.pt",
        "mixed_qat", "4/8", "Mixed INT4/INT8 QAT, 50 epochs", 48.0,
        lic_lambda=0.025, lic_quant='qat_best',
    ),
    # ── KD v2 ablation (λ=0.0018 only — peaked at 27.53 dB, below Mixed P3) ──
    ModelEntry(
        "kd_v2",      "KD QAT v2 (ablation)",
        REPO_ROOT / "results/4-bit/phase4b-kd-v2/LoC_LIC_kd.pt",
        "mixed_qat", "4/8", "KD from P3, alpha=0.05, 50 epochs", 48.0,
        lic_lambda=0.0018, lic_quant='qat_best',
    ),
]


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class ModelResult:
    key:                 str
    name:                str
    bits:                str
    method:              str
    deploy_size_mb:      float
    checkpoint_size_mb:  float
    # Image quality
    psnr:           float = float("nan")   # RGB-PSNR
    y_psnr:         float = float("nan")   # Y-PSNR (luma only — matches LIC paper convention)
    ssim:           float = float("nan")
    ms_ssim:        float = float("nan")
    lpips:          float = float("nan")
    likelihood_bpp: float = float("nan")   # BPP from forward-pass likelihoods (inflated ~10%)
    actual_bpp:     float = float("nan")   # BPP from real bitstream (per-image avg over timing subset)
    delta_psnr:     float = float("nan")   # vs FP32 (RGB), filled after all models run
    delta_y_psnr:   float = float("nan")   # vs FP32 (Y),   filled after all models run
    # Codec timing  (full pipeline: NN + arithmetic range coding)
    enc_time_mean_ms:  float = float("nan")
    enc_time_std_ms:   float = float("nan")
    dec_time_mean_ms:  float = float("nan")
    dec_time_std_ms:   float = float("nan")
    bitstream_kb:      float = float("nan")   # average compressed bytes per image
    # NN-only latency (forward pass without range coding)
    latency_mean_ms:   float = float("nan")
    latency_std_ms:    float = float("nan")
    vram_mb:           float = float("nan")
    gmacs:             float = float("nan")
    kmac_per_pixel:    float = float("nan")
    # Status
    loaded:        bool = False
    error:         str  = ""
    lic_roundtrip: str  = "—"   # "✓" | "✗" | "—" (skipped)


# ── Model construction ────────────────────────────────────────────────────────

def _build_model(device: torch.device) -> nn.Module:
    """Create an FP32 LoC-LIC model from the YAML config (no weights loaded)."""
    cfg = model_config(str(_CFG_PATH))
    args = argparse.Namespace(lmbda=0.0018)
    net, _, _ = get_model(cfg, args, device)
    return net.to(device)


def _extract_state(raw) -> dict:
    """Extract the weight state_dict from a raw torch.load result."""
    if isinstance(raw, dict):
        return raw.get("state_dict", raw.get("model", raw))
    return raw  # full model object — caller detects via isinstance check


def _load_weights(model: nn.Module, state: dict) -> None:
    """
    Load a state_dict into model via CompressAI's CompressionModel.load_state_dict.

    Using CompressionModel.load_state_dict instead of nn.Module.load_state_dict is
    required because:

    1. Key-name remapping (old → new CompressAI):
       EntropyBottleneck density-estimator parameters were renamed in CompressAI 1.2+
       from _matrix0/_bias1/_factor2 (flat attributes) to matrices.0/biases.1/factors.2
       (ParameterList).  CompressionModel.load_state_dict calls remap_old_keys() for
       every EntropyBottleneck it finds, so checkpoints saved with either naming load
       correctly.  Calling nn.Module.load_state_dict directly skips this remap, leaving
       the density estimator at random initial values and inflating z_likelihoods BPP
       by ~8× (observed: 0.169 instead of ~0.020 per image at λ=0.0018).

    2. Buffer size pre-allocation:
       CompressAI's CompressionModel.load_state_dict calls update_registered_buffers
       for both EntropyBottleneck and GaussianConditional automatically, so the CDF
       buffers are resized to match the checkpoint before the state_dict is applied.

    3. QAT parametrization compatibility:
       strict=False allows keys that differ between the bare model and a QAT checkpoint
       (.parametrizations.weight.original) to be skipped gracefully.
    """
    from compressai.models import CompressionModel
    CompressionModel.load_state_dict(model, state, strict=False)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_fp32(entry: ModelEntry, device: torch.device) -> nn.Module:
    raw = torch.load(entry.checkpoint, map_location=device, weights_only=False)
    state = _extract_state(raw)
    if not isinstance(state, dict):
        return state.eval().to(device)
    model = _build_model(device)
    _load_weights(model, state)
    return model.eval()


def load_gptq_ptq(entry: ModelEntry, device: torch.device) -> nn.Module:
    """
    GPTQ and PTQ checkpoints store FP32 weight tensors whose values have been
    rounded to the INT8/INT4 quantization grid.  No special module structure is
    needed — load as a standard FP32 model.
    """
    raw = torch.load(entry.checkpoint, map_location=device, weights_only=False)
    state = _extract_state(raw)
    if not isinstance(state, dict):
        return state.eval().to(device)
    model = _build_model(device)
    _load_weights(model, state)
    return model.eval()


def load_int4_qat(entry: ModelEntry, device: torch.device) -> nn.Module:
    """
    INT4 QAT (Phase 1): checkpoints were saved with plain .weight keys (no
    parametrization structure).  Load weights first into a vanilla model, then
    apply fake-quant parametrizations.  register_parametrization() promotes the
    already-loaded .weight to .parametrizations.weight.original, so the trained
    values are preserved and model.weight returns quantize(trained_weight).
    """
    from qat_modules import register_qat_parametrizations  # type: ignore
    model = _build_model(device)
    raw = torch.load(entry.checkpoint, map_location=device, weights_only=False)
    state = _extract_state(raw)
    if not isinstance(state, dict):
        raise RuntimeError("Expected state_dict, got model object")
    _load_weights(model, state)                                      # weights loaded first
    register_qat_parametrizations(model, bits=4, skip_first_last_conv=True)
    return model.eval()


def load_mixed_qat(entry: ModelEntry, device: torch.device) -> nn.Module:
    """
    Mixed QAT (Phase 2/3/4): checkpoints were saved by extract_deployment_state_dict_mixed()
    which permanently bakes per-layer fake-quantization into the .weight tensors.
    Load those already-quantized weights directly — no parametrizations needed at
    inference time.  Re-applying fake_quant_ste on GPU produces subtly different
    per-channel scale factors than on CPU (reduction-ordering differences for the
    INT8 max computation), causing ~5 dB PSNR degradation on CUDA.
    """
    raw = torch.load(entry.checkpoint, map_location=device, weights_only=False)
    state = _extract_state(raw)
    if not isinstance(state, dict):
        return state.eval().to(device)
    model = _build_model(device)
    _load_weights(model, state)
    return model.eval()


_LOADERS = {
    "fp32":      load_fp32,
    "gptq_ptq":  load_gptq_ptq,
    "int4_qat":  load_int4_qat,
    "mixed_qat": load_mixed_qat,
}


def load_model(entry: ModelEntry, device: torch.device) -> Tuple[Optional[nn.Module], str]:
    if not entry.checkpoint.exists():
        return None, f"checkpoint not found: {entry.checkpoint}"
    try:
        return _LOADERS[entry.loader](entry, device), ""
    except Exception as exc:
        return None, str(exc)


# ── FLOPs estimation ──────────────────────────────────────────────────────────

def estimate_gmacs(
    model: nn.Module,
    h: int = 512,
    w: int = 768,
) -> Tuple[float, float]:
    """
    Count multiply-accumulate operations for Conv2d and Linear layers via
    forward hooks on a dummy (h × w) input.

    Returns (GMACs, kMAC_per_pixel).

    Note: entropy coding arithmetic (range coder) is NOT counted here.
    These are neural-network MACs only.  All precision variants have the same
    MAC count — quantization changes arithmetic precision, not operation count.
    """
    macs_log: List[int] = []
    hooks = []

    def conv_hook(m: nn.Conv2d, inp, out):
        _, C_out, H_out, W_out = out.shape
        macs_log.append(
            C_out * H_out * W_out * (m.in_channels // m.groups)
            * m.kernel_size[0] * m.kernel_size[1]
        )

    def linear_hook(m: nn.Linear, inp, out):
        n_outputs = out.numel() // out.shape[-1]
        macs_log.append(n_outputs * m.in_features * m.out_features)

    for _, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(linear_hook))

    dummy = torch.rand(1, 3, h, w)
    try:
        with torch.no_grad():
            model(dummy)
    except Exception:
        pass  # hooks fire before entropy coding — partial forward is fine

    for h_handle in hooks:
        h_handle.remove()

    total_macs = sum(macs_log)
    gmacs = total_macs / 1e9
    kmac_per_pixel = (total_macs / 1e3) / (h * w)
    return gmacs, kmac_per_pixel


# ── Padding helper (LoC-LIC requires input dims divisible by 64) ─────────────

def _pad_to_multiple(x: torch.Tensor, p: int = 64) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad spatial dims to next multiple of p. Returns (padded, (orig_H, orig_W))."""
    _, _, H, W = x.shape
    pad_h = (p - H % p) % p
    pad_w = (p - W % p) % p
    if pad_h or pad_w:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (H, W)


# ── Inference timing & VRAM ───────────────────────────────────────────────────

def measure_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_warmup: int = 3,
    n_measure: int = 15,
) -> Tuple[float, float, float]:
    """
    Returns (mean_latency_ms, std_latency_ms, peak_vram_mb).

    Latency: wall-clock time for one full forward pass (includes entropy
             probability estimation but NOT arithmetic range coding).
    VRAM:    peak GPU memory allocated during the measurement loop
             (model weights + forward activations).
    """
    model.eval()

    def _next(it):
        try:
            return next(it)
        except StopIteration:
            return next(iter(loader))

    it = iter(loader)

    # Warmup (fills CUDA kernel launch caches)
    with torch.no_grad():
        for _ in range(n_warmup):
            img, _ = _pad_to_multiple(_next(it).to(device))
            _ = model(img)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Reset VRAM counter after warmup
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(n_measure):
            img, _ = _pad_to_multiple(_next(it).to(device))
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(img)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

    vram_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else float("nan")
    )
    return float(np.mean(latencies)), float(np.std(latencies)), vram_mb


# ── Codec timing (full encode + decode pipeline) ──────────────────────────────

def measure_codec_time(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    entry: "ModelEntry",
    n_images: int = 10,
) -> Dict[str, float]:
    """
    Measure FULL encode and decode time including arithmetic range coding.

    This is the standard metric used in learned image compression papers.
    It captures everything: neural network forward pass (g_a, h_a, etc.) +
    the RANS arithmetic coder (the part that produces actual compressed bytes).

    Unlike the NN-only latency, this WILL differ between model variants because:
    - Entropy coding speed depends on the entropy model's distribution accuracy
    - More accurate distributions (from better QAT training) can code faster
    - The autoregressive context loop (10 slices) dominates encoding time

    The model already measures cost_time internally with cuda.synchronize(),
    so we use that rather than adding an outer timer.

    When entry.lic_quant is set, the bitstream is written to a temporary .lic
    file and read back before decoding, exercising the full round-trip.

    Returns dict with keys:
        enc_mean_ms, enc_std_ms, dec_mean_ms, dec_std_ms, bitstream_kb,
        lic_roundtrip ("✓" | "✗" | "—")
    """
    import tempfile
    import os as _os

    # Build CDF tables required for the range coder
    try:
        model.update(force=True)
    except Exception as exc:
        print(f"   [warn] model.update() failed ({exc}) — codec timing skipped")
        return {}

    model.eval()
    enc_times, dec_times, bitstream_sizes, bpp_values = [], [], [], []
    lic_ok = True   # flipped to False on first round-trip failure

    do_lic = entry.lic_quant is not None

    with torch.no_grad():
        for i, images in enumerate(loader):
            if i >= n_images:
                break
            x, (orig_H, orig_W) = _pad_to_multiple(images.to(device))

            # Encode
            try:
                enc_out = model.compress(x)
            except Exception as exc:
                print(f"   [warn] compress() failed on image {i}: {exc}")
                continue

            enc_times.append(enc_out["cost_time"] * 1000.0)   # s → ms

            # Bitstream size: sum bytes across all string lists
            strings = enc_out["strings"]
            total_bytes = 0
            for s in strings:
                if isinstance(s, (list, tuple)):
                    total_bytes += sum(len(b) for b in s if isinstance(b, (bytes, bytearray)))
                elif isinstance(s, (bytes, bytearray)):
                    total_bytes += len(s)
            bitstream_sizes.append(total_bytes / 1024.0)   # bytes → KB
            bpp_values.append(total_bytes * 8 / (orig_H * orig_W))

            # .lic round-trip: write → read → decode from reloaded data
            strings_for_dec = enc_out["strings"]
            shape_for_dec   = enc_out["shape"]
            if do_lic:
                fd, tmp_path = tempfile.mkstemp(suffix='.lic')
                _os.close(fd)
                try:
                    write_lic(tmp_path, enc_out,
                              lambda_val=entry.lic_lambda,
                              quant_type=entry.lic_quant,
                              img_shape=(orig_H, orig_W))
                    reloaded, _, _, _ = read_lic(tmp_path)
                    strings_for_dec = reloaded["strings"]
                    shape_for_dec   = reloaded["shape"]
                except Exception as exc:
                    print(f"   [warn] .lic round-trip failed on image {i}: {exc}")
                    lic_ok = False
                finally:
                    _os.unlink(tmp_path)

            # Decode
            try:
                dec_out = model.decompress(strings_for_dec, shape_for_dec)
                dec_times.append(dec_out["cost_time"] * 1000.0)
            except Exception as exc:
                print(f"   [warn] decompress() failed on image {i}: {exc}")

    if not enc_times:
        return {}

    if do_lic:
        lic_status = "✓" if lic_ok else "✗"
    else:
        lic_status = "—"

    return {
        "enc_mean_ms":    float(np.mean(enc_times)),
        "enc_std_ms":     float(np.std(enc_times)),
        "dec_mean_ms":    float(np.mean(dec_times)) if dec_times else float("nan"),
        "dec_std_ms":     float(np.std(dec_times))  if dec_times else float("nan"),
        "bitstream_kb":   float(np.mean(bitstream_sizes)) if bitstream_sizes else float("nan"),
        "actual_bpp":     float(np.mean(bpp_values))      if bpp_values     else float("nan"),
        "lic_roundtrip":  lic_status,
    }


# ── Quality metrics ───────────────────────────────────────────────────────────

def _rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """Convert [B,3,H,W] float [0,1] → Y luma [B,1,H,W] (BT.601)."""
    return (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]).unsqueeze(1)


def evaluate_quality(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_metric: ImageMetric,
) -> Dict[str, float]:
    """Run full-dataset evaluation and return averaged quality metrics."""
    psnr_l, y_psnr_l, ssim_l, msssim_l, lpips_l, bpp_l = [], [], [], [], [], []

    model.eval()
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            B, _, H, W = images.shape
            num_pixels = B * H * W

            images_padded, _ = _pad_to_multiple(images)
            out = model(images_padded)
            x_hat = out["x_hat"].clamp(0, 1)[:, :, :H, :W]

            m = image_metric.metric_image(x_hat, images)
            psnr_l.append(m.get("psnr",   float("nan")))
            ssim_l.append(m.get("ssim",   float("nan")))
            msssim_l.append(m.get("msssim", float("nan")))
            lpips_l.append(m.get("lpips",  float("nan")))

            # Y-PSNR (luma only — standard in LIC papers)
            y_ref  = _rgb_to_y(images.cpu())
            y_pred = _rgb_to_y(x_hat.cpu())
            mse_y  = ((y_ref - y_pred) ** 2).mean().item()
            y_psnr_l.append(10 * np.log10(1.0 / mse_y) if mse_y > 0 else float("inf"))

            bpp = sum(
                -torch.log2(lk).sum().item() / num_pixels
                for lk in out["likelihoods"].values()
            )
            bpp_l.append(bpp)

    return {
        "psnr":           float(np.nanmean(psnr_l)),
        "y_psnr":         float(np.nanmean(y_psnr_l)),
        "ssim":           float(np.nanmean(ssim_l)),
        "ms_ssim":        float(np.nanmean(msssim_l)),
        "lpips":          float(np.nanmean(lpips_l)),
        "likelihood_bpp": float(np.nanmean(bpp_l)),
    }


# ── Dataset loader ────────────────────────────────────────────────────────────

def get_loader(dataset_name: str, num_workers: int = 4) -> DataLoader:
    path = _DATASET_PATHS.get(dataset_name)
    if path is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {list(_DATASET_PATHS)}"
        )
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    ds = ImageFolder(path, transform=transforms.ToTensor())
    return DataLoader(
        ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )


# ── Output helpers ────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "key", "name", "bits", "method",
    "deploy_size_mb", "checkpoint_size_mb",
    "psnr", "y_psnr", "ssim", "ms_ssim", "lpips",
    "likelihood_bpp", "actual_bpp",
    "delta_psnr", "delta_y_psnr",
    "enc_time_mean_ms", "enc_time_std_ms",
    "dec_time_mean_ms", "dec_time_std_ms",
    "bitstream_kb",
    "latency_mean_ms", "latency_std_ms", "vram_mb",
    "gmacs", "kmac_per_pixel",
    "loaded", "error", "lic_roundtrip",
]


def save_csv(results: List[ModelResult], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k) for k in _CSV_FIELDS})


def save_json(results: List[ModelResult], path: Path) -> None:
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def plot_rd_curves(results: List[ModelResult], dataset: str, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("  [warn] matplotlib not available — plots skipped")
        return

    loaded = [r for r in results if r.loaded and not math.isnan(r.y_psnr)]
    if not loaded:
        return

    colors = cm.tab10.colors

    def _scatter(ax, xs, ys, labels):
        for i, (x, y, lbl) in enumerate(zip(xs, ys, labels)):
            ax.scatter(x, y, color=colors[i % 10], s=100, zorder=3)
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(6, 3), fontsize=8)

    # Y-PSNR vs actual BPP
    fig, ax = plt.subplots(figsize=(10, 6))
    _scatter(ax, [r.actual_bpp for r in loaded], [r.y_psnr for r in loaded],
             [r.name for r in loaded])
    ax.set_xlabel("BPP (actual bitstream)")
    ax.set_ylabel("Y-PSNR (dB)")
    ax.set_title(f"Rate–Distortion: Y-PSNR vs BPP — {dataset}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"rd_psnr_bpp_{dataset}.png", dpi=150)
    plt.close()

    # Y-PSNR vs deployment weight size
    fig, ax = plt.subplots(figsize=(10, 6))
    _scatter(ax, [r.deploy_size_mb for r in loaded], [r.y_psnr for r in loaded],
             [r.name for r in loaded])
    ax.set_xlabel("Deployment weight size (MB)")
    ax.set_ylabel("Y-PSNR (dB)")
    ax.set_title(f"Quality vs Model Size — {dataset}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"quality_vs_size_{dataset}.png", dpi=150)
    plt.close()


def _fmt(v: float, width: int = 7, dec: int = 3) -> str:
    return f"{v:{width}.{dec}f}" if not math.isnan(v) else f"{'N/A':>{width}}"


def print_table(results: List[ModelResult]) -> None:
    hdr = (
        f"  {'Model':<30} {'Bits':>5} {'PSNR':>7} {'Y-PSNR':>8} {'ΔPSNR':>7} {'ΔY-PSNR':>9} "
        f"{'ActBPP':>7} {'LikBPP':>7} {'SizeMB':>7} "
        f"{'Enc ms':>8} {'Dec ms':>8} {'BS KB':>7} "
        f"{'NN ms':>7} {'VRAM':>6} {'GMACs':>7} {'.lic':>4}"
    )
    print("\n" + hdr)
    print("  " + "─" * (len(hdr) - 2))
    for r in results:
        ok = "✓" if r.loaded else "✗"
        print(
            f"  {ok} {r.name:<29} {r.bits:>5} "
            f"{_fmt(r.psnr)} {_fmt(r.y_psnr, 8)} "
            f"{_fmt(r.delta_psnr)} {_fmt(r.delta_y_psnr, 9)} "
            f"{_fmt(r.actual_bpp, 7, 4)} {_fmt(r.likelihood_bpp, 7, 4)} {r.deploy_size_mb:>7.0f} "
            f"{_fmt(r.enc_time_mean_ms, 8, 1)} {_fmt(r.dec_time_mean_ms, 8, 1)} "
            f"{_fmt(r.bitstream_kb, 7, 1)} "
            f"{_fmt(r.latency_mean_ms, 7, 1)} {_fmt(r.vram_mb, 6, 0)} "
            f"{_fmt(r.gmacs, 7, 2)} {r.lic_roundtrip:>4}"
        )
    print()


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\nDevice:  {device}")
    if device.type == "cuda":
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"VRAM:    {total_vram:.0f} MB")
    print(f"PyTorch: {torch.__version__}\n")

    # Select models
    if args.models == "all":
        entries = MODEL_REGISTRY
    else:
        keys = {k.strip() for k in args.models.split(",")}
        entries = [e for e in MODEL_REGISTRY if e.key in keys]
        if not entries:
            valid = ", ".join(e.key for e in MODEL_REGISTRY)
            raise ValueError(f"No matching keys. Valid keys: {valid}")

    datasets = [d.strip() for d in args.datasets.split(",")]

    # Output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out_dir}\n")

    image_metric = ImageMetric().to(device)
    # Cache FLOPs (same network architecture → same MACs for all variants)
    _flops_cache: Dict[str, Tuple[float, float]] = {}

    for dataset_name in datasets:
        print(f"\n{'=' * 70}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'=' * 70}")

        try:
            loader = get_loader(dataset_name, num_workers=args.num_workers)
        except FileNotFoundError as exc:
            print(f"  [skip] {exc}")
            continue

        results: List[ModelResult] = []

        for entry in entries:
            print(f"\n▶  {entry.name}")
            print(f"   checkpoint: {entry.checkpoint.name}")

            ckpt_mb = (
                entry.checkpoint.stat().st_size / 1024**2
                if entry.checkpoint.exists() else 0.0
            )
            result = ModelResult(
                key=entry.key, name=entry.name, bits=entry.bits,
                method=entry.method, deploy_size_mb=entry.deploy_size_mb,
                checkpoint_size_mb=ckpt_mb,
            )

            model, err = load_model(entry, device)
            if model is None:
                result.error = err
                print(f"   [FAILED] {err}")
                results.append(result)
                continue

            result.loaded = True

            # Codec timing — full encode + decode including arithmetic range coding
            lic_label = f"λ={entry.lic_lambda}, {entry.lic_quant}" if entry.lic_quant else "skipped"
            print(f"   Measuring encode/decode time (full codec pipeline, .lic round-trip: {lic_label}) …")
            try:
                ct = measure_codec_time(
                    model, loader, device, entry, n_images=args.n_codec_images
                )
                if ct:
                    result.enc_time_mean_ms = ct["enc_mean_ms"]
                    result.enc_time_std_ms  = ct["enc_std_ms"]
                    result.dec_time_mean_ms = ct["dec_mean_ms"]
                    result.dec_time_std_ms  = ct["dec_std_ms"]
                    result.bitstream_kb     = ct["bitstream_kb"]
                    result.actual_bpp       = ct["actual_bpp"]
                    result.lic_roundtrip    = ct.get("lic_roundtrip", "—")
                    print(
                        f"   → enc {result.enc_time_mean_ms:.1f} ± {result.enc_time_std_ms:.1f} ms  |"
                        f"  dec {result.dec_time_mean_ms:.1f} ± {result.dec_time_std_ms:.1f} ms  |"
                        f"  bitstream {result.bitstream_kb:.1f} KB  |"
                        f"  .lic {result.lic_roundtrip}"
                    )
            except Exception as exc:
                print(f"   [warn] Codec timing failed: {exc}")

            # FLOPs — compute once (all variants share the same architecture)
            if "any" not in _flops_cache:
                print("   Computing FLOPs (Conv2d + Linear, neural-net only) …")
                try:
                    g, k = estimate_gmacs(model.cpu(), h=512, w=768)
                    _flops_cache["any"] = (g, k)
                    print(f"   → {g:.2f} GMACs  ({k:.2f} kMAC/pixel)")
                    model = model.to(device)
                except Exception as exc:
                    print(f"   [warn] FLOPs failed: {exc}")
                    _flops_cache["any"] = (float("nan"), float("nan"))
                    model = model.to(device)
            result.gmacs, result.kmac_per_pixel = _flops_cache["any"]

            # Inference timing + VRAM (GPU only)
            if device.type == "cuda":
                print("   Measuring latency + VRAM …")
                try:
                    lat_m, lat_s, vram = measure_inference(
                        model, loader, device,
                        n_warmup=3, n_measure=args.n_measure,
                    )
                    result.latency_mean_ms = lat_m
                    result.latency_std_ms  = lat_s
                    result.vram_mb         = vram
                    print(
                        f"   → latency {lat_m:.1f} ± {lat_s:.1f} ms/image"
                        f"   VRAM {vram:.0f} MB"
                    )
                except Exception as exc:
                    print(f"   [warn] Timing failed: {exc}")
            else:
                print("   [info] No GPU — latency/VRAM skipped")

            # Quality metrics
            print("   Evaluating quality …")
            try:
                q = evaluate_quality(model, loader, device, image_metric)
                result.psnr           = q["psnr"]
                result.y_psnr         = q["y_psnr"]
                result.ssim           = q["ssim"]
                result.ms_ssim        = q["ms_ssim"]
                result.lpips          = q["lpips"]
                result.likelihood_bpp = q["likelihood_bpp"]
                print(
                    f"   → PSNR {result.psnr:.3f} dB  Y-PSNR {result.y_psnr:.3f} dB  "
                    f"SSIM {result.ssim:.4f}  "
                    f"likelihood BPP {result.likelihood_bpp:.4f}  actual BPP {result.actual_bpp:.4f}"
                )
            except Exception as exc:
                print(f"   [warn] Quality eval failed: {exc}")

            results.append(result)

            # Free GPU memory before next model
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Fill ΔPSNR vs FP32
        fp32_psnr = next(
            (r.psnr for r in results if r.key == "fp32" and not math.isnan(r.psnr)),
            float("nan"),
        )
        fp32_y_psnr = next(
            (r.y_psnr for r in results if r.key == "fp32" and not math.isnan(r.y_psnr)),
            float("nan"),
        )
        for r in results:
            if not math.isnan(r.psnr) and not math.isnan(fp32_psnr):
                r.delta_psnr = r.psnr - fp32_psnr
            if not math.isnan(r.y_psnr) and not math.isnan(fp32_y_psnr):
                r.delta_y_psnr = r.y_psnr - fp32_y_psnr

        # Save outputs
        print_table(results)
        csv_path  = out_dir / f"evaluation_{dataset_name}.csv"
        json_path = out_dir / f"evaluation_{dataset_name}.json"
        save_csv(results, csv_path)
        save_json(results, json_path)
        plot_rd_curves(results, dataset_name, out_dir)

        print(f"  Saved → {csv_path.name}")
        print(f"  Saved → {json_path.name}")
        print(f"  Plots → rd_psnr_bpp_{dataset_name}.png  |  quality_vs_size_{dataset_name}.png")

    print(f"\n✓ Done.  All results in: {out_dir}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _valid_keys() -> str:
    return ", ".join(e.key for e in MODEL_REGISTRY)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoC-LIC full evaluation harness — all 9 quantization variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available model keys: {_valid_keys()}",
    )
    p.add_argument(
        "--models", default="all",
        help="'all' or comma-separated model keys, e.g. fp32,mixed_p3,kd_p4",
    )
    p.add_argument(
        "--datasets", default="kodak,pro_2020",
        help="Comma-separated dataset names: kodak, pro_2020",
    )
    p.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "results" / "eval_harness"),
        help="Root directory for outputs (a timestamped sub-folder is created)",
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--n_measure", type=int, default=15,
        help="Number of timed NN-forward passes per model (after warmup)",
    )
    p.add_argument(
        "--n_codec_images", type=int, default=10,
        help="Number of images to run through full compress/decompress for timing",
    )
    p.add_argument(
        "--device", default=None,
        help="'cuda' or 'cpu' — auto-detected if not set",
    )
    return p.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
