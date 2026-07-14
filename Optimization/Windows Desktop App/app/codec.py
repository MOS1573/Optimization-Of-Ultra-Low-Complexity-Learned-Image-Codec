"""
LoC-LIC Codec — model loading and compress/decompress for the desktop app.

Uses PyTorch + CompressAI for all inference.  The ONNX backend (Phase 2a)
can be plugged in later by replacing the _load_model internals while keeping
the LocLicCodec API identical.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    from torchmetrics.functional.image import (
        multiscale_structural_similarity_index_measure as _ms_ssim_fn,
    )
    _HAS_MS_SSIM = True
except ImportError:
    _HAS_MS_SSIM = False
from torchvision import transforms as T

# ── Repo path setup ───────────────────────────────────────────────────────────
# Dev:    set LOCLIC_REPO_ROOT or rely on this file's location (3 levels up).
# Bundle: PyInstaller includes the repo modules via the spec's datas/pathex.
REPO_ROOT = Path(os.environ.get("LOCLIC_REPO_ROOT", Path(__file__).parents[3]))

for _p in [
    str(REPO_ROOT),
    str(REPO_ROOT / "Optimization" / "4-bit" / "4-bit Mixed-Precision"),
    str(REPO_ROOT / "Optimization" / "Lic Container Format"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.get_model import get_model               # noqa: E402
from utils.config import model_config                # noqa: E402
from lic_format import write_lic, read_lic, lic_info # noqa: E402

_CFG_PATH = REPO_ROOT / "configs" / "models" / "loclic.yaml"


# ── Model registry ────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    key:            str
    name:           str     # shown in UI dropdowns
    checkpoint:     Path
    loader:         str     # 'fp32' | 'gptq_ptq' | 'mixed_qat'
    lambda_val:     float
    lic_quant:      str     # quant type written into .lic header
    deploy_size_mb: float   # theoretical deployment weight footprint


MODEL_CONFIGS: List[ModelConfig] = [
    # ── FP32 baselines ──────────────────────────────────────────────────────
    ModelConfig("fp32_0018", "FP32  λ=0.0018",
                REPO_ROOT / "checkpoints/loc-lic/0.0018.pth.tar",
                "fp32", 0.0018, "fp32", 268.0),
    ModelConfig("fp32_0035", "FP32  λ=0.0035",
                REPO_ROOT / "checkpoints/loc-lic/0.0035.pth.tar",
                "fp32", 0.0035, "fp32", 268.0),
    ModelConfig("fp32_0067", "FP32  λ=0.0067",
                REPO_ROOT / "checkpoints/loc-lic/0.0067.pth.tar",
                "fp32", 0.0067, "fp32", 268.0),
    ModelConfig("fp32_025",  "FP32  λ=0.025",
                REPO_ROOT / "checkpoints/loc-lic/0.025.pth.tar",
                "fp32", 0.025,  "fp32", 268.0),
    # ── INT8 GPTQ (4× weight compression) ───────────────────────────────────
    # λ=0.0018 uses the original GPTQ checkpoint — per-lambda re-quantisation
    # was only run for λ=0.0035/0.0067/0.025; no gptq-per-lambda/lambda-0018 exists.
    ModelConfig("gptq_0018", "INT8 GPTQ  λ=0.0018",
                REPO_ROOT / "results/legacy/old_kodak_test/gptq_second_try/LoC_LIC_int8_gptq_8bit.pt",
                "gptq_ptq", 0.0018, "int8_gptq", 67.0),
    ModelConfig("gptq_0035", "INT8 GPTQ  λ=0.0035",
                REPO_ROOT / "results/8-bit/gptq-per-lambda/lambda-0035/LoC_LIC_int8_gptq_8bit.pt",
                "gptq_ptq", 0.0035, "int8_gptq", 67.0),
    ModelConfig("gptq_0067", "INT8 GPTQ  λ=0.0067",
                REPO_ROOT / "results/8-bit/gptq-per-lambda/lambda-0067/LoC_LIC_int8_gptq_8bit.pt",
                "gptq_ptq", 0.0067, "int8_gptq", 67.0),
    ModelConfig("gptq_025",  "INT8 GPTQ  λ=0.025",
                REPO_ROOT / "results/8-bit/gptq-per-lambda/lambda-025/LoC_LIC_int8_gptq_8bit.pt",
                "gptq_ptq", 0.025,  "int8_gptq", 67.0),
    # ── Mixed QAT Phase 3 ★ (5.5× weight compression, all λ) ────────────────
    ModelConfig("qat_p3_0018", "Mixed QAT ★  λ=0.0018",
                REPO_ROOT / "results/4-bit/phase3-mixed-finetuned/lambda-0018/LoC_LIC_mixed_qat.pt",
                "mixed_qat", 0.0018, "qat_best", 48.0),
    ModelConfig("qat_p3_0035", "Mixed QAT ★  λ=0.0035",
                REPO_ROOT / "results/4-bit/phase3-mixed-finetuned/lambda-0035/LoC_LIC_mixed_qat.pt",
                "mixed_qat", 0.0035, "qat_best", 48.0),
    ModelConfig("qat_p3_0067", "Mixed QAT ★  λ=0.0067",
                REPO_ROOT / "results/4-bit/phase3-mixed-finetuned/lambda-0067/LoC_LIC_mixed_qat.pt",
                "mixed_qat", 0.0067, "qat_best", 48.0),
    ModelConfig("qat_p3_025",  "Mixed QAT ★  λ=0.025",
                REPO_ROOT / "results/4-bit/phase3-mixed-finetuned/lambda-025/LoC_LIC_mixed_qat.pt",
                "mixed_qat", 0.025,  "qat_best", 48.0),
]


# ── Internal model loading (mirrors evaluate_all.py) ─────────────────────────

def _build_base_model(device: torch.device) -> nn.Module:
    cfg  = model_config(str(_CFG_PATH))
    args = argparse.Namespace(lmbda=0.0018)
    net, _, _ = get_model(cfg, args, device)
    return net.to(device)


def _load_weights(model: nn.Module, state: dict) -> None:
    # Must call through the model instance (CompressionModel.load_state_dict),
    # not nn.Module.load_state_dict directly.  CompressAI's override calls
    # remap_old_keys() which maps old checkpoint names (_matrix0, _bias1,
    # _factor2) to the current names (matrices.0, biases.1, factors.2).
    # Bypassing it leaves the density estimator at random initialisation on
    # FP32 checkpoints, which inflates BPP by ~2× at decode time.
    model.load_state_dict(state, strict=False)


def _load_model(cfg: ModelConfig, device: torch.device) -> nn.Module:
    raw = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    state = raw.get("state_dict", raw.get("model", raw)) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        return state.eval().to(device)
    model = _build_base_model(device)
    _load_weights(model, state)
    return model.eval()


# ── Public codec class ────────────────────────────────────────────────────────

class LocLicCodec:
    """
    Wraps one LoC-LIC model for compress / decompress.

    Typical usage:
        codec = LocLicCodec(cfg)
        codec.load()                          # loads weights + builds CDF tables
        result = codec.compress_image(path)   # returns dict with x_hat + metrics
        codec.save_lic(result, "out.lic")     # persists bitstream
        codec.unload()                        # frees GPU memory
    """

    def __init__(self, cfg: ModelConfig, device: Optional[torch.device] = None):
        self.cfg    = cfg
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._model: Optional[nn.Module] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model weights and build CDF tables for the range coder."""
        self._model = _load_model(self.cfg, self.device)
        self._model.update(force=True)

    def unload(self) -> None:
        self._model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # ── Compress ──────────────────────────────────────────────────────────────

    def compress_image(self, image_path: str) -> Dict[str, Any]:
        """
        Compress a PNG/JPEG/BMP image file.

        Returns:
            x_hat:        torch.Tensor (3, H, W) float32 — reconstructed image
            compress_out: dict from model.compress() — pass to save_lic()
            img_shape:    (H, W) original size before padding
            psnr:         float (dB)
            bpp:          float (bits per pixel, from bitstream bytes)
            file_kb:      float (compressed size in KB)
            enc_ms:       float (encode time in ms, neural net + range coder)
            dec_ms:       float (decode time in ms)
        """
        if not self.loaded:
            self.load()

        img              = Image.open(image_path).convert("RGB")
        W_orig, H_orig   = img.size   # PIL returns (W, H)
        x                = T.ToTensor()(img).unsqueeze(0).to(self.device)

        x_pad, (H_orig, W_orig) = _pad64(x)

        with torch.no_grad():
            t0           = time.perf_counter()
            compress_out = self._model.compress(x_pad)
            enc_ms       = (time.perf_counter() - t0) * 1000.0

            t0           = time.perf_counter()
            dec_out      = self._model.decompress(
                compress_out["strings"], compress_out["shape"]
            )
            dec_ms       = (time.perf_counter() - t0) * 1000.0

        x_hat  = dec_out["x_hat"].clamp(0, 1).squeeze(0)[:, :H_orig, :W_orig]
        x_orig = x.squeeze(0)[:, :H_orig, :W_orig].cpu()

        total_bytes = _count_bytes(compress_out["strings"])
        bpp         = (total_bytes * 8) / (H_orig * W_orig)

        x_hat_cpu = x_hat.cpu()
        return {
            "x_hat":        x_hat_cpu,
            "compress_out": compress_out,
            "img_shape":    (H_orig, W_orig),
            "psnr":         _psnr(x_orig, x_hat_cpu),
            "ms_ssim":      _ms_ssim(x_orig, x_hat_cpu),
            "bpp":          bpp,
            "file_kb":      total_bytes / 1024.0,
            "enc_ms":       enc_ms,
            "dec_ms":       dec_ms,
        }

    # ── Decompress ────────────────────────────────────────────────────────────

    def decompress_lic(self, lic_path: str) -> Dict[str, Any]:
        """
        Decompress a .lic file using this codec's model.

        Returns same dict structure as compress_image.
        psnr and enc_ms are NaN (no original image to compare against).
        """
        if not self.loaded:
            self.load()

        compress_out, lambda_val, quant_type, (H_orig, W_orig) = read_lic(lic_path)

        with torch.no_grad():
            t0      = time.perf_counter()
            dec_out = self._model.decompress(
                compress_out["strings"], compress_out["shape"]
            )
            dec_ms  = (time.perf_counter() - t0) * 1000.0

        x_hat       = dec_out["x_hat"].clamp(0, 1).squeeze(0)[:, :H_orig, :W_orig]
        total_bytes = _count_bytes(compress_out["strings"])
        bpp         = (total_bytes * 8) / (H_orig * W_orig)

        return {
            "x_hat":        x_hat.cpu(),
            "compress_out": compress_out,
            "img_shape":    (H_orig, W_orig),
            "psnr":         float("nan"),
            "ms_ssim":      float("nan"),
            "bpp":          bpp,
            "file_kb":      total_bytes / 1024.0,
            "enc_ms":       float("nan"),
            "dec_ms":       dec_ms,
            "lic_lambda":   lambda_val,
            "lic_quant":    quant_type,
        }

    # ── Save ─────────────────────────────────────────────────────────────────

    def save_lic(self, result: Dict[str, Any], lic_path: str) -> None:
        """Write a compress result to a .lic file."""
        write_lic(
            lic_path,
            result["compress_out"],
            lambda_val=self.cfg.lambda_val,
            quant_type=self.cfg.lic_quant,
            img_shape=result["img_shape"],
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pad64(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad spatial dims to next multiple of 64. Returns (padded, (orig_H, orig_W))."""
    _, _, H, W = x.shape
    pH = (64 - H % 64) % 64
    pW = (64 - W % 64) % 64
    if pH or pW:
        x = torch.nn.functional.pad(x, (0, pW, 0, pH), mode="reflect")
    return x, (H, W)


def _count_bytes(strings) -> int:
    total = 0
    for group in strings:
        items = group if isinstance(group, (list, tuple)) else [group]
        total += sum(len(s) for s in items if isinstance(s, (bytes, bytearray)))
    return total


def _psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    mse = torch.mean((original.float() - reconstructed.float()) ** 2).item()
    return float("inf") if mse == 0 else -10.0 * np.log10(mse)


def _ms_ssim(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    if not _HAS_MS_SSIM:
        return float("nan")
    try:
        o = original.unsqueeze(0).clamp(0, 1).cpu()
        r = reconstructed.unsqueeze(0).clamp(0, 1).cpu()
        return float(_ms_ssim_fn(r, o, data_range=1.0))
    except Exception:
        return float("nan")


def find_cfg(lambda_val: float, loader: str) -> Optional[ModelConfig]:
    """Look up a ModelConfig by lambda and loader type."""
    for cfg in MODEL_CONFIGS:
        if cfg.lambda_val == lambda_val and cfg.loader == loader:
            return cfg
    return None
