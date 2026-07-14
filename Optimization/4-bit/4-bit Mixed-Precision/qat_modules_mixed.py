"""
Mixed-precision QAT building blocks for LoC-LIC.

Extension of qat_modules.py that supports per-layer bit-width assignment.
The core fake-quantization and STE logic is identical; the only addition is
that `register_mixed_qat_parametrizations()` accepts a `layer_bits` dict
mapping layer names to their target bit-width (4, 8, or 32).

  bits=4  → INT4 fake-quant (same as the 4-bit QAT baseline)
  bits=8  → INT8 fake-quant (for entropy-sensitive layers)
  bits=32 → skip (keep FP32) — used for first/last conv or manually forced layers

The `extract_deployment_state_dict_mixed()` function reads the bit-width from
each layer's registered FakeQuantizeNBit hook, so the deployment state dict
always uses the correct quantization per layer.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

try:
    from compressai.entropy_models import EntropyBottleneck, GaussianConditional
    _ENTROPY_TYPES: Tuple = (EntropyBottleneck, GaussianConditional)
except ImportError:
    _ENTROPY_TYPES = ()

try:
    from compressai.layers import GDN
    _GDN_TYPES: Tuple = (GDN,)
except ImportError:
    _GDN_TYPES = ()

logger = logging.getLogger(__name__)

_QUANTIZABLE = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)

_SKIP_PATTERNS: Tuple[str, ...] = (
    "entropy_bottleneck",
    "gaussian_conditional",
    "quantizer",
    "_quantiles",
)


# ─────────────────────────────────────────────────────────────────────────────
# STE fake-quantization (same as qat_modules.py — duplicated to keep this
# file self-contained and not depend on the 4-bit QAT directory)
# ─────────────────────────────────────────────────────────────────────────────

def fake_quant_ste(W: torch.Tensor, maxv: float) -> torch.Tensor:
    """Symmetric per-output-channel fake quantization with STE."""
    if W.dim() > 1:
        with torch.no_grad():
            scale = W.reshape(W.shape[0], -1).abs().max(dim=1)[0].clamp(min=1e-8) / maxv
        s = scale.view(-1, *([1] * (W.dim() - 1)))
    else:
        with torch.no_grad():
            scale = W.abs().max().clamp(min=1e-8) / maxv
        s = scale

    W_norm    = W / s
    W_clamped = W_norm.clamp(-maxv, maxv)
    W_q_norm  = W_clamped + (W_clamped.round() - W_clamped).detach()
    return W_q_norm * s


# ─────────────────────────────────────────────────────────────────────────────
# Parametrize-compatible FakeQuantize module
# ─────────────────────────────────────────────────────────────────────────────

class FakeQuantizeNBit(nn.Module):
    def __init__(self, bits: int = 4):
        super().__init__()
        self.bits = bits
        self.maxv = float(2 ** (bits - 1) - 1)

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return fake_quant_ste(W, self.maxv)

    def right_inverse(self, W: torch.Tensor) -> torch.Tensor:
        return W


# ─────────────────────────────────────────────────────────────────────────────
# Layer selection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _should_skip(name: str, module: nn.Module) -> bool:
    if _ENTROPY_TYPES and isinstance(module, _ENTROPY_TYPES):
        return True
    if _GDN_TYPES and isinstance(module, _GDN_TYPES):
        return True
    return any(pat in name.lower() for pat in _SKIP_PATTERNS)


def _collect_conv_names(model: nn.Module) -> list:
    names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and not _should_skip(name, module):
            names.append(name)
    return names


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def register_mixed_qat_parametrizations(
    model: nn.Module,
    layer_bits: Dict[str, int],
    default_bits: int = 4,
    skip_first_last_conv: bool = True,
) -> Tuple[int, int, int]:
    """
    Register per-layer fake-quantization parametrizations.

    Args:
        model:               The model to quantize.
        layer_bits:          Dict mapping layer name → bit-width. Layers not in
                             this dict get `default_bits`. A value of 32 means
                             skip (keep FP32).
        default_bits:        Bit-width for layers absent from `layer_bits`.
        skip_first_last_conv: Always keep first and last conv at FP32.

    Returns:
        (n_int4, n_int8, n_fp32_skipped)
    """
    conv_names = _collect_conv_names(model)
    first_conv = conv_names[0] if conv_names else None
    last_conv  = conv_names[-1] if conv_names else None

    n_int4 = n_int8 = n_skip = 0

    for name, module in model.named_modules():
        if not isinstance(module, _QUANTIZABLE):
            continue
        if _should_skip(name, module):
            n_skip += 1
            continue
        if skip_first_last_conv and isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if name in (first_conv, last_conv):
                logger.info(f"  Mixed-QAT skip (first/last conv): {name}")
                n_skip += 1
                continue

        bits = layer_bits.get(name, default_bits)

        if bits == 32:
            logger.info(f"  Mixed-QAT skip (FP32 forced): {name}")
            n_skip += 1
            continue

        parametrize.register_parametrization(module, "weight", FakeQuantizeNBit(bits=bits))

        if bits == 8:
            n_int8 += 1
        else:
            n_int4 += 1

    logger.info(
        f"Mixed-QAT parametrizations: {n_int4} INT4 layers, "
        f"{n_int8} INT8 layers, {n_skip} FP32 (skipped)."
    )
    return n_int4, n_int8, n_skip


def remove_qat_parametrizations(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if parametrize.is_parametrized(module) and "weight" in getattr(module, "parametrizations", {}):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)


def extract_deployment_state_dict_mixed(model: nn.Module) -> dict:
    """
    Build a deployment state dict with permanently-applied quantized weights.
    Reads each layer's actual bit-width from its FakeQuantizeNBit hook, so
    mixed INT4/INT8 layers are handled correctly.
    """
    sd = {}
    for name, module in model.named_modules():
        try:
            hook = module.parametrizations.weight[0]
            bits = hook.bits
            maxv = float(2 ** (bits - 1) - 1)
            W_orig = module.parametrizations.weight.original.data
            with torch.no_grad():
                W_q = fake_quant_ste(W_orig.float(), maxv).to(W_orig.dtype)
            sd[name + ".weight"] = W_q
        except AttributeError:
            w = getattr(module, "weight", None)
            if isinstance(w, torch.Tensor) and w is not None:
                sd[name + ".weight"] = w.data

    full_sd = model.state_dict()
    for key, val in full_sd.items():
        if "parametrizations" in key:
            base = key.replace(".parametrizations.weight.original", ".weight")
            if base not in sd:
                sd[base] = val
        else:
            if key not in sd:
                sd[key] = val

    return sd
