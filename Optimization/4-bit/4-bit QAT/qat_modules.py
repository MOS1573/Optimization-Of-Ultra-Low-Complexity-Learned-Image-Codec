"""
QAT building blocks for LoC-LIC 4-bit Quantization-Aware Training.

Core idea: wrap each Conv2d / ConvTranspose2d / Linear weight with a fake-
quantization operation that:
  - In the FORWARD pass: applies symmetric per-output-channel INT4 rounding.
  - In the BACKWARD pass: passes gradients straight through (STE).

This is done via `torch.nn.utils.parametrize`, which injects the transform as
a registered forward hook so that `module.weight` always returns the fake-
quantized version during both training forward and loss.backward, while the
actual stored parameter (`module.parametrizations.weight.original`) is the
full-precision weight that the optimizer updates.

Entropy model layers (EntropyBottleneck, GaussianConditional, GDN) are always
kept at FP32 — they manage CDF tables that must not be perturbed.
The first and last Conv2d in the network are also kept at FP32 by default
(highest sensitivity, same policy as GPTQ).
"""

from __future__ import annotations

import logging
from typing import Set, Tuple

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
# STE fake-quantization function
# ─────────────────────────────────────────────────────────────────────────────

def fake_quant_ste(W: torch.Tensor, maxv: float) -> torch.Tensor:
    """
    Symmetric per-output-channel fake quantization with Straight-Through Estimator.

    Forward:  W_q = round(clamp(W / scale, -maxv, maxv)) * scale
    Backward: dL/dW = dL/dW_q  (identity — the STE approximation)

    The scale is derived from the current weight values but detached so that
    the gradient flows cleanly through the identity path.

    Args:
        W:    weight tensor, first dim = output channels.
        maxv: 7.0 for INT4, 127.0 for INT8.

    Returns:
        Fake-quantized weight (same shape, same dtype, gradient-enabled).
    """
    if W.dim() > 1:
        with torch.no_grad():
            scale = W.reshape(W.shape[0], -1).abs().max(dim=1)[0].clamp(min=1e-8) / maxv
        s = scale.view(-1, *([1] * (W.dim() - 1)))
    else:
        with torch.no_grad():
            scale = W.abs().max().clamp(min=1e-8) / maxv
        s = scale

    W_norm = W / s
    W_clamped = W_norm.clamp(-maxv, maxv)
    # STE: forward = rounded, backward = identity gradient
    W_q_norm = W_clamped + (W_clamped.round() - W_clamped).detach()
    return W_q_norm * s


# ─────────────────────────────────────────────────────────────────────────────
# Parametrize-compatible FakeQuantize module
# ─────────────────────────────────────────────────────────────────────────────

class FakeQuantizeNBit(nn.Module):
    """
    Parametrize wrapper: registers as a weight transform on nn.Module via
    `torch.nn.utils.parametrize.register_parametrization`.

    Usage (internal — called by register_qat_parametrizations):
        parametrize.register_parametrization(layer, 'weight', FakeQuantizeNBit(bits=4))

    After registration:
        layer.weight           → fake-quantized (used in forward)
        layer.parametrizations.weight.original  → full-precision (updated by optimizer)
    """

    def __init__(self, bits: int = 4):
        super().__init__()
        self.bits = bits
        self.maxv = float(2 ** (bits - 1) - 1)  # 7 for INT4

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return fake_quant_ste(W, self.maxv)

    def right_inverse(self, W: torch.Tensor) -> torch.Tensor:
        # Inverse transform for initialization: identity (no change to stored weights)
        return W


# ─────────────────────────────────────────────────────────────────────────────
# Layer selection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _should_skip(name: str, module: nn.Module) -> bool:
    """Return True if this module must stay at FP32."""
    if _ENTROPY_TYPES and isinstance(module, _ENTROPY_TYPES):
        return True
    if _GDN_TYPES and isinstance(module, _GDN_TYPES):
        return True
    name_lower = name.lower()
    return any(pat in name_lower for pat in _SKIP_PATTERNS)


def _collect_conv_names(model: nn.Module) -> list:
    """Return ordered list of Conv2d / ConvTranspose2d names, skipping entropy layers."""
    names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and not _should_skip(name, module):
            names.append(name)
    return names


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def register_qat_parametrizations(
    model: nn.Module,
    bits: int = 4,
    skip_first_last_conv: bool = True,
) -> Tuple[int, int]:
    """
    Register FakeQuantizeNBit on all eligible weight tensors in *model*.

    Returns:
        (n_quantized, n_skipped)
    """
    conv_names = _collect_conv_names(model)
    first_conv = conv_names[0] if conv_names else None
    last_conv  = conv_names[-1] if conv_names else None

    n_q = n_skip = 0
    for name, module in model.named_modules():
        if not isinstance(module, _QUANTIZABLE):
            continue
        if _should_skip(name, module):
            n_skip += 1
            continue
        if skip_first_last_conv and isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if name in (first_conv, last_conv):
                logger.info(f"  QAT skip (first/last conv): {name}")
                n_skip += 1
                continue
        parametrize.register_parametrization(module, "weight", FakeQuantizeNBit(bits=bits))
        n_q += 1

    logger.info(f"QAT parametrizations registered: {n_q} layers quantized, {n_skip} skipped.")
    return n_q, n_skip


def remove_qat_parametrizations(model: nn.Module) -> None:
    """Remove all FakeQuantize parametrizations (restores original weight behaviour)."""
    for name, module in model.named_modules():
        if parametrize.is_parametrized(module) and "weight" in getattr(module, "parametrizations", {}):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)


def extract_deployment_state_dict(model: nn.Module, bits: int = 4) -> dict:
    """
    Build a standard state dict with permanently-applied INT4 QDQ weights.
    Use this for the final model saved for evaluation / deployment.

    The returned state dict has the same keys as the original FP32 model,
    so it can be loaded via the standard LoC_LIC.load_state_dict().
    """
    maxv = float(2 ** (bits - 1) - 1)
    sd = {}
    for name, module in model.named_modules():
        # Use try/except to detect QAT parametrizations robustly.
        # Direct attribute access avoids relying on is_parametrized() which
        # checks isinstance(parametrizations, ModuleDict) — that check can fail
        # after DDP unwrap if the class reference changes across processes.
        try:
            W_orig = module.parametrizations.weight.original.data
            with torch.no_grad():
                W_q = fake_quant_ste(W_orig.float(), maxv).to(W_orig.dtype)
            sd[name + ".weight"] = W_q
        except AttributeError:
            w = getattr(module, "weight", None)
            if isinstance(w, torch.Tensor) and w is not None:
                sd[name + ".weight"] = w.data

    # Include all other parameters and buffers
    full_sd = model.state_dict()
    for key, val in full_sd.items():
        # Skip parametrize internal keys — already handled above
        if "parametrizations" in key:
            # Extract base name and replace
            base = key.replace(".parametrizations.weight.original", ".weight")
            if base not in sd:
                sd[base] = val
        else:
            if key not in sd:
                sd[key] = val

    return sd
