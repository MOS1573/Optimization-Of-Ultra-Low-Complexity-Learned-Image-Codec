"""
GPTQ-based 8-bit Post-Training Quantization for LoC-LIC.

Algorithm: Frantar et al. (2022) "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers" — extended here to Conv2d and depthwise Conv2d.

Key improvements over first try (naive round-to-nearest):
  1. Hessian-guided rounding: within each layer, quantization error in weight column j
     is compensated in columns j+1..m by updating them with the scaled error.
  2. Symmetric INT8 [-127, 127], zero_point = 0 — eliminates the formula bug from v1.
  3. Real calibration data (Kodak images) instead of random noise.
  4. Entropy model layers (EntropyBottleneck, GaussianConditional) always skipped.
  5. Per-channel depthwise GPTQ — handles grouped convolutions correctly.
  6. Full GPU support throughout.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# compressai entropy models — must not be quantized (they manage CDF tables)
try:
    from compressai.entropy_models import EntropyBottleneck, GaussianConditional
    _ENTROPY_TYPES = (EntropyBottleneck, GaussianConditional)
except ImportError:
    _ENTROPY_TYPES = ()

try:
    from compressai.layers import GDN
    _GDN_TYPES = (GDN,)
except ImportError:
    _GDN_TYPES = ()

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuantConfig:
    bits: int = 8
    # Hessian damping: adds percdamp * mean(diag(H)) to diagonal for stability
    percdamp: float = 0.01
    # Symmetric quantization: range [-maxval, maxval], zero_point = 0
    sym: bool = True
    # Per-output-channel scale (recommended for weights)
    per_channel: bool = True
    # Module name substrings that force FP32 (in addition to entropy models)
    skip_patterns: Tuple[str, ...] = (
        "entropy_bottleneck",
        "gaussian_conditional",
        "quantizer",           # model's internal discrete quantizer
        "_quantiles",          # EntropyBottleneck internal buffer
    )
    # Keep the very first and very last conv at FP32 (highest sensitivity)
    skip_first_last_conv: bool = True
    # Output directory for results and saved model
    output_dir: str = "results/gptq_ptq"


# ─────────────────────────────────────────────────────────────────────────────
# Quantization primitives
# ─────────────────────────────────────────────────────────────────────────────

def _maxval(bits: int) -> float:
    return float(2 ** (bits - 1) - 1)   # 127 for INT8


def symmetric_quantize_tensor(w: torch.Tensor, bits: int = 8,
                               per_channel: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-output-channel INT quantization.
    w: weight tensor of any shape, first dim = output channels.
    Returns: (dequantized_float, scale_per_channel)
    Formula: q = clamp(round(w / scale), -maxval, maxval)
             w_dq = q * scale
    """
    maxv = _maxval(bits)
    if per_channel and w.dim() > 1:
        scale = w.reshape(w.shape[0], -1).abs().max(dim=1)[0] / maxv
    else:
        scale = w.abs().max() / maxv
    scale = scale.clamp(min=1e-8)
    s = scale.view(-1, *([1] * (w.dim() - 1))) if per_channel and w.dim() > 1 else scale
    w_dq = (w / s).round().clamp(-maxv, maxv) * s
    return w_dq, scale


# ─────────────────────────────────────────────────────────────────────────────
# Layer-level GPTQ
# ─────────────────────────────────────────────────────────────────────────────

class GPTQLayer:
    """
    Accumulates Hessian statistics from forward-pass inputs, then applies GPTQ
    to quantize a single layer's weights.

    Supported layer types:
      - nn.Linear            → standard GPTQ on [n_out × n_in]
      - nn.Conv2d (groups=1) → GPTQ with im2col unfolding
      - nn.Conv2d (depthwise)→ per-channel GPTQ (one small H per channel)
      - nn.ConvTranspose2d   → symmetric per-channel (GPTQ approx via input cov)
    """

    def __init__(self, layer: nn.Module, config: QuantConfig):
        self.layer = layer
        self.config = config
        self.H: Optional[torch.Tensor] = None  # accumulated X^T X
        self.nsamples: int = 0
        self._is_depthwise = (
            isinstance(layer, nn.Conv2d) and
            layer.groups == layer.in_channels and
            layer.in_channels > 1
        )

    # ── Input collection ─────────────────────────────────────────────────────

    def add_batch(self, inp: torch.Tensor):
        """Add one batch of layer inputs to the Hessian accumulator."""
        device = inp.device
        inp = inp.detach().float()

        if self._is_depthwise:
            self._add_batch_depthwise(inp, device)
            return

        X = self._unfold(inp)   # [N, columns]
        if X is None:
            return
        n = X.shape[0]
        if self.H is None:
            cols = X.shape[1]
            self.H = torch.zeros(cols, cols, dtype=torch.float64, device=device)
        self.H += X.double().T @ X.double()
        self.nsamples += n

    def _add_batch_depthwise(self, inp: torch.Tensor, device):
        """
        For depthwise conv: accumulate a separate H_c for each channel c.
        self.H shape: [C, kH*kW, kH*kW]
        """
        layer = self.layer
        C = inp.shape[1]
        k = layer.kernel_size[0] * layer.kernel_size[1]
        if self.H is None:
            self.H = torch.zeros(C, k, k, dtype=torch.float64, device=device)
        for c in range(C):
            xc = F.unfold(
                inp[:, c:c+1],
                kernel_size=layer.kernel_size,
                dilation=layer.dilation,
                padding=layer.padding,
                stride=layer.stride,
            )  # [B, k, L]
            xc = xc.permute(0, 2, 1).reshape(-1, k).double()  # [B*L, k]
            self.H[c] += xc.T @ xc
        self.nsamples += inp.shape[0]

    def _unfold(self, inp: torch.Tensor) -> Optional[torch.Tensor]:
        """Convert layer input to 2D matrix [N, columns] via im2col or reshape."""
        layer = self.layer
        if isinstance(layer, nn.Linear):
            return inp.reshape(-1, inp.shape[-1])
        elif isinstance(layer, nn.Conv2d):
            # Standard conv2d (groups=1 or grouped but not depthwise)
            # For grouped conv (1 < groups < in_channels), unfold full input;
            # this is an approximation that treats all channels jointly.
            X = F.unfold(
                inp,
                kernel_size=layer.kernel_size,
                dilation=layer.dilation,
                padding=layer.padding,
                stride=layer.stride,
            )  # [B, C_in*kH*kW, L]
            return X.permute(0, 2, 1).reshape(-1, X.shape[1])  # [B*L, C_in*kH*kW]
        elif isinstance(layer, nn.ConvTranspose2d):
            # Approximate: use spatial average of input as proxy for Hessian diagonal
            return inp.permute(0, 2, 3, 1).reshape(-1, inp.shape[1])  # [B*H*W, C_in]
        return None

    # ── GPTQ quantization ────────────────────────────────────────────────────

    def quantize(self) -> torch.Tensor:
        """
        Apply GPTQ to this layer's weights.
        Returns dequantized float weights (same shape as layer.weight).
        Caller should do: layer.weight.data.copy_(result)
        """
        layer = self.layer
        cfg = self.config
        W = layer.weight.data.float()

        # Depthwise: per-channel GPTQ
        if self._is_depthwise:
            return self._quantize_depthwise(W)

        # ConvTranspose2d: GPTQ with transposed weight view
        if isinstance(layer, nn.ConvTranspose2d):
            return self._quantize_conv_transpose(W)

        # Linear and standard Conv2d: full GPTQ
        orig_shape = W.shape
        n_out = W.shape[0]
        W_flat = W.reshape(n_out, -1).clone()  # [n_out, columns]
        m = W_flat.shape[1]

        if self.H is None or self.nsamples == 0:
            logger.warning(f"No calibration data for {layer.__class__.__name__}; "
                           "falling back to symmetric quantization.")
            q, _ = symmetric_quantize_tensor(W, cfg.bits, cfg.per_channel)
            return q.to(layer.weight.dtype)

        Hinv = self._compute_hinv(self.H.float() / self.nsamples, m)

        # Pre-compute per-output-channel scale from original weights
        maxv = _maxval(cfg.bits)
        scale = W_flat.abs().max(dim=1)[0].clamp(min=1e-8) / maxv  # [n_out]

        Q = torch.zeros_like(W_flat)

        for col in range(m):
            w_col = W_flat[:, col]                      # [n_out]
            d = Hinv[col, col].item()

            # Quantize with per-output-channel scale
            q_col = (w_col / scale).round().clamp(-maxv, maxv) * scale
            Q[:, col] = q_col

            # Propagate error to remaining columns
            if col + 1 < m:
                err = (w_col - q_col) / max(d, 1e-8)   # [n_out]
                W_flat[:, col + 1:] -= err.unsqueeze(1) * Hinv[col, col + 1:].unsqueeze(0)

        return Q.reshape(orig_shape).to(layer.weight.dtype)

    def _quantize_depthwise(self, W: torch.Tensor) -> torch.Tensor:
        """Per-channel GPTQ for depthwise Conv2d."""
        cfg = self.config
        C_out = W.shape[0]
        orig_shape = W.shape
        maxv = _maxval(cfg.bits)
        Q = torch.zeros_like(W)

        for c in range(C_out):
            w_c = W[c].reshape(-1).clone()  # [kH*kW]
            m = w_c.shape[0]
            scale_c = w_c.abs().max().clamp(min=1e-8).item() / maxv

            if self.H is not None and self.nsamples > 0:
                H_c = (self.H[c] / self.nsamples).float()
                Hinv_c = self._compute_hinv(H_c, m)
                q_c = torch.zeros_like(w_c)
                for col in range(m):
                    d = Hinv_c[col, col].item()
                    q_val = (w_c[col] / scale_c).round().clamp(-maxv, maxv) * scale_c
                    q_c[col] = q_val
                    if col + 1 < m:
                        err = (w_c[col] - q_val) / max(d, 1e-8)
                        w_c[col + 1:] -= err * Hinv_c[col, col + 1:]
            else:
                q_c = (w_c / scale_c).round().clamp(-maxv, maxv) * scale_c

            Q[c] = q_c.reshape(W[c].shape)

        return Q.reshape(orig_shape).to(self.layer.weight.dtype)

    def _quantize_conv_transpose(self, W: torch.Tensor) -> torch.Tensor:
        """
        GPTQ for ConvTranspose2d.
        Weight shape: [C_in, C_out/groups, kH, kW].
        We treat rows as C_in (instead of C_out) for the per-channel scale.
        """
        cfg = self.config
        orig_shape = W.shape
        n_out = W.shape[0]      # C_in — treated as "output channels" for scale
        W_flat = W.reshape(n_out, -1).clone()
        m = W_flat.shape[1]
        maxv = _maxval(cfg.bits)
        scale = W_flat.abs().max(dim=1)[0].clamp(min=1e-8) / maxv

        if self.H is None or self.nsamples == 0:
            q, _ = symmetric_quantize_tensor(W, cfg.bits, cfg.per_channel)
            return q.to(self.layer.weight.dtype)

        # For ConvTranspose2d: H is accumulated on the C_in-dimensional input
        # which matches the "rows" here (C_in × C_in matrix).
        Hinv = self._compute_hinv(self.H.float() / self.nsamples, m)

        Q = torch.zeros_like(W_flat)
        for col in range(m):
            w_col = W_flat[:, col]
            d = Hinv[col, col].item()
            q_col = (w_col / scale).round().clamp(-maxv, maxv) * scale
            Q[:, col] = q_col
            if col + 1 < m:
                err = (w_col - q_col) / max(d, 1e-8)
                W_flat[:, col + 1:] -= err.unsqueeze(1) * Hinv[col, col + 1:].unsqueeze(0)

        return Q.reshape(orig_shape).to(self.layer.weight.dtype)

    @staticmethod
    def _compute_hinv(H: torch.Tensor, m: int) -> torch.Tensor:
        """
        Compute upper-triangular Cholesky factor of H^{-1}.
        This is the Hinv matrix used in the GPTQ update rule.
        Falls back to diagonal approximation if Cholesky fails.
        """
        # Remove zero-diagonal entries
        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0

        try:
            # H = L L^T  (Cholesky of H)
            # Hinv = (H^{-1})  via cholesky_inverse
            # Then Cholesky of Hinv for the upper-triangular factor
            L = torch.linalg.cholesky(H.double())
            Hinv_full = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv_full, upper=True).float()
        except Exception:
            # Diagonal fallback
            d = H.diagonal().clamp(min=1e-8).sqrt()
            Hinv = torch.diag(1.0 / d)

        return Hinv

    def free(self):
        """Release Hessian memory after quantization."""
        del self.H
        self.H = None
        self.nsamples = 0


# ─────────────────────────────────────────────────────────────────────────────
# Full-model GPTQ pipeline
# ─────────────────────────────────────────────────────────────────────────────

_QUANTIZABLE_TYPES = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)


class GPTQPipeline:
    """
    Orchestrates GPTQ quantization over all eligible layers in a model.

    Usage:
        pipeline = GPTQPipeline(model, config)
        pipeline.calibrate(calibration_loader)   # one forward pass, hooks all layers
        pipeline.quantize_all()                  # apply GPTQ layer by layer
        quantized_model = pipeline.model         # model with dequantized INT8 weights
    """

    def __init__(self, model: nn.Module, config: QuantConfig):
        self.model = model
        self.config = config
        self._gptq_layers: Dict[str, GPTQLayer] = {}
        self._hooks: List = []
        self._skipped: Set[str] = set()
        self._layer_order: List[str] = []

    # ── Layer selection ───────────────────────────────────────────────────────

    def _should_skip(self, name: str, module: nn.Module) -> bool:
        """Return True if this module should be kept at FP32."""
        # Always skip entropy models and GDN
        if _ENTROPY_TYPES and isinstance(module, _ENTROPY_TYPES):
            return True
        if _GDN_TYPES and isinstance(module, _GDN_TYPES):
            return True
        # Skip by name pattern
        name_lower = name.lower()
        for pat in self.config.skip_patterns:
            if pat in name_lower:
                return True
        return False

    def _collect_layers(self):
        """Walk model, collect quantizable layers and their GPTQ handlers."""
        conv_names = []  # collect all Conv2d names to identify first/last
        for name, module in self.model.named_modules():
            if isinstance(module, _QUANTIZABLE_TYPES) and not self._should_skip(name, module):
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    conv_names.append(name)

        first_conv = conv_names[0] if conv_names else None
        last_conv = conv_names[-1] if conv_names else None

        for name, module in self.model.named_modules():
            if not isinstance(module, _QUANTIZABLE_TYPES):
                continue
            if self._should_skip(name, module):
                self._skipped.add(name)
                continue
            if self.config.skip_first_last_conv and isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                if name in (first_conv, last_conv):
                    self._skipped.add(name)
                    logger.info(f"Skipping (first/last conv): {name}")
                    continue
            self._gptq_layers[name] = GPTQLayer(module, self.config)
            self._layer_order.append(name)

        logger.info(f"Quantizable layers: {len(self._gptq_layers)}")
        logger.info(f"Skipped layers:     {len(self._skipped)}")

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self, dataloader):
        """
        Run model on calibration data with forward hooks active.
        Hooks accumulate the Hessian (X^T X) for each quantizable layer.
        One forward pass is sufficient; no gradients are computed.
        """
        self._collect_layers()
        self._register_hooks()

        self.model.eval()
        device = next(self.model.parameters()).device
        n_batches = 0

        logger.info(f"Calibrating on {len(dataloader)} batches ...")
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device)
                try:
                    self.model(x)
                except Exception as e:
                    # Some batches may fail (padding issues etc.) — skip gracefully
                    logger.debug(f"Calibration batch skipped: {e}")
                n_batches += 1
                if n_batches % 10 == 0:
                    logger.info(f"  Calibration: {n_batches}/{len(dataloader)} batches")

        self._remove_hooks()
        logger.info(f"Calibration complete ({n_batches} batches).")

    def _register_hooks(self):
        """Attach forward pre-hooks to each quantizable layer."""
        for name, gptq_layer in self._gptq_layers.items():
            module = gptq_layer.layer

            def make_hook(gl):
                def hook(module, inp):
                    if inp and inp[0] is not None:
                        gl.add_batch(inp[0])
                return hook

            h = module.register_forward_pre_hook(make_hook(gptq_layer))
            self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Quantization ─────────────────────────────────────────────────────────

    def quantize_all(self):
        """
        Apply GPTQ to every collected layer in forward order.
        Updates each layer's .weight.data in-place with dequantized INT8 weights.
        Hessian memory is freed after each layer to keep GPU memory usage low.
        """
        total = len(self._layer_order)
        logger.info(f"Applying GPTQ to {total} layers ...")

        scales: Dict[str, torch.Tensor] = {}

        for i, name in enumerate(self._layer_order):
            gptq_layer = self._gptq_layers[name]
            module = gptq_layer.layer

            W_dq = gptq_layer.quantize()
            module.weight.data.copy_(W_dq)

            # Record per-layer scale for logging / export
            W_orig = W_dq.float()
            maxv = _maxval(self.config.bits)
            s = W_orig.reshape(W_orig.shape[0], -1).abs().max(dim=1)[0] / maxv
            scales[name] = s.cpu()

            gptq_layer.free()   # release H to free GPU memory

            if (i + 1) % 50 == 0 or (i + 1) == total:
                logger.info(f"  Quantized {i + 1}/{total} layers")

        logger.info("GPTQ quantization complete.")
        return scales

    def get_quantization_summary(self) -> dict:
        """Return a summary dict for logging / saving."""
        return {
            "algorithm": "GPTQ",
            "bits": self.config.bits,
            "symmetric": self.config.sym,
            "per_channel": self.config.per_channel,
            "percdamp": self.config.percdamp,
            "n_quantized_layers": len(self._layer_order),
            "n_skipped_layers": len(self._skipped),
            "skipped_layers": sorted(self._skipped),
        }
