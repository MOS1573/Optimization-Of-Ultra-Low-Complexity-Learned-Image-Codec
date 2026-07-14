#!/usr/bin/env python3
"""
Knowledge Distillation modules for LoC-LIC mixed-precision QAT.

Uses forward hooks to capture intermediate feature maps from both the FP32
teacher and the quantized student, then computes a per-layer MSE distillation
loss that supplements the standard rate-distortion objective.

Loss:
    L_total = L_RD + alpha * mean_i( MSE(student_feat_i, teacher_feat_i) )

Default distillation layers:
    g_a  — analysis transform   (most sensitive; feeds entropy model)
    h_s  — hyper decoder        (feeds entropy parameter prediction heads)
    g_s  — synthesis transform  (reconstruction quality)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Modules distilled by default (most impactful based on sensitivity analysis)
DEFAULT_KD_LAYERS: List[str] = ["g_a", "h_s", "g_s"]


# ─────────────────────────────────────────────────────────────────────────────
# Hook
# ─────────────────────────────────────────────────────────────────────────────

class FeatureHook:
    """Captures the primary output tensor of a module via a forward hook."""

    def __init__(self) -> None:
        self.features: Optional[torch.Tensor] = None
        self._handle = None

    def hook_fn(self, module: nn.Module, input, output) -> None:
        if isinstance(output, (tuple, list)):
            self.features = output[0]
        else:
            self.features = output

    def register(self, module: nn.Module) -> "FeatureHook":
        self._handle = module.register_forward_hook(self.hook_fn)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

def register_kd_hooks(
    model: nn.Module,
    layer_names: List[str],
) -> Dict[str, FeatureHook]:
    """
    Register forward hooks on the named modules of *model*.

    Returns a dict {layer_name: FeatureHook}. Warns if a requested name is
    not found in the model.
    """
    hooks: Dict[str, FeatureHook] = {}
    registered: set = set()

    for name, module in model.named_modules():
        if name in layer_names:
            hooks[name] = FeatureHook().register(module)
            registered.add(name)

    missing = set(layer_names) - registered
    if missing:
        print(f"[KD] Warning — layers not found in model, skipping: {sorted(missing)}")
    if registered:
        print(f"[KD] Hooks registered on: {sorted(registered)}")

    return hooks


def remove_kd_hooks(hooks: Dict[str, FeatureHook]) -> None:
    """Remove all registered hooks (call at the end of training)."""
    for hook in hooks.values():
        hook.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_kd_loss(
    teacher_hooks: Dict[str, FeatureHook],
    student_hooks: Dict[str, FeatureHook],
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    MSE distillation loss between corresponding teacher and student features.

    Gradients flow only through the student; teacher features are detached.
    Returns a scalar tensor (0.0 if no matching hooks have captured features).
    """
    losses: List[torch.Tensor] = []

    for name, t_hook in teacher_hooks.items():
        s_hook = student_hooks.get(name)
        if s_hook is None:
            continue
        t_feat = t_hook.features
        s_feat = s_hook.features
        if t_feat is None or s_feat is None:
            continue
        losses.append(F.mse_loss(s_feat.float(), t_feat.float().detach()))

    if not losses:
        return torch.tensor(0.0, requires_grad=True)

    return alpha * torch.stack(losses).mean()
