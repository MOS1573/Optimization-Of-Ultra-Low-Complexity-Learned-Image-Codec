#!/usr/bin/env python3
"""
Per-image PSNR extraction for box-plot figure.

Evaluates FP32, INT8 GPTQ, INT8 PTQ, and Mixed QAT Phase 3 on all 24 Kodak
images at λ=0.0018 and writes per-image PSNR to a JSON file.

Usage (SLURM):
    sbatch Optimization/run_per_image_psnr.sh
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in [
    str(REPO_ROOT),
    str(REPO_ROOT / "Optimization" / "4-bit" / "4-bit Mixed-Precision"),
    str(REPO_ROOT / "Optimization" / "4-bit" / "4-bit QAT"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets.datasets import ImageFolder          # noqa: E402
from models.get_model import get_model             # noqa: E402
from utils.config import model_config              # noqa: E402
from utils.func import update_registered_buffers   # noqa: E402

KODAK_PATH = "/home/atuin/b290dc/b290dc10/TestDataset/images/Kodak"
CFG_PATH   = REPO_ROOT / "configs" / "models" / "loclic.yaml"
OUT_JSON   = REPO_ROOT / "results" / "per_image_psnr_kodak.json"


def _build_model(device):
    cfg = model_config(str(CFG_PATH))
    args = argparse.Namespace(lmbda=0.0018)
    net, _, _ = get_model(cfg, args, device)
    return net.to(device)


def _extract_state(raw):
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(raw, dict) and key in raw:
            return raw[key]
    return raw


def _load_weights(model, state):
    if hasattr(model, "entropy_bottleneck"):
        update_registered_buffers(
            model.entropy_bottleneck, "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"], state,
        )
    if hasattr(model, "gaussian_conditional"):
        update_registered_buffers(
            model.gaussian_conditional, "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"], state,
        )
    nn.Module.load_state_dict(model, state, strict=False)


def load_fp32(ckpt, device):
    raw = torch.load(ckpt, map_location=device, weights_only=False)
    state = _extract_state(raw)
    if not isinstance(state, dict):
        return state.eval().to(device)
    model = _build_model(device)
    _load_weights(model, state)
    return model.eval()


def load_gptq_ptq(ckpt, device):
    raw = torch.load(ckpt, map_location=device, weights_only=False)
    state = _extract_state(raw)
    if not isinstance(state, dict):
        return state.eval().to(device)
    model = _build_model(device)
    _load_weights(model, state)
    return model.eval()


def load_mixed_qat(ckpt, device):
    raw = torch.load(ckpt, map_location=device, weights_only=False)
    state = _extract_state(raw)
    if not isinstance(state, dict):
        return state.eval().to(device)
    model = _build_model(device)
    _load_weights(model, state)
    return model.eval()


def _pad_to_multiple(x, p=64):
    H, W = x.shape[-2:]
    pH = (p - H % p) % p
    pW = (p - W % p) % p
    if pH or pW:
        import torch.nn.functional as F
        x = F.pad(x, (0, pW, 0, pH))
    return x, (H, W)


def _psnr(x_hat, x):
    mse = torch.mean((x_hat - x) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def per_image_psnr(model, loader, device):
    psnr_list = []
    model.eval()
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            _, _, H, W = images.shape
            x_pad, _ = _pad_to_multiple(images)
            out = model(x_pad)
            x_hat = out["x_hat"].clamp(0, 1)[:, :, :H, :W]
            psnr_list.append(_psnr(x_hat, images))
    return psnr_list


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = ImageFolder(KODAK_PATH, transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    print(f"Images: {len(ds)}")

    models_to_eval = [
        ("FP32 Baseline",       REPO_ROOT / "checkpoints/loc-lic/0.0018.pth.tar",                                          load_fp32),
        ("INT8 GPTQ",           REPO_ROOT / "results/legacy/old_kodak_test/gptq_second_try/LoC_LIC_int8_gptq_8bit.pt",     load_gptq_ptq),
        ("INT8 PTQ",            REPO_ROOT / "results/legacy/old_kodak_test/ptq_fast_test/LoC_LIC_int8_ptq.pt",              load_gptq_ptq),
        ("Mixed QAT Phase 3", REPO_ROOT / "results/4-bit/phase3-mixed-finetuned/lambda-0018/LoC_LIC_mixed_qat.pt",    load_mixed_qat),
    ]

    results = {}
    for name, ckpt, loader_fn in models_to_eval:
        print(f"\nLoading {name} from {ckpt.name} ...", flush=True)
        if not ckpt.exists():
            print(f"  [SKIP] checkpoint not found")
            continue
        model = loader_fn(ckpt, device)
        psnr = per_image_psnr(model, loader, device)
        results[name] = psnr
        print(f"  mean PSNR = {sum(psnr)/len(psnr):.3f} dB  (n={len(psnr)})")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved per-image PSNR to {OUT_JSON}")


if __name__ == "__main__":
    main()
