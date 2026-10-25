#!/usr/bin/env python3
"""
ONNX Export + INT8 Quantization for LoC-LIC (Phase 2a)

WHAT GETS EXPORTED vs WHAT STAYS IN PYTHON
───────────────────────────────────────────
✓ Exported to ONNX:
    model.forward(x) — the full neural-network pipeline:
        g_a (analysis)  →  h_a (hyper-analysis)  →  h_s (hyper-synthesis)  →  g_s (synthesis)
    This path produces x_hat directly from x, sufficient for latency benchmarking.

✗ NOT exportable (stays Python / CompressAI):
    entropy_bottleneck, gaussian_conditional — the RANS arithmetic range coder.
    These are Python loops with integer arithmetic; ONNX has no equivalent operators.

CONSEQUENCE FOR THE APP
───────────────────────
The desktop app uses model.compress() / model.decompress() for actual compression,
which requires CompressAI's Python entropy coder.  The ONNX model is used for:
  1. Latency benchmarking (Phase 2a goal — measure NN speedup from INT8)
  2. Future: swap NN transforms to ONNX Runtime while keeping CompressAI for coding

INT4 NOTE
─────────
ONNX Runtime does not support INT4 for Conv2d (only MatMulNBits for Linear).
Use --quant int8 for real hardware speedup.  Our QAT 4-bit model quality still
shows in the reconstructed image because the weights encode the trained distribution
even when stored as float32.

USAGE
─────
  # Export FP32 ONNX
  python scripts/export_onnx.py --lambda 0.0067 --model fp32 \\
      --output models/loclic_fp32_0067.onnx

  # Export and apply ONNX Runtime INT8 static quantization
  python scripts/export_onnx.py --lambda 0.0067 --model int8_gptq \\
      --output models/loclic_gptq_0067.onnx --quant int8

  # Benchmark ONNX RT latency (CUDA or DirectML on Windows)
  python scripts/export_onnx.py --lambda 0.0067 --model fp32 \\
      --output models/loclic_fp32_0067.onnx --benchmark
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Repo path setup ───────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parents[3]
for _p in [
    str(REPO_ROOT),
    str(REPO_ROOT / "Optimization" / "mixed-precision QAT"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.get_model import get_model               # noqa: E402
from utils.config import model_config                # noqa: E402
from utils.func import update_registered_buffers     # noqa: E402

_CFG_PATH = REPO_ROOT / "configs" / "models" / "loclic.yaml"

CHECKPOINTS = {
    ("fp32",      0.0018): REPO_ROOT / "checkpoints/loc-lic/0.0018.pth.tar",
    ("fp32",      0.0035): REPO_ROOT / "checkpoints/loc-lic/0.0035.pth.tar",
    ("fp32",      0.0067): REPO_ROOT / "checkpoints/loc-lic/0.0067.pth.tar",
    ("fp32",      0.025):  REPO_ROOT / "checkpoints/loc-lic/0.025.pth.tar",
    ("int8_gptq", 0.0018): REPO_ROOT / "results/legacy/old_kodak_test/gptq_second_try/LoC_LIC_int8_gptq_8bit.pt",
    ("int8_gptq", 0.0035): REPO_ROOT / "results/8-bit/gptq-per-lambda/lambda-0035/LoC_LIC_int8_gptq_8bit.pt",
    ("int8_gptq", 0.0067): REPO_ROOT / "results/8-bit/gptq-per-lambda/lambda-0067/LoC_LIC_int8_gptq_8bit.pt",
    ("int8_gptq", 0.025):  REPO_ROOT / "results/8-bit/gptq-per-lambda/lambda-025/LoC_LIC_int8_gptq_8bit.pt",
    ("qat_best",  0.0018): REPO_ROOT / "results/4-bit/phase3-mixed-finetuned/lambda-0018/LoC_LIC_mixed_qat.pt",
}


def load_model(model_type: str, lambda_val: float,
               device: torch.device) -> nn.Module:
    ckpt = CHECKPOINTS.get((model_type, lambda_val))
    if ckpt is None:
        raise ValueError(f"No checkpoint for model={model_type}, λ={lambda_val}.\n"
                         f"Valid: {list(CHECKPOINTS.keys())}")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    cfg  = model_config(str(_CFG_PATH))
    net, _, _ = get_model(cfg, argparse.Namespace(lmbda=0.0018), device)

    raw   = torch.load(ckpt, map_location=device, weights_only=False)
    state = raw.get("state_dict", raw.get("model", raw)) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        return state.eval().to(device)

    for attr, keys in [
        ("entropy_bottleneck",  ["_quantized_cdf", "_offset", "_cdf_length"]),
        ("gaussian_conditional", ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"]),
    ]:
        if hasattr(net, attr):
            update_registered_buffers(getattr(net, attr), attr, keys, state)

    nn.Module.load_state_dict(net, state, strict=False)
    return net.eval().to(device)


def export_onnx(model: nn.Module, output_path: Path,
                h: int = 512, w: int = 768) -> None:
    """Export model.forward() (NN transforms only) to ONNX."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    dummy  = torch.zeros(1, 3, h, w, device=device)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["x_hat"],
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "x_hat": {0: "batch", 2: "height", 3: "width"},
            },
        )

    size_mb = output_path.stat().st_size / 1024 ** 2
    print(f"  Exported → {output_path}  ({size_mb:.1f} MB)")


def quantize_int8(onnx_path: Path, output_path: Path, calib_dir: str) -> None:
    """Apply ONNX Runtime static INT8 quantization with Kodak calibration."""
    try:
        from onnxruntime.quantization import (
            CalibrationDataReader, QuantFormat, QuantType, quantize_static,
        )
        from PIL import Image as PILImage
        from torchvision import transforms as T
    except ImportError as exc:
        print(f"  [skip] INT8 quantization requires onnxruntime: {exc}")
        return

    class _KodakReader(CalibrationDataReader):
        def __init__(self, calib_dir: str, n: int = 24):
            imgs = sorted(Path(calib_dir).glob("*.png")) + \
                   sorted(Path(calib_dir).glob("*.jpg"))
            tf = T.ToTensor()
            self._data = []
            for p in sorted(set(imgs))[:n]:
                img = PILImage.open(p).convert("RGB")
                t   = tf(img).unsqueeze(0).numpy()
                H, W = t.shape[2], t.shape[3]
                pH = (64 - H % 64) % 64
                pW = (64 - W % 64) % 64
                if pH or pW:
                    t = np.pad(t, ((0, 0), (0, 0), (0, pH), (0, pW)), mode="reflect")
                self._data.append({"image": t})
            self._iter = iter(self._data)

        def get_next(self):
            return next(self._iter, None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(onnx_path),
        str(output_path),
        calibration_data_reader=_KodakReader(calib_dir),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    size_mb = output_path.stat().st_size / 1024 ** 2
    print(f"  INT8 ONNX → {output_path}  ({size_mb:.1f} MB)")


def benchmark(onnx_path: Path, n_runs: int = 50) -> None:
    """Benchmark ONNX Runtime latency. Uses CUDA EP if available."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [skip] onnxruntime not installed")
        return

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    ep   = sess.get_providers()[0]

    dummy = np.zeros((1, 3, 512, 768), dtype=np.float32)
    for _ in range(5):                          # warmup
        sess.run(None, {"image": dummy})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"image": dummy})
        times.append((time.perf_counter() - t0) * 1000.0)

    print(f"  EP:      {ep}")
    print(f"  Latency: {np.mean(times):.1f} ± {np.std(times):.1f} ms  (n={n_runs})")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Export LoC-LIC NN backbone to ONNX (Phase 2a)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--lambda", dest="lam", type=float, required=True,
                   choices=[0.0018, 0.0035, 0.0067, 0.025])
    p.add_argument("--model", choices=["fp32", "int8_gptq", "qat_best"],
                   default="fp32")
    p.add_argument("--output", type=Path, required=True,
                   help="Output .onnx path")
    p.add_argument("--quant", choices=["none", "int8"], default="none",
                   help="Apply ONNX Runtime INT8 static quantization")
    p.add_argument("--calib",
                   default="path/to/dataset/Kodak",
                   help="Calibration dataset (Kodak 24 images) for --quant int8")
    p.add_argument("--benchmark", action="store_true",
                   help="Measure ONNX Runtime latency after export")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"\nLoading: model={args.model}  λ={args.lam}  device={device}")
    model = load_model(args.model, args.lam, device)

    print(f"Exporting to ONNX …")
    export_onnx(model, args.output)

    if args.benchmark:
        print("\nBenchmark (FP32 ONNX):")
        benchmark(args.output)

    if args.quant == "int8":
        q_path = args.output.with_stem(args.output.stem + "_int8")
        print(f"\nApplying INT8 quantization …")
        quantize_int8(args.output, q_path, args.calib)
        if args.benchmark:
            print("\nBenchmark (INT8 ONNX):")
            benchmark(q_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
