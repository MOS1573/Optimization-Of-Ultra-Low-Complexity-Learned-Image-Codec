#!/usr/bin/env python3
"""
Phase 2a — ONNX Export + INT8 Benchmark for LoC-LIC

Benchmarks four variants on the RTX 3080 (CUDA EP):
  1. PyTorch FP32          — baseline, fake-quant weights (what the app uses now)
  2. ONNX FP32             — same weights, ONNX Runtime CPU/CUDA EP
  3. ONNX INT8 (PTQ)       — ONNX Runtime static INT8, calibrated on Kodak
  4. ONNX INT8 (QAT→INT8)  — Mixed QAT weights exported then INT8-quantized
                              (best quality + INT8 speed)

NOTE: Only the NN backbone is benchmarked (g_a, h_a, h_s, g_s).
CompressAI's arithmetic range coder stays in Python and is NOT included.
The NN transforms dominate inference time (~85–90%).

Usage:
  python benchmark_onnx.py --output_dir results/benchmarks/onnx \
                            --calib /path/to/Kodak --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).parents[3]
for _p in [str(REPO_ROOT), str(REPO_ROOT / "Optimization" / "mixed-precision QAT")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.get_model import get_model          # noqa: E402
from utils.config import model_config           # noqa: E402
from utils.func import update_registered_buffers  # noqa: E402

_CFG_PATH  = REPO_ROOT / "configs" / "models" / "loclic.yaml"
_FP32_CKPT = REPO_ROOT / "checkpoints/loc-lic/0.0018.pth.tar"
_QAT_CKPT  = REPO_ROOT / "results/4-bit/phase3-mixed-finetuned/lambda-0018/LoC_LIC_mixed_qat.pt"

H, W = 512, 768   # standard Kodak landscape resolution


# ── Model loading ─────────────────────────────────────────────────────────────

def _load(ckpt_path: Path, device: torch.device) -> nn.Module:
    cfg = model_config(str(_CFG_PATH))
    net, _, _ = get_model(cfg, argparse.Namespace(lmbda=0.0018), device)
    raw   = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = raw.get("state_dict", raw.get("model", raw)) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        return state.eval().to(device)
    for attr, keys in [
        ("entropy_bottleneck",   ["_quantized_cdf", "_offset", "_cdf_length"]),
        ("gaussian_conditional", ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"]),
    ]:
        if hasattr(net, attr):
            update_registered_buffers(getattr(net, attr), attr, keys, state)
    nn.Module.load_state_dict(net, state, strict=False)
    return net.eval().to(device)


# ── Backbone wrapper (exports only NN transforms, skips entropy coding) ───────

class _BackboneWrapper(nn.Module):
    """Wraps g_a → h_a → h_s → g_s without the arithmetic range coder.

    The entropy model (gaussian_conditional / entropy_bottleneck) uses erfc
    which is not an ONNX-supported operator.  This wrapper benchmarks the NN
    transforms only — they account for ~85-90% of total inference time.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.g_a = model.g_a
        self.h_a = model.h_a
        self.h_s = model.h_s
        self.g_s = model.g_s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y      = self.g_a(x)
        z      = self.h_a(torch.abs(y))
        params = self.h_s(z)          # hyper-synthesis (returns Gaussian params)
        x_hat  = self.g_s(y)          # synthesis — skip entropy quantization
        return x_hat


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(model: nn.Module, path: Path, device: torch.device) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backbone = _BackboneWrapper(model).to(device).eval()
    dummy = torch.zeros(1, 3, H, W, device=device)
    with torch.no_grad():
        torch.onnx.export(
            backbone, dummy, str(path),
            export_params=True, opset_version=17,
            do_constant_folding=True,
            input_names=["image"], output_names=["x_hat"],
            dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                          "x_hat": {0: "batch", 2: "height", 3: "width"}},
            dynamo=False,
        )
    mb = path.stat().st_size / 1024**2
    print(f"  Exported → {path.name}  ({mb:.1f} MB)")


# ── INT8 quantization ─────────────────────────────────────────────────────────

def quantize_int8(fp32_onnx: Path, int8_onnx: Path, calib_dir: str) -> None:
    from onnxruntime.quantization import (
        CalibrationDataReader, QuantFormat, QuantType, quantize_static,
    )
    from PIL import Image as PILImage
    from torchvision import transforms as T

    class _Reader(CalibrationDataReader):
        def __init__(self):
            imgs = sorted(Path(calib_dir).glob("*.png")) + \
                   sorted(Path(calib_dir).glob("*.jpg"))
            tf = T.ToTensor()
            data = []
            for p in sorted(set(imgs))[:24]:
                img = PILImage.open(p).convert("RGB")
                t   = tf(img).unsqueeze(0).numpy()
                pH  = (64 - t.shape[2] % 64) % 64
                pW  = (64 - t.shape[3] % 64) % 64
                if pH or pW:
                    t = np.pad(t, ((0,0),(0,0),(0,pH),(0,pW)), mode="reflect")
                data.append({"image": t})
            self._it = iter(data)
        def get_next(self): return next(self._it, None)

    int8_onnx.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(fp32_onnx), str(int8_onnx),
        calibration_data_reader=_Reader(),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    mb = int8_onnx.stat().st_size / 1024**2
    print(f"  INT8 ONNX → {int8_onnx.name}  ({mb:.1f} MB)")


# ── Benchmarking ──────────────────────────────────────────────────────────────

def bench_pytorch(model: nn.Module, device: torch.device, n: int) -> tuple[float, float]:
    dummy = torch.zeros(1, 3, H, W, device=device)
    with torch.no_grad():
        for _ in range(10): model(dummy)   # warmup
        if device.type == "cuda": torch.cuda.synchronize()
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            model(dummy)
            if device.type == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def bench_onnx(onnx_path: Path, use_cuda: bool, n: int) -> tuple[float, float, str]:
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 8   # explicit count suppresses SLURM affinity warnings
    opts.inter_op_num_threads = 1
    if use_cuda:
        cuda_opts = {"device_id": 0, "arena_extend_strategy": "kNextPowerOfTwo",
                     "cudnn_conv_algo_search": "EXHAUSTIVE", "do_copy_in_default_stream": True}
        providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
    ep   = sess.get_providers()[0].replace("ExecutionProvider", "")
    dummy = np.zeros((1, 3, H, W), dtype=np.float32)
    for _ in range(10): sess.run(None, {"image": dummy})   # warmup
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {"image": dummy})
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times)), ep


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=Path,
                   default=REPO_ROOT / "results" / "onnx_benchmark")
    p.add_argument("--calib", default=
                   "path/to/dataset/Kodak")
    p.add_argument("--n_runs", type=int, default=100)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = p.parse_args()

    device    = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_cuda  = device.type == "cuda"
    out       = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  LoC-LIC ONNX Phase 2a Benchmark")
    print(f"  Device : {device}  |  Image: {H}×{W}  |  N={args.n_runs}")
    print(f"  Output : {out}")
    print(f"{'='*60}\n")

    results = {}

    # ── 1. PyTorch FP32 full model (current app — includes entropy model fwd) ─
    print("[1/6] PyTorch FP32 full model — loading & benchmarking …")
    m_fp32 = _load(_FP32_CKPT, device)
    mu, sd = bench_pytorch(m_fp32, device, args.n_runs)
    results["pytorch_fp32_full"] = (mu, sd, str(device).upper())
    print(f"       {mu:.1f} ± {sd:.1f} ms  (full model incl. entropy fwd)\n")

    # ── 1b. PyTorch FP32 backbone only (fair comparison with ONNX) ───────────
    print("[1b]   PyTorch FP32 backbone only — benchmarking …")
    m_fp32_bb = _BackboneWrapper(m_fp32).to(device).eval()
    mu, sd = bench_pytorch(m_fp32_bb, device, args.n_runs)
    results["pytorch_fp32"] = (mu, sd, str(device).upper())
    print(f"       {mu:.1f} ± {sd:.1f} ms  (backbone only, fair vs ONNX)\n")

    # ── 1c. torch.compile (graph fusion — same model, compiled kernel) ───────
    print("[1c]   PyTorch FP32 backbone + torch.compile — benchmarking …")
    try:
        m_compiled = torch.compile(_BackboneWrapper(m_fp32).to(device).eval(),
                                   mode="reduce-overhead")
        dummy_w = torch.zeros(1, 3, H, W, device=device)
        with torch.no_grad():
            for _ in range(5): m_compiled(dummy_w)   # warmup / compile
        mu, sd = bench_pytorch(m_compiled, device, args.n_runs)
        results["pytorch_compiled"] = (mu, sd, str(device).upper())
        print(f"       {mu:.1f} ± {sd:.1f} ms  (torch.compile reduce-overhead)\n")
    except Exception as e:
        print(f"       torch.compile failed: {e}\n")
        results["pytorch_compiled"] = results["pytorch_fp32"]

    # ── 2. PyTorch Mixed QAT (fake-quant FP32 — same speed as above) ─────────
    print("[2/6] PyTorch Mixed QAT (fake-quant) — loading & benchmarking …")
    m_qat = _load(_QAT_CKPT, device)
    mu, sd = bench_pytorch(m_qat, device, args.n_runs)
    results["pytorch_qat_fakequant"] = (mu, sd, str(device).upper())
    print(f"       {mu:.1f} ± {sd:.1f} ms  (same as FP32 — fake quant)\n")

    # ── 3. ONNX FP32 export + benchmark ──────────────────────────────────────
    fp32_onnx = out / "loclic_fp32.onnx"
    qat_onnx  = out / "loclic_qat_fp32.onnx"
    fp32_int8 = out / "loclic_fp32_int8.onnx"
    qat_int8  = out / "loclic_qat_int8.onnx"

    print("[3/6] Exporting FP32 → ONNX …")
    export_onnx(m_fp32, fp32_onnx, device)
    del m_fp32; torch.cuda.empty_cache() if use_cuda else None
    mu, sd, ep = bench_onnx(fp32_onnx, use_cuda, args.n_runs)
    results["onnx_fp32"] = (mu, sd, ep)
    print(f"       {mu:.1f} ± {sd:.1f} ms  [{ep}]\n")

    print("[4/6] Exporting Mixed QAT → ONNX …")
    export_onnx(m_qat, qat_onnx, device)
    del m_qat; torch.cuda.empty_cache() if use_cuda else None
    mu, sd, ep = bench_onnx(qat_onnx, use_cuda, args.n_runs)
    results["onnx_qat_fp32"] = (mu, sd, ep)
    print(f"       {mu:.1f} ± {sd:.1f} ms  [{ep}]\n")

    # ── 4. INT8 quantization + benchmark ─────────────────────────────────────
    print("[5/6] Applying ONNX Runtime INT8 to FP32 model …")
    quantize_int8(fp32_onnx, fp32_int8, args.calib)
    mu, sd, ep = bench_onnx(fp32_int8, use_cuda, args.n_runs)
    results["onnx_fp32_int8"] = (mu, sd, ep)
    print(f"       {mu:.1f} ± {sd:.1f} ms  [{ep}]\n")

    print("[6/6] Applying ONNX Runtime INT8 to Mixed QAT model …")
    quantize_int8(qat_onnx, qat_int8, args.calib)
    mu, sd, ep = bench_onnx(qat_int8, use_cuda, args.n_runs)
    results["onnx_qat_int8"] = (mu, sd, ep)
    print(f"       {mu:.1f} ± {sd:.1f} ms  [{ep}]\n")

    # ── Report ────────────────────────────────────────────────────────────────
    baseline = results["pytorch_fp32"][0]   # backbone-only is the fair baseline
    rows = [
        ("PyTorch FP32 full model (app today)",  "pytorch_fp32_full"),
        ("PyTorch FP32 backbone only",           "pytorch_fp32"),
        ("PyTorch FP32 + torch.compile",         "pytorch_compiled"),
        ("PyTorch Mixed QAT fake-quant",         "pytorch_qat_fakequant"),
        ("ONNX Runtime FP32",                    "onnx_fp32"),
        ("ONNX Runtime FP32 + INT8 (PTQ)",       "onnx_fp32_int8"),
        ("ONNX Runtime Mixed QAT + INT8 ★",      "onnx_qat_int8"),
    ]

    header = f"\n{'='*70}\n  RESULTS — NN backbone latency  ({H}×{W} image, n={args.n_runs})\n{'='*70}"
    table  = f"{'Model':<40} {'ms':>8} {'±':>6} {'EP':>10} {'Speedup':>8}\n"
    table += "-" * 70 + "\n"
    for label, key in rows:
        mu, sd, ep = results[key]
        sx = f"{baseline/mu:.2f}×"
        table += f"{label:<40} {mu:>7.1f} {sd:>6.1f} {ep:>10} {sx:>8}\n"
    table += "=" * 70

    print(header)
    print(table)

    report_path = out / "benchmark_results.txt"
    with open(report_path, "w") as f:
        import datetime
        f.write(f"LoC-LIC ONNX Phase 2a Benchmark\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"GPU:  {torch.cuda.get_device_name(0) if use_cuda else 'CPU'}\n")
        f.write(f"Image size: {H}×{W}  |  N={args.n_runs} runs\n\n")
        f.write(table)
    print(f"\nSaved → {report_path}\n")


if __name__ == "__main__":
    main()
