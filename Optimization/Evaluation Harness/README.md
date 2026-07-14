# LoC-LIC Evaluation Harness

This folder contains a full evaluation harness for all 9 LoC-LIC quantization
variants. It measures **image quality**, **inference speed**, **GPU memory**,
and **model complexity** in one script and saves everything to CSV, JSON, and
PNG plots.

---

## Files

| File | Purpose |
|------|---------|
| `evaluate_all.py` | Main evaluation script — runs on GPU, measures all metrics |
| `run_eval.sh` | SLURM launcher — submits to rtx3080 with 1 GPU for 6 hours |
| `RESULTS.md` | Final evaluation results with analysis and caveats |

---

## How to Run

### On the cluster (recommended — needs GPU for timing/VRAM)

```bash
sbatch --partition=rtx3080 --gres=gpu:1 --cpus-per-task=8 --time=6:00:00 \
       "Optimization/Evaluation Harness/run_eval.sh"
```

### Only specific models (faster — ~1–2 hours)

```bash
sbatch --partition=rtx3080 --gres=gpu:1 --cpus-per-task=8 --time=2:00:00 \
       "Optimization/Evaluation Harness/run_eval.sh" fp32,mixed_p3,kd_p4
```

### Locally (no GPU — quality metrics only, timing/VRAM will be N/A)

```bash
python "Optimization/Evaluation Harness/evaluate_all.py" \
    --models fp32,mixed_p3 \
    --datasets kodak
```

---

## Model Keys

| Key | Model | Bits | Method |
|-----|-------|:----:|--------|
| `fp32` | FP32 Baseline | 32 | No quantization |
| `int8_gptq` | INT8 GPTQ | 8 | GPTQ, no fine-tune |
| `int8_ptq` | INT8 PTQ | 8 | Weight-only QDQ |
| `int4_gptq` | INT4 GPTQ | 4 | GPTQ, no fine-tune |
| `int4_qat` | INT4 QAT Phase 1 | 4 | QAT, 100 epochs |
| `mixed_p2` | Mixed QAT Phase 2 | 4/8 | Mixed INT8/INT4 QAT, 100 epochs |
| `mixed_p3` | Mixed QAT Phase 3 ★ | 4/8 | Mixed QAT, lower LR, 50 epochs |
| `kd_p4` | KD QAT Phase 4 | 4/8 | QAT + feature distillation, 100 epochs |

---

## Output Files

Everything is saved to `results/eval_harness/eval_<timestamp>/`:

| File | Contents |
|------|---------|
| `evaluation_kodak.csv` | One row per model, all metrics |
| `evaluation_pro_2020.csv` | Same for the pro_2020 dataset |
| `evaluation_kodak.json` | Full results in JSON format |
| `rd_psnr_bpp_kodak.png` | PSNR vs BPP scatter plot |
| `quality_vs_size_kodak.png` | PSNR vs deployment weight size plot |

---

## Metrics — What Each One Means

### Image Quality Metrics

**PSNR (Peak Signal-to-Noise Ratio)**
Measured in decibels (dB). Compares the reconstructed image to the original
pixel-by-pixel. Higher is better. The formula is:
`PSNR = 10 × log10(1 / MSE)` where MSE is the mean squared error per pixel.
- > 40 dB → nearly lossless (hard to see any difference)
- 30–40 dB → good quality
- < 30 dB → visible degradation

**SSIM (Structural Similarity Index)**
Range: 0 to 1. Measures luminance, contrast, and structural similarity.
More aligned with human perception than PSNR.
- > 0.95 → very high quality
- 0.80–0.95 → good
- < 0.80 → noticeable degradation

**MS-SSIM (Multi-Scale SSIM)**
Same idea as SSIM but computed at multiple image scales (resolutions).
More robust to viewing distance than single-scale SSIM.

**LPIPS (Learned Perceptual Image Patch Similarity)**
Range: 0 to 1. Uses deep neural network features to measure perceptual
similarity. Lower is better (0 = identical, 1 = very different).
Unlike PSNR and SSIM, LPIPS captures whether textures and fine details
*look* correct, not just whether they're numerically close.

**ΔPSNR**
The PSNR gap vs the FP32 baseline. Negative means worse than FP32.
The project target was < −1.0 dB. Phase 3 achieved −0.456 dB.

**BPP (Bits Per Pixel)**
How many bits the compressed image uses per pixel.
Lower = smaller compressed file. The FP32 baseline at λ=0.0018 uses ~0.286 BPP
on Kodak. Quantized models operate at a different point (~0.12 BPP) because
quantizing the weights shifts the entropy model's probability assignments; BPP
should not be compared directly across variants without full R-D curves.
This is the rate part of the rate-distortion trade-off.

---

### Complexity & Resource Metrics

**Latency (ms/image)**
Average time for one forward pass on the GPU, measured in milliseconds.
- Includes: all neural network layers (g_a, h_a, h_s, g_s, attention, etc.)
           and entropy probability estimation (likelihood computation).
- Excludes: actual arithmetic range coding (compress/decompress API).
- Measured with 3 warmup passes + 15 timed passes, then averaged.
- Important: QAT models (Phase 1–4) use fake quantization — they run as
  FP32 at inference time. Their latency equals the FP32 baseline.
  To get real INT8/INT4 speedup, the model must be exported via TensorRT.

**VRAM (MB) — Peak GPU Memory**
Peak GPU memory allocated during the inference measurement loop.
Includes: model weights loaded on GPU + forward pass activations.
- FP32 model: weights occupy 268 MB on GPU + activations (~300–400 MB) ≈ 600–700 MB total
- QAT fake-quant models: same as FP32 (weights still stored as float32)
- True INT8 model (after TensorRT export): weights ~67 MB → ~200–300 MB total

**GMACs (Giga Multiply-Accumulate Operations)**
Total multiply-add operations in one forward pass, in billions.
Counted for Conv2d and Linear layers via forward hooks.
All precision variants have the SAME GMACs — quantization changes the
bit-width of arithmetic, not the number of operations.
Entropy coding (range coder) arithmetic is NOT included in this count.

**kMAC/pixel**
GMACs normalized by image size: (total MACs / 1000) / (H × W).
Useful for comparing complexity across different image resolutions.

**Deployment weight size (MB)**
The theoretical size of the model weights if truly stored in the target
precision (not the checkpoint file on disk).
Formula:
- FP32:  69.9M params × 4 B = 268 MB
- INT8:  69.9M params × 1 B =  67 MB  (4× smaller)
- INT4:  69.9M params × 0.5 B = 34 MB (8× smaller)
- Mixed: 69.9M × (0.45×1 + 0.55×0.5) B = 48 MB (~5.5× smaller)

**Checkpoint size on disk (MB)**
Actual size of the saved `.pt` / `.pth.tar` file.
All checkpoints are ~268 MB because QAT weights are saved as FP32 tensors
with quantization metadata (scales/zeros). To achieve the smaller deployment
sizes above, you would export the model with TensorRT or ONNX Runtime.

---

## Important Notes

### Fake Quantization vs True Quantization

Our QAT checkpoints (Phase 1–4) use **fake quantization** during training:
- Weights are stored as FP32 (full precision) in the checkpoint file.
- During the forward pass, a quantize + dequantize operation is applied
  to the weights, simulating the rounding noise of INT8/INT4 arithmetic.
- This means: **inference speed = FP32 speed**, **VRAM = FP32 VRAM**.
- The quality metrics (PSNR, SSIM) DO reflect the quantization noise correctly.

To get real INT8/INT4 speed and memory benefits:
1. Export the model using `extract_deployment_state_dict_mixed()` from
   `qat_modules_mixed.py` to get quantized weight values.
2. Convert to TensorRT or ONNX Runtime quantized format.

### FLOPs Coverage

The FLOPs counter (GMACs) captures **Conv2d and Linear layers only**.
It does NOT count:
- Entropy coding operations (arithmetic range coder)
- Batch normalization
- Activation functions (ReLU, GELU, etc.)
- Softmax in attention layers

For the purposes of comparing quantized vs FP32 complexity, these omitted
operations are the same in all variants, so the relative comparison is valid.

### GPTQ / PTQ Models

GPTQ and PTQ checkpoints were saved as FP32 state_dicts where the weights
have been rounded to the INT8/INT4 quantization grid. They are loaded as
standard FP32 models. Their quality metrics reflect the quantization effect
(rounded weights), but their latency and VRAM are FP32 figures.

---

## Environment

| Item | Value |
|------|-------|
| PyTorch | 2.11.0+cu130 |
| Python | 3.10 |
| Training GPU | 2× NVIDIA RTX 3080 (9.6 GiB) |
| Eval GPU | 1× NVIDIA RTX 3080 |
| Conda env | `loclic` |
