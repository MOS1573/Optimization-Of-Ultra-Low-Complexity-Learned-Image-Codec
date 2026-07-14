# LoC-LIC Windows Desktop App

PyQt6 desktop application for learned image compression using LoC-LIC models.
Supports compress, decompress, and `.lic` container file I/O across all 4
rate-distortion operating points.

---

## Features

- Load PNG / JPEG / BMP images
- Compress with any available model (FP32, INT8 GPTQ, Mixed QAT ★)
- Lambda selector: 0.0018 / 0.0035 / 0.0067 / 0.025
- Side-by-side original vs reconstructed image display
- Metrics: PSNR, BPP, file size, encode time, decode time
- Save compressed image as `.lic` file
- Open any `.lic` file — model auto-detected from file header
- Dark Fusion theme, non-blocking background inference

---

## Folder Structure

```
Windows Desktop App/
├── app/
│   ├── __init__.py
│   ├── main.py       ← entry point (PyQt6 QApplication)
│   ├── codec.py      ← model loading + compress/decompress logic
│   └── ui.py         ← MainWindow, ImagePanel, MetricsBar
├── scripts/
│   └── export_onnx.py  ← Phase 2a: export backbone to ONNX + INT8 quant
├── build/
│   ├── loclic.spec     ← PyInstaller spec
│   └── build_windows.bat
└── requirements.txt
```

---

## Running in Development

```bash
# From the repo root, with the loclic conda env active:
conda activate loclic
python "Optimization/Windows Desktop App/app/main.py"

# Or with an explicit repo root (if running from a different directory):
LOCLIC_REPO_ROOT=/path/to/loc-lic-new \
    python "Optimization/Windows Desktop App/app/main.py"
```

---

## Building the Windows .exe

On a Windows machine with the conda env installed:

```bat
cd "Optimization\Windows Desktop App"
pip install -r requirements.txt
build\build_windows.bat
```

Output: `dist\LoC-LIC Compressor\LoC-LIC Compressor.exe`

After building, copy the model checkpoints next to the `.exe`:

```
dist\LoC-LIC Compressor\
├── LoC-LIC Compressor.exe
├── checkpoints\loc-lic\
│   ├── 0.0018.pth.tar
│   ├── 0.0035.pth.tar
│   ├── 0.0067.pth.tar
│   └── 0.025.pth.tar
└── results\
    ├── legacy\old_kodak_test\gptq_second_try\LoC_LIC_int8_gptq_8bit.pt
    ├── 8-bit\gptq-per-lambda\lambda-0035\LoC_LIC_int8_gptq_8bit.pt
    ├── 8-bit\gptq-per-lambda\lambda-0067\LoC_LIC_int8_gptq_8bit.pt
    ├── 8-bit\gptq-per-lambda\lambda-025\LoC_LIC_int8_gptq_8bit.pt
    ├── 4-bit\phase3-mixed-finetuned\lambda-0018\LoC_LIC_mixed_qat.pt
    └── 4-bit\phase4b-kd-v2\LoC_LIC_kd.pt
```

---

## How the .lic Format Is Used

| Action | What happens |
|--------|--------------|
| Compress | model.compress(x) → bitstream → `write_lic()` → `.lic` file |
| Open .lic | `read_lic()` reads header (lambda + quant type) → auto-loads model → `model.decompress()` |
| Save .lic | `write_lic()` from the last compress result |

The `.lic` header encodes lambda ID and quant type, so any copy of the app
can open any `.lic` file as long as the corresponding checkpoint is present.

---

## Model Availability

| Model | λ values available |
|-------|--------------------|
| FP32 Baseline | 0.0018, 0.0035, 0.0067, 0.025 |
| INT8 GPTQ | 0.0018, 0.0035, 0.0067, 0.025 |
| Mixed QAT Phase 3 ★ | 0.0018 |
| KD QAT Phase 4b v2 ★ | 0.0018 |

---

## Phase 2a — ONNX Export (future)

Once Phase 1c is complete, use `scripts/export_onnx.py` to:
1. Export the NN backbone (g_a, h_a, h_s, g_s) to ONNX
2. Apply ONNX Runtime INT8 static quantization (calibrated on Kodak)
3. Benchmark FP32 vs INT8 ONNX latency (CUDA EP / DirectML)

Note: the arithmetic range coder (CompressAI) cannot be exported to ONNX and
always runs in Python. INT4 is not supported for Conv2d in ONNX Runtime.
