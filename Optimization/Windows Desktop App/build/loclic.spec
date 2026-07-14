# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec — LoC-LIC Desktop App (Windows)

Build (run from the repo root):
    cd "Optimization/Windows Desktop App"
    pyinstaller build/loclic.spec

Output:
    dist/LoC-LIC Compressor/        ← folder distribution
    dist/LoC-LIC Compressor.exe     ← single-exe (optional, slower startup)

After building, copy checkpoint files into dist/models/ (see README).
"""

from pathlib import Path

APP_DIR   = Path(SPECPATH)             # .../Optimization/Windows Desktop App
REPO_ROOT = APP_DIR.parents[1]

a = Analysis(
    [str(APP_DIR / "app" / "main.py")],
    pathex=[
        str(REPO_ROOT),
        str(APP_DIR / "app"),
        str(REPO_ROOT / "Optimization" / "mixed-precision QAT"),
        str(REPO_ROOT / "Optimization" / "Lic Container Format"),
    ],
    binaries=[],
    datas=[
        # LoC-LIC Python source modules bundled into the app
        (str(REPO_ROOT / "models"),   "models"),
        (str(REPO_ROOT / "modules"),  "modules"),
        (str(REPO_ROOT / "utils"),    "utils"),
        (str(REPO_ROOT / "datasets"), "datasets"),
        (str(REPO_ROOT / "configs"),  "configs"),
        # Mixed-QAT helpers
        (str(REPO_ROOT / "Optimization" / "mixed-precision QAT"), "mixed_qat"),
        # .lic format reader/writer
        (str(REPO_ROOT / "Optimization" / "Lic Container Format" / "lic_format.py"), "."),
    ],
    hiddenimports=[
        "compressai",
        "compressai.models",
        "compressai.entropy_models",
        "compressai.layers",
        "compressai.ops",
        "timm",
        "timm.models",
        "einops",
        "PyQt6",
        "PyQt6.QtWidgets",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PIL",
        "PIL.Image",
        "numpy",
        "torch",
        "torch.nn",
        "torchvision",
        "torchvision.transforms",
    ],
    hookspath=[],
    runtime_hooks=[],
    # onnxruntime excluded until Phase 2a ONNX backend is integrated
    excludes=["onnxruntime", "matplotlib", "jupyter", "IPython"],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LoC-LIC Compressor",
    debug=False,
    strip=False,
    upx=True,
    console=False,          # no black terminal window
    icon=None,              # replace with path to .ico if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="LoC-LIC Compressor",
)
