"""
LoC-LIC Desktop App — PyQt6 main window.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image as PilImage
from PyQt6.QtCore import QObject, QSize, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup, QComboBox, QFileDialog, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QStatusBar, QToolBar, QVBoxLayout, QWidget,
)

try:
    from .codec import MODEL_CONFIGS, ModelConfig, LocLicCodec, find_cfg
except ImportError:
    from codec import MODEL_CONFIGS, ModelConfig, LocLicCodec, find_cfg  # type: ignore[no-redef]

try:
    from lic_format import lic_info
except ImportError:
    lic_info = None


# ── Constants ─────────────────────────────────────────────────────────────────

_LAMBDA_VALUES = [0.0018, 0.0035, 0.0067, 0.025]
_LAMBDA_LABELS = [
    "λ=0.0018  (smallest file)",
    "λ=0.0035  (balanced)",
    "λ=0.0067  (balanced+)",
    "λ=0.025   (best quality)",
]
_LAMBDA_SHORT  = ["λ=0.0018", "λ=0.0035", "λ=0.0067", "λ=0.025"]

_MODEL_LABELS  = ["FP32 Baseline", "INT8 GPTQ", "Mixed QAT ★"]
_LOADER_KEYS   = ["fp32",          "gptq_ptq",  "mixed_qat"]

_SLOT_DEFAULTS = [
    (1, 2),   # INT8 GPTQ,   λ=0.0067
    (2, 2),   # Mixed QAT ★, λ=0.0067
    (0, 3),   # FP32,         λ=0.025
]
_SLOT_POSITIONS = {
    2: [(0, 1)],
    3: [(0, 1), (1, 0)],
    4: [(0, 1), (1, 0), (1, 1)],
}

_SAVE_FILTERS = (
    "LIC compressed (*.lic);;"
    "PNG image (*.png);;"
    "JPEG image (*.jpg);;"
    "WebP image (*.webp);;"
    "BMP image (*.bmp);;"
    "TIFF image (*.tiff)"
)
_FILTER_EXT = {
    "*.lic": ".lic", "*.png": ".png", "*.jpg": ".jpg",
    "*.webp": ".webp", "*.bmp": ".bmp", "*.tiff": ".tiff",
}

# Zoom steps: 0.0 = fit-to-panel, others are absolute scale factors
_ZOOM_STEPS = [0.0, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
_ZOOM_LABELS = ["Fit", "33%", "50%", "75%", "100%", "150%", "200%", "300%"]


# ── Background worker ─────────────────────────────────────────────────────────

class _Worker(QObject):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    @pyqtSlot()
    def run(self):
        try:
            self.finished.emit(self._fn())
        except Exception as exc:
            self.error.emit(str(exc))


# ── Image viewer with scroll + zoom ──────────────────────────────────────────

class ImageViewer(QFrame):
    """
    Scrollable image display with zoom controls.
    No title (managed by the parent panel).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._orig_px:  Optional[QPixmap] = None
        self._zoom_idx: int               = 0    # index into _ZOOM_STEPS

        # Inner label inside the scroll area
        self._img_lbl = QLabel()
        self._img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_lbl.setStyleSheet("background: #1e1e1e;")
        self._img_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._scroll = QScrollArea()
        self._scroll.setWidget(self._img_lbl)
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setStyleSheet("background: #1e1e1e; border: none;")

        # Zoom bar
        self._btn_zm  = QPushButton("−")
        self._btn_zp  = QPushButton("+")
        self._btn_fit = QPushButton("Fit")
        self._lbl_z   = QLabel("Fit")
        for b in (self._btn_zm, self._btn_zp, self._btn_fit):
            b.setFixedHeight(20)
            b.setFixedWidth(28 if b is not self._btn_fit else 34)
        self._lbl_z.setFixedWidth(38)
        self._lbl_z.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_z.setFont(QFont("Segoe UI", 8))

        zbar = QHBoxLayout()
        zbar.setContentsMargins(4, 1, 4, 1)
        zbar.setSpacing(2)
        zbar.addStretch()
        zbar.addWidget(self._btn_zm)
        zbar.addWidget(self._lbl_z)
        zbar.addWidget(self._btn_zp)
        zbar.addWidget(self._btn_fit)
        zbar.addStretch()

        self._vlay = QVBoxLayout(self)
        self._vlay.setContentsMargins(2, 2, 2, 2)
        self._vlay.setSpacing(2)
        self._vlay.addWidget(self._scroll, stretch=1)
        self._vlay.addLayout(zbar)

        # Optional detail widget embedded between scroll area and zoom bar.
        # Inserted via embed_detail(); lives inside this frame so it never
        # affects the grid row height — only borrows space from the scroll area.
        self._detail_widget: Optional[QWidget] = None

        self._btn_zm.clicked.connect(self.zoom_out)
        self._btn_zp.clicked.connect(self.zoom_in)
        self._btn_fit.clicked.connect(self.zoom_fit)

    # ── Public ────────────────────────────────────────────────────────────────

    def embed_detail(self, widget: QWidget) -> None:
        """
        Insert a detail/metrics widget between the scroll area and zoom bar.
        The widget starts hidden. Toggling its visibility only steals space
        from the scroll area — it never changes this frame's grid row height.
        """
        self._detail_widget = widget
        widget.setMinimumHeight(0)
        widget.setVisible(False)
        # Insert before the zoom bar (last item is the zbar QHBoxLayout)
        self._vlay.insertWidget(self._vlay.count() - 1, widget)

    def show_detail(self, visible: bool) -> None:
        if self._detail_widget is not None:
            self._detail_widget.setVisible(visible)

    def set_tensor(self, t: torch.Tensor) -> None:
        arr = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        H, W, _ = arr.shape
        qi = QImage(arr.tobytes(), W, H, W * 3, QImage.Format.Format_RGB888)
        self._orig_px = QPixmap.fromImage(qi)
        self._refresh()

    def set_file(self, path: str) -> None:
        self._orig_px = QPixmap(path)
        self._refresh()

    def clear(self, placeholder: str = "") -> None:
        self._orig_px = None
        self._img_lbl.setPixmap(QPixmap())
        self._img_lbl.setText(placeholder)
        vp = self._scroll.viewport().size()
        self._img_lbl.resize(vp)
        self._lbl_z.setText("Fit")
        self._zoom_idx = 0

    def zoom_in(self) -> None:
        if self._zoom_idx < len(_ZOOM_STEPS) - 1:
            self._zoom_idx += 1
            self._refresh()

    def zoom_out(self) -> None:
        if self._zoom_idx > 0:
            self._zoom_idx -= 1
            self._refresh()

    def zoom_fit(self) -> None:
        self._zoom_idx = 0
        self._refresh()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        if not self._orig_px or self._orig_px.isNull():
            return
        factor = _ZOOM_STEPS[self._zoom_idx]
        vp = self._scroll.viewport().size()

        if factor == 0.0:
            # Fit: scale pixmap to viewport, no scrollbars
            scaled = self._orig_px.scaled(
                vp.width(), vp.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._img_lbl.setPixmap(scaled)
            self._img_lbl.resize(vp)
            self._lbl_z.setText("Fit")
        else:
            # Fixed zoom: show at factor × natural size, scrollbars if needed
            w = max(1, int(self._orig_px.width()  * factor))
            h = max(1, int(self._orig_px.height() * factor))
            scaled = self._orig_px.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._img_lbl.setPixmap(scaled)
            self._img_lbl.resize(scaled.size())
            self._lbl_z.setText(_ZOOM_LABELS[self._zoom_idx])

    def resizeEvent(self, e):
        if _ZOOM_STEPS[self._zoom_idx] == 0.0:
            self._refresh()
        super().resizeEvent(e)


# ── Info bars ─────────────────────────────────────────────────────────────────

def _make_info_box(key: str, title: str) -> tuple:
    """Return (QGroupBox, QLabel) pair for a metric/info field."""
    box = QGroupBox(title)
    box.setAlignment(Qt.AlignmentFlag.AlignCenter)
    inner = QVBoxLayout(box)
    inner.setContentsMargins(2, 2, 2, 2)
    lbl = QLabel("—")
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setFont(QFont("Segoe UI", 9))
    lbl.setWordWrap(True)
    inner.addWidget(lbl)
    return box, lbl


class OriginalInfoBar(QWidget):
    """
    One-row info bar shown beneath the original image.
    Switches between image-file mode and .lic-decode mode.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 1, 2, 1)
        lay.setSpacing(2)
        self._labels: dict[str, QLabel] = {}
        for key, title in [
            ("f1", "—"), ("f2", "—"), ("f3", "—"), ("f4", "—"), ("f5", "—"),
        ]:
            box, lbl = _make_info_box(key, title)
            lay.addWidget(box)
            self._labels[key] = lbl
        self._boxes = [lay.itemAt(i).widget() for i in range(5)]

    def _set_fields(self, fields: list[tuple[str, str]]) -> None:
        for i, (title, value) in enumerate(fields[:5]):
            box = self._boxes[i]
            box.setTitle(title)
            box.setVisible(True)
            self._labels[f"f{i+1}"].setText(value)
        for i in range(len(fields), 5):
            self._boxes[i].setVisible(False)

    def update_image(self, path: str) -> None:
        p = Path(path)
        try:
            stat_kb = p.stat().st_size / 1024.0
            px = QPixmap(path)
            w, h = (px.width(), px.height()) if not px.isNull() else (0, 0)
            raw_mb = (w * h * 3) / (1024 * 1024)
            fmt = p.suffix.upper().lstrip(".") or "?"
        except Exception:
            stat_kb, w, h, raw_mb, fmt = 0, 0, 0, 0, "?"
        self._set_fields([
            ("Name",        p.name[:22] + ("…" if len(p.name) > 22 else "")),
            ("Resolution",  f"{w} × {h} px"),
            ("File size",   f"{stat_kb:.1f} KB"),
            ("Format",      fmt),
            ("Raw (24bpp)", f"{raw_mb:.2f} MB"),
        ])

    def update_lic(self, lam: float, quant: str, bpp: float,
                   file_kb: float, dec_ms: float) -> None:
        def fm(v): return f"{v:.0f} ms" if math.isfinite(v) else "—"
        model_short = {"fp32": "FP32", "int8_gptq": "INT8 GPTQ",
                       "qat_best": "Mixed QAT"}.get(quant, quant)
        self._set_fields([
            ("Lambda",   str(lam)),
            ("Codec",    model_short),
            ("BPP",      f"{bpp:.4f}"),
            ("File",     f"{file_kb:.1f} KB"),
            ("Decode",   fm(dec_ms)),
        ])

    def clear(self) -> None:
        self._set_fields([("—", "—")] * 5)


class MetricsBar(QWidget):
    """
    Two-row metrics bar for comparison slots.
      Row 1 — quality:  PSNR | ΔPSNR | MS-SSIM | BPP | Ratio
      Row 2 — system:   File  | Wts.(disk) | Encode  | Decode
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._labels: dict[str, QLabel]  = {}
        self._boxes:  dict[str, QGroupBox] = {}
        vlay = QVBoxLayout(self)
        vlay.setContentsMargins(2, 1, 2, 1)
        vlay.setSpacing(2)

        for fields in (
            [("psnr","PSNR"),("ms_ssim","MS-SSIM"),("bpp","BPP"),("ratio","Ratio")],
            [("size","File size"),("deploy","Wts. (disk)"),("enc","Encode"),("dec","Decode")],
        ):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(2)
            for key, title in fields:
                box, lbl = _make_info_box(key, title)
                row.addWidget(box)
                self._labels[key] = lbl
                self._boxes[key]  = box
            vlay.addLayout(row)

        self._boxes["deploy"].setToolTip(
            "Theoretical weight file size after quantization conversion to native INT8/INT4.\n"
            "Runtime GPU VRAM is higher: FP32 ~268 MB, INT8 GPTQ ~268 MB (weights still\n"
            "float32 in PyTorch), Mixed QAT ~931 MB (fake-quant parametrizations retained)."
        )

    def update(self, psnr: float, ms_ssim: float, bpp: float,
               file_kb: float, enc_ms: float, dec_ms: float,
               deploy_mb: float) -> None:
        def fp(v): return f"{v:.2f} dB" if math.isfinite(v) else "—"
        def fm(v): return f"{v:.0f} ms" if math.isfinite(v) else "—"
        ratio = (24.0 / bpp) if bpp > 0 else 0.0
        self._labels["psnr"].setText(fp(psnr))
        self._labels["ms_ssim"].setText(f"{ms_ssim:.4f}" if math.isfinite(ms_ssim) else "—")
        self._labels["bpp"].setText(f"{bpp:.4f}")
        self._labels["ratio"].setText(f"{ratio:.0f}×")
        self._labels["size"].setText(f"{file_kb:.1f} KB")
        self._labels["deploy"].setText(f"{deploy_mb:.0f} MB")
        self._labels["enc"].setText(fm(enc_ms))
        self._labels["dec"].setText(fm(dec_ms))

    def clear(self) -> None:
        for lbl in self._labels.values():
            lbl.setText("—")


# ── Original panel ────────────────────────────────────────────────────────────

class OriginalPanel(QWidget):
    """
    Left/top-left panel: always shows source image or decoded .lic.
    Has its own zoom controls and optional info bar.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._path:       Optional[str]  = None
        self._lic_result: Optional[dict] = None

        # Header
        self._title_lbl = QLabel("—")
        self._title_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))

        self._btn_details = QPushButton("Details ▾")
        self._btn_details.setCheckable(True)
        self._btn_details.setMaximumWidth(82)
        self._btn_details.setFixedHeight(22)

        self._btn_save = QPushButton("Export…")
        self._btn_save.setEnabled(False)
        self._btn_save.setMaximumWidth(68)
        self._btn_save.setFixedHeight(22)

        hdr = QHBoxLayout()
        hdr.setContentsMargins(4, 2, 4, 0)
        hdr.addWidget(self._title_lbl)
        hdr.addStretch()
        hdr.addWidget(self._btn_save)
        hdr.addWidget(self._btn_details)

        # Viewer — info bar embedded inside so it never affects grid row heights
        self._viewer   = ImageViewer()
        self._info_bar = OriginalInfoBar()
        self._viewer.embed_detail(self._info_bar)

        vlay = QVBoxLayout(self)
        vlay.setContentsMargins(2, 2, 2, 2)
        vlay.setSpacing(2)
        vlay.addLayout(hdr)
        vlay.addWidget(self._viewer, stretch=1)

        self._btn_details.toggled.connect(self._toggle_details)
        self._btn_save.clicked.connect(self._on_save)
        self.clear()

    def set_file(self, path: str) -> None:
        self._path       = path
        self._lic_result = None
        self._title_lbl.setText(Path(path).name)
        self._viewer.set_file(path)
        self._info_bar.update_image(path)
        self._btn_save.setEnabled(False)

    def set_lic_result(self, result: dict, cfg: Optional[ModelConfig] = None) -> None:
        """Show a decoded .lic image and auto-reveal its details."""
        self._lic_result = result
        self._path       = None
        self._viewer.set_tensor(result["x_hat"])

        lam   = result.get("lic_lambda", "?")
        quant = result.get("lic_quant",  "?")
        model_short = {"fp32": "FP32", "int8_gptq": "INT8 GPTQ",
                       "qat_best": "Mixed QAT ★"}.get(quant, quant)
        self._title_lbl.setText(f"{model_short}  ·  λ={lam}")
        self._info_bar.update_lic(
            lam=lam, quant=quant,
            bpp=result.get("bpp", 0.0),
            file_kb=result.get("file_kb", 0.0),
            dec_ms=result.get("dec_ms", float("nan")),
        )
        self._btn_save.setEnabled(True)
        # Auto-open details so user immediately sees the .lic metadata
        self._btn_details.setChecked(True)

    def clear(self) -> None:
        self._path       = None
        self._lic_result = None
        self._title_lbl.setText("—")
        self._viewer.clear("Open an image to start")
        self._info_bar.clear()
        self._btn_save.setEnabled(False)
        self._btn_details.setChecked(False)

    def _toggle_details(self, checked: bool) -> None:
        self._viewer.show_detail(checked)
        self._btn_details.setText("Details ▴" if checked else "Details ▾")

    def _on_save(self) -> None:
        """Export the decoded .lic image (or loaded image) to a standard format."""
        if self._lic_result is None:
            return
        path, sel_filter = QFileDialog.getSaveFileName(
            self, "Export image", "decoded_output",
            # .lic re-save excluded: we already have the file on disk
            "PNG image (*.png);;"
            "JPEG image (*.jpg);;"
            "WebP image (*.webp);;"
            "BMP image (*.bmp);;"
            "TIFF image (*.tiff)"
        )
        if not path:
            return
        if not Path(path).suffix:
            for pattern, ext in [("*.png",".png"),("*.jpg",".jpg"),
                                   ("*.webp",".webp"),("*.bmp",".bmp"),("*.tiff",".tiff")]:
                if pattern in sel_filter:
                    path += ext
                    break
        try:
            _save_image(self._lic_result["x_hat"], path)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))


# ── Comparison slot ───────────────────────────────────────────────────────────

class ComparisonSlot(QWidget):
    compression_done  = pyqtSignal(object)
    compression_error = pyqtSignal(object, str)

    def __init__(self, slot_index: int, parent=None):
        super().__init__(parent)
        self._codec:  Optional[LocLicCodec] = None
        self._thread: Optional[QThread]     = None
        self._worker: Optional[_Worker]     = None
        self._result: Optional[dict]        = None

        # ── Selectors ────────────────────────────────────────────────────────
        self._cmb_model  = QComboBox()
        self._cmb_model.addItems(_MODEL_LABELS)
        self._cmb_model.setMinimumWidth(105)

        self._cmb_lambda = QComboBox()
        self._cmb_lambda.addItems(_LAMBDA_LABELS)

        model_idx, lam_idx = _SLOT_DEFAULTS[slot_index % len(_SLOT_DEFAULTS)]
        self._cmb_model.setCurrentIndex(model_idx)
        self._cmb_lambda.setCurrentIndex(lam_idx)

        self._spinner = QProgressBar()
        self._spinner.setRange(0, 0)
        self._spinner.setVisible(False)
        self._spinner.setMaximumWidth(44)
        self._spinner.setMaximumHeight(13)

        self._btn_save = QPushButton("Save…")
        self._btn_save.setEnabled(False)
        self._btn_save.setMaximumWidth(60)
        self._btn_save.setFixedHeight(22)

        self._btn_details = QPushButton("Details ▾")
        self._btn_details.setCheckable(True)
        self._btn_details.setMaximumWidth(82)
        self._btn_details.setFixedHeight(22)

        hdr = QHBoxLayout()
        hdr.setContentsMargins(2, 2, 2, 0)
        hdr.setSpacing(3)
        hdr.addWidget(self._cmb_model,  stretch=3)
        hdr.addWidget(self._cmb_lambda, stretch=3)
        hdr.addWidget(self._spinner)
        hdr.addWidget(self._btn_save)
        hdr.addWidget(self._btn_details)

        # ── Viewer + metrics — metrics embedded inside viewer ─────────────────
        self._viewer  = ImageViewer()
        self._metrics = MetricsBar()
        self._viewer.embed_detail(self._metrics)

        vlay = QVBoxLayout(self)
        vlay.setContentsMargins(2, 2, 2, 2)
        vlay.setSpacing(2)
        vlay.addLayout(hdr)
        vlay.addWidget(self._viewer, stretch=1)

        self._cmb_model.currentIndexChanged.connect(self._update_title)
        self._cmb_lambda.currentIndexChanged.connect(self._update_title)
        self._btn_details.toggled.connect(self._toggle_details)
        self._btn_save.clicked.connect(self._on_save)
        self._update_title()

    # ── Public ───────────────────────────────────────────────────────────────

    def selected_cfg(self) -> Optional[ModelConfig]:
        lam    = _LAMBDA_VALUES[self._cmb_lambda.currentIndex()]
        loader = _LOADER_KEYS[self._cmb_model.currentIndex()]
        return find_cfg(lam, loader)

    def start_compress(self, image_path: str) -> None:
        cfg = self.selected_cfg()
        if cfg is None or not cfg.checkpoint.exists():
            name = _MODEL_LABELS[self._cmb_model.currentIndex()]
            lam  = _LAMBDA_VALUES[self._cmb_lambda.currentIndex()]
            self.compression_error.emit(self, f"{name} — no checkpoint for λ={lam}")
            return
        self._launch(self._make_compress_task(cfg, image_path))

    def start_decompress(self, cfg: ModelConfig, lic_path: str) -> None:
        self._launch(self._make_decompress_task(cfg, lic_path))

    def clear(self) -> None:
        self._viewer.clear("—")
        self._metrics.clear()
        self._result = None
        self._btn_save.setEnabled(False)

    # ── Title / details ───────────────────────────────────────────────────────

    def _update_title(self) -> None:
        model = _MODEL_LABELS[self._cmb_model.currentIndex()]
        lam   = _LAMBDA_SHORT[self._cmb_lambda.currentIndex()]
        # No title label inside ImageViewer; use window title or status only.
        # Show as a tooltip on the viewer instead for compactness.
        self._viewer.setToolTip(f"{model}  ·  {lam}")
        # Also set the header as a visible label-like effect via frame title approach:
        self._viewer._img_lbl.setText(f"{model}\n{lam}")

    def _toggle_details(self, checked: bool) -> None:
        self._viewer.show_detail(checked)
        self._btn_details.setText("Details ▴" if checked else "Details ▾")

    # ── Tasks ─────────────────────────────────────────────────────────────────

    def _make_compress_task(self, cfg: ModelConfig, image_path: str):
        def task():
            if self._codec is None or self._codec.cfg.key != cfg.key:
                if self._codec:
                    self._codec.unload()
                self._codec = LocLicCodec(cfg)
                self._codec.load()
            return self._codec.compress_image(image_path)
        return task

    def _make_decompress_task(self, cfg: ModelConfig, lic_path: str):
        def task():
            if self._codec is None or self._codec.cfg.key != cfg.key:
                if self._codec:
                    self._codec.unload()
                self._codec = LocLicCodec(cfg)
                self._codec.load()
            return self._codec.decompress_lic(lic_path)
        return task

    # ── Threading ─────────────────────────────────────────────────────────────

    def _launch(self, task) -> None:
        self._set_busy(True)
        self._result = None
        self._btn_save.setEnabled(False)
        worker = _Worker(task)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_done)
        worker.error.connect(self._on_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    @pyqtSlot(dict)
    def _on_done(self, result: dict) -> None:
        self._result = result
        self._set_busy(False)
        self._viewer.set_tensor(result["x_hat"])
        deploy_mb = self._codec.cfg.deploy_size_mb if self._codec else 0.0
        self._metrics.update(
            psnr=result.get("psnr",    float("nan")),
            ms_ssim=result.get("ms_ssim", float("nan")),
            bpp=result.get("bpp",      0.0),
            file_kb=result.get("file_kb", 0.0),
            enc_ms=result.get("enc_ms", float("nan")),
            dec_ms=result.get("dec_ms", float("nan")),
            deploy_mb=deploy_mb,
        )
        self._btn_save.setEnabled(True)
        self.compression_done.emit(self)

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._set_busy(False)
        self.compression_error.emit(self, msg)

    @pyqtSlot()
    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None

    def _set_busy(self, busy: bool) -> None:
        self._spinner.setVisible(busy)
        self._cmb_model.setEnabled(not busy)
        self._cmb_lambda.setEnabled(not busy)

    # ── Save ──────────────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        if self._result is None:
            return
        path, sel_filter = QFileDialog.getSaveFileName(
            self, "Save image", "output", _SAVE_FILTERS
        )
        if not path:
            return
        if not Path(path).suffix:
            for pattern, ext in _FILTER_EXT.items():
                if pattern in sel_filter:
                    path += ext
                    break
        try:
            if Path(path).suffix.lower() == ".lic":
                if self._codec is None:
                    QMessageBox.warning(self, "Cannot save .lic",
                        "No active codec. Compress the image first.")
                    return
                self._codec.save_lic(self._result, path)
            else:
                _save_image(self._result["x_hat"], path)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LoC-LIC Image Compressor")
        self.resize(1440, 880)

        self._current_image_path: Optional[str]         = None
        self._slots:               List[ComparisonSlot] = []
        self._compress_queue:      List[ComparisonSlot] = []
        self._n_panels: int = 2

        # Codec used for 1-panel .lic decompression
        self._single_codec:  Optional[LocLicCodec] = None
        self._single_thread: Optional[QThread]     = None
        self._single_worker: Optional[_Worker]     = None

        self._build_ui()
        self._set_layout(2)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        tb = QToolBar("Controls")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)

        self._btn_open     = QPushButton("Open Image")
        self._btn_open_lic = QPushButton("Open .lic")
        self._btn_clear    = QPushButton("✕ Clear")
        self._btn_clear.setEnabled(False)
        self._btn_compress = QPushButton("⚡ Compress All")
        self._btn_compress.setEnabled(False)

        self._lyt_group = QButtonGroup(self)
        self._lyt_group.setExclusive(True)
        self._btn_lyt: dict[int, QPushButton] = {}
        for n in (1, 2, 3, 4):
            btn = QPushButton(str(n))
            btn.setCheckable(True)
            btn.setFixedWidth(28)
            self._btn_lyt[n] = btn
            self._lyt_group.addButton(btn, n)
        self._btn_lyt[2].setChecked(True)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        self._progress.setMaximumWidth(80)

        for w in [
            self._btn_open, self._btn_open_lic, self._btn_clear,
            _sep(),
            QLabel("  Panels:"),
            self._btn_lyt[1], self._btn_lyt[2],
            self._btn_lyt[3], self._btn_lyt[4],
            _sep(),
            self._btn_compress, self._progress,
        ]:
            tb.addWidget(w)

        central = QWidget()
        self.setCentralWidget(central)
        self._grid = QGridLayout(central)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setSpacing(6)

        self._orig = OriginalPanel()
        self._grid.addWidget(self._orig, 0, 0)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(f"Ready — {_device_name()}")

        self._btn_open.clicked.connect(self._on_open_image)
        self._btn_open_lic.clicked.connect(self._on_open_lic)
        self._btn_clear.clicked.connect(self._on_clear)
        self._btn_compress.clicked.connect(self._on_compress_all)
        self._lyt_group.idClicked.connect(self._set_layout)

    # ── Layout management ─────────────────────────────────────────────────────

    def _set_layout(self, n: int) -> None:
        self._n_panels = n

        for slot in self._slots:
            self._grid.removeWidget(slot)
            slot.setParent(None)
            slot.deleteLater()
        self._slots.clear()
        self._compress_queue.clear()

        for i in range(4):
            self._grid.setRowStretch(i, 0)
            self._grid.setColumnStretch(i, 0)

        if n == 1:
            # Original panel fills everything
            self._grid.setRowStretch(0, 1)
            self._grid.setColumnStretch(0, 1)
            self._btn_compress.setEnabled(False)
        else:
            rows = 2 if n >= 3 else 1
            for r in range(rows):
                self._grid.setRowStretch(r, 1)
            for c in range(2):
                self._grid.setColumnStretch(c, 1)
            for i, (row, col) in enumerate(_SLOT_POSITIONS[n]):
                slot = ComparisonSlot(slot_index=i)
                slot.compression_done.connect(self._on_slot_done)
                slot.compression_error.connect(self._on_slot_error)
                self._grid.addWidget(slot, row, col)
                self._slots.append(slot)
            self._btn_compress.setEnabled(bool(self._current_image_path))

        self._btn_lyt[n].setChecked(True)
        self._status.showMessage(
            f"{n}-panel layout" +
            (" — open an image, then ⚡ Compress All" if n > 1 else " — open an image or .lic to view")
        )

    # ── File operations ───────────────────────────────────────────────────────

    def _on_open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)",
        )
        if not path:
            return
        self._current_image_path = path
        self._orig.set_file(path)
        for slot in self._slots:
            slot.clear()
        self._btn_clear.setEnabled(True)
        self._btn_compress.setEnabled(self._n_panels > 1)
        self._status.showMessage(f"Loaded: {Path(path).name}")

    def _on_open_lic(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open .lic File", "", "LIC Files (*.lic);;All Files (*)"
        )
        if not path:
            return
        if lic_info is None:
            QMessageBox.critical(self, "Error", "lic_format module not available.")
            return
        try:
            info = lic_info(path)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot read .lic", str(exc))
            return

        lam    = info["lambda"]
        qtype  = info["quant_type"]
        loader = {"fp32": "fp32", "int8_gptq": "gptq_ptq",
                  "qat_best": "mixed_qat"}.get(qtype, "fp32")
        cfg = find_cfg(lam, loader) or find_cfg(lam, "fp32")
        if cfg is None:
            QMessageBox.critical(self, "No model",
                f"No checkpoint for λ={lam}.")
            return

        # Always switch to 1-panel mode for .lic viewing
        self._set_layout(1)
        self._decode_lic_into_orig(cfg, path)
        self._btn_clear.setEnabled(True)
        self._status.showMessage(f"Decompressing .lic  (λ={lam}, {qtype})…")

    def _decode_lic_into_orig(self, cfg: ModelConfig, lic_path: str) -> None:
        """1-panel mode: decode .lic and show result in original panel."""
        def task():
            if self._single_codec is None or self._single_codec.cfg.key != cfg.key:
                if self._single_codec:
                    self._single_codec.unload()
                self._single_codec = LocLicCodec(cfg)
                self._single_codec.load()
            return self._single_codec.decompress_lic(lic_path)

        worker = _Worker(task)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_single_done)
        worker.error.connect(self._on_single_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_single_thread_finished)
        self._single_thread = thread
        self._single_worker = worker  # keep alive until thread finishes
        thread.start()

    @pyqtSlot(dict)
    def _on_single_done(self, result: dict) -> None:
        self._orig.set_lic_result(result)   # also auto-shows details + sets title
        bpp = result.get("bpp", 0.0)
        self._status.showMessage(
            f"Decoded .lic — BPP {bpp:.4f}  {result.get('file_kb', 0):.1f} KB"
        )

    @pyqtSlot(str)
    def _on_single_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Decode error", msg)
        self._status.showMessage(f"Error: {msg[:100]}")

    @pyqtSlot()
    def _on_single_thread_finished(self) -> None:
        self._single_thread = None
        self._single_worker = None

    # ── Clear ─────────────────────────────────────────────────────────────────

    def _on_clear(self) -> None:
        self._current_image_path = None
        self._orig.clear()
        for slot in self._slots:
            slot.clear()
        self._btn_clear.setEnabled(False)
        self._btn_compress.setEnabled(False)
        self._status.showMessage("Cleared.")

    # ── Compression queue ─────────────────────────────────────────────────────

    def _on_compress_all(self) -> None:
        if not self._current_image_path:
            return
        self._compress_queue = list(self._slots)
        self._btn_compress.setEnabled(False)
        self._progress.setVisible(True)
        self._advance_queue()

    def _advance_queue(self) -> None:
        if not self._compress_queue:
            self._progress.setVisible(False)
            self._btn_compress.setEnabled(bool(self._current_image_path))
            self._status.showMessage("All done.")
            return
        slot = self._compress_queue[0]
        idx  = self._slots.index(slot) + 1
        self._status.showMessage(f"Compressing panel {idx} / {len(self._slots)}…")
        slot.start_compress(self._current_image_path)

    @pyqtSlot(object)
    def _on_slot_done(self, slot) -> None:
        if slot in self._compress_queue:
            self._compress_queue.remove(slot)
        result = slot._result
        if result:
            idx   = self._slots.index(slot) + 1
            psnr  = result.get("psnr", float("nan"))
            bpp   = result.get("bpp", 0.0)
            ratio = (24.0 / bpp) if bpp > 0 else 0.0
            psnrs = f"PSNR {psnr:.2f} dB  " if math.isfinite(psnr) else ""
            self._status.showMessage(
                f"Panel {idx} — {psnrs}BPP {bpp:.4f}  Ratio {ratio:.0f}×"
            )
        self._advance_queue()

    @pyqtSlot(object, str)
    def _on_slot_error(self, slot, msg: str) -> None:
        if slot in self._compress_queue:
            self._compress_queue.remove(slot)
        idx = (self._slots.index(slot) + 1) if slot in self._slots else "?"
        QMessageBox.warning(self, f"Panel {idx} — skipped", msg)
        self._advance_queue()



# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_image(x_hat: torch.Tensor, path: str) -> None:
    arr = (x_hat.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    img = PilImage.fromarray(arr, mode="RGB")
    suffix = Path(path).suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        img.save(path, format="JPEG", quality=95, subsampling=0)
    elif suffix == ".webp":
        img.save(path, format="WEBP", quality=95)
    elif suffix == ".bmp":
        img.save(path, format="BMP")
    elif suffix in (".tif", ".tiff"):
        img.save(path, format="TIFF")
    else:
        img.save(path, format="PNG")


def _sep() -> QWidget:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def _device_name() -> str:
    if torch.cuda.is_available():
        return f"CUDA — {torch.cuda.get_device_name(0)}"
    return "CPU"
