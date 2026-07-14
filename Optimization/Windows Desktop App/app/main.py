#!/usr/bin/env python3
"""
LoC-LIC Desktop App — entry point.

Run from repo root:
    python -m "Optimization.Windows Desktop App.app.main"

Or directly:
    python "Optimization/Windows Desktop App/app/main.py"
"""

from __future__ import annotations
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

# Support both package (-m) and plain-script invocation.
try:
    from .ui import MainWindow           # package mode: python -m ...
except ImportError:
    _app_dir = Path(__file__).parent
    _repo    = _app_dir.parents[2]
    for _p in (str(_repo), str(_app_dir)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from ui import MainWindow            # type: ignore[no-redef]  # script mode


def _dark_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor(45,  45,  48))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Base,            QColor(30,  30,  30))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(53,  53,  53))
    p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(255, 255, 220))
    p.setColor(QPalette.ColorRole.ToolTipText,     QColor(0,   0,   0))
    p.setColor(QPalette.ColorRole.Text,            QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.Button,          QColor(53,  53,  53))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(220, 220, 220))
    p.setColor(QPalette.ColorRole.BrightText,      QColor(255, 80,  80))
    p.setColor(QPalette.ColorRole.Link,            QColor(42,  130, 218))
    p.setColor(QPalette.ColorRole.Highlight,       QColor(42,  130, 218))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    return p


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("LoC-LIC Compressor")
    app.setOrganizationName("LoC-LIC")
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
