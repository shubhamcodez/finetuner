from __future__ import annotations

import pytest
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QLabel

from finetuner.ui.theme import Token, apply_theme


@pytest.fixture(scope="module")
def themed_app():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    return app


@pytest.mark.ui
def test_labels_do_not_paint_an_opaque_background(themed_app):
    sheet = themed_app.styleSheet()
    assert "QMainWindow, QWidget" not in sheet
    assert "QLabel {\n    background-color: transparent;" in sheet.replace("\r\n", "\n")

    label = QLabel("System status")
    color = label.palette().color(label.backgroundRole())
    assert color.alpha() == 0 or color == QColor(Token.BG)
