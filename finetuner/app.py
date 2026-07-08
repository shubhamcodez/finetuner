from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from finetuner.ui.branding import app_icon, configure_platform_app_id, logo_pixmap
from finetuner.ui.theme import Theme, apply_theme
from finetuner.datasets.hf_datasets import load_env_file


def _make_splash() -> QSplashScreen:
    pixmap = QPixmap(520, 180)
    pixmap.fill(QColor(Theme.BG))
    painter = QPainter(pixmap)

    logo = logo_pixmap(56)
    text_x = 40
    if logo is not None:
        painter.drawPixmap(40, 36, logo)
        text_x = 40 + logo.width() + 16

    painter.setPen(QColor(Theme.TEXT))
    painter.setFont(QFont(Theme.FONT_FAMILY, 22, QFont.Weight.DemiBold))
    painter.drawText(text_x, 70, "Finetuner")
    painter.setPen(QColor(Theme.TEXT_SECONDARY))
    painter.setFont(QFont(Theme.FONT_FAMILY, 11))
    painter.drawText(text_x, 98, "GPU LLM Fine-tuning Platform")
    painter.setPen(QColor(Theme.PRIMARY))
    painter.setBrush(QColor(Theme.PRIMARY))
    painter.drawRoundedRect(text_x, 120, 120, 3, 2, 2)
    painter.end()
    splash = QSplashScreen(pixmap)
    splash.show()
    return splash


def main() -> None:
    configure_platform_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName("Finetuner")
    app.setOrganizationName("Finetuner")
    app.setWindowIcon(app_icon())
    apply_theme(app)
    load_env_file()

    splash = _make_splash()
    splash.showMessage(
        "Initializing workspace…",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
        QColor(Theme.TEXT_SECONDARY),
    )
    app.processEvents()

    from finetuner.ui.main_window import MainWindow

    window = MainWindow()
    splash.finish(window)
    window.show()
    window.raise_()
    window.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
