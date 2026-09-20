from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap

from finetuner.core.paths import icon_path, logo_path

PRODUCT_NAME = "Inferna"
APP_USER_MODEL_ID = "Inferna.Inferna.1"

_ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def configure_platform_app_id() -> None:
    """Give Inferna its own taskbar identity on Windows (not grouped under python.exe)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        pass


def app_icon() -> QIcon:
    ico = icon_path()
    if ico.exists() and ico.suffix.lower() == ".ico":
        icon = QIcon(str(ico))
        if not icon.isNull():
            return icon

    logo = logo_path()
    if not logo.exists():
        return QIcon()

    base = QPixmap(str(logo))
    if base.isNull():
        return QIcon()

    icon = QIcon()
    for size in _ICON_SIZES:
        scaled = base.scaled(
            size,
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        icon.addPixmap(scaled)
    return icon


def logo_pixmap(height: int | None = 24, width: int | None = None) -> QPixmap | None:
    path = logo_path()
    if not path.exists():
        return None
    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        return None
    if width:
        return pixmap.scaledToWidth(
            width,
            Qt.TransformationMode.SmoothTransformation,
        )
    return pixmap.scaledToHeight(
        height or 24,
        Qt.TransformationMode.SmoothTransformation,
    )
