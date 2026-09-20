from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase

from finetuner.core.paths import bundled_assets_dir


def font_dir() -> Path:
    bundled = bundled_assets_dir() / "fonts"
    if bundled.is_dir():
        return bundled
    return Path(__file__).resolve().parents[2] / "assets" / "fonts"


def load_application_fonts() -> str:
    """Register Inter if present; otherwise use a system technical sans."""
    for name in ("InterVariable.ttf", "Inter.ttf", "Inter-Regular.ttf"):
        path = font_dir() / name
        if not path.is_file():
            continue
        font_id = QFontDatabase.addApplicationFont(str(path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            return families[0]
    installed = set(QFontDatabase().families())
    for candidate in ("Inter", "Geist", "SF Pro Text", "Segoe UI Variable Text", "Segoe UI"):
        if candidate in installed:
            return candidate
    return "Segoe UI"


def application_font(family: str, size: int = 13, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont(family, size)
    font.setWeight(weight)
    font.setHintingPreference(QFont.HintingPreference.PreferDefaultHinting)
    font.setStyleStrategy(QFont.StyleStrategy.PreferQuality)
    return font
