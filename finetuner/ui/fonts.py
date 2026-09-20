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
    installed = set(QFontDatabase.families())
    for candidate in ("Inter", "Geist", "SF Pro Text", "Segoe UI Variable Text", "Segoe UI"):
        if candidate in installed:
            return candidate
    return "Segoe UI"


def _family_has_weight(family: str, weight: QFont.Weight) -> bool:
    styles = QFontDatabase.styles(family)
    if not styles:
        return weight == QFont.Weight.Normal
    wanted = int(weight)
    return any(int(QFontDatabase.weight(family, style)) == wanted for style in styles)


def _apply_variable_weight(font: QFont, weight: QFont.Weight) -> None:
    numeric = int(weight)
    if numeric == int(QFont.Weight.Normal):
        return
    setter = getattr(font, "setVariableAxis", None)
    if setter is None:
        return
    try:
        tag = QFont.Tag("wght") if hasattr(QFont, "Tag") else "wght"
        setter(tag, float(numeric))
    except (TypeError, ValueError, RuntimeError):
        return


def application_font(family: str, size: int = 13, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont(family, size)
    if _family_has_weight(family, weight):
        font.setWeight(weight)
    else:
        font.setWeight(QFont.Weight.Normal)
        _apply_variable_weight(font, weight)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    font.setStyleStrategy(QFont.StyleStrategy.PreferOutline | QFont.StyleStrategy.PreferAntialias)
    return font
