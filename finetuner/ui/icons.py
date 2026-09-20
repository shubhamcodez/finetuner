"""Phosphor icons via QtAwesome — one outline family for the whole workbench."""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtGui import QIcon

import qtawesome as qta

# Regular Phosphor outlines. Keep names stable for sidebar and tool cards.
_PHOSPHOR = {
    "project": "ph.squares-four",
    "models": "ph.cube",
    "data": "ph.database",
    "training": "ph.play-circle",
    "distillation": "ph.git-branch",
    "evaluation": "ph.chart-line",
    "analysis": "ph.share-network",
    "deployment": "ph.package",
    "inference": "ph.lightning",
    "results": "ph.list-bullets",
    "system": "ph.cpu",
    "docs": "ph.book-open",
    "settings": "ph.gear",
    "user": "ph.user-circle",
    "search": "ph.magnifying-glass",
    "help": "ph.question",
    "chevron": "ph.caret-right",
    "chevron-down": "ph.caret-down",
}


def line_icon(name: str, color: str = "#A7AFBC", size: int = 18) -> QIcon:
    glyph = _PHOSPHOR.get(name, "ph.square")
    icon = qta.icon(glyph, color=color)
    icon.pixmap(size, size)
    return icon
