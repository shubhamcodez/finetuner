"""Shared downloaded-model menu used by the tool pages."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox

from finetuner.core.model_catalog import downloaded_model_choices, find_downloaded_model


def reload_model_combo(combo: QComboBox, jobs: list, current: str) -> None:
    choices = downloaded_model_choices(jobs)
    wanted = current.strip()
    if wanted and not any(path.casefold() == wanted.casefold() for _label, path in choices):
        resolved = find_downloaded_model(wanted)
        if resolved:
            wanted = resolved
    combo.blockSignals(True)
    combo.clear()
    for label, path in choices:
        combo.addItem(label, path)
        combo.setItemData(combo.count() - 1, path, Qt.ItemDataRole.ToolTipRole)
    index = combo.findData(wanted) if wanted else -1
    if index < 0 and combo.count():
        index = 0
    if index >= 0:
        combo.setCurrentIndex(index)
    combo.blockSignals(False)
