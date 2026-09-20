from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton


class ToolRunBar(QFrame):
    """Primary action for a tool page."""

    run_requested = Signal()

    def __init__(self, action_label: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("PipelineContext")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 8)
        row.setSpacing(12)
        self.label = QLabel("Uses this page's settings. Other tools keep running independently.")
        self.label.setObjectName("MutedLabel")
        row.addWidget(self.label, 1)
        self.button = QPushButton(action_label)
        self.button.setObjectName("PrimaryButton")
        self.button.clicked.connect(self.run_requested.emit)
        row.addWidget(self.button)

    def set_enabled(self, enabled: bool) -> None:
        self.button.setEnabled(enabled)
