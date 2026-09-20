from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton


class ToolRunBar(QFrame):
    """Start this tool by itself; it does not wait on other pages."""

    run_requested = Signal()

    def __init__(self, action_label: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("PipelineContext")
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 3, 4, 3)
        row.setSpacing(6)
        self.label = QLabel("Runs independently. Uses this page's settings.")
        self.label.setObjectName("MutedLabel")
        row.addWidget(self.label, 1)
        self.button = QPushButton(action_label)
        self.button.setObjectName("PrimaryButton")
        self.button.clicked.connect(self.run_requested.emit)
        row.addWidget(self.button)

    def set_enabled(self, enabled: bool) -> None:
        self.button.setEnabled(enabled)
