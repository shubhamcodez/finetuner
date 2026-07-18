from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton


class PipelineContextBar(QFrame):
    """A compact reminder of how an independently usable tool fits the active workflow."""

    workflow_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("PipelineContext")
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 3, 4, 3)
        row.setSpacing(6)
        self.label = QLabel()
        self.label.setObjectName("MutedLabel")
        self.label.setWordWrap(False)
        row.addWidget(self.label, 1)
        button = QPushButton("Workflow")
        button.setObjectName("ContextButton")
        button.clicked.connect(self.workflow_requested.emit)
        row.addWidget(button)

    def set_context(self, workflow_name: str, stage_names: list[str]) -> None:
        if stage_names:
            if len(stage_names) <= 2:
                compact_stages = ", ".join(stage_names)
            else:
                compact_stages = f"{stage_names[0]} +{len(stage_names) - 1} stages"
            self.label.setText(f"{workflow_name}  |  {compact_stages}")
            self.label.setToolTip(
                "This page supplies shared project settings to: " + ", ".join(stage_names)
            )
        else:
            self.label.setText(f"Available independently  |  not used by {workflow_name}")
            self.label.setToolTip(
                "You can configure this tool now and add its stage to the workflow later."
            )
