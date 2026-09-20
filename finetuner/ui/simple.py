"""Small layout helpers for a novice-first desktop UI."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """Advanced settings stay closed until someone asks for them."""

    def __init__(self, title: str = "Advanced", parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.toggle = QToolButton()
        self.toggle.setObjectName("AdvancedToggle")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(False)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.setText(title)
        self.toggle.toggled.connect(self._set_open)
        self.body = QWidget()
        self.body.setVisible(False)
        layout.addWidget(self.toggle)
        layout.addWidget(self.body)

    def set_body_layout(self, layout) -> None:
        self.body.setLayout(layout)

    def _set_open(self, open_: bool) -> None:
        self.body.setVisible(open_)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if open_ else Qt.ArrowType.RightArrow)


class StepCard(QFrame):
    def __init__(self, number: str, title: str, body: str, action: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("StepCard")
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 14, 16, 14)
        row.setSpacing(14)
        badge = QLabel(number)
        badge.setObjectName("StepNumber")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(32, 32)
        row.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(4)
        heading = QLabel(title)
        heading.setObjectName("StepTitle")
        copy = QLabel(body)
        copy.setObjectName("HintLabel")
        copy.setWordWrap(True)
        text.addWidget(heading)
        text.addWidget(copy)
        row.addLayout(text, 1)
        self.button = QPushButton(action)
        self.button.setObjectName("PrimaryButton")
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        row.addWidget(self.button, 0, Qt.AlignmentFlag.AlignVCenter)
