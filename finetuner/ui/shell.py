from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from finetuner.ui.branding import logo_pixmap
from finetuner.ui.icons import line_icon
from finetuner.ui.theme import Token

PRIMARY_NAV = (
    ("project", "Project", "project"),
    ("models", "Models", "models"),
    ("data", "Data", "data"),
    ("training", "Training", "training"),
    ("distillation", "Distillation", "distillation"),
    ("evals", "Evaluation", "evaluation"),
    ("analysis", "Analysis", "analysis"),
    ("deployment", "Deployment", "deployment"),
    ("inference", "Inference", "inference"),
    ("results", "Results", "results"),
    ("monitor", "System", "system"),
)

BOTTOM_NAV = (
    ("docs", "Documentation", "docs"),
    ("settings", "Settings", "settings"),
)

PAGE_COPY = {
    "project": ("PROJECT", "Build, evaluate, and deploy your model", "A seamless workflow for modern model development."),
    "models": ("MODELS", "Model registry", "Queue local or Hugging Face checkpoints for every tool."),
    "data": ("DATA", "Datasets", "Choose a preset, local JSONL, or Hugging Face dataset."),
    "training": ("TRAINING", "Train model", "Fine-tune a queued model with the selected method and data."),
    "distillation": ("DISTILLATION", "Distill a teacher into a student", "Transfer behavior from a larger model into a smaller one."),
    "evals": ("EVALUATION", "Evaluate models", "Score queued models on shared benchmarks."),
    "analysis": ("ANALYSIS", "Analyze representations", "Inspect hidden states, activations, and layer similarity."),
    "deployment": ("DEPLOYMENT", "Quantize for a device", "Compress a model for a concrete backend and accelerator."),
    "inference": ("INFERENCE", "Optimize and serve", "Pick the strongest engine for this machine and bind port 1234."),
    "results": ("RESULTS", "Recent runs", "Compare artifacts and metrics from the latest tool runs."),
    "monitor": ("SYSTEM", "System status", "Compute, storage, and process health for this workstation."),
    "docs": ("DOCUMENTATION", "Documentation", "How the workbench is meant to be used."),
    "settings": ("SETTINGS", "Settings", "Workspace defaults and credentials."),
}


class NavButton(QPushButton):
    def __init__(self, area: str, label: str, icon_name: str, parent=None) -> None:
        super().__init__(label, parent)
        self.area = area
        self.icon_name = icon_name
        self.setObjectName("NavItem")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setCheckable(False)
        self.setIcon(line_icon(icon_name, Token.TEXT_SECONDARY, 18))
        self.setIconSize(QSize(18, 18))

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        color = Token.ACCENT if selected else Token.TEXT_SECONDARY
        self.setIcon(line_icon(self.icon_name, color, 18))
        self.style().unpolish(self)
        self.style().polish(self)


class Sidebar(QFrame):
    navigate = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(220)
        self.buttons: dict[str, NavButton] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 20, 12, 16)
        layout.setSpacing(4)

        brand = QHBoxLayout()
        brand.setSpacing(10)
        logo = QLabel()
        pixmap = logo_pixmap(22)
        if pixmap is not None:
            logo.setPixmap(pixmap)
        title = QLabel("Finetuner")
        title.setObjectName("SidebarBrand")
        brand.addWidget(logo)
        brand.addWidget(title, 1)
        layout.addLayout(brand)
        layout.addSpacing(20)

        for area, label, icon in PRIMARY_NAV:
            button = NavButton(area, label, icon)
            button.clicked.connect(lambda _=False, chosen=area: self.navigate.emit(chosen))
            self.buttons[area] = button
            layout.addWidget(button)

        layout.addStretch(1)
        for area, label, icon in BOTTOM_NAV:
            button = NavButton(area, label, icon)
            button.clicked.connect(lambda _=False, chosen=area: self.navigate.emit(chosen))
            self.buttons[area] = button
            layout.addWidget(button)

        profile = QFrame()
        profile.setObjectName("SurfaceCard")
        row = QHBoxLayout(profile)
        row.setContentsMargins(10, 8, 10, 8)
        icon = QLabel()
        icon.setPixmap(line_icon("user", Token.TEXT_SECONDARY, 16).pixmap(16, 16))
        text = QVBoxLayout()
        text.setSpacing(0)
        name = QLabel("Local workspace")
        name.setObjectName("CardTitle")
        name.setStyleSheet("font-size: 13px;")
        path = QLabel("This machine")
        path.setObjectName("MetaLabel")
        text.addWidget(name)
        text.addWidget(path)
        row.addWidget(icon)
        row.addLayout(text, 1)
        layout.addSpacing(8)
        layout.addWidget(profile)

    def select(self, area: str) -> None:
        for key, button in self.buttons.items():
            button.set_selected(key == area)


class PageHeader(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        copy = QVBoxLayout()
        copy.setSpacing(6)
        self.eyebrow = QLabel()
        self.eyebrow.setObjectName("Eyebrow")
        self.title = QLabel()
        self.title.setObjectName("PageTitle")
        self.title.setWordWrap(True)
        self.subtitle = QLabel()
        self.subtitle.setObjectName("PageSubtitle")
        self.subtitle.setWordWrap(True)
        copy.addWidget(self.eyebrow)
        copy.addWidget(self.title)
        copy.addWidget(self.subtitle)
        layout.addLayout(copy, 1)
        actions_wrap = QWidget()
        self.actions = QHBoxLayout(actions_wrap)
        self.actions.setContentsMargins(0, 0, 0, 0)
        self.actions.setSpacing(8)
        layout.addWidget(actions_wrap, 0, Qt.AlignmentFlag.AlignTop)

    def set_page(self, area: str) -> None:
        eyebrow, title, subtitle = PAGE_COPY.get(area, ("FINETUNER", area.title(), ""))
        self.eyebrow.setText(eyebrow)
        self.title.setText(title)
        self.subtitle.setText(subtitle)


class WorkspaceBar(QFrame):
    command_requested = Signal()
    help_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addStretch()
        self.status = QLabel("Ready")
        self.status.setObjectName("StatusBadge")
        search = QPushButton()
        search.setObjectName("IconButton")
        search.setToolTip("Command palette  Ctrl+K")
        search.setIcon(line_icon("search", Token.TEXT_SECONDARY, 16))
        search.clicked.connect(self.command_requested.emit)
        help_btn = QPushButton()
        help_btn.setObjectName("IconButton")
        help_btn.setToolTip("Documentation")
        help_btn.setIcon(line_icon("help", Token.TEXT_SECONDARY, 16))
        help_btn.clicked.connect(self.help_requested.emit)
        row.addWidget(self.status)
        row.addWidget(search)
        row.addWidget(help_btn)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
