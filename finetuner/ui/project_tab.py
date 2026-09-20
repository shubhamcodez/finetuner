from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from finetuner.core.actions import ActionEvent
from finetuner.core.job import ModelRunResult, ProjectConfig
from finetuner.ui.icons import line_icon
from finetuner.ui.theme import Token

_TOOLS = (
    ("models", "Models", "models", "Queue Hugging Face and local checkpoints."),
    ("training", "Finetune", "training", "Fine-tune a queued model on selected data."),
    ("distillation", "Distill", "distillation", "Transfer a teacher into a smaller student."),
    ("evals", "Evaluate", "evaluation", "Score models on shared benchmarks."),
    ("analysis", "Analyze", "analysis", "Inspect representations and layer similarity."),
    ("deployment", "Deploy", "deployment", "Quantize for a concrete backend and device."),
    ("inference", "Optimize Inference", "inference", "Bind the strongest ready engine on port 1234."),
)


class ToolCard(QFrame):
    clicked = Signal(str)

    def __init__(self, area: str, title: str, icon: str, body: str, parent=None) -> None:
        super().__init__(parent)
        self.area = area
        self.setObjectName("ToolCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        row = QHBoxLayout(self)
        row.setContentsMargins(20, 20, 20, 20)
        row.setSpacing(16)
        mark = QLabel()
        mark.setPixmap(line_icon(icon, Token.ACCENT, 20).pixmap(20, 20))
        text = QVBoxLayout()
        text.setSpacing(4)
        name = QLabel(title)
        name.setObjectName("CardTitle")
        copy = QLabel(body)
        copy.setObjectName("HintLabel")
        copy.setWordWrap(True)
        text.addWidget(name)
        text.addWidget(copy)
        chevron = QLabel()
        chevron.setPixmap(line_icon("chevron", Token.TEXT_TERTIARY, 16).pixmap(16, 16))
        row.addWidget(mark, 0, Qt.AlignmentFlag.AlignTop)
        row.addLayout(text, 1)
        row.addWidget(chevron, 0, Qt.AlignmentFlag.AlignVCenter)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.area)
        super().mouseReleaseEvent(event)


class ProjectTab(QWidget):
    navigate_requested = Signal(str)
    run_requested = Signal(str)

    def __init__(self, config: ProjectConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self._running = False
        self._results: list[ModelRunResult] = []
        self._cards: dict[str, ToolCard] = {}
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 32)
        layout.setSpacing(24)

        section = QLabel("Tools")
        section.setObjectName("SectionTitle")
        layout.addWidget(section)
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)
        for index, (area, title, icon, body) in enumerate(_TOOLS):
            card = ToolCard(area, title, icon, body)
            card.clicked.connect(self.navigate_requested.emit)
            self._cards[area] = card
            grid.addWidget(card, index // 3, index % 3)
        layout.addLayout(grid)

        run_header = QHBoxLayout()
        run_title = QLabel("Current run")
        run_title.setObjectName("SectionTitle")
        run_header.addWidget(run_title)
        run_header.addStretch()
        self.current_stage = QLabel("Not running")
        self.current_stage.setObjectName("MutedLabel")
        run_header.addWidget(self.current_stage)
        layout.addLayout(run_header)

        outputs_title = QLabel("Recent runs")
        outputs_title.setObjectName("SectionTitle")
        layout.addWidget(outputs_title)
        self.empty = QLabel("No runs yet. Finetune a model to see runs and metrics here.")
        self.empty.setObjectName("HintLabel")
        layout.addWidget(self.empty)
        self.outputs = QTableWidget(0, 6)
        self.outputs.setHorizontalHeaderLabels(
            ["Name", "Outcome", "Policy", "Analysis", "Deployment", "Inference"]
        )
        self.outputs.verticalHeader().setVisible(False)
        self.outputs.setShowGrid(False)
        self.outputs.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.outputs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.outputs.verticalHeader().setDefaultSectionSize(46)
        self.outputs.setMinimumHeight(120)
        self.outputs.setMaximumHeight(220)
        layout.addWidget(self.outputs)
        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def refresh(self) -> None:
        self._refresh_outputs()

    def set_running(self, running: bool) -> None:
        self._running = running
        self.current_stage.setText("Running" if running else ("Completed" if self._results else "Not running"))
        self.refresh()

    def handle_action_event(self, event: ActionEvent) -> None:
        labels = {
            "running": "Running",
            "completed": "Completed",
            "failed": "Failed",
            "cancelled": "Cancelled",
        }
        subject = f"{event.subject} · " if event.subject else ""
        self.current_stage.setText(
            f"{subject}{event.action_name} ({event.index}/{event.total}) · "
            f"{labels.get(event.status, event.status.title())}"
        )

    def clear_results(self) -> None:
        self._results.clear()
        self._refresh_outputs()

    def add_result(self, result: ModelRunResult) -> None:
        self._results.append(result)
        self._refresh_outputs()

    def _refresh_outputs(self) -> None:
        empty = not self._results
        self.empty.setVisible(empty)
        self.outputs.setVisible(not empty)
        self.outputs.setRowCount(len(self._results))
        for row, result in enumerate(self._results):
            outcome = "Failed" if result.training_error else "Completed"
            values = (
                result.model_name,
                outcome,
                "Ready" if result.output_path else "—",
                "Ready" if result.analysis_path else "—",
                "Ready" if result.deployment_path else "—",
                "Ready" if result.inference_path else "—",
            )
            for column, value in enumerate(values):
                self.outputs.setItem(row, column, QTableWidgetItem(value))
