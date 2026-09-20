from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from finetuner.core.actions import ActionEvent
from finetuner.core.job import ModelRunResult, ProjectConfig
from finetuner.core.project_state import ProjectAreaState, build_project_snapshot


class ProjectAreaCard(QGroupBox):
    navigate_requested = Signal(str)
    run_requested = Signal(str)

    def __init__(self, area_id: str, parent=None) -> None:
        super().__init__(parent)
        self.area_id = area_id
        self._action = ""
        layout = QGridLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(2)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary, 0, 0)
        self.state = QLabel()
        self.state.setObjectName("MutedLabel")
        self.state.setWordWrap(True)
        layout.addWidget(self.state, 1, 0)
        buttons = QVBoxLayout()
        self.configure = QPushButton("Open")
        self.configure.setObjectName("SecondaryButton")
        self.configure.clicked.connect(lambda: self.navigate_requested.emit(self.area_id))
        self.run = QPushButton("Run")
        self.run.setObjectName("PrimaryButton")
        self.run.clicked.connect(lambda: self.run_requested.emit(self._action))
        buttons.addWidget(self.configure)
        buttons.addWidget(self.run)
        layout.addLayout(buttons, 0, 1, 2, 1)

    def apply(self, area: ProjectAreaState, running: bool) -> None:
        self.setTitle(area.title)
        self.summary.setText(area.summary)
        self._action = area.action
        self.run.setVisible(bool(area.action))
        self.run.setEnabled(bool(area.action) and area.ready and not running)
        if area.ready:
            self.state.setText("Ready to run" if area.action else "Ready")
            self.configure.setText("Open")
        else:
            self.state.setText("Needs attention | " + "; ".join(area.issues))
            self.configure.setText("Fix")


class ProjectTab(QWidget):
    """Overview of independent tools that share queued models and optional data."""

    navigate_requested = Signal(str)
    run_requested = Signal(str)

    def __init__(self, config: ProjectConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self._running = False
        self._results: list[ModelRunResult] = []
        self._cards: dict[str, ProjectAreaCard] = {}
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
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        hero = QFrame()
        hero.setObjectName("SummaryBanner")
        hero_layout = QHBoxLayout(hero)
        hero_text = QVBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("LogPanelTitle")
        hero_text.addWidget(self.title)
        self.readiness = QLabel()
        self.readiness.setWordWrap(True)
        hero_text.addWidget(self.readiness)
        hero_layout.addLayout(hero_text, 1)
        layout.addWidget(hero)

        section = QLabel("Tools")
        section.setObjectName("LogPanelTitle")
        layout.addWidget(section)
        grid = QGridLayout()
        for index, area_id in enumerate(
            (
                "models",
                "training",
                "distillation",
                "evals",
                "analysis",
                "deployment",
                "inference",
            )
        ):
            card = ProjectAreaCard(area_id)
            card.navigate_requested.connect(self.navigate_requested.emit)
            card.run_requested.connect(self.run_requested.emit)
            self._cards[area_id] = card
            grid.addWidget(card, index // 3, index % 3)
        layout.addLayout(grid)

        status_header = QHBoxLayout()
        status_title = QLabel("Current run")
        status_title.setObjectName("LogPanelTitle")
        status_header.addWidget(status_title)
        status_header.addStretch()
        self.current_stage = QLabel("Not running")
        self.current_stage.setObjectName("MutedLabel")
        status_header.addWidget(self.current_stage)
        layout.addLayout(status_header)

        outputs_header = QHBoxLayout()
        outputs_title = QLabel("Latest outputs")
        outputs_title.setObjectName("LogPanelTitle")
        outputs_header.addWidget(outputs_title)
        outputs_header.addStretch()
        results_button = QPushButton("Open Results")
        results_button.clicked.connect(lambda: self.navigate_requested.emit("results"))
        outputs_header.addWidget(results_button)
        layout.addLayout(outputs_header)
        self.outputs = QTableWidget(0, 6)
        self.outputs.setHorizontalHeaderLabels(
            ["Model", "Outcome", "Policy", "Analysis", "Deployment", "Inference"]
        )
        self.outputs.verticalHeader().setVisible(False)
        self.outputs.setShowGrid(False)
        self.outputs.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.outputs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.outputs.setMinimumHeight(100)
        self.outputs.setMaximumHeight(150)
        layout.addWidget(self.outputs)
        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def refresh(self) -> None:
        snapshot = build_project_snapshot(self.config)
        self.title.setText(snapshot.title)
        ready_tools = [area for area in snapshot.areas if area.action and area.ready]
        if ready_tools:
            names = ", ".join(area.title for area in ready_tools)
            self.readiness.setText(
                f"{len(ready_tools)} tool{'s' if len(ready_tools) != 1 else ''} ready: {names}. "
                "Run any of them from its card or its page."
            )
        else:
            unique = list(dict.fromkeys(issue.message for issue in snapshot.issues))
            self.readiness.setText("Configure a tool before running: " + " | ".join(unique))
        for area in snapshot.areas:
            self._cards[area.area_id].apply(area, self._running)
        self._refresh_outputs()

    def set_running(self, running: bool) -> None:
        self._running = running
        if running:
            self.current_stage.setText("Preparing run...")
        else:
            self.current_stage.setText("Run complete" if self._results else "Not running")
        self.refresh()

    def handle_action_event(self, event: ActionEvent) -> None:
        labels = {"running": "Running", "completed": "Complete", "failed": "Failed"}
        subject = f"{event.subject} | " if event.subject else ""
        self.current_stage.setText(
            f"{subject}{event.action_name} ({event.index}/{event.total}) | "
            f"{labels.get(event.status, event.status.title())}"
        )

    def clear_results(self) -> None:
        self._results.clear()
        self._refresh_outputs()

    def add_result(self, result: ModelRunResult) -> None:
        self._results.append(result)
        self._refresh_outputs()

    def _refresh_outputs(self) -> None:
        self.outputs.setRowCount(len(self._results))
        for row, result in enumerate(self._results):
            outcome = "Failed" if result.training_error else "Complete"
            values = (
                result.model_name,
                outcome,
                "Ready" if result.output_path else "-",
                "Ready" if result.analysis_path else "-",
                "Ready" if result.deployment_path else "-",
                "Ready" if result.inference_path else "-",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if result.training_error:
                    item.setToolTip(result.training_error)
                elif column > 1:
                    paths = (
                        result.output_path,
                        result.analysis_path,
                        result.deployment_path,
                        result.inference_path,
                    )
                    if paths[column - 2]:
                        item.setToolTip(paths[column - 2])
                self.outputs.setItem(row, column, item)
