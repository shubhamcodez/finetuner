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

from finetuner.core.job import ModelRunResult, ProjectConfig
from finetuner.core.project_state import ProjectAreaState, build_project_snapshot
from finetuner.workflows.executor import StageEvent


class ProjectAreaCard(QGroupBox):
    navigate_requested = Signal(str)

    def __init__(self, area_id: str, parent=None) -> None:
        super().__init__(parent)
        self.area_id = area_id
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
        self.configure = QPushButton("Configure")
        self.configure.setObjectName("SecondaryButton")
        self.configure.clicked.connect(lambda: self.navigate_requested.emit(self.area_id))
        layout.addWidget(self.configure, 0, 1, 2, 1)

    def apply(self, area: ProjectAreaState) -> None:
        self.setTitle(area.title)
        self.summary.setText(area.summary)
        if not area.included:
            self.state.setText("Available | not in active workflow")
            self.configure.setText("Open")
        elif area.ready:
            self.state.setText("Ready | in active workflow")
            self.configure.setText("Review")
        else:
            self.state.setText("Needs attention | " + "; ".join(area.issues))
            self.configure.setText("Fix")


class ProjectTab(QWidget):
    """Connective project surface; specialized tools remain independently usable."""

    navigate_requested = Signal(str)
    run_requested = Signal()

    def __init__(self, config: ProjectConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self._running = False
        self._stage_rows: dict[str, int] = {}
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
        self.workflow_name = QLabel()
        self.workflow_name.setObjectName("LogPanelTitle")
        hero_text.addWidget(self.workflow_name)
        self.readiness = QLabel()
        self.readiness.setWordWrap(True)
        hero_text.addWidget(self.readiness)
        hero_layout.addLayout(hero_text, 1)
        workflow_button = QPushButton("Edit Workflow")
        workflow_button.clicked.connect(lambda: self.navigate_requested.emit("workflow"))
        self.run_button = QPushButton("Run Active Workflow")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self.run_requested.emit)
        hero_layout.addWidget(workflow_button)
        hero_layout.addWidget(self.run_button)
        layout.addWidget(hero)

        section = QLabel("Project context")
        section.setObjectName("LogPanelTitle")
        layout.addWidget(section)
        grid = QGridLayout()
        for index, area_id in enumerate(
            ("models", "training", "distillation", "evals", "analysis", "deployment")
        ):
            card = ProjectAreaCard(area_id)
            card.navigate_requested.connect(self.navigate_requested.emit)
            self._cards[area_id] = card
            grid.addWidget(card, index // 3, index % 3)
        layout.addLayout(grid)

        pipeline_header = QHBoxLayout()
        pipeline_title = QLabel("Active workflow")
        pipeline_title.setObjectName("LogPanelTitle")
        pipeline_header.addWidget(pipeline_title)
        pipeline_header.addStretch()
        self.current_stage = QLabel("Not running")
        self.current_stage.setObjectName("MutedLabel")
        pipeline_header.addWidget(self.current_stage)
        layout.addLayout(pipeline_header)

        self.stage_table = QTableWidget(0, 5)
        self.stage_table.setHorizontalHeaderLabels(
            ["#", "Stage", "Configuration", "Depends on", "Status"]
        )
        self.stage_table.verticalHeader().setVisible(False)
        self.stage_table.setShowGrid(False)
        self.stage_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.stage_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.stage_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.stage_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.stage_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.stage_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        self.stage_table.setMinimumHeight(150)
        self.stage_table.setMaximumHeight(210)
        layout.addWidget(self.stage_table)

        outputs_header = QHBoxLayout()
        outputs_title = QLabel("Latest outputs")
        outputs_title.setObjectName("LogPanelTitle")
        outputs_header.addWidget(outputs_title)
        outputs_header.addStretch()
        results_button = QPushButton("Open Results")
        results_button.clicked.connect(lambda: self.navigate_requested.emit("results"))
        outputs_header.addWidget(results_button)
        layout.addLayout(outputs_header)
        self.outputs = QTableWidget(0, 5)
        self.outputs.setHorizontalHeaderLabels(
            ["Model", "Outcome", "Policy", "Analysis", "Deployment"]
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
        self.workflow_name.setText(snapshot.workflow_name)
        if snapshot.ready:
            self.readiness.setText(
                "Ready to run. Each tool shares this project context and contributes only when "
                "its stage is present in the active workflow."
            )
        else:
            unique = list(dict.fromkeys(issue.message for issue in snapshot.issues))
            self.readiness.setText("Before running: " + " | ".join(unique))
        self.run_button.setEnabled(snapshot.ready and not self._running)
        for area in snapshot.areas:
            self._cards[area.area_id].apply(area)
        self._stage_rows.clear()
        self.stage_table.setRowCount(len(snapshot.stages))
        for row, stage in enumerate(snapshot.stages):
            self._stage_rows[stage.stage_id] = row
            values = (
                str(row + 1),
                stage.name,
                stage.summary,
                ", ".join(stage.depends_on) or "-",
                "Pending",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (0, 4):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.stage_table.setItem(row, column, item)
        self._refresh_outputs()

    def set_running(self, running: bool) -> None:
        self._running = running
        if running:
            for row in range(self.stage_table.rowCount()):
                self.stage_table.item(row, 4).setText("Pending")
            self.current_stage.setText("Preparing run...")
            self.run_button.setEnabled(False)
        else:
            self.current_stage.setText("Run complete" if self._results else "Not running")
            self.run_button.setEnabled(build_project_snapshot(self.config).ready)

    def handle_stage_event(self, event: StageEvent) -> None:
        row = self._stage_rows.get(event.stage_id)
        if row is None:
            return
        labels = {"running": "Running", "completed": "Complete", "failed": "Failed"}
        self.stage_table.item(row, 4).setText(labels.get(event.status, event.status.title()))
        subject = f"{event.subject} | " if event.subject else ""
        self.current_stage.setText(
            f"{subject}{event.stage_name} ({event.index}/{event.total}) | {event.status}"
        )
        if event.status == "completed" and event.metrics:
            metrics = ", ".join(f"{key} {value:.2f}" for key, value in event.metrics.items())
            self.stage_table.item(row, 2).setToolTip(metrics)
        if event.status == "failed" and event.message:
            self.stage_table.item(row, 4).setToolTip(event.message)

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
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if result.training_error:
                    item.setToolTip(result.training_error)
                elif column > 1:
                    paths = (result.output_path, result.analysis_path, result.deployment_path)
                    if paths[column - 2]:
                        item.setToolTip(paths[column - 2])
                self.outputs.setItem(row, column, item)
