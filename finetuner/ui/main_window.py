from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from finetuner.core.config_store import load_config, save_config
from finetuner.core.job import ModelRunResult, ProjectConfig
from finetuner.core.project_state import build_project_snapshot
from finetuner.ui.branding import app_icon
from finetuner.ui.analysis_tab import AnalysisTab
from finetuner.ui.deployment_tab import DeploymentTab
from finetuner.ui.distillation_tab import DistillationTab
from finetuner.ui.evals_tab import EvalsTab
from finetuner.ui.models_tab import ModelsTab
from finetuner.ui.monitor_tab import MonitorTab
from finetuner.ui.project_tab import ProjectTab
from finetuner.ui.results_tab import ResultsTab
from finetuner.ui.training_tab import TrainingTab
from finetuner.ui.workflows_tab import WorkflowsTab
from finetuner.workflows.schema import StageKind

if TYPE_CHECKING:
    from finetuner.core.queue import JobQueue


class QueueWorker(QThread):
    log_line = Signal(str)
    progress = Signal(str, int, int)
    download_progress = Signal(int, str)
    model_done = Signal(object)
    finished_all = Signal(object)
    stage_event = Signal(object)

    def __init__(self, config: ProjectConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self._queue: JobQueue | None = None

    def cancel(self) -> None:
        if self._queue:
            self._queue.cancel()

    def run(self) -> None:
        from finetuner.core.queue import JobQueue

        self._queue = JobQueue(
            config=self.config,
            log_callback=lambda msg: self.log_line.emit(msg),
            progress_callback=lambda phase, cur, total: self.progress.emit(phase, cur, total),
            model_done_callback=lambda r: self.model_done.emit(r),
            download_progress_callback=lambda pct, desc: self.download_progress.emit(pct, desc),
            stage_event_callback=lambda event: self.stage_event.emit(event),
        )
        results = self._queue.run()
        self.finished_all.emit(results)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Finetuner")
        self.setWindowIcon(app_icon())
        self.resize(1180, 720)
        self.setMinimumSize(960, 560)

        self.config = load_config()
        self._worker: QueueWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 6, 10, 8)
        main_layout.setSpacing(6)

        main_layout.addWidget(self._build_header())

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.project_tab = ProjectTab(self.config)
        self.models_tab = ModelsTab(self.config)
        self.training_tab = TrainingTab(self.config)
        self.workflows_tab = WorkflowsTab(self.config)
        self.distillation_tab = DistillationTab(self.config)
        self.deployment_tab = DeploymentTab(self.config)
        self.evals_tab = EvalsTab(self.config)
        self.analysis_tab = AnalysisTab(self.config)
        self.results_tab = ResultsTab()
        self.monitor_tab = MonitorTab()

        self.tabs.addTab(self.project_tab, "Project")
        self.tabs.addTab(self.models_tab, "Models")
        self.tabs.addTab(self.training_tab, "Data & Train")
        self.tabs.addTab(self.workflows_tab, "Workflow")
        self.tabs.addTab(self.distillation_tab, "Distillation")
        self.tabs.addTab(self.evals_tab, "Evaluation")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.addTab(self.deployment_tab, "Deployment")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.monitor_tab, "System")

        self._tab_by_area = {
            "project": self.project_tab,
            "models": self.models_tab,
            "training": self.training_tab,
            "workflow": self.workflows_tab,
            "distillation": self.distillation_tab,
            "evals": self.evals_tab,
            "analysis": self.analysis_tab,
            "deployment": self.deployment_tab,
            "results": self.results_tab,
            "monitor": self.monitor_tab,
        }

        splitter.addWidget(self.tabs)
        splitter.addWidget(self._build_log_panel())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter, stretch=1)

        for tab in (
            self.models_tab,
            self.training_tab,
            self.workflows_tab,
            self.distillation_tab,
            self.deployment_tab,
            self.evals_tab,
            self.analysis_tab,
        ):
            tab.config_changed.connect(self._on_config_changed)

        self.training_tab.evals_suggest.connect(self._on_evals_suggest)
        self.results_tab.artifact_requested.connect(self._open_result_artifact)
        self.project_tab.navigate_requested.connect(self._navigate_to)
        self.project_tab.run_requested.connect(self._start_run)
        for tab in (
            self.training_tab,
            self.distillation_tab,
            self.deployment_tab,
            self.evals_tab,
            self.analysis_tab,
        ):
            tab.pipeline_context.workflow_requested.connect(lambda: self._navigate_to("workflow"))
        self._refresh_project_context()

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("AppHeader")
        header.setFixedHeight(32)

        row = QHBoxLayout(header)
        row.setContentsMargins(10, 0, 10, 0)
        row.setSpacing(8)
        row.addStretch()

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusBadge")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self.status_label)

        return header

    def _build_log_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("LogPanel")
        log_layout = QVBoxLayout(panel)
        log_layout.setContentsMargins(8, 6, 8, 6)
        log_layout.setSpacing(4)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        title = QLabel("Run Console")
        title.setObjectName("LogPanelTitle")
        toolbar.addWidget(title)
        toolbar.addStretch()

        self.run_btn = QPushButton("Start Run")
        self.run_btn.setObjectName("PrimaryButton")
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_btn.clicked.connect(self._start_run)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("SecondaryButton")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.clicked.connect(self._cancel_run)

        toolbar.addWidget(self.run_btn)
        toolbar.addWidget(self.cancel_btn)
        log_layout.addLayout(toolbar)

        self.run_progress = QProgressBar()
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(0)
        self.run_progress.setVisible(False)
        self.run_progress.setTextVisible(True)
        self.run_progress.setFixedHeight(8)
        log_layout.addWidget(self.run_progress)

        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("LogConsole")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setPlaceholderText("Run logs will appear here...")
        self.log_view.setMinimumHeight(72)
        log_layout.addWidget(self.log_view, stretch=1)

        return panel

    def _save_config(self) -> None:
        save_config(self.config)

    def _on_config_changed(self) -> None:
        self._save_config()
        self.project_tab.refresh()
        self._refresh_project_context()

    def _navigate_to(self, area: str) -> None:
        target = self._tab_by_area.get(area)
        if target is not None:
            self.tabs.setCurrentWidget(target)

    def _open_result_artifact(self, area: str, path: str) -> None:
        if area == "analysis":
            self.analysis_tab.set_artifact(path)
        self._navigate_to(area)

    def _refresh_project_context(self) -> None:
        stages_by_kind: dict[StageKind, list[str]] = {kind: [] for kind in StageKind}
        for stage in self.config.workflow.topological_stages():
            stages_by_kind[stage.kind].append(stage.name)
        workflow_name = self.config.workflow.name
        data_consumers = [
            stage.name
            for stage in self.config.workflow.topological_stages()
            if stage.kind in {StageKind.TRAIN, StageKind.DISTILL, StageKind.ANALYZE}
        ]
        self.training_tab.pipeline_context.set_context(workflow_name, data_consumers)
        self.distillation_tab.pipeline_context.set_context(
            workflow_name, stages_by_kind[StageKind.DISTILL]
        )
        self.evals_tab.pipeline_context.set_context(
            workflow_name, stages_by_kind[StageKind.EVALUATE]
        )
        self.analysis_tab.pipeline_context.set_context(
            workflow_name, stages_by_kind[StageKind.ANALYZE]
        )
        self.deployment_tab.pipeline_context.set_context(
            workflow_name, stages_by_kind[StageKind.QUANTIZE]
        )

    def _on_evals_suggest(self, eval_ids: list[str]) -> None:
        self.evals_tab.apply_selection(eval_ids)
        self._on_config_changed()

    def _append_log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)

    def _start_run(self) -> None:
        snapshot = build_project_snapshot(self.config)
        if not snapshot.ready:
            messages = list(dict.fromkeys(issue.message for issue in snapshot.issues))
            QMessageBox.warning(
                self,
                "Project Not Ready",
                "Resolve these project items before running:\n\n" + "\n".join(messages),
            )
            first_area = snapshot.issues[0].area if snapshot.issues else "project"
            self._navigate_to(first_area)
            return
        if self._worker and self._worker.isRunning():
            return

        self._save_config()
        self.log_view.clear()
        self.results_tab.set_results([])
        self.project_tab.clear_results()
        self.project_tab.set_running(True)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status_label.setText("Running")
        self._append_log(f"Starting workflow: {self.config.workflow.name}")

        self._worker = QueueWorker(self.config, self)
        self._worker.log_line.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.download_progress.connect(self._on_download_progress)
        self._worker.model_done.connect(self._on_model_done)
        self._worker.stage_event.connect(self._on_stage_event)
        self._worker.finished_all.connect(self._on_finished)
        self._worker.start()

    def _cancel_run(self) -> None:
        if self._worker:
            self._worker.cancel()
            self.status_label.setText("Cancelling")

    def _on_progress(self, phase: str, current: int, total: int) -> None:
        self.run_progress.setVisible(False)
        self.status_label.setText(f"{phase.title()} {current}/{total}")

    def _on_download_progress(self, percent: int, desc: str) -> None:
        self.run_progress.setVisible(True)
        self.run_progress.setValue(percent)
        self.status_label.setText(f"Download {percent}%")

    def _on_stage_event(self, event) -> None:
        self.project_tab.handle_stage_event(event)
        self.run_progress.setVisible(True)
        complete = event.index if event.status == "completed" else event.index - 1
        self.run_progress.setValue(round(complete / max(event.total, 1) * 100))
        self.status_label.setText(f"{event.stage_name} {event.index}/{event.total}")

    def _on_model_done(self, result: ModelRunResult) -> None:
        self.results_tab.add_result(result)
        self.project_tab.add_result(result)
        if result.analysis_path:
            self.analysis_tab.set_artifact(result.analysis_path)

    def _on_finished(self, _results: list) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.run_progress.setVisible(False)
        self.run_progress.setValue(0)
        self.status_label.setText("Complete")
        self.project_tab.set_running(False)
        self._append_log("All models processed.")

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Run in progress",
                "A training run is in progress. Cancel and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._worker.cancel()
            self._worker.wait(5000)

        self.monitor_tab.shutdown()
        self._save_config()
        super().closeEvent(event)
