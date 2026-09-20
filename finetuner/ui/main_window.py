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

from finetuner.core.actions import ActionKind, get_action
from finetuner.core.config_store import load_config, save_config
from finetuner.core.job import ModelRunResult, ProjectConfig
from finetuner.core.preflight import collect_action_issues
from finetuner.ui.branding import app_icon
from finetuner.ui.analysis_tab import AnalysisTab
from finetuner.ui.deployment_tab import DeploymentTab
from finetuner.ui.distillation_tab import DistillationTab
from finetuner.ui.inference_tab import InferenceTab
from finetuner.ui.evals_tab import EvalsTab
from finetuner.ui.models_tab import ModelsTab
from finetuner.ui.monitor_tab import MonitorTab
from finetuner.ui.project_tab import ProjectTab
from finetuner.ui.results_tab import ResultsTab
from finetuner.ui.training_tab import TrainingTab

if TYPE_CHECKING:
    from finetuner.core.queue import JobQueue


class QueueWorker(QThread):
    log_line = Signal(str)
    progress = Signal(str, int, int)
    download_progress = Signal(int, str)
    model_done = Signal(object)
    finished_all = Signal(object)
    stage_event = Signal(object)

    def __init__(self, config: ProjectConfig, action: str, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.action = action
        self._queue: JobQueue | None = None

    def cancel(self) -> None:
        if self._queue:
            self._queue.cancel()

    def run(self) -> None:
        from finetuner.core.queue import JobQueue

        self._queue = JobQueue(
            config=self.config,
            action=self.action,
            log_callback=lambda msg: self.log_line.emit(msg),
            progress_callback=lambda phase, cur, total: self.progress.emit(phase, cur, total),
            model_done_callback=lambda r: self.model_done.emit(r),
            download_progress_callback=lambda pct, desc: self.download_progress.emit(pct, desc),
            stage_event_callback=lambda event: self.stage_event.emit(event),
        )
        results = self._queue.run()
        self.finished_all.emit(results)


class ServeWorker(QThread):
    log_line = Signal(str)
    ready = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        config: ProjectConfig,
        model_path: str,
        *,
        optimize: bool,
        plan_dir: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.model_path = model_path
        self.optimize = optimize
        self.plan_dir = plan_dir

    def run(self) -> None:
        from finetuner.core.paths import runs_dir
        from finetuner.inference.serve import launch_from_plan, launch_inference_server

        try:
            if self.plan_dir:
                status = launch_from_plan(
                    self.plan_dir, self.config, log=lambda msg: self.log_line.emit(msg)
                )
            else:
                status = launch_inference_server(
                    self.model_path,
                    str(runs_dir() / "serve"),
                    self.config,
                    optimize=self.optimize,
                    log=lambda msg: self.log_line.emit(msg),
                )
            self.ready.emit(status)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Finetuner")
        self.setWindowIcon(app_icon())
        self.resize(1180, 720)
        self.setMinimumSize(960, 560)

        self.config = load_config()
        self._worker: QueueWorker | None = None
        self._serve_worker: ServeWorker | None = None

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
        self.distillation_tab = DistillationTab(self.config)
        self.deployment_tab = DeploymentTab(self.config)
        self.inference_tab = InferenceTab(self.config)
        self.evals_tab = EvalsTab(self.config)
        self.analysis_tab = AnalysisTab(self.config)
        self.results_tab = ResultsTab()
        self.monitor_tab = MonitorTab()

        self.tabs.addTab(self.project_tab, "Project")
        self.tabs.addTab(self.models_tab, "Models")
        self.tabs.addTab(self.training_tab, "Data & Train")
        self.tabs.addTab(self.distillation_tab, "Distillation")
        self.tabs.addTab(self.evals_tab, "Evaluation")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.addTab(self.deployment_tab, "Deployment")
        self.tabs.addTab(self.inference_tab, "Inference")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.monitor_tab, "System")

        self._tab_by_area = {
            "project": self.project_tab,
            "models": self.models_tab,
            "training": self.training_tab,
            "distillation": self.distillation_tab,
            "evals": self.evals_tab,
            "analysis": self.analysis_tab,
            "deployment": self.deployment_tab,
            "inference": self.inference_tab,
            "results": self.results_tab,
            "monitor": self.monitor_tab,
        }
        self._action_by_tab = {
            self.training_tab: ActionKind.TRAIN.value,
            self.distillation_tab: ActionKind.DISTILL.value,
            self.evals_tab: ActionKind.EVALUATE.value,
            self.analysis_tab: ActionKind.ANALYZE.value,
            self.deployment_tab: ActionKind.QUANTIZE.value,
            self.inference_tab: ActionKind.OPTIMIZE.value,
        }

        splitter.addWidget(self.tabs)
        splitter.addWidget(self._build_log_panel())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter, stretch=1)

        for tab in (
            self.models_tab,
            self.training_tab,
            self.distillation_tab,
            self.deployment_tab,
            self.inference_tab,
            self.evals_tab,
            self.analysis_tab,
        ):
            tab.config_changed.connect(self._on_config_changed)

        self.training_tab.evals_suggest.connect(self._on_evals_suggest)
        self.results_tab.artifact_requested.connect(self._open_result_artifact)
        self.project_tab.navigate_requested.connect(self._navigate_to)
        self.project_tab.run_requested.connect(self._start_run)
        self.training_tab.run_requested.connect(lambda: self._start_run(ActionKind.TRAIN.value))
        self.distillation_tab.run_requested.connect(lambda: self._start_run(ActionKind.DISTILL.value))
        self.evals_tab.run_requested.connect(lambda: self._start_run(ActionKind.EVALUATE.value))
        self.analysis_tab.run_requested.connect(lambda: self._start_run(ActionKind.ANALYZE.value))
        self.deployment_tab.run_requested.connect(lambda: self._start_run(ActionKind.QUANTIZE.value))
        self.inference_tab.run_requested.connect(lambda: self._start_run(ActionKind.OPTIMIZE.value))
        self.inference_tab.serve_requested.connect(self._serve_queued_model)
        self.inference_tab.stop_requested.connect(self._stop_server)
        self.inference_tab.quantization_changed.connect(self.deployment_tab.reload_from_config)
        self.models_tab.model_ready.connect(self._on_model_ready)
        self.tabs.currentChanged.connect(self._update_run_button)
        self._update_run_button()

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
        self.run_btn.clicked.connect(lambda: self._start_run())

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
        self._update_run_button()

    def _navigate_to(self, area: str) -> None:
        target = self._tab_by_area.get(area)
        if target is not None:
            self.tabs.setCurrentWidget(target)

    def _open_result_artifact(self, area: str, path: str) -> None:
        if area == "analysis":
            self.analysis_tab.set_artifact(path)
        if area == "inference":
            self.inference_tab.set_artifact(path)
        self._navigate_to(area)

    def _current_action(self) -> str | None:
        return self._action_by_tab.get(self.tabs.currentWidget())

    def _update_run_button(self, _index: int = 0) -> None:
        action = self._current_action()
        running = bool(self._worker and self._worker.isRunning())
        if action:
            spec = get_action(action)
            self.run_btn.setText(f"Run {spec.title}")
            self.run_btn.setEnabled(not running)
        else:
            self.run_btn.setText("Run selected tool")
            self.run_btn.setEnabled(False)

    def _on_evals_suggest(self, eval_ids: list[str]) -> None:
        self.evals_tab.apply_selection(eval_ids)
        self._on_config_changed()

    def _append_log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)

    def _start_run(self, action: str | None = None) -> None:
        selected = action or self._current_action()
        if not selected:
            QMessageBox.information(
                self,
                "Choose a tool",
                "Open Training, Distillation, Evaluation, Analysis, Deployment, or Inference, then run that tool.",
            )
            return
        issues = collect_action_issues(self.config, selected)
        if issues:
            messages = list(dict.fromkeys(issue.message for issue in issues))
            QMessageBox.warning(
                self,
                "Tool Not Ready",
                "Resolve these items before running:\n\n" + "\n".join(messages),
            )
            self._navigate_to(issues[0].area)
            return
        if self._worker and self._worker.isRunning():
            return

        spec = get_action(selected)
        self._save_config()
        self.log_view.clear()
        self.results_tab.set_results([])
        self.project_tab.clear_results()
        self.project_tab.set_running(True)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status_label.setText("Running")
        self._append_log(f"Starting {spec.title.lower()}")

        self._worker = QueueWorker(self.config, selected, self)
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
        self.project_tab.handle_action_event(event)
        self.run_progress.setVisible(True)
        complete = event.index if event.status == "completed" else event.index - 1
        self.run_progress.setValue(round(complete / max(event.total, 1) * 100))
        self.status_label.setText(f"{event.action_name} {event.index}/{event.total}")

    def _on_model_done(self, result: ModelRunResult) -> None:
        self.results_tab.add_result(result)
        self.project_tab.add_result(result)
        if result.analysis_path:
            self.analysis_tab.set_artifact(result.analysis_path)
        if result.inference_path:
            self.inference_tab.set_artifact(result.inference_path)

    def _on_finished(self, results: list) -> None:
        self.cancel_btn.setEnabled(False)
        self.run_progress.setVisible(False)
        self.run_progress.setValue(0)
        self.status_label.setText("Complete")
        self.project_tab.set_running(False)
        self._update_run_button()
        self._append_log("Run finished.")
        inference_path = next(
            (result.inference_path for result in results if getattr(result, "inference_path", "")),
            "",
        )
        if inference_path:
            from finetuner.inference.serve import current_server

            if current_server() is None:
                self._start_serve(
                    "", optimize=False, plan_dir=inference_path, ignore_queue=True
                )

    def _on_model_ready(self, model) -> None:
        from pathlib import Path

        from finetuner.inference.serve import resolved_model_path

        path = resolved_model_path(model)
        if not path or not Path(path).exists():
            return
        self._offer_serve(path)

    def _offer_serve(self, model_path: str) -> None:
        from finetuner.inference.devices import describe_device_offer

        offer = describe_device_offer(model_path)
        box = QMessageBox(self)
        box.setWindowTitle(offer.title)
        box.setText(offer.message)
        optimize_btn = box.addButton(
            "Optimize and serve", QMessageBox.ButtonRole.AcceptRole
        )
        plain_btn = box.addButton(
            "Serve without optimizing", QMessageBox.ButtonRole.ActionRole
        )
        box.addButton("Not now", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is optimize_btn:
            self.tabs.setCurrentWidget(self.inference_tab)
            self._start_serve(model_path, optimize=True)
        elif clicked is plain_btn:
            self.tabs.setCurrentWidget(self.inference_tab)
            self._start_serve(model_path, optimize=False)

    def _serve_queued_model(self, optimize: bool) -> None:
        from pathlib import Path

        from finetuner.inference.serve import resolved_model_path

        for model in reversed(self.config.models):
            path = resolved_model_path(model)
            if path and Path(path).exists():
                self._start_serve(path, optimize=optimize)
                return
        QMessageBox.information(
            self,
            "Serve",
            "Add or download a model first. Finetuner serves it on port 1234.",
        )

    def _start_serve(
        self,
        model_path: str,
        *,
        optimize: bool,
        plan_dir: str = "",
        ignore_queue: bool = False,
    ) -> None:
        if not ignore_queue and self._worker and self._worker.isRunning():
            QMessageBox.information(
                self, "Serve", "Wait for the current tool run to finish before serving."
            )
            return
        if self._serve_worker and self._serve_worker.isRunning():
            return
        self._save_config()
        self._append_log("Starting local inference server on port 1234")
        self.status_label.setText("Starting server")
        self._serve_worker = ServeWorker(
            self.config, model_path, optimize=optimize, plan_dir=plan_dir, parent=self
        )
        self._serve_worker.log_line.connect(self._append_log)
        self._serve_worker.ready.connect(self._on_serve_ready)
        self._serve_worker.failed.connect(self._on_serve_failed)
        self._serve_worker.start()

    def _on_serve_ready(self, status) -> None:
        self.inference_tab._load_config()
        self.deployment_tab.reload_from_config()
        detail = f"{status.backend} / {status.engine} on {status.target.replace('_', ' ')}"
        if status.optimized:
            detail += " (optimized)"
        else:
            detail += " (not optimized)"
        self.inference_tab.set_serve_status(status.url, detail)
        self.status_label.setText(f"Serving :{status.port}")
        self._append_log(f"Model is serving at {status.url}")
        self._save_config()

    def _on_serve_failed(self, error: str) -> None:
        self.inference_tab.set_serve_status("", f"Serve failed: {error}")
        self.status_label.setText("Serve failed")
        self._append_log(f"Serve failed: {error}")
        QMessageBox.warning(self, "Serve failed", error)

    def _stop_server(self) -> None:
        from finetuner.inference.serve import stop_server

        stop_server()
        self.inference_tab.set_serve_status("")
        self.status_label.setText("Ready")
        self._append_log("Stopped local inference server")

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

        from finetuner.inference.serve import stop_server

        stop_server()
        if self._serve_worker and self._serve_worker.isRunning():
            self._serve_worker.wait(3000)
        self.monitor_tab.shutdown()
        self._save_config()
        super().closeEvent(event)
