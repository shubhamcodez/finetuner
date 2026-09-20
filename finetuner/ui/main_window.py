from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from finetuner.core.actions import ActionKind, get_action
from finetuner.core.config_store import load_config, save_config
from finetuner.core.job import ModelRunResult, ProjectConfig
from finetuner.core.preflight import collect_action_issues
from finetuner.ui.analysis_tab import AnalysisTab
from finetuner.ui.branding import PRODUCT_NAME, app_icon
from finetuner.ui.command_palette import CommandPalette
from finetuner.ui.deployment_tab import DeploymentTab
from finetuner.ui.distillation_tab import DistillationTab
from finetuner.ui.evals_tab import EvalsTab
from finetuner.ui.inference_tab import InferenceTab
from finetuner.ui.models_tab import ModelsTab
from finetuner.ui.monitor_tab import MonitorTab
from finetuner.ui.project_tab import ProjectTab
from finetuner.ui.results_tab import ResultsTab
from finetuner.ui.shell import Sidebar, PageHeader, WorkspaceBar
from finetuner.ui.training_tab import TrainingTab
from finetuner.ui.utility_pages import DocsPage, SettingsPage

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
        self.setWindowTitle(PRODUCT_NAME)
        self.setWindowIcon(app_icon())
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)

        self.config = load_config()
        self._worker: QueueWorker | None = None
        self._serve_worker: ServeWorker | None = None
        self._area = "project"

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
        self.docs_page = DocsPage()
        self.settings_page = SettingsPage(self.config)

        self._page_by_area = {
            "project": self.project_tab,
            "models": self.models_tab,
            "data": self.training_tab,
            "training": self.training_tab,
            "distillation": self.distillation_tab,
            "evals": self.evals_tab,
            "analysis": self.analysis_tab,
            "deployment": self.deployment_tab,
            "inference": self.inference_tab,
            "results": self.results_tab,
            "monitor": self.monitor_tab,
            "docs": self.docs_page,
            "settings": self.settings_page,
        }
        self._action_by_area = {
            "training": ActionKind.TRAIN.value,
            "distillation": ActionKind.DISTILL.value,
            "evals": ActionKind.EVALUATE.value,
            "analysis": ActionKind.ANALYZE.value,
            "deployment": ActionKind.QUANTIZE.value,
            "inference": ActionKind.OPTIMIZE.value,
        }

        shell = QWidget()
        shell.setObjectName("AppShell")
        self.setCentralWidget(shell)
        row = QHBoxLayout(shell)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.sidebar = Sidebar()
        self.sidebar.navigate.connect(self._navigate_to)
        row.addWidget(self.sidebar)

        workspace = QWidget()
        self._workspace_layout = QVBoxLayout(workspace)
        self._workspace_layout.setContentsMargins(32, 24, 32, 24)
        self._workspace_layout.setSpacing(24)
        top = QHBoxLayout()
        self.page_header = PageHeader()
        top.addWidget(self.page_header, 1)
        self.workspace_bar = WorkspaceBar()
        self.workspace_bar.command_requested.connect(self._open_palette)
        self.workspace_bar.help_requested.connect(lambda: self._navigate_to("docs"))
        top.addWidget(self.workspace_bar, 0)
        self._workspace_layout.addLayout(top)

        self.pages = QStackedWidget()
        seen: set[int] = set()
        for page in self._page_by_area.values():
            if id(page) in seen:
                continue
            seen.add(id(page))
            self.pages.addWidget(page)
        self._workspace_layout.addWidget(self.pages, 1)
        row.addWidget(workspace, 1)

        self._build_log_drawer()
        self.palette = CommandPalette(self)
        self.palette.activated.connect(self._navigate_to)
        QShortcut(QKeySequence("Ctrl+K"), self, self._open_palette)

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
        self._navigate_to("project")

    def _build_log_drawer(self) -> None:
        self.log_drawer = QDockWidget("Run details", self)
        self.log_drawer.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        heading = QHBoxLayout()
        title = QLabel("Run details")
        title.setObjectName("SectionTitle")
        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("GhostButton")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_run)
        heading.addWidget(title)
        heading.addStretch()
        heading.addWidget(self.status_label)
        heading.addWidget(self.cancel_btn)
        layout.addLayout(heading)
        self.run_progress = QProgressBar()
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(0)
        self.run_progress.setVisible(False)
        layout.addWidget(self.run_progress)
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("LogConsole")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setPlaceholderText("Logs for the active run appear here.")
        layout.addWidget(self.log_view, 1)
        self.log_drawer.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.log_drawer)
        self.log_drawer.hide()

    def _open_palette(self) -> None:
        self.palette.open_palette()

    def _save_config(self) -> None:
        save_config(self.config)

    def _on_config_changed(self) -> None:
        self._save_config()
        self.project_tab.refresh()
        self._update_run_button()

    def _navigate_to(self, area: str) -> None:
        target = self._page_by_area.get(area)
        if target is None:
            return
        self._area = area
        self.pages.setCurrentWidget(target)
        self.sidebar.select(area)
        self.page_header.set_page(area)
        compact = area == "monitor"
        self._workspace_layout.setContentsMargins(32, 8 if compact else 24, 32, 8 if compact else 24)
        self._workspace_layout.setSpacing(8 if compact else 24)
        self._update_run_button()

    def _open_result_artifact(self, area: str, path: str) -> None:
        if area == "analysis":
            self.analysis_tab.set_artifact(path)
        if area == "inference":
            self.inference_tab.set_artifact(path)
        self._navigate_to(area)

    def _current_action(self) -> str | None:
        return self._action_by_area.get(self._area)

    def _update_run_button(self, _index: int = 0) -> None:
        running = bool(self._worker and self._worker.isRunning())
        self.cancel_btn.setEnabled(running)
        for tab in (
            self.training_tab,
            self.distillation_tab,
            self.evals_tab,
            self.analysis_tab,
            self.deployment_tab,
            self.inference_tab,
        ):
            tab.run_bar.set_enabled(not running)

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
                "Open Finetune, Distillation, Evaluation, Analysis, Deployment, or Inference, then start that tool.",
            )
            return
        issues = collect_action_issues(self.config, selected)
        if issues:
            messages = list(dict.fromkeys(issue.message for issue in issues))
            QMessageBox.warning(
                self,
                "Needs attention",
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
        self.status_label.setText("Running")
        self.log_drawer.show()
        self._append_log(f"Starting {spec.title.lower()}")

        self._worker = QueueWorker(self.config, selected, self)
        self._worker.log_line.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.download_progress.connect(self._on_download_progress)
        self._worker.model_done.connect(self._on_model_done)
        self._worker.stage_event.connect(self._on_stage_event)
        self._worker.finished_all.connect(self._on_finished)
        self._worker.start()
        self._update_run_button()

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
        self.status_label.setText("Completed")
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
        optimize_btn = box.addButton("Optimize and serve", QMessageBox.ButtonRole.AcceptRole)
        plain_btn = box.addButton("Serve without optimizing", QMessageBox.ButtonRole.ActionRole)
        box.addButton("Not now", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is optimize_btn:
            self._navigate_to("inference")
            self._start_serve(model_path, optimize=True)
        elif clicked is plain_btn:
            self._navigate_to("inference")
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
            "Add or download a model first. Inferna serves it on port 1234.",
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
        self.log_drawer.show()
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
        detail += " (optimized)" if status.optimized else " (not optimized)"
        self.inference_tab.set_serve_status(status.url, detail)
        self.status_label.setText(f"Serving :{status.port}")
        self._append_log(f"Model is serving at {status.url}")
        self._save_config()

    def _on_serve_failed(self, error: str) -> None:
        self.inference_tab.set_serve_status("", f"Serve failed: {error}")
        self.status_label.setText("Failed")
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
                "A run is in progress. Cancel and exit?",
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
