from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from finetuner.core.job import ProjectConfig
from finetuner.eval.tasks import EVAL_TASKS
from finetuner.ui.model_menu import reload_model_combo
from finetuner.ui.tool_run import ToolRunBar


class EvalsTab(QWidget):
    config_changed = Signal()
    run_requested = Signal()
    models_requested = Signal()

    def __init__(self, config: ProjectConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self._checkboxes: dict[str, QCheckBox] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        self.run_bar = ToolRunBar("Run evaluation")
        self.run_bar.run_requested.connect(self.run_requested.emit)
        layout.addWidget(self.run_bar)

        settings_group = QGroupBox("Evaluation Settings")
        samples_row = QFormLayout(settings_group)
        samples_row.setVerticalSpacing(4)
        samples_row.setContentsMargins(0, 0, 0, 0)
        self.model = QComboBox()
        self.model.currentIndexChanged.connect(self._on_model_changed)
        samples_row.addRow("Model", self.model)
        self.max_samples_spin = QSpinBox()
        self.max_samples_spin.setRange(10, 10000)
        self.max_samples_spin.setValue(self.config.eval_max_samples)
        self.max_samples_spin.valueChanged.connect(self._on_max_samples_changed)
        samples_row.addRow("Max samples per eval", self.max_samples_spin)
        layout.addWidget(settings_group)

        group = QGroupBox("Available Benchmarks")
        group_layout = QGridLayout(group)
        group_layout.setSpacing(4)
        for index, (task_id, task) in enumerate(EVAL_TASKS.items()):
            cb = QCheckBox(task.name)
            cb.setToolTip(task.description)
            cb.setChecked(task_id in self.config.enabled_evals)
            cb.stateChanged.connect(lambda _state, tid=task_id: self._on_toggle(tid))
            self._checkboxes[task_id] = cb
            group_layout.addWidget(cb, index // 2, index % 2)
        layout.addWidget(group)
        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)
        self.reload_models()

    def selected_model_path(self) -> str:
        value = self.model.currentData()
        return str(value) if value else ""

    def reload_models(self) -> None:
        reload_model_combo(
            self.model,
            self.config.models,
            self.selected_model_path() or self.config.eval_model_path,
        )
        self.config.eval_model_path = self.selected_model_path()

    def showEvent(self, event) -> None:
        self.reload_models()
        super().showEvent(event)
        if self.model.count() == 0:
            self.config.eval_model_path = ""
            QTimer.singleShot(0, self.models_requested.emit)

    def _on_model_changed(self, _index: int = 0) -> None:
        self.config.eval_model_path = self.selected_model_path()
        self.config_changed.emit()

    def apply_selection(self, eval_ids: list[str]) -> None:
        self.config.enabled_evals = list(eval_ids)
        for task_id, cb in self._checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(task_id in eval_ids)
            cb.blockSignals(False)

    def _on_toggle(self, task_id: str) -> None:
        cb = self._checkboxes[task_id]
        if cb.isChecked():
            if task_id not in self.config.enabled_evals:
                self.config.enabled_evals.append(task_id)
        else:
            self.config.enabled_evals = [e for e in self.config.enabled_evals if e != task_id]
        self.config_changed.emit()

    def _on_max_samples_changed(self, value: int) -> None:
        self.config.eval_max_samples = value
        self.config_changed.emit()
