from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from finetuner.core.job import ProjectConfig
from finetuner.eval.tasks import EVAL_TASKS


class EvalsTab(QWidget):
    config_changed = Signal()

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

        hint = QLabel("Benchmarks run after each model is fine-tuned.")
        hint.setObjectName("HintLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        settings_group = QGroupBox("Evaluation Settings")
        samples_row = QFormLayout(settings_group)
        samples_row.setVerticalSpacing(4)
        samples_row.setContentsMargins(0, 0, 0, 0)
        self.max_samples_spin = QSpinBox()
        self.max_samples_spin.setRange(10, 10000)
        self.max_samples_spin.setValue(self.config.eval_max_samples)
        self.max_samples_spin.valueChanged.connect(self._on_max_samples_changed)
        samples_row.addRow("Max samples per eval", self.max_samples_spin)
        layout.addWidget(settings_group)

        group = QGroupBox("Available Benchmarks")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(4)
        for task_id, task in EVAL_TASKS.items():
            cb = QCheckBox(f"{task.name} — {task.description}")
            cb.setChecked(task_id in self.config.enabled_evals)
            cb.stateChanged.connect(lambda _state, tid=task_id: self._on_toggle(tid))
            self._checkboxes[task_id] = cb
            group_layout.addWidget(cb)
        layout.addWidget(group)
        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

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
