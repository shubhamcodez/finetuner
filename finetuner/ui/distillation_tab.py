from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from finetuner.core.job import ProjectConfig
from finetuner.core.model_catalog import discover_downloaded_models
from finetuner.distillation.config import DistillationTechnique
from finetuner.distillation.domains import DOMAIN_PRESETS
from finetuner.ui.pipeline_context import PipelineContextBar


class DistillationTab(QWidget):
    config_changed = Signal()

    def __init__(self, config: ProjectConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.domain_checks: dict[str, QCheckBox] = {}
        self._build_ui()
        self._load_config()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        self.pipeline_context = PipelineContextBar()
        layout.addWidget(self.pipeline_context)
        intro = QLabel(
            "Transfer a teacher into a smaller student; tokenizer compatibility depends on technique."
        )
        intro.setObjectName("HintLabel")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setVerticalSpacing(4)
        self.form = form
        self.teacher = QComboBox()
        self.teacher.setEditable(True)
        self.teacher.lineEdit().setPlaceholderText(
            "Select a downloaded model or enter a Hugging Face ID/path"
        )
        self.student = QComboBox()
        self.student.setEditable(True)
        self.student.lineEdit().setPlaceholderText(
            "Select a downloaded model or enter a Hugging Face ID/path"
        )
        self.technique = QComboBox()
        techniques = (
            ("Sequence KD (portable)", DistillationTechnique.SEQUENCE.value),
            ("Logit KL (experimental)", DistillationTechnique.LOGIT_KL.value),
            ("On-policy GKD (experimental)", DistillationTechnique.ON_POLICY_GKD.value),
            ("Reverse-KL / MiniLLM-style (experimental)", DistillationTechnique.REVERSE_KL.value),
        )
        for name, value in techniques:
            self.technique.addItem(name, value)
        self.domain_mode = QComboBox()
        self.domain_mode.addItem("All / general", "all")
        self.domain_mode.addItem("Selected fields", "presets")
        self.domain_mode.addItem("Custom topic", "custom")
        self.custom_domain = QLineEdit()
        self.custom_domain.setPlaceholderText("e.g. compiler optimization, medical coding")
        self.max_samples = QSpinBox()
        self.max_samples.setRange(1, 10_000_000)
        self.max_samples.setValue(1000)
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0, 5)
        self.temperature.setSingleStep(0.1)
        form.addRow("Teacher model", self.teacher)
        form.addRow("Student model", self.student)
        form.addRow("Technique", self.technique)
        form.addRow("Knowledge scope", self.domain_mode)
        form.addRow("Custom topic(s)", self.custom_domain)
        form.addRow("Maximum samples", self.max_samples)
        form.addRow("Teacher temperature", self.temperature)
        layout.addLayout(form)

        domains = QGroupBox("Field presets")
        domain_layout = QGridLayout(domains)
        domain_layout.setVerticalSpacing(3)
        for index, (domain_id, preset) in enumerate(DOMAIN_PRESETS.items()):
            check = QCheckBox(preset.name)
            check.toggled.connect(self._sync)
            self.domain_checks[domain_id] = check
            domain_layout.addWidget(check, index // 3, index % 3)
        layout.addWidget(domains)
        self.domain_group = domains
        self.status = QLabel("")
        self.status.setObjectName("MutedLabel")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch()

        for widget in (self.technique, self.domain_mode):
            widget.currentIndexChanged.connect(self._sync)
        self.teacher.currentTextChanged.connect(self._sync)
        self.student.currentTextChanged.connect(self._sync)
        self.custom_domain.textChanged.connect(self._sync)
        self.max_samples.valueChanged.connect(self._sync)
        self.temperature.valueChanged.connect(self._sync)

    def _load_models(self) -> None:
        current_teacher = self.config.distillation.teacher_model
        current_student = self.config.distillation.student_model
        for combo, current in ((self.teacher, current_teacher), (self.student, current_student)):
            combo.blockSignals(True)
            combo.clear()
            values: set[str] = set()
            downloaded_aliases: dict[str, str] = {}
            for model in discover_downloaded_models():
                combo.addItem(f"Downloaded | {model.name}", model.path)
                combo.setItemData(combo.count() - 1, model.path, Qt.ItemDataRole.ToolTipRole)
                values.add(model.path.casefold())
                if model.source_id:
                    values.add(model.source_id.casefold())
                    downloaded_aliases[model.source_id.casefold()] = model.path
            for model in self.config.models:
                value = model.output_path or model.identifier
                if not value or value.casefold() in values or model.identifier.casefold() in values:
                    continue
                combo.addItem(f"Queue | {model.name}", value)
                combo.setItemData(combo.count() - 1, value, Qt.ItemDataRole.ToolTipRole)
                values.add(value.casefold())
            selected_value = downloaded_aliases.get(current.casefold(), current)
            selected = combo.findData(selected_value)
            if selected >= 0:
                combo.setCurrentIndex(selected)
            else:
                combo.setCurrentText(current)
            combo.blockSignals(False)

    @staticmethod
    def _selected_model(combo: QComboBox) -> str:
        index = combo.currentIndex()
        if index >= 0 and combo.currentText() == combo.itemText(index):
            value = combo.itemData(index)
            if value:
                return str(value)
        return combo.currentText().strip()

    def _load_config(self) -> None:
        self._load_models()
        d = self.config.distillation
        self.technique.setCurrentIndex(max(0, self.technique.findData(d.technique)))
        self.domain_mode.setCurrentIndex(max(0, self.domain_mode.findData(d.domain.mode)))
        self.custom_domain.setText(d.domain.custom)
        self.max_samples.setValue(d.max_samples)
        self.temperature.setValue(d.temperature)
        for domain_id, check in self.domain_checks.items():
            check.setChecked(domain_id in d.domain.fields)
        self._sync()

    def showEvent(self, event) -> None:
        self._load_models()
        super().showEvent(event)

    def _sync(self, _value=None) -> None:
        d = self.config.distillation
        d.teacher_model = self._selected_model(self.teacher)
        d.student_model = self._selected_model(self.student)
        d.technique = self.technique.currentData() or "sequence"
        d.domain.mode = self.domain_mode.currentData() or "all"
        d.domain.custom = self.custom_domain.text().strip()
        d.domain.fields = [key for key, check in self.domain_checks.items() if check.isChecked()]
        d.max_samples = self.max_samples.value()
        d.temperature = self.temperature.value()
        self.domain_group.setVisible(d.domain.mode == "presets")
        self.form.setRowVisible(self.custom_domain, d.domain.mode == "custom")
        self.custom_domain.setEnabled(d.domain.mode == "custom")
        errors = d.validate()
        self.status.setText("; ".join(errors) if errors else "Ready for a Distill stage.")
        self.config_changed.emit()
