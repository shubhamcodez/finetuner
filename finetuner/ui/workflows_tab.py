from __future__ import annotations

import json

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from finetuner.core.job import ProjectConfig
from finetuner.workflows.schema import WorkflowSpec, WorkflowValidationError
from finetuner.workflows.templates import workflow_templates


class WorkflowsTab(QWidget):
    config_changed = Signal()

    def __init__(self, config: ProjectConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self._templates = workflow_templates()
        self._build_ui()
        self._show_workflow(config.workflow)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        intro = QLabel("Choose a reusable graph; matching tool pages provide its shared settings.")
        intro.setObjectName("HintLabel")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        toolbar = QHBoxLayout()
        self.template_combo = QComboBox()
        for workflow in self._templates:
            self.template_combo.addItem(workflow.name, workflow.workflow_id)
        load_button = QPushButton("Load Template")
        load_button.clicked.connect(self._load_template)
        validate_button = QPushButton("Validate & Save")
        validate_button.setObjectName("PrimaryButton")
        validate_button.clicked.connect(self._validate_and_save)
        toolbar.addWidget(QLabel("Starting point"))
        toolbar.addWidget(self.template_combo, 1)
        toolbar.addWidget(load_button)
        toolbar.addWidget(validate_button)
        layout.addLayout(toolbar)

        self.stage_table = QTableWidget(0, 4)
        self.stage_table.setHorizontalHeaderLabels(["Stage", "Kind", "Depends on", "Parameters"])
        self.stage_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stage_table.verticalHeader().setVisible(False)
        self.stage_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.stage_table.setMaximumHeight(170)
        layout.addWidget(self.stage_table)

        self.advanced_button = QPushButton("Show advanced JSON")
        self.advanced_button.setObjectName("SecondaryButton")
        self.advanced_button.setCheckable(True)
        self.advanced_button.toggled.connect(self._toggle_editor)
        layout.addWidget(self.advanced_button)
        self.editor_label = QLabel("Workflow JSON")
        layout.addWidget(self.editor_label)
        self.editor = QPlainTextEdit()
        self.editor.setObjectName("LogConsole")
        self.editor.setPlaceholderText("Workflow JSON")
        self.editor.setMaximumHeight(240)
        layout.addWidget(self.editor)
        self._toggle_editor(False)

        self.status = QLabel("")
        self.status.setObjectName("MutedLabel")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch()

    def _toggle_editor(self, visible: bool) -> None:
        self.editor_label.setVisible(visible)
        self.editor.setVisible(visible)
        self.advanced_button.setText("Hide advanced JSON" if visible else "Show advanced JSON")

    def _load_template(self) -> None:
        workflow_id = self.template_combo.currentData()
        workflow = next(item for item in self._templates if item.workflow_id == workflow_id)
        self._show_workflow(workflow)
        self.status.setText("Template loaded. Review parameters, then validate and save.")

    def _show_workflow(self, workflow: WorkflowSpec) -> None:
        self.editor.setPlainText(json.dumps(workflow.to_dict(), indent=2))
        self._refresh_table(workflow)

    def _refresh_table(self, workflow: WorkflowSpec) -> None:
        stages = workflow.topological_stages()
        self.stage_table.setRowCount(len(stages))
        for row, stage in enumerate(stages):
            values = (
                stage.name,
                stage.kind.value,
                ", ".join(stage.depends_on) or "-",
                json.dumps(stage.parameters, sort_keys=True),
            )
            for column, value in enumerate(values):
                self.stage_table.setItem(row, column, QTableWidgetItem(value))

    def _validate_and_save(self) -> None:
        try:
            payload = json.loads(self.editor.toPlainText())
            workflow = WorkflowSpec.from_dict(payload)
        except (json.JSONDecodeError, WorkflowValidationError, TypeError, ValueError) as exc:
            self.status.setText(f"Validation failed: {exc}")
            QMessageBox.warning(self, "Invalid Workflow", str(exc))
            return
        self.config.workflow = workflow
        self._refresh_table(workflow)
        self.status.setText(f"Saved {workflow.name} ({len(workflow.topological_stages())} stages).")
        self.config_changed.emit()
