from __future__ import annotations

import json

import pytest
from PySide6.QtWidgets import QApplication

from finetuner.core.job import ModelRunResult, ProjectConfig
from finetuner.ui.analysis_tab import AnalysisTab
from finetuner.ui.deployment_tab import DeploymentTab
from finetuner.ui.distillation_tab import DistillationTab
from finetuner.ui.project_tab import ProjectTab
from finetuner.ui.results_tab import ResultsTab
from finetuner.ui.training_tab import TrainingTab
from finetuner.ui.workflows_tab import WorkflowsTab


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.ui
def test_product_tabs_construct_offscreen(app):
    config = ProjectConfig()
    tabs = [
        WorkflowsTab(config),
        DistillationTab(config),
        DeploymentTab(config),
        AnalysisTab(config),
        TrainingTab(config),
        ProjectTab(config),
    ]
    assert all(tab is not None for tab in tabs)


@pytest.mark.ui
def test_workflow_editor_saves_valid_custom_graph(app):
    config = ProjectConfig()
    tab = WorkflowsTab(config)
    payload = config.workflow.to_dict()
    payload["id"] = "custom_sft"
    payload["name"] = "Custom SFT"
    tab.editor.setPlainText(json.dumps(payload))
    tab._validate_and_save()
    assert config.workflow.workflow_id == "custom_sft"


@pytest.mark.ui
def test_project_overview_and_context_bars_follow_active_workflow(app):
    config = ProjectConfig()
    project = ProjectTab(config)
    training = TrainingTab(config)
    training.pipeline_context.set_context(
        config.workflow.name,
        [stage.name for stage in config.workflow.stages if stage.kind.value == "train"],
    )

    assert project.stage_table.rowCount() == len(config.workflow.stages)
    assert "Instruction tuning" in training.pipeline_context.label.text()
    assert training.dataset_group.isHidden()


@pytest.mark.ui
def test_results_stays_compact_when_run_has_only_artifacts(app):
    tab = ResultsTab()
    tab.set_results([ModelRunResult("Model", "org/model", "/policy")])

    assert tab.table.columnCount() == 4
    assert tab.table.horizontalHeaderItem(1).text() == "Policy"


@pytest.mark.ui
def test_distillation_selectors_include_downloaded_models(app, monkeypatch, tmp_path):
    model_path = tmp_path / "org__teacher"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    (model_path / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr("finetuner.core.model_catalog.models_dir", lambda: tmp_path)
    config = ProjectConfig()
    tab = DistillationTab(config)

    teacher_index = tab.teacher.findData(str(model_path))
    student_index = tab.student.findData(str(model_path))
    assert teacher_index >= 0 and student_index >= 0
    assert tab.teacher.itemText(teacher_index) == "Downloaded | org/teacher"

    tab.teacher.setCurrentIndex(teacher_index)
    assert config.distillation.teacher_model == str(model_path)
