from __future__ import annotations

import json

import pytest
from PySide6.QtWidgets import QApplication

from finetuner.core.job import ProjectConfig
from finetuner.ui.analysis_tab import AnalysisTab
from finetuner.ui.deployment_tab import DeploymentTab
from finetuner.ui.distillation_tab import DistillationTab
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
