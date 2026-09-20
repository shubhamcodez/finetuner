from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QLabel

from finetuner.core.job import ModelRunResult, ProjectConfig
from finetuner.monitor.stats import SystemStats
from finetuner.ui.analysis_tab import AnalysisTab
from finetuner.ui.deployment_tab import DeploymentTab
from finetuner.ui.distillation_tab import DistillationTab
from finetuner.ui.inference_tab import InferenceTab
from finetuner.ui.models_tab import AddModelDialog
from finetuner.ui.monitor_tab import MonitorTab
from finetuner.ui.project_tab import ProjectTab
from finetuner.ui.results_tab import ResultsTab
from finetuner.ui.training_tab import TrainingTab


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.ui
def test_product_tabs_construct_offscreen(app):
    config = ProjectConfig()
    tabs = [
        DistillationTab(config),
        DeploymentTab(config),
        InferenceTab(config),
        AnalysisTab(config),
        TrainingTab(config),
        ProjectTab(config),
    ]
    assert all(tab is not None for tab in tabs)
    assert all(hasattr(tab, "run_bar") for tab in tabs[:-1])


@pytest.mark.ui
def test_project_overview_lists_independent_tools(app):
    config = ProjectConfig()
    project = ProjectTab(config)
    training = TrainingTab(config)

    assert set(project._cards) == {
        "models",
        "training",
        "distillation",
        "evals",
        "analysis",
        "deployment",
        "inference",
    }
    labels = [child.text() for child in project.findChildren(QLabel)]
    assert "Prepare data" not in labels
    assert "→" not in labels
    assert "independently" in training.run_bar.label.text().lower()
    assert training.dataset_group.isHidden()


@pytest.mark.ui
def test_results_stays_compact_when_run_has_only_artifacts(app):
    tab = ResultsTab()
    tab.set_results([ModelRunResult("Model", "org/model", "/policy")])

    assert tab.table.columnCount() == 5
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


@pytest.mark.ui
def test_add_model_dialog_lists_huggingface_choices(app, monkeypatch):
    monkeypatch.setattr("finetuner.ui.models_tab.AddModelDialog._start_trending_fetch", lambda self, token="": None)
    dialog = AddModelDialog()
    labels = [dialog.name_combo.itemText(i) for i in range(dialog.name_combo.count())]
    assert dialog.name_combo.currentIndex() == 0
    assert not dialog.name_combo.isEditable()
    assert labels[0].startswith("Select")
    assert dialog.identifier_edit.text() == ""
    assert dialog.name_combo.count() > 4
    assert any("Qwen" in label or "qwen" in label.lower() for label in labels)
    dialog.name_combo.setCurrentIndex(1)
    assert "/" in dialog.identifier_edit.text()
    dialog.close()


@pytest.mark.ui
def test_system_tab_renders_npu_stats(app):
    tab = MonitorTab()
    tab._on_stats(
        SystemStats(
            cpu_percent=10.0,
            ram_used_gb=8.0,
            ram_total_gb=16.0,
            npu_available=True,
            npu_name="Qualcomm Hexagon NPU",
            npu_util_percent=42.0,
            npu_mem_used_gb=0.7,
            npu_mem_shared_gb=0.2,
            tpu_available=True,
            tpu_name="Coral Edge TPU",
            tpu_util_percent=18.0,
            tpu_mem_used_gb=0.5,
            tpu_mem_total_gb=8.0,
            tpu_detail="Local Edge TPU",
        )
    )
    assert "42" in tab.npu_card.value_label.text()
    assert "Hexagon" in tab.npu_card.detail_label.text()
    assert "0.7" in tab.npu_mem_card.value_label.text()
    assert "18" in tab.tpu_card.value_label.text()
    assert "Coral" in tab.tpu_card.detail_label.text()
    assert tab.cpu_chart._view.minimumHeight() >= 180
    assert tab.npu_chart._view.minimumHeight() >= 180
    tab.shutdown()
