from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QScrollArea

from finetuner.core.job import ModelRunResult, ProjectConfig
from finetuner.monitor.stats import SystemStats
from finetuner.ui.analysis_tab import AnalysisTab
from finetuner.ui.deployment_tab import DeploymentTab
from finetuner.ui.distillation_tab import DistillationTab
from finetuner.inference.planner import AcceleratorInventory
from finetuner.ui.inference_tab import InferenceTab
from finetuner.core.hf_trending import HubModel
from finetuner.ui.models_tab import AddModelDialog, HubModelCard
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
    picker = dialog.name_combo
    labels = [f"{model.name} {model.repo_id}" for model in picker._models]
    assert picker.selected() is None
    assert picker.caption.text() == "Select a trending model"
    assert dialog.identifier_edit.text() == ""
    assert len(picker._models) > 4
    assert len(picker._cards) == len(picker._models)
    assert picker._cards[0].model.description
    vision = next(card for card in picker._cards if "vision" in card.model.shown_capabilities())
    texts = [child.text() for child in vision.findChildren(QLabel) if child.text()]
    assert "CAPABILITIES" in texts
    assert "Vision" in texts
    assert "Tool Use" in texts
    pills = [child for child in vision.findChildren(QFrame) if child.objectName() == "CapabilityPill"]
    assert pills
    assert any("Qwen" in label or "qwen" in label.lower() for label in labels)
    picker.choose_index(0)
    assert "/" in dialog.identifier_edit.text()
    assert picker._cards[0].property("selected") is True
    dialog.close()


@pytest.mark.ui
def test_model_card_shows_huggingface_downloads(app):
    card = HubModelCard(
        HubModel("org/hot-model", "Hot Model", downloads=1_250_000, likes=880)
    )
    texts = [child.text() for child in card.findChildren(QLabel) if child.text()]
    assert any("downloads so far" in text for text in texts)
    assert any("1.2M" in text for text in texts)
    assert any("likes" in text for text in texts)


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
    assert tab.findChild(QScrollArea) is not None
    tab.resize(800, 360)
    tab.show()
    app.processEvents()
    assert tab._scroll.verticalScrollBar().maximum() > 0
    tab.shutdown()
    tab.close()


@pytest.mark.ui
def test_inference_page_enables_only_detected_methods(app, monkeypatch):
    monkeypatch.setattr(
        "finetuner.inference.planner.detect_accelerator_inventory",
        lambda: AcceleratorInventory(1, "cuda", 12.0, ("RTX 4070",)),
    )
    tab = InferenceTab(ProjectConfig())
    tab._set_advanced_visible(True)
    tab.engine.setCurrentIndex(max(0, tab.engine.findData("vllm")))
    tab.target.setCurrentIndex(max(0, tab.target.findData("nvidia_gpu")))
    tab._apply_features(auto_fill=True)
    assert tab.advanced_form.isRowVisible(tab.tensor_parallel)
    assert not tab.tensor_parallel.isEnabled()
    assert tab.cuda_graphs.isEnabled()
    assert not tab.flash_attention.isVisible()

    monkeypatch.setattr(
        "finetuner.inference.planner.detect_accelerator_inventory",
        lambda: AcceleratorInventory(4, "cuda", 80.0, ("A100", "A100", "A100", "A100")),
    )
    tab._apply_features(auto_fill=True)
    assert tab.tensor_parallel.isEnabled()
    assert tab.tensor_parallel.value() == 4

    tab.engine.setCurrentIndex(max(0, tab.engine.findData("llamacpp")))
    tab.target.setCurrentIndex(max(0, tab.target.findData("cpu")))
    assert not tab.advanced_form.isRowVisible(tab.tensor_parallel)
    assert not tab.cuda_graphs.isVisible()
    tab.close()
