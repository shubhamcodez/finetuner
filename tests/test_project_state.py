from __future__ import annotations

from finetuner.core.job import ModelJob, ModelSource, ProjectConfig
from finetuner.core.project_state import build_project_snapshot


def _configured_model() -> ModelJob:
    return ModelJob("Student", ModelSource.HUGGINGFACE, "org/student")


def test_snapshot_lists_every_tool_independently():
    snapshot = build_project_snapshot(ProjectConfig())

    areas = {area.area_id: area for area in snapshot.areas}
    assert set(areas) == {
        "models",
        "training",
        "distillation",
        "evals",
        "analysis",
        "deployment",
        "inference",
    }
    assert all(area.included for area in snapshot.areas)
    assert not areas["models"].ready
    assert not areas["training"].ready
    assert not areas["evals"].ready
    assert not areas["deployment"].ready
    assert not areas["inference"].ready


def test_training_is_ready_when_models_and_dataset_are_configured():
    config = ProjectConfig(models=[_configured_model()])
    config.training.dataset_preset_id = "alpaca"
    config.training.dataset_use_bundled_only = True

    snapshot = build_project_snapshot(config)
    training = next(area for area in snapshot.areas if area.area_id == "training")
    evals = next(area for area in snapshot.areas if area.area_id == "evals")
    assert training.ready
    assert evals.ready
    assert training.action == "train"


def test_distillation_readiness_is_owned_by_the_distillation_tool():
    config = ProjectConfig(models=[_configured_model()])
    config.training.dataset_preset_id = "alpaca"

    snapshot = build_project_snapshot(config)
    distillation = next(area for area in snapshot.areas if area.area_id == "distillation")
    assert not distillation.ready
    assert "teacher model is required" in distillation.issues
    assert "student model is required" in distillation.issues


def test_deployment_and_inference_do_not_require_a_dataset():
    config = ProjectConfig(models=[_configured_model()])

    snapshot = build_project_snapshot(config)
    deployment = next(area for area in snapshot.areas if area.area_id == "deployment")
    inference = next(area for area in snapshot.areas if area.area_id == "inference")
    assert deployment.ready
    assert inference.ready
    assert "dataset" not in " ".join(deployment.issues).lower()
    assert "dataset" not in " ".join(inference.issues).lower()
