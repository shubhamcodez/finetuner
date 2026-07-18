from __future__ import annotations

from finetuner.core.job import ModelJob, ModelSource, ProjectConfig
from finetuner.core.project_state import build_project_snapshot, workflow_requires_dataset
from finetuner.workflows.schema import StageKind, WorkflowSpec, WorkflowStage
from finetuner.workflows.templates import get_workflow_template


def _configured_model() -> ModelJob:
    return ModelJob("Student", ModelSource.HUGGINGFACE, "org/student")


def test_snapshot_collects_cross_tool_readiness_issues():
    snapshot = build_project_snapshot(ProjectConfig())

    assert not snapshot.ready
    assert {issue.area for issue in snapshot.issues} == {"models", "training"}
    assert [stage.stage_id for stage in snapshot.stages] == ["sft", "evaluate"]
    training = next(area for area in snapshot.areas if area.area_id == "training")
    assert training.included
    assert not training.ready


def test_snapshot_is_ready_when_shared_sft_inputs_are_configured():
    config = ProjectConfig(models=[_configured_model()])
    config.training.dataset_preset_id = "alpaca"
    config.training.dataset_use_bundled_only = True

    snapshot = build_project_snapshot(config)

    assert snapshot.ready
    assert not snapshot.issues
    evals = next(area for area in snapshot.areas if area.area_id == "evals")
    deployment = next(area for area in snapshot.areas if area.area_id == "deployment")
    assert evals.included and evals.ready
    assert not deployment.included


def test_distillation_readiness_is_owned_by_distillation_area():
    config = ProjectConfig(
        models=[_configured_model()], workflow=get_workflow_template("distill_deploy")
    )
    config.training.dataset_preset_id = "alpaca"

    snapshot = build_project_snapshot(config)

    messages = [issue.message for issue in snapshot.issues if issue.area == "distillation"]
    assert "teacher model is required" in messages
    assert "student model is required" in messages
    deployment = next(area for area in snapshot.areas if area.area_id == "deployment")
    assert deployment.included and deployment.ready


def test_deployment_only_workflow_does_not_require_a_dataset():
    workflow = WorkflowSpec(
        "deploy",
        "Deploy only",
        (WorkflowStage("quantize", "Quantize", StageKind.QUANTIZE),),
    )
    config = ProjectConfig(models=[_configured_model()], workflow=workflow)

    assert not workflow_requires_dataset(config)
    snapshot = build_project_snapshot(config)
    assert snapshot.ready
    training = next(area for area in snapshot.areas if area.area_id == "training")
    assert not training.included
