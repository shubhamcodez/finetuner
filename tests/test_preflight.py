from __future__ import annotations

import pytest

from finetuner.core.job import ProjectConfig
from finetuner.workflows.preflight import PreflightError, validate_project_workflow
from finetuner.workflows.schema import StageKind, WorkflowSpec, WorkflowStage
from finetuner.workflows.templates import get_workflow_template


def test_default_sft_project_passes_preflight():
    validate_project_workflow(ProjectConfig())


def test_distillation_template_reports_missing_model_choices_before_execution():
    config = ProjectConfig(workflow=get_workflow_template("distill_deploy"))
    with pytest.raises(PreflightError, match="teacher model is required"):
        validate_project_workflow(config)


def test_custom_ppo_requires_reward_source():
    config = ProjectConfig(
        workflow=WorkflowSpec(
            "bad_ppo",
            "Bad PPO",
            (WorkflowStage("ppo", "PPO", StageKind.TRAIN, parameters={"method": "ppo"}),),
        )
    )
    with pytest.raises(PreflightError, match="reward-model dependency"):
        validate_project_workflow(config)
