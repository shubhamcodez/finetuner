from __future__ import annotations

import pytest

from finetuner.core.actions import ActionKind
from finetuner.core.job import ProjectConfig
from finetuner.core.preflight import PreflightError, collect_action_issues, validate_action


def test_default_training_config_is_valid_without_models_or_data():
    issues = collect_action_issues(ProjectConfig(), ActionKind.TRAIN)
    messages = [issue.message for issue in issues]
    assert "Add at least one model" in messages
    assert "Select a local, preset, or Hugging Face dataset" in messages


def test_distillation_reports_missing_model_choices_before_execution():
    with pytest.raises(PreflightError, match="Select a teacher model"):
        validate_action(ProjectConfig(), ActionKind.DISTILL)


def test_optimize_rejects_incompatible_engine_targets():
    config = ProjectConfig()
    config.inference.engine = "vllm"
    config.inference.target = "apple_gpu"
    with pytest.raises(PreflightError, match="does not support"):
        validate_action(config, ActionKind.OPTIMIZE)


def test_ppo_requires_a_reward_model_id():
    config = ProjectConfig()
    config.training.training_method = "ppo"
    with pytest.raises(PreflightError, match="Reward Model ID"):
        validate_action(config, ActionKind.TRAIN)


def test_quantize_and_optimize_validate_independently():
    config = ProjectConfig()
    config.quantization.backend = "gguf"
    config.quantization.target = "cpu"
    config.inference.engine = "vllm"
    config.inference.target = "nvidia_gpu"
    assert not [
        issue
        for issue in collect_action_issues(config, ActionKind.QUANTIZE)
        if "consume" in issue.message
    ]
    assert not [
        issue
        for issue in collect_action_issues(config, ActionKind.OPTIMIZE)
        if "consume" in issue.message
    ]
