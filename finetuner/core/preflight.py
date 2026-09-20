from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from finetuner.core.actions import ActionKind, get_action
from finetuner.core.job import ProjectConfig
from finetuner.training.validation import validate_training_config


class PreflightError(ValueError):
    pass


@dataclass(frozen=True)
class PreflightIssue:
    area: str
    message: str
    action: str = ""


def _dataset_issues(config: ProjectConfig) -> list[str]:
    training = config.training
    if training.dataset_path:
        return [] if Path(training.dataset_path).is_file() else ["Local dataset does not exist"]
    if training.dataset_preset_id or training.dataset_hf_id:
        return []
    return ["Select a local, preset, or Hugging Face dataset"]


def _model_issues(config: ProjectConfig) -> list[str]:
    if not config.models:
        return ["Add at least one model"]
    if not any(model.identifier.strip() for model in config.models):
        return ["Every queued model is missing an identifier"]
    return []


def collect_action_issues(config: ProjectConfig, action: ActionKind | str) -> list[PreflightIssue]:
    spec = get_action(action)
    issues: list[PreflightIssue] = []

    def add(message: str, area: str | None = None) -> None:
        issues.append(PreflightIssue(area or spec.area, message, spec.action.value))

    if spec.needs_models:
        for message in _model_issues(config):
            add(message, "models")
    if spec.needs_dataset:
        for message in _dataset_issues(config):
            add(message, "training")

    if spec.action == ActionKind.TRAIN:
        for message in validate_training_config(config.training):
            add(message)
        if config.training.training_method == "ppo" and not config.training.reward_model_id.strip():
            add("PPO needs a Training Reward Model ID")
    elif spec.action == ActionKind.DISTILL:
        for message in config.distillation.validate():
            add(message)
    elif spec.action == ActionKind.QUANTIZE:
        for message in config.quantization.validate():
            add(message)
    elif spec.action == ActionKind.OPTIMIZE:
        for message in config.inference.validate():
            add(message)
    elif spec.action == ActionKind.ANALYZE:
        for message in config.analysis.validate():
            add(message)
    elif spec.action == ActionKind.EVALUATE:
        if not config.enabled_evals:
            add("select at least one evaluation task")
    return issues


def collect_project_issues(config: ProjectConfig) -> list[PreflightIssue]:
    issues = [PreflightIssue("models", message) for message in _model_issues(config)]
    seen = {(issue.area, issue.message) for issue in issues}
    for spec in (
        get_action(ActionKind.TRAIN),
        get_action(ActionKind.DISTILL),
        get_action(ActionKind.EVALUATE),
        get_action(ActionKind.ANALYZE),
        get_action(ActionKind.QUANTIZE),
        get_action(ActionKind.OPTIMIZE),
    ):
        for issue in collect_action_issues(config, spec.action):
            key = (issue.area, issue.message)
            if key not in seen:
                issues.append(issue)
                seen.add(key)
    return issues


def validate_action(config: ProjectConfig, action: ActionKind | str) -> None:
    spec = get_action(action)
    issues = collect_action_issues(config, spec.action)
    if issues:
        raise PreflightError(
            "\n".join(f"{spec.title}: {issue.message}" for issue in issues)
        )
