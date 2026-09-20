from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from finetuner.core.actions import ActionKind, action_specs, get_action
from finetuner.core.job import ProjectConfig
from finetuner.core.preflight import PreflightIssue, collect_action_issues


@dataclass(frozen=True)
class ProjectAreaState:
    area_id: str
    title: str
    summary: str
    included: bool
    ready: bool
    issues: tuple[str, ...] = ()
    action: str = ""


@dataclass(frozen=True)
class ProjectSnapshot:
    title: str
    ready: bool
    issues: tuple[PreflightIssue, ...]
    areas: tuple[ProjectAreaState, ...]


def action_requires_dataset(action: ActionKind | str) -> bool:
    return get_action(action).needs_dataset


def _dataset_summary(config: ProjectConfig) -> str:
    training = config.training
    if training.dataset_path:
        path = Path(training.dataset_path)
        return path.name or str(path)
    if training.dataset_preset_id:
        mode = "offline sample" if training.dataset_use_bundled_only else "Hugging Face"
        return f"{training.dataset_preset_id} ({mode})"
    if training.dataset_hf_id:
        return training.dataset_hf_id
    return "No dataset selected"


def _area_summary(area_id: str, config: ProjectConfig) -> str:
    if area_id == "models":
        names = ", ".join(model.name for model in config.models[:2])
        if len(config.models) > 2:
            names += f" +{len(config.models) - 2}"
        return names or "No models queued"
    if area_id == "training":
        method = config.training.training_method.upper()
        return f"{_dataset_summary(config)} | {method}"
    if area_id == "distillation":
        return (
            f"{config.distillation.teacher_model or 'Teacher unset'} -> "
            f"{config.distillation.student_model or 'student unset'}"
        )
    if area_id == "evals":
        return ", ".join(config.enabled_evals) or "No benchmarks selected"
    if area_id == "analysis":
        return f"{config.analysis.reducer.upper()} representation projection"
    if area_id == "deployment":
        return (
            f"{config.quantization.backend.upper()} {config.quantization.bits}-bit for "
            f"{config.quantization.target.replace('_', ' ')}"
        )
    if area_id == "inference":
        return f"{config.inference.engine} on {config.inference.target.replace('_', ' ')}"
    return area_id


def build_project_snapshot(config: ProjectConfig) -> ProjectSnapshot:
    model_messages = []
    if not config.models:
        model_messages.append("Add at least one model")
    elif not any(model.identifier.strip() for model in config.models):
        model_messages.append("Every queued model is missing an identifier")

    areas = [
        ProjectAreaState(
            "models",
            "Models",
            _area_summary("models", config),
            True,
            not model_messages,
            tuple(model_messages),
        )
    ]
    all_issues: list[PreflightIssue] = [
        PreflightIssue("models", message) for message in model_messages
    ]
    for spec in action_specs():
        issues = collect_action_issues(config, spec.action)
        all_issues.extend(issues)
        areas.append(
            ProjectAreaState(
                spec.area,
                spec.title,
                _area_summary(spec.area, config),
                True,
                not issues,
                tuple(issue.message for issue in issues),
                spec.action.value,
            )
        )
    ready_tools = sum(1 for area in areas if area.action and area.ready)
    return ProjectSnapshot(
        "Independent tools",
        ready_tools > 0,
        tuple(all_issues),
        tuple(areas),
    )
