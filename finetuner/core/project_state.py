from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from finetuner.core.job import ProjectConfig
from finetuner.workflows.preflight import PreflightIssue, collect_workflow_issues
from finetuner.workflows.schema import StageKind, WorkflowStage


@dataclass(frozen=True)
class ProjectAreaState:
    area_id: str
    title: str
    summary: str
    included: bool
    ready: bool
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectStageState:
    stage_id: str
    name: str
    kind: str
    depends_on: tuple[str, ...]
    summary: str


@dataclass(frozen=True)
class ProjectSnapshot:
    workflow_name: str
    ready: bool
    issues: tuple[PreflightIssue, ...]
    areas: tuple[ProjectAreaState, ...]
    stages: tuple[ProjectStageState, ...]


def workflow_requires_dataset(config: ProjectConfig) -> bool:
    return any(
        stage.kind in {StageKind.TRAIN, StageKind.DISTILL, StageKind.ANALYZE}
        for stage in config.workflow.topological_stages()
    )


def _dataset_summary(config: ProjectConfig) -> tuple[str, list[str]]:
    training = config.training
    if training.dataset_path:
        path = Path(training.dataset_path)
        return path.name or str(path), ([] if path.is_file() else ["Local dataset does not exist"])
    if training.dataset_preset_id:
        mode = "offline sample" if training.dataset_use_bundled_only else "Hugging Face"
        return f"{training.dataset_preset_id} ({mode})", []
    if training.dataset_hf_id:
        return training.dataset_hf_id, []
    return "No dataset selected", ["Select a local, preset, or Hugging Face dataset"]


def _stage_summary(stage: WorkflowStage, config: ProjectConfig) -> str:
    if stage.kind == StageKind.TRAIN:
        method = stage.parameters.get("method", config.training.training_method)
        return f"{str(method).upper()} | {config.training.max_steps} steps"
    if stage.kind == StageKind.DISTILL:
        domain = config.distillation.domain
        scope = domain.custom if domain.mode == "custom" else domain.mode
        return f"{config.distillation.technique} | {scope}"
    if stage.kind == StageKind.QUANTIZE:
        quant = config.quantization
        return f"{quant.backend.upper()} {quant.bits}-bit | {quant.target.replace('_', ' ')}"
    if stage.kind == StageKind.EVALUATE:
        tasks = stage.parameters.get("task_ids", config.enabled_evals)
        return f"{len(tasks)} benchmark{'s' if len(tasks) != 1 else ''}"
    if stage.kind == StageKind.ANALYZE:
        return f"{config.analysis.reducer.upper()} | {config.analysis.pooling} pooling"
    return stage.kind.value


def build_project_snapshot(config: ProjectConfig) -> ProjectSnapshot:
    issues = list(collect_workflow_issues(config))
    if not config.models:
        issues.append(PreflightIssue("models", "Add at least one model"))
    elif not any(model.identifier.strip() for model in config.models):
        issues.append(PreflightIssue("models", "Every queued model is missing an identifier"))

    needs_dataset = workflow_requires_dataset(config)
    dataset_summary, dataset_issues = _dataset_summary(config)
    if needs_dataset:
        issues.extend(PreflightIssue("training", message) for message in dataset_issues)

    stages = tuple(
        ProjectStageState(
            stage.stage_id,
            stage.name,
            stage.kind.value,
            stage.depends_on,
            _stage_summary(stage, config),
        )
        for stage in config.workflow.topological_stages()
    )
    included_kinds = {stage.kind for stage in config.workflow.topological_stages()}
    by_area: dict[str, list[str]] = {}
    for issue in issues:
        by_area.setdefault(issue.area, []).append(issue.message)

    model_names = ", ".join(model.name for model in config.models[:2])
    if len(config.models) > 2:
        model_names += f" +{len(config.models) - 2}"
    areas = (
        ProjectAreaState(
            "models",
            "Models",
            model_names or "No models queued",
            True,
            not by_area.get("models"),
            tuple(by_area.get("models", ())),
        ),
        ProjectAreaState(
            "training",
            "Data & training",
            dataset_summary,
            needs_dataset or StageKind.TRAIN in included_kinds,
            not by_area.get("training"),
            tuple(by_area.get("training", ())),
        ),
        ProjectAreaState(
            "distillation",
            "Distillation",
            f"{config.distillation.teacher_model or 'Teacher unset'} -> "
            f"{config.distillation.student_model or 'student unset'}",
            StageKind.DISTILL in included_kinds,
            not by_area.get("distillation"),
            tuple(by_area.get("distillation", ())),
        ),
        ProjectAreaState(
            "evals",
            "Evaluation",
            ", ".join(config.enabled_evals) or "No benchmarks selected",
            StageKind.EVALUATE in included_kinds,
            not by_area.get("evals"),
            tuple(by_area.get("evals", ())),
        ),
        ProjectAreaState(
            "analysis",
            "Analysis",
            f"{config.analysis.reducer.upper()} representation projection",
            StageKind.ANALYZE in included_kinds,
            not by_area.get("analysis"),
            tuple(by_area.get("analysis", ())),
        ),
        ProjectAreaState(
            "deployment",
            "Deployment",
            f"{config.quantization.backend.upper()} {config.quantization.bits}-bit for "
            f"{config.quantization.target.replace('_', ' ')}",
            StageKind.QUANTIZE in included_kinds,
            not by_area.get("deployment"),
            tuple(by_area.get("deployment", ())),
        ),
    )
    return ProjectSnapshot(
        config.workflow.name,
        not issues,
        tuple(issues),
        areas,
        stages,
    )
