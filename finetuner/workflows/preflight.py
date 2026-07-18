from __future__ import annotations

from dataclasses import dataclass

from finetuner.analysis.config import AnalysisConfig
from finetuner.distillation.config import DistillationConfig
from finetuner.quantization.specs import QuantizationConfig
from finetuner.training.methods import get_method
from finetuner.training.validation import training_config_for_stage, validate_training_config
from finetuner.workflows.schema import StageKind, WorkflowSpec


class PreflightError(ValueError):
    pass


@dataclass(frozen=True)
class PreflightIssue:
    area: str
    message: str
    stage_id: str = ""


_STAGE_AREAS = {
    StageKind.TRAIN: "training",
    StageKind.DISTILL: "distillation",
    StageKind.QUANTIZE: "deployment",
    StageKind.EVALUATE: "evals",
    StageKind.ANALYZE: "analysis",
}


def collect_workflow_issues(project_config) -> list[PreflightIssue]:
    workflow: WorkflowSpec = project_config.workflow
    try:
        workflow.validate()
    except ValueError as exc:
        return [PreflightIssue("workflow", str(exc))]
    issues: list[PreflightIssue] = []
    ordered = workflow.topological_stages()
    for stage in ordered:
        area = _STAGE_AREAS[stage.kind]

        def add(message: str) -> None:
            issues.append(PreflightIssue(area, message, stage.stage_id))

        if stage.kind == StageKind.TRAIN:
            method_id = str(
                stage.parameters.get("method", project_config.training.training_method or "sft")
            )
            if get_method(method_id) is None:
                add(f"unknown training method {method_id!r}")
            stage_training = training_config_for_stage(project_config.training, stage.parameters)
            for message in validate_training_config(stage_training, method_id):
                if message.startswith("unknown training method") and get_method(method_id) is None:
                    continue
                add(message)
            if method_id == "ppo" and not project_config.training.reward_model_id:
                dependency_methods = {
                    workflow.stage(dependency).parameters.get("method")
                    for dependency in stage.depends_on
                }
                if "reward" not in dependency_methods:
                    add("PPO needs a reward-model dependency or Training Reward Model ID")
        elif stage.kind == StageKind.DISTILL:
            payload = project_config.distillation.to_dict()
            payload.update(stage.parameters)
            for message in DistillationConfig.from_dict(payload).validate():
                add(message)
        elif stage.kind == StageKind.QUANTIZE:
            payload = project_config.quantization.to_dict()
            payload.update(stage.parameters)
            for message in QuantizationConfig.from_dict(payload).validate():
                add(message)
        elif stage.kind == StageKind.ANALYZE:
            payload = project_config.analysis.to_dict()
            payload.update(stage.parameters)
            for message in AnalysisConfig.from_dict(payload).validate():
                add(message)
        elif stage.kind == StageKind.EVALUATE:
            task_ids = stage.parameters.get("task_ids", project_config.enabled_evals)
            if not task_ids:
                add("select at least one evaluation task")
    return issues


def validate_project_workflow(project_config) -> None:
    issues = collect_workflow_issues(project_config)
    if issues:
        workflow = project_config.workflow
        labels = {stage.stage_id: stage.name for stage in workflow.stages}
        raise PreflightError(
            "\n".join(
                f"{labels.get(issue.stage_id, issue.area.title())}: {issue.message}"
                for issue in issues
            )
        )
