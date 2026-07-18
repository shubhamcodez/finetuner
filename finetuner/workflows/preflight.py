from __future__ import annotations

from finetuner.analysis.config import AnalysisConfig
from finetuner.distillation.config import DistillationConfig
from finetuner.quantization.specs import QuantizationConfig
from finetuner.training.methods import get_method
from finetuner.training.validation import training_config_for_stage, validate_training_config
from finetuner.workflows.schema import StageKind, WorkflowSpec


class PreflightError(ValueError):
    pass


def validate_project_workflow(project_config) -> None:
    workflow: WorkflowSpec = project_config.workflow
    workflow.validate()
    errors: list[str] = []
    ordered = workflow.topological_stages()
    for stage in ordered:
        prefix = f"{stage.name}: "
        if stage.kind == StageKind.TRAIN:
            method_id = str(
                stage.parameters.get("method", project_config.training.training_method or "sft")
            )
            if get_method(method_id) is None:
                errors.append(prefix + f"unknown training method {method_id!r}")
            stage_training = training_config_for_stage(project_config.training, stage.parameters)
            errors.extend(
                prefix + message
                for message in validate_training_config(stage_training, method_id)
                if not (
                    message.startswith("unknown training method") and get_method(method_id) is None
                )
            )
            if method_id == "ppo" and not project_config.training.reward_model_id:
                dependency_methods = {
                    workflow.stage(dependency).parameters.get("method")
                    for dependency in stage.depends_on
                }
                if "reward" not in dependency_methods:
                    errors.append(
                        prefix + "PPO needs a reward-model dependency or Training Reward Model ID"
                    )
        elif stage.kind == StageKind.DISTILL:
            payload = project_config.distillation.to_dict()
            payload.update(stage.parameters)
            errors.extend(
                prefix + message for message in DistillationConfig.from_dict(payload).validate()
            )
        elif stage.kind == StageKind.QUANTIZE:
            payload = project_config.quantization.to_dict()
            payload.update(stage.parameters)
            errors.extend(
                prefix + message for message in QuantizationConfig.from_dict(payload).validate()
            )
        elif stage.kind == StageKind.ANALYZE:
            payload = project_config.analysis.to_dict()
            payload.update(stage.parameters)
            errors.extend(
                prefix + message for message in AnalysisConfig.from_dict(payload).validate()
            )
        elif stage.kind == StageKind.EVALUATE:
            task_ids = stage.parameters.get("task_ids", project_config.enabled_evals)
            if not task_ids:
                errors.append(prefix + "select at least one evaluation task")
    if errors:
        raise PreflightError("\n".join(errors))
