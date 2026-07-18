from __future__ import annotations

from dataclasses import fields, replace

from finetuner.analysis.config import AnalysisConfig
from finetuner.distillation.config import DistillationConfig
from finetuner.quantization.specs import QuantizationConfig
from finetuner.workflows.executor import (
    StageOutput,
    WorkflowContext,
    dependency_artifact,
)
from finetuner.workflows.schema import StageKind, WorkflowStage


def _policy_input(context: WorkflowContext, dependencies: dict[str, StageOutput]) -> str:
    return str(dependency_artifact(dependencies, "policy_model", context.model_path))


def _merge_parameters(config, parameters: dict, excluded: set[str] | None = None):
    excluded = excluded or set()
    allowed = {item.name for item in fields(config)} - excluded
    overrides = {key: value for key, value in parameters.items() if key in allowed}
    return replace(config, **overrides)


def train_stage(stage: WorkflowStage, context: WorkflowContext, dependencies) -> StageOutput:
    from finetuner.training.runner import train

    method = str(
        stage.parameters.get("method", context.project_config.training.training_method or "sft")
    )
    training = _merge_parameters(
        context.project_config.training,
        stage.parameters,
        {"training_method"},
    )
    training = replace(training, training_method=method)
    reward_model = dependency_artifact(dependencies, "reward_model", "")
    if method == "ppo" and reward_model:
        training = replace(training, reward_model_id=str(reward_model))
    model_path = _policy_input(context, dependencies)
    stage_dir = context.run_dir / stage.stage_id
    trained_path = train(
        model_path=model_path,
        output_dir=str(stage_dir),
        training=training,
        dataset_path=context.dataset_path,
        log_callback=context.log,
    )
    artifact_name = "reward_model" if method == "reward" else "policy_model"
    return StageOutput(artifacts={artifact_name: trained_path}, metadata={"method": method})


def distill_stage(stage: WorkflowStage, context: WorkflowContext, dependencies) -> StageOutput:
    from finetuner.distillation.runner import run_distillation

    base = context.project_config.distillation.to_dict()
    base.update(stage.parameters)
    config = DistillationConfig.from_dict(base)
    student, manifest = run_distillation(
        context.dataset_path,
        str(context.run_dir / stage.stage_id),
        config,
        context.project_config.training,
        context.log,
    )
    return StageOutput(
        artifacts={"policy_model": student, "distillation_manifest": manifest},
        metadata={"technique": config.technique, "domain": config.domain.to_dict()},
    )


def quantize_stage(stage: WorkflowStage, context: WorkflowContext, dependencies) -> StageOutput:
    from finetuner.quantization.runner import quantize_model

    base = context.project_config.quantization.to_dict()
    base.update(stage.parameters)
    config = QuantizationConfig.from_dict(base)
    quantized = quantize_model(
        _policy_input(context, dependencies),
        str(context.run_dir / stage.stage_id),
        config,
        context.log,
    )
    return StageOutput(
        artifacts={"deployment_model": quantized},
        metadata={"backend": config.backend, "target": config.target, "bits": config.bits},
    )


def evaluate_stage(stage: WorkflowStage, context: WorkflowContext, dependencies) -> StageOutput:
    from finetuner.eval.runner import run_evals

    task_ids = list(stage.parameters.get("task_ids", context.project_config.enabled_evals))
    max_samples = int(stage.parameters.get("max_samples", context.project_config.eval_max_samples))
    results = run_evals(_policy_input(context, dependencies), task_ids, max_samples, context.log)
    metrics = {result.task_id: result.score for result in results}
    return StageOutput(artifacts={"eval_results": results}, metrics=metrics)


def analyze_stage(stage: WorkflowStage, context: WorkflowContext, dependencies) -> StageOutput:
    from finetuner.analysis.runner import analyze_model

    base = context.project_config.analysis.to_dict()
    base.update(stage.parameters)
    config = AnalysisConfig.from_dict(base)
    artifact = analyze_model(
        _policy_input(context, dependencies),
        context.dataset_path,
        str(context.run_dir / stage.stage_id),
        config,
        context.log,
    )
    return StageOutput(artifacts={"analysis": artifact}, metadata=config.to_dict())


def production_handlers():
    return {
        StageKind.TRAIN: train_stage,
        StageKind.DISTILL: distill_stage,
        StageKind.QUANTIZE: quantize_stage,
        StageKind.EVALUATE: evaluate_stage,
        StageKind.ANALYZE: analyze_stage,
    }
