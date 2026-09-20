from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from finetuner.core.actions import ActionCancelled, ActionEvent, ActionKind, ActionOutput, get_action
from finetuner.core.artifacts import ArtifactRecord, RunManifest, artifact_metadata, stable_digest


@dataclass
class ActionContext:
    run_id: str
    run_dir: Path
    model_path: str
    dataset_path: str
    project_config: Any
    action: ActionKind
    log: Callable[[str], None] = lambda _message: None
    is_cancelled: Callable[[], bool] = lambda: False
    event_callback: Callable[[ActionEvent], None] = lambda _event: None
    subject: str = ""
    index: int = 1
    total: int = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_train(context: ActionContext) -> ActionOutput:
    from finetuner.training.runner import train

    trained_path = train(
        model_path=context.model_path,
        output_dir=str(context.run_dir / "train"),
        training=context.project_config.training,
        dataset_path=context.dataset_path,
        log_callback=context.log,
    )
    method = context.project_config.training.training_method
    artifact_name = "reward_model" if method == "reward" else "policy_model"
    return ActionOutput(artifacts={artifact_name: trained_path}, metadata={"method": method})


def run_distill(context: ActionContext) -> ActionOutput:
    from finetuner.distillation.runner import run_distillation

    student, manifest = run_distillation(
        context.dataset_path,
        str(context.run_dir / "distill"),
        context.project_config.distillation,
        context.project_config.training,
        context.log,
    )
    config = context.project_config.distillation
    return ActionOutput(
        artifacts={"policy_model": student, "distillation_manifest": manifest},
        metadata={"technique": config.technique, "domain": config.domain.to_dict()},
    )


def run_evaluate(context: ActionContext) -> ActionOutput:
    from finetuner.eval.runner import run_evals

    results = run_evals(
        context.model_path,
        context.project_config.enabled_evals,
        context.project_config.eval_max_samples,
        context.log,
    )
    metrics = {result.task_id: result.score for result in results}
    return ActionOutput(artifacts={"eval_results": results}, metrics=metrics)


def run_analyze(context: ActionContext) -> ActionOutput:
    from finetuner.analysis.runner import analyze_model

    artifact = analyze_model(
        context.model_path,
        context.dataset_path,
        str(context.run_dir / "analyze"),
        context.project_config.analysis,
        context.log,
    )
    return ActionOutput(
        artifacts={"analysis": artifact}, metadata=context.project_config.analysis.to_dict()
    )


def run_quantize(context: ActionContext) -> ActionOutput:
    from finetuner.quantization.runner import quantize_model

    config = context.project_config.quantization
    quantized = quantize_model(
        context.model_path,
        str(context.run_dir / "quantize"),
        config,
        context.log,
    )
    return ActionOutput(
        artifacts={"deployment_model": quantized},
        metadata={"backend": config.backend, "target": config.target, "bits": config.bits},
    )


def run_optimize(context: ActionContext) -> ActionOutput:
    from finetuner.inference.devices import apply_best_runtime
    from finetuner.inference.runner import optimize_inference_engine
    from finetuner.quantization.specs import DeviceTarget

    config = context.project_config.inference
    if config.target == DeviceTarget.AUTO.value:
        choice = apply_best_runtime(
            context.project_config,
            model_path=context.model_path,
            run_on_device=bool((config.extra_options or {}).get("run_on_device")),
        )
        context.log(choice.reason)
        for skipped in choice.skipped:
            context.log(skipped)
        config = context.project_config.inference
    optimized = optimize_inference_engine(
        context.model_path,
        str(context.run_dir / "optimize"),
        config,
        context.log,
    )
    return ActionOutput(
        artifacts={"inference_engine": optimized},
        metadata={"engine": config.engine, "target": config.target, "compiled": config.compile},
    )


def _handler(action: ActionKind):
    return {
        ActionKind.TRAIN: run_train,
        ActionKind.DISTILL: run_distill,
        ActionKind.EVALUATE: run_evaluate,
        ActionKind.ANALYZE: run_analyze,
        ActionKind.QUANTIZE: run_quantize,
        ActionKind.OPTIMIZE: run_optimize,
    }[action]


def execute_action(context: ActionContext) -> ActionOutput:
    spec = get_action(context.action)
    handler = _handler(spec.action)
    context.run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = context.run_dir / "manifest.json"
    safe_config = (
        context.project_config.to_dict()
        if hasattr(context.project_config, "to_dict")
        else context.project_config
    )
    manifest = RunManifest(
        run_id=context.run_id,
        action={"id": spec.action.value, "name": spec.title},
        config_digest=stable_digest(safe_config),
    )
    manifest.save(manifest_path)
    record: dict[str, Any] = {
        "id": spec.action.value,
        "kind": spec.action.value,
        "status": "running",
        "started_at": _now(),
    }
    manifest.stages.append(record)
    manifest.save(manifest_path)
    _notify(
        context,
        ActionEvent(
            context.run_id,
            spec.action.value,
            spec.title,
            "running",
            context.index,
            context.total,
            context.subject,
        ),
    )
    started = time.monotonic()
    try:
        if context.is_cancelled():
            raise ActionCancelled("Run cancelled by user")
        context.log(f"\n--- {spec.title} ---")
        output = handler(context)
    except Exception as exc:
        record.update(
            status="failed",
            finished_at=_now(),
            duration_seconds=round(time.monotonic() - started, 3),
            error=f"{type(exc).__name__}: {exc}",
        )
        manifest.finish("cancelled" if isinstance(exc, ActionCancelled) else "failed")
        manifest.save(manifest_path)
        _notify(
            context,
            ActionEvent(
                context.run_id,
                spec.action.value,
                spec.title,
                "failed",
                context.index,
                context.total,
                context.subject,
                str(exc),
            ),
        )
        raise
    record.update(
        status="completed",
        finished_at=_now(),
        duration_seconds=round(time.monotonic() - started, 3),
        metrics=output.metrics,
        metadata=output.metadata,
    )
    for name, value in output.artifacts.items():
        if isinstance(value, (str, Path)):
            manifest.artifacts.append(
                ArtifactRecord(
                    name=name,
                    kind=name,
                    uri=str(value),
                    producer_stage=spec.action.value,
                    metadata=artifact_metadata(value),
                )
            )
    manifest.finish("completed")
    manifest.save(manifest_path)
    _notify(
        context,
        ActionEvent(
            context.run_id,
            spec.action.value,
            spec.title,
            "completed",
            context.index,
            context.total,
            context.subject,
            metrics=dict(output.metrics),
            artifact_names=tuple(output.artifacts),
        ),
    )
    output.artifacts["manifest"] = str(manifest_path)
    return output


def _notify(context: ActionContext, event: ActionEvent) -> None:
    try:
        context.event_callback(event)
    except Exception as exc:
        context.log(f"Action progress listener failed: {exc}")
