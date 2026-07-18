from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from finetuner.core.artifacts import (
    ArtifactRecord,
    RunManifest,
    artifact_metadata,
    stable_digest,
)
from finetuner.workflows.schema import StageKind, WorkflowSpec, WorkflowStage


class WorkflowCancelled(RuntimeError):
    pass


@dataclass
class WorkflowContext:
    run_id: str
    run_dir: Path
    model_path: str
    dataset_path: str
    project_config: Any
    log: Callable[[str], None] = lambda _message: None
    is_cancelled: Callable[[], bool] = lambda: False


@dataclass
class StageOutput:
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecutionResult:
    status: str
    outputs: dict[str, StageOutput]
    manifest_path: str

    def latest_artifact(self, name: str, default: Any = None) -> Any:
        for output in reversed(list(self.outputs.values())):
            if name in output.artifacts:
                return output.artifacts[name]
        return default


class StageHandler(Protocol):
    def __call__(
        self,
        stage: WorkflowStage,
        context: WorkflowContext,
        dependencies: dict[str, StageOutput],
    ) -> StageOutput: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkflowExecutor:
    def __init__(self, handlers: dict[StageKind, StageHandler]) -> None:
        self.handlers = dict(handlers)

    def execute(self, workflow: WorkflowSpec, context: WorkflowContext) -> WorkflowExecutionResult:
        workflow.validate()
        context.run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = context.run_dir / "manifest.json"
        safe_config = (
            context.project_config.to_dict()
            if hasattr(context.project_config, "to_dict")
            else context.project_config
        )
        manifest = RunManifest(
            run_id=context.run_id,
            workflow=workflow.to_dict(),
            config_digest=stable_digest(safe_config),
        )
        manifest.save(manifest_path)
        outputs: dict[str, StageOutput] = {}

        try:
            for stage in workflow.topological_stages():
                if context.is_cancelled():
                    raise WorkflowCancelled("Workflow cancelled by user")
                handler = self.handlers.get(stage.kind)
                if handler is None:
                    raise RuntimeError(f"No handler registered for stage kind {stage.kind.value!r}")
                dependencies = {stage_id: outputs[stage_id] for stage_id in stage.depends_on}
                context.log(f"\n--- {stage.name} [{stage.kind.value}] ---")
                started = time.monotonic()
                stage_record: dict[str, Any] = {
                    "id": stage.stage_id,
                    "kind": stage.kind.value,
                    "status": "running",
                    "started_at": _now(),
                }
                manifest.stages.append(stage_record)
                manifest.save(manifest_path)
                try:
                    output = handler(stage, context, dependencies)
                except Exception as exc:
                    stage_record.update(
                        status="failed",
                        finished_at=_now(),
                        duration_seconds=round(time.monotonic() - started, 3),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    raise
                outputs[stage.stage_id] = output
                stage_record.update(
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
                                producer_stage=stage.stage_id,
                                metadata=artifact_metadata(value),
                            )
                        )
                manifest.save(manifest_path)
            manifest.finish("completed")
        except WorkflowCancelled:
            manifest.finish("cancelled")
            manifest.save(manifest_path)
            raise
        except Exception:
            manifest.finish("failed")
            manifest.save(manifest_path)
            raise

        manifest.save(manifest_path)
        return WorkflowExecutionResult("completed", outputs, str(manifest_path))


def dependency_artifact(
    dependencies: dict[str, StageOutput], name: str, default: Any = None
) -> Any:
    for output in reversed(list(dependencies.values())):
        if name in output.artifacts:
            return output.artifacts[name]
    return default
