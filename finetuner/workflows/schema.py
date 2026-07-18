from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class WorkflowValidationError(ValueError):
    """Raised when a workflow cannot be executed safely."""


class StageKind(str, Enum):
    TRAIN = "train"
    DISTILL = "distill"
    QUANTIZE = "quantize"
    EVALUATE = "evaluate"
    ANALYZE = "analyze"


@dataclass(frozen=True)
class WorkflowStage:
    stage_id: str
    name: str
    kind: StageKind
    depends_on: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.stage_id,
            "name": self.name,
            "kind": self.kind.value,
            "depends_on": list(self.depends_on),
            "parameters": dict(self.parameters),
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowStage:
        try:
            kind = StageKind(data["kind"])
        except (KeyError, ValueError) as exc:
            raise WorkflowValidationError(
                f"Unknown workflow stage kind: {data.get('kind')!r}"
            ) from exc
        stage_id = str(data.get("id", "")).strip()
        return cls(
            stage_id=stage_id,
            name=str(data.get("name") or stage_id).strip(),
            kind=kind,
            depends_on=tuple(str(item) for item in data.get("depends_on", ())),
            parameters=dict(data.get("parameters") or {}),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass(frozen=True)
class WorkflowSpec:
    workflow_id: str
    name: str
    stages: tuple[WorkflowStage, ...]
    description: str = ""
    schema_version: int = 1

    def validate(self) -> None:
        errors: list[str] = []
        if not _ID_PATTERN.fullmatch(self.workflow_id):
            errors.append(
                "workflow id must start with a lowercase letter and contain only "
                "lowercase letters, numbers, '_' or '-'"
            )
        if self.schema_version != 1:
            errors.append(f"unsupported workflow schema version: {self.schema_version}")
        if not self.name.strip():
            errors.append("workflow name is required")
        enabled = [stage for stage in self.stages if stage.enabled]
        if not enabled:
            errors.append("workflow must contain at least one enabled stage")

        ids: set[str] = set()
        all_ids = {stage.stage_id for stage in self.stages}
        for stage in self.stages:
            if not _ID_PATTERN.fullmatch(stage.stage_id):
                errors.append(f"invalid stage id: {stage.stage_id!r}")
            if stage.stage_id in ids:
                errors.append(f"duplicate stage id: {stage.stage_id}")
            ids.add(stage.stage_id)
            if not stage.name.strip():
                errors.append(f"stage {stage.stage_id!r} has no name")
            if stage.stage_id in stage.depends_on:
                errors.append(f"stage {stage.stage_id!r} depends on itself")
            missing = sorted(set(stage.depends_on) - all_ids)
            if missing:
                errors.append(
                    f"stage {stage.stage_id!r} has missing dependencies: {', '.join(missing)}"
                )
            disabled_dependencies = [
                dep
                for dep in stage.depends_on
                if dep in all_ids
                and not next(item for item in self.stages if item.stage_id == dep).enabled
            ]
            if stage.enabled and disabled_dependencies:
                errors.append(
                    f"stage {stage.stage_id!r} depends on disabled stages: "
                    f"{', '.join(disabled_dependencies)}"
                )

        if not errors:
            try:
                self.topological_stages()
            except WorkflowValidationError as exc:
                errors.append(str(exc))
        if errors:
            raise WorkflowValidationError("; ".join(errors))

    def topological_stages(self) -> list[WorkflowStage]:
        """Return enabled stages in deterministic dependency order."""
        stages = {stage.stage_id: stage for stage in self.stages if stage.enabled}
        indegree = {
            stage_id: sum(dep in stages for dep in stage.depends_on)
            for stage_id, stage in stages.items()
        }
        children: dict[str, list[str]] = {stage_id: [] for stage_id in stages}
        for stage in stages.values():
            for dependency in stage.depends_on:
                if dependency in children:
                    children[dependency].append(stage.stage_id)

        order_index = {stage.stage_id: index for index, stage in enumerate(self.stages)}
        ready = sorted(
            (stage_id for stage_id, count in indegree.items() if count == 0),
            key=order_index.__getitem__,
        )
        result: list[WorkflowStage] = []
        while ready:
            stage_id = ready.pop(0)
            result.append(stages[stage_id])
            for child in sorted(children[stage_id], key=order_index.__getitem__):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
            ready.sort(key=order_index.__getitem__)

        if len(result) != len(stages):
            cyclic = sorted(stage_id for stage_id, count in indegree.items() if count > 0)
            raise WorkflowValidationError(
                f"workflow contains a dependency cycle involving: {', '.join(cyclic)}"
            )
        return result

    def stage(self, stage_id: str) -> WorkflowStage:
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(stage_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, validate: bool = True) -> WorkflowSpec:
        workflow = cls(
            workflow_id=str(data.get("id", "")).strip(),
            name=str(data.get("name", "")).strip(),
            description=str(data.get("description", "")).strip(),
            schema_version=int(data.get("schema_version", 1)),
            stages=tuple(WorkflowStage.from_dict(item) for item in data.get("stages", ())),
        )
        if validate:
            workflow.validate()
        return workflow


def validate_workflows(workflows: Iterable[WorkflowSpec]) -> None:
    seen: set[str] = set()
    for workflow in workflows:
        workflow.validate()
        if workflow.workflow_id in seen:
            raise WorkflowValidationError(f"duplicate workflow id: {workflow.workflow_id}")
        seen.add(workflow.workflow_id)
