from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionKind(str, Enum):
    TRAIN = "train"
    DISTILL = "distill"
    EVALUATE = "evaluate"
    ANALYZE = "analyze"
    QUANTIZE = "quantize"
    OPTIMIZE = "optimize"


@dataclass(frozen=True)
class ActionSpec:
    action: ActionKind
    title: str
    area: str
    needs_models: bool
    needs_dataset: bool
    description: str


_ACTIONS: dict[ActionKind, ActionSpec] = {
    ActionKind.TRAIN: ActionSpec(
        ActionKind.TRAIN,
        "Train",
        "training",
        True,
        True,
        "Fine-tune queued models with the selected dataset and method.",
    ),
    ActionKind.DISTILL: ActionSpec(
        ActionKind.DISTILL,
        "Distill",
        "distillation",
        False,
        True,
        "Transfer a teacher into a student. Uses the distillation models, not the queue.",
    ),
    ActionKind.EVALUATE: ActionSpec(
        ActionKind.EVALUATE,
        "Evaluate",
        "evals",
        True,
        False,
        "Score queued models on the selected benchmarks.",
    ),
    ActionKind.ANALYZE: ActionSpec(
        ActionKind.ANALYZE,
        "Analyze",
        "analysis",
        True,
        True,
        "Extract representations from queued models using the selected dataset.",
    ),
    ActionKind.QUANTIZE: ActionSpec(
        ActionKind.QUANTIZE,
        "Deploy",
        "deployment",
        True,
        False,
        "Compress queued models for a concrete backend and device.",
    ),
    ActionKind.OPTIMIZE: ActionSpec(
        ActionKind.OPTIMIZE,
        "Optimize inference",
        "inference",
        True,
        False,
        "Plan or compile a serving engine for queued model artifacts.",
    ),
}


def action_specs() -> list[ActionSpec]:
    return list(_ACTIONS.values())


def get_action(action: ActionKind | str) -> ActionSpec:
    return _ACTIONS[ActionKind(action)]


@dataclass
class ActionEvent:
    run_id: str
    action: str
    action_name: str
    status: str
    index: int
    total: int
    subject: str = ""
    message: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_names: tuple[str, ...] = ()


@dataclass
class ActionOutput:
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ActionCancelled(RuntimeError):
    pass
