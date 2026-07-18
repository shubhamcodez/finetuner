from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finetuner.core.artifacts import atomic_write_json


@dataclass
class RepresentationPoint:
    sample_id: str
    label: str
    text: str
    x: float
    y: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class LayerAnalysis:
    layer: int
    points: list[RepresentationPoint]
    activation_norm_mean: float
    activation_norm_std: float
    attention_entropy_mean: float | None = None
    reducer_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "points": [point.to_dict() for point in self.points],
            "activation_norm_mean": self.activation_norm_mean,
            "activation_norm_std": self.activation_norm_std,
            "attention_entropy_mean": self.attention_entropy_mean,
            "reducer_metadata": self.reducer_metadata,
        }


def write_analysis_artifact(
    path: str | Path,
    *,
    model: str,
    reducer: str,
    pooling: str,
    layers: list[LayerAnalysis],
    cka: list[list[float]],
) -> str:
    destination = Path(path)
    atomic_write_json(
        destination,
        {
            "schema_version": 1,
            "model": model,
            "reducer": reducer,
            "pooling": pooling,
            "layers": [layer.to_dict() for layer in layers],
            "cka": cka,
        },
    )
    return str(destination.resolve())
