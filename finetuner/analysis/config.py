from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AnalysisConfig:
    reducer: str = "pca"
    pooling: str = "mean"
    layers: list[int] | None = None
    max_samples: int = 200
    max_length: int = 512
    perplexity: float = 30.0
    seed: int = 42
    include_attention_entropy: bool = True

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.reducer not in {"pca", "tsne", "umap"}:
            errors.append("reducer must be pca, tsne, or umap")
        if self.pooling not in {"mean", "last"}:
            errors.append("pooling must be mean or last")
        if not 2 <= self.max_samples <= 100_000:
            errors.append("max_samples must be between 2 and 100,000")
        if not 8 <= self.max_length <= 32_768:
            errors.append("max_length must be between 8 and 32,768")
        if self.perplexity <= 0:
            errors.append("perplexity must be positive")
        return errors

    def require_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "reducer": self.reducer,
            "pooling": self.pooling,
            "layers": self.layers,
            "max_samples": self.max_samples,
            "max_length": self.max_length,
            "perplexity": self.perplexity,
            "seed": self.seed,
            "include_attention_entropy": self.include_attention_entropy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AnalysisConfig:
        data = data or {}
        valid = cls().__dict__.keys()
        return cls(**{key: value for key, value in data.items() if key in valid})
