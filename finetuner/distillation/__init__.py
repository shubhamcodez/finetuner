"""Teacher/student knowledge-distillation pipelines."""

from finetuner.distillation.config import (
    DistillationConfig,
    DistillationTechnique,
    DomainSelection,
)

__all__ = ["DistillationConfig", "DistillationTechnique", "DomainSelection"]
