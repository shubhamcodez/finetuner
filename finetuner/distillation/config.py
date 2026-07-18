from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DistillationTechnique(str, Enum):
    SEQUENCE = "sequence"
    LOGIT_KL = "logit_kl"
    ON_POLICY_GKD = "on_policy_gkd"
    REVERSE_KL = "reverse_kl"


@dataclass
class DomainSelection:
    mode: str = "all"
    fields: list[str] = field(default_factory=list)
    custom: str = ""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.mode not in {"all", "presets", "custom"}:
            errors.append("domain mode must be all, presets, or custom")
        if self.mode == "presets" and not self.fields:
            errors.append("select at least one domain preset")
        if self.mode == "custom" and not self.custom.strip():
            errors.append("custom domain is required")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "fields": list(self.fields), "custom": self.custom}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DomainSelection:
        data = data or {}
        return cls(
            mode=str(data.get("mode", "all")),
            fields=[str(item) for item in data.get("fields", ())],
            custom=str(data.get("custom", "")),
        )


@dataclass
class DistillationConfig:
    teacher_model: str = ""
    student_model: str = ""
    technique: str = DistillationTechnique.SEQUENCE.value
    domain: DomainSelection = field(default_factory=DomainSelection)
    prompt_field: str = "prompt"
    response_field: str = "response"
    max_samples: int = 1000
    max_new_tokens: int = 512
    temperature: float = 0.7
    kd_temperature: float = 1.0
    alpha: float = 0.5
    student_generated_fraction: float = 0.5
    seed: int = 42

    def validate(self) -> list[str]:
        errors = self.domain.validate()
        if not self.teacher_model.strip():
            errors.append("teacher model is required")
        if not self.student_model.strip():
            errors.append("student model is required")
        if self.teacher_model.strip() == self.student_model.strip():
            errors.append("teacher and student models must be different")
        try:
            DistillationTechnique(self.technique)
        except ValueError:
            errors.append(f"unknown distillation technique: {self.technique}")
        if not 1 <= self.max_samples <= 10_000_000:
            errors.append("max_samples must be between 1 and 10,000,000")
        if not 1 <= self.max_new_tokens <= 32_768:
            errors.append("max_new_tokens must be between 1 and 32,768")
        if not 0 <= self.temperature <= 5:
            errors.append("temperature must be between 0 and 5")
        if self.kd_temperature <= 0:
            errors.append("kd_temperature must be positive")
        if not 0 <= self.alpha <= 1:
            errors.append("alpha must be between 0 and 1")
        if not 0 <= self.student_generated_fraction <= 1:
            errors.append("student_generated_fraction must be between 0 and 1")
        return errors

    def require_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "teacher_model": self.teacher_model,
            "student_model": self.student_model,
            "technique": self.technique,
            "domain": self.domain.to_dict(),
            "prompt_field": self.prompt_field,
            "response_field": self.response_field,
            "max_samples": self.max_samples,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "kd_temperature": self.kd_temperature,
            "alpha": self.alpha,
            "student_generated_fraction": self.student_generated_fraction,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DistillationConfig:
        data = dict(data or {})
        data["domain"] = DomainSelection.from_dict(data.get("domain"))
        valid = cls().__dict__.keys()
        return cls(**{key: value for key, value in data.items() if key in valid})
