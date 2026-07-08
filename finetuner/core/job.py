from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any


class ModelSource(str, Enum):
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class JobStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ModelJob:
    name: str
    source: ModelSource
    identifier: str
    status: JobStatus = JobStatus.PENDING
    output_path: str = ""
    error: str = ""

    def display_source(self) -> str:
        if self.source == ModelSource.HUGGINGFACE:
            return f"HF: {self.identifier}"
        return f"Local: {self.identifier}"


@dataclass
class TrainingConfig:
    dataset_path: str = ""
    dataset_hf_id: str = ""
    dataset_preset_id: str = ""
    dataset_use_bundled_only: bool = False
    training_method: str = "sft"
    reward_function: str = "exact_match"
    reward_model_id: str = ""
    dpo_beta: float = 0.1
    grpo_num_generations: int = 2
    ppo_kl_coef: float = 0.05
    ppo_cliprange: float = 0.2
    max_steps: int = 100
    learning_rate: float = 2e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    use_qlora: bool = True


@dataclass
class EvalResult:
    task_id: str
    task_name: str
    score: float
    metric: str = "accuracy"


@dataclass
class ModelRunResult:
    model_name: str
    model_identifier: str
    output_path: str
    eval_results: list[EvalResult] = field(default_factory=list)
    training_error: str = ""


@dataclass
class ProjectConfig:
    models: list[ModelJob] = field(default_factory=list)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    enabled_evals: list[str] = field(default_factory=lambda: ["mmlu", "gsm8k"])
    hf_token: str = ""
    eval_max_samples: int = 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": [
                {
                    "name": m.name,
                    "source": m.source.value,
                    "identifier": m.identifier,
                    "status": m.status.value,
                    "output_path": m.output_path,
                    "error": m.error,
                }
                for m in self.models
            ],
            "training": {
                "dataset_path": self.training.dataset_path,
                "dataset_hf_id": self.training.dataset_hf_id,
                "dataset_preset_id": self.training.dataset_preset_id,
                "dataset_use_bundled_only": self.training.dataset_use_bundled_only,
                "training_method": self.training.training_method,
                "reward_function": self.training.reward_function,
                "reward_model_id": self.training.reward_model_id,
                "dpo_beta": self.training.dpo_beta,
                "grpo_num_generations": self.training.grpo_num_generations,
                "ppo_kl_coef": self.training.ppo_kl_coef,
                "ppo_cliprange": self.training.ppo_cliprange,
                "max_steps": self.training.max_steps,
                "learning_rate": self.training.learning_rate,
                "lora_rank": self.training.lora_rank,
                "lora_alpha": self.training.lora_alpha,
                "batch_size": self.training.batch_size,
                "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
                "max_seq_length": self.training.max_seq_length,
                "use_qlora": self.training.use_qlora,
            },
            "enabled_evals": self.enabled_evals,
            "hf_token": self.hf_token,
            "eval_max_samples": self.eval_max_samples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectConfig:
        training_data = data.get("training", {})
        valid = {f.name for f in fields(TrainingConfig)}
        filtered = {k: v for k, v in training_data.items() if k in valid}
        models = [
            ModelJob(
                name=m["name"],
                source=ModelSource(m["source"]),
                identifier=m["identifier"],
                status=JobStatus(m.get("status", JobStatus.PENDING.value)),
                output_path=m.get("output_path", ""),
                error=m.get("error", ""),
            )
            for m in data.get("models", [])
        ]
        return cls(
            models=models,
            training=TrainingConfig(**filtered) if filtered else TrainingConfig(),
            enabled_evals=data.get("enabled_evals", ["mmlu", "gsm8k"]),
            hf_token=data.get("hf_token", ""),
            eval_max_samples=data.get("eval_max_samples", 100),
        )


def resolve_model_path(job: ModelJob) -> Path:
    if job.source == ModelSource.LOCAL:
        return Path(job.identifier)
    from finetuner.core.paths import model_download_path

    return model_download_path(job.identifier)
