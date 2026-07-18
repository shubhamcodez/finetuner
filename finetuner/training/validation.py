from __future__ import annotations

from dataclasses import fields, replace

from finetuner.core.job import TrainingConfig
from finetuner.training.methods import get_method


def training_config_for_stage(base: TrainingConfig, parameters: dict) -> TrainingConfig:
    allowed = {item.name for item in fields(TrainingConfig)}
    overrides = {key: value for key, value in parameters.items() if key in allowed}
    return replace(base, **overrides)


def validate_training_config(config: TrainingConfig, method_id: str | None = None) -> list[str]:
    method = method_id or config.training_method
    errors: list[str] = []
    if get_method(method) is None:
        errors.append(f"unknown training method {method!r}")
    if config.max_steps <= 0:
        errors.append("max_steps must be positive")
    if not 0 < config.learning_rate <= 0.1:
        errors.append("learning_rate must be greater than 0 and at most 0.1")
    if config.lora_rank <= 0 or config.lora_alpha <= 0:
        errors.append("LoRA rank and alpha must be positive")
    if len(set(config.lora_target_modules)) != len(config.lora_target_modules):
        errors.append("LoRA target modules must not contain duplicates")
    if config.batch_size <= 0 or config.gradient_accumulation_steps <= 0:
        errors.append("batch size and gradient accumulation must be positive")
    if config.max_seq_length < 32:
        errors.append("max_seq_length must be at least 32")
    if config.dpo_beta <= 0:
        errors.append("DPO/KTO beta must be positive")
    if config.grpo_num_generations <= 0:
        errors.append("online RL generations must be positive")
    if method in {"grpo", "rloo"}:
        effective_batch = config.batch_size * config.gradient_accumulation_steps
        if effective_batch % config.grpo_num_generations != 0:
            errors.append(
                f"{method.upper()} generations ({config.grpo_num_generations}) must divide "
                f"effective batch size ({effective_batch})"
            )
    return errors
