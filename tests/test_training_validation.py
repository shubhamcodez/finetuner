from __future__ import annotations

from finetuner.core.job import TrainingConfig
from finetuner.training.validation import validate_training_config


def test_online_rl_generation_count_must_divide_effective_batch():
    config = TrainingConfig(batch_size=1, gradient_accumulation_steps=4, grpo_num_generations=3)
    assert any("must divide" in error for error in validate_training_config(config, "grpo"))
    config.grpo_num_generations = 2
    assert not any("must divide" in error for error in validate_training_config(config, "grpo"))


def test_duplicate_lora_targets_are_rejected():
    config = TrainingConfig(lora_target_modules=["q_proj", "q_proj"])
    assert any("duplicates" in error for error in validate_training_config(config, "sft"))
