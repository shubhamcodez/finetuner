from __future__ import annotations

from dataclasses import replace

from finetuner.core.job import TrainingConfig

QUALITY_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def apply_quality_recipe(training: TrainingConfig) -> TrainingConfig:
    """LoRA on the real transformer, chat template, assistant-only loss.

    The NPU logit adapter cannot change hidden states, so it barely moves
    GSM8K/HellaSwag/ARC. This recipe trains the attention and MLP adapters
    the model actually uses at inference.
    """
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    seq = training.max_seq_length
    if seq >= 1024:
        seq = 768 if cuda else 384
    rank = max(training.lora_rank, 32)
    accum = training.gradient_accumulation_steps
    if accum < 4:
        accum = 8 if cuda else 8
    steps = training.max_steps
    if steps < 200:
        steps = 400 if cuda else 200
    targets = list(training.lora_target_modules) or list(QUALITY_LORA_TARGETS)
    return replace(
        training,
        quality_recipe=True,
        use_chat_template=True,
        use_qlora=cuda and training.use_qlora,
        accelerator="cuda" if training.accelerator in {"", "auto", "npu", "tpu", "cpu"} else training.accelerator,
        max_seq_length=seq,
        lora_rank=rank,
        lora_alpha=max(training.lora_alpha, rank * 2),
        lora_target_modules=targets,
        gradient_accumulation_steps=accum,
        max_steps=steps,
        learning_rate=min(max(training.learning_rate, 1e-4), 3e-4),
        batch_size=1,
    )
