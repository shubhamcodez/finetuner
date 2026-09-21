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
    """Chat template + assistant-only loss on the real transformer.

    Full-weight SFT is the default when LoRA is off. LoRA/QLoRA stay opt-in
    via the training toggles; this recipe only raises adapter rank if those
    toggles are already on. The NPU logit adapter cannot change hidden
    states, so this path always leaves the frozen-decoder engine.
    """
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    seq = training.max_seq_length
    if seq >= 1024:
        seq = 768 if cuda else 384
    accum = training.gradient_accumulation_steps
    if accum < 4:
        accum = 8
    steps = training.max_steps
    if steps < 200:
        steps = 400 if cuda else 200
    updates: dict = {
        "quality_recipe": True,
        "use_chat_template": True,
        "accelerator": (
            "cuda"
            if training.accelerator in {"", "auto", "npu", "tpu", "cpu"}
            else training.accelerator
        ),
        "max_seq_length": seq,
        "gradient_accumulation_steps": accum,
        "max_steps": steps,
        "batch_size": 1,
    }
    if training.uses_lora():
        rank = max(training.lora_rank, 32)
        targets = list(training.lora_target_modules) or list(QUALITY_LORA_TARGETS)
        updates.update(
            use_lora=True,
            use_qlora=cuda and training.use_qlora,
            lora_rank=rank,
            lora_alpha=max(training.lora_alpha, rank * 2),
            lora_target_modules=targets,
            learning_rate=min(max(training.learning_rate, 1e-4), 3e-4),
        )
    else:
        updates.update(
            use_lora=False,
            use_qlora=False,
            learning_rate=min(max(training.learning_rate, 1e-5), 5e-5),
        )
    return replace(training, **updates)
