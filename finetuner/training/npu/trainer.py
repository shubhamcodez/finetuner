from __future__ import annotations

from typing import Callable

from finetuner.core.job import TrainingConfig


def train_npu_sft(
    model_path: str,
    output_dir: str,
    training: TrainingConfig,
    dataset,
    log_callback: Callable[[str], None] | None = None,
) -> str:
    from finetuner.training.accel.engine import train_accel

    return train_accel(model_path, output_dir, training, dataset, log_callback)
