"""Run a short NPU-SFT pass on the bundled GSM8K sample."""

from __future__ import annotations

import json
from pathlib import Path

from finetuner.core.job import TrainingConfig
from finetuner.datasets.presets import FORMATTERS
from finetuner.datasets.rows import row_to_text
from finetuner.training.npu.trainer import train_npu_sft


def _bundled_rows() -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "assets" / "datasets" / "gsm8k_sample.jsonl"
    formatter = FORMATTERS["gsm8k"]
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        text = row_to_text(json.loads(line), formatter)
        if text:
            rows.append({"text": text})
    return rows


def main() -> None:
    training = TrainingConfig(
        training_method="npu",
        max_steps=3,
        learning_rate=5e-3,
        lora_rank=8,
        lora_alpha=16,
        gradient_accumulation_steps=1,
        max_seq_length=128,
        use_qlora=False,
        seed=42,
    )
    output = Path.home() / ".finetuner" / "bench" / "npu-sft"
    path = train_npu_sft("", str(output), training, _bundled_rows(), print)
    print("adapter", path)


if __name__ == "__main__":
    main()
