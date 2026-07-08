from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

RowFormatter = Callable[[dict], str | None]


@dataclass(frozen=True)
class DatasetPreset:
    preset_id: str
    name: str
    description: str
    related_eval_id: str
    hf_dataset: str
    hf_config: str | None = None
    split: str = "train"
    max_samples: int = 500
    bundled_filename: str | None = None


def _format_gsm8k(row: dict) -> str | None:
    question = row.get("question", "")
    answer = row.get("answer", "")
    if not question or not answer:
        return None
    return (
        f"### Instruction:\nSolve this grade-school math word problem.\n"
        f"### Input:\n{question}\n"
        f"### Response:\n{answer}"
    )


def _format_mmlu(row: dict) -> str | None:
    question = row.get("question", "")
    choices = row.get("choices", [])
    answer = row.get("answer", "")
    if not question or not choices or not answer:
        return None
    labels = ["A", "B", "C", "D"]
    choice_lines = []
    answer_text = str(answer)
    for i, choice in enumerate(choices[:4]):
        label = labels[i]
        choice_lines.append(f"{label}) {choice}")
        if answer == label or answer == i:
            answer_text = str(choice)
    return (
        f"### Instruction:\nAnswer the multiple-choice question.\n"
        f"### Input:\n{question}\n"
        + "\n".join(choice_lines)
        + f"\n### Response:\n{answer_text}"
    )


def _format_hellaswag(row: dict) -> str | None:
    ctx = row.get("ctx", "") or row.get("context", "")
    endings = row.get("endings", [])
    label = row.get("label", row.get("labels", None))
    if not ctx or not endings:
        return None
    if isinstance(label, list):
        label = label[0] if label else 0
    try:
        idx = int(label)
    except (TypeError, ValueError):
        idx = 0
    idx = max(0, min(idx, len(endings) - 1))
    completion = endings[idx]
    return (
        f"### Instruction:\nComplete the sentence with the most plausible ending.\n"
        f"### Input:\n{ctx}\n"
        f"### Response:\n{completion}"
    )


def _format_arc(row: dict) -> str | None:
    question = row.get("question", "")
    choices = row.get("choices", {})
    answer_key = row.get("answerKey", "")
    if not question or not choices:
        return None
    texts = choices.get("text", [])
    labels = choices.get("label", [])
    choice_lines = []
    answer_text = answer_key
    for label, text in zip(labels, texts):
        choice_lines.append(f"{label}) {text}")
        if label == answer_key:
            answer_text = text
    return (
        f"### Instruction:\nAnswer the science question.\n"
        f"### Input:\n{question}\n"
        + "\n".join(choice_lines)
        + f"\n### Response:\n{answer_text}"
    )


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "gsm8k": DatasetPreset(
        preset_id="gsm8k",
        name="GSM8K (Math)",
        description="Grade-school math word problems — pairs with GSM8K eval",
        related_eval_id="gsm8k",
        hf_dataset="openai/gsm8k",
        hf_config="main",
        split="train",
        max_samples=500,
        bundled_filename="gsm8k_sample.jsonl",
    ),
    "mmlu": DatasetPreset(
        preset_id="mmlu",
        name="MMLU (Knowledge)",
        description="Multitask multiple-choice QA — pairs with MMLU eval",
        related_eval_id="mmlu",
        hf_dataset="cais/mmlu",
        hf_config="all",
        split="test",
        max_samples=500,
        bundled_filename="mmlu_sample.jsonl",
    ),
    "hellaswag": DatasetPreset(
        preset_id="hellaswag",
        name="HellaSwag (Commonsense)",
        description="Commonsense sentence completion — pairs with HellaSwag eval",
        related_eval_id="hellaswag",
        hf_dataset="Rowan/hellaswag",
        hf_config=None,
        split="train",
        max_samples=500,
        bundled_filename="hellaswag_sample.jsonl",
    ),
    "arc_challenge": DatasetPreset(
        preset_id="arc_challenge",
        name="ARC Challenge (Science)",
        description="Science exam questions — pairs with ARC Challenge eval",
        related_eval_id="arc_challenge",
        hf_dataset="allenai/ai2_arc",
        hf_config="ARC-Challenge",
        split="train",
        max_samples=500,
        bundled_filename="arc_sample.jsonl",
    ),
    "sample": DatasetPreset(
        preset_id="sample",
        name="Bundled sample (smoke test)",
        description="Tiny local mix for quick end-to-end testing",
        related_eval_id="gsm8k",
        hf_dataset="",
        split="train",
        max_samples=50,
        bundled_filename="sample_sft.jsonl",
    ),
}

FORMATTERS: dict[str, RowFormatter] = {
    "gsm8k": _format_gsm8k,
    "mmlu": _format_mmlu,
    "hellaswag": _format_hellaswag,
    "arc_challenge": _format_arc,
    "sample": None,
}


def preset_list() -> list[DatasetPreset]:
    return list(DATASET_PRESETS.values())


def get_preset(preset_id: str) -> DatasetPreset | None:
    return DATASET_PRESETS.get(preset_id)
