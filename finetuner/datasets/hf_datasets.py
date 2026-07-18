from __future__ import annotations

import os
from pathlib import Path


def load_env_file() -> None:
    """Load HF_TOKEN from project .env into the environment if not already set."""
    if os.environ.get("HF_TOKEN"):
        return
    for candidate in (
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ):
        if not candidate.exists():
            continue
        raw = candidate.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        # Bare token file: hf_...
        if raw.startswith("hf_") and "\n" not in raw and "=" not in raw:
            os.environ["HF_TOKEN"] = raw
            return
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key == "HF_TOKEN" and value:
                os.environ["HF_TOKEN"] = value
                return


def extract_gsm8k_answer(text: str) -> str:
    if "####" in text:
        return text.split("####")[-1].strip()
    return text.strip()


# Legacy dataset slugs (pre-namespace Hub IDs) -> current repo ids.
DATASET_ID_ALIASES: dict[str, str] = {
    "gsm8k": "openai/gsm8k",
    "ai2_arc": "allenai/ai2_arc",
    "hellaswag": "Rowan/hellaswag",
    "mmlu": "cais/mmlu",
}


def normalize_hf_dataset_id(name: str) -> str:
    """Map bare dataset slugs to namespace/name ids required by huggingface_hub."""
    cleaned = name.strip()
    if not cleaned or "/" in cleaned:
        return cleaned
    return DATASET_ID_ALIASES.get(cleaned, cleaned)


def load_hf_split(name: str, config: str | None, split: str):
    from datasets import load_dataset

    load_env_file()
    repo_id = normalize_hf_dataset_id(name)
    token = os.environ.get("HF_TOKEN") or None
    kwargs = {"token": token} if token else {}

    if config:
        return load_dataset(repo_id, config, split=split, **kwargs)
    return load_dataset(repo_id, split=split, **kwargs)


EVAL_DATASET_SPECS: dict[str, dict] = {
    "gsm8k": {
        "name": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "question": "question",
        "answer": "answer",
        "answer_parser": "gsm8k",
    },
    "mmlu": {
        "name": "cais/mmlu",
        "config": "all",
        "split": "test",
        "question": "question",
        "choices": "choices",
        "answer": "answer",
        "answer_parser": "mmlu",
    },
    "hellaswag": {
        "name": "Rowan/hellaswag",
        "config": None,
        "split": "validation",
        "question": "ctx",
        "choices": "endings",
        "answer": "label",
        "answer_parser": "index",
    },
    "arc_challenge": {
        "name": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
        "question": "question",
        "choices": "choices",
        "answer": "answerKey",
        "answer_parser": "arc",
    },
}
