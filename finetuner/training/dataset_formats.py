from __future__ import annotations

import random

from datasets import Dataset


def split_instruction_response(text: str) -> tuple[str, str]:
    if "### Response:" in text:
        prompt, response = text.split("### Response:", 1)
        return prompt.strip() + "\n### Response:\n", response.strip()
    if "### response:" in text.lower():
        idx = text.lower().index("### response:")
        prompt = text[:idx].strip()
        response = text[idx + len("### response:") :].strip()
        return prompt + "\n### Response:\n", response
    return text.strip() + "\n### Response:\n", ""


def _weak_rejected(answer: str) -> str:
    if not answer:
        return "I don't know."
    if "####" in answer:
        tail = answer.split("####")[-1].strip()
        if tail.isdigit():
            wrong = str(int(tail) + random.randint(1, 9))
            return answer.rsplit("####", 1)[0] + f"#### {wrong}"
    tokens = answer.split()
    if tokens:
        tokens[-1] = "unknown"
        return " ".join(tokens)
    return "Unknown."


def sft_to_prompt_rows(dataset: Dataset) -> list[dict]:
    rows = []
    for ex in dataset:
        text = ex.get("text", "")
        prompt, response = split_instruction_response(text)
        if not prompt.strip():
            continue
        rows.append({"prompt": prompt, "ground_truth": response})
    return rows


def sft_to_dpo_rows(dataset: Dataset) -> list[dict]:
    rows = []
    for ex in dataset:
        text = ex.get("text", "")
        prompt, chosen = split_instruction_response(text)
        if not prompt.strip() or not chosen:
            continue
        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": _weak_rejected(chosen),
            }
        )
    return rows


def sft_to_kto_rows(dataset: Dataset) -> list[dict]:
    rows = []
    for ex in dataset:
        text = ex.get("text", "")
        prompt, completion = split_instruction_response(text)
        if not prompt.strip() or not completion:
            continue
        rows.append({"prompt": prompt, "completion": completion, "label": True})
        rows.append(
            {
                "prompt": prompt,
                "completion": _weak_rejected(completion),
                "label": False,
            }
        )
    return rows


def sft_to_reward_rows(dataset: Dataset) -> list[dict]:
    return sft_to_dpo_rows(dataset)


def prepare_method_dataset(dataset: Dataset, method: str) -> Dataset:
    if method == "sft":
        return dataset
    if method in ("grpo", "ppo"):
        rows = sft_to_prompt_rows(dataset)
        if not rows:
            raise ValueError("Could not extract prompts from the dataset for RL training.")
        return Dataset.from_list(rows)
    if method in ("dpo", "reward"):
        rows = sft_to_dpo_rows(dataset)
        if not rows:
            raise ValueError("Could not build preference pairs from the dataset.")
        return Dataset.from_list(rows)
    if method == "kto":
        rows = sft_to_kto_rows(dataset)
        if not rows:
            raise ValueError("Could not build KTO examples from the dataset.")
        return Dataset.from_list(rows)
    raise ValueError(f"Unsupported training method: {method}")
