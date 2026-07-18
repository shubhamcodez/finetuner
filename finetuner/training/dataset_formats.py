from __future__ import annotations

import hashlib

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


def _weak_rejected(answer: str, seed: int = 42) -> str:
    if not answer:
        return "I don't know."
    if "####" in answer:
        tail = answer.split("####")[-1].strip()
        if tail.isdigit():
            digest = hashlib.sha256(f"{seed}:{answer}".encode("utf-8")).digest()
            wrong = str(int(tail) + 1 + digest[0] % 9)
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


def sft_to_dpo_rows(dataset: Dataset, seed: int = 42) -> list[dict]:
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
                "rejected": _weak_rejected(chosen, seed),
            }
        )
    return rows


def sft_to_kto_rows(dataset: Dataset, seed: int = 42) -> list[dict]:
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
                "completion": _weak_rejected(completion, seed),
                "label": False,
            }
        )
    return rows


def sft_to_reward_rows(dataset: Dataset) -> list[dict]:
    return sft_to_dpo_rows(dataset)


def prepare_method_dataset(
    dataset: Dataset,
    method: str,
    *,
    allow_synthetic_preferences: bool = False,
    seed: int = 42,
) -> Dataset:
    if method == "sft":
        if "text" in dataset.column_names or "messages" in dataset.column_names:
            return dataset
        if {"prompt", "chosen"}.issubset(dataset.column_names):
            return Dataset.from_list(
                [
                    {"text": f"{row['prompt']}\n{row['chosen']}"}
                    for row in dataset
                    if row.get("prompt") and row.get("chosen")
                ]
            )
        if {"prompt", "response"}.issubset(dataset.column_names):
            return Dataset.from_list(
                [
                    {"text": f"{row['prompt']}\n{row['response']}"}
                    for row in dataset
                    if row.get("prompt") and row.get("response")
                ]
            )
        return dataset
    if method in ("grpo", "ppo", "rloo"):
        if "prompt" in dataset.column_names:
            return dataset
        rows = sft_to_prompt_rows(dataset)
        if not rows:
            raise ValueError("Could not extract prompts from the dataset for RL training.")
        return Dataset.from_list(rows)
    if method in ("dpo", "reward", "orpo"):
        required = {"prompt", "chosen", "rejected"}
        if required.issubset(dataset.column_names):
            return dataset
        if not allow_synthetic_preferences:
            raise ValueError(
                f"{method.upper()} requires real prompt/chosen/rejected preference pairs. "
                "Synthetic rejected answers are disabled by default because they can bias evaluation."
            )
        rows = sft_to_dpo_rows(dataset, seed)
        if not rows:
            raise ValueError("Could not build preference pairs from the dataset.")
        return Dataset.from_list(rows)
    if method == "kto":
        required = {"prompt", "completion", "label"}
        if required.issubset(dataset.column_names):
            return dataset
        if not allow_synthetic_preferences:
            raise ValueError(
                "KTO requires prompt/completion/label feedback rows. Synthetic feedback is disabled."
            )
        rows = sft_to_kto_rows(dataset, seed)
        if not rows:
            raise ValueError("Could not build KTO examples from the dataset.")
        return Dataset.from_list(rows)
    raise ValueError(f"Unsupported training method: {method}")
