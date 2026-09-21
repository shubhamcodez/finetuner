from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SequenceExample:
    ids: np.ndarray
    labels: np.ndarray
    mask: np.ndarray


@dataclass
class PreferenceExample:
    chosen: SequenceExample
    rejected: SequenceExample


@dataclass
class KtoExample:
    sequence: SequenceExample
    desirable: bool


@dataclass
class PromptExample:
    prompt_ids: np.ndarray
    ground_truth: str | None


def split_response(text: str) -> tuple[str, str]:
    if "### Response:" in text:
        prompt, response = text.split("### Response:", 1)
        return prompt.strip() + "\n### Response:\n", response.strip()
    lowered = text.lower()
    if "### response:" in lowered:
        idx = lowered.index("### response:")
        return text[:idx].strip() + "\n### Response:\n", text[idx + len("### response:") :].strip()
    return text.strip() + "\n### Response:\n", ""


def encode_supervised(prompt: str, response: str, tokenizer, max_length: int) -> SequenceExample | None:
    if not response:
        return None
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(prompt + response)[:max_length]
    if len(full_ids) < 2:
        return None
    labels = np.array(full_ids[1:] + [full_ids[-1]], dtype=np.int64)
    mask = np.zeros(len(full_ids), dtype=np.float32)
    start = min(max(len(prompt_ids) - 1, 0), len(full_ids) - 1)
    mask[start:] = 1.0
    return SequenceExample(np.array(full_ids, dtype=np.int64), labels, mask)


def dataset_rows(dataset) -> list[dict]:
    if hasattr(dataset, "to_list"):
        return list(dataset.to_list())
    return [dict(item) for item in dataset]


def sft_examples(dataset, tokenizer, max_length: int) -> list[SequenceExample]:
    rows: list[SequenceExample] = []
    for item in dataset_rows(dataset):
        text = str(item.get("text") or "")
        if text:
            prompt, response = split_response(text)
        else:
            prompt = str(item.get("prompt") or "")
            response = str(item.get("response") or item.get("chosen") or "")
        example = encode_supervised(prompt, response, tokenizer, max_length)
        if example is not None:
            rows.append(example)
    return rows


def preference_examples(dataset, tokenizer, max_length: int) -> list[PreferenceExample]:
    rows: list[PreferenceExample] = []
    for item in dataset_rows(dataset):
        prompt = str(item.get("prompt") or "")
        chosen = encode_supervised(prompt, str(item.get("chosen") or ""), tokenizer, max_length)
        rejected = encode_supervised(prompt, str(item.get("rejected") or ""), tokenizer, max_length)
        if chosen is not None and rejected is not None:
            rows.append(PreferenceExample(chosen, rejected))
    return rows


def kto_examples(dataset, tokenizer, max_length: int) -> list[KtoExample]:
    rows: list[KtoExample] = []
    for item in dataset_rows(dataset):
        prompt = str(item.get("prompt") or "")
        completion = str(item.get("completion") or item.get("chosen") or "")
        sequence = encode_supervised(prompt, completion, tokenizer, max_length)
        if sequence is None:
            continue
        label = item.get("label", True)
        rows.append(KtoExample(sequence, bool(label)))
    return rows


def prompt_examples(dataset, tokenizer, max_length: int) -> list[PromptExample]:
    rows: list[PromptExample] = []
    for item in dataset_rows(dataset):
        prompt = str(item.get("prompt") or "")
        if not prompt:
            text = str(item.get("text") or "")
            prompt, _ = split_response(text)
        ids = np.array(tokenizer.encode(prompt)[: max(1, max_length // 2)], dtype=np.int64)
        if ids.size == 0:
            continue
        truth = item.get("ground_truth")
        if truth is None:
            _, response = split_response(str(item.get("text") or ""))
            truth = response or None
        rows.append(PromptExample(ids, None if truth is None else str(truth)))
    return rows
