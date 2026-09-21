from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from finetuner.training.accel.data import encode_supervised, split_response
from finetuner.training.accel.generate import generate
from finetuner.training.accel.math import sequence_logprob
from finetuner.training.npu.adapter import LogitAdapter
from finetuner.training.npu.runtime import NpuBackbone, load_spec, resolve_npu_artifact
from finetuner.training.npu.tokenizer import NpuTokenizer

LogFn = Callable[[str], None]


def load_runtime(artifact_path: str = "", adapter_dir: str = ""):
    artifact = resolve_npu_artifact(artifact_path)
    spec = load_spec(artifact)
    backbone = NpuBackbone(spec, lambda _msg: None)
    tokenizer = NpuTokenizer(artifact)
    adapter = None
    if adapter_dir:
        path = Path(adapter_dir)
        if (path / "npu_adapter.json").is_file():
            adapter = LogitAdapter.load(path)
    return backbone, tokenizer, adapter


def score_holdout_accel(
    holdout: list[dict],
    *,
    artifact_path: str = "",
    adapter_dir: str = "",
    max_new_tokens: int = 32,
    log: LogFn | None = None,
) -> float:
    if not holdout:
        return 0.0
    from finetuner.datasets.hf_datasets import extract_gsm8k_answer

    backbone, tokenizer, adapter = load_runtime(artifact_path, adapter_dir)
    correct = 0
    for row in holdout:
        prompt, gold = split_response(str(row.get("text") or ""))
        if not gold:
            continue
        ids = np.array(tokenizer.encode(prompt), dtype=np.int64)
        generated_ids, _ = generate(
            backbone, adapter, ids, max_new_tokens=max_new_tokens, temperature=0.0
        )
        text = tokenizer.decode(generated_ids[len(ids) :].tolist())
        expected = extract_gsm8k_answer(gold)
        predicted = extract_gsm8k_answer(text)
        if expected and predicted and expected.strip() == predicted.strip():
            correct += 1
        elif expected and expected.lower() in text.lower():
            correct += 1
    score = 100.0 * correct / len(holdout)
    if log:
        log(f"Accel holdout: {correct}/{len(holdout)} = {score:.1f}%")
    return score


def score_reward_ranking_accel(
    pairs: list[dict],
    *,
    artifact_path: str = "",
    adapter_dir: str = "",
    max_pairs: int = 16,
    log: LogFn | None = None,
) -> float:
    backbone, tokenizer, adapter = load_runtime(artifact_path, adapter_dir)
    wins = 0
    total = min(max_pairs, len(pairs))
    for index in range(total):
        row = pairs[index]
        prompt = str(row.get("prompt") or "")
        chosen = encode_supervised(prompt, str(row.get("chosen") or ""), tokenizer, 128)
        rejected = encode_supervised(prompt, str(row.get("rejected") or ""), tokenizer, 128)
        if chosen is None or rejected is None:
            continue
        chosen_logits = np.asarray(backbone.logits(chosen.ids), dtype=np.float32)
        rejected_logits = np.asarray(backbone.logits(rejected.ids), dtype=np.float32)
        if adapter is not None:
            chosen_logits = adapter.apply(chosen_logits, chosen.ids)
            rejected_logits = adapter.apply(rejected_logits, rejected.ids)
        if sequence_logprob(chosen_logits, chosen.labels, chosen.mask) > sequence_logprob(
            rejected_logits, rejected.labels, rejected.mask
        ):
            wins += 1
    score = 100.0 * wins / total if total else 0.0
    if log:
        log(f"Accel reward ranking: {wins}/{total} = {score:.1f}%")
    return score
