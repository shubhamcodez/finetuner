from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from finetuner.datasets.hf_datasets import extract_gsm8k_answer
from finetuner.training.accel.data import encode_supervised, split_response
from finetuner.training.accel.generate import generate
from finetuner.training.accel.math import sequence_logprob
from finetuner.training.npu.adapter import LogitAdapter
from finetuner.training.npu.runtime import NpuBackbone, load_spec, resolve_npu_artifact
from finetuner.training.npu.tokenizer import NpuTokenizer

LogFn = Callable[[str], None]
_RUNTIME: dict[str, tuple] = {}


def load_runtime(artifact_path: str = "", adapter_dir: str = ""):
    artifact = str(resolve_npu_artifact(artifact_path))
    cached = _RUNTIME.get(artifact)
    if cached is None:
        spec = load_spec(Path(artifact))
        cached = (NpuBackbone(spec, lambda _msg: None), NpuTokenizer(Path(artifact)))
        _RUNTIME[artifact] = cached
    backbone, tokenizer = cached
    adapter = None
    if adapter_dir:
        path = Path(adapter_dir)
        if (path / "npu_adapter.json").is_file():
            adapter = LogitAdapter.load(path)
    return backbone, tokenizer, adapter


def _combined_logits(backbone, adapter, ids: np.ndarray) -> np.ndarray:
    logits = np.asarray(backbone.logits(ids), dtype=np.float32)
    if adapter is not None:
        return adapter.apply(logits, ids)
    return logits


def _gold_stats(backbone, adapter, tokenizer, prompt: str, gold: str, max_length: int) -> tuple[float, float]:
    example = encode_supervised(prompt, gold, tokenizer, max_length)
    if example is None:
        return 0.0, 0.0
    logits = _combined_logits(backbone, adapter, example.ids)
    logprob = sequence_logprob(logits, example.labels, example.mask)
    count = float(example.mask.sum())
    token_acc = 0.0
    if count:
        pred = np.argmax(logits, axis=-1)
        token_acc = float(((pred == example.labels) * example.mask).sum() / count)
        logprob = logprob / count
    return logprob, token_acc


def _choice_index(backbone, adapter, tokenizer, prompt: str, choices: list[str], max_length: int) -> int:
    scores = []
    for choice in choices:
        example = encode_supervised(prompt, choice, tokenizer, max_length)
        if example is None:
            scores.append(float("-inf"))
            continue
        logits = _combined_logits(backbone, adapter, example.ids)
        count = max(float(example.mask.sum()), 1.0)
        scores.append(sequence_logprob(logits, example.labels, example.mask) / count)
    return int(np.argmax(np.asarray(scores)))


def score_benchmark_accel(
    rows: list[dict],
    *,
    artifact_path: str = "",
    adapter_dir: str = "",
    max_length: int = 160,
    generate_gsm8k: bool = True,
    max_new_tokens: int = 24,
    log: LogFn | None = None,
) -> dict[str, float]:
    if not rows:
        return {"accuracy": 0.0, "gold_logprob": 0.0, "n": 0.0}
    backbone, tokenizer, adapter = load_runtime(artifact_path, adapter_dir)
    correct = 0
    logprobs: list[float] = []
    for row in rows:
        text = str(row.get("text") or "")
        prompt = str(row.get("prompt") or "")
        gold = str(row.get("gold") or "")
        if not prompt or not gold:
            prompt, gold = split_response(text)
        choices = [str(item) for item in (row.get("choices") or []) if str(item)]
        mean_lp, _ = _gold_stats(backbone, adapter, tokenizer, prompt, gold, max_length)
        logprobs.append(mean_lp)
        task = str(row.get("task") or "")
        if choices:
            predicted = _choice_index(backbone, adapter, tokenizer, prompt, choices, max_length)
            gold_index = int(row.get("gold_index") or 0)
            if predicted == gold_index:
                correct += 1
            continue
        if generate_gsm8k or task == "gsm8k":
            prompt_ids = np.array(tokenizer.encode(prompt)[: max(8, max_length // 2)], dtype=np.int64)

            def _stop(decoded: str) -> bool:
                answer = extract_gsm8k_answer(decoded)
                return "####" in decoded and bool(answer)

            generated_ids, _ = generate(
                backbone,
                adapter,
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                stop=_stop,
                tokenizer=tokenizer,
            )
            text_out = tokenizer.decode(generated_ids[len(prompt_ids) :].tolist())
            expected = extract_gsm8k_answer(gold or str(row.get("answer") or ""))
            predicted_ans = extract_gsm8k_answer(text_out)
            if expected and predicted_ans and expected.strip() == predicted_ans.strip():
                correct += 1
            elif expected and expected.lower() in text_out.lower():
                correct += 1
    n = float(len(rows))
    scores = {
        "accuracy": 100.0 * correct / n,
        "gold_logprob": float(np.mean(logprobs)) if logprobs else 0.0,
        "n": n,
    }
    if log:
        log(
            f"Accel benchmark: {correct}/{int(n)} acc={scores['accuracy']:.1f}% "
            f"gold_logprob={scores['gold_logprob']:.3f}"
        )
    return scores


def score_holdout_accel(
    holdout: list[dict],
    *,
    artifact_path: str = "",
    adapter_dir: str = "",
    max_new_tokens: int = 32,
    log: LogFn | None = None,
) -> float:
    return float(
        score_benchmark_accel(
            holdout,
            artifact_path=artifact_path,
            adapter_dir=adapter_dir,
            max_new_tokens=max_new_tokens,
            log=log,
        ).get("accuracy", 0.0)
    )


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
        chosen_logits = _combined_logits(backbone, adapter, chosen.ids)
        rejected_logits = _combined_logits(backbone, adapter, rejected.ids)
        if sequence_logprob(chosen_logits, chosen.labels, chosen.mask) > sequence_logprob(
            rejected_logits, rejected.labels, rejected.mask
        ):
            wins += 1
    score = 100.0 * wins / total if total else 0.0
    if log:
        log(f"Accel reward ranking: {wins}/{total} = {score:.1f}%")
    return score
