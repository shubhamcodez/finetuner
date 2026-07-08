from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from finetuner.datasets.hf_datasets import extract_gsm8k_answer


@dataclass(frozen=True)
class RewardFunctionSpec:
    reward_id: str
    name: str
    description: str
    needs_ground_truth: bool = True
    needs_hf_model: bool = False


REWARD_FUNCTIONS: dict[str, RewardFunctionSpec] = {
    "exact_match": RewardFunctionSpec(
        reward_id="exact_match",
        name="Exact match",
        description="1.0 if completion matches ground truth (after GSM8K normalization)",
    ),
    "partial_match": RewardFunctionSpec(
        reward_id="partial_match",
        name="Partial match",
        description="Fraction of ground-truth tokens found in the completion",
    ),
    "format_bonus": RewardFunctionSpec(
        reward_id="format_bonus",
        name="Format + match",
        description="Exact match plus bonus for structured ### Response output",
    ),
    "length_penalty": RewardFunctionSpec(
        reward_id="length_penalty",
        name="Length penalty",
        description="Exact match minus penalty for overly long completions",
        needs_ground_truth=True,
    ),
    "hf_reward_model": RewardFunctionSpec(
        reward_id="hf_reward_model",
        name="HF reward model",
        description="Score completions with a Hugging Face sequence-classification model",
        needs_ground_truth=False,
        needs_hf_model=True,
    ),
}


def reward_function_list() -> list[RewardFunctionSpec]:
    return list(REWARD_FUNCTIONS.values())


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _exact_match_reward(completions: list[str], ground_truths: list[str | None]) -> list[float]:
    scores = []
    for completion, truth in zip(completions, ground_truths):
        if not truth:
            scores.append(0.0)
            continue
        pred = extract_gsm8k_answer(completion)
        exp = extract_gsm8k_answer(str(truth))
        if pred and exp and pred.strip() == exp.strip():
            scores.append(1.0)
        elif exp and exp.lower() in completion.lower():
            scores.append(0.5)
        else:
            scores.append(0.0)
    return scores


def _partial_match_reward(completions: list[str], ground_truths: list[str | None]) -> list[float]:
    scores = []
    for completion, truth in zip(completions, ground_truths):
        if not truth:
            scores.append(0.0)
            continue
        exp = _normalize(extract_gsm8k_answer(str(truth)))
        comp = _normalize(completion)
        if not exp:
            scores.append(0.0)
            continue
        hits = sum(1 for token in exp.split() if token in comp)
        scores.append(hits / max(len(exp.split()), 1))
    return scores


def _format_bonus_reward(completions: list[str], ground_truths: list[str | None]) -> list[float]:
    base = _exact_match_reward(completions, ground_truths)
    scores = []
    for completion, score in zip(completions, base):
        bonus = 0.1 if "### Response" in completion or "####" in completion else 0.0
        scores.append(min(1.0, score + bonus))
    return scores


def _length_penalty_reward(completions: list[str], ground_truths: list[str | None]) -> list[float]:
    base = _exact_match_reward(completions, ground_truths)
    scores = []
    for completion, score in zip(completions, base):
        over = max(0, len(completion) - 256)
        penalty = min(0.5, over / 512)
        scores.append(max(0.0, score - penalty))
    return scores


def build_reward_function(
    reward_id: str,
    reward_model_id: str = "",
    log: Callable[[str], None] | None = None,
) -> Callable | str:
    """Return a TRL-compatible reward callable or HF model id string."""
    spec = REWARD_FUNCTIONS.get(reward_id)
    if spec is None:
        raise ValueError(f"Unknown reward function: {reward_id}")

    if spec.needs_hf_model:
        if not reward_model_id:
            raise ValueError(
                "Reward function 'HF reward model' requires a Reward Model ID "
                "(e.g. OpenAssistant/reward-model-deberta-v3-large-v2)."
            )
        if log:
            log(f"Using HF reward model: {reward_model_id}")
        return reward_model_id

    def _fn(completions, **kwargs):
        truths = kwargs.get("ground_truth")
        if truths is None:
            raise ValueError(
                f"Reward function '{reward_id}' requires a ground_truth column in the dataset."
            )
        if not isinstance(truths, list):
            truths = [truths] * len(completions)
        if reward_id == "exact_match":
            return _exact_match_reward(completions, truths)
        if reward_id == "partial_match":
            return _partial_match_reward(completions, truths)
        if reward_id == "format_bonus":
            return _format_bonus_reward(completions, truths)
        if reward_id == "length_penalty":
            return _length_penalty_reward(completions, truths)
        return [0.0] * len(completions)

    return _fn
