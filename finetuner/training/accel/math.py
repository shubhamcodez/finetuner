from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0))))


def sequence_logprob(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    logp = log_softmax(logits)
    rows = np.arange(labels.shape[0])
    safe = np.clip(labels, 0, logits.shape[-1] - 1)
    return float((logp[rows, safe] * mask.astype(np.float32)).sum())


def logprob_to_logit_grads(
    logits: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    weight: float,
) -> np.ndarray:
    """d(weight * sum log π) / d logits."""
    probs = softmax(logits)
    grads = -probs
    rows = np.arange(labels.shape[0])
    safe = np.clip(labels, 0, logits.shape[-1] - 1)
    grads[rows, safe] += 1.0
    grads *= (mask.astype(np.float32) * float(weight))[:, None]
    return grads.astype(np.float32)


def dpo_weights(
    pi_chosen: float,
    pi_rejected: float,
    ref_chosen: float,
    ref_rejected: float,
    beta: float,
) -> tuple[float, float, float]:
    delta = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)
    logit = float(beta) * delta
    prob = sigmoid(logit)
    loss = float(-np.log(max(prob, 1e-9)))
    d_logit = -(1.0 - prob)
    return loss, d_logit * float(beta), -d_logit * float(beta)


def kto_weights(pi: float, ref: float, desirable: bool, beta: float) -> tuple[float, float]:
    signed = (pi - ref) if desirable else (ref - pi)
    logit = float(beta) * signed
    prob = sigmoid(logit)
    loss = float(-np.log(max(prob, 1e-9)))
    d_logit = -(1.0 - prob)
    weight = d_logit * float(beta)
    return loss, weight if desirable else -weight


def orpo_odds_weights(pi_chosen: float, pi_rejected: float) -> tuple[float, float, float]:
    odds = pi_chosen - pi_rejected
    prob = sigmoid(odds)
    loss = float(-np.log(max(prob, 1e-9)))
    d_odds = -(1.0 - prob)
    return loss, d_odds, -d_odds


def bradley_terry_weights(score_chosen: float, score_rejected: float) -> tuple[float, float, float]:
    return orpo_odds_weights(score_chosen, score_rejected)


def ppo_logprob_weight(
    new_logp: float,
    old_logp: float,
    advantage: float,
    cliprange: float,
    kl_coef: float,
) -> tuple[float, float]:
    ratio = float(np.exp(np.clip(new_logp - old_logp, -20.0, 20.0)))
    unclipped = ratio * advantage
    clipped_ratio = float(np.clip(ratio, 1.0 - cliprange, 1.0 + cliprange))
    clipped = clipped_ratio * advantage
    if unclipped <= clipped:
        loss = -unclipped
        d_ratio = -advantage
    elif 1.0 - cliprange <= ratio <= 1.0 + cliprange:
        loss = -clipped
        d_ratio = -advantage
    else:
        loss = -clipped
        d_ratio = 0.0
    loss += kl_coef * (new_logp - old_logp)
    return float(loss), float(d_ratio * ratio + kl_coef)


def group_advantages(rewards: np.ndarray, *, leave_one_out: bool) -> np.ndarray:
    values = np.asarray(rewards, dtype=np.float32)
    if values.size == 0:
        return values
    if leave_one_out and values.size > 1:
        total = float(values.sum())
        others = (total - values) / (values.size - 1)
        return values - others
    centered = values - float(values.mean())
    scale = float(values.std())
    if scale > 1e-6:
        return centered / scale
    return centered
