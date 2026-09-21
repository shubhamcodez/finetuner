from __future__ import annotations

import numpy as np

from finetuner.training.accel.math import log_softmax, softmax
from finetuner.training.npu.adapter import LogitAdapter


def generate(
    backbone,
    adapter: LogitAdapter | None,
    prompt_ids: np.ndarray,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    ids = [int(token) for token in np.asarray(prompt_ids).reshape(-1).tolist()]
    logprob = 0.0
    rng = np.random.default_rng(seed)
    for _ in range(max(1, max_new_tokens)):
        tokens = np.array(ids, dtype=np.int64)
        logits = np.asarray(backbone.logits(tokens), dtype=np.float32)
        if adapter is not None:
            logits = adapter.apply(logits, tokens)
        last = logits[-1]
        if temperature and temperature > 0:
            probs = softmax(last / float(temperature))
            nxt = int(rng.choice(probs.shape[0], p=probs / max(float(probs.sum()), 1e-9)))
        else:
            nxt = int(np.argmax(last))
        logprob += float(log_softmax(last)[nxt])
        ids.append(nxt)
    return np.array(ids, dtype=np.int64), logprob
