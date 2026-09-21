from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class AdapterConfig:
    vocab_size: int
    rank: int
    alpha: float
    seed: int = 42


class LogitAdapter:
    """Low-rank residual on NPU logits. HTP stays frozen; only A/B train on CPU."""

    def __init__(self, config: AdapterConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        scale = 1.0 / max(config.rank, 1) ** 0.5
        self.A = rng.normal(0.0, scale, (config.vocab_size, config.rank)).astype(np.float32)
        self.B = np.zeros((config.rank, config.vocab_size), dtype=np.float32)
        self._m_A = np.zeros_like(self.A)
        self._v_A = np.zeros_like(self.A)
        self._m_B = np.zeros_like(self.B)
        self._v_B = np.zeros_like(self.B)
        self._step = 0

    @property
    def scale(self) -> float:
        return float(self.config.alpha) / max(self.config.rank, 1)

    def delta(self, input_ids: np.ndarray) -> np.ndarray:
        tokens = np.clip(input_ids.astype(np.int64), 0, self.config.vocab_size - 1)
        embedded = self.A[tokens]
        return embedded @ self.B * self.scale

    def apply(self, logits: np.ndarray, input_ids: np.ndarray) -> np.ndarray:
        return logits.astype(np.float32, copy=False) + self.delta(input_ids)

    def adam_step(
        self,
        grad_A: np.ndarray,
        grad_B: np.ndarray,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self._step += 1
        for param, grad, moment, velocity in (
            (self.A, grad_A, self._m_A, self._v_A),
            (self.B, grad_B, self._m_B, self._v_B),
        ):
            moment *= beta1
            moment += (1.0 - beta1) * grad
            velocity *= beta2
            velocity += (1.0 - beta2) * (grad * grad)
            m_hat = moment / (1.0 - beta1**self._step)
            v_hat = velocity / (1.0 - beta2**self._step)
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "adapter.npz", A=self.A, B=self.B)
        (directory / "npu_adapter.json").write_text(
            json.dumps({"schema_version": 1, **asdict(self.config)}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "LogitAdapter":
        payload = json.loads((directory / "npu_adapter.json").read_text(encoding="utf-8"))
        adapter = cls(
            AdapterConfig(
                vocab_size=int(payload["vocab_size"]),
                rank=int(payload["rank"]),
                alpha=float(payload["alpha"]),
                seed=int(payload.get("seed", 42)),
            )
        )
        weights = np.load(directory / "adapter.npz")
        adapter.A = weights["A"].astype(np.float32)
        adapter.B = weights["B"].astype(np.float32)
        return adapter


def softmax_ce_grads(
    logits: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return mean CE and d(loss)/d(logits) for a [T, V] teacher-forced step."""
    active = mask.astype(np.float32)
    count = float(active.sum())
    if count <= 0:
        return 0.0, np.zeros_like(logits, dtype=np.float32)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    rows = np.arange(labels.shape[0])
    safe = np.clip(labels, 0, logits.shape[-1] - 1)
    gathered = np.clip(probs[rows, safe], 1e-9, 1.0)
    loss = float((-(np.log(gathered)) * active).sum() / count)
    grads = probs
    grads[rows, safe] -= 1.0
    grads *= (active / count)[:, None]
    return loss, grads.astype(np.float32)


def adapter_grads(
    adapter: LogitAdapter,
    input_ids: np.ndarray,
    logit_grads: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.clip(input_ids.astype(np.int64), 0, adapter.config.vocab_size - 1)
    embedded = adapter.A[tokens]
    grad_delta = logit_grads * adapter.scale
    grad_B = embedded.T @ grad_delta
    grad_embed = grad_delta @ adapter.B.T
    grad_A = np.zeros_like(adapter.A)
    np.add.at(grad_A, tokens, grad_embed)
    return grad_A.astype(np.float32), grad_B.astype(np.float32)
