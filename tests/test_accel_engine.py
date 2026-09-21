from __future__ import annotations

import numpy as np

from finetuner.core.job import TrainingConfig
from finetuner.training.accel.detect import engine_method, resolve_accelerator, uses_accel_engine
from finetuner.training.accel.engine import train_accel
from finetuner.training.accel.math import (
    dpo_weights,
    group_advantages,
    kto_weights,
    ppo_logprob_weight,
    sequence_logprob,
)
from finetuner.training.npu.adapter import AdapterConfig, LogitAdapter, softmax_ce_grads


class FakeBackbone:
    def __init__(self, vocab: int = 16) -> None:
        self.vocab_size = vocab
        self.provider = "fake"
        rng = np.random.default_rng(1)
        self.table = rng.normal(0.0, 0.25, (vocab, vocab)).astype(np.float32)

    def logits(self, ids: np.ndarray) -> np.ndarray:
        tokens = np.clip(np.asarray(ids, dtype=np.int64).reshape(-1), 0, self.vocab_size - 1)
        return self.table[tokens]


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [1] + [1 + (ord(char) % 14) for char in (text or "x")[:10]]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(97 + (int(item) % 26)) for item in ids)


def _train(tmp_path, method: str, dataset, **kwargs) -> str:
    training = TrainingConfig(
        training_method=method,
        accelerator=kwargs.get("accelerator", "cpu"),
        max_steps=kwargs.get("max_steps", 3),
        learning_rate=0.05,
        lora_rank=2,
        lora_alpha=4,
        batch_size=1,
        gradient_accumulation_steps=1,
        max_seq_length=64,
        use_qlora=False,
        allow_synthetic_preferences=True,
        grpo_num_generations=2,
        seed=0,
    )
    return train_accel(
        "unused",
        str(tmp_path),
        training,
        dataset,
        backbone=FakeBackbone(),
        tokenizer=FakeTokenizer(),
    )


def test_resolve_accelerator_aliases_and_auto():
    assert resolve_accelerator(TrainingConfig(training_method="npu")) == "npu"
    assert resolve_accelerator(TrainingConfig(training_method="tpu")) == "tpu"
    assert resolve_accelerator(TrainingConfig(training_method="sft", accelerator="npu")) == "npu"
    assert engine_method(TrainingConfig(training_method="tpu")) == "sft"
    assert uses_accel_engine(TrainingConfig(accelerator="tpu"))
    assert not uses_accel_engine(TrainingConfig(training_method="sft", accelerator="cuda"))


def test_dpo_and_kto_math_prefer_the_better_sequence():
    loss_good, w_c, w_r = dpo_weights(0.0, -2.0, -1.0, -1.0, 0.5)
    loss_bad, _, _ = dpo_weights(-2.0, 0.0, -1.0, -1.0, 0.5)
    assert loss_good < loss_bad
    assert w_c < 0
    assert w_r > 0
    want_loss, want_w = kto_weights(1.0, 0.0, True, 0.5)
    avoid_loss, avoid_w = kto_weights(1.0, 0.0, False, 0.5)
    assert want_loss < avoid_loss
    assert want_w < 0
    assert avoid_w > 0


def test_rloo_advantages_leave_the_current_reward_out():
    values = group_advantages(np.array([1.0, 3.0, 5.0]), leave_one_out=True)
    assert np.allclose(values[0], 1.0 - 4.0)
    assert np.allclose(values[1], 3.0 - 3.0)


def test_ppo_clip_zeroes_far_ratio_when_advantage_is_positive():
    loss, weight = ppo_logprob_weight(4.0, 0.0, 1.0, cliprange=0.2, kl_coef=0.0)
    assert loss < 0
    assert weight == 0.0


def test_sft_on_cpu_engine_reduces_loss(tmp_path):
    from finetuner.training.accel.data import sft_examples

    dataset = [{"text": "### Instruction:\n2+2\n### Response:\n4"}]
    backbone = FakeBackbone()
    tokenizer = FakeTokenizer()
    example = sft_examples(dataset, tokenizer, 64)[0]
    adapter = LogitAdapter(AdapterConfig(vocab_size=16, rank=2, alpha=4.0, seed=0))
    start, _ = softmax_ce_grads(
        adapter.apply(backbone.logits(example.ids), example.ids),
        example.labels,
        example.mask,
    )
    _train(tmp_path, "sft", dataset, max_steps=20)
    restored = LogitAdapter.load(tmp_path / "accel_adapter")
    end, _ = softmax_ce_grads(
        restored.apply(backbone.logits(example.ids), example.ids),
        example.labels,
        example.mask,
    )
    assert end < start


def test_every_page_method_runs_on_the_accel_engine(tmp_path):
    sft = [{"text": "### Instruction:\n2+2\n### Response:\n4"}]
    pref = [{"prompt": "2+2", "chosen": "4", "rejected": "5"}]
    kto = [{"prompt": "2+2", "completion": "4", "label": True}]
    prompts = [{"prompt": "2+2\n### Response:\n", "ground_truth": "4"}]
    datasets = {
        "sft": sft,
        "dpo": pref,
        "orpo": pref,
        "reward": pref,
        "kto": kto,
        "grpo": prompts,
        "rloo": prompts,
        "ppo": prompts,
    }
    for method, rows in datasets.items():
        out = _train(tmp_path / method, method, rows, accelerator="tpu")
        assert (tmp_path / method / "accel_adapter" / "npu_adapter.json").is_file()
        assert out.endswith("accel_adapter")


def test_sequence_logprob_matches_masked_tokens():
    logits = np.zeros((2, 4), dtype=np.float32)
    logits[0, 1] = 5.0
    logits[1, 2] = 5.0
    labels = np.array([1, 2], dtype=np.int64)
    mask = np.array([1.0, 0.0], dtype=np.float32)
    assert sequence_logprob(logits, labels, mask) < 0
    assert sequence_logprob(logits, labels, np.ones(2, dtype=np.float32)) < sequence_logprob(
        logits, labels, mask
    )
