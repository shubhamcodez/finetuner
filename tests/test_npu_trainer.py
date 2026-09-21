from __future__ import annotations

import json

import numpy as np

from finetuner.core.job import TrainingConfig
from finetuner.training.npu.adapter import (
    AdapterConfig,
    LogitAdapter,
    adapter_grads,
    softmax_ce_grads,
)
from finetuner.training.npu.runtime import load_spec
from finetuner.training.validation import validate_training_config


def test_logit_adapter_reduces_cross_entropy():
    adapter = LogitAdapter(AdapterConfig(vocab_size=8, rank=2, alpha=4.0, seed=0))
    ids = np.array([1, 2, 3], dtype=np.int64)
    labels = np.array([2, 3, 4], dtype=np.int64)
    mask = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    base = np.zeros((3, 8), dtype=np.float32)
    start, _ = softmax_ce_grads(adapter.apply(base, ids), labels, mask)
    for _ in range(40):
        logits = adapter.apply(base, ids)
        _, grads = softmax_ce_grads(logits, labels, mask)
        grad_A, grad_B = adapter_grads(adapter, ids, grads)
        adapter.adam_step(grad_A, grad_B, learning_rate=0.05)
    end, _ = softmax_ce_grads(adapter.apply(base, ids), labels, mask)
    assert end < start


def test_adapter_roundtrip(tmp_path):
    adapter = LogitAdapter(AdapterConfig(vocab_size=6, rank=2, alpha=2.0, seed=3))
    adapter.B[0, 1] = 0.5
    adapter.save(tmp_path)
    restored = LogitAdapter.load(tmp_path)
    ids = np.array([0, 1], dtype=np.int64)
    assert np.allclose(adapter.delta(ids), restored.delta(ids))


def test_load_spec_reads_genai_config(tmp_path):
    (tmp_path / "genai_config.json").write_text(
        json.dumps(
            {
                "model": {
                    "vocab_size": 32,
                    "decoder": {
                        "filename": "model.onnx",
                        "hidden_size": 16,
                        "num_hidden_layers": 2,
                        "num_key_value_heads": 1,
                        "head_size": 4,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model.onnx").write_bytes(b"onnx")
    spec = load_spec(tmp_path)
    assert spec.vocab_size == 32
    assert spec.num_layers == 2
    assert spec.head_dim == 4


def test_npu_method_rejects_qlora():
    config = TrainingConfig(training_method="npu", use_qlora=True)
    assert any("QLoRA" in error for error in validate_training_config(config, "npu"))
    config.use_qlora = False
    assert validate_training_config(config, "npu") == []


def test_accel_engine_rejects_qlora():
    config = TrainingConfig(
        training_method="dpo", accelerator="tpu", use_qlora=True, quality_recipe=False
    )
    assert any("QLoRA" in error for error in validate_training_config(config, "dpo"))
