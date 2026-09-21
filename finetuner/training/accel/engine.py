from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from finetuner.core.job import TrainingConfig
from finetuner.training.accel.data import (
    PreferenceExample,
    PromptExample,
    SequenceExample,
    kto_examples,
    preference_examples,
    prompt_examples,
    sft_examples,
)
from finetuner.training.accel.detect import engine_method, npu_info, resolve_accelerator, tpu_info
from finetuner.training.accel.generate import generate
from finetuner.training.accel.math import (
    bradley_terry_weights,
    dpo_weights,
    group_advantages,
    kto_weights,
    logprob_to_logit_grads,
    orpo_odds_weights,
    ppo_logprob_weight,
    sequence_logprob,
)
from finetuner.training.npu.adapter import (
    AdapterConfig,
    LogitAdapter,
    adapter_grads,
    softmax_ce_grads,
)
from finetuner.training.npu.runtime import NpuBackbone, load_spec, resolve_npu_artifact
from finetuner.training.npu.tokenizer import NpuTokenizer

LogFn = Callable[[str], None]


def train_accel(
    model_path: str,
    output_dir: str,
    training: TrainingConfig,
    dataset,
    log_callback: LogFn | None = None,
    backbone=None,
    tokenizer=None,
) -> str:
    def log(message: str) -> None:
        if log_callback:
            log_callback(message)

    accelerator = resolve_accelerator(training)
    method = engine_method(training)
    if backbone is None or tokenizer is None:
        artifact = resolve_npu_artifact(training.npu_artifact_path or model_path)
        spec = load_spec(artifact)
        log(f"Accel backbone: {artifact}")
        tokenizer = tokenizer or NpuTokenizer(artifact)
        backbone = backbone or NpuBackbone(spec, log)
        vocab_size = spec.vocab_size
    else:
        vocab_size = int(getattr(backbone, "vocab_size", 0) or 0)
        if vocab_size <= 0:
            probe = np.asarray(backbone.logits(np.array([1], dtype=np.int64)))
            vocab_size = int(probe.shape[-1])

    _log_devices(accelerator, getattr(backbone, "provider", "unknown"), log)
    adapter = LogitAdapter(
        AdapterConfig(
            vocab_size=vocab_size,
            rank=training.lora_rank,
            alpha=float(training.lora_alpha),
            seed=training.seed,
        )
    )
    examples = _load_examples(method, dataset, tokenizer, training.max_seq_length)
    if not examples:
        raise ValueError(f"Accel {method} needs usable rows for that method's data format.")

    trainable = int(adapter.A.size + adapter.B.size)
    log(
        f"Accel {method} on {accelerator}: trainable params {trainable:,} "
        f"(logit LoRA rank={training.lora_rank}); frozen decoder provider="
        f"{getattr(backbone, 'provider', 'unknown')}"
    )

    accum_A = np.zeros_like(adapter.A)
    accum_B = np.zeros_like(adapter.B)
    pending = 0
    losses: list[float] = []
    steps = 0
    cursor = 0
    reward_fn = _reward_fn(training, method)

    while steps < training.max_steps:
        example = examples[cursor % len(examples)]
        cursor += 1
        loss, pairs = _step(method, adapter, backbone, tokenizer, example, training, reward_fn)
        for ids, logit_grads in pairs:
            grad_A, grad_B = adapter_grads(adapter, ids, logit_grads)
            accum_A += grad_A
            accum_B += grad_B
        pending += 1
        losses.append(loss)
        if pending < training.gradient_accumulation_steps and steps + 1 < training.max_steps:
            continue
        adapter.adam_step(accum_A / pending, accum_B / pending, training.learning_rate)
        accum_A.fill(0)
        accum_B.fill(0)
        pending = 0
        steps += 1
        if steps == 1 or steps % 5 == 0 or steps == training.max_steps:
            recent = sum(losses[-5:]) / min(5, len(losses))
            log(
                f"Accel-{method} step {steps}/{training.max_steps} "
                f"loss={recent:.4f} provider={getattr(backbone, 'provider', 'unknown')}"
            )

    out = Path(output_dir) / "accel_adapter"
    adapter.save(out)
    (out / "README.txt").write_text(
        "Frozen NPU/TPU/CPU decoder + CPU (or TPU-host) logit LoRA.\n"
        "Hexagon HTP and Coral/Cloud TPU do not receive gradients. The accelerator "
        "only produces base logits; A/B train outside the frozen graph.\n",
        encoding="utf-8",
    )
    last = losses[-1] if losses else 0.0
    (out / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": steps,
                "log_history": [{"loss": last, "step": steps}],
                "accelerator": accelerator,
                "method": method,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Keep the older NPU filename so existing tests and scripts still find a package.
    adapter.save(Path(output_dir) / "npu_adapter")
    log(f"Accel adapter saved to {out}")
    return str(out)


def _log_devices(accelerator: str, provider: str, log: LogFn) -> None:
    npu = npu_info()
    tpu = tpu_info()
    log(
        f"Accel device request={accelerator} decoder={provider} "
        f"npu={'yes' if npu.available else 'no'}"
        + (f" ({npu.name})" if npu.name else "")
        + f" tpu={'yes' if tpu.available else 'no'}"
        + (f" ({tpu.name})" if tpu.name else "")
    )
    if accelerator == "tpu" and not tpu.available:
        log("TPU chips not visible; adapter trains on the host. Frozen decoder stays ONNX.")
    if accelerator == "npu" and "QNN" not in provider:
        log("NPU EP not bound; using the ONNX fallback provider for frozen logits.")


def _load_examples(method: str, dataset, tokenizer, max_length: int):
    if method == "sft":
        return sft_examples(dataset, tokenizer, max_length)
    if method in {"dpo", "orpo", "reward"}:
        return preference_examples(dataset, tokenizer, max_length)
    if method == "kto":
        return kto_examples(dataset, tokenizer, max_length)
    if method in {"grpo", "ppo", "rloo"}:
        return prompt_examples(dataset, tokenizer, max_length)
    raise ValueError(f"Accel engine does not implement {method}")


def _reward_fn(training: TrainingConfig, method: str):
    if method not in {"grpo", "ppo", "rloo"}:
        return None
    from finetuner.training.rewards import build_reward_function

    reward_id = training.reward_function or "exact_match"
    if reward_id == "hf_reward_model":
        reward_id = "exact_match"
    built = build_reward_function(reward_id, training.reward_model_id)
    if not callable(built):
        built = build_reward_function("exact_match")
    return built


def _step(
    method: str,
    adapter: LogitAdapter,
    backbone,
    tokenizer,
    example,
    training: TrainingConfig,
    reward_fn,
) -> tuple[float, list[tuple[np.ndarray, np.ndarray]]]:
    if method == "sft":
        return _sft_step(adapter, backbone, example)
    if method == "dpo":
        return _dpo_step(adapter, backbone, example, training.dpo_beta)
    if method == "kto":
        return _kto_step(adapter, backbone, example, training.dpo_beta)
    if method == "orpo":
        return _orpo_step(adapter, backbone, example)
    if method == "reward":
        return _reward_step(adapter, backbone, example)
    if method == "grpo":
        return _pg_step(adapter, backbone, tokenizer, example, training, reward_fn, leave_one_out=False)
    if method == "rloo":
        return _pg_step(adapter, backbone, tokenizer, example, training, reward_fn, leave_one_out=True)
    if method == "ppo":
        return _ppo_step(adapter, backbone, tokenizer, example, training, reward_fn)
    raise ValueError(f"Accel engine does not implement {method}")


def _forward(adapter: LogitAdapter, backbone, example: SequenceExample) -> np.ndarray:
    base = np.asarray(backbone.logits(example.ids), dtype=np.float32)
    return adapter.apply(base, example.ids), base


def _sft_step(adapter, backbone, example: SequenceExample):
    combined, _ = _forward(adapter, backbone, example)
    loss, grads = softmax_ce_grads(combined, example.labels, example.mask)
    return loss, [(example.ids, grads)]


def _dpo_step(adapter, backbone, example: PreferenceExample, beta: float):
    pi_w, ref_w_logits = _forward(adapter, backbone, example.chosen)
    pi_l, ref_l_logits = _forward(adapter, backbone, example.rejected)
    pi_chosen = sequence_logprob(pi_w, example.chosen.labels, example.chosen.mask)
    pi_rejected = sequence_logprob(pi_l, example.rejected.labels, example.rejected.mask)
    ref_chosen = sequence_logprob(ref_w_logits, example.chosen.labels, example.chosen.mask)
    ref_rejected = sequence_logprob(ref_l_logits, example.rejected.labels, example.rejected.mask)
    loss, w_chosen, w_rejected = dpo_weights(
        pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta
    )
    return loss, [
        (example.chosen.ids, logprob_to_logit_grads(pi_w, example.chosen.labels, example.chosen.mask, w_chosen)),
        (
            example.rejected.ids,
            logprob_to_logit_grads(pi_l, example.rejected.labels, example.rejected.mask, w_rejected),
        ),
    ]


def _kto_step(adapter, backbone, example, beta: float):
    combined, ref = _forward(adapter, backbone, example.sequence)
    pi = sequence_logprob(combined, example.sequence.labels, example.sequence.mask)
    ref_lp = sequence_logprob(ref, example.sequence.labels, example.sequence.mask)
    loss, weight = kto_weights(pi, ref_lp, example.desirable, beta)
    grads = logprob_to_logit_grads(
        combined, example.sequence.labels, example.sequence.mask, weight
    )
    return loss, [(example.sequence.ids, grads)]


def _orpo_step(adapter, backbone, example: PreferenceExample):
    chosen, _ = _forward(adapter, backbone, example.chosen)
    rejected, _ = _forward(adapter, backbone, example.rejected)
    sft_loss, sft_grads = softmax_ce_grads(chosen, example.chosen.labels, example.chosen.mask)
    pi_w = sequence_logprob(chosen, example.chosen.labels, example.chosen.mask)
    pi_l = sequence_logprob(rejected, example.rejected.labels, example.rejected.mask)
    odds_loss, w_c, w_r = orpo_odds_weights(pi_w, pi_l)
    chosen_grads = sft_grads + logprob_to_logit_grads(
        chosen, example.chosen.labels, example.chosen.mask, w_c
    )
    rejected_grads = logprob_to_logit_grads(
        rejected, example.rejected.labels, example.rejected.mask, w_r
    )
    return sft_loss + odds_loss, [
        (example.chosen.ids, chosen_grads),
        (example.rejected.ids, rejected_grads),
    ]


def _reward_step(adapter, backbone, example: PreferenceExample):
    chosen, _ = _forward(adapter, backbone, example.chosen)
    rejected, _ = _forward(adapter, backbone, example.rejected)
    score_c = sequence_logprob(chosen, example.chosen.labels, example.chosen.mask)
    score_r = sequence_logprob(rejected, example.rejected.labels, example.rejected.mask)
    loss, w_c, w_r = bradley_terry_weights(score_c, score_r)
    return loss, [
        (example.chosen.ids, logprob_to_logit_grads(chosen, example.chosen.labels, example.chosen.mask, w_c)),
        (example.rejected.ids, logprob_to_logit_grads(rejected, example.rejected.labels, example.rejected.mask, w_r)),
    ]


def _pg_step(
    adapter,
    backbone,
    tokenizer,
    example: PromptExample,
    training: TrainingConfig,
    reward_fn,
    *,
    leave_one_out: bool,
):
    max_new = max(8, min(16, training.max_seq_length // 4))
    generations = max(2, training.grpo_num_generations)
    completions = []
    decoded = []
    for index in range(generations):
        ids, _ = generate(
            backbone,
            adapter,
            example.prompt_ids,
            max_new_tokens=max_new,
            temperature=0.8,
            seed=training.seed + index,
        )
        prompt_len = int(example.prompt_ids.shape[0])
        text = tokenizer.decode(ids[prompt_len:].tolist())
        completions.append(ids)
        decoded.append(text)
    rewards = np.array(
        reward_fn(decoded, ground_truth=[example.ground_truth] * len(decoded)),
        dtype=np.float32,
    )
    advantages = group_advantages(rewards, leave_one_out=leave_one_out)
    pairs = []
    losses = []
    for ids, advantage in zip(completions, advantages):
        seq = _completion_example(ids, int(example.prompt_ids.shape[0]))
        combined, _ = _forward(adapter, backbone, seq)
        logp = sequence_logprob(combined, seq.labels, seq.mask)
        losses.append(float(-advantage * logp))
        pairs.append(
            (
                seq.ids,
                logprob_to_logit_grads(combined, seq.labels, seq.mask, -float(advantage)),
            )
        )
    return float(np.mean(losses) if losses else 0.0), pairs


def _ppo_step(adapter, backbone, tokenizer, example: PromptExample, training, reward_fn):
    max_new = max(8, min(16, training.max_seq_length // 4))
    ids, old_logp = generate(
        backbone,
        adapter,
        example.prompt_ids,
        max_new_tokens=max_new,
        temperature=0.7,
        seed=training.seed,
    )
    prompt_len = int(example.prompt_ids.shape[0])
    text = tokenizer.decode(ids[prompt_len:].tolist())
    reward = float(reward_fn([text], ground_truth=[example.ground_truth])[0])
    seq = _completion_example(ids, prompt_len)
    combined, _ = _forward(adapter, backbone, seq)
    new_logp = sequence_logprob(combined, seq.labels, seq.mask)
    loss, weight = ppo_logprob_weight(
        new_logp,
        old_logp,
        reward,
        training.ppo_cliprange,
        training.ppo_kl_coef,
    )
    return loss, [(seq.ids, logprob_to_logit_grads(combined, seq.labels, seq.mask, weight))]


def _completion_example(ids: np.ndarray, prompt_len: int) -> SequenceExample:
    tokens = np.asarray(ids, dtype=np.int64)
    labels = np.array(tokens[1:].tolist() + [int(tokens[-1])], dtype=np.int64)
    mask = np.zeros(len(tokens), dtype=np.float32)
    start = min(max(prompt_len - 1, 0), len(tokens) - 1)
    mask[start:] = 1.0
    return SequenceExample(tokens, labels, mask)
