from __future__ import annotations

from pathlib import Path
from typing import Callable

from finetuner.core.job import TrainingConfig
from finetuner.training.dataset_formats import prepare_method_dataset
from finetuner.training.methods import get_method
from finetuner.training.validation import validate_training_config


def train(
    model_path: str,
    output_dir: str,
    training: TrainingConfig,
    dataset_path: str,
    log_callback: Callable[[str], None] | None = None,
) -> str:
    def log(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    method = training.training_method or "sft"
    spec = get_method(method)
    if spec is None:
        raise ValueError(f"Unknown training method: {method}")
    config_errors = validate_training_config(training, method)
    if config_errors:
        raise ValueError("; ".join(config_errors))

    from finetuner.training.accel.detect import resolve_accelerator, uses_accel_engine

    if training.quality_recipe and (training.training_method or "sft") not in {"npu", "tpu"}:
        from finetuner.training.recipe import apply_quality_recipe

        training = apply_quality_recipe(training)
        log(
            "Quality recipe: chat template + assistant-only loss + LoRA on attention/MLP. "
            "This trains the real transformer, not the NPU logit adapter."
        )

    accelerator = resolve_accelerator(training)
    if uses_accel_engine(training):
        runtime = {"device": accelerator}
    else:
        from finetuner.training.common import require_cuda, train_runtime

        require_cuda()
        runtime = train_runtime()
    log(f"Training method: {spec.name} — {spec.description}")
    log(f"Device: {runtime['device']}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if uses_accel_engine(training):
        from finetuner.training.accel.engine import train_accel

        raw = _load_accel_rows(dataset_path)
        dataset = prepare_method_dataset(
            raw,
            method,
            allow_synthetic_preferences=training.allow_synthetic_preferences,
            seed=training.seed,
        )
        log(f"Dataset size: {len(dataset)} examples ({method} format)")
        log(f"Using NPU/TPU accelerator engine ({accelerator})")
        return train_accel(model_path, str(out), training, dataset, log)

    from finetuner.training.common import load_raw_dataset, load_tokenizer

    raw = load_raw_dataset(dataset_path, training=training, log=log)
    dataset = prepare_method_dataset(
        raw,
        method,
        allow_synthetic_preferences=training.allow_synthetic_preferences,
        seed=training.seed,
    )
    log(f"Dataset size: {len(dataset)} examples ({method} format)")
    log(f"Loading tokenizer from {model_path}")
    tokenizer = load_tokenizer(model_path)

    if method == "sft":
        return _train_sft(model_path, training, dataset, tokenizer, out, log)
    if method == "dpo":
        return _train_dpo(model_path, training, dataset, tokenizer, out, log)
    if method == "grpo":
        return _train_grpo(model_path, training, dataset, tokenizer, out, log)
    if method == "kto":
        return _train_kto(model_path, training, dataset, tokenizer, out, log)
    if method == "reward":
        return _train_reward(model_path, training, dataset, tokenizer, out, log)
    if method == "ppo":
        return _train_ppo(model_path, training, dataset, tokenizer, out, log)
    if method == "orpo":
        return _train_orpo(model_path, training, dataset, tokenizer, out, log)
    if method == "rloo":
        return _train_rloo(model_path, training, dataset, tokenizer, out, log)
    raise ValueError(f"Unsupported training method: {method}")


def _load_accel_rows(dataset_path: str) -> list[dict]:
    import json

    path = Path(dataset_path)
    if path.is_file() and path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        if rows:
            return rows
    if path.is_file() and path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
    from finetuner.training.common import load_raw_dataset

    return [dict(row) for row in load_raw_dataset(dataset_path)]


def train_sft(
    model_path: str,
    output_dir: str,
    training: TrainingConfig,
    dataset_path: str,
    log_callback: Callable[[str], None] | None = None,
) -> str:
    """Backward-compatible SFT entry point."""
    from dataclasses import replace

    return train(
        model_path,
        output_dir,
        replace(training, training_method="sft"),
        dataset_path,
        log_callback,
    )


def _train_sft(model_path, training, dataset, tokenizer, out, log):
    from trl import SFTConfig, SFTTrainer

    from finetuner.training.chat_format import ensure_messages
    from finetuner.training.common import (
        base_training_kwargs,
        detect_text_field,
        load_lora_model,
        save_and_merge,
    )

    model = load_lora_model(model_path, training, log)
    use_chat = bool(training.use_chat_template or training.quality_recipe)
    if use_chat:
        rows = ensure_messages(dataset)
        from datasets import Dataset

        dataset = Dataset.from_list(rows)
        if "text" in dataset.column_names:
            dataset = dataset.remove_columns(["text"])
        log("SFT using conversational messages + assistant-only loss")
        sft_config = SFTConfig(
            **base_training_kwargs(training, out),
            max_length=training.max_seq_length,
            assistant_only_loss=True,
            packing=False,
        )
    else:
        text_field = detect_text_field(dataset)
        log(f"Using text field: {text_field}")
        sft_config = SFTConfig(
            **base_training_kwargs(training, out),
            max_length=training.max_seq_length,
            dataset_text_field=text_field,
            packing=False,
        )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    log("Starting SFT training...")
    trainer.train()
    log("Training complete. Saving adapter...")
    return save_and_merge(trainer, tokenizer, out, log)


def _train_dpo(model_path, training, dataset, tokenizer, out, log):
    from trl import DPOConfig, DPOTrainer

    from finetuner.training.common import base_training_kwargs, load_lora_model, save_and_merge

    model = load_lora_model(model_path, training, log)
    dpo_config = DPOConfig(
        **base_training_kwargs(training, out),
        beta=training.dpo_beta,
        max_length=training.max_seq_length,
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    log(f"Starting DPO training (beta={training.dpo_beta})...")
    trainer.train()
    log("DPO training complete. Saving adapter...")
    return save_and_merge(trainer, tokenizer, out, log)


def _train_grpo(model_path, training, dataset, tokenizer, out, log):
    from trl import GRPOConfig, GRPOTrainer

    from finetuner.training.common import base_training_kwargs, load_lora_model, save_and_merge
    from finetuner.training.rewards import build_reward_function

    model = load_lora_model(model_path, training, log)
    num_gen = training.grpo_num_generations
    effective_batch = training.batch_size * training.gradient_accumulation_steps
    if effective_batch % num_gen != 0:
        raise ValueError(
            f"GRPO num_generations ({num_gen}) must divide effective batch size "
            f"({effective_batch} = batch {training.batch_size} × grad accum "
            f"{training.gradient_accumulation_steps})."
        )

    reward_func = build_reward_function(
        training.reward_function,
        training.reward_model_id,
        log=log,
    )
    grpo_config = GRPOConfig(
        **base_training_kwargs(training, out),
        num_generations=num_gen,
        max_completion_length=min(256, training.max_seq_length // 2),
        remove_unused_columns=False,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    log(f"Starting GRPO training (generations={num_gen}, reward={training.reward_function})...")
    trainer.train()
    log("GRPO training complete. Saving adapter...")
    return save_and_merge(trainer, tokenizer, out, log)


def _train_kto(model_path, training, dataset, tokenizer, out, log):
    from trl import KTOConfig, KTOTrainer

    from finetuner.training.common import base_training_kwargs, load_lora_model, save_and_merge

    model = load_lora_model(model_path, training, log)
    kto_config = KTOConfig(
        **base_training_kwargs(training, out),
        beta=training.dpo_beta,
        max_length=training.max_seq_length,
    )
    trainer = KTOTrainer(
        model=model,
        args=kto_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    log(f"Starting KTO training (beta={training.dpo_beta})...")
    trainer.train()
    log("KTO training complete. Saving adapter...")
    return save_and_merge(trainer, tokenizer, out, log)


def _train_reward(model_path, training, dataset, tokenizer, out, log):
    from transformers import AutoModelForSequenceClassification
    from trl import RewardConfig, RewardTrainer

    from finetuner.training.common import base_training_kwargs, train_runtime

    runtime = train_runtime()
    log("Loading base model for reward-model training...")
    kwargs = {
        "num_labels": 1,
        "torch_dtype": runtime["dtype"],
        "trust_remote_code": True,
    }
    if runtime["device_map"]:
        kwargs["device_map"] = runtime["device_map"]
    base = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
    if runtime["device"] == "cpu":
        base = base.to("cpu")
    reward_config = RewardConfig(
        **base_training_kwargs(training, out),
        max_length=training.max_seq_length,
    )
    trainer = RewardTrainer(
        model=base,
        args=reward_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    log("Starting reward-model training...")
    trainer.train()
    reward_dir = out / "reward_model"
    reward_dir.mkdir(exist_ok=True)
    trainer.model.save_pretrained(str(reward_dir))
    tokenizer.save_pretrained(str(reward_dir))
    log(f"Reward model saved to {reward_dir}")
    return str(reward_dir)


def _train_ppo(model_path, training, dataset, tokenizer, out, log):
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
    from trl.experimental.ppo import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

    from finetuner.training.common import train_runtime

    if not training.reward_model_id:
        raise ValueError(
            "PPO requires a Reward Model ID (sequence-classification checkpoint on Hugging Face)."
        )

    runtime = train_runtime()
    log("Loading policy and value models for PPO...")
    policy_kwargs = {"torch_dtype": runtime["dtype"], "trust_remote_code": True}
    if runtime["device_map"]:
        policy_kwargs["device_map"] = runtime["device_map"]
    policy = AutoModelForCausalLM.from_pretrained(model_path, **policy_kwargs)
    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path,
        torch_dtype=runtime["dtype"],
        trust_remote_code=True,
    )
    if runtime["device"] == "cpu":
        policy = policy.to("cpu")
        value_model = value_model.to("cpu")

    log(f"Loading reward model: {training.reward_model_id}")
    reward_kwargs = {"torch_dtype": runtime["dtype"], "trust_remote_code": True}
    if runtime["device_map"]:
        reward_kwargs["device_map"] = runtime["device_map"]
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training.reward_model_id,
        **reward_kwargs,
    )
    if runtime["device"] == "cpu":
        reward_model = reward_model.to("cpu")

    lora_config = LoraConfig(
        r=training.lora_rank,
        lora_alpha=training.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=training.lora_target_modules or "all-linear",
    )

    def _tokenize(row):
        encoded = tokenizer(
            row["prompt"],
            truncation=True,
            max_length=min(512, training.max_seq_length),
            padding=False,
        )
        return {"input_ids": encoded["input_ids"], "query": row["prompt"]}

    tokenized = dataset.map(
        _tokenize, remove_columns=[c for c in dataset.column_names if c != "prompt"]
    )

    ppo_config = PPOConfig(
        output_dir=str(out),
        learning_rate=training.learning_rate,
        per_device_train_batch_size=training.batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        total_episodes=training.max_steps,
        response_length=min(128, training.max_seq_length // 4),
        kl_coef=training.ppo_kl_coef,
        cliprange=training.ppo_cliprange,
        fp16=runtime["fp16"],
        report_to="none",
        logging_steps=5,
    )

    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy,
        ref_model=None,
        reward_model=reward_model,
        train_dataset=tokenized,
        value_model=value_model,
        peft_config=lora_config,
    )
    log(f"Starting PPO training (kl_coef={training.ppo_kl_coef})...")
    trainer.train()
    ppo_dir = out / "ppo_policy"
    ppo_dir.mkdir(exist_ok=True)
    trainer.policy_model.save_pretrained(str(ppo_dir))
    tokenizer.save_pretrained(str(ppo_dir))
    log(f"PPO policy saved to {ppo_dir}")
    return str(ppo_dir)


def _train_orpo(model_path, training, dataset, tokenizer, out, log):
    try:
        from trl.experimental.orpo import ORPOConfig, ORPOTrainer
    except ImportError as exc:
        raise RuntimeError("ORPO requires a TRL build containing trl.experimental.orpo") from exc

    from finetuner.training.common import base_training_kwargs, load_lora_model, save_and_merge

    model = load_lora_model(model_path, training, log)
    config = ORPOConfig(
        **base_training_kwargs(training, out),
        beta=training.dpo_beta,
        max_length=training.max_seq_length,
    )
    trainer = ORPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    log(f"Starting ORPO training (beta={training.dpo_beta})...")
    trainer.train()
    return save_and_merge(trainer, tokenizer, out, log)


def _train_rloo(model_path, training, dataset, tokenizer, out, log):
    from trl import RLOOConfig, RLOOTrainer

    from finetuner.training.common import base_training_kwargs, load_lora_model, save_and_merge
    from finetuner.training.rewards import build_reward_function

    model = load_lora_model(model_path, training, log)
    reward_func = build_reward_function(
        training.reward_function,
        training.reward_model_id,
        log=log,
    )
    config = RLOOConfig(
        **base_training_kwargs(training, out),
        num_generations=training.grpo_num_generations,
        max_completion_length=min(256, training.max_seq_length // 2),
        beta=training.ppo_kl_coef,
    )
    trainer = RLOOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    log(f"Starting RLOO training (generations={training.grpo_num_generations})...")
    trainer.train()
    return save_and_merge(trainer, tokenizer, out, log)
