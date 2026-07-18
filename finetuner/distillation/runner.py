from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

from finetuner.core.job import TrainingConfig
from finetuner.distillation.config import DistillationConfig, DistillationTechnique
from finetuner.distillation.pipeline import (
    build_sequence_distillation_dataset,
    transformers_teacher_generator,
)


def run_distillation(
    dataset_path: str,
    output_dir: str,
    config: DistillationConfig,
    training: TrainingConfig,
    log: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    """Run sequence KD or TRL GKD and return (student model, provenance manifest)."""
    config.require_valid()
    technique = DistillationTechnique(config.technique)
    if technique == DistillationTechnique.SEQUENCE:
        from finetuner.training.common import load_raw_dataset

        raw = load_raw_dataset(dataset_path, training=training, log=log)
        generated = build_sequence_distillation_dataset(
            raw,
            Path(output_dir) / "synthetic_data",
            config,
            transformers_teacher_generator(config.teacher_model),
            log=log,
        )
        from finetuner.training.runner import train

        student = train(
            config.student_model,
            str(Path(output_dir) / "student"),
            replace(training, training_method="sft"),
            generated.dataset_path,
            log,
        )
        return student, generated.manifest_path

    if technique in {DistillationTechnique.LOGIT_KL, DistillationTechnique.ON_POLICY_GKD}:
        return _run_gkd(dataset_path, output_dir, config, training, log)
    if technique == DistillationTechnique.REVERSE_KL:
        return _run_minillm(dataset_path, output_dir, config, training, log)
    raise AssertionError(technique)


def _run_gkd(dataset_path, output_dir, config, training, log):
    """Use TRL's experimental GKD implementation with explicit compatibility checks."""
    try:
        from trl.experimental.gkd import GKDConfig, GKDTrainer
    except ImportError as exc:
        raise RuntimeError(
            "This distillation technique requires a TRL build containing trl.experimental.gkd"
        ) from exc
    from peft import LoraConfig
    from transformers import AutoTokenizer

    from finetuner.training.common import load_raw_dataset

    student_tokenizer = AutoTokenizer.from_pretrained(config.student_model, trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_model, trust_remote_code=True)
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError(
            "Logit/GKD distillation requires teacher and student to share a tokenizer vocabulary. "
            "Use sequence distillation for unrelated model families."
        )
    dataset = load_raw_dataset(dataset_path, training=training, log=log)
    if config.domain.mode != "all":
        from datasets import Dataset
        from finetuner.distillation.domains import select_domain_rows

        dataset = Dataset.from_list(
            select_domain_rows(dataset, config.domain, limit=config.max_samples)
        )
    elif len(dataset) > config.max_samples:
        dataset = dataset.select(range(config.max_samples))

    forward_kl = config.technique == DistillationTechnique.LOGIT_KL.value
    loss_beta = 0.0 if forward_kl else config.alpha
    on_policy_fraction = 0.0 if forward_kl else config.student_generated_fraction
    args = GKDConfig(
        output_dir=str(Path(output_dir) / "student"),
        max_steps=training.max_steps,
        learning_rate=training.learning_rate,
        per_device_train_batch_size=training.batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        max_length=training.max_seq_length,
        temperature=config.kd_temperature,
        lmbda=on_policy_fraction,
        beta=loss_beta,
        seq_kd=False,
        fp16=True,
        report_to="none",
        save_steps=training.max_steps,
    )
    peft = LoraConfig(
        r=training.lora_rank,
        lora_alpha=training.lora_alpha,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    trainer = GKDTrainer(
        model=config.student_model,
        teacher_model=config.teacher_model,
        args=args,
        train_dataset=dataset,
        processing_class=student_tokenizer,
        peft_config=peft,
    )
    if log:
        log(f"Starting {config.technique} distillation with TRL GKD")
    trainer.train()
    student_dir = Path(output_dir) / "student" / "adapter"
    trainer.save_model(str(student_dir))
    student_tokenizer.save_pretrained(str(student_dir))
    from finetuner.core.artifacts import atomic_write_json

    manifest_path = Path(output_dir) / "distillation_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "technique": config.technique,
            "teacher_model": config.teacher_model,
            "student_model": config.student_model,
            "domain": config.domain.to_dict(),
        },
    )
    return str(student_dir), str(manifest_path)


def _run_minillm(dataset_path, output_dir, config, training, log):
    try:
        from trl.experimental.minillm import MiniLLMConfig, MiniLLMTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Reverse-KL distillation requires a TRL build containing trl.experimental.minillm"
        ) from exc
    from peft import LoraConfig
    from transformers import AutoTokenizer

    from finetuner.training.common import load_raw_dataset

    student_tokenizer = AutoTokenizer.from_pretrained(config.student_model, trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_model, trust_remote_code=True)
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError(
            "MiniLLM reverse-KL distillation requires teacher and student to share a tokenizer "
            "vocabulary. Use sequence distillation for unrelated model families."
        )
    dataset = load_raw_dataset(dataset_path, training=training, log=log)
    if config.domain.mode != "all":
        from datasets import Dataset

        from finetuner.distillation.domains import select_domain_rows

        dataset = Dataset.from_list(
            select_domain_rows(dataset, config.domain, limit=config.max_samples)
        )
    from finetuner.training.dataset_formats import prepare_method_dataset

    dataset = prepare_method_dataset(dataset, "rloo")
    if len(dataset) > config.max_samples:
        dataset = dataset.select(range(config.max_samples))
    args = MiniLLMConfig(
        output_dir=str(Path(output_dir) / "student"),
        max_steps=training.max_steps,
        learning_rate=training.learning_rate,
        per_device_train_batch_size=training.batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        num_generations=training.grpo_num_generations,
        max_completion_length=min(config.max_new_tokens, training.max_seq_length // 2),
        temperature=max(config.temperature, 1e-5),
        kd_temperature=config.kd_temperature,
        fp16=True,
        report_to="none",
        save_steps=training.max_steps,
    )
    peft = LoraConfig(
        r=training.lora_rank,
        lora_alpha=training.lora_alpha,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    trainer = MiniLLMTrainer(
        model=config.student_model,
        teacher_model=config.teacher_model,
        args=args,
        train_dataset=dataset,
        processing_class=student_tokenizer,
        peft_config=peft,
    )
    if log:
        log("Starting reverse-KL distillation with TRL MiniLLM")
    trainer.train()
    student_dir = Path(output_dir) / "student" / "adapter"
    trainer.save_model(str(student_dir))
    student_tokenizer.save_pretrained(str(student_dir))
    from finetuner.core.artifacts import atomic_write_json

    manifest_path = Path(output_dir) / "distillation_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "technique": config.technique,
            "teacher_model": config.teacher_model,
            "student_model": config.student_model,
            "domain": config.domain.to_dict(),
        },
    )
    return str(student_dir), str(manifest_path)
