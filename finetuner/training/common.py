from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from finetuner.core.job import TrainingConfig


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        version = getattr(torch, "__version__", "unknown")
        raise RuntimeError(
            "CUDA GPU not available. Fine-tuning requires an NVIDIA GPU with PyTorch CUDA support.\n"
            f"Installed torch: {version}\n"
            "If this shows '+cpu', reinstall with:\n"
            "  pip install torch==2.11.0 torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu128"
        )


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_lora_model(model_path: str, training: TrainingConfig, log: Callable[[str], None]):
    import torch

    if training.use_qlora:
        log("Loading model with QLoRA (4-bit)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"QLoRA load failed: {exc}\n"
                "Disable QLoRA in Training settings to use fp16 LoRA instead."
            ) from exc
        model = prepare_model_for_kbit_training(model)
        log("QLoRA load successful.")
    else:
        log("Loading model with fp16 LoRA...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    lora_config = LoraConfig(
        r=training.lora_rank,
        lora_alpha=training.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def base_training_kwargs(training: TrainingConfig, output_dir: Path) -> dict:
    return {
        "output_dir": str(output_dir),
        "max_steps": training.max_steps,
        "learning_rate": training.learning_rate,
        "per_device_train_batch_size": training.batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "logging_steps": 5,
        "save_steps": training.max_steps,
        "save_total_limit": 1,
        "fp16": True,
        "report_to": "none",
    }


def save_and_merge(trainer, tokenizer, output_dir: Path, log: Callable[[str], None]) -> str:
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    merged_dir = output_dir / "merged"
    log("Merging LoRA weights for eval...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    log(f"Merged model saved to {merged_dir}")
    return str(merged_dir)


def load_raw_dataset(
    dataset_path: str,
    training: TrainingConfig | None = None,
    log: Callable[[str], None] | None = None,
) -> Dataset:
    if dataset_path.startswith("preset://"):
        from finetuner.datasets.loader import load_preset_dataset

        preset_id = dataset_path.removeprefix("preset://")
        use_bundled = training.dataset_use_bundled_only if training else False
        return load_preset_dataset(preset_id, log=log, bundled_only=use_bundled)

    path = Path(dataset_path)
    if path.exists():
        if path.suffix == ".jsonl":
            rows = []
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return _format_sft_dataset(rows)
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return _format_sft_dataset(data)
            raise ValueError("JSON dataset must be a list of records")

    from finetuner.datasets.hf_datasets import load_hf_split, normalize_hf_dataset_id

    repo_id = normalize_hf_dataset_id(dataset_path)
    return _format_sft_dataset(load_hf_split(repo_id, None, "train"))


def _format_sft_dataset(rows) -> Dataset:
    formatted = []
    for row in rows:
        if isinstance(row, dict):
            if "text" in row:
                formatted.append({"text": row["text"]})
            elif "messages" in row:
                formatted.append({"text": _messages_to_text(row["messages"])})
            elif "instruction" in row and "output" in row:
                inst = row.get("instruction", "")
                inp = row.get("input", "")
                out = row.get("output", "")
                prompt = f"### Instruction:\n{inst}\n"
                if inp:
                    prompt += f"### Input:\n{inp}\n"
                prompt += f"### Response:\n{out}"
                formatted.append({"text": prompt})
            elif "prompt" in row and "chosen" in row:
                formatted.append(row)
            else:
                formatted.append({"text": json.dumps(row)})
        else:
            formatted.append({"text": str(row)})
    return Dataset.from_list(formatted)


def _messages_to_text(messages: list) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def detect_text_field(dataset: Dataset) -> str:
    if "text" in dataset.column_names:
        return "text"
    if "messages" in dataset.column_names:
        return "messages"
    return dataset.column_names[0]
