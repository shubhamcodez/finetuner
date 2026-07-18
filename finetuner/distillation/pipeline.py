from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from finetuner.core.artifacts import atomic_write_json, stable_digest
from finetuner.distillation.config import DistillationConfig, DistillationTechnique
from finetuner.distillation.domains import select_domain_rows


class TeacherGenerator(Protocol):
    def __call__(self, prompts: list[str], config: DistillationConfig) -> list[str]: ...


@dataclass(frozen=True)
class DistillationDatasetResult:
    dataset_path: str
    manifest_path: str
    selected_count: int


def _extract_prompt(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None:
        for fallback in ("prompt", "instruction", "question", "text", "input"):
            if row.get(fallback):
                value = row[fallback]
                break
    return str(value or "").strip()


def build_sequence_distillation_dataset(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    config: DistillationConfig,
    teacher_generate: TeacherGenerator,
    *,
    batch_size: int = 8,
    log: Callable[[str], None] | None = None,
) -> DistillationDatasetResult:
    config.require_valid()
    selected = select_domain_rows(rows, config.domain, limit=config.max_samples)
    prompts = [prompt for row in selected if (prompt := _extract_prompt(row, config.prompt_field))]
    if not prompts:
        raise ValueError("No prompts matched the selected distillation domains")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "teacher_sequences.jsonl"
    generated_rows: list[dict[str, Any]] = []
    with dataset_path.open("w", encoding="utf-8", newline="\n") as stream:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            completions = teacher_generate(batch, config)
            if len(completions) != len(batch):
                raise RuntimeError(
                    f"Teacher returned {len(completions)} responses for {len(batch)} prompts"
                )
            for prompt, response in zip(batch, completions):
                record = {
                    "prompt": prompt,
                    "response": str(response),
                    "text": f"### Instruction:\n{prompt}\n\n### Response:\n{response}",
                }
                generated_rows.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if log:
                log(f"Teacher generation: {min(start + len(batch), len(prompts))}/{len(prompts)}")

    manifest = {
        "schema_version": 1,
        "technique": DistillationTechnique.SEQUENCE.value,
        "teacher_model": config.teacher_model,
        "student_model": config.student_model,
        "domain": config.domain.to_dict(),
        "selected_count": len(generated_rows),
        "dataset_digest": stable_digest(generated_rows),
        "generation": {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "seed": config.seed,
        },
    }
    manifest_path = output / "distillation_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return DistillationDatasetResult(str(dataset_path), str(manifest_path), len(generated_rows))


def transformers_teacher_generator(model_id: str) -> TeacherGenerator:
    """Create a batched local teacher. Loading is deferred until execution."""
    state: dict[str, Any] = {}

    def generate(prompts: list[str], config: DistillationConfig) -> list[str]:
        if not state:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            model.eval()
            state.update(model=model, tokenizer=tokenizer)
        model, tokenizer = state["model"], state["tokenizer"]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        import torch

        with torch.inference_mode():
            outputs = model.generate(
                **encoded,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.temperature > 0,
                temperature=max(config.temperature, 1e-5),
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        return tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)

    return generate
