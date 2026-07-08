from __future__ import annotations

from typing import Callable

from finetuner.core.job import EvalResult
from finetuner.datasets.hf_datasets import (
    EVAL_DATASET_SPECS,
    extract_gsm8k_answer,
    load_env_file,
    load_hf_split,
)
from finetuner.eval.tasks import EVAL_TASKS


def run_evals(
    model_path: str,
    task_ids: list[str],
    max_samples: int = 100,
    log_callback: Callable[[str], None] | None = None,
) -> list[EvalResult]:
    if not task_ids:
        return []

    load_env_file()

    def log(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    log(f"Running evals: {', '.join(task_ids)}")
    log(f"Model: {model_path}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("Loading model for eval...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    results: list[EvalResult] = []
    for task_id in task_ids:
        score = _run_task(model, tokenizer, task_id, max_samples, log)
        results.append(
            EvalResult(
                task_id=task_id,
                task_name=EVAL_TASKS[task_id].name,
                score=score,
                metric="acc",
            )
        )
    return results


def _build_prompt(row: dict, spec: dict) -> str:
    question = str(row.get(spec["question"], ""))
    parser = spec.get("answer_parser", "")

    if parser == "mmlu":
        choices = row.get("choices", [])
        lines = [question, ""]
        for i, choice in enumerate(choices[:4]):
            lines.append(f"{chr(65 + i)}) {choice}")
        lines.append("\nAnswer:")
        return "\n".join(lines)

    if parser == "index":
        return f"{question}\n\nComplete the sentence:"

    if parser == "arc":
        choices = row.get("choices", {})
        texts = choices.get("text", [])
        labels = choices.get("label", [])
        lines = [question, ""]
        for label, text in zip(labels, texts):
            lines.append(f"{label}) {text}")
        lines.append("\nAnswer:")
        return "\n".join(lines)

    return f"Question: {question}\n\nAnswer:"


def _expected_answer(row: dict, spec: dict) -> str:
    parser = spec.get("answer_parser", "")
    raw = row.get(spec["answer"])

    if parser == "gsm8k":
        return extract_gsm8k_answer(str(raw))

    if parser == "mmlu":
        choices = row.get("choices", [])
        try:
            idx = int(raw)
            return str(choices[idx])
        except (TypeError, ValueError, IndexError):
            return str(raw)

    if parser == "index":
        endings = row.get(spec.get("choices", "endings"), [])
        try:
            idx = int(raw)
            return str(endings[idx])
        except (TypeError, ValueError, IndexError):
            return str(raw)

    if parser == "arc":
        choices = row.get("choices", {})
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        for label, text in zip(labels, texts):
            if label == raw:
                return str(text)
        return str(raw)

    return str(raw)


def _run_task(model, tokenizer, task_id: str, max_samples: int, log) -> float:
    import torch

    if task_id not in EVAL_TASKS:
        raise ValueError(f"Unknown eval task: {task_id}")

    spec = EVAL_DATASET_SPECS.get(task_id)
    if not spec:
        raise ValueError(f"No eval dataset configured for task: {task_id}")

    ds = load_hf_split(spec["name"], spec.get("config"), spec["split"])

    correct = 0
    total = min(max_samples, len(ds))
    if total == 0:
        raise ValueError(f"Eval dataset for {task_id} is empty.")

    for i in range(total):
        row = ds[i]
        prompt = _build_prompt(row, spec)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        expected = _expected_answer(row, spec)
        if task_id == "gsm8k":
            gen_answer = extract_gsm8k_answer(generated)
            if gen_answer and expected and gen_answer.strip() == expected.strip():
                correct += 1
            elif expected and expected in generated:
                correct += 1
        elif expected and expected.lower() in generated.lower():
            correct += 1

    score = correct / total * 100
    log(f"{EVAL_TASKS[task_id].name}: {correct}/{total} = {score:.1f}%")
    return score
