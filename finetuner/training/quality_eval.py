from __future__ import annotations

from typing import Callable

from finetuner.datasets.hf_datasets import extract_gsm8k_answer
from finetuner.training.chat_format import apply_chat_prompt, user_and_assistant

LogFn = Callable[[str], None]


def score_generate(
    model_path: str,
    rows: list[dict],
    *,
    max_new_tokens: int = 256,
    max_samples: int = 0,
    log: LogFn | None = None,
) -> dict[str, float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sample = rows[:max_samples] if max_samples else rows
    if not sample:
        return {"accuracy": 0.0, "n": 0.0}
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    correct = 0
    for index, row in enumerate(sample, start=1):
        user, gold = user_and_assistant(row)
        prompt = apply_chat_prompt(tokenizer, user)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        if _match(row, gold, text):
            correct += 1
        if log and (index == 1 or index % 20 == 0 or index == len(sample)):
            log(f"Quality eval {index}/{len(sample)} acc={100.0 * correct / index:.1f}%")
    n = float(len(sample))
    scores = {"accuracy": 100.0 * correct / n, "n": n}
    if log:
        log(f"Quality generate: {correct}/{int(n)} = {scores['accuracy']:.1f}%")
    del model
    return scores


def _match(row: dict, gold: str, predicted: str) -> bool:
    choices = [str(item) for item in (row.get("choices") or []) if str(item)]
    if choices:
        gold_index = int(row.get("gold_index") or 0)
        picked = _pick_choice(predicted, choices)
        return picked == gold_index
    expected = extract_gsm8k_answer(gold or str(row.get("answer") or ""))
    got = extract_gsm8k_answer(predicted)
    if expected and got and expected.strip() == got.strip():
        return True
    return bool(expected) and expected.lower() in predicted.lower()


def _pick_choice(predicted: str, choices: list[str]) -> int:
    lowered = predicted.lower()
    for index, choice in enumerate(choices):
        if choice.lower() in lowered:
            return index
    return -1
