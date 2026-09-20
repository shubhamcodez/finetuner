"""Turn common finetune dataset rows into a single text example."""

from __future__ import annotations


def messages_to_text(messages: list) -> str:
    parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("from") or "user"
        content = msg.get("content") or msg.get("value") or ""
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _pair_to_text(instruction: str, response: str, source: str = "") -> str | None:
    if not str(response).strip():
        return None
    prompt = str(instruction).strip()
    heading = source or "Instruction"
    if prompt:
        return f"### {heading}:\n{prompt}\n### Response:\n{response}"
    return f"### Response:\n{response}"


def row_to_text(row: dict, formatter=None) -> str | None:
    if formatter:
        text = formatter(row)
        if text:
            return text
    if row.get("text"):
        return str(row["text"])
    for key in ("messages", "conversations", "conversation"):
        if isinstance(row.get(key), list):
            rendered = messages_to_text(row[key])
            if rendered:
                return rendered
    if row.get("instruction") and row.get("output"):
        inst = str(row.get("instruction", ""))
        inp = str(row.get("input", "")).strip()
        out = str(row.get("output", ""))
        text = f"### Instruction:\n{inst}\n"
        if inp:
            text += f"### Input:\n{inp}\n"
        text += f"### Response:\n{out}"
        return text
    if row.get("prompt") and row.get("response"):
        return _pair_to_text(row["prompt"], row["response"])
    if row.get("prompt") and row.get("completion"):
        return _pair_to_text(row["prompt"], row["completion"])
    if row.get("question") and row.get("answer"):
        return _pair_to_text(row["question"], row["answer"], "Question")
    if row.get("input") and row.get("output"):
        return _pair_to_text(row["input"], row["output"])
    chosen = row.get("chosen")
    if isinstance(chosen, str) and chosen.strip():
        return _pair_to_text(row.get("prompt") or row.get("question") or "", chosen)
    if isinstance(chosen, list):
        rendered = messages_to_text(chosen)
        if rendered:
            return rendered
    return None
