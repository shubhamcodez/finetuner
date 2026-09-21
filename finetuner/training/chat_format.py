from __future__ import annotations

import re

_HEADING = re.compile(
    r"^(?:###\s*(?:Instruction|Input|Response)\s*:?\s*\n?)+",
    re.IGNORECASE | re.MULTILINE,
)


def _unwrap_prompt(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"###\s*Instruction:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Solve this grade-school math word problem\.\s*", "", cleaned)
    cleaned = re.sub(r"Answer the (?:multiple-choice|science) question\.\s*", "", cleaned)
    cleaned = re.sub(r"Complete the sentence with the most plausible ending\.\s*", "", cleaned)
    cleaned = re.sub(r"###\s*Input:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"###\s*Response:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def user_and_assistant(row: dict) -> tuple[str, str]:
    if row.get("question") and (row.get("answer") or row.get("gold")):
        return str(row["question"]).strip(), str(row.get("answer") or row.get("gold")).strip()
    if row.get("ctx") and row.get("gold"):
        return str(row["ctx"]).strip(), str(row["gold"]).strip()
    prompt = str(row.get("prompt") or "")
    gold = str(row.get("gold") or row.get("response") or row.get("chosen") or "")
    if prompt and gold:
        return _unwrap_prompt(prompt), gold.strip()
    text = str(row.get("text") or "")
    if "### Response:" in text:
        left, right = text.split("### Response:", 1)
        return _unwrap_prompt(left), right.strip()
    return text.strip(), ""


def row_to_messages(row: dict) -> list[dict[str, str]]:
    raw = row.get("messages")
    if isinstance(raw, list) and raw:
        messages = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if content:
                messages.append(
                    {"role": str(item.get("role") or "user"), "content": content}
                )
        if len(messages) >= 2:
            return messages
    user, assistant = user_and_assistant(row)
    if not user:
        user = "Answer the question."
    messages = [{"role": "user", "content": user}]
    if assistant:
        messages.append({"role": "assistant", "content": assistant})
    return messages


def ensure_messages(rows) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        item["messages"] = row_to_messages(item)
        out.append(item)
    return out


def apply_chat_prompt(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    if getattr(tokenizer, "apply_chat_template", None) and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{user_text.strip()}\n"