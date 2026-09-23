"""Send one turn to the local OpenAI-compatible inference server."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatTurn:
    text: str
    metrics: str = ""


def reply_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise RuntimeError("Server returned no reply")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content")
    if content is None:
        content = choice.get("text")
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
        content = "".join(parts)
    text = str(content or "").strip()
    if not text:
        raise RuntimeError("Server returned an empty reply")
    return text


def _number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            return number
    return None


def _latency(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def inference_metrics(payload: dict[str, Any], elapsed_s: float) -> str:
    """Compact tok/s, latency, and token counts for the last completion."""
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
    prompt = _number(usage.get("prompt_tokens"), timings.get("prompt_n"))
    completion = _number(usage.get("completion_tokens"), timings.get("predicted_n"))
    prefill_ms = _number(timings.get("prompt_ms"))
    predicted_ms = _number(timings.get("predicted_ms"))
    gen_tps = _number(timings.get("predicted_per_second"))
    if gen_tps is None and completion and predicted_ms:
        gen_tps = completion / (predicted_ms / 1000)
    if gen_tps is None and completion and elapsed_s > 0:
        gen_tps = completion / elapsed_s
    parts: list[str] = []
    if gen_tps is not None:
        parts.append(f"{gen_tps:.1f} tok/s")
    if elapsed_s >= 0:
        parts.append(_latency(elapsed_s))
    if prefill_ms is not None:
        parts.append(f"prefill {_latency(prefill_ms / 1000)}")
    if completion is not None:
        generated = str(int(completion))
        if prompt is not None:
            parts.append(f"{int(prompt)}→{generated} tok")
        else:
            parts.append(f"{generated} tok")
    return " · ".join(parts)


def request_chat(
    base_url: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 256,
    timeout: float = 180,
) -> ChatTurn:
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps(
        {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Chat request failed ({exc.code}): {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach the server at {base_url}: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Server returned an unexpected reply")
    elapsed = time.perf_counter() - started
    return ChatTurn(reply_text(payload), inference_metrics(payload, elapsed))
