"""OpenAI-compatible HTTP fallback when a specialist serve CLI is unavailable."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse


class GenerationBackend:
    def model_id(self) -> str:
        return "finetuner"

    def generate(self, prompt: str, max_tokens: int) -> str:
        raise NotImplementedError


class CallbackBackend(GenerationBackend):
    def __init__(self, generate: Callable[[str, int], str], model_id: str = "finetuner") -> None:
        self._generate = generate
        self._model_id = model_id

    def model_id(self) -> str:
        return self._model_id

    def generate(self, prompt: str, max_tokens: int) -> str:
        return self._generate(prompt, max_tokens)


def _messages_to_prompt(payload: dict[str, Any]) -> str:
    if payload.get("prompt"):
        return str(payload["prompt"])
    messages = payload.get("messages") or []
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


def make_handler(backend: GenerationBackend) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/health", "/v1/health"}:
                self._json(200, {"status": "ok"})
                return
            if path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": backend.model_id(), "object": "model"}]})
                return
            self._json(404, {"error": {"message": f"unknown path {path}"}})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path not in {"/v1/chat/completions", "/v1/completions"}:
                self._json(404, {"error": {"message": f"unknown path {path}"}})
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._json(400, {"error": {"message": "invalid JSON"}})
                return
            prompt = _messages_to_prompt(payload if isinstance(payload, dict) else {})
            max_tokens = int(payload.get("max_tokens") or 64) if isinstance(payload, dict) else 64
            try:
                text = backend.generate(prompt, max(1, min(max_tokens, 2048)))
            except Exception as exc:
                self._json(503, {"error": {"message": str(exc)}})
                return
            self._json(
                200,
                {
                    "id": "finetuner-cmpl",
                    "object": "chat.completion" if path.endswith("chat/completions") else "text_completion",
                    "model": backend.model_id(),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "text": text,
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

    return Handler


def start_builtin_server(
    host: str,
    port: int,
    backend: GenerationBackend,
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(backend))
    server.daemon_threads = True
    return server
