"""Measure Qwen2.5-0.5B on Hexagon HTP vs CPU, using a QNN-bound ORT session."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn

MODEL = (
    Path.home()
    / ".finetuner"
    / "bench"
    / "models"
    / "qwen2.5-0.5b-instruct-onnx-genai"
    / "model.onnx"
)
OUT = Path.home() / ".finetuner" / "bench" / "qwen05b-npu-metrics.json"
LAYERS = 24
HEADS = 2
HEAD_DIM = 64
REPEATS = 3


def _register() -> None:
    ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())


def _npu_device():
    return next(
        device
        for device in ort.get_ep_devices()
        if device.ep_name == "QNNExecutionProvider"
        and device.device.type == ort.OrtHardwareDeviceType.NPU
    )


def _session(kind: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    if kind == "qnn":
        options.add_provider_for_devices(
            [_npu_device()],
            {
                "backend_path": qnn.get_qnn_htp_path(),
                "htp_performance_mode": "burst",
                "enable_htp_fp16_precision": "1",
            },
        )
    else:
        options.add_provider("CPUExecutionProvider", {})
    started = time.perf_counter()
    session = ort.InferenceSession(str(MODEL), sess_options=options)
    print(
        f"{kind} session {session.get_providers()} in {1000 * (time.perf_counter() - started):.0f} ms"
    )
    return session


def _feeds(seq_len: int, past_len: int) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {
        "input_ids": np.ones((1, seq_len), dtype=np.int64),
        "attention_mask": np.ones((1, past_len + seq_len), dtype=np.int64),
    }
    key = np.zeros((1, HEADS, past_len, HEAD_DIM), dtype=np.float32)
    value = np.zeros((1, HEADS, past_len, HEAD_DIM), dtype=np.float32)
    for layer in range(LAYERS):
        feeds[f"past_key_values.{layer}.key"] = key
        feeds[f"past_key_values.{layer}.value"] = value
    return feeds


def _run_once(session: ort.InferenceSession, seq_len: int, past_len: int) -> float:
    feeds = _feeds(seq_len, past_len)
    started = time.perf_counter()
    session.run(None, feeds)
    return 1000 * (time.perf_counter() - started)


def _repeat(session: ort.InferenceSession, name: str, seq_len: int, past_len: int, steps: int) -> dict:
    _run_once(session, seq_len, past_len)
    samples = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        tokens = 0
        for _step in range(steps):
            _run_once(session, seq_len, past_len)
            tokens += seq_len
        elapsed_ms = 1000 * (time.perf_counter() - started)
        samples.append(
            {
                "tokens": tokens,
                "ms": elapsed_ms,
                "tps": tokens / (elapsed_ms / 1000),
                "ms_per_token": elapsed_ms / tokens,
            }
        )
        print(
            f"{name} {tokens} tok: {samples[-1]['tps']:.1f} tok/s "
            f"({samples[-1]['ms_per_token']:.2f} ms/tok)"
        )
    return {
        "mean_tps": sum(item["tps"] for item in samples) / len(samples),
        "mean_ms_per_token": sum(item["ms_per_token"] for item in samples) / len(samples),
        "samples": samples,
    }


def bench(kind: str) -> dict:
    session = _session(kind)
    return {
        "providers": session.get_providers(),
        "pp128": _repeat(session, f"{kind} pp128", seq_len=128, past_len=0, steps=1),
        "tg64": _repeat(session, f"{kind} tg64", seq_len=1, past_len=128, steps=64),
        "tg128": _repeat(session, f"{kind} tg128", seq_len=1, past_len=128, steps=128),
    }


def main() -> None:
    _register()
    payload = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "artifact": "justinchuby/qwen2.5-0.5b-instruct-onnx-genai",
        "npu": "Snapdragon X Elite Hexagon HTP via QNNExecutionProvider",
        "qnn": bench("qnn"),
        "cpu": bench("cpu"),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
