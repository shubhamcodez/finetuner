"""Bind the INT8 Qwen 0.5B graph to Hexagon and measure TPS."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn

MODEL = (
    Path.home()
    / ".finetuner"
    / "bench"
    / "models"
    / "qwen2.5-0.5b-onnx-int8"
    / "onnx"
    / "model_int8.onnx"
)
OUT = Path.home() / ".finetuner" / "bench" / "qwen05b-npu-int8-metrics.json"
LAYERS = 24
REPEATS = 3


def session(kind: str) -> ort.InferenceSession:
    ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_profiling = kind == "qnn"
    options.profile_file_prefix = str(Path.home() / ".finetuner" / "bench" / "qnn-int8-profile")
    if kind == "qnn":
        npu = next(
            device
            for device in ort.get_ep_devices()
            if device.ep_name == "QNNExecutionProvider"
            and device.device.type == ort.OrtHardwareDeviceType.NPU
        )
        options.add_provider_for_devices(
            [npu],
            {
                "backend_path": qnn.get_qnn_htp_path(),
                "htp_performance_mode": "burst",
                "enable_htp_fp16_precision": "1",
            },
        )
    else:
        options.add_provider("CPUExecutionProvider", {})
    started = time.perf_counter()
    loaded = ort.InferenceSession(str(MODEL), sess_options=options)
    print(kind, loaded.get_providers(), f"{time.perf_counter() - started:.1f}s")
    return loaded


def feeds(seq_len: int, past_len: int) -> dict[str, np.ndarray]:
    payload = {
        "input_ids": np.ones((1, seq_len), dtype=np.int64),
        "attention_mask": np.ones((1, past_len + seq_len), dtype=np.int64),
        "position_ids": np.arange(past_len, past_len + seq_len, dtype=np.int64).reshape(1, -1),
    }
    key = np.zeros((1, 2, past_len, 64), dtype=np.float32)
    value = np.zeros((1, 2, past_len, 64), dtype=np.float32)
    for layer in range(LAYERS):
        payload[f"past_key_values.{layer}.key"] = key
        payload[f"past_key_values.{layer}.value"] = value
    return payload


def profile_providers(loaded: ort.InferenceSession) -> dict[str, int]:
    loaded.run(None, feeds(1, 128))
    path = loaded.end_profiling()
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    for row in rows:
        args = row.get("args") or {}
        if isinstance(args, dict) and "provider" in args:
            counts[args["provider"]] += 1
    print("node providers", dict(counts))
    return dict(counts)


def repeat(loaded: ort.InferenceSession, label: str, seq_len: int, past_len: int, steps: int) -> dict:
    loaded.run(None, feeds(seq_len, past_len))
    samples = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        tokens = 0
        for _step in range(steps):
            loaded.run(None, feeds(seq_len, past_len))
            tokens += seq_len
        ms = 1000 * (time.perf_counter() - started)
        samples.append({"tps": tokens / (ms / 1000), "ms_per_token": ms / tokens, "ms": ms})
        print(f"{label}: {samples[-1]['tps']:.1f} tok/s ({samples[-1]['ms_per_token']:.2f} ms/tok)")
    return {
        "mean_tps": sum(item["tps"] for item in samples) / len(samples),
        "mean_ms_per_token": sum(item["ms_per_token"] for item in samples) / len(samples),
        "samples": samples,
    }


def main() -> None:
    qnn_sess = session("qnn")
    split = profile_providers(qnn_sess)
    qnn_metrics = {
        "providers": qnn_sess.get_providers(),
        "node_providers": split,
        "pp128": repeat(qnn_sess, "qnn pp128", 128, 0, 1),
        "tg64": repeat(qnn_sess, "qnn tg64", 1, 128, 64),
    }
    cpu_sess = session("cpu")
    cpu_metrics = {
        "providers": cpu_sess.get_providers(),
        "pp128": repeat(cpu_sess, "cpu pp128", 128, 0, 1),
        "tg64": repeat(cpu_sess, "cpu tg64", 1, 128, 64),
    }
    OUT.write_text(
        json.dumps({"qnn": qnn_metrics, "cpu": cpu_metrics}, indent=2),
        encoding="utf-8",
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
