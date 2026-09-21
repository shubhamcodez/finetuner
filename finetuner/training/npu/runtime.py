from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NpuModelSpec:
    model_dir: Path
    onnx_path: Path
    hidden_size: int
    vocab_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    input_ids_name: str = "input_ids"
    attention_mask_name: str = "attention_mask"
    logits_name: str = "logits"
    past_key: str = "past_key_values.%d.key"
    past_value: str = "past_key_values.%d.value"


def default_npu_artifact() -> Path:
    return Path.home() / ".finetuner" / "bench" / "models" / "qwen2.5-0.5b-instruct-onnx-genai"


def resolve_npu_artifact(path: str = "") -> Path:
    candidates = []
    if path.strip():
        candidates.append(Path(path))
    candidates.append(default_npu_artifact())
    for candidate in candidates:
        if (candidate / "genai_config.json").is_file() and (
            (candidate / "model.onnx").is_file() or any(candidate.glob("*.onnx"))
        ):
            return candidate
    raise FileNotFoundError(
        "No QNN/ONNX-GenAI package found. Download a Hexagon-ready package "
        f"(expected {default_npu_artifact()} or set the NPU artifact path)."
    )


def load_spec(model_dir: Path) -> NpuModelSpec:
    payload = json.loads((model_dir / "genai_config.json").read_text(encoding="utf-8"))
    decoder = payload.get("model", {}).get("decoder", {})
    inputs = decoder.get("inputs", {})
    outputs = decoder.get("outputs", {})
    onnx_name = str(decoder.get("filename") or "model.onnx")
    onnx_path = model_dir / onnx_name
    if not onnx_path.is_file():
        found = next(iter(sorted(model_dir.glob("*.onnx"))), None)
        if found is None:
            raise FileNotFoundError(f"No ONNX decoder in {model_dir}")
        onnx_path = found
    return NpuModelSpec(
        model_dir=model_dir,
        onnx_path=onnx_path,
        hidden_size=int(decoder.get("hidden_size") or 896),
        vocab_size=int(payload.get("model", {}).get("vocab_size") or 151936),
        num_layers=int(decoder.get("num_hidden_layers") or 24),
        num_kv_heads=int(decoder.get("num_key_value_heads") or 2),
        head_dim=int(decoder.get("head_size") or 64),
        input_ids_name=str(inputs.get("input_ids") or "input_ids"),
        attention_mask_name=str(inputs.get("attention_mask") or "attention_mask"),
        logits_name=str(outputs.get("logits") or "logits"),
        past_key=str(inputs.get("past_key_names") or "past_key_values.%d.key"),
        past_value=str(inputs.get("past_value_names") or "past_key_values.%d.value"),
    )


def npu_provider_available() -> bool:
    try:
        import onnxruntime as ort
        import onnxruntime_qnn as qnn

        ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())
        return any(
            device.ep_name == "QNNExecutionProvider"
            and device.device.type == ort.OrtHardwareDeviceType.NPU
            for device in ort.get_ep_devices()
        )
    except Exception:
        return False


class NpuBackbone:
    """Frozen Qwen decoder on Hexagon HTP (or CPU EP if QNN is missing)."""

    def __init__(self, spec: NpuModelSpec, log=lambda _msg: None) -> None:
        self.spec = spec
        self.session, self.provider = _open_session(spec, log)
        self.output_names = [item.name for item in self.session.get_outputs()]

    def logits(self, input_ids: np.ndarray, past_len: int = 0) -> np.ndarray:
        feeds = _feeds(self.spec, input_ids, past_len)
        outputs = self.session.run([self.spec.logits_name], feeds)
        logits = np.asarray(outputs[0])
        if logits.ndim == 3:
            return logits[0]
        return logits


def _open_session(spec: NpuModelSpec, log) -> tuple[Any, str]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    provider = "CPUExecutionProvider"
    try:
        import onnxruntime_qnn as qnn

        ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())
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
        provider = "QNNExecutionProvider"
        log("NPU backbone: Qualcomm Hexagon HTP via QNNExecutionProvider")
    except Exception as exc:
        options.add_provider("CPUExecutionProvider", {})
        log(f"QNN/HTP unavailable ({exc}); NPU trainer falling back to ONNX CPU EP")
    session = ort.InferenceSession(str(spec.onnx_path), sess_options=options)
    log(f"ONNX providers: {session.get_providers()}")
    return session, provider


def _feeds(spec: NpuModelSpec, input_ids: np.ndarray, past_len: int) -> dict[str, np.ndarray]:
    tokens = np.asarray(input_ids, dtype=np.int64)
    if tokens.ndim == 1:
        tokens = tokens[None, :]
    seq_len = tokens.shape[1]
    feeds: dict[str, np.ndarray] = {
        spec.input_ids_name: tokens,
        spec.attention_mask_name: np.ones((1, past_len + seq_len), dtype=np.int64),
    }
    empty_key = np.zeros((1, spec.num_kv_heads, past_len, spec.head_dim), dtype=np.float32)
    empty_value = np.zeros((1, spec.num_kv_heads, past_len, spec.head_dim), dtype=np.float32)
    for layer in range(spec.num_layers):
        feeds[spec.past_key % layer] = empty_key
        feeds[spec.past_value % layer] = empty_value
    return feeds
