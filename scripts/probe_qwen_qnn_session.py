"""Try to compile the 0.5B decoder ONNX onto Hexagon HTP."""

from __future__ import annotations

import time
from pathlib import Path

import onnxruntime as ort
import onnxruntime_qnn as qnn

MODEL = Path.home() / ".finetuner" / "bench" / "models" / "qwen2.5-0.5b-instruct-onnx-genai" / "model.onnx"

ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())
npu = next(
    device
    for device in ort.get_ep_devices()
    if device.ep_name == "QNNExecutionProvider" and device.device.type == ort.OrtHardwareDeviceType.NPU
)
print("NPU", npu.device.metadata)
session_options = ort.SessionOptions()
session_options.log_severity_level = 1
session_options.add_provider_for_devices(
    [npu],
    {
        "backend_path": qnn.get_qnn_htp_path(),
        "htp_performance_mode": "burst",
        "enable_htp_fp16_precision": "1",
    },
)
started = time.perf_counter()
session = ort.InferenceSession(str(MODEL), sess_options=session_options)
print(f"compiled in {time.perf_counter() - started:.1f}s")
print("providers", session.get_providers())
print("inputs", len(session.get_inputs()), [item.name for item in session.get_inputs()[:6]])
