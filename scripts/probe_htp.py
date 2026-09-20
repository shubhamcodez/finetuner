"""Bind a tiny graph to Hexagon HTP via ORT's device API."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_qnn as qnn
from onnx import TensorProto, helper, numpy_helper

ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())

weights = np.random.randn(256, 256).astype(np.float32)
graph = helper.make_graph(
    [helper.make_node("MatMul", ["X", "W"], ["Y"])],
    "htp_probe",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 256])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 256])],
    [numpy_helper.from_array(weights, name="W")],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 10
path = Path(tempfile.gettempdir()) / "htp_probe.onnx"
onnx.save(model, path)

npu = next(
    device
    for device in ort.get_ep_devices()
    if device.ep_name == "QNNExecutionProvider"
    and device.device.type == ort.OrtHardwareDeviceType.NPU
)
print("selected", npu.ep_name, npu.device.type, nnu_meta := npu.device.metadata)

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
print("has_providers", session_options.has_providers())
session = ort.InferenceSession(str(path), sess_options=session_options)
print("providers", session.get_providers())
print("provider options", session.get_provider_options())
output = session.run(None, {"X": np.random.randn(1, 256).astype(np.float32)})[0]
print("output", output.shape, float(output.mean()))
