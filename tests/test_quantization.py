from __future__ import annotations

import pytest

from finetuner.quantization.planner import compatible_backends, recommended_config
from finetuner.quantization.runner import build_quantization_commands
from finetuner.quantization.specs import (
    DeviceTarget,
    QuantizationBackend,
    QuantizationConfig,
)


def test_backend_target_matrix_rejects_false_portability_claims():
    config = QuantizationConfig(backend="openvino", target="apple_gpu", bits=4)
    assert "does not support" in "; ".join(config.validate())
    assert QuantizationBackend.OPENVINO not in compatible_backends(DeviceTarget.APPLE_GPU)
    assert QuantizationBackend.GGUF in compatible_backends(DeviceTarget.APPLE_GPU)


def test_awq_requires_calibration_data():
    config = QuantizationConfig(backend="awq", target="nvidia_gpu", bits=4)
    assert any("calibration" in error for error in config.validate())


def test_recommendations_choose_npu_specific_runtimes():
    assert recommended_config(DeviceTarget.INTEL_NPU).backend == "openvino"
    with pytest.raises(ValueError, match="QNN/QAIRT"):
        recommended_config(DeviceTarget.QUALCOMM_NPU)
    assert recommended_config(DeviceTarget.CPU).backend == "gguf"


def test_openvino_command_is_an_argv_array(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "finetuner.quantization.runner._resolve_executable", lambda name: f"/bin/{name}"
    )
    config = QuantizationConfig(backend="openvino", target="intel_npu", bits=4)
    commands = build_quantization_commands("model", str(tmp_path), config)
    assert commands[0][:4] == ["/bin/optimum-cli", "export", "openvino", "--model"]
    assert "--weight-format" in commands[0]
    assert commands[0][-1] == str(tmp_path.resolve())


def test_onnx_int8_plan_exports_then_quantizes(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "finetuner.quantization.runner._resolve_executable", lambda name: f"/bin/{name}"
    )
    config = QuantizationConfig(backend="onnx", target="cpu", bits=8)
    commands = build_quantization_commands("model", str(tmp_path), config)
    assert len(commands) == 2
    assert commands[0][1:3] == ["export", "onnx"]
    assert commands[1][1:3] == ["onnxruntime", "quantize"]
    assert "--per_channel" in commands[1]


def test_gguf_command_requires_explicit_toolchain(monkeypatch, tmp_path):
    root = tmp_path / "llama cpp"
    root.mkdir()
    (root / "convert_hf_to_gguf.py").write_text("", encoding="utf-8")
    (root / "llama-quantize.exe").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "finetuner.quantization.runner._resolve_executable", lambda name: "python.exe"
    )
    config = QuantizationConfig(backend="gguf", target="cpu", bits=4, llama_cpp_path=str(root))
    commands = build_quantization_commands("model", str(tmp_path / "out"), config)
    assert len(commands) == 2
    assert commands[1][-1] == "Q4_K_M"
    assert all(isinstance(command, list) for command in commands)
