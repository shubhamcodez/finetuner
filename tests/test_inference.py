from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetuner.core.job import ProjectConfig
from finetuner.inference.devices import (
    apply_device_recipe,
    detect_htp_ready,
    preferred_npu_target,
    recipe_for_target,
    select_best_runtime,
)
from finetuner.inference.planner import (
    AcceleratorInventory,
    backend_engine_compatibility_error,
    compatible_engines,
    estimate_kv_cache_bytes,
    recommended_config,
    runtime_features,
)
from finetuner.inference.runner import (
    build_compile_commands,
    build_serve_command,
    detect_source_format,
    optimize_inference_engine,
    runtime_options,
)
from finetuner.inference.specs import (
    InferenceEngine,
    InferenceOptimizationConfig,
)
from finetuner.quantization.planner import HardwareCapability
from finetuner.quantization.specs import DeviceTarget


def test_engine_target_matrix_rejects_false_portability_claims():
    config = InferenceOptimizationConfig(engine="vllm", target="apple_gpu")
    assert "does not support" in "; ".join(config.validate())
    assert InferenceEngine.VLLM not in compatible_engines(DeviceTarget.APPLE_GPU)
    assert InferenceEngine.LLAMACPP in compatible_engines(DeviceTarget.APPLE_GPU)


def test_quantization_backend_must_match_inference_engine():
    assert backend_engine_compatibility_error("gguf", "llamacpp") is None
    assert backend_engine_compatibility_error("awq", "vllm") is None
    assert "does not consume gguf" in (backend_engine_compatibility_error("gguf", "vllm") or "")


def test_qualcomm_npu_requires_the_qnn_provider_recipe():
    generic = InferenceOptimizationConfig(engine="onnxruntime", target="qualcomm_npu")
    assert any("QNN" in error for error in generic.validate())
    vllm = InferenceOptimizationConfig(engine="vllm", target="qualcomm_npu")
    assert any("QNN" in error or "does not support" in error for error in vllm.validate())
    config = recommended_config(DeviceTarget.QUALCOMM_NPU)
    assert config.engine == "onnxruntime"
    assert config.extra_options["execution_provider"] == "QNNExecutionProvider"
    assert not config.validate()


def test_recommendations_choose_runtime_specific_engines():
    assert recommended_config(DeviceTarget.INTEL_NPU).engine == "openvino"
    assert recommended_config(DeviceTarget.NVIDIA_GPU, vllm_available=True).engine == "vllm"
    assert recommended_config(DeviceTarget.NVIDIA_GPU, vllm_available=False).engine == "llamacpp"
    assert recommended_config(DeviceTarget.AMD_GPU).engine == "llamacpp"
    assert recommended_config(DeviceTarget.CPU).engine == "llamacpp"
    tight = recommended_config(DeviceTarget.NVIDIA_GPU, memory_gb=10, vllm_available=True)
    assert tight.kv_cache_dtype == "fp8"
    assert tight.max_context == 4096


def test_adaptive_runtime_skips_hexagon_without_an_htp_package():
    caps = [
        HardwareCapability(DeviceTarget.CPU, True, "Oryon"),
        HardwareCapability(DeviceTarget.QUALCOMM_NPU, True, "Hexagon NPU"),
        HardwareCapability(DeviceTarget.NVIDIA_GPU, False, "missing"),
    ]
    choice = select_best_runtime(caps, htp_ready=False)
    assert choice.recipe.target == DeviceTarget.CPU
    assert choice.recipe.inference.engine == "llamacpp"
    assert any("QNN/HTP" in item for item in choice.skipped)


def test_adaptive_runtime_uses_qnn_when_the_package_is_htp_ready():
    caps = [
        HardwareCapability(DeviceTarget.CPU, True, "Oryon"),
        HardwareCapability(DeviceTarget.QUALCOMM_NPU, True, "Hexagon NPU"),
    ]
    choice = select_best_runtime(caps, htp_ready=True)
    assert choice.recipe.target == DeviceTarget.QUALCOMM_NPU
    assert choice.recipe.inference.engine == "onnxruntime"


def test_adaptive_runtime_prefers_nvidia_when_present():
    caps = [
        HardwareCapability(DeviceTarget.CPU, True, "cpu"),
        HardwareCapability(DeviceTarget.NVIDIA_GPU, True, "RTX"),
        HardwareCapability(DeviceTarget.QUALCOMM_NPU, True, "Hexagon"),
    ]
    choice = select_best_runtime(caps, htp_ready=True, vllm_available=True)
    assert choice.recipe.target == DeviceTarget.NVIDIA_GPU
    assert choice.recipe.inference.engine == "vllm"


def test_detect_htp_ready_requires_a_qnn_context_binary(tmp_path):
    assert detect_htp_ready(str(tmp_path)) is False
    (tmp_path / "genai_config.json").write_text('{"model":{}}', encoding="utf-8")
    assert detect_htp_ready(str(tmp_path)) is False
    (tmp_path / "decoder.bin").write_bytes(b"qnn")
    (tmp_path / "genai_config.json").write_text(
        '{"model":{"decoder":{"session_options":{"provider_options":[{"qnn":{}}]}}}}',
        encoding="utf-8",
    )
    assert detect_htp_ready(str(tmp_path)) is True


def test_device_recipe_applies_npu_and_gpu_targets():
    project = ProjectConfig()
    npu = apply_device_recipe(project, DeviceTarget.QUALCOMM_NPU, run_on_device=True)
    assert npu.inference.engine == "onnxruntime"
    assert npu.quantization.backend == "onnx"
    assert project.inference.extra_options["run_on_device"] is True
    nvidia = recipe_for_target(DeviceTarget.NVIDIA_GPU)
    assert nvidia.inference.target == "nvidia_gpu"
    amd = recipe_for_target(DeviceTarget.AMD_GPU)
    assert amd.inference.extra_options["gpu_backend"] == "vulkan"


def test_preferred_npu_follows_detected_hardware():
    caps = [
        HardwareCapability(DeviceTarget.QUALCOMM_NPU, True, "Hexagon NPU"),
        HardwareCapability(DeviceTarget.INTEL_NPU, False, "missing"),
    ]
    assert preferred_npu_target(caps) == DeviceTarget.QUALCOMM_NPU
    caps = [
        HardwareCapability(DeviceTarget.QUALCOMM_NPU, False, "missing"),
        HardwareCapability(DeviceTarget.INTEL_NPU, True, "OpenVINO NPU"),
    ]
    assert preferred_npu_target(caps) == DeviceTarget.INTEL_NPU


def test_runtime_options_use_qnn_for_qualcomm_npu(tmp_path):
    config = recommended_config(DeviceTarget.QUALCOMM_NPU)
    options = runtime_options("model.onnx", str(tmp_path), config)
    assert options["execution_provider"] == "QNNExecutionProvider"
    assert options["backend_path"] == "QnnHtp.dll"


def test_llamacpp_gpu_serve_offloads_layers(monkeypatch, tmp_path):
    model = tmp_path / "model-q4.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(
        "finetuner.inference.runner._resolve_tool", lambda name, toolchain_path="", required=False: name
    )
    config = recommended_config(DeviceTarget.NVIDIA_GPU, vllm_available=False)
    command = build_serve_command(str(model), str(tmp_path / "out"), config)
    assert "-ngl" in command
    assert command[command.index("-ngl") + 1] == "99"
    assert "--device" in command
    assert command[command.index("--device") + 1] == "cuda"


def test_compile_is_reserved_for_ahead_of_time_engines():
    config = InferenceOptimizationConfig(engine="vllm", target="nvidia_gpu", compile=True)
    assert any("ahead-of-time" in error for error in config.validate())
    ok = InferenceOptimizationConfig(engine="tensorrt_llm", target="nvidia_gpu", compile=True)
    assert not ok.validate()


def test_vllm_serve_command_is_an_argv_array(monkeypatch):
    monkeypatch.setattr(
        "finetuner.inference.runner._resolve_tool", lambda name, toolchain_path="", required=False: f"/bin/{name}"
    )
    config = InferenceOptimizationConfig(
        engine="vllm",
        target="nvidia_gpu",
        max_context=8192,
        max_batch_size=16,
        kv_cache_dtype="fp8",
        prefix_caching=True,
        cuda_graphs=False,
    )
    command = build_serve_command("model", "out", config)
    assert command[:2] == ["/bin/vllm", "serve"]
    assert command[2].endswith("model")
    assert "--max-model-len" in command
    assert "--enable-prefix-caching" in command
    assert "--enforce-eager" in command
    assert "--kv-cache-dtype" in command
    assert all(isinstance(part, str) for part in command)


def test_llamacpp_serve_command_uses_gguf_and_cache_flags(monkeypatch, tmp_path):
    model = tmp_path / "model-q4.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(
        "finetuner.inference.runner._resolve_tool", lambda name, toolchain_path="", required=False: name
    )
    config = InferenceOptimizationConfig(
        engine="llamacpp",
        target="cpu",
        kv_cache_dtype="q8_0",
        flash_attention=False,
        prefix_caching=True,
    )
    command = build_serve_command(str(model), str(tmp_path / "out"), config)
    assert command[0] == "llama-server"
    assert command[1:3] == ["-m", str(model.resolve())]
    assert "--cache-type-k" in command
    assert "--prompt-cache-all" in command
    assert "-ngl" not in command


def test_tensorrt_compile_command_is_an_argv_array(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "finetuner.inference.runner._resolve_tool", lambda name, toolchain_path="", required=False: f"/opt/{name}"
    )
    config = InferenceOptimizationConfig(
        engine="tensorrt_llm",
        target="nvidia_gpu",
        compile=True,
        max_context=4096,
        max_batch_size=8,
    )
    commands = build_compile_commands("weights", str(tmp_path), config)
    assert commands[0][:2] == ["/opt/trtllm-build", "--checkpoint_dir"]
    assert commands[0][2].endswith("weights")
    assert "--max_seq_len" in commands[0]
    assert commands[0][commands[0].index("--output_dir") + 1] == str((tmp_path / "engine").resolve())


def test_detect_source_format_prefers_quantization_manifest(tmp_path):
    artifact = tmp_path / "quantized"
    artifact.mkdir()
    (artifact / "quantization_manifest.json").write_text(
        json.dumps({"quantization": {"backend": "gguf"}}),
        encoding="utf-8",
    )
    (artifact / "model-q4.gguf").write_bytes(b"gguf")
    assert detect_source_format(str(artifact)).value == "gguf"


def test_optimize_writes_plan_without_compiling(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "hidden_size": 64,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "engine"
    config = InferenceOptimizationConfig(engine="llamacpp", target="cpu", compile=False)
    result = optimize_inference_engine(str(model), str(output), config)
    plan = json.loads((output / "inference_plan.json").read_text(encoding="utf-8"))
    assert result == str(output.resolve())
    assert plan["engine"] == "llamacpp"
    assert plan["compiled"] is False
    assert plan["serve_command"][0] == "llama-server"
    assert plan["source_format"] == "gguf"
    assert "device_bind" in plan


def test_run_on_device_fails_when_accelerator_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "finetuner.inference.runner.detect_inference_hardware",
        lambda: [HardwareCapability(DeviceTarget.NVIDIA_GPU, False, "missing")],
    )
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    config = recommended_config(DeviceTarget.NVIDIA_GPU, vllm_available=False)
    config.extra_options = {**(config.extra_options or {}), "run_on_device": True}
    with pytest.raises(RuntimeError, match="not present"):
        optimize_inference_engine(str(model), str(tmp_path / "out"), config)


def test_vllm_rejects_gguf_artifacts(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    config = InferenceOptimizationConfig(engine="vllm", target="nvidia_gpu")
    with pytest.raises(ValueError, match="does not consume gguf"):
        optimize_inference_engine(str(model), str(tmp_path / "out"), config)


def test_llamacpp_rejects_raw_huggingface_weights(tmp_path):
    model = tmp_path / "hf"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    config = InferenceOptimizationConfig(engine="llamacpp", target="cpu")
    with pytest.raises(ValueError, match="requires a GGUF artifact"):
        optimize_inference_engine(str(model), str(tmp_path / "out"), config)


def test_optimize_action_uses_the_selected_model(monkeypatch, tmp_path):
    from finetuner.core.actions import ActionKind
    from finetuner.core.job import ProjectConfig
    from finetuner.core.runner import ActionContext, execute_action

    seen: list[str] = []

    def fake_optimize(model_path, output_dir, config, log=None):
        seen.append(model_path)
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        return str(destination)

    monkeypatch.setattr("finetuner.inference.runner.optimize_inference_engine", fake_optimize)
    output = execute_action(
        ActionContext(
            "run",
            tmp_path,
            "/quantized",
            "",
            ProjectConfig(),
            ActionKind.OPTIMIZE,
        )
    )
    assert seen == ["/quantized"]
    assert "inference_engine" in output.artifacts


def test_runtime_features_detect_tensor_parallel_and_related_methods():
    one = AcceleratorInventory(1, "cuda", 24.0, ("RTX 4090",))
    many = AcceleratorInventory(4, "cuda", 80.0, ("A100", "A100", "A100", "A100"))

    blocked = runtime_features("vllm", "nvidia_gpu", one)
    assert blocked.tensor_parallel.supported
    assert not blocked.tensor_parallel.available
    assert blocked.tensor_parallel.value == 1
    assert "2+" in blocked.tensor_parallel.reason
    assert blocked.cuda_graphs.available
    assert blocked.gpu_memory_fraction.available
    assert not blocked.flash_attention.supported

    ready = runtime_features("vllm", "nvidia_gpu", many)
    assert ready.tensor_parallel.available
    assert ready.tensor_parallel.value == 4

    cpu = runtime_features("llamacpp", "cpu", many)
    assert not cpu.tensor_parallel.supported
    assert not cpu.cuda_graphs.supported
    assert cpu.prefix_caching.available
    assert not cpu.flash_attention.available


def test_recommended_vllm_uses_detected_gpu_count(monkeypatch):
    monkeypatch.setattr(
        "finetuner.inference.planner.detect_accelerator_inventory",
        lambda: AcceleratorInventory(2, "cuda", 48.0, ("A6000", "A6000")),
    )
    config = recommended_config(DeviceTarget.NVIDIA_GPU, vllm_available=True)
    assert config.tensor_parallel == 2
    assert config.engine == "vllm"


def test_kv_cache_estimate_scales_with_batch_and_context():
    architecture = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_size": 64,
    }
    small = InferenceOptimizationConfig(max_context=256, max_batch_size=1, kv_cache_dtype="fp16")
    large = InferenceOptimizationConfig(max_context=512, max_batch_size=2, kv_cache_dtype="fp16")
    assert estimate_kv_cache_bytes(architecture, large) == 4 * estimate_kv_cache_bytes(architecture, small)
