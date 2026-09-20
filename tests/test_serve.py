from __future__ import annotations

import json
from urllib.request import urlopen

from finetuner.core.job import ProjectConfig
from finetuner.inference.devices import describe_device_offer, select_best_runtime
from finetuner.inference.http_server import CallbackBackend, start_builtin_server
from finetuner.inference.runner import build_serve_command
from finetuner.inference.serve import (
    DEFAULT_SERVE_PORT,
    portable_serve_config,
    prepare_serve_model,
    resolve_launch_command,
    serve_url,
    stop_server,
)
from finetuner.inference.specs import InferenceOptimizationConfig
from finetuner.quantization.planner import HardwareCapability
from finetuner.quantization.specs import DeviceTarget


def test_default_serve_port_is_1234():
    assert DEFAULT_SERVE_PORT == 1234
    assert InferenceOptimizationConfig().serve_port == 1234
    assert serve_url() == "http://127.0.0.1:1234"


def test_serve_commands_bind_port_1234(monkeypatch, tmp_path):
    model = tmp_path / "model-q4.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(
        "finetuner.inference.runner._resolve_tool",
        lambda name, toolchain_path="", required=False: name,
    )
    llamacpp = build_serve_command(
        str(model),
        str(tmp_path / "out"),
        InferenceOptimizationConfig(engine="llamacpp", target="cpu"),
    )
    assert llamacpp[llamacpp.index("--port") + 1] == "1234"
    vllm = build_serve_command(
        "model",
        "out",
        InferenceOptimizationConfig(engine="vllm", target="nvidia_gpu"),
    )
    assert vllm[vllm.index("--port") + 1] == "1234"


def test_device_offer_names_detected_hardware_and_best_engine():
    caps = [
        HardwareCapability(DeviceTarget.CPU, True, "Oryon"),
        HardwareCapability(DeviceTarget.QUALCOMM_NPU, True, "Hexagon NPU"),
        HardwareCapability(DeviceTarget.NVIDIA_GPU, False, "missing"),
    ]
    offer = describe_device_offer(capabilities=caps)
    assert offer.title == "Optimize for your device?"
    assert "Hexagon" in offer.message
    assert "1234" in offer.message
    assert offer.choice.recipe.target == DeviceTarget.CPU
    assert "port 1234" in offer.message


def test_unoptimized_prepare_keeps_portable_cpu_recipe(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    project = ProjectConfig()
    project.inference.serve_port = 1234
    served, config, _plan = prepare_serve_model(
        str(model), str(tmp_path / "serve"), project, optimize=False
    )
    assert served == str(model)
    assert config.engine == "llamacpp"
    assert config.target == "cpu"
    assert config.extra_options["optimized"] is False
    assert config.serve_port == 1234


def test_optimized_prepare_skips_hexagon_without_htp(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    caps = [
        HardwareCapability(DeviceTarget.CPU, True, "Oryon"),
        HardwareCapability(DeviceTarget.QUALCOMM_NPU, True, "Hexagon NPU"),
    ]
    monkeypatch.setattr(
        "finetuner.inference.devices.detect_inference_hardware", lambda: caps
    )
    monkeypatch.setattr(
        "finetuner.inference.serve.optimize_inference_engine",
        lambda model_path, output_dir, config, log=None: output_dir,
    )
    project = ProjectConfig()
    _served, config, _plan = prepare_serve_model(
        str(model), str(tmp_path / "serve"), project, optimize=True
    )
    assert config.engine == "llamacpp"
    assert config.target == "cpu"
    assert config.extra_options["optimized"] is True


def test_resolve_launch_uses_discovered_llama_server(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    server = tmp_path / "llama-server.exe"
    server.write_bytes(b"exe")
    monkeypatch.setattr("finetuner.inference.serve.find_llama_server", lambda toolchain="": str(server))
    command = resolve_launch_command(
        str(model),
        str(tmp_path),
        InferenceOptimizationConfig(engine="llamacpp", target="cpu"),
    )
    assert command[0] == str(server)
    assert "--port" in command
    assert command[command.index("--port") + 1] == "1234"


def test_builtin_server_answers_health_and_chat(tmp_path):
    backend = CallbackBackend(lambda prompt, max_tokens: f"echo:{prompt[:8]}", "tiny")
    httpd = start_builtin_server("127.0.0.1", 0, backend)
    host, port = httpd.server_address[:2]
    from threading import Thread

    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        health = json.loads(urlopen(f"http://{host}:{port}/health", timeout=2).read())
        assert health["status"] == "ok"
        request = urlopen(
            f"http://{host}:{port}/v1/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}).encode(),
            timeout=2,
        )
        payload = json.loads(request.read())
        assert payload["choices"][0]["message"]["content"].startswith("echo:")
    finally:
        httpd.shutdown()
        httpd.server_close()
        stop_server()


def test_portable_config_maps_onnx_to_ort(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx")
    config = portable_serve_config(str(model))
    assert config.engine == "onnxruntime"
    assert config.serve_port == 1234


def test_select_best_runtime_still_used_by_offer():
    caps = [HardwareCapability(DeviceTarget.CPU, True, "cpu")]
    choice = select_best_runtime(caps)
    offer = describe_device_offer(capabilities=caps)
    assert offer.choice.recipe.inference.engine == choice.recipe.inference.engine
