from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

from finetuner.core.artifacts import atomic_write_json
from finetuner.inference.planner import detect_inference_hardware, estimate_kv_cache_bytes
from finetuner.inference.specs import (
    InferenceEngine,
    InferenceOptimizationConfig,
    SourceFormat,
    get_engine_spec,
)
from finetuner.quantization.specs import DeviceTarget


def detect_source_format(model_path: str) -> SourceFormat:
    path = Path(model_path)
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".gguf":
            return SourceFormat.GGUF
        if suffix == ".onnx":
            return SourceFormat.ONNX
        if suffix == ".xml":
            return SourceFormat.OPENVINO_IR
        return SourceFormat.HF

    manifest = path / "quantization_manifest.json"
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            backend = str(payload.get("quantization", {}).get("backend", ""))
        except (OSError, json.JSONDecodeError, TypeError):
            backend = ""
        mapping = {
            "gguf": SourceFormat.GGUF,
            "openvino": SourceFormat.OPENVINO_IR,
            "onnx": SourceFormat.ONNX,
            "awq": SourceFormat.AWQ,
        }
        if backend in mapping:
            return mapping[backend]

    if any(path.glob("*.gguf")):
        return SourceFormat.GGUF
    if (path / "openvino_model.xml").exists() or any(path.glob("*.xml")):
        return SourceFormat.OPENVINO_IR
    if any(path.glob("*.onnx")):
        return SourceFormat.ONNX
    if (path / "quant_config.json").exists() or (path / "quantization_config.json").exists():
        return SourceFormat.AWQ
    return SourceFormat.HF


def _read_architecture(model_path: str) -> dict[str, Any] | None:
    path = Path(model_path)
    candidates = [path / "config.json"] if path.is_dir() else [path.with_name("config.json")]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _resolve_tool(name: str, toolchain_path: str = "", required: bool = False) -> str:
    names = [name]
    if not name.endswith(".exe"):
        names.append(f"{name}.exe")
    root = Path(toolchain_path).expanduser().resolve() if toolchain_path else None
    if root:
        for candidate in names:
            located = root / candidate
            if located.exists():
                return str(located)
    for candidate in names:
        found = shutil.which(candidate)
        if found:
            return found
    if required:
        raise RuntimeError(f"Required executable {name!r} was not found in toolchain_path or PATH")
    return name


def _model_file(model_path: str, pattern: str) -> str:
    path = Path(model_path)
    if path.is_file():
        return str(path.resolve())
    matches = sorted(path.glob(pattern))
    if matches:
        return str(matches[0].resolve())
    return str(path.resolve())


def build_serve_command(
    model_path: str, output_dir: str, config: InferenceOptimizationConfig
) -> list[str]:
    """Build a serve argv array (never a shell string) for a validated engine plan."""
    config.require_valid()
    engine = InferenceEngine(config.engine)
    spec = get_engine_spec(engine)
    if not spec.has_serve_cli:
        return []
    model = str(Path(model_path).resolve())
    target = DeviceTarget(config.target)

    if engine == InferenceEngine.LLAMACPP:
        server = _resolve_tool("llama-server", config.toolchain_path)
        command = [
            server,
            "-m",
            _model_file(model, "*.gguf"),
            "-c",
            str(config.max_context),
            "-b",
            str(max(config.max_batch_size, 1) * 64),
            "--parallel",
            str(config.max_batch_size),
        ]
        extras = config.extra_options or {}
        gpu_layers = extras.get("gpu_layers")
        if target != DeviceTarget.CPU:
            command.extend(["-ngl", str(int(gpu_layers) if gpu_layers is not None else 99)])
        if extras.get("gpu_backend"):
            command.extend(["--device", str(extras["gpu_backend"])])
        if config.flash_attention:
            command.extend(["--flash-attn", "on"])
        if config.kv_cache_dtype not in {"auto", "fp16"}:
            command.extend(["--cache-type-k", config.kv_cache_dtype])
            command.extend(["--cache-type-v", config.kv_cache_dtype])
        if config.prefix_caching:
            command.append("--prompt-cache-all")
        command.extend(["--host", "127.0.0.1", "--port", str(config.serve_port)])
        return command

    if engine == InferenceEngine.VLLM:
        command = [
            _resolve_tool("vllm", config.toolchain_path),
            "serve",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(config.serve_port),
            "--max-model-len",
            str(config.max_context),
            "--max-num-seqs",
            str(config.max_batch_size),
            "--tensor-parallel-size",
            str(config.tensor_parallel),
            "--gpu-memory-utilization",
            str(config.gpu_memory_utilization),
        ]
        if config.kv_cache_dtype != "auto":
            command.extend(["--kv-cache-dtype", config.kv_cache_dtype])
        if config.prefix_caching:
            command.append("--enable-prefix-caching")
        if not config.cuda_graphs:
            command.append("--enforce-eager")
        if config.speculative_tokens:
            command.extend(["--num-speculative-tokens", str(config.speculative_tokens)])
        return command

    if engine == InferenceEngine.TENSORRT_LLM:
        return [
            _resolve_tool("trtllm-serve", config.toolchain_path),
            str(Path(output_dir).resolve() / "engine"),
            "--host",
            "127.0.0.1",
            "--port",
            str(config.serve_port),
            "--max_batch_size",
            str(config.max_batch_size),
        ]

    if engine == InferenceEngine.TGI:
        command = [
            _resolve_tool("text-generation-launcher", config.toolchain_path),
            "--model-id",
            model,
            "--hostname",
            "127.0.0.1",
            "--port",
            str(config.serve_port),
            "--max-total-tokens",
            str(config.max_context),
            "--max-concurrent-requests",
            str(config.max_batch_size),
        ]
        if config.tensor_parallel > 1:
            command.extend(["--num-shard", str(config.tensor_parallel)])
        if config.prefix_caching:
            command.append("--enable-prefix-caching")
        if config.speculative_tokens:
            command.extend(["--speculate", str(config.speculative_tokens)])
        return command

    raise AssertionError(engine)


def build_compile_commands(
    model_path: str, output_dir: str, config: InferenceOptimizationConfig
) -> list[list[str]]:
    """Build compile argv arrays. Empty when compilation is in-process or not required."""
    config.require_valid()
    engine = InferenceEngine(config.engine)
    model = str(Path(model_path).resolve())
    output = Path(output_dir).resolve()

    if engine == InferenceEngine.TENSORRT_LLM:
        builder = _resolve_tool("trtllm-build", config.toolchain_path, required=config.compile)
        command = [
            builder,
            "--checkpoint_dir",
            model,
            "--output_dir",
            str(output / "engine"),
            "--max_batch_size",
            str(config.max_batch_size),
            "--max_seq_len",
            str(config.max_context),
        ]
        if config.flash_attention:
            command.extend(["--gpt_attention_plugin", "auto"])
        if config.kv_cache_dtype == "int8":
            command.extend(["--paged_kv_cache", "enable"])
        if config.kv_cache_dtype == "fp8":
            command.extend(["--use_fp8_context_fmha", "enable"])
        return [command]
    return []


def runtime_options(
    model_path: str, output_dir: str, config: InferenceOptimizationConfig
) -> dict[str, Any]:
    engine = InferenceEngine(config.engine)
    target = DeviceTarget(config.target)
    output = str(Path(output_dir).resolve())
    if engine == InferenceEngine.OPENVINO:
        device = {
            DeviceTarget.CPU: "CPU",
            DeviceTarget.INTEL_GPU: "GPU",
            DeviceTarget.INTEL_NPU: "NPU",
        }[target]
        return {
            "device": device,
            "CACHE_DIR": str(Path(output) / "ov_cache"),
            "PERFORMANCE_HINT": "THROUGHPUT" if config.max_batch_size > 1 else "LATENCY",
            "INFERENCE_NUM_THREADS": 0,
            "KV_CACHE_PRECISION": config.kv_cache_dtype,
        }
    if engine == InferenceEngine.ONNXRUNTIME:
        extras = config.extra_options or {}
        provider = {
            DeviceTarget.CPU: "CPUExecutionProvider",
            DeviceTarget.NVIDIA_GPU: "CUDAExecutionProvider",
            DeviceTarget.AMD_GPU: extras.get("execution_provider", "DmlExecutionProvider"),
            DeviceTarget.QUALCOMM_NPU: "QNNExecutionProvider",
        }.get(target, extras.get("execution_provider", "CPUExecutionProvider"))
        provider = str(extras.get("execution_provider") or provider)
        options = {
            "execution_provider": provider,
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "enable_mem_pattern": True,
            "enable_cpu_mem_arena": provider == "CPUExecutionProvider",
            "session_cache": str(Path(output) / "ort_cache"),
        }
        if provider == "QNNExecutionProvider":
            options.update(
                {
                    "backend_path": extras.get("backend_path", "QnnHtp.dll"),
                    "qnn_backend": extras.get("qnn_backend", "htp"),
                    "htp_performance_mode": extras.get("htp_performance_mode", "burst"),
                    "htp_graph_finalization_optimization_mode": extras.get(
                        "htp_graph_finalization_optimization_mode", "3"
                    ),
                }
            )
        return options
    return {
        "engine": engine.value,
        "target": target.value,
        "max_context": config.max_context,
        "max_batch_size": config.max_batch_size,
    }


def _compile_openvino(
    model_path: str,
    output: Path,
    config: InferenceOptimizationConfig,
    log: Callable[[str], None] | None,
) -> None:
    try:
        import openvino as ov
    except ImportError as exc:
        raise RuntimeError("OpenVINO compilation requires the optional 'openvino' package") from exc
    options = runtime_options(model_path, str(output), config)
    cache_dir = Path(options["CACHE_DIR"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    xml = Path(_model_file(model_path, "*.xml"))
    if log:
        log(f"Compiling OpenVINO model for {options['device']}...")
    core = ov.Core()
    core.set_property({"CACHE_DIR": str(cache_dir)})
    model = core.read_model(str(xml))
    compiled = core.compile_model(model, options["device"])
    atomic_write_json(
        output / "compiled_engine.json",
        {
            "schema_version": 1,
            "runtime": "openvino",
            "device": options["device"],
            "inputs": [item.get_any_name() for item in compiled.inputs],
            "cache_dir": str(cache_dir),
        },
    )


def _compile_onnxruntime(
    model_path: str,
    output: Path,
    config: InferenceOptimizationConfig,
    log: Callable[[str], None] | None,
) -> None:
    try:
        import onnxruntime
    except ImportError as exc:
        raise RuntimeError("ONNX Runtime optimization requires the optional 'onnxruntime' package") from exc
    options = runtime_options(model_path, str(output), config)
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_mem_pattern = True
    optimized = output / "optimized.onnx"
    session_options.optimized_model_filepath = str(optimized)
    provider = str(options["execution_provider"])
    available = list(onnxruntime.get_available_providers())
    if provider not in available:
        raise RuntimeError(
            f"{provider} is not in this ONNX Runtime build "
            f"(available: {', '.join(available) or 'none'})"
        )
    provider_options = _onnx_provider_options(options)
    if log:
        log(f"Applying ONNX Runtime graph optimizations on {provider}...")
    onnxruntime.InferenceSession(
        _model_file(model_path, "*.onnx"),
        sess_options=session_options,
        providers=[(provider, provider_options)] if provider_options else [provider],
    )
    atomic_write_json(
        output / "compiled_engine.json",
        {
            "schema_version": 1,
            "runtime": "onnxruntime",
            "execution_provider": options["execution_provider"],
            "optimized_model": str(optimized),
        },
    )


def _onnx_provider_options(options: dict[str, Any]) -> dict[str, str]:
    if options.get("execution_provider") != "QNNExecutionProvider":
        return {}
    return {
        "backend_path": str(options.get("backend_path", "QnnHtp.dll")),
        "htp_performance_mode": str(options.get("htp_performance_mode", "burst")),
        "htp_graph_finalization_optimization_mode": str(
            options.get("htp_graph_finalization_optimization_mode", "3")
        ),
    }


def probe_device_bind(config: InferenceOptimizationConfig) -> dict[str, Any]:
    """Record whether the selected accelerator is present and bindable."""
    target = DeviceTarget(config.target)
    extras = config.extra_options or {}
    capabilities = {item.target: item for item in detect_inference_hardware()}
    capability = capabilities.get(target)
    available = bool(capability and capability.available)
    detail = capability.detail if capability else f"{target.value} not detected"
    providers: list[str] = []
    try:
        import onnxruntime

        providers = list(onnxruntime.get_available_providers())
    except Exception:
        providers = []
    required_provider = extras.get("execution_provider")
    provider_ready = required_provider is None or required_provider in providers
    bound = available and (
        InferenceEngine(config.engine) != InferenceEngine.ONNXRUNTIME or provider_ready
    )
    error = ""
    if not available:
        error = f"{target.value.replace('_', ' ')} is not present on this machine ({detail})"
    elif required_provider and not provider_ready:
        error = (
            f"{target.value.replace('_', ' ')} is present ({detail}) but {required_provider} "
            f"is not in this ONNX Runtime build. Available providers: "
            f"{', '.join(providers) or 'none'}."
        )
    return {
        "target": target.value,
        "available": available,
        "bound": bound,
        "detail": detail,
        "execution_provider": required_provider,
        "available_providers": providers,
        "error": error,
    }


def optimize_inference_engine(
    model_path: str,
    output_dir: str,
    config: InferenceOptimizationConfig,
    log: Callable[[str], None] | None = None,
) -> str:
    config.require_valid()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = (
        SourceFormat(config.source_format)
        if config.source_format != SourceFormat.AUTO.value
        else detect_source_format(model_path)
    )
    spec = get_engine_spec(config.engine)
    if source not in spec.source_formats:
        raise ValueError(
            f"{spec.name} does not consume {source.value} artifacts; "
            f"use {', '.join(item.value for item in spec.source_formats)}"
        )
    if spec.engine == InferenceEngine.LLAMACPP and source == SourceFormat.HF:
        raise ValueError(
            "llama.cpp serving requires a GGUF artifact; run a GGUF quantization stage first"
        )
    if spec.engine == InferenceEngine.OPENVINO and source == SourceFormat.HF:
        raise ValueError(
            "OpenVINO serving requires an OpenVINO IR artifact; run an OpenVINO quantization stage first"
        )

    serve = build_serve_command(model_path, str(output), config)
    compile_commands = build_compile_commands(model_path, str(output), config)
    architecture = _read_architecture(model_path)
    kv_bytes = estimate_kv_cache_bytes(architecture, config)
    compiled = False

    if config.compile:
        engine = InferenceEngine(config.engine)
        if engine == InferenceEngine.OPENVINO:
            _compile_openvino(model_path, output, config, log)
            compiled = True
        elif engine == InferenceEngine.ONNXRUNTIME:
            _compile_onnxruntime(model_path, output, config, log)
            compiled = True
        elif compile_commands:
            import subprocess

            for command in compile_commands:
                if log:
                    log(f"Running inference compile step: {Path(command[0]).name}")
                subprocess.run(command, check=True, text=True)
            compiled = True

    plan = {
        "schema_version": 1,
        "source_model": str(Path(model_path).resolve()),
        "source_format": source.value,
        "output_dir": str(output.resolve()),
        "engine": spec.engine.value,
        "engine_name": spec.name,
        "target": config.target,
        "optimization": config.to_dict(),
        "runtime_options": runtime_options(model_path, str(output), config),
        "serve_command": [Path(serve[0]).name, *serve[1:]] if serve else [],
        "serve_url": f"http://127.0.0.1:{config.serve_port}",
        "serve_port": config.serve_port,
        "compile_commands": [
            [Path(command[0]).name, *command[1:]] for command in compile_commands
        ],
        "compiled": compiled,
        "device_bind": {},
        "recommended": {
            "kv_cache_bytes_estimate": kv_bytes,
            "notes": [
                spec.description,
                *(
                    [f"Estimated KV-cache working set: {kv_bytes / (1024 ** 3):.2f} GiB"]
                    if kv_bytes
                    else []
                ),
            ],
        },
    }
    bind = probe_device_bind(config)
    plan["device_bind"] = bind
    if bind["detail"]:
        plan["recommended"]["notes"].append(f"Device: {bind['detail']}")
    destination = output / "inference_plan.json"
    atomic_write_json(destination, plan)
    if log:
        log(f"Inference engine plan saved to {destination}")
        if bind["error"]:
            log(bind["error"])
    extras = config.extra_options or {}
    if extras.get("run_on_device") and not bind["bound"]:
        raise RuntimeError(bind["error"] or f"Could not bind {config.target}")
    return str(output.resolve())
