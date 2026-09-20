from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from finetuner.inference.specs import (
    InferenceEngine,
    InferenceOptimizationConfig,
    KVCacheDType,
    SourceFormat,
    engine_specs,
    get_engine_spec,
    qnn_runtime_options,
)
from finetuner.quantization.planner import HardwareCapability, detect_hardware
from finetuner.quantization.specs import DeviceTarget


@dataclass(frozen=True)
class InferenceRecommendation:
    config: InferenceOptimizationConfig
    rationale: str
    notes: tuple[str, ...] = ()


_BACKEND_ENGINES = {
    "gguf": (InferenceEngine.LLAMACPP,),
    "openvino": (InferenceEngine.OPENVINO,),
    "onnx": (InferenceEngine.ONNXRUNTIME,),
    "awq": (InferenceEngine.VLLM, InferenceEngine.TENSORRT_LLM, InferenceEngine.TGI),
}


def compatible_engines(target: DeviceTarget) -> list[InferenceEngine]:
    return [spec.engine for spec in engine_specs() if target in spec.targets]


def compatible_engines_for_backend(backend: str) -> tuple[InferenceEngine, ...]:
    return _BACKEND_ENGINES.get(backend, ())


def backend_engine_compatibility_error(backend: str, engine: str) -> str | None:
    allowed = compatible_engines_for_backend(backend)
    if not allowed:
        return None
    try:
        selected = InferenceEngine(engine)
    except ValueError:
        return None
    if selected in allowed:
        return None
    names = ", ".join(item.value for item in allowed)
    return (
        f"{engine} does not consume {backend} artifacts; "
        f"use {names} or change the quantization backend"
    )


def recommended_config(
    target: DeviceTarget,
    memory_gb: float | None = None,
    *,
    vllm_available: bool | None = None,
) -> InferenceOptimizationConfig:
    if target == DeviceTarget.AUTO:
        from finetuner.inference.devices import select_best_runtime

        return select_best_runtime(
            memory_gb=memory_gb,
            vllm_available=vllm_available,
        ).recipe.inference
    if target == DeviceTarget.QUALCOMM_NPU:
        return InferenceOptimizationConfig(
            engine=InferenceEngine.ONNXRUNTIME.value,
            target=target.value,
            source_format=SourceFormat.ONNX.value,
            max_context=_context_for_memory(memory_gb, default=2048),
            max_batch_size=1,
            compile=False,
            prefix_caching=False,
            cuda_graphs=False,
            flash_attention=False,
            extra_options=qnn_runtime_options(),
        )
    if target == DeviceTarget.INTEL_NPU:
        return InferenceOptimizationConfig(
            engine=InferenceEngine.OPENVINO.value,
            target=target.value,
            source_format=SourceFormat.OPENVINO_IR.value,
            max_context=_context_for_memory(memory_gb, default=2048),
            max_batch_size=1,
            kv_cache_dtype=KVCacheDType.INT8.value,
            compile=True,
            prefix_caching=False,
            cuda_graphs=False,
            flash_attention=False,
        )
    if target == DeviceTarget.INTEL_GPU:
        return InferenceOptimizationConfig(
            engine=InferenceEngine.OPENVINO.value,
            target=target.value,
            source_format=SourceFormat.OPENVINO_IR.value,
            max_context=_context_for_memory(memory_gb, default=4096),
            max_batch_size=4,
            compile=True,
            cuda_graphs=False,
        )
    if target == DeviceTarget.NVIDIA_GPU:
        tight = memory_gb is not None and memory_gb < 16
        use_vllm = _vllm_available() if vllm_available is None else vllm_available
        if use_vllm:
            return InferenceOptimizationConfig(
                engine=InferenceEngine.VLLM.value,
                target=target.value,
                source_format=SourceFormat.HF.value,
                max_context=_context_for_memory(memory_gb, default=4096 if tight else 8192),
                max_batch_size=4 if tight else 16,
                kv_cache_dtype=KVCacheDType.FP8.value if tight else KVCacheDType.AUTO.value,
                gpu_memory_utilization=0.85 if tight else 0.90,
                extra_options={"gpu_backend": "cuda"},
            )
        return InferenceOptimizationConfig(
            engine=InferenceEngine.LLAMACPP.value,
            target=target.value,
            source_format=SourceFormat.GGUF.value,
            max_context=_context_for_memory(memory_gb, default=4096 if tight else 8192),
            max_batch_size=4 if tight else 8,
            kv_cache_dtype=KVCacheDType.FP8.value if tight else KVCacheDType.AUTO.value,
            cuda_graphs=False,
            flash_attention=True,
            extra_options={"gpu_backend": "cuda", "gpu_layers": 99},
        )
    if target in {
        DeviceTarget.CPU,
        DeviceTarget.AMD_GPU,
        DeviceTarget.APPLE_GPU,
    }:
        gpu_backend = {
            DeviceTarget.AMD_GPU: "vulkan",
            DeviceTarget.APPLE_GPU: "metal",
        }.get(target)
        extras = {"gpu_backend": gpu_backend, "gpu_layers": 99} if gpu_backend else {}
        return InferenceOptimizationConfig(
            engine=InferenceEngine.LLAMACPP.value,
            target=target.value,
            source_format=SourceFormat.GGUF.value,
            max_context=_context_for_memory(memory_gb, default=4096),
            max_batch_size=4 if target == DeviceTarget.CPU else 8,
            kv_cache_dtype=(
                KVCacheDType.Q8_0.value if target == DeviceTarget.CPU else KVCacheDType.AUTO.value
            ),
            cuda_graphs=False,
            flash_attention=target != DeviceTarget.CPU,
            extra_options=extras or None,
        )
    raise ValueError(f"No inference-engine recommendation for {target.value}")


def recommend_for_target(
    target: DeviceTarget, memory_gb: float | None = None
) -> InferenceRecommendation:
    config = recommended_config(target, memory_gb)
    spec = get_engine_spec(config.engine)
    notes = [
        spec.description,
        f"Native source formats: {', '.join(item.value for item in spec.source_formats)}",
    ]
    if config.compile:
        notes.append("This runtime materializes a compiled/cached engine when compile is enabled.")
    if memory_gb is not None:
        notes.append(f"Sized for approximately {memory_gb:.1f} GB device memory.")
    return InferenceRecommendation(config, f"{spec.name} on {target.value.replace('_', ' ')}", tuple(notes))


def detect_inference_hardware() -> list[HardwareCapability]:
    return detect_hardware()


def estimate_kv_cache_bytes(
    architecture: dict[str, Any] | None, config: InferenceOptimizationConfig
) -> int | None:
    if not architecture:
        return None
    layers = architecture.get("num_hidden_layers")
    kv_heads = architecture.get("num_key_value_heads") or architecture.get("num_attention_heads")
    hidden = architecture.get("hidden_size")
    heads = architecture.get("num_attention_heads")
    if not all(isinstance(value, int) and value > 0 for value in (layers, kv_heads, hidden, heads)):
        return None
    head_dim = hidden // heads
    bytes_per = {
        KVCacheDType.AUTO.value: 2.0,
        KVCacheDType.FP16.value: 2.0,
        KVCacheDType.FP8.value: 1.0,
        KVCacheDType.INT8.value: 1.0,
        KVCacheDType.Q8_0.value: 1.0,
        KVCacheDType.Q4_0.value: 0.5,
    }[config.kv_cache_dtype]
    # Key + value tensors for every layer, sequence, and concurrent request.
    return int(
        2
        * layers
        * kv_heads
        * head_dim
        * config.max_context
        * config.max_batch_size
        * bytes_per
    )


def _vllm_available() -> bool:
    import importlib.util
    import shutil

    return shutil.which("vllm") is not None or importlib.util.find_spec("vllm") is not None


def _context_for_memory(memory_gb: float | None, default: int) -> int:
    if memory_gb is None:
        return default
    if memory_gb < 8:
        return 2048
    if memory_gb < 16:
        return 4096
    if memory_gb < 24:
        return 8192
    return 16384
