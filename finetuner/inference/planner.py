from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
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
            inventory = detect_accelerator_inventory()
            tensor_parallel = inventory.gpu_count if inventory.gpu_count >= 2 else 1
            return InferenceOptimizationConfig(
                engine=InferenceEngine.VLLM.value,
                target=target.value,
                source_format=SourceFormat.HF.value,
                max_context=_context_for_memory(memory_gb, default=4096 if tight else 8192),
                max_batch_size=4 if tight else 16,
                kv_cache_dtype=KVCacheDType.FP8.value if tight else KVCacheDType.AUTO.value,
                tensor_parallel=tensor_parallel,
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


@dataclass(frozen=True)
class AcceleratorInventory:
    gpu_count: int
    kind: str
    memory_gb: float | None
    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class MethodSupport:
    supported: bool
    available: bool
    value: int | bool
    reason: str


@dataclass(frozen=True)
class RuntimeFeatures:
    inventory: AcceleratorInventory
    tensor_parallel: MethodSupport
    cuda_graphs: MethodSupport
    flash_attention: MethodSupport
    prefix_caching: MethodSupport
    speculative: MethodSupport
    gpu_memory_fraction: MethodSupport
    compile: MethodSupport

    def summary(self, capabilities: list[HardwareCapability] | None = None) -> str:
        present = []
        if self.inventory.gpu_count:
            label = ", ".join(self.inventory.names[:2]) or self.inventory.kind.upper()
            present.append(f"{self.inventory.gpu_count}x {label}")
        if capabilities:
            for item in capabilities:
                if item.available and item.target not in {
                    DeviceTarget.NVIDIA_GPU,
                    DeviceTarget.AMD_GPU,
                    DeviceTarget.CPU,
                }:
                    present.append(item.target.value.replace("_", " "))
        machine = ", ".join(present) or "CPU only"
        if self.tensor_parallel.supported and not self.tensor_parallel.available:
            return f"This machine: {machine}. {self.tensor_parallel.reason}."
        if self.tensor_parallel.available:
            return (
                f"This machine: {machine}. "
                f"Tensor parallel can use {self.tensor_parallel.value} GPUs."
            )
        return f"This machine: {machine}."


_TP_ENGINES = {
    InferenceEngine.VLLM,
    InferenceEngine.TENSORRT_LLM,
    InferenceEngine.TGI,
}
_CUDA_GRAPH_ENGINES = {InferenceEngine.VLLM}
_FLASH_ENGINES = {InferenceEngine.LLAMACPP, InferenceEngine.TENSORRT_LLM}
_PREFIX_ENGINES = {InferenceEngine.LLAMACPP, InferenceEngine.VLLM, InferenceEngine.TGI}
_SPECULATIVE_ENGINES = {InferenceEngine.VLLM, InferenceEngine.TGI}
_GPU_MEM_ENGINES = {InferenceEngine.VLLM}
_MULTI_GPU_TARGETS = {DeviceTarget.NVIDIA_GPU, DeviceTarget.AMD_GPU}
_GPU_TARGETS = {
    DeviceTarget.NVIDIA_GPU,
    DeviceTarget.AMD_GPU,
    DeviceTarget.INTEL_GPU,
    DeviceTarget.APPLE_GPU,
}


def detect_accelerator_inventory() -> AcceleratorInventory:
    try:
        import torch

        if torch.cuda.is_available():
            count = int(torch.cuda.device_count())
            names = tuple(torch.cuda.get_device_name(index) for index in range(count))
            memory = None
            if count:
                memory = float(torch.cuda.get_device_properties(0).total_memory) / (1024**3)
            kind = "rocm" if getattr(torch.version, "hip", None) else "cuda"
            return AcceleratorInventory(count, kind, memory, names)
    except Exception:
        pass
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=6,
        )
        rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if completed.returncode == 0 and rows:
            names: list[str] = []
            memory = None
            for row in rows:
                name, _, rest = row.partition(",")
                names.append(name.strip())
                if memory is None and rest.strip():
                    try:
                        memory = float(rest.strip()) / 1024
                    except ValueError:
                        memory = None
            return AcceleratorInventory(len(names), "cuda", memory, tuple(names))
    except (OSError, subprocess.SubprocessError):
        pass
    if shutil.which("rocm-smi") or shutil.which("hipinfo"):
        return AcceleratorInventory(1, "rocm", None, ())
    return AcceleratorInventory(0, "", None, ())


def runtime_features(
    engine: InferenceEngine | str,
    target: DeviceTarget | str,
    inventory: AcceleratorInventory | None = None,
) -> RuntimeFeatures:
    try:
        engine_id = InferenceEngine(engine)
    except ValueError:
        engine_id = InferenceEngine.LLAMACPP
    try:
        target_id = DeviceTarget(target)
    except ValueError:
        target_id = DeviceTarget.CPU
    spec = get_engine_spec(engine_id)
    inventory = inventory or detect_accelerator_inventory()
    if target_id == DeviceTarget.AUTO:
        if inventory.kind == "rocm" and inventory.gpu_count:
            target_id = DeviceTarget.AMD_GPU
        elif inventory.gpu_count or inventory.kind == "cuda":
            target_id = DeviceTarget.NVIDIA_GPU
        else:
            target_id = DeviceTarget.CPU
    gpu_count = inventory.gpu_count
    nvidia = target_id == DeviceTarget.NVIDIA_GPU and (gpu_count >= 1 or inventory.kind == "cuda")
    gpu_target = target_id in _GPU_TARGETS

    if engine_id not in _TP_ENGINES:
        tensor_parallel = MethodSupport(False, False, 1, "This engine does not support tensor parallelism")
    elif target_id not in _MULTI_GPU_TARGETS:
        tensor_parallel = MethodSupport(
            True, False, 1, "Tensor parallelism needs a CUDA or ROCm multi-GPU target"
        )
    elif gpu_count < 2:
        tensor_parallel = MethodSupport(
            True, False, 1, f"Tensor parallelism needs 2+ GPUs; detected {gpu_count}"
        )
    else:
        tensor_parallel = MethodSupport(
            True, True, gpu_count, f"Tensor parallelism can use {gpu_count} GPUs"
        )

    if engine_id not in _CUDA_GRAPH_ENGINES:
        cuda_graphs = MethodSupport(False, False, False, "This engine does not use CUDA graphs")
    elif not nvidia:
        cuda_graphs = MethodSupport(True, False, False, "CUDA graphs need a present NVIDIA GPU")
    else:
        cuda_graphs = MethodSupport(True, True, True, "CUDA graphs are available on this NVIDIA GPU")

    if engine_id not in _FLASH_ENGINES:
        flash_attention = MethodSupport(False, False, False, "This engine manages attention internally")
    elif not gpu_target:
        flash_attention = MethodSupport(True, False, False, "Flash attention needs a GPU target")
    else:
        flash_attention = MethodSupport(True, True, True, "Flash attention is available on this GPU")

    if engine_id not in _PREFIX_ENGINES:
        prefix_caching = MethodSupport(False, False, False, "This engine has no prefix cache")
    else:
        prefix_caching = MethodSupport(True, True, True, "Prefix cache is available")

    if engine_id not in _SPECULATIVE_ENGINES:
        speculative = MethodSupport(False, False, 0, "This engine does not support speculative decoding")
    else:
        speculative = MethodSupport(True, True, 0, "Speculative decoding is optional on this engine")

    if engine_id not in _GPU_MEM_ENGINES:
        gpu_memory = MethodSupport(False, False, 0.9, "GPU memory fraction is not used by this engine")
    elif not gpu_target:
        gpu_memory = MethodSupport(True, False, 0.9, "GPU memory fraction needs a GPU target")
    else:
        gpu_memory = MethodSupport(True, True, 0.9, "GPU memory fraction is available")

    compile = MethodSupport(
        spec.compiles_ahead_of_time,
        spec.compiles_ahead_of_time,
        spec.compiles_ahead_of_time,
        "Compile a cached engine artifact" if spec.compiles_ahead_of_time else "This engine has no compiled artifact",
    )
    return RuntimeFeatures(
        inventory,
        tensor_parallel,
        cuda_graphs,
        flash_attention,
        prefix_caching,
        speculative,
        gpu_memory,
        compile,
    )


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
