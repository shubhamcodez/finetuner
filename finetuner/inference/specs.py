"""Inference-engine optimization: compile and tune a model for a concrete runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from finetuner.quantization.specs import DeviceTarget


class InferenceEngine(str, Enum):
    LLAMACPP = "llamacpp"
    VLLM = "vllm"
    TENSORRT_LLM = "tensorrt_llm"
    OPENVINO = "openvino"
    ONNXRUNTIME = "onnxruntime"
    TGI = "tgi"


class SourceFormat(str, Enum):
    AUTO = "auto"
    HF = "hf"
    GGUF = "gguf"
    OPENVINO_IR = "openvino_ir"
    ONNX = "onnx"
    AWQ = "awq"


class KVCacheDType(str, Enum):
    AUTO = "auto"
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"


@dataclass(frozen=True)
class EngineSpec:
    engine: InferenceEngine
    name: str
    targets: tuple[DeviceTarget, ...]
    source_formats: tuple[SourceFormat, ...]
    kv_cache_dtypes: tuple[KVCacheDType, ...]
    package_hint: str
    description: str
    compiles_ahead_of_time: bool = False
    has_serve_cli: bool = True


_ENGINES: dict[InferenceEngine, EngineSpec] = {
    InferenceEngine.LLAMACPP: EngineSpec(
        InferenceEngine.LLAMACPP,
        "llama.cpp",
        (
            DeviceTarget.CPU,
            DeviceTarget.NVIDIA_GPU,
            DeviceTarget.AMD_GPU,
            DeviceTarget.INTEL_GPU,
            DeviceTarget.APPLE_GPU,
        ),
        (SourceFormat.GGUF, SourceFormat.HF),
        (KVCacheDType.AUTO, KVCacheDType.FP16, KVCacheDType.Q8_0, KVCacheDType.Q4_0),
        "llama.cpp",
        "Broad CPU/GPU serving through llama-server. GGUF is the native artifact; "
        "Hugging Face weights still need a GGUF conversion stage.",
    ),
    InferenceEngine.VLLM: EngineSpec(
        InferenceEngine.VLLM,
        "vLLM",
        (DeviceTarget.NVIDIA_GPU,),
        (SourceFormat.HF, SourceFormat.AWQ),
        (KVCacheDType.AUTO, KVCacheDType.FP16, KVCacheDType.FP8),
        "pip install vllm",
        "Paged-attention serving with continuous batching, prefix caching, and CUDA graphs.",
    ),
    InferenceEngine.TENSORRT_LLM: EngineSpec(
        InferenceEngine.TENSORRT_LLM,
        "TensorRT-LLM",
        (DeviceTarget.NVIDIA_GPU,),
        (SourceFormat.HF, SourceFormat.AWQ),
        (KVCacheDType.AUTO, KVCacheDType.FP16, KVCacheDType.FP8, KVCacheDType.INT8),
        "TensorRT-LLM (trtllm-build)",
        "Ahead-of-time NVIDIA engine build for high-throughput CUDA serving.",
        compiles_ahead_of_time=True,
    ),
    InferenceEngine.OPENVINO: EngineSpec(
        InferenceEngine.OPENVINO,
        "OpenVINO",
        (DeviceTarget.CPU, DeviceTarget.INTEL_GPU, DeviceTarget.INTEL_NPU),
        (SourceFormat.OPENVINO_IR, SourceFormat.HF),
        (KVCacheDType.AUTO, KVCacheDType.FP16, KVCacheDType.INT8),
        "pip install openvino",
        "Compiled/cached OpenVINO runtime for Intel CPU, GPU, and supported NPUs.",
        compiles_ahead_of_time=True,
        has_serve_cli=False,
    ),
    InferenceEngine.ONNXRUNTIME: EngineSpec(
        InferenceEngine.ONNXRUNTIME,
        "ONNX Runtime",
        (
            DeviceTarget.CPU,
            DeviceTarget.NVIDIA_GPU,
            DeviceTarget.AMD_GPU,
            DeviceTarget.QUALCOMM_NPU,
        ),
        (SourceFormat.ONNX,),
        (KVCacheDType.AUTO,),
        "pip install onnxruntime",
        "Session-level graph optimization. CPU uses the default EP; NVIDIA needs CUDA, "
        "AMD needs DirectML/ROCm, and Qualcomm NPU needs the QNN/HTP provider recipe.",
        compiles_ahead_of_time=True,
        has_serve_cli=False,
    ),
    InferenceEngine.TGI: EngineSpec(
        InferenceEngine.TGI,
        "Text Generation Inference",
        (DeviceTarget.NVIDIA_GPU, DeviceTarget.CPU),
        (SourceFormat.HF, SourceFormat.AWQ),
        (KVCacheDType.AUTO, KVCacheDType.FP16, KVCacheDType.FP8),
        "text-generation-launcher",
        "Hugging Face TGI serving with paged attention and optional prefix caching.",
    ),
}


def engine_specs() -> list[EngineSpec]:
    return list(_ENGINES.values())


def get_engine_spec(engine: InferenceEngine | str) -> EngineSpec:
    return _ENGINES[InferenceEngine(engine)]


@dataclass
class InferenceOptimizationConfig:
    engine: str = InferenceEngine.LLAMACPP.value
    target: str = DeviceTarget.CPU.value
    source_format: str = SourceFormat.AUTO.value
    max_context: int = 4096
    max_batch_size: int = 8
    kv_cache_dtype: str = KVCacheDType.AUTO.value
    tensor_parallel: int = 1
    gpu_memory_utilization: float = 0.90
    prefix_caching: bool = True
    cuda_graphs: bool = True
    flash_attention: bool = True
    speculative_tokens: int = 0
    compile: bool = False
    toolchain_path: str = ""
    serve_port: int = 1234
    extra_options: dict[str, Any] | None = None

    def validate(self) -> list[str]:
        errors: list[str] = []
        try:
            engine = InferenceEngine(self.engine)
            spec = get_engine_spec(engine)
        except ValueError:
            return [f"Unknown inference engine: {self.engine}"]
        try:
            target = DeviceTarget(self.target)
        except ValueError:
            return [f"Unknown deployment target: {self.target}"]
        extras = self.extra_options or {}
        if target == DeviceTarget.AUTO:
            # Resolved to a concrete specialist in run_optimize / Run best.
            pass
        elif target not in spec.targets:
            errors.append(f"{spec.name} does not support target {target.value}")
            errors.extend(_provider_recipe_errors(engine, target, extras))
        else:
            errors.extend(_provider_recipe_errors(engine, target, extras))
        try:
            source = SourceFormat(self.source_format)
        except ValueError:
            return [f"Unknown source format: {self.source_format}"]
        if source != SourceFormat.AUTO and source not in spec.source_formats:
            errors.append(f"{spec.name} does not consume {source.value} artifacts")
        try:
            kv_dtype = KVCacheDType(self.kv_cache_dtype)
        except ValueError:
            return [f"Unknown KV-cache dtype: {self.kv_cache_dtype}"]
        if kv_dtype not in spec.kv_cache_dtypes:
            errors.append(
                f"{spec.name} supports KV-cache dtypes "
                f"{', '.join(item.value for item in spec.kv_cache_dtypes)}, not {kv_dtype.value}"
            )
        if not 256 <= self.max_context <= 1_048_576:
            errors.append("max_context must be between 256 and 1,048,576")
        if not 1 <= self.max_batch_size <= 4096:
            errors.append("max_batch_size must be between 1 and 4,096")
        if not 1 <= self.tensor_parallel <= 256:
            errors.append("tensor_parallel must be between 1 and 256")
        if not 0.1 <= self.gpu_memory_utilization <= 1.0:
            errors.append("gpu_memory_utilization must be between 0.1 and 1.0")
        if not 0 <= self.speculative_tokens <= 32:
            errors.append("speculative_tokens must be between 0 and 32")
        if self.tensor_parallel > 1 and target not in {DeviceTarget.NVIDIA_GPU, DeviceTarget.AMD_GPU}:
            errors.append("tensor_parallel > 1 requires a multi-GPU CUDA or ROCm target")
        if self.compile and not spec.compiles_ahead_of_time:
            errors.append(
                f"{spec.name} has no ahead-of-time compiled engine; "
                "leave compile disabled and use the serve plan"
            )
        if not 1 <= self.serve_port <= 65535:
            errors.append("serve_port must be between 1 and 65,535")
        return errors

    def require_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "target": self.target,
            "source_format": self.source_format,
            "max_context": self.max_context,
            "max_batch_size": self.max_batch_size,
            "kv_cache_dtype": self.kv_cache_dtype,
            "tensor_parallel": self.tensor_parallel,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "prefix_caching": self.prefix_caching,
            "cuda_graphs": self.cuda_graphs,
            "flash_attention": self.flash_attention,
            "speculative_tokens": self.speculative_tokens,
            "compile": self.compile,
            "toolchain_path": self.toolchain_path,
            "serve_port": self.serve_port,
            "extra_options": self.extra_options or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> InferenceOptimizationConfig:
        data = data or {}
        valid = cls().__dict__.keys()
        return cls(**{key: value for key, value in data.items() if key in valid})


_REQUIRED_PROVIDERS = {
    (InferenceEngine.ONNXRUNTIME, DeviceTarget.QUALCOMM_NPU): (
        "QNNExecutionProvider",
        "Qualcomm NPU serving requires the QNN/HTP execution-provider recipe, "
        "not a generic CPU ONNX Runtime or vLLM plan",
    ),
    (InferenceEngine.ONNXRUNTIME, DeviceTarget.NVIDIA_GPU): (
        "CUDAExecutionProvider",
        "ONNX Runtime on NVIDIA GPU requires the CUDA execution-provider recipe",
    ),
    (InferenceEngine.ONNXRUNTIME, DeviceTarget.AMD_GPU): (
        "DmlExecutionProvider",
        "ONNX Runtime on AMD GPU requires the DirectML or ROCm execution-provider recipe",
    ),
}


def _provider_recipe_errors(
    engine: InferenceEngine, target: DeviceTarget, extras: dict[str, Any]
) -> list[str]:
    required = _REQUIRED_PROVIDERS.get((engine, target))
    if required is None:
        if target == DeviceTarget.QUALCOMM_NPU and engine != InferenceEngine.ONNXRUNTIME:
            return [
                "Qualcomm NPU serving requires the QNN/HTP execution-provider recipe, "
                "not a generic CPU ONNX Runtime or vLLM plan"
            ]
        return []
    provider, message = required
    selected = str(extras.get("execution_provider", ""))
    allowed = {provider}
    if target == DeviceTarget.AMD_GPU:
        allowed.add("ROCMExecutionProvider")
    if selected not in allowed:
        return [message]
    return []


def qnn_runtime_options() -> dict[str, Any]:
    return {
        "execution_provider": "QNNExecutionProvider",
        "qnn_backend": "htp",
        "backend_path": "QnnHtp.dll",
        "htp_performance_mode": "burst",
        "htp_graph_finalization_optimization_mode": "3",
    }
