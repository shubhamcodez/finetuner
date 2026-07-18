from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class QuantizationBackend(str, Enum):
    GGUF = "gguf"
    OPENVINO = "openvino"
    ONNX = "onnx"
    AWQ = "awq"


class DeviceTarget(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    NVIDIA_GPU = "nvidia_gpu"
    AMD_GPU = "amd_gpu"
    INTEL_GPU = "intel_gpu"
    APPLE_GPU = "apple_gpu"
    INTEL_NPU = "intel_npu"
    QUALCOMM_NPU = "qualcomm_npu"


@dataclass(frozen=True)
class BackendSpec:
    backend: QuantizationBackend
    name: str
    output_format: str
    supported_bits: tuple[int, ...]
    targets: tuple[DeviceTarget, ...]
    package_hint: str
    description: str
    experimental_targets: tuple[DeviceTarget, ...] = ()


_BACKENDS: dict[QuantizationBackend, BackendSpec] = {
    QuantizationBackend.GGUF: BackendSpec(
        QuantizationBackend.GGUF,
        "GGUF / llama.cpp",
        "gguf",
        (2, 3, 4, 5, 6, 8),
        (
            DeviceTarget.CPU,
            DeviceTarget.NVIDIA_GPU,
            DeviceTarget.AMD_GPU,
            DeviceTarget.INTEL_GPU,
            DeviceTarget.APPLE_GPU,
        ),
        "llama.cpp",
        "Broad CPU/GPU deployment through llama.cpp backends (CUDA, HIP, Metal, Vulkan, SYCL).",
    ),
    QuantizationBackend.OPENVINO: BackendSpec(
        QuantizationBackend.OPENVINO,
        "OpenVINO IR",
        "openvino_ir",
        (4, 8),
        (DeviceTarget.CPU, DeviceTarget.INTEL_GPU, DeviceTarget.INTEL_NPU),
        "pip install optimum[openvino]",
        "Intel CPU, GPU, and supported Core Ultra NPU deployment with NNCF compression.",
    ),
    QuantizationBackend.ONNX: BackendSpec(
        QuantizationBackend.ONNX,
        "ONNX Runtime",
        "onnx",
        (8,),
        (DeviceTarget.CPU,),
        "pip install optimum[onnxruntime]",
        "Portable INT8 graph for CPU inference. GPU/NPU execution needs a provider-specific recipe.",
    ),
    QuantizationBackend.AWQ: BackendSpec(
        QuantizationBackend.AWQ,
        "Activation-aware Weight Quantization",
        "safetensors",
        (4,),
        (DeviceTarget.NVIDIA_GPU,),
        "pip install llm-awq",
        "Calibration-aware 4-bit weights for compatible CUDA inference engines.",
    ),
}


def backend_specs() -> list[BackendSpec]:
    return list(_BACKENDS.values())


def get_backend_spec(backend: QuantizationBackend | str) -> BackendSpec:
    return _BACKENDS[QuantizationBackend(backend)]


@dataclass
class QuantizationConfig:
    backend: str = QuantizationBackend.GGUF.value
    target: str = DeviceTarget.CPU.value
    bits: int = 4
    group_size: int = 128
    scheme: str = "symmetric"
    calibration_dataset: str = ""
    llama_cpp_path: str = ""
    extra_options: dict[str, Any] | None = None

    def validate(self) -> list[str]:
        errors: list[str] = []
        try:
            backend = QuantizationBackend(self.backend)
            spec = get_backend_spec(backend)
        except ValueError:
            return [f"Unknown quantization backend: {self.backend}"]
        try:
            target = DeviceTarget(self.target)
        except ValueError:
            return [f"Unknown deployment target: {self.target}"]
        if target == DeviceTarget.AUTO:
            errors.append("Select a concrete deployment target before quantizing")
        elif target not in spec.targets:
            errors.append(f"{spec.name} does not support target {target.value}")
        if self.bits not in spec.supported_bits:
            errors.append(
                f"{spec.name} supports {', '.join(map(str, spec.supported_bits))}-bit artifacts, "
                f"not {self.bits}-bit"
            )
        if self.group_size not in (-1, 0) and self.group_size < 16:
            errors.append("group_size must be -1/0 (per-channel) or at least 16")
        if self.scheme not in {"symmetric", "asymmetric"}:
            errors.append("scheme must be 'symmetric' or 'asymmetric'")
        if backend == QuantizationBackend.AWQ and not self.calibration_dataset:
            errors.append("AWQ requires a representative calibration dataset")
        return errors

    def require_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "target": self.target,
            "bits": self.bits,
            "group_size": self.group_size,
            "scheme": self.scheme,
            "calibration_dataset": self.calibration_dataset,
            "llama_cpp_path": self.llama_cpp_path,
            "extra_options": self.extra_options or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> QuantizationConfig:
        data = data or {}
        valid = cls().__dict__.keys()
        return cls(**{key: value for key, value in data.items() if key in valid})
