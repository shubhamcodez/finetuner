from __future__ import annotations

import importlib.util
import platform
import shutil
from dataclasses import dataclass

from finetuner.quantization.specs import (
    DeviceTarget,
    QuantizationBackend,
    QuantizationConfig,
    backend_specs,
)


@dataclass(frozen=True)
class HardwareCapability:
    target: DeviceTarget
    available: bool
    detail: str


def detect_hardware() -> list[HardwareCapability]:
    capabilities = [HardwareCapability(DeviceTarget.CPU, True, platform.processor() or "CPU")]
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        cuda_name = torch.cuda.get_device_name(0) if cuda else "CUDA unavailable"
    except Exception:
        cuda, cuda_name = False, "PyTorch unavailable"
    capabilities.append(HardwareCapability(DeviceTarget.NVIDIA_GPU, cuda, cuda_name))

    onnx_available = importlib.util.find_spec("onnxruntime") is not None
    providers: list[str] = []
    if onnx_available:
        try:
            import onnxruntime

            providers = onnxruntime.get_available_providers()
        except Exception:
            providers = []
    capabilities.extend(
        [
            HardwareCapability(
                DeviceTarget.AMD_GPU,
                "DmlExecutionProvider" in providers or shutil.which("llama-cli") is not None,
                "DirectML or llama.cpp backend",
            ),
            HardwareCapability(
                DeviceTarget.QUALCOMM_NPU,
                "QNNExecutionProvider" in providers,
                "ONNX Runtime QNN execution provider",
            ),
            HardwareCapability(
                DeviceTarget.APPLE_GPU,
                platform.system() == "Darwin",
                "Metal via llama.cpp",
            ),
        ]
    )
    try:
        import openvino as ov

        devices = {item.split(".", 1)[0] for item in ov.Core().available_devices}
    except Exception:
        devices = set()
    capabilities.extend(
        [
            HardwareCapability(DeviceTarget.INTEL_GPU, "GPU" in devices, "OpenVINO GPU"),
            HardwareCapability(DeviceTarget.INTEL_NPU, "NPU" in devices, "OpenVINO NPU"),
        ]
    )
    return capabilities


def recommended_config(target: DeviceTarget, memory_gb: float | None = None) -> QuantizationConfig:
    if target == DeviceTarget.INTEL_NPU:
        return QuantizationConfig("openvino", target.value, 4)
    if target == DeviceTarget.QUALCOMM_NPU:
        raise ValueError(
            "Qualcomm NPU deployment requires a device-specific QNN/QAIRT conversion recipe; "
            "generic ONNX quantization is not sufficient."
        )
    if target in {DeviceTarget.NVIDIA_GPU} and memory_gb is not None and memory_gb < 16:
        return QuantizationConfig("awq", target.value, 4)
    if target in {
        DeviceTarget.NVIDIA_GPU,
        DeviceTarget.AMD_GPU,
        DeviceTarget.INTEL_GPU,
        DeviceTarget.APPLE_GPU,
        DeviceTarget.CPU,
    }:
        return QuantizationConfig("gguf", target.value, 4)
    raise ValueError(f"No deployment recommendation for {target.value}")


def compatible_backends(target: DeviceTarget) -> list[QuantizationBackend]:
    return [spec.backend for spec in backend_specs() if target in spec.targets]
