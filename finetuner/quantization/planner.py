from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
from dataclasses import dataclass

from finetuner.quantization.specs import (
    DeviceTarget,
    QuantizationBackend,
    QuantizationConfig,
    backend_specs,
)

_ACCELERATOR_CACHE: list[str] | None = None


def list_windows_accelerator_names() -> list[str]:
    """PnP display and compute-accelerator names. Cached for the process."""
    global _ACCELERATOR_CACHE
    if _ACCELERATOR_CACHE is not None:
        return _ACCELERATOR_CACHE
    if platform.system() != "Windows":
        _ACCELERATOR_CACHE = []
        return _ACCELERATOR_CACHE
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-PnpDevice -Class ComputeAccelerator,Display -Status OK "
                "-ErrorAction SilentlyContinue | Select-Object -ExpandProperty FriendlyName",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        names = []
    _ACCELERATOR_CACHE = names
    return names


def _name_matches(names: list[str], *needles: str) -> str:
    for name in names:
        lowered = name.lower()
        if all(needle in lowered for needle in needles):
            return name
    return ""


@dataclass(frozen=True)
class HardwareCapability:
    target: DeviceTarget
    available: bool
    detail: str


def detect_hardware() -> list[HardwareCapability]:
    capabilities = [HardwareCapability(DeviceTarget.CPU, True, platform.processor() or "CPU")]
    accelerators = list_windows_accelerator_names()
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        cuda_name = torch.cuda.get_device_name(0) if cuda else "CUDA unavailable"
    except Exception:
        cuda, cuda_name = False, "PyTorch unavailable"
    nvidia_pnp = _name_matches(accelerators, "nvidia")
    nvidia = cuda or shutil.which("nvidia-smi") is not None or bool(nvidia_pnp)
    nvidia_detail = cuda_name if cuda else (nvidia_pnp or "NVIDIA GPU not detected")
    capabilities.append(HardwareCapability(DeviceTarget.NVIDIA_GPU, nvidia, nvidia_detail))

    onnx_available = importlib.util.find_spec("onnxruntime") is not None
    providers: list[str] = []
    if onnx_available:
        try:
            import onnxruntime

            providers = onnxruntime.get_available_providers()
        except Exception:
            providers = []
    amd_pnp = _name_matches(accelerators, "radeon") or _name_matches(accelerators, "amd", "gpu")
    if "adreno" in amd_pnp.lower():
        amd_pnp = ""
    amd = bool(amd_pnp) or (
        "DmlExecutionProvider" in providers and bool(_name_matches(accelerators, "amd"))
    ) or shutil.which("hipinfo") is not None or shutil.which("rocm-smi") is not None
    hexagon = _name_matches(accelerators, "hexagon", "npu") or _name_matches(
        accelerators, "qualcomm", "npu"
    )
    qnn = "QNNExecutionProvider" in providers
    capabilities.extend(
        [
            HardwareCapability(
                DeviceTarget.AMD_GPU,
                amd,
                amd_pnp or ("DirectML" if "DmlExecutionProvider" in providers else "AMD GPU not detected"),
            ),
            HardwareCapability(
                DeviceTarget.QUALCOMM_NPU,
                qnn or bool(hexagon),
                hexagon or ("ONNX Runtime QNN execution provider" if qnn else "Qualcomm NPU not detected"),
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
        return QuantizationConfig(
            "onnx",
            target.value,
            8,
            extra_options={
                "execution_provider": "QNNExecutionProvider",
                "qnn_backend": "htp",
                "backend_path": "QnnHtp.dll",
            },
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
