"""One-button device recipes: NVIDIA GPU, AMD GPU, and NPU."""

from __future__ import annotations

import platform
from dataclasses import dataclass

from finetuner.inference.planner import detect_inference_hardware, recommended_config
from finetuner.inference.specs import InferenceOptimizationConfig
from finetuner.quantization.planner import HardwareCapability, recommended_config as quant_recommended
from finetuner.quantization.specs import DeviceTarget, QuantizationConfig


@dataclass(frozen=True)
class DeviceRecipe:
    target: DeviceTarget
    label: str
    inference: InferenceOptimizationConfig
    quantization: QuantizationConfig
    available: bool
    detail: str


_BUTTON_LABELS = {
    DeviceTarget.NVIDIA_GPU: "NVIDIA GPU",
    DeviceTarget.AMD_GPU: "AMD GPU",
    DeviceTarget.QUALCOMM_NPU: "Qualcomm NPU",
    DeviceTarget.INTEL_NPU: "Intel NPU",
}


def preferred_npu_target(
    capabilities: list[HardwareCapability] | None = None,
) -> DeviceTarget:
    caps = {item.target: item for item in (capabilities or detect_inference_hardware())}
    qualcomm = caps.get(DeviceTarget.QUALCOMM_NPU)
    intel = caps.get(DeviceTarget.INTEL_NPU)
    if qualcomm and qualcomm.available:
        return DeviceTarget.QUALCOMM_NPU
    if intel and intel.available:
        return DeviceTarget.INTEL_NPU
    processor = (platform.processor() or "").lower()
    machine = (platform.machine() or "").lower()
    if "qualcomm" in processor or "arm" in machine or "aarch64" in machine:
        return DeviceTarget.QUALCOMM_NPU
    return DeviceTarget.INTEL_NPU


def capability_for(
    target: DeviceTarget, capabilities: list[HardwareCapability] | None = None
) -> HardwareCapability:
    for item in capabilities or detect_inference_hardware():
        if item.target == target:
            return item
    return HardwareCapability(target, False, f"{target.value} not detected")


def recipe_for_target(
    target: DeviceTarget,
    memory_gb: float | None = None,
    *,
    run_on_device: bool = False,
    capabilities: list[HardwareCapability] | None = None,
) -> DeviceRecipe:
    inference = recommended_config(target, memory_gb)
    extras = dict(inference.extra_options or {})
    extras["run_on_device"] = run_on_device
    inference.extra_options = extras
    capability = capability_for(target, capabilities)
    return DeviceRecipe(
        target=target,
        label=_BUTTON_LABELS.get(target, target.value.replace("_", " ").title()),
        inference=inference,
        quantization=quant_recommended(target, memory_gb),
        available=capability.available,
        detail=capability.detail,
    )


def apply_device_recipe(
    project_config: object,
    target: DeviceTarget,
    *,
    run_on_device: bool = True,
    memory_gb: float | None = None,
) -> DeviceRecipe:
    recipe = recipe_for_target(target, memory_gb, run_on_device=run_on_device)
    project_config.inference = recipe.inference
    project_config.quantization = recipe.quantization
    return recipe


def launch_targets(
    capabilities: list[HardwareCapability] | None = None,
) -> tuple[DeviceTarget, ...]:
    caps = capabilities or detect_inference_hardware()
    return (
        DeviceTarget.NVIDIA_GPU,
        DeviceTarget.AMD_GPU,
        preferred_npu_target(caps),
    )
