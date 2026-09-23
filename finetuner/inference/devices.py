"""Device recipes and the adaptive picker: best specialist per machine."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class AdaptiveChoice:
    recipe: DeviceRecipe
    reason: str
    skipped: tuple[str, ...] = ()


# Highest-ranked present accelerator wins. Qualcomm HTP is only chosen when the
# artifact is a real QNN graph; otherwise this machine's measured winner is CPU.
_HOST_PRIORITY = (
    DeviceTarget.NVIDIA_GPU,
    DeviceTarget.AMD_GPU,
    DeviceTarget.INTEL_NPU,
    DeviceTarget.APPLE_GPU,
    DeviceTarget.INTEL_GPU,
    DeviceTarget.QUALCOMM_NPU,
    DeviceTarget.CPU,
)


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
    previous = getattr(project_config, "inference", None)
    if previous is not None:
        recipe.inference.serve_port = previous.serve_port or 1234
    project_config.inference = recipe.inference
    project_config.quantization = recipe.quantization
    return recipe


def detect_htp_ready(model_path: str = "") -> bool:
    """True only for a Qualcomm context-binary / QNN genai package, not generic ONNX."""
    if not model_path:
        return False
    root = Path(model_path)
    if root.is_file():
        root = root.parent
    if not root.is_dir():
        return False
    binaries = list(root.glob("*.bin")) or list(root.glob("**/*qnn*.bin"))
    genai = root / "genai_config.json"
    if not binaries or not genai.exists():
        return False
    try:
        payload = json.loads(genai.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    blob = json.dumps(payload).lower()
    return "qnn" in blob or "htp" in blob


def select_best_runtime(
    capabilities: list[HardwareCapability] | None = None,
    *,
    memory_gb: float | None = None,
    htp_ready: bool = False,
    vllm_available: bool | None = None,
    run_on_device: bool = False,
    model_path: str = "",
) -> AdaptiveChoice:
    """Pick the strongest specialist that can actually run on this host + artifact."""
    caps = capabilities or detect_inference_hardware()
    present = {item.target: item for item in caps if item.available}
    ready = htp_ready or detect_htp_ready(model_path)
    skipped: list[str] = []
    for target in _HOST_PRIORITY:
        if target == DeviceTarget.CPU:
            recipe = recipe_for_target(
                target, memory_gb, run_on_device=run_on_device, capabilities=caps
            )
            extras = dict(recipe.inference.extra_options or {})
            extras["adaptive"] = True
            recipe.inference.extra_options = extras
            return AdaptiveChoice(
                recipe,
                "llama.cpp on CPU is the portable floor when no faster accelerator is ready",
                tuple(skipped),
            )
        if target not in present:
            continue
        if target == DeviceTarget.QUALCOMM_NPU and not ready:
            skipped.append(
                "Qualcomm NPU is present, but the model is not a QNN/HTP context-binary graph"
            )
            continue
        if target == DeviceTarget.NVIDIA_GPU and vllm_available is not None:
            inference = recommended_config(target, memory_gb, vllm_available=vllm_available)
            extras = dict(inference.extra_options or {})
            extras["run_on_device"] = run_on_device
            extras["adaptive"] = True
            inference.extra_options = extras
            capability = present[target]
            recipe = DeviceRecipe(
                target=target,
                label=_BUTTON_LABELS.get(target, target.value.replace("_", " ").title()),
                inference=inference,
                quantization=quant_recommended(target, memory_gb),
                available=capability.available,
                detail=capability.detail,
            )
        else:
            recipe = recipe_for_target(
                target, memory_gb, run_on_device=run_on_device, capabilities=caps
            )
            extras = dict(recipe.inference.extra_options or {})
            extras["adaptive"] = True
            recipe.inference.extra_options = extras
        return AdaptiveChoice(
            recipe,
            f"{recipe.label} is the highest-ranked accelerator that is present and ready",
            tuple(skipped),
        )
    raise AssertionError("CPU is always selectable")


def apply_best_runtime(
    project_config: object,
    *,
    model_path: str = "",
    run_on_device: bool = True,
    memory_gb: float | None = None,
    capabilities: list[HardwareCapability] | None = None,
    htp_ready: bool = False,
    vllm_available: bool | None = None,
) -> AdaptiveChoice:
    choice = select_best_runtime(
        capabilities,
        memory_gb=memory_gb,
        htp_ready=htp_ready,
        vllm_available=vllm_available,
        run_on_device=run_on_device,
        model_path=model_path,
    )
    extras = dict(choice.recipe.inference.extra_options or {})
    extras["adaptive_reason"] = choice.reason
    extras["adaptive_skipped"] = list(choice.skipped)
    choice.recipe.inference.extra_options = extras
    previous = getattr(project_config, "inference", None)
    if previous is not None:
        choice.recipe.inference.serve_port = previous.serve_port or 1234
    project_config.inference = choice.recipe.inference
    project_config.quantization = choice.recipe.quantization
    return choice


@dataclass(frozen=True)
class DeviceOffer:
    detected: tuple[str, ...]
    choice: AdaptiveChoice
    title: str
    message: str


def describe_device_offer(
    model_path: str = "",
    capabilities: list[HardwareCapability] | None = None,
) -> DeviceOffer:
    """Text for the load-time 'optimize for your device?' prompt."""
    caps = capabilities or detect_inference_hardware()
    choice = select_best_runtime(caps, model_path=model_path, run_on_device=False)
    detected = tuple(
        f"{item.target.value.replace('_', ' ')} ({item.detail})"
        for item in caps
        if item.available
    )
    engine = choice.recipe.inference.engine
    target = choice.recipe.target.value.replace("_", " ")
    skipped = ""
    if choice.skipped:
        skipped = "\n\n" + "\n".join(choice.skipped)
    message = (
        f"Detected: {', '.join(detected) or 'CPU only'}.\n\n"
        f"Fastest ready engine on this machine: {engine} on {target}.\n"
        f"{choice.reason}.{skipped}\n\n"
        "Optimize for this device, then serve the model on port 1234? "
        "You can also serve without optimizing."
    )
    return DeviceOffer(detected, choice, "Optimize for your device?", message)


@dataclass(frozen=True)
class DeviceMemory:
    available: bool
    free_gb: float | None = None
    total_gb: float | None = None


def _nvidia_memory_gb() -> tuple[float, float] | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
        free = getattr(info, "free", None)
        if free is None:
            free = info.total - info.used
        return float(free) / (1024**3), float(info.total) / (1024**3)
    except Exception:
        return None


def _host_memory_gb() -> tuple[float, float]:
    import psutil

    memory = psutil.virtual_memory()
    return memory.available / (1024**3), memory.total / (1024**3)


def device_memory_snapshot(
    capabilities: list[HardwareCapability] | None = None,
) -> dict[str, DeviceMemory]:
    """Availability and memory for every device row. Missing devices stay empty."""
    caps = capabilities if capabilities is not None else detect_inference_hardware()
    present = {item.target: item.available for item in caps}
    ram_free, ram_total = _host_memory_gb()
    nvidia = _nvidia_memory_gb()
    snapshot: dict[str, DeviceMemory] = {
        DeviceTarget.CPU.value: DeviceMemory(True, ram_free, ram_total),
    }
    for target in (
        DeviceTarget.NVIDIA_GPU,
        DeviceTarget.AMD_GPU,
        DeviceTarget.INTEL_GPU,
        DeviceTarget.APPLE_GPU,
        DeviceTarget.INTEL_NPU,
        DeviceTarget.QUALCOMM_NPU,
    ):
        if not present.get(target, False):
            snapshot[target.value] = DeviceMemory(False)
            continue
        if target == DeviceTarget.NVIDIA_GPU and nvidia is not None:
            snapshot[target.value] = DeviceMemory(True, nvidia[0], nvidia[1])
        else:
            snapshot[target.value] = DeviceMemory(True, ram_free, ram_total)
    return snapshot


def launch_targets(
    capabilities: list[HardwareCapability] | None = None,
) -> tuple[DeviceTarget, ...]:
    caps = capabilities or detect_inference_hardware()
    return (
        DeviceTarget.NVIDIA_GPU,
        DeviceTarget.AMD_GPU,
        preferred_npu_target(caps),
    )
