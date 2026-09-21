from __future__ import annotations

from dataclasses import dataclass

from finetuner.core.job import TrainingConfig

ACCEL_ENGINES = frozenset({"npu", "tpu", "cpu"})
ALIAS_METHODS = frozenset({"npu", "tpu"})


@dataclass(frozen=True)
class AcceleratorInfo:
    kind: str
    available: bool
    name: str
    detail: str = ""


def npu_available() -> bool:
    try:
        from finetuner.training.npu.runtime import npu_provider_available

        return bool(npu_provider_available())
    except Exception:
        return False


def tpu_info() -> AcceleratorInfo:
    try:
        from finetuner.monitor.tpu import TpuCollector

        snap = TpuCollector.collect()
        if snap.available:
            return AcceleratorInfo("tpu", True, snap.name or "TPU", snap.detail)
    except Exception:
        pass
    try:
        import jax

        devices = [str(device) for device in jax.devices() if "tpu" in str(device).lower()]
        if devices:
            return AcceleratorInfo("tpu", True, devices[0], "JAX TPU")
    except Exception:
        pass
    return AcceleratorInfo("tpu", False, "", "")


def npu_info() -> AcceleratorInfo:
    if npu_available():
        return AcceleratorInfo("npu", True, "Qualcomm Hexagon HTP", "QNNExecutionProvider")
    return AcceleratorInfo("npu", False, "", "")


def detect_best() -> str:
    if npu_info().available:
        return "npu"
    if tpu_info().available:
        return "tpu"
    return "cuda"


def resolve_accelerator(training: TrainingConfig) -> str:
    raw = (getattr(training, "accelerator", "") or "cuda").strip().lower()
    method = (training.training_method or "sft").strip().lower()
    if method in ALIAS_METHODS and raw in {"", "cuda", "auto"}:
        return method
    if raw == "auto":
        return detect_best()
    return raw or "cuda"


def uses_accel_engine(training: TrainingConfig) -> bool:
    if (training.training_method or "") in ALIAS_METHODS:
        return True
    return resolve_accelerator(training) in ACCEL_ENGINES


def engine_method(training: TrainingConfig) -> str:
    method = (training.training_method or "sft").strip().lower()
    if method in ALIAS_METHODS:
        return "sft"
    return method
