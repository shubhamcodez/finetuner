"""Local Coral / Cloud TPU detection for the System page."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

_TPU_ENV_KEYS = (
    "TPU_NAME",
    "TPU_ACCELERATOR_TYPE",
    "TPU_VISIBLE_CHIPS",
    "TPU_WORKER_HOSTNAMES",
    "KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS",
)
_LOCAL_CACHE: list[str] | None = None


@dataclass
class TpuSnapshot:
    available: bool = False
    name: str = ""
    util_percent: float = 0.0
    mem_used_gb: float = 0.0
    mem_total_gb: float = 0.0
    detail: str = ""


def name_from_local_devices(names: list[str]) -> str:
    for name in names:
        lowered = name.lower()
        if "npu" in lowered or "hexagon" in lowered:
            continue
        if any(token in lowered for token in ("edge tpu", "coral", "apex tpu")):
            return name
        if "tpu" in lowered:
            return name
    return ""


def snapshot_from_env(env: Mapping[str, str]) -> TpuSnapshot | None:
    values = {key: (env.get(key) or "").strip() for key in _TPU_ENV_KEYS}
    if not any(values.values()):
        return None
    name = values["TPU_NAME"]
    accel = values["TPU_ACCELERATOR_TYPE"]
    if name and accel:
        label = f"{name} ({accel})"
    else:
        label = name or accel or "Cloud TPU"
    return TpuSnapshot(available=True, name=label, detail="Google Cloud TPU")


def linux_edge_tpu_name() -> str:
    if Path("/dev/apex_0").exists() or Path("/sys/class/apex/apex_0").exists():
        return "Coral Edge TPU"
    return ""


def list_windows_tpu_names() -> list[str]:
    global _LOCAL_CACHE
    if _LOCAL_CACHE is not None:
        return _LOCAL_CACHE
    if platform.system() != "Windows":
        _LOCAL_CACHE = []
        return _LOCAL_CACHE
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-PnpDevice -Status OK -ErrorAction SilentlyContinue | "
                "Where-Object { $_.FriendlyName -match 'TPU|Coral' } | "
                "Select-Object -ExpandProperty FriendlyName",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        names = []
    _LOCAL_CACHE = names
    return names


class TpuCollector:
    _name_loaded = False
    _cached: TpuSnapshot = TpuSnapshot()

    @classmethod
    def collect(cls) -> TpuSnapshot:
        if not cls._name_loaded:
            cls._cached = cls._detect()
            cls._name_loaded = True
        return cls._cached

    @classmethod
    def shutdown(cls) -> None:
        cls._name_loaded = False
        cls._cached = TpuSnapshot()

    @classmethod
    def _detect(cls) -> TpuSnapshot:
        local = name_from_local_devices(list_windows_tpu_names())
        if not local and sys.platform.startswith("linux"):
            local = linux_edge_tpu_name()
        if local:
            return TpuSnapshot(available=True, name=local, detail="Local Edge TPU")
        return snapshot_from_env(os.environ) or TpuSnapshot()
