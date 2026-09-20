"""Windows NPU utilization via PDH GPU Engine counters.

Task Manager treats Hexagon / Intel NPUs as compute adapters. Their load is
exposed as ``\\GPU Engine(*)\\Utilization Percentage`` instances whose engine
types are only Compute/Neural (no 3D/Video). Some hosts also publish
``\\NPU Engine(*)``.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass

_NPU_ENGINE_PREFIXES = ("Compute", "Neural", "NPU", "Inference")
_BYTES_PER_GB = 1024**3


@dataclass(frozen=True)
class EngineInstance:
    luid: str
    engine: int
    engine_type: str


@dataclass
class NpuSnapshot:
    available: bool = False
    name: str = ""
    util_percent: float = 0.0
    mem_used_gb: float = 0.0
    mem_shared_gb: float = 0.0


def parse_gpu_engine_instance(instance: str) -> EngineInstance | None:
    """Parse ``pid_X_luid_0xA_0xB_phys_N_eng_N_engtype_TYPE``."""
    parts = instance.split("_")
    if "luid" not in parts or "eng" not in parts or "engtype" not in parts:
        return None
    luid_i = parts.index("luid")
    eng_i = parts.index("eng")
    type_i = parts.index("engtype")
    try:
        luid = f"{parts[luid_i + 1]}_{parts[luid_i + 2]}"
        engine = int(parts[eng_i + 1])
    except (IndexError, ValueError):
        return None
    engine_type = "_".join(parts[type_i + 1 :])
    if not luid or not engine_type:
        return None
    return EngineInstance(luid, engine, engine_type)


def parse_adapter_luid(instance: str) -> str | None:
    """LUID from ``luid_0xA_0xB_phys_N`` or a full GPU Engine instance name."""
    parts = instance.split("_")
    if "luid" not in parts:
        return None
    i = parts.index("luid")
    try:
        return f"{parts[i + 1]}_{parts[i + 2]}"
    except IndexError:
        return None


def is_npu_engine(engine_type: str) -> bool:
    return engine_type.startswith(_NPU_ENGINE_PREFIXES)


def is_npu_adapter(engine_types: set[str]) -> bool:
    return bool(engine_types) and all(is_npu_engine(item) for item in engine_types)


def npu_luid_from_instances(instances: list[str]) -> str | None:
    engines: dict[str, set[str]] = {}
    for instance in instances:
        parsed = parse_gpu_engine_instance(instance)
        if parsed is None:
            continue
        engines.setdefault(parsed.luid, set()).add(parsed.engine_type)
    for luid, types in engines.items():
        if is_npu_adapter(types):
            return luid
    return None


def aggregate_npu_util(samples: list[tuple[str, float]], npu_luid: str) -> float:
    """Task Manager-style load: sum PIDs per engine, then max across engines."""
    by_engine: dict[int, float] = {}
    for instance, value in samples:
        parsed = parse_gpu_engine_instance(instance)
        if parsed is None or parsed.luid != npu_luid or not is_npu_engine(parsed.engine_type):
            continue
        by_engine[parsed.engine] = by_engine.get(parsed.engine, 0.0) + max(value, 0.0)
    if not by_engine:
        return 0.0
    return min(max(by_engine.values()), 100.0)


def memory_for_luid(
    committed: list[tuple[str, float]],
    shared: list[tuple[str, float]],
    npu_luid: str,
) -> tuple[float, float]:
    def _sum(samples: list[tuple[str, float]]) -> float:
        total = 0.0
        for instance, value in samples:
            if parse_adapter_luid(instance) == npu_luid:
                total += max(value, 0.0)
        return total / _BYTES_PER_GB

    return _sum(committed), _sum(shared)


def npu_display_name() -> str:
    try:
        from finetuner.quantization.planner import list_windows_accelerator_names
    except Exception:
        return ""
    try:
        names = list_windows_accelerator_names()
    except Exception:
        return ""
    for name in names:
        lowered = name.lower()
        if "npu" in lowered or "hexagon" in lowered:
            return name
    return ""


class WindowsNpuCollector:
    """Keeps a PDH query open so utilization rates have a previous sample."""

    _PDH_MORE_DATA = 0x800007D2
    _PDH_FMT_DOUBLE = 0x00000200
    _PDH_FMT_NOCAP100 = 0x00008000

    def __init__(self) -> None:
        self._pdh = ctypes.WinDLL("pdh")
        self._configure_pdh()
        self._query = wintypes.HANDLE()
        self._util = wintypes.HANDLE()
        self._npu_util = wintypes.HANDLE()
        self._committed = wintypes.HANDLE()
        self._shared = wintypes.HANDLE()
        self._ready = False
        self._bind()

    def _configure_pdh(self) -> None:
        pdh = self._pdh
        pdh.PdhOpenQueryW.argtypes = [
            wintypes.LPCWSTR,
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        pdh.PdhOpenQueryW.restype = wintypes.DWORD
        pdh.PdhAddEnglishCounterW.argtypes = [
            wintypes.HANDLE,
            wintypes.LPCWSTR,
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        pdh.PdhAddEnglishCounterW.restype = wintypes.DWORD
        pdh.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
        pdh.PdhCollectQueryData.restype = wintypes.DWORD
        pdh.PdhGetFormattedCounterArrayW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
        ]
        pdh.PdhGetFormattedCounterArrayW.restype = wintypes.DWORD
        pdh.PdhCloseQuery.argtypes = [wintypes.HANDLE]
        pdh.PdhCloseQuery.restype = wintypes.DWORD

    def _bind(self) -> None:
        if self._status(self._pdh.PdhOpenQueryW(None, None, ctypes.byref(self._query))) != 0:
            return
        self._add(r"\GPU Engine(*)\Utilization Percentage", self._util)
        self._add(r"\NPU Engine(*)\Utilization Percentage", self._npu_util)
        self._add(r"\GPU Adapter Memory(*)\Total Committed", self._committed)
        self._add(r"\GPU Adapter Memory(*)\Shared Usage", self._shared)
        if self._util.value or self._npu_util.value:
            self._pdh.PdhCollectQueryData(self._query)
            self._ready = True

    def _add(self, path: str, handle: wintypes.HANDLE) -> None:
        status = self._pdh.PdhAddEnglishCounterW(self._query, path, None, ctypes.byref(handle))
        if self._status(status) != 0:
            handle.value = None

    def close(self) -> None:
        if self._query.value:
            self._pdh.PdhCloseQuery(self._query)
        self._query = wintypes.HANDLE()
        self._ready = False

    def read(self, name: str) -> NpuSnapshot:
        snap = NpuSnapshot(name=name)
        if not self._ready:
            snap.available = bool(name)
            return snap
        self._pdh.PdhCollectQueryData(self._query)
        engine_samples = self._array(self._util)
        npu_engine_samples = self._array(self._npu_util)
        luid = npu_luid_from_instances([instance for instance, _ in engine_samples])
        if luid:
            snap.available = True
            snap.util_percent = aggregate_npu_util(engine_samples, luid)
            snap.mem_used_gb, snap.mem_shared_gb = memory_for_luid(
                self._array(self._committed),
                self._array(self._shared),
                luid,
            )
        elif npu_engine_samples:
            snap.available = True
            snap.util_percent = min(max(value for _, value in npu_engine_samples), 100.0)
        elif name:
            snap.available = True
        if snap.available and not snap.name:
            snap.name = "NPU"
        return snap

    def _array(self, counter: wintypes.HANDLE) -> list[tuple[str, float]]:
        if not counter.value:
            return []
        size = wintypes.DWORD(0)
        count = wintypes.DWORD(0)
        fmt = self._PDH_FMT_DOUBLE | self._PDH_FMT_NOCAP100
        status = self._pdh.PdhGetFormattedCounterArrayW(
            counter, fmt, ctypes.byref(size), ctypes.byref(count), None
        )
        if self._status(status) != self._PDH_MORE_DATA or size.value == 0:
            return []
        buf = ctypes.create_string_buffer(size.value)
        status = self._pdh.PdhGetFormattedCounterArrayW(
            counter,
            fmt,
            ctypes.byref(size),
            ctypes.byref(count),
            ctypes.cast(buf, ctypes.POINTER(_PdhItem)),
        )
        if self._status(status) != 0:
            return []
        items = ctypes.cast(buf, ctypes.POINTER(_PdhItem))
        samples: list[tuple[str, float]] = []
        for index in range(count.value):
            item = items[index]
            if item.FmtValue.CStatus != 0 or not item.szName:
                continue
            samples.append((item.szName, float(item.FmtValue.doubleValue)))
        return samples

    @staticmethod
    def _status(code: int) -> int:
        return code & 0xFFFFFFFF


class _PdhValue(ctypes.Structure):
    class _Union(ctypes.Union):
        _fields_ = [
            ("longValue", wintypes.LONG),
            ("doubleValue", ctypes.c_double),
            ("largeValue", ctypes.c_longlong),
            ("AnsiStringValue", wintypes.LPCSTR),
            ("WideStringValue", wintypes.LPCWSTR),
        ]

    _anonymous_ = ("u",)
    _fields_ = [("CStatus", wintypes.DWORD), ("u", _Union)]


class _PdhItem(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _PdhValue)]


class NpuCollector:
    _impl: WindowsNpuCollector | None = None
    _name: str = ""
    _name_loaded = False

    @classmethod
    def collect(cls) -> NpuSnapshot:
        if sys.platform != "win32":
            return NpuSnapshot()
        if not cls._name_loaded:
            cls._name = npu_display_name()
            cls._name_loaded = True
        if cls._impl is None:
            try:
                cls._impl = WindowsNpuCollector()
            except Exception:
                return NpuSnapshot(available=bool(cls._name), name=cls._name)
        try:
            return cls._impl.read(cls._name)
        except Exception:
            return NpuSnapshot(available=bool(cls._name), name=cls._name)

    @classmethod
    def shutdown(cls) -> None:
        if cls._impl is not None:
            try:
                cls._impl.close()
            except Exception:
                pass
            cls._impl = None
        cls._name_loaded = False
        cls._name = ""
