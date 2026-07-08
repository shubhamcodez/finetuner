from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, QTimer, Signal


@dataclass
class SystemStats:
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_util_percent: float = 0.0
    gpu_mem_used_gb: float = 0.0
    gpu_mem_total_gb: float = 0.0
    gpu_temp_c: float = 0.0


class StatsCollector:
    """Collects system stats. NVML is initialized once per process."""

    _nvml_initialized = False
    _gpu_handle = None

    @classmethod
    def _init_nvml(cls) -> None:
        if cls._nvml_initialized:
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            cls._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            cls._nvml_initialized = True
        except Exception:
            cls._nvml_initialized = False
            cls._gpu_handle = None

    @classmethod
    def shutdown_nvml(cls) -> None:
        if not cls._nvml_initialized:
            return
        try:
            import pynvml

            pynvml.nvmlShutdown()
        except Exception:
            pass
        cls._nvml_initialized = False
        cls._gpu_handle = None

    @classmethod
    def collect(cls) -> SystemStats:
        import psutil

        stats = SystemStats()
        stats.cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        stats.ram_used_gb = mem.used / (1024**3)
        stats.ram_total_gb = mem.total / (1024**3)

        cls._init_nvml()
        if cls._gpu_handle is not None:
            try:
                import pynvml

                stats.gpu_available = True
                stats.gpu_name = pynvml.nvmlDeviceGetName(cls._gpu_handle)
                if isinstance(stats.gpu_name, bytes):
                    stats.gpu_name = stats.gpu_name.decode("utf-8", errors="replace")
                util = pynvml.nvmlDeviceGetUtilizationRates(cls._gpu_handle)
                stats.gpu_util_percent = float(util.gpu)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(cls._gpu_handle)
                stats.gpu_mem_used_gb = mem_info.used / (1024**3)
                stats.gpu_mem_total_gb = mem_info.total / (1024**3)
                try:
                    stats.gpu_temp_c = float(
                        pynvml.nvmlDeviceGetTemperature(
                            cls._gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    )
                except Exception:
                    stats.gpu_temp_c = 0.0
            except Exception:
                stats.gpu_available = False
        return stats


class StatsPoller(QObject):
    """Main-thread timer poller — avoids QThread + Qt Charts shutdown crashes."""

    stats_updated = Signal(object)

    def __init__(self, interval_ms: int = 1000, parent=None) -> None:
        super().__init__(parent)
        import psutil

        psutil.cpu_percent(interval=None)
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    def _poll(self) -> None:
        self.stats_updated.emit(StatsCollector.collect())

    def stop(self) -> None:
        self._timer.stop()
        StatsCollector.shutdown_nvml()
