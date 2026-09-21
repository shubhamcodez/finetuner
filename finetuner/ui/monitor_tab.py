from __future__ import annotations

from collections import deque

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from finetuner.monitor.stats import StatsPoller, SystemStats
from finetuner.ui.theme import Theme, chart_colors

_CHART_HEIGHT = 300


class Sparkline(QWidget):
    def __init__(self, y_max: float = 100.0, parent=None) -> None:
        super().__init__(parent)
        self._y_max = max(y_max, 1.0)
        self._history: deque[float] = deque(maxlen=60)
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def append(self, value: float) -> None:
        self._history.append(max(0.0, float(value)))
        self.update()

    def paintEvent(self, event) -> None:
        del event
        colors = chart_colors()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(colors.get("background", Theme.SURFACE)))
        rect = self.rect().adjusted(6, 8, -6, -8)
        if rect.width() < 4 or rect.height() < 4:
            return
        painter.setPen(QPen(QColor(colors["grid"]), 1))
        for step in range(5):
            y = rect.top() + rect.height() * step / 4
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
        if len(self._history) < 2:
            return
        span = max(len(self._history) - 1, 1)
        painter.setPen(QPen(QColor(colors["line"]), 2.5))
        points = []
        for index, value in enumerate(self._history):
            x = rect.left() + rect.width() * index / span
            y = rect.bottom() - rect.height() * min(value / self._y_max, 1.0)
            points.append((x, y))
        for start, end in zip(points, points[1:]):
            painter.drawLine(int(start[0]), int(start[1]), int(end[0]), int(end[1]))


class MetricCard(QFrame):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("SurfaceCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setSpacing(1)
        layout.setContentsMargins(10, 8, 10, 8)
        heading = QLabel(title)
        heading.setObjectName("MetaLabel")
        self.value_label = QLabel("--")
        self.value_label.setObjectName("MetricValue")
        self.detail_label = QLabel("")
        self.detail_label.setObjectName("MetricDetail")
        self.detail_label.setWordWrap(False)
        layout.addWidget(heading)
        layout.addWidget(self.value_label)
        layout.addWidget(self.detail_label)

    def set_value(self, value: str, detail: str = "") -> None:
        self.value_label.setText(value)
        self.detail_label.setText(detail)


class HistoryChart(QFrame):
    def __init__(self, title: str, y_max: float = 100.0, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("SurfaceCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(_CHART_HEIGHT)
        heading = QLabel(title)
        heading.setObjectName("CardTitle")
        self._view = Sparkline(y_max)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        layout.addWidget(heading)
        layout.addWidget(self._view, 1)

    def append(self, value: float) -> None:
        self._view.append(value)


class MonitorTab(QScrollArea):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidgetResizable(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._build_ui()
        self._poller = StatsPoller(interval_ms=1000, parent=self)
        self._poller.stats_updated.connect(self._on_stats)

    def minimumSizeHint(self) -> QSize:
        return QSize(0, 0)

    def sizeHint(self) -> QSize:
        return QSize(400, 240)

    def _build_ui(self) -> None:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 16, 16)
        layout.setSpacing(8)

        health = QFrame()
        health.setObjectName("SurfaceCard")
        health_layout = QHBoxLayout(health)
        health_layout.setContentsMargins(14, 8, 14, 8)
        health_layout.setSpacing(12)
        self.health_rows = {}
        for key, label in (
            ("compute", "Compute"),
            ("storage", "Storage"),
            ("models", "Model registry"),
            ("jobs", "Job scheduler"),
            ("services", "Local services"),
        ):
            cell = QVBoxLayout()
            cell.setSpacing(2)
            name = QLabel(label)
            name.setObjectName("MetaLabel")
            state = QLabel("Healthy")
            state.setObjectName("CardTitle")
            cell.addWidget(name)
            cell.addWidget(state)
            health_layout.addLayout(cell, 1)
            self.health_rows[key] = state
        layout.addWidget(health)

        cards = QGridLayout()
        cards.setSpacing(8)
        self.cpu_card = MetricCard("CPU Utilization")
        self.ram_card = MetricCard("System Memory")
        self.gpu_card = MetricCard("GPU Utilization")
        self.vram_card = MetricCard("GPU Memory")
        self.npu_card = MetricCard("NPU Utilization")
        self.npu_mem_card = MetricCard("NPU Memory")
        self.tpu_card = MetricCard("TPU Utilization")
        self.tpu_mem_card = MetricCard("TPU Memory")
        cards.addWidget(self.cpu_card, 0, 0)
        cards.addWidget(self.ram_card, 0, 1)
        cards.addWidget(self.gpu_card, 0, 2)
        cards.addWidget(self.vram_card, 0, 3)
        cards.addWidget(self.npu_card, 1, 0)
        cards.addWidget(self.npu_mem_card, 1, 1)
        cards.addWidget(self.tpu_card, 1, 2)
        cards.addWidget(self.tpu_mem_card, 1, 3)
        layout.addLayout(cards)

        charts = QGridLayout()
        charts.setSpacing(8)
        charts.setColumnStretch(0, 1)
        charts.setColumnStretch(1, 1)
        self.cpu_chart = HistoryChart("CPU History", y_max=100)
        self.gpu_chart = HistoryChart("GPU History", y_max=100)
        self.npu_chart = HistoryChart("NPU History", y_max=100)
        self.tpu_chart = HistoryChart("TPU History", y_max=100)
        charts.addWidget(self.cpu_chart, 0, 0)
        charts.addWidget(self.gpu_chart, 0, 1)
        charts.addWidget(self.npu_chart, 1, 0)
        charts.addWidget(self.tpu_chart, 1, 1)
        layout.addLayout(charts)

        self.setWidget(content)
        self._scroll = self
        self._fit_content_width()

    def _fit_content_width(self) -> None:
        content = self.widget()
        if content is None or content.layout() is None:
            return
        height = max(content.layout().sizeHint().height(), 800)
        content.setMinimumHeight(height)
        content.resize(max(self.viewport().width(), 1), height)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_content_width()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._fit_content_width()

    def _on_stats(self, stats: SystemStats) -> None:
        self.cpu_card.set_value(f"{stats.cpu_percent:.1f}%")
        self.cpu_chart.append(stats.cpu_percent)

        ram_pct = (stats.ram_used_gb / stats.ram_total_gb * 100) if stats.ram_total_gb else 0
        self.ram_card.set_value(
            f"{stats.ram_used_gb:.1f} GB",
            f"{ram_pct:.0f}% of {stats.ram_total_gb:.1f} GB total",
        )

        if stats.gpu_available:
            self.gpu_card.set_value(
                f"{stats.gpu_util_percent:.0f}%",
                stats.gpu_name,
            )
            self.gpu_chart.append(stats.gpu_util_percent)
            self.vram_card.set_value(
                f"{stats.gpu_mem_used_gb:.1f} GB",
                f"{stats.gpu_mem_total_gb:.1f} GB · {stats.gpu_temp_c:.0f}°C",
            )
            self.gpu_card.setToolTip(f"NVIDIA GPU active — {stats.gpu_name}")
        else:
            self.gpu_card.set_value("Unavailable", "No NVIDIA GPU detected")
            self.vram_card.set_value("—", "")
            self.gpu_card.setToolTip(
                "GPU unavailable. Install NVIDIA drivers to enable CUDA fine-tuning."
            )

        if stats.npu_available:
            self.npu_card.set_value(
                f"{stats.npu_util_percent:.0f}%",
                stats.npu_name,
            )
            self.npu_chart.append(stats.npu_util_percent)
            mem_detail = (
                f"{stats.npu_mem_shared_gb:.1f} GB shared"
                if stats.npu_mem_shared_gb
                else "Windows adapter committed memory"
            )
            if stats.npu_mem_used_gb or stats.npu_mem_shared_gb:
                self.npu_mem_card.set_value(f"{stats.npu_mem_used_gb:.1f} GB", mem_detail)
            else:
                self.npu_mem_card.set_value("—", mem_detail)
            self.npu_card.setToolTip(f"NPU active — {stats.npu_name or 'Windows NPU'}")
        else:
            self.npu_card.set_value("Unavailable", "No NPU detected")
            self.npu_mem_card.set_value("—", "")
            self.npu_card.setToolTip(
                "NPU unavailable. Qualcomm Hexagon and Intel NPUs appear here when Windows exposes them."
            )

        if stats.tpu_available:
            self.tpu_card.set_value(
                f"{stats.tpu_util_percent:.0f}%",
                stats.tpu_name,
            )
            self.tpu_chart.append(stats.tpu_util_percent)
            if stats.tpu_mem_total_gb:
                self.tpu_mem_card.set_value(
                    f"{stats.tpu_mem_used_gb:.1f} GB",
                    f"{stats.tpu_mem_total_gb:.1f} GB · {stats.tpu_detail}".strip(" ·"),
                )
            elif stats.tpu_mem_used_gb:
                self.tpu_mem_card.set_value(f"{stats.tpu_mem_used_gb:.1f} GB", stats.tpu_detail)
            else:
                self.tpu_mem_card.set_value("—", stats.tpu_detail or "No HBM counters")
            self.tpu_card.setToolTip(f"TPU active — {stats.tpu_name or 'TPU'}")
        else:
            self.tpu_card.set_value("Unavailable", "No TPU detected")
            self.tpu_mem_card.set_value("—", "")
            self.tpu_card.setToolTip(
                "TPU unavailable. Coral Edge TPU or a Cloud TPU runtime appears here when present."
            )

        if "compute" in getattr(self, "health_rows", {}):
            if stats.gpu_available:
                compute = "Healthy"
            elif stats.npu_available:
                compute = "NPU"
            elif stats.tpu_available:
                compute = "TPU"
            else:
                compute = "CPU only"
            self.health_rows["compute"].setText(compute)

    def shutdown(self) -> None:
        self._poller.stop()
