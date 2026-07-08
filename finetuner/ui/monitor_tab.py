from __future__ import annotations

from collections import deque

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from finetuner.monitor.stats import StatsPoller, SystemStats
from finetuner.ui.theme import Theme, chart_colors


class MetricCard(QGroupBox):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(title, parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(6, 8, 6, 6)
        self.value_label = QLabel("--")
        self.value_label.setObjectName("MetricValue")
        self.detail_label = QLabel("")
        self.detail_label.setObjectName("MetricDetail")
        layout.addWidget(self.value_label)
        layout.addWidget(self.detail_label)

    def set_value(self, value: str, detail: str = "") -> None:
        self.value_label.setText(value)
        self.detail_label.setText(detail)


class HistoryChart(QGroupBox):
    def __init__(self, title: str, y_max: float = 100.0, parent=None) -> None:
        super().__init__(title, parent)
        colors = chart_colors()
        self._series = QLineSeries()
        self._history: deque[float] = deque(maxlen=60)

        chart = QChart()
        chart.addSeries(self._series)
        chart.legend().hide()
        chart.setBackgroundVisible(False)
        chart.setBackgroundBrush(QColor(colors.get("background", Theme.SURFACE)))
        chart.setPlotAreaBackgroundVisible(True)
        chart.setPlotAreaBackgroundBrush(QColor(Theme.SURFACE_ALT))

        axis_x = QValueAxis()
        axis_x.setRange(0, 60)
        axis_x.setLabelFormat("%d")
        axis_x.setTitleText("Time (s)")
        axis_x.setLabelsColor(QColor(colors["label"]))
        axis_x.setTitleBrush(QColor(colors["label"]))
        axis_x.setGridLineColor(QColor(colors["grid"]))

        axis_y = QValueAxis()
        axis_y.setRange(0, y_max)
        axis_y.setTitleText("Utilization %")
        axis_y.setLabelsColor(QColor(colors["label"]))
        axis_y.setTitleBrush(QColor(colors["label"]))
        axis_y.setGridLineColor(QColor(colors["grid"]))

        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        self._series.attachAxis(axis_x)
        self._series.attachAxis(axis_y)

        pen = QPen(QColor(colors["line"]))
        pen.setWidthF(2.5)
        self._series.setPen(pen)

        self._view = QChartView(chart)
        self._view.setMinimumHeight(130)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(0)
        layout.addWidget(self._view)

    def append(self, value: float) -> None:
        self._history.append(value)
        self._series.clear()
        for i, v in enumerate(self._history):
            self._series.append(i, v)


class MonitorTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._poller = StatsPoller(interval_ms=1000, parent=self)
        self._poller.stats_updated.connect(self._on_stats)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        intro = QLabel("Live CPU, RAM, and GPU metrics.")
        intro.setObjectName("HintLabel")
        layout.addWidget(intro)

        cards = QGridLayout()
        cards.setSpacing(8)
        self.cpu_card = MetricCard("CPU Utilization")
        self.ram_card = MetricCard("System Memory")
        self.gpu_card = MetricCard("GPU Utilization")
        self.vram_card = MetricCard("GPU Memory")
        cards.addWidget(self.cpu_card, 0, 0)
        cards.addWidget(self.ram_card, 0, 1)
        cards.addWidget(self.gpu_card, 1, 0)
        cards.addWidget(self.vram_card, 1, 1)
        layout.addLayout(cards)

        charts = QHBoxLayout()
        charts.setSpacing(8)
        self.cpu_chart = HistoryChart("CPU History", y_max=100)
        self.gpu_chart = HistoryChart("GPU History", y_max=100)
        charts.addWidget(self.cpu_chart)
        charts.addWidget(self.gpu_chart)
        layout.addLayout(charts)

        self.gpu_status = QLabel("")
        self.gpu_status.setObjectName("MutedLabel")
        self.gpu_status.setWordWrap(True)
        layout.addWidget(self.gpu_status)
        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

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
            self.gpu_status.setText(f"NVIDIA GPU active — {stats.gpu_name}")
        else:
            self.gpu_card.set_value("Unavailable", "No NVIDIA GPU detected")
            self.vram_card.set_value("—", "")
            self.gpu_status.setText(
                "GPU unavailable. Install NVIDIA drivers to enable CUDA fine-tuning."
            )

    def shutdown(self) -> None:
        self._poller.stop()
