from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QRectF, Signal
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from finetuner.core.job import ProjectConfig
from finetuner.ui.theme import Theme


class RepresentationView(QGraphicsView):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setMinimumHeight(260)
        self.setRenderHint(self.renderHints().Antialiasing, True)

    def set_points(self, points: list[dict]) -> None:
        scene = self.scene()
        scene.clear()
        if not points:
            return
        xs = [float(point["x"]) for point in points]
        ys = [float(point["y"]) for point in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1e-9)
        span_y = max(max_y - min_y, 1e-9)
        width, height, margin = 720.0, 360.0, 24.0
        colors = [Theme.PRIMARY, Theme.SUCCESS, Theme.WARNING, Theme.DANGER, "#C084FC"]
        labels = sorted({str(point.get("label", "unlabeled")) for point in points})
        palette = {label: colors[index % len(colors)] for index, label in enumerate(labels)}
        scene.addLine(
            margin, height - margin, width - margin, height - margin, QPen(QColor(Theme.BORDER))
        )
        scene.addLine(margin, margin, margin, height - margin, QPen(QColor(Theme.BORDER)))
        for point in points:
            x = margin + (float(point["x"]) - min_x) / span_x * (width - 2 * margin)
            y = height - margin - (float(point["y"]) - min_y) / span_y * (height - 2 * margin)
            label = str(point.get("label", "unlabeled"))
            item = scene.addEllipse(
                QRectF(x - 3.5, y - 3.5, 7, 7),
                QPen(QColor(palette[label])),
                QBrush(QColor(palette[label])),
            )
            text = str(point.get("text", "")).replace("\n", " ")[:240]
            item.setToolTip(f"{label}\n{text}")
        scene.setSceneRect(0, 0, width, height)
        self.fitInView(scene.sceneRect())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self.scene().sceneRect().isEmpty():
            self.fitInView(self.scene().sceneRect())


class AnalysisTab(QWidget):
    config_changed = Signal()

    def __init__(self, config: ProjectConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self._artifact: dict | None = None
        self._build_ui()
        self._load_config()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        intro = QLabel(
            "Explore pooled hidden-state geometry by layer, activation norms, attention entropy, "
            "and cross-layer CKA similarity. Labels come from dataset topic/subject metadata when available."
        )
        intro.setObjectName("HintLabel")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        settings = QHBoxLayout()
        form = QFormLayout()
        self.reducer = QComboBox()
        self.reducer.addItem("PCA (fast, deterministic)", "pca")
        self.reducer.addItem("t-SNE", "tsne")
        self.reducer.addItem("UMAP", "umap")
        self.pooling = QComboBox()
        self.pooling.addItem("Mean token", "mean")
        self.pooling.addItem("Last token", "last")
        self.max_samples = QSpinBox()
        self.max_samples.setRange(2, 100_000)
        form.addRow("Projection", self.reducer)
        form.addRow("Pooling", self.pooling)
        form.addRow("Maximum samples", self.max_samples)
        settings.addLayout(form)
        settings.addStretch()
        load_button = QPushButton("Load Analysis…")
        load_button.clicked.connect(self._browse)
        settings.addWidget(load_button)
        layout.addLayout(settings)

        layer_row = QHBoxLayout()
        layer_row.addWidget(QLabel("Layer"))
        self.layer = QComboBox()
        self.layer.currentIndexChanged.connect(self._render_layer)
        layer_row.addWidget(self.layer)
        layer_row.addStretch()
        self.summary = QLabel(
            "Run a workflow containing an Analyze stage, or load representations.json."
        )
        self.summary.setObjectName("MutedLabel")
        layer_row.addWidget(self.summary)
        layout.addLayout(layer_row)
        self.plot = RepresentationView()
        layout.addWidget(self.plot, 1)

        self.reducer.currentIndexChanged.connect(self._sync)
        self.pooling.currentIndexChanged.connect(self._sync)
        self.max_samples.valueChanged.connect(self._sync)

    def _load_config(self) -> None:
        a = self.config.analysis
        self.reducer.setCurrentIndex(max(0, self.reducer.findData(a.reducer)))
        self.pooling.setCurrentIndex(max(0, self.pooling.findData(a.pooling)))
        self.max_samples.setValue(a.max_samples)

    def _sync(self, _value=None) -> None:
        a = self.config.analysis
        a.reducer = self.reducer.currentData() or "pca"
        a.pooling = self.pooling.currentData() or "mean"
        a.max_samples = self.max_samples.value()
        self.config_changed.emit()

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Representation Analysis", "", "JSON (*.json)"
        )
        if path:
            self.set_artifact(path)

    def set_artifact(self, path: str) -> None:
        if not path:
            return
        try:
            artifact = json.loads(Path(path).read_text(encoding="utf-8"))
            if artifact.get("schema_version") != 1 or not artifact.get("layers"):
                raise ValueError("Unsupported or empty analysis artifact")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            self.summary.setText(f"Could not load analysis: {exc}")
            return
        self._artifact = artifact
        self.layer.blockSignals(True)
        self.layer.clear()
        for layer in artifact["layers"]:
            self.layer.addItem(str(layer["layer"]), int(layer["layer"]))
        self.layer.blockSignals(False)
        self._render_layer()

    def _render_layer(self, _index: int = 0) -> None:
        if not self._artifact or self.layer.currentData() is None:
            return
        layer_id = int(self.layer.currentData())
        layer = next(item for item in self._artifact["layers"] if int(item["layer"]) == layer_id)
        self.plot.set_points(layer["points"])
        entropy = layer.get("attention_entropy_mean")
        entropy_text = f" · attention entropy {entropy:.3f}" if entropy is not None else ""
        self.summary.setText(
            f"{len(layer['points'])} samples · activation norm "
            f"{layer['activation_norm_mean']:.3f} ± {layer['activation_norm_std']:.3f}{entropy_text}"
        )
