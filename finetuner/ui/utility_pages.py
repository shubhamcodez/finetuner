from __future__ import annotations

from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from finetuner.core.job import ProjectConfig


class DocsPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 32)
        layout.setSpacing(16)
        for title, body in (
            (
                "Load a model",
                "Add a Hugging Face repo or a local GGUF/ONNX folder. When weights are on disk, "
                "Inferna detects this machine and asks whether to optimize before serving on port 1234.",
            ),
            (
                "Run one tool at a time",
                "Finetune, distillation, evaluation, analysis, deployment, and inference are independent. "
                "They share the model queue. They do not wait on a DAG.",
            ),
            (
                "Serve",
                "The live endpoint is http://127.0.0.1:1234. Optimize picks the strongest ready engine. "
                "Skip optimize to serve the artifact as-is.",
            ),
        ):
            card = QFrame()
            card.setObjectName("SurfaceCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(20, 20, 20, 20)
            heading = QLabel(title)
            heading.setObjectName("CardTitle")
            copy = QLabel(body)
            copy.setObjectName("HintLabel")
            copy.setWordWrap(True)
            card_layout.addWidget(heading)
            card_layout.addWidget(copy)
            layout.addWidget(card)
        layout.addStretch()


class SettingsPage(QWidget):
    def __init__(self, config: ProjectConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 32)
        card = QFrame()
        card.setObjectName("SurfaceCard")
        form = QFormLayout(card)
        form.setContentsMargins(20, 20, 20, 20)
        form.setVerticalSpacing(12)
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("HF_TOKEN or leave empty")
        self.token.setText(config.hf_token)
        self.token.textChanged.connect(self._sync)
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(config.inference.serve_port)
        self.port.valueChanged.connect(self._sync)
        form.addRow("Hugging Face token", self.token)
        form.addRow("Serve port", self.port)
        hint = QLabel("The token stays in memory or HF_TOKEN. It is not written to config.json.")
        hint.setObjectName("MetaLabel")
        hint.setWordWrap(True)
        form.addRow("", hint)
        layout.addWidget(card)
        layout.addStretch()

    def _sync(self) -> None:
        self.config.hf_token = self.token.text().strip()
        self.config.inference.serve_port = self.port.value()
