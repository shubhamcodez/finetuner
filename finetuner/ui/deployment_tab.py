from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from finetuner.core.job import ProjectConfig
from finetuner.quantization.planner import detect_hardware
from finetuner.quantization.specs import DeviceTarget, backend_specs, get_backend_spec
from finetuner.ui.pipeline_context import PipelineContextBar


class DeploymentTab(QWidget):
    config_changed = Signal()

    def __init__(self, config: ProjectConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self._build_ui()
        self._load_config()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        self.pipeline_context = PipelineContextBar()
        layout.addWidget(self.pipeline_context)
        intro = QLabel("Create a quantized artifact matched to an inference backend and device.")
        intro.setObjectName("HintLabel")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setVerticalSpacing(4)
        self.form = form
        self.backend = QComboBox()
        for spec in backend_specs():
            self.backend.addItem(spec.name, spec.backend.value)
        self.target = QComboBox()
        for target in DeviceTarget:
            if target != DeviceTarget.AUTO:
                self.target.addItem(target.value.replace("_", " ").title(), target.value)
        self.bits = QComboBox()
        self.group_size = QSpinBox()
        self.group_size.setRange(16, 1024)
        self.group_size.setSingleStep(16)
        self.calibration = QLineEdit()
        self.calibration.setPlaceholderText("Required for AWQ; optional for data-aware compression")
        browse_calibration = QPushButton("Browse")
        browse_calibration.clicked.connect(self._browse_calibration)
        calibration_row = QHBoxLayout()
        calibration_row.addWidget(self.calibration)
        calibration_row.addWidget(browse_calibration)
        self.llama_path = QLineEdit()
        self.llama_path.setPlaceholderText("llama.cpp checkout/build directory for GGUF")
        browse_llama = QPushButton("Browse")
        browse_llama.clicked.connect(self._browse_llama)
        llama_row = QHBoxLayout()
        llama_row.addWidget(self.llama_path)
        llama_row.addWidget(browse_llama)
        form.addRow("Backend", self.backend)
        form.addRow("Device target", self.target)
        form.addRow("Weight bits", self.bits)
        form.addRow("Group size", self.group_size)
        form.addRow("Calibration data", calibration_row)
        form.addRow("llama.cpp path", llama_row)
        self.calibration_row = calibration_row
        self.llama_row = llama_row
        layout.addLayout(form)

        row = QHBoxLayout()
        detect = QPushButton("Detect This Device")
        detect.clicked.connect(self._detect)
        row.addWidget(detect)
        row.addStretch()
        layout.addLayout(row)
        self.status = QLabel("")
        self.status.setObjectName("MutedLabel")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch()

        self.backend.currentIndexChanged.connect(self._backend_changed)
        self.target.currentIndexChanged.connect(self._sync)
        self.bits.currentIndexChanged.connect(self._sync)
        self.group_size.valueChanged.connect(self._sync)
        self.calibration.textChanged.connect(self._sync)
        self.llama_path.textChanged.connect(self._sync)

    def _load_config(self) -> None:
        q = self.config.quantization
        self.backend.setCurrentIndex(max(0, self.backend.findData(q.backend)))
        self._backend_changed()
        self.target.setCurrentIndex(max(0, self.target.findData(q.target)))
        self.bits.setCurrentIndex(max(0, self.bits.findData(q.bits)))
        self.group_size.setValue(q.group_size)
        self.calibration.setText(q.calibration_dataset)
        self.llama_path.setText(q.llama_cpp_path)
        self._sync()

    def _backend_changed(self, _index: int = 0) -> None:
        spec = get_backend_spec(self.backend.currentData())
        current = self.bits.currentData()
        self.bits.blockSignals(True)
        self.bits.clear()
        for bits in spec.supported_bits:
            self.bits.addItem(f"{bits}-bit", bits)
        desired = self.config.quantization.bits if current is None else current
        self.bits.setCurrentIndex(max(0, self.bits.findData(desired)))
        self.bits.blockSignals(False)
        self.form.setRowVisible(self.calibration_row, spec.backend.value == "awq")
        self.form.setRowVisible(self.llama_row, spec.backend.value == "gguf")
        self._sync()

    def _sync(self, _value=None) -> None:
        q = self.config.quantization
        q.backend = self.backend.currentData() or "gguf"
        q.target = self.target.currentData() or "cpu"
        q.bits = int(self.bits.currentData() or 4)
        q.group_size = self.group_size.value()
        q.calibration_dataset = self.calibration.text().strip()
        q.llama_cpp_path = self.llama_path.text().strip()
        errors = q.validate()
        spec = get_backend_spec(q.backend)
        self.status.setText(
            "; ".join(errors) if errors else f"Ready: {spec.name} -> {q.target.replace('_', ' ')}"
        )
        self.status.setToolTip(spec.description)
        self.config_changed.emit()

    def _browse_calibration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Calibration Dataset")
        if path:
            self.calibration.setText(path)

    def _browse_llama(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "llama.cpp Directory")
        if path:
            self.llama_path.setText(path)

    def _detect(self) -> None:
        capabilities = detect_hardware()
        available = [item.target.value.replace("_", " ") for item in capabilities if item.available]
        self.status.setText("Detected: " + ", ".join(available))
