from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
from finetuner.inference.devices import (
    apply_best_runtime,
    apply_device_recipe,
    launch_targets,
    recipe_for_target,
)
from finetuner.inference.planner import (
    backend_engine_compatibility_error,
    detect_inference_hardware,
    recommend_for_target,
)
from finetuner.inference.specs import InferenceEngine, engine_specs, get_engine_spec
from finetuner.quantization.specs import DeviceTarget
from finetuner.ui.tool_run import ToolRunBar


class InferenceTab(QWidget):
    config_changed = Signal()
    run_requested = Signal()
    quantization_changed = Signal()
    serve_requested = Signal(bool)
    stop_requested = Signal()

    def __init__(self, config: ProjectConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self._build_ui()
        self._load_config()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        self.run_bar = ToolRunBar("Run best engine for this machine")
        self.run_bar.run_requested.connect(self._run_best)
        layout.addWidget(self.run_bar)
        intro = QLabel(
            "Load a model and Finetuner detects this device, then asks whether to "
            "optimize before serving at http://127.0.0.1:1234. "
            "One adaptive engine picks the strongest specialist: "
            "vLLM or llama.cpp CUDA on NVIDIA, llama.cpp on AMD/Apple/CPU, "
            "OpenVINO on Intel NPU/GPU, and QNN/HTP only when the artifact is a "
            "Hexagon context-binary graph. You can also serve without optimizing."
        )
        intro.setObjectName("HintLabel")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setVerticalSpacing(4)
        self.form = form
        self.engine = QComboBox()
        for spec in engine_specs():
            self.engine.addItem(spec.name, spec.engine.value)
        self.target = QComboBox()
        self.target.addItem("Best for this machine", DeviceTarget.AUTO.value)
        for target in DeviceTarget:
            if target != DeviceTarget.AUTO:
                self.target.addItem(target.value.replace("_", " ").title(), target.value)
        self.kv_cache = QComboBox()
        self.max_context = QSpinBox()
        self.max_context.setRange(256, 1_048_576)
        self.max_context.setSingleStep(256)
        self.max_batch = QSpinBox()
        self.max_batch.setRange(1, 4096)
        self.tensor_parallel = QSpinBox()
        self.tensor_parallel.setRange(1, 256)
        self.gpu_memory = QDoubleSpinBox()
        self.gpu_memory.setRange(0.1, 1.0)
        self.gpu_memory.setSingleStep(0.05)
        self.gpu_memory.setDecimals(2)
        self.speculative = QSpinBox()
        self.speculative.setRange(0, 32)
        self.prefix_caching = QCheckBox("Prefix / prompt cache")
        self.cuda_graphs = QCheckBox("CUDA graphs")
        self.flash_attention = QCheckBox("Flash attention")
        self.compile = QCheckBox("Compile or cache an engine artifact")
        self.serve_port = QSpinBox()
        self.serve_port.setRange(1, 65535)
        self.serve_port.setValue(1234)
        flags = QHBoxLayout()
        flags.addWidget(self.prefix_caching)
        flags.addWidget(self.cuda_graphs)
        flags.addWidget(self.flash_attention)
        flags.addStretch()
        self.toolchain = QLineEdit()
        self.toolchain.setPlaceholderText("Optional llama.cpp / TensorRT-LLM / TGI checkout")
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_toolchain)
        toolchain_row = QHBoxLayout()
        toolchain_row.addWidget(self.toolchain)
        toolchain_row.addWidget(browse)
        self.toolchain_row = toolchain_row
        form.addRow("Engine", self.engine)
        form.addRow("Device target", self.target)
        form.addRow("KV-cache dtype", self.kv_cache)
        form.addRow("Max context", self.max_context)
        form.addRow("Max batch", self.max_batch)
        form.addRow("Tensor parallel", self.tensor_parallel)
        form.addRow("GPU memory fraction", self.gpu_memory)
        form.addRow("Speculative tokens", self.speculative)
        form.addRow("Runtime flags", flags)
        form.addRow("Compile engine", self.compile)
        form.addRow("Serve port", self.serve_port)
        form.addRow("Toolchain path", toolchain_row)
        layout.addLayout(form)

        devices = QHBoxLayout()
        self.device_buttons: dict[str, QPushButton] = {}
        for target in launch_targets():
            recipe = recipe_for_target(target)
            if target in {DeviceTarget.QUALCOMM_NPU, DeviceTarget.INTEL_NPU}:
                label = "Run on NPU"
            else:
                label = f"Run on {recipe.label}"
            button = QPushButton(label)
            button.setObjectName("PrimaryButton" if recipe.available else "SecondaryButton")
            button.setToolTip(recipe.detail)
            button.clicked.connect(lambda _checked=False, chosen=target: self._run_on_device(chosen))
            self.device_buttons[target.value] = button
            devices.addWidget(button)
        devices.addStretch()
        layout.addLayout(devices)

        serve_row = QHBoxLayout()
        self.serve_plain_btn = QPushButton("Serve without optimizing")
        self.serve_plain_btn.clicked.connect(lambda: self.serve_requested.emit(False))
        self.stop_serve_btn = QPushButton("Stop server")
        self.stop_serve_btn.clicked.connect(self.stop_requested.emit)
        serve_row.addWidget(self.serve_plain_btn)
        serve_row.addWidget(self.stop_serve_btn)
        serve_row.addStretch()
        layout.addLayout(serve_row)
        self.serve_status = QLabel("Not serving. After a model is loaded it will be offered on port 1234.")
        self.serve_status.setObjectName("HintLabel")
        self.serve_status.setWordWrap(True)
        self.serve_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.serve_status)

        row = QHBoxLayout()
        detect = QPushButton("Detect This Device")
        detect.clicked.connect(self._detect)
        recommend = QPushButton("Recommend")
        recommend.clicked.connect(self._recommend)
        load_plan = QPushButton("Load Plan...")
        load_plan.clicked.connect(self._browse_plan)
        row.addWidget(detect)
        row.addWidget(recommend)
        row.addWidget(load_plan)
        row.addStretch()
        layout.addLayout(row)

        self.status = QLabel("")
        self.status.setObjectName("MutedLabel")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.plan_preview = QLabel(
            "Run inference optimization on queued models, or load inference_plan.json."
        )
        self.plan_preview.setObjectName("HintLabel")
        self.plan_preview.setWordWrap(True)
        layout.addWidget(self.plan_preview)
        layout.addStretch()

        self.engine.currentIndexChanged.connect(self._engine_changed)
        self.target.currentIndexChanged.connect(self._sync)
        self.kv_cache.currentIndexChanged.connect(self._sync)
        self.max_context.valueChanged.connect(self._sync)
        self.max_batch.valueChanged.connect(self._sync)
        self.tensor_parallel.valueChanged.connect(self._sync)
        self.gpu_memory.valueChanged.connect(self._sync)
        self.speculative.valueChanged.connect(self._sync)
        self.prefix_caching.toggled.connect(self._sync)
        self.cuda_graphs.toggled.connect(self._sync)
        self.flash_attention.toggled.connect(self._sync)
        self.compile.toggled.connect(self._sync)
        self.serve_port.valueChanged.connect(self._sync)
        self.toolchain.textChanged.connect(self._sync)

    def _load_config(self) -> None:
        cfg = self.config.inference
        self.engine.setCurrentIndex(max(0, self.engine.findData(cfg.engine)))
        self._engine_changed()
        self.target.setCurrentIndex(max(0, self.target.findData(cfg.target)))
        self.kv_cache.setCurrentIndex(max(0, self.kv_cache.findData(cfg.kv_cache_dtype)))
        self.max_context.setValue(cfg.max_context)
        self.max_batch.setValue(cfg.max_batch_size)
        self.tensor_parallel.setValue(cfg.tensor_parallel)
        self.gpu_memory.setValue(cfg.gpu_memory_utilization)
        self.speculative.setValue(cfg.speculative_tokens)
        self.prefix_caching.setChecked(cfg.prefix_caching)
        self.cuda_graphs.setChecked(cfg.cuda_graphs)
        self.flash_attention.setChecked(cfg.flash_attention)
        self.compile.setChecked(cfg.compile)
        self.serve_port.setValue(cfg.serve_port or 1234)
        self.toolchain.setText(cfg.toolchain_path)
        self._sync()

    def _engine_changed(self, _index: int = 0) -> None:
        spec = get_engine_spec(self.engine.currentData() or InferenceEngine.LLAMACPP.value)
        current = self.kv_cache.currentData()
        self.kv_cache.blockSignals(True)
        self.kv_cache.clear()
        for dtype in spec.kv_cache_dtypes:
            self.kv_cache.addItem(dtype.value, dtype.value)
        desired = self.config.inference.kv_cache_dtype if current is None else current
        self.kv_cache.setCurrentIndex(max(0, self.kv_cache.findData(desired)))
        self.kv_cache.blockSignals(False)
        self.compile.setEnabled(spec.compiles_ahead_of_time)
        if not spec.compiles_ahead_of_time:
            self.compile.setChecked(False)
        self._sync()

    def _sync(self, _value=None) -> None:
        cfg = self.config.inference
        cfg.engine = self.engine.currentData() or InferenceEngine.LLAMACPP.value
        cfg.target = self.target.currentData() or DeviceTarget.CPU.value
        cfg.kv_cache_dtype = self.kv_cache.currentData() or "auto"
        cfg.max_context = self.max_context.value()
        cfg.max_batch_size = self.max_batch.value()
        cfg.tensor_parallel = self.tensor_parallel.value()
        cfg.gpu_memory_utilization = self.gpu_memory.value()
        cfg.speculative_tokens = self.speculative.value()
        cfg.prefix_caching = self.prefix_caching.isChecked()
        cfg.cuda_graphs = self.cuda_graphs.isChecked()
        cfg.flash_attention = self.flash_attention.isChecked()
        cfg.compile = self.compile.isChecked()
        cfg.serve_port = self.serve_port.value()
        cfg.toolchain_path = self.toolchain.text().strip()
        errors = cfg.validate()
        mismatch = backend_engine_compatibility_error(self.config.quantization.backend, cfg.engine)
        if mismatch:
            errors.append(mismatch)
        spec = get_engine_spec(cfg.engine)
        if errors:
            self.status.setText("; ".join(errors))
        else:
            mode = "compiled engine" if spec.compiles_ahead_of_time and cfg.compile else "serve plan"
            self.status.setText(
                f"Ready: {spec.name} -> {cfg.target.replace('_', ' ')} ({mode}, "
                f"ctx {cfg.max_context}, batch {cfg.max_batch_size}, port {cfg.serve_port})"
            )
        self.status.setToolTip(spec.description)
        self.config_changed.emit()

    def _browse_toolchain(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Inference Toolchain Directory")
        if path:
            self.toolchain.setText(path)

    def _browse_plan(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Inference Engine Plan", "", "JSON (*.json)"
        )
        if path:
            self.set_artifact(path)

    def _detect(self) -> None:
        capabilities = detect_inference_hardware()
        available = [item.target.value.replace("_", " ") for item in capabilities if item.available]
        self.status.setText("Detected: " + ", ".join(available or ["CPU only"]))
        for target in launch_targets(capabilities):
            button = self.device_buttons.get(target.value)
            if button is None:
                continue
            capability = next((item for item in capabilities if item.target == target), None)
            present = bool(capability and capability.available)
            button.setObjectName("PrimaryButton" if present else "SecondaryButton")
            button.setToolTip(capability.detail if capability else "")
            button.style().unpolish(button)
            button.style().polish(button)

    def _run_best(self) -> None:
        choice = apply_best_runtime(self.config, run_on_device=True)
        self._load_config()
        self.quantization_changed.emit()
        skipped = f" Skipped: {'; '.join(choice.skipped)}." if choice.skipped else ""
        self.status.setText(
            f"{choice.reason} ({choice.recipe.inference.engine} / "
            f"{choice.recipe.quantization.backend}).{skipped}"
        )
        self.run_requested.emit()

    def _run_on_device(self, target: DeviceTarget) -> None:
        recipe = apply_device_recipe(self.config, target, run_on_device=True)
        self._load_config()
        self.quantization_changed.emit()
        presence = "present" if recipe.available else "not detected on this machine"
        self.status.setText(
            f"Running {recipe.label} recipe ({recipe.inference.engine} / "
            f"{recipe.quantization.backend}, {presence}: {recipe.detail})"
        )
        self.run_requested.emit()

    def _recommend(self) -> None:
        target = DeviceTarget(self.target.currentData() or DeviceTarget.AUTO.value)
        if target == DeviceTarget.AUTO:
            choice = apply_best_runtime(self.config, run_on_device=False)
            self._load_config()
            self.quantization_changed.emit()
            skipped = f" Skipped: {'; '.join(choice.skipped)}." if choice.skipped else ""
            self.status.setText(f"{choice.reason}.{skipped}")
            return
        try:
            recommendation = recommend_for_target(target)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        recommended = recommendation.config
        self.config.inference = recommended
        self._load_config()
        self.status.setText(recommendation.rationale + " | " + " ".join(recommendation.notes[:1]))

    def set_artifact(self, path: str) -> None:
        if not path:
            return
        plan_path = Path(path)
        if plan_path.is_dir():
            plan_path = plan_path / "inference_plan.json"
        try:
            artifact = json.loads(plan_path.read_text(encoding="utf-8"))
            if artifact.get("schema_version") != 1 or not artifact.get("engine"):
                raise ValueError("Unsupported or empty inference plan")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            self.plan_preview.setText(f"Could not load inference plan: {exc}")
            return
        serve = artifact.get("serve_command") or []
        serve_text = " ".join(str(part) for part in serve) if serve else "in-process runtime options"
        compiled = "compiled" if artifact.get("compiled") else "plan only"
        bind = artifact.get("device_bind") or {}
        bind_text = ""
        if bind:
            state = "bound" if bind.get("bound") else "not bound"
            bind_text = f" | {state}: {bind.get('detail', '')}"
        self.plan_preview.setText(
            f"{artifact.get('engine_name', artifact['engine'])} | "
            f"{artifact.get('target', '')} | {compiled} | {serve_text}{bind_text}"
        )
        self.plan_preview.setToolTip(plan_path.as_posix())

    def set_serve_status(self, url: str = "", detail: str = "") -> None:
        if url:
            extra = f" — {detail}" if detail else ""
            self.serve_status.setText(f"Serving at {url}{extra}")
        else:
            self.serve_status.setText(
                detail or "Not serving. After a model is loaded it will be offered on port 1234."
            )
