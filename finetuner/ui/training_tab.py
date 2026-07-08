from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from finetuner.core.job import ProjectConfig
from finetuner.datasets.presets import DATASET_PRESETS, get_preset
from finetuner.eval.tasks import EVAL_TASKS
from finetuner.training.methods import TRAINING_METHODS
from finetuner.training.rewards import REWARD_FUNCTIONS


class TrainingTab(QWidget):
    config_changed = Signal()
    evals_suggest = Signal(list)

    def __init__(self, config: ProjectConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self._block_sync = False
        self._build_ui()
        self._load_from_config()

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

        preset_group = QGroupBox("Ready-made Datasets")
        preset_layout = QVBoxLayout(preset_group)
        preset_layout.setSpacing(4)

        preset_hint = QLabel("Pick a Hugging Face preset aligned with your eval, or use offline sample mode.")
        preset_hint.setObjectName("HintLabel")
        preset_hint.setWordWrap(True)
        preset_layout.addWidget(preset_hint)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Select preset…", "")
        for preset in DATASET_PRESETS.values():
            eval_name = (
                EVAL_TASKS[preset.related_eval_id].name
                if preset.related_eval_id in EVAL_TASKS
                else preset.related_eval_id
            )
            self.preset_combo.addItem(
                f"{preset.name} → {eval_name}",
                preset.preset_id,
            )
            idx = self.preset_combo.count() - 1
            self.preset_combo.setItemData(idx, preset.description, Qt.ItemDataRole.ToolTipRole)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.preset_combo, stretch=1)

        self.clear_preset_btn = QPushButton("Clear")
        self.clear_preset_btn.setObjectName("SecondaryButton")
        self.clear_preset_btn.clicked.connect(self._clear_preset)
        preset_row.addWidget(self.clear_preset_btn)
        preset_layout.addLayout(preset_row)

        self.bundled_only_check = QCheckBox("Use offline sample only (fast, no HF download)")
        self.bundled_only_check.stateChanged.connect(self._sync_config)
        preset_layout.addWidget(self.bundled_only_check)

        self.auto_eval_check = QCheckBox("Auto-enable matching eval when preset is selected")
        self.auto_eval_check.setChecked(True)
        preset_layout.addWidget(self.auto_eval_check)

        self.preset_status = QLabel("No preset selected")
        self.preset_status.setObjectName("MutedLabel")
        self.preset_status.setWordWrap(True)
        preset_layout.addWidget(self.preset_status)

        layout.addWidget(preset_group)

        dataset_group = QGroupBox("Custom Dataset Override")
        dataset_form = QFormLayout(dataset_group)
        dataset_form.setVerticalSpacing(4)
        dataset_form.setContentsMargins(0, 0, 0, 0)

        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.textChanged.connect(self._sync_config)
        browse_btn = QPushButton("Browse JSONL...")
        browse_btn.clicked.connect(self._browse_dataset)
        path_row = QHBoxLayout()
        path_row.addWidget(self.dataset_path_edit)
        path_row.addWidget(browse_btn)
        dataset_form.addRow("Local JSONL/JSON", path_row)

        self.dataset_hf_edit = QLineEdit()
        self.dataset_hf_edit.setPlaceholderText("e.g. tatsu-lab/alpaca (optional)")
        self.dataset_hf_edit.textChanged.connect(self._sync_config)
        dataset_form.addRow("HF Dataset ID", self.dataset_hf_edit)

        layout.addWidget(dataset_group)

        method_group = QGroupBox("Optimization & Rewards")
        method_grid = QGridLayout(method_group)
        method_grid.setVerticalSpacing(4)
        method_grid.setHorizontalSpacing(12)
        method_grid.setContentsMargins(0, 0, 0, 0)

        self.method_combo = QComboBox()
        for spec in TRAINING_METHODS.values():
            self.method_combo.addItem(spec.name, spec.method_id)
            idx = self.method_combo.count() - 1
            self.method_combo.setItemData(idx, spec.description, Qt.ItemDataRole.ToolTipRole)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_grid.addWidget(QLabel("Method"), 0, 0)
        method_grid.addWidget(self.method_combo, 0, 1, 1, 3)

        self.method_hint = QLabel("")
        self.method_hint.setObjectName("HintLabel")
        self.method_hint.setWordWrap(True)
        method_grid.addWidget(self.method_hint, 1, 0, 1, 4)

        self.reward_combo = QComboBox()
        for spec in REWARD_FUNCTIONS.values():
            self.reward_combo.addItem(spec.name, spec.reward_id)
            idx = self.reward_combo.count() - 1
            self.reward_combo.setItemData(idx, spec.description, Qt.ItemDataRole.ToolTipRole)
        self.reward_combo.currentIndexChanged.connect(self._on_reward_changed)
        method_grid.addWidget(QLabel("Reward fn"), 2, 0)
        method_grid.addWidget(self.reward_combo, 2, 1)

        self.reward_model_edit = QLineEdit()
        self.reward_model_edit.setPlaceholderText("HF reward model (PPO / HF reward fn)")
        self.reward_model_edit.textChanged.connect(self._sync_config)
        method_grid.addWidget(QLabel("Reward model"), 2, 2)
        method_grid.addWidget(self.reward_model_edit, 2, 3)

        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setDecimals(3)
        self.beta_spin.setRange(0.001, 2.0)
        self.beta_spin.setSingleStep(0.01)
        self.beta_spin.valueChanged.connect(self._sync_config)
        method_grid.addWidget(QLabel("Beta (DPO/KTO)"), 3, 0)
        method_grid.addWidget(self.beta_spin, 3, 1)

        self.num_gen_spin = QSpinBox()
        self.num_gen_spin.setRange(1, 16)
        self.num_gen_spin.valueChanged.connect(self._sync_config)
        method_grid.addWidget(QLabel("GRPO gens"), 3, 2)
        method_grid.addWidget(self.num_gen_spin, 3, 3)

        self.kl_spin = QDoubleSpinBox()
        self.kl_spin.setDecimals(3)
        self.kl_spin.setRange(0.0, 1.0)
        self.kl_spin.setSingleStep(0.01)
        self.kl_spin.valueChanged.connect(self._sync_config)
        method_grid.addWidget(QLabel("PPO KL coef"), 4, 0)
        method_grid.addWidget(self.kl_spin, 4, 1)

        self.clip_spin = QDoubleSpinBox()
        self.clip_spin.setDecimals(2)
        self.clip_spin.setRange(0.05, 0.5)
        self.clip_spin.setSingleStep(0.05)
        self.clip_spin.valueChanged.connect(self._sync_config)
        method_grid.addWidget(QLabel("PPO clip"), 4, 2)
        method_grid.addWidget(self.clip_spin, 4, 3)

        layout.addWidget(method_group)

        params_group = QGroupBox("Training Hyperparameters")
        params_grid = QGridLayout(params_group)
        params_grid.setVerticalSpacing(4)
        params_grid.setHorizontalSpacing(12)
        params_grid.setContentsMargins(0, 0, 0, 0)

        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(10, 100000)
        self.max_steps_spin.valueChanged.connect(self._sync_config)
        params_grid.addWidget(QLabel("Max steps"), 0, 0)
        params_grid.addWidget(self.max_steps_spin, 0, 1)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(1e-6, 1e-2)
        self.lr_spin.setSingleStep(1e-5)
        self.lr_spin.valueChanged.connect(self._sync_config)
        params_grid.addWidget(QLabel("Learning rate"), 0, 2)
        params_grid.addWidget(self.lr_spin, 0, 3)

        self.lora_rank_spin = QSpinBox()
        self.lora_rank_spin.setRange(4, 128)
        self.lora_rank_spin.valueChanged.connect(self._sync_config)
        params_grid.addWidget(QLabel("LoRA rank"), 1, 0)
        params_grid.addWidget(self.lora_rank_spin, 1, 1)

        self.lora_alpha_spin = QSpinBox()
        self.lora_alpha_spin.setRange(8, 256)
        self.lora_alpha_spin.valueChanged.connect(self._sync_config)
        params_grid.addWidget(QLabel("LoRA alpha"), 1, 2)
        params_grid.addWidget(self.lora_alpha_spin, 1, 3)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.valueChanged.connect(self._sync_config)
        params_grid.addWidget(QLabel("Batch size"), 2, 0)
        params_grid.addWidget(self.batch_spin, 2, 1)

        self.grad_accum_spin = QSpinBox()
        self.grad_accum_spin.setRange(1, 64)
        self.grad_accum_spin.valueChanged.connect(self._sync_config)
        params_grid.addWidget(QLabel("Grad accumulation"), 2, 2)
        params_grid.addWidget(self.grad_accum_spin, 2, 3)

        self.max_seq_spin = QSpinBox()
        self.max_seq_spin.setRange(256, 8192)
        self.max_seq_spin.setSingleStep(256)
        self.max_seq_spin.valueChanged.connect(self._sync_config)
        params_grid.addWidget(QLabel("Max seq length"), 3, 0)
        params_grid.addWidget(self.max_seq_spin, 3, 1)

        self.qlora_check = QCheckBox("Use QLoRA (4-bit)")
        self.qlora_check.stateChanged.connect(self._sync_config)
        params_grid.addWidget(self.qlora_check, 3, 2, 1, 2)

        layout.addWidget(params_group)

        settings_group = QGroupBox("Hugging Face Token (for gated models)")
        settings_form = QFormLayout(settings_group)
        settings_form.setVerticalSpacing(4)
        settings_form.setContentsMargins(0, 0, 0, 0)
        self.hf_token_edit = QLineEdit()
        self.hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_edit.textChanged.connect(self._sync_config)
        settings_form.addRow("HF Token", self.hf_token_edit)
        layout.addWidget(settings_group)

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _load_from_config(self) -> None:
        self._block_sync = True
        t = self.config.training
        self.dataset_path_edit.setText(t.dataset_path)
        self.dataset_hf_edit.setText(t.dataset_hf_id)
        self.bundled_only_check.setChecked(t.dataset_use_bundled_only)
        self.max_steps_spin.setValue(t.max_steps)
        self.lr_spin.setValue(t.learning_rate)
        self.lora_rank_spin.setValue(t.lora_rank)
        self.lora_alpha_spin.setValue(t.lora_alpha)
        self.batch_spin.setValue(t.batch_size)
        self.grad_accum_spin.setValue(t.gradient_accumulation_steps)
        self.max_seq_spin.setValue(t.max_seq_length)
        self.qlora_check.setChecked(t.use_qlora)
        self.hf_token_edit.setText(self.config.hf_token)

        method_idx = max(0, self.method_combo.findData(t.training_method))
        self.method_combo.setCurrentIndex(method_idx)
        reward_idx = max(0, self.reward_combo.findData(t.reward_function))
        self.reward_combo.setCurrentIndex(reward_idx)
        self.reward_model_edit.setText(t.reward_model_id)
        self.beta_spin.setValue(t.dpo_beta)
        self.num_gen_spin.setValue(t.grpo_num_generations)
        self.kl_spin.setValue(t.ppo_kl_coef)
        self.clip_spin.setValue(t.ppo_cliprange)

        self._update_method_hint()
        self._update_method_fields()
        self._update_preset_status()
        self._sync_preset_combo()
        self._block_sync = False

    def _on_reward_changed(self, _index: int = 0) -> None:
        self._update_method_fields()
        self._sync_config()

    def _on_method_changed(self, _index: int = 0) -> None:
        self._update_method_hint()
        self._update_method_fields()
        self._sync_config()

    def _update_method_hint(self) -> None:
        method_id = self.method_combo.currentData() or "sft"
        spec = TRAINING_METHODS.get(method_id)
        self.method_hint.setText(spec.description if spec else "")

    def _update_method_fields(self) -> None:
        method_id = self.method_combo.currentData() or "sft"
        uses_reward = method_id in ("grpo", "ppo")
        uses_reward_model = method_id == "ppo"
        uses_beta = method_id in ("dpo", "kto")
        uses_grpo = method_id == "grpo"
        uses_ppo = method_id == "ppo"

        self.reward_combo.setEnabled(uses_reward)
        self.reward_model_edit.setEnabled(uses_reward_model or self.reward_combo.currentData() == "hf_reward_model")
        self.beta_spin.setEnabled(uses_beta)
        self.num_gen_spin.setEnabled(uses_grpo)
        self.kl_spin.setEnabled(uses_ppo)
        self.clip_spin.setEnabled(uses_ppo)

    def _sync_preset_combo(self) -> None:
        preset_id = self.config.training.dataset_preset_id
        if not preset_id:
            self.preset_combo.setCurrentIndex(0)
            return
        for i in range(self.preset_combo.count()):
            if self.preset_combo.itemData(i) == preset_id:
                self.preset_combo.setCurrentIndex(i)
                return
        self.preset_combo.setCurrentIndex(0)

    def _update_preset_status(self) -> None:
        preset_id = self.config.training.dataset_preset_id
        if not preset_id:
            self.preset_status.setText("No preset selected.")
            return
        preset = get_preset(preset_id)
        if not preset:
            self.preset_status.setText(f"Unknown preset: {preset_id}")
            return
        eval_name = EVAL_TASKS[preset.related_eval_id].name if preset.related_eval_id in EVAL_TASKS else preset.related_eval_id
        mode = "offline" if self.config.training.dataset_use_bundled_only else "HF"
        self.preset_status.setText(
            f"Active: {preset.name} → {eval_name} · {mode}"
        )

    def _on_preset_changed(self, index: int) -> None:
        if self._block_sync:
            return
        preset_id = self.preset_combo.itemData(index) or ""
        if not preset_id:
            self.config.training.dataset_preset_id = ""
            self._update_preset_status()
            self._sync_config()
            return
        self._apply_preset(preset_id)

    def _apply_preset(self, preset_id: str) -> None:
        preset = get_preset(preset_id)
        if not preset:
            return

        self._block_sync = True
        self.config.training.dataset_preset_id = preset_id
        self.config.training.dataset_path = ""
        self.config.training.dataset_hf_id = ""
        self.dataset_path_edit.clear()
        self.dataset_hf_edit.clear()
        self._block_sync = False

        if self.auto_eval_check.isChecked() and preset.related_eval_id:
            if preset.related_eval_id not in self.config.enabled_evals:
                self.config.enabled_evals.append(preset.related_eval_id)
            self.evals_suggest.emit(list(self.config.enabled_evals))

        self._update_preset_status()
        self._sync_config()

    def _clear_preset(self) -> None:
        self._block_sync = True
        self.config.training.dataset_preset_id = ""
        self.preset_combo.setCurrentIndex(0)
        self._block_sync = False
        self._update_preset_status()
        self._sync_config()

    def _browse_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            "",
            "Data Files (*.jsonl *.json);;All Files (*)",
        )
        if path:
            self._block_sync = True
            self.config.training.dataset_preset_id = ""
            self.preset_combo.setCurrentIndex(0)
            self._block_sync = False
            self._update_preset_status()
            self.dataset_path_edit.setText(path)
            self._sync_config()

    def _sync_config(self) -> None:
        if self._block_sync:
            return
        t = self.config.training
        t.dataset_path = self.dataset_path_edit.text().strip()
        t.dataset_hf_id = self.dataset_hf_edit.text().strip()
        t.dataset_use_bundled_only = self.bundled_only_check.isChecked()
        if t.dataset_path or t.dataset_hf_id:
            t.dataset_preset_id = ""
            self._block_sync = True
            self.preset_combo.setCurrentIndex(0)
            self._block_sync = False
            self._update_preset_status()
        t.max_steps = self.max_steps_spin.value()
        t.learning_rate = self.lr_spin.value()
        t.lora_rank = self.lora_rank_spin.value()
        t.lora_alpha = self.lora_alpha_spin.value()
        t.batch_size = self.batch_spin.value()
        t.gradient_accumulation_steps = self.grad_accum_spin.value()
        t.max_seq_length = self.max_seq_spin.value()
        t.use_qlora = self.qlora_check.isChecked()
        t.training_method = self.method_combo.currentData() or "sft"
        t.reward_function = self.reward_combo.currentData() or "exact_match"
        t.reward_model_id = self.reward_model_edit.text().strip()
        t.dpo_beta = self.beta_spin.value()
        t.grpo_num_generations = self.num_gen_spin.value()
        t.ppo_kl_coef = self.kl_spin.value()
        t.ppo_cliprange = self.clip_spin.value()
        self.config.hf_token = self.hf_token_edit.text().strip()
        self._update_method_fields()
        self.config_changed.emit()

    def apply_eval_selection(self, eval_ids: list[str]) -> None:
        self.config.enabled_evals = list(eval_ids)
