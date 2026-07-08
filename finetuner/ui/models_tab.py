from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from finetuner.core.download_worker import DownloadWorker
from finetuner.core.job import ModelJob, ModelSource, ProjectConfig
from finetuner.core.paths import DEFAULT_MODEL_ID, DEFAULT_MODELS


def validate_local_model(path: Path) -> tuple[bool, str]:
    if not path.is_dir():
        return False, "Path is not a directory"
    config = path / "config.json"
    if not config.exists():
        return False, "Missing config.json — not a valid Hugging Face model folder"
    weight_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    if not weight_files:
        return False, "No model weight files (.safetensors or .bin) found"
    return True, ""


class AddModelDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Model")
        self.setMinimumWidth(420)

        layout = QFormLayout(self)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Hugging Face", "Local Path"])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)

        self.identifier_edit = QLineEdit()
        self.identifier_edit.setPlaceholderText(DEFAULT_MODEL_ID)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Display name (optional)")

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse)
        id_row = QHBoxLayout()
        id_row.addWidget(self.identifier_edit)
        id_row.addWidget(self.browse_btn)

        layout.addRow("Source", self.source_combo)
        layout.addRow("Identifier / Path", id_row)
        layout.addRow("Name", self.name_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self._on_source_changed(0)
        self._result: ModelJob | None = None

    def _on_source_changed(self, index: int) -> None:
        is_local = index == 1
        self.browse_btn.setVisible(is_local)
        if is_local:
            self.identifier_edit.setPlaceholderText("C:\\models\\my-model")
        else:
            self.identifier_edit.setPlaceholderText(DEFAULT_MODEL_ID)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Model Folder")
        if path:
            self.identifier_edit.setText(path)

    def validate_and_accept(self) -> None:
        identifier = self.identifier_edit.text().strip()
        if not identifier:
            QMessageBox.warning(self, "Validation", "Enter a model ID or path.")
            return

        is_local = self.source_combo.currentIndex() == 1
        if is_local:
            ok, msg = validate_local_model(Path(identifier))
            if not ok:
                QMessageBox.warning(self, "Validation", msg)
                return
            source = ModelSource.LOCAL
            name = self.name_edit.text().strip() or Path(identifier).name
        else:
            source = ModelSource.HUGGINGFACE
            name = self.name_edit.text().strip() or identifier.split("/")[-1]

        self._result = ModelJob(name=name, source=source, identifier=identifier)
        self.accept()

    def get_model(self) -> ModelJob | None:
        return self._result


class ModelsTab(QWidget):
    config_changed = Signal()

    def __init__(self, config: ProjectConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self._download_worker: DownloadWorker | None = None
        self._build_ui()
        self.refresh_table()

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

        hint = QLabel(
            "Fine-tuning queue — HF or local. Defaults: "
            + ", ".join(name for name, _ in DEFAULT_MODELS)
        )
        hint.setObjectName("HintLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.add_btn = QPushButton("Add Model")
        self.add_btn.setObjectName("PrimaryButton")
        self.add_btn.clicked.connect(self._add_model)
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.setObjectName("SecondaryButton")
        self.remove_btn.clicked.connect(self._remove_selected)
        self.download_btn = QPushButton("Download Selected")
        self.download_btn.clicked.connect(self._download_selected)
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.remove_btn)
        btn_row.addWidget(self.download_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        table_group = QGroupBox("Model Queue")
        table_layout = QVBoxLayout(table_group)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "Source", "Identifier", "Local Path"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setMinimumHeight(100)
        table_layout.addWidget(self.table)
        layout.addWidget(table_group)

        self.download_progress = QProgressBar()
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_progress.setVisible(False)
        layout.addWidget(self.download_progress)

        self.status_label = QLabel("")
        self.status_label.setObjectName("MutedLabel")
        layout.addWidget(self.status_label)
        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def refresh_table(self) -> None:
        self.table.setRowCount(len(self.config.models))
        for row, model in enumerate(self.config.models):
            local_path = ""
            if model.source == ModelSource.LOCAL:
                local_path = model.identifier
            elif model.output_path:
                local_path = model.output_path
            else:
                from finetuner.core.paths import model_download_path

                p = model_download_path(model.identifier)
                if p.exists():
                    local_path = str(p)

            self.table.setItem(row, 0, QTableWidgetItem(model.name))
            self.table.setItem(row, 1, QTableWidgetItem(model.source.value))
            self.table.setItem(row, 2, QTableWidgetItem(model.identifier))
            item = QTableWidgetItem(local_path)
            item.setToolTip(local_path)
            self.table.setItem(row, 3, item)

    def _add_model(self) -> None:
        dialog = AddModelDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        model = dialog.get_model()
        if model:
            self.config.models.append(model)
            self.refresh_table()
            self.config_changed.emit()

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for row in rows:
            del self.config.models[row]
        self.refresh_table()
        self.config_changed.emit()

    def _download_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            QMessageBox.information(self, "Download", "Select a model row first.")
            return
        row = rows[0]
        model = self.config.models[row]
        if model.source != ModelSource.HUGGINGFACE:
            QMessageBox.information(self, "Download", "Only Hugging Face models can be downloaded here.")
            return
        if self._download_worker and self._download_worker.isRunning():
            QMessageBox.information(self, "Download", "A download is already in progress.")
            return

        self.download_btn.setEnabled(False)
        self.download_progress.setVisible(True)
        self.download_progress.setValue(0)
        self.status_label.setText(f"Downloading {model.identifier}...")
        self._download_worker = DownloadWorker(model.identifier, self.config.hf_token, self)
        self._download_worker.progress_text.connect(self.status_label.setText)
        self._download_worker.progress_percent.connect(self._on_download_progress)
        self._download_worker.finished_ok.connect(
            lambda path: self._on_download_done(row, path)
        )
        self._download_worker.failed.connect(self._on_download_failed)
        self._download_worker.start()

    def _on_download_progress(self, percent: int, desc: str) -> None:
        self.download_progress.setValue(percent)
        short = desc if len(desc) <= 80 else f"...{desc[-77:]}"
        self.status_label.setText(f"{percent}% — {short}")

    def _on_download_done(self, row: int, path: str) -> None:
        self.config.models[row].output_path = path
        self.download_btn.setEnabled(True)
        self.download_progress.setValue(100)
        self.download_progress.setVisible(False)
        self.status_label.setText(f"Download complete: {path}")
        self.refresh_table()
        self.config_changed.emit()

    def _on_download_failed(self, error: str) -> None:
        self.download_btn.setEnabled(True)
        self.download_progress.setVisible(False)
        self.download_progress.setValue(0)
        self.status_label.setText(f"Download failed: {error}")
        QMessageBox.critical(self, "Download Failed", error)
