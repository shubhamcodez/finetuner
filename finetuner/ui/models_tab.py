from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from finetuner.core.download_worker import DownloadWorker
from finetuner.core.hf_trending import FEATURED_MODELS, HubModel, fetch_trending_models
from finetuner.core.job import ModelJob, ModelSource, ProjectConfig
from finetuner.ui.icons import line_icon
from finetuner.ui.theme import Token


class _TrendingWorker(QThread):
    ready = Signal(object)

    def __init__(self, token: str = "", parent=None) -> None:
        super().__init__(parent)
        self._token = token

    def run(self) -> None:
        self.ready.emit(fetch_trending_models(token=self._token))


class TrendingModelSelect(QFrame):
    """Rounded select that starts empty and lists trending Hub models."""

    model_chosen = Signal(object)
    _PLACEHOLDER = "Select a trending model"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("TrendingSelect")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(40)
        self._models: list[HubModel] = []
        self._selected: HubModel | None = None

        row = QHBoxLayout(self)
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(8)
        self.caption = QLabel(self._PLACEHOLDER)
        self.caption.setObjectName("TrendingSelectLabel")
        self.caption.setProperty("filled", False)
        caption_font = QFont(self.caption.font())
        caption_font.setWeight(QFont.Weight.Normal)
        caption_font.setStyleStrategy(QFont.StyleStrategy.PreferNoHinting)
        self.caption.setFont(caption_font)
        chevron = QLabel()
        chevron.setPixmap(line_icon("chevron-down", Token.TEXT_TERTIARY, 16).pixmap(16, 16))
        row.addWidget(self.caption, 1)
        row.addWidget(chevron, 0, Qt.AlignmentFlag.AlignVCenter)

        self._popup = QFrame(None, Qt.WindowType.Popup)
        self._popup.setObjectName("TrendingPopup")
        self._popup.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        pop = QVBoxLayout(self._popup)
        pop.setContentsMargins(4, 4, 4, 4)
        self._list = QListWidget()
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.itemClicked.connect(self._pick)
        pop.addWidget(self._list)

    def set_models(self, models: list[HubModel] | tuple[HubModel, ...]) -> None:
        current = self._selected.repo_id if self._selected else ""
        self._models = list(models)
        self._list.clear()
        for model in self._models:
            item = QListWidgetItem(f"{model.name}    {model.repo_id}")
            item.setData(Qt.ItemDataRole.UserRole, model)
            self._list.addItem(item)
        if current:
            match = next((model for model in self._models if model.repo_id == current), None)
            if match is not None:
                self._apply(match, emit=False)

    def selected(self) -> HubModel | None:
        return self._selected

    def choose_index(self, index: int) -> None:
        if 0 <= index < len(self._models):
            self._apply(self._models[index])

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self._open()
        super().mousePressEvent(event)

    def _open(self) -> None:
        if self._list.count() == 0:
            return
        width = max(self.width(), 360)
        self._popup.resize(width, min(320, 36 * self._list.count() + 12))
        self._popup.move(self.mapToGlobal(QPoint(0, self.height() + 4)))
        self._popup.show()

    def _pick(self, item: QListWidgetItem) -> None:
        model = item.data(Qt.ItemDataRole.UserRole)
        self._popup.hide()
        if isinstance(model, HubModel):
            self._apply(model)

    def _apply(self, model: HubModel, emit: bool = True) -> None:
        self._selected = model
        self.caption.setText(f"{model.name}  {model.repo_id}")
        self.caption.setProperty("filled", True)
        self.caption.style().unpolish(self.caption)
        self.caption.style().polish(self.caption)
        if emit:
            self.model_chosen.emit(model)


def validate_local_model(path: Path) -> tuple[bool, str]:
    # Compatibility wrapper: validation belongs in core so workers do not import Qt.
    from finetuner.core.model_validation import validate_local_model as validate

    return validate(path)


class AddModelDialog(QDialog):
    def __init__(self, parent=None, token: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Model")
        self.setMinimumWidth(480)
        self._trending_worker: _TrendingWorker | None = None

        layout = QFormLayout(self)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Hugging Face", "Local Path"])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)

        self.name_combo = TrendingModelSelect()
        self.name_combo.setMinimumWidth(320)
        self.name_combo.model_chosen.connect(self._on_hub_model)

        self.identifier_edit = QLineEdit()
        self.identifier_edit.setPlaceholderText("org/model")

        self.browse_btn = QPushButton("Folder...")
        self.browse_btn.clicked.connect(self._browse)
        self.browse_file_btn = QPushButton("GGUF / ONNX...")
        self.browse_file_btn.clicked.connect(self._browse_file)
        id_row = QHBoxLayout()
        id_row.addWidget(self.identifier_edit)
        id_row.addWidget(self.browse_btn)
        id_row.addWidget(self.browse_file_btn)

        layout.addRow("Source", self.source_combo)
        layout.addRow("Name", self.name_combo)
        layout.addRow("Identifier / Path", id_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self._fill_names(FEATURED_MODELS)
        self._on_source_changed(0)
        self._result: ModelJob | None = None
        self._start_trending_fetch(token)

    def _fill_names(self, models: list[HubModel] | tuple[HubModel, ...]) -> None:
        self.name_combo.set_models(models)
        if self.name_combo.selected() is None and self.source_combo.currentIndex() == 0:
            self.identifier_edit.clear()

    def _start_trending_fetch(self, token: str) -> None:
        self._trending_worker = _TrendingWorker(token, self)
        self._trending_worker.ready.connect(self._fill_names)
        self._trending_worker.start()

    def _on_hub_model(self, model: HubModel) -> None:
        if self.source_combo.currentIndex() == 0:
            self.identifier_edit.setText(model.repo_id)

    def _on_source_changed(self, index: int) -> None:
        is_local = index == 1
        self.browse_btn.setVisible(is_local)
        self.browse_file_btn.setVisible(is_local)
        self.name_combo.setEnabled(not is_local)
        if is_local:
            self.identifier_edit.setPlaceholderText("C:\\models\\my-model or model.gguf")
        else:
            self.identifier_edit.setPlaceholderText("org/model or pick a name above")
            chosen = self.name_combo.selected()
            if chosen is not None:
                self._on_hub_model(chosen)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Model Folder")
        if path:
            self.identifier_edit.setText(path)

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Models (*.gguf *.onnx);;All files (*)"
        )
        if path:
            self.identifier_edit.setText(path)

    def validate_and_accept(self) -> None:
        identifier = self.identifier_edit.text().strip()
        chosen = self.name_combo.selected()
        typed = chosen.name if chosen else ""
        is_local = self.source_combo.currentIndex() == 1
        if not is_local and chosen is not None:
            identifier = chosen.repo_id or identifier
        if not identifier:
            QMessageBox.warning(self, "Validation", "Enter a model ID or path.")
            return

        if is_local:
            ok, msg = validate_local_model(Path(identifier))
            if not ok:
                QMessageBox.warning(self, "Validation", msg)
                return
            source = ModelSource.LOCAL
            name = typed or Path(identifier).name
        else:
            source = ModelSource.HUGGINGFACE
            name = typed or identifier.split("/")[-1]

        self._result = ModelJob(name=name, source=source, identifier=identifier)
        self.accept()

    def get_model(self) -> ModelJob | None:
        return self._result


class ModelsTab(QWidget):
    config_changed = Signal()
    model_ready = Signal(object)

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
            "Queued models are the input for training, evaluation, analysis, deployment, and inference. "
            "After a model is on disk, Inferna detects this device and asks whether to optimize before serving on port 1234."
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
                from finetuner.core.model_catalog import find_downloaded_model

                local_path = find_downloaded_model(model.identifier)

            self.table.setItem(row, 0, QTableWidgetItem(model.name))
            self.table.setItem(row, 1, QTableWidgetItem(model.source.value))
            self.table.setItem(row, 2, QTableWidgetItem(model.identifier))
            item = QTableWidgetItem(local_path)
            item.setToolTip(local_path)
            self.table.setItem(row, 3, item)

    def _add_model(self) -> None:
        dialog = AddModelDialog(self, token=self.config.hf_token)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        model = dialog.get_model()
        if model:
            self.config.models.append(model)
            self.refresh_table()
            self.config_changed.emit()
            self.model_ready.emit(model)

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
            QMessageBox.information(
                self, "Download", "Only Hugging Face models can be downloaded here."
            )
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
        self._download_worker.finished_ok.connect(lambda path: self._on_download_done(row, path))
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
        self.model_ready.emit(self.config.models[row])

    def _on_download_failed(self, error: str) -> None:
        self.download_btn.setEnabled(True)
        self.download_progress.setVisible(False)
        self.download_progress.setValue(0)
        self.status_label.setText(f"Download failed: {error}")
        QMessageBox.critical(self, "Download Failed", error)
