from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QHeaderView,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from finetuner.core.job import ModelRunResult
from finetuner.eval.tasks import EVAL_TASKS
from finetuner.ui.theme import Theme


class ResultsTab(QWidget):
    artifact_requested = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._results: list[ModelRunResult] = []
        self._build_ui()

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

        self.summary_frame = QFrame()
        self.summary_frame.setObjectName("SummaryBanner")
        summary_layout = QVBoxLayout(self.summary_frame)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        self.summary_label = QLabel("Run training to see comparison results.")
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        layout.addWidget(self.summary_frame)

        self.table = QTableWidget(0, 0)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setMinimumHeight(120)
        self.table.itemDoubleClicked.connect(self._open_artifact)
        layout.addWidget(self.table)
        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def set_results(self, results: list[ModelRunResult]) -> None:
        self._results = results
        self._refresh()

    def add_result(self, result: ModelRunResult) -> None:
        self._results.append(result)
        self._refresh()

    def _refresh(self) -> None:
        if not self._results:
            self.summary_label.setText("Run training to see comparison results.")
            self.summary_frame.setVisible(True)
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        task_ids: list[str] = []
        for r in self._results:
            for e in r.eval_results:
                if e.task_id not in task_ids:
                    task_ids.append(e.task_id)

        headers = ["Model"] + [EVAL_TASKS[t].name if t in EVAL_TASKS else t for t in task_ids]
        headers += ["Policy", "Analysis", "Deployment"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(self._results))

        best_per_task: dict[str, tuple[int, float]] = {}
        for task_id in task_ids:
            best_score = -1.0
            best_row = -1
            for row, result in enumerate(self._results):
                for e in result.eval_results:
                    if e.task_id == task_id and e.score > best_score:
                        best_score = e.score
                        best_row = row
            if best_row >= 0:
                best_per_task[task_id] = (best_row, best_score)

        winner_bg = QColor(Theme.SUCCESS_BG)
        winner_fg = QColor(Theme.SUCCESS_TEXT)

        for row, result in enumerate(self._results):
            name_item = QTableWidgetItem(result.model_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if result.training_error:
                name_item.setToolTip(result.training_error)
            self.table.setItem(row, 0, name_item)

            for col, task_id in enumerate(task_ids, start=1):
                score = None
                for e in result.eval_results:
                    if e.task_id == task_id:
                        score = e.score
                        break
                text = (
                    f"{score:.1f}%"
                    if score is not None
                    else ("ERR" if result.training_error else "-")
                )
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if task_id in best_per_task and best_per_task[task_id][0] == row:
                    item.setBackground(winner_bg)
                    item.setForeground(winner_fg)
                    font = item.font()
                    font.setWeight(QFont.Weight.DemiBold)
                    item.setFont(font)
                self.table.setItem(row, col, item)

            artifact_start = 1 + len(task_ids)
            artifacts = (
                ("models", result.output_path),
                ("analysis", result.analysis_path),
                ("deployment", result.deployment_path),
            )
            for offset, (area, path) in enumerate(artifacts):
                item = QTableWidgetItem("Ready" if path else "-")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if path:
                    item.setToolTip(f"Double-click to open\n{path}")
                    item.setData(Qt.ItemDataRole.UserRole, (area, path))
                self.table.setItem(row, artifact_start + offset, item)

        summary_lines = []
        for task_id, (row, score) in best_per_task.items():
            name = EVAL_TASKS[task_id].name if task_id in EVAL_TASKS else task_id
            model_name = self._results[row].model_name
            summary_lines.append(f"Best for {name}: {model_name} ({score:.1f}%)")

        if summary_lines:
            self.summary_label.setText("\n".join(summary_lines))
            self.summary_frame.setVisible(True)
        else:
            self.summary_label.setText("Run finished; evaluation scores are unavailable.")
            self.summary_frame.setVisible(True)

    def _open_artifact(self, item: QTableWidgetItem) -> None:
        destination = item.data(Qt.ItemDataRole.UserRole)
        if destination:
            area, path = destination
            self.artifact_requested.emit(area, path)
