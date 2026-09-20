from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QDialog, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from finetuner.ui.shell import BOTTOM_NAV, PRIMARY_NAV


class CommandPalette(QDialog):
    activated = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Command palette")
        self.setModal(True)
        self.resize(520, 420)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search tools, runs, and settings")
        self.search.textChanged.connect(self._filter)
        self.search.returnPressed.connect(self._accept_current)
        self.results = QListWidget()
        self.results.itemActivated.connect(self._emit)
        layout.addWidget(self.search)
        layout.addWidget(self.results, 1)
        self._commands = [
            *( (area, f"Open {label}") for area, label, _icon in PRIMARY_NAV ),
            *( (area, f"Open {label}") for area, label, _icon in BOTTOM_NAV ),
            ("training", "Start training"),
            ("evals", "Run evaluation"),
            ("inference", "Serve model on port 1234"),
        ]
        self._filter("")
        QShortcut(QKeySequence("Escape"), self, self.reject)

    def _filter(self, query: str) -> None:
        needle = query.strip().lower()
        self.results.clear()
        for area, title in self._commands:
            if needle and needle not in title.lower() and needle not in area:
                continue
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, area)
            self.results.addItem(item)
        if self.results.count():
            self.results.setCurrentRow(0)

    def _accept_current(self) -> None:
        item = self.results.currentItem()
        if item:
            self._emit(item)

    def _emit(self, item: QListWidgetItem) -> None:
        self.activated.emit(str(item.data(Qt.ItemDataRole.UserRole)))
        self.accept()

    def open_palette(self) -> None:
        self.search.clear()
        self._filter("")
        self.search.setFocus()
        self.exec()
