"""Compact device picker: availability, memory, and the current choice."""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from finetuner.inference.devices import DeviceMemory
from finetuner.quantization.specs import DeviceTarget
from finetuner.ui.icons import line_icon
from finetuner.ui.theme import Token

_ROWS = (
    (DeviceTarget.CPU.value, "CPU", "cpu"),
    (DeviceTarget.NVIDIA_GPU.value, "Nvidia GPU", "nvidia"),
    (DeviceTarget.AMD_GPU.value, "AMD GPU", "amd"),
    (DeviceTarget.INTEL_GPU.value, "Intel GPU", "intel"),
    (DeviceTarget.APPLE_GPU.value, "Apple GPU", "apple"),
    (DeviceTarget.INTEL_NPU.value, "Intel NPU", "intel"),
    (DeviceTarget.QUALCOMM_NPU.value, "Qualcomm NPU", "qualcomm"),
)


def _gb(value: float) -> str:
    rounded = round(value, 1)
    if abs(rounded - round(rounded)) < 0.05:
        return f"{round(rounded):.0f}"
    return f"{rounded:.1f}"


def _device_icon(kind: str) -> QPixmap:
    if kind == "cpu":
        return line_icon("cpu", "#C5CDD8", 16).pixmap(16, 16)
    if kind == "apple":
        return line_icon("apple", "#E8E8ED", 16).pixmap(16, 16)
    if kind == "intel":
        pixmap = QPixmap(30, 16)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#0071C5"), 1.1))
        painter.drawRoundedRect(0, 1, 29, 14, 7, 7)
        painter.setPen(QColor("#0071C5"))
        font = QFont(painter.font())
        font.setPixelSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "intel")
        painter.end()
        return pixmap
    pixmap = QPixmap(18, 18)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    if kind == "nvidia":
        painter.setPen(QPen(QColor("#76B900"), 1.4))
        painter.drawEllipse(1, 4, 15, 10)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#76B900"))
        painter.drawEllipse(7, 6, 5, 5)
    elif kind == "amd":
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ED1C24"))
        path = QPainterPath()
        path.moveTo(2, 2)
        path.lineTo(15, 9)
        path.lineTo(2, 16)
        path.lineTo(6.5, 9)
        path.closeSubpath()
        painter.drawPath(path)
    else:
        painter.setPen(QColor("#7C9BFF"))
        font = QFont(painter.font())
        font.setPixelSize(15)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "Q")
    painter.end()
    return pixmap


class MemoryBar(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._fraction = 0.0
        self._low = False
        self._active = False
        self.setFixedSize(72, 6)

    def set_fraction(self, fraction: float | None) -> None:
        if fraction is None:
            self._active = False
            self.update()
            return
        self._active = True
        self._fraction = max(0.0, min(1.0, fraction))
        self._low = self._fraction < 0.4
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        if not self._active:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#2A3140"))
        painter.drawRoundedRect(self.rect(), 3, 3)
        width = max(4, int(self.width() * self._fraction)) if self._fraction > 0 else 0
        if width:
            painter.setBrush(QColor(Token.WARNING if self._low else "#3DDC97"))
            painter.drawRoundedRect(0, 0, width, self.height(), 3, 3)
        painter.end()


class StatusPill(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._available = False
        self.setFixedSize(92, 20)

    def set_available(self, available: bool) -> None:
        self._available = available
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._available:
            background, foreground, text = QColor("#143028"), QColor("#3DDC97"), "Available"
        else:
            background, foreground, text = QColor("#1A2028"), QColor(Token.TEXT_TERTIARY), "Not found"
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(background)
        painter.drawRoundedRect(self.rect(), 9, 9)
        painter.setBrush(foreground)
        painter.drawEllipse(8, 7, 6, 6)
        painter.setPen(QPen(foreground))
        painter.drawText(self.rect().adjusted(18, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter, text)
        painter.end()


class _DeviceRow(QFrame):
    chosen = Signal(str)

    def __init__(self, value: str, title: str, icon_kind: str, parent=None) -> None:
        super().__init__(parent)
        self.value = value
        self.title = title
        self.setObjectName("DeviceRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(36)

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 4, 10, 4)
        row.setSpacing(10)
        icon = QLabel()
        icon.setPixmap(_device_icon(icon_kind))
        icon.setFixedWidth(32)
        name = QLabel(title)
        name.setObjectName("DeviceName")
        self.pill = StatusPill()
        self.memory = QLabel("—")
        self.memory.setObjectName("DeviceMemory")
        self.memory.setProperty("missing", True)
        self.memory.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.memory.setFixedWidth(118)
        self.bar = MemoryBar()
        self.check = QLabel()
        self.check.setPixmap(line_icon("check", Token.ACCENT, 14).pixmap(14, 14))
        self.check.setFixedWidth(16)
        self.check.setVisible(False)
        row.addWidget(icon)
        row.addWidget(name, 1)
        row.addWidget(self.pill)
        row.addWidget(self.memory)
        row.addWidget(self.bar)
        row.addWidget(self.check)
        for child in (icon, name, self.pill, self.memory, self.bar, self.check):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self.check.setVisible(selected)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_memory(self, state: DeviceMemory) -> None:
        self.pill.set_available(state.available)
        if not state.available or state.free_gb is None or state.total_gb is None or state.total_gb <= 0:
            self.memory.setText("—")
            self.memory.setProperty("missing", True)
            self.bar.set_fraction(None)
        else:
            self.memory.setText(f"{_gb(state.free_gb)} GB / {_gb(state.total_gb)} GB")
            self.memory.setProperty("missing", False)
            self.bar.set_fraction(state.free_gb / state.total_gb)
        self.memory.style().unpolish(self.memory)
        self.memory.style().polish(self.memory)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.chosen.emit(self.value)
        super().mouseReleaseEvent(event)


class _AutoRow(QFrame):
    chosen = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("DeviceRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(52)
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 10, 6)
        row.setSpacing(10)
        spark = QLabel()
        spark.setPixmap(line_icon("sparkle", Token.TEXT_SECONDARY, 16).pixmap(16, 16))
        spark.setFixedWidth(32)
        text = QVBoxLayout()
        text.setSpacing(0)
        title = QLabel("Auto")
        title.setObjectName("DeviceName")
        subtitle = QLabel("Automatically choose the best available device")
        subtitle.setObjectName("DeviceSubtitle")
        text.addWidget(title)
        text.addWidget(subtitle)
        badge = QLabel("Recommended")
        badge.setObjectName("RecommendedBadge")
        badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.check = QLabel()
        self.check.setPixmap(line_icon("check", Token.ACCENT, 14).pixmap(14, 14))
        self.check.setFixedWidth(16)
        self.check.setVisible(False)
        row.addWidget(spark)
        row.addLayout(text, 1)
        row.addWidget(badge)
        row.addWidget(self.check)
        for child in (spark, title, subtitle, badge, self.check):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self.check.setVisible(selected)
        self.style().unpolish(self)
        self.style().polish(self)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.chosen.emit()
        super().mouseReleaseEvent(event)


class _DevicePanel(QFrame):
    """Popup contents for the device menu."""

    selected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("DevicePanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self._value = DeviceTarget.AUTO.value
        self._rows: list[_DeviceRow] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(2)

        self._auto = _AutoRow()
        self._auto.chosen.connect(lambda: self._choose(DeviceTarget.AUTO.value, notify=True))
        layout.addWidget(self._auto)

        divider = QFrame()
        divider.setObjectName("DeviceDivider")
        divider.setFixedHeight(1)
        divider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(divider)

        for value, title, icon_kind in _ROWS:
            row = _DeviceRow(value, title, icon_kind)
            row.chosen.connect(lambda chosen: self._choose(chosen, notify=True))
            self._rows.append(row)
            layout.addWidget(row)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 90))
        self.setGraphicsEffect(shadow)
        self._refresh_selection()

    def currentData(self) -> str:
        return self._value

    def count(self) -> int:
        return 1 + len(self._rows)

    def findData(self, value: str) -> int:
        if value == DeviceTarget.AUTO.value:
            return 0
        for index, row in enumerate(self._rows, start=1):
            if row.value == value:
                return index
        return -1

    def itemText(self, index: int) -> str:
        if index == 0:
            return "Auto"
        if 1 <= index <= len(self._rows):
            return self._rows[index - 1].title
        return ""

    def setCurrentIndex(self, index: int) -> None:
        if index <= 0:
            self._choose(DeviceTarget.AUTO.value, notify=False)
            return
        if index <= len(self._rows):
            self._choose(self._rows[index - 1].value, notify=False)

    def set_memory(self, snapshot: dict[str, DeviceMemory]) -> None:
        for row in self._rows:
            row.set_memory(snapshot.get(row.value, DeviceMemory(False)))

    def _choose(self, value: str, *, notify: bool) -> None:
        changed = value != self._value
        self._value = value
        self._refresh_selection()
        if notify:
            if changed:
                self.selected.emit(value)
            self.hide()

    def _refresh_selection(self) -> None:
        selected_auto = self._value == DeviceTarget.AUTO.value
        self._auto.set_selected(selected_auto)
        for row in self._rows:
            row.set_selected(row.value == self._value)


class DeviceSelector(QWidget):
    """Device menu. The closed control matches the other form fields; the list opens underneath."""

    currentIndexChanged = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._panel = _DevicePanel(self)
        self._panel.selected.connect(self._on_selected)

        self._face = QFrame()
        self._face.setObjectName("DeviceMenu")
        self._face.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._face.setCursor(Qt.CursorShape.PointingHandCursor)
        self._face.setFixedHeight(40)
        face = QHBoxLayout(self._face)
        face.setContentsMargins(10, 0, 10, 0)
        face.setSpacing(8)
        self._label = QLabel("Auto")
        self._label.setObjectName("DeviceMenuLabel")
        arrow = QLabel()
        arrow.setPixmap(line_icon("chevron-down", Token.TEXT_SECONDARY, 12).pixmap(12, 12))
        face.addWidget(self._label, 1)
        face.addWidget(arrow)
        for child in (self._label, arrow):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._face.mouseReleaseEvent = self._open_from_face  # type: ignore[method-assign]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._face)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def currentData(self) -> str:
        return self._panel.currentData()

    def count(self) -> int:
        return self._panel.count()

    def findData(self, value: str) -> int:
        return self._panel.findData(value)

    def itemText(self, index: int) -> str:
        return self._panel.itemText(index)

    def setCurrentIndex(self, index: int) -> None:
        before = self.currentData()
        self._panel.setCurrentIndex(index)
        self._label.setText(self.itemText(max(0, self.findData(self.currentData()))))
        if self.currentData() != before and not self.signalsBlocked():
            self.currentIndexChanged.emit(self.findData(self.currentData()))

    def set_memory(self, snapshot: dict[str, DeviceMemory]) -> None:
        self._panel.set_memory(snapshot)

    def _open_from_face(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._open()
        QFrame.mouseReleaseEvent(self._face, event)

    def _open(self) -> None:
        width = max(self.width(), 560)
        self._panel.setFixedWidth(width)
        self._panel.adjustSize()
        self._panel.move(self.mapToGlobal(QPoint(0, self.height() + 4)))
        self._panel.show()

    def _on_selected(self, value: str) -> None:
        self._label.setText(self.itemText(max(0, self.findData(value))))
        if not self.signalsBlocked():
            self.currentIndexChanged.emit(self.findData(value))
