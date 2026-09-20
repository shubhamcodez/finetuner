"""Monochrome line icons, one stroke family."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def line_icon(name: str, color: str = "#A7AFBC", size: int = 18) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    _draw(painter, name, float(size))
    painter.end()
    icon = QIcon()
    icon.addPixmap(pixmap)
    return icon


def _draw(painter: QPainter, name: str, size: float) -> None:
    s = size
    m = s * 0.18
    box = QRectF(m, m, s - 2 * m, s - 2 * m)
    if name == "project":
        painter.drawRoundedRect(box, 2.2, 2.2)
        painter.drawLine(QPointF(box.left() + 3, box.top() + 4), QPointF(box.right() - 3, box.top() + 4))
    elif name == "models":
        painter.drawRoundedRect(QRectF(box.left(), box.top() + 2, box.width() * 0.62, box.height() * 0.72), 2, 2)
        painter.drawRoundedRect(QRectF(box.left() + 5, box.top(), box.width() * 0.62, box.height() * 0.72), 2, 2)
    elif name == "data":
        painter.drawEllipse(QRectF(box.left() + 2, box.top(), box.width() - 4, box.height() * 0.28))
        painter.drawArc(QRectF(box.left() + 2, box.center().y() - 4, box.width() - 4, box.height() * 0.28), 0, 180 * 16)
        painter.drawArc(QRectF(box.left() + 2, box.bottom() - box.height() * 0.28, box.width() - 4, box.height() * 0.28), 0, 180 * 16)
        painter.drawLine(QPointF(box.left() + 2, box.top() + 4), QPointF(box.left() + 2, box.bottom() - 4))
        painter.drawLine(QPointF(box.right() - 2, box.top() + 4), QPointF(box.right() - 2, box.bottom() - 4))
    elif name == "training":
        path = QPainterPath()
        path.moveTo(box.left() + 2, box.top() + 2)
        path.lineTo(box.right() - 1, box.center().y())
        path.lineTo(box.left() + 2, box.bottom() - 2)
        path.closeSubpath()
        painter.drawPath(path)
    elif name == "distillation":
        painter.drawEllipse(QRectF(box.left() + 3, box.top(), 6, 6))
        painter.drawLine(QPointF(box.center().x(), box.top() + 7), QPointF(box.center().x(), box.bottom() - 7))
        painter.drawEllipse(QRectF(box.left() + 4, box.bottom() - 6, 5, 5))
    elif name == "evaluation":
        painter.drawLine(QPointF(box.left(), box.bottom()), QPointF(box.left(), box.top() + 3))
        painter.drawLine(QPointF(box.left(), box.bottom()), QPointF(box.right(), box.bottom()))
        painter.drawLine(QPointF(box.left() + 3, box.bottom() - 3), QPointF(box.left() + 5, box.top() + 4))
        painter.drawLine(QPointF(box.left() + 5, box.top() + 4), QPointF(box.left() + 8, box.bottom() - 6))
        painter.drawLine(QPointF(box.left() + 8, box.bottom() - 6), QPointF(box.right() - 1, box.top() + 2))
    elif name == "analysis":
        painter.drawEllipse(QRectF(box.left() + 1, box.top() + 4, 5, 5))
        painter.drawEllipse(QRectF(box.right() - 6, box.top() + 1, 5, 5))
        painter.drawEllipse(QRectF(box.center().x() - 2, box.bottom() - 6, 5, 5))
        painter.drawLine(QPointF(box.left() + 5, box.top() + 7), QPointF(box.right() - 4, box.top() + 4))
        painter.drawLine(QPointF(box.right() - 4, box.top() + 5), QPointF(box.center().x() + 1, box.bottom() - 4))
    elif name == "deployment":
        painter.drawRoundedRect(box, 2, 2)
        painter.drawLine(QPointF(box.center().x(), box.top() + 3), QPointF(box.center().x(), box.bottom() - 3))
        painter.drawLine(QPointF(box.center().x() - 3, box.bottom() - 6), QPointF(box.center().x(), box.bottom() - 3))
        painter.drawLine(QPointF(box.center().x() + 3, box.bottom() - 6), QPointF(box.center().x(), box.bottom() - 3))
    elif name == "inference":
        painter.drawEllipse(QRectF(box.center().x() - 2, box.center().y() - 2, 4, 4))
        painter.drawEllipse(QRectF(box.left(), box.top(), box.width(), box.height()))
        painter.drawEllipse(QRectF(box.left() + 3, box.top() + 3, box.width() - 6, box.height() - 6))
    elif name == "results":
        painter.drawLine(QPointF(box.left() + 1, box.top() + 3), QPointF(box.right() - 1, box.top() + 3))
        painter.drawLine(QPointF(box.left() + 1, box.center().y()), QPointF(box.right() - 1, box.center().y()))
        painter.drawLine(QPointF(box.left() + 1, box.bottom() - 3), QPointF(box.right() - 1, box.bottom() - 3))
    elif name == "system":
        painter.drawRoundedRect(box, 2, 2)
        painter.drawLine(QPointF(box.left() + 3, box.center().y()), QPointF(box.right() - 3, box.center().y()))
        painter.drawEllipse(QRectF(box.left() + 3, box.top() + 3, 3, 3))
        painter.drawEllipse(QRectF(box.right() - 6, box.bottom() - 6, 3, 3))
    elif name == "docs":
        painter.drawRoundedRect(box, 1.5, 1.5)
        painter.drawLine(QPointF(box.left() + 3, box.top() + 4), QPointF(box.right() - 3, box.top() + 4))
        painter.drawLine(QPointF(box.left() + 3, box.top() + 7), QPointF(box.right() - 3, box.top() + 7))
    elif name == "settings":
        painter.drawEllipse(QRectF(box.center().x() - 2.2, box.center().y() - 2.2, 4.4, 4.4))
        painter.drawEllipse(box)
    elif name == "user":
        painter.drawEllipse(QRectF(box.center().x() - 2.4, box.top() + 1, 4.8, 4.8))
        painter.drawArc(QRectF(box.left() + 1, box.center().y(), box.width() - 2, box.height() * 0.55), 0, 180 * 16)
    elif name == "search":
        painter.drawEllipse(QRectF(box.left() + 1, box.top() + 1, box.width() * 0.62, box.height() * 0.62))
        painter.drawLine(QPointF(box.right() - 2, box.bottom() - 2), QPointF(box.center().x() + 2, box.center().y() + 2))
    elif name == "help":
        painter.drawEllipse(box)
        painter.drawArc(QRectF(box.left() + 4, box.top() + 3, box.width() - 8, 6), 20 * 16, 200 * 16)
        painter.drawPoint(QPointF(box.center().x(), box.bottom() - 3))
    elif name == "chevron":
        painter.drawLine(QPointF(box.left() + 4, box.top() + 3), QPointF(box.right() - 3, box.center().y()))
        painter.drawLine(QPointF(box.right() - 3, box.center().y()), QPointF(box.left() + 4, box.bottom() - 3))
    else:
        painter.drawRoundedRect(box, 2, 2)
