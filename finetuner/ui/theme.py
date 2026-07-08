"""Matte black theme for Finetuner."""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication


class Theme:
    # Matte black palette
    BG = "#0C0C0E"
    SURFACE = "#161618"
    SURFACE_ALT = "#1E1E22"
    SURFACE_RAISED = "#242428"
    BORDER = "#2C2C32"
    BORDER_STRONG = "#3A3A42"
    TEXT = "#ECECF0"
    TEXT_SECONDARY = "#9898A6"
    TEXT_MUTED = "#6E6E7A"
    PRIMARY = "#6B9FFF"
    PRIMARY_HOVER = "#5590F7"
    PRIMARY_LIGHT = "#1A2740"
    PRIMARY_BORDER = "#3D5A8C"
    SUCCESS = "#5EEAD4"
    SUCCESS_BG = "#122820"
    SUCCESS_BORDER = "#1E4038"
    SUCCESS_TEXT = "#8FF0D4"
    WARNING = "#FBBF24"
    DANGER = "#F87171"
    ACCENT = "#6B9FFF"
    CHART_LINE = "#6B9FFF"
    CHART_GRID = "#2C2C32"
    LOG_BG = "#08080A"
    LOG_TEXT = "#B4B4BE"
    FONT_FAMILY = "Segoe UI"


STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {Theme.BG};
    color: {Theme.TEXT};
    font-family: "{Theme.FONT_FAMILY}", "Inter", sans-serif;
    font-size: 12px;
}}

#AppHeader {{
    background-color: {Theme.SURFACE};
    border-bottom: 1px solid {Theme.BORDER};
}}
#AppLogo {{
    background: transparent;
    padding: 0;
    margin: 0;
}}
#AppTitle {{
    font-size: 15px;
    font-weight: 600;
    color: {Theme.TEXT};
    padding: 0;
    margin: 0;
}}
#AppSubtitle {{
    font-size: 11px;
    color: {Theme.TEXT_SECONDARY};
    padding: 0;
    margin: 0;
}}
#StatusBadge {{
    background-color: {Theme.SURFACE_ALT};
    border: 1px solid {Theme.BORDER_STRONG};
    border-radius: 10px;
    padding: 3px 10px;
    color: {Theme.TEXT_SECONDARY};
    font-size: 11px;
    font-weight: 500;
}}

QTabWidget::pane {{
    border: 1px solid {Theme.BORDER};
    border-radius: 6px;
    background: {Theme.SURFACE};
    top: -1px;
    padding: 2px;
}}
QTabBar::tab {{
    background: transparent;
    color: {Theme.TEXT_MUTED};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 6px 14px;
    margin-right: 2px;
    font-weight: 500;
    min-width: 72px;
}}
QTabBar::tab:selected {{
    color: {Theme.PRIMARY};
    border-bottom: 2px solid {Theme.PRIMARY};
    background: transparent;
}}
QTabBar::tab:hover:!selected {{
    color: {Theme.TEXT};
    background: {Theme.SURFACE_ALT};
    border-radius: 6px 6px 0 0;
}}

QGroupBox {{
    background-color: {Theme.SURFACE};
    border: 1px solid {Theme.BORDER};
    border-radius: 8px;
    margin-top: 8px;
    padding: 10px 10px 8px 10px;
    font-weight: 600;
    color: {Theme.TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: {Theme.TEXT_SECONDARY};
    font-size: 11px;
    font-weight: 600;
}}

QPushButton {{
    background-color: {Theme.SURFACE_ALT};
    color: {Theme.TEXT};
    border: 1px solid {Theme.BORDER_STRONG};
    border-radius: 6px;
    padding: 4px 12px;
    font-weight: 500;
    min-height: 14px;
}}
QPushButton:hover {{
    background-color: {Theme.SURFACE_RAISED};
    border-color: {Theme.TEXT_MUTED};
}}
QPushButton:pressed {{
    background-color: {Theme.BORDER};
}}
QPushButton:disabled {{
    color: {Theme.TEXT_MUTED};
    background-color: {Theme.SURFACE};
    border-color: {Theme.BORDER};
}}
QPushButton#PrimaryButton {{
    background-color: {Theme.PRIMARY};
    color: #0C0C0E;
    border: 1px solid {Theme.PRIMARY};
    font-weight: 600;
}}
QPushButton#PrimaryButton:hover {{
    background-color: {Theme.PRIMARY_HOVER};
    border-color: {Theme.PRIMARY_HOVER};
    color: #0C0C0E;
}}
QPushButton#PrimaryButton:disabled {{
    background-color: {Theme.PRIMARY_BORDER};
    border-color: {Theme.PRIMARY_BORDER};
    color: {Theme.TEXT_MUTED};
}}
QPushButton#SecondaryButton {{
    background-color: transparent;
    color: {Theme.TEXT_SECONDARY};
    border: 1px solid {Theme.BORDER_STRONG};
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
    background-color: {Theme.SURFACE_ALT};
    border: 1px solid {Theme.BORDER_STRONG};
    border-radius: 6px;
    padding: 4px 8px;
    color: {Theme.TEXT};
    min-height: 14px;
    selection-background-color: {Theme.PRIMARY_LIGHT};
    selection-color: {Theme.TEXT};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{
    border: 1px solid {Theme.PRIMARY};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background-color: {Theme.SURFACE_ALT};
    color: {Theme.TEXT};
    border: 1px solid {Theme.BORDER_STRONG};
    selection-background-color: {Theme.PRIMARY_LIGHT};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    border: none;
    background: transparent;
    width: 18px;
}}

QListWidget {{
    background-color: {Theme.SURFACE_ALT};
    border: 1px solid {Theme.BORDER};
    border-radius: 6px;
    padding: 2px;
    outline: none;
}}
QListWidget::item {{
    border-radius: 4px;
    padding: 4px 8px;
    margin: 1px 0;
    color: {Theme.TEXT};
}}
QListWidget::item:selected {{
    background-color: {Theme.PRIMARY_LIGHT};
    color: {Theme.PRIMARY};
    border: 1px solid {Theme.PRIMARY_BORDER};
}}
QListWidget::item:hover:!selected {{
    background-color: {Theme.SURFACE_RAISED};
}}

QTableWidget {{
    background-color: {Theme.SURFACE_ALT};
    border: 1px solid {Theme.BORDER};
    border-radius: 8px;
    gridline-color: {Theme.BORDER};
    selection-background-color: {Theme.PRIMARY_LIGHT};
    selection-color: {Theme.PRIMARY};
    alternate-background-color: {Theme.SURFACE};
    outline: none;
}}
QTableWidget::item {{
    padding: 4px 6px;
    color: {Theme.TEXT};
}}
QHeaderView::section {{
    background-color: {Theme.SURFACE};
    color: {Theme.TEXT_SECONDARY};
    border: none;
    border-bottom: 1px solid {Theme.BORDER};
    border-right: 1px solid {Theme.BORDER};
    padding: 5px 6px;
    font-weight: 600;
    font-size: 11px;
}}

QCheckBox {{
    spacing: 6px;
    color: {Theme.TEXT};
}}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border-radius: 4px;
    border: 1px solid {Theme.BORDER_STRONG};
    background: {Theme.SURFACE_ALT};
}}
QCheckBox::indicator:checked {{
    background-color: {Theme.PRIMARY};
    border-color: {Theme.PRIMARY};
}}

QProgressBar {{
    background-color: {Theme.SURFACE_ALT};
    border: 1px solid {Theme.BORDER};
    border-radius: 4px;
    height: 8px;
    max-height: 8px;
    text-align: center;
    color: {Theme.TEXT_SECONDARY};
    font-size: 10px;
}}
QProgressBar::chunk {{
    background-color: {Theme.PRIMARY};
    border-radius: 5px;
}}

#LogConsole {{
    background-color: {Theme.LOG_BG};
    color: {Theme.LOG_TEXT};
    border: 1px solid {Theme.BORDER};
    border-radius: 6px;
    padding: 6px;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 11px;
    selection-background-color: {Theme.PRIMARY_LIGHT};
}}
#LogPanel {{
    background-color: {Theme.SURFACE};
    border: 1px solid {Theme.BORDER};
    border-radius: 8px;
}}
#LogPanelTitle {{
    font-size: 11px;
    font-weight: 600;
    color: {Theme.TEXT_MUTED};
    letter-spacing: 0.3px;
}}

QLabel {{
    color: {Theme.TEXT};
    background: transparent;
}}
#HintLabel, #MutedLabel {{
    color: {Theme.TEXT_SECONDARY};
    font-size: 11px;
}}
#MetricValue {{
    font-size: 20px;
    font-weight: 600;
    color: {Theme.TEXT};
}}
#MetricDetail {{
    font-size: 11px;
    color: {Theme.TEXT_SECONDARY};
}}
#SummaryBanner {{
    background-color: {Theme.SUCCESS_BG};
    border: 1px solid {Theme.SUCCESS_BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    color: {Theme.SUCCESS_TEXT};
    font-weight: 500;
    font-size: 11px;
}}

QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 4px;
}}
QScrollBar::handle:vertical {{
    background: {Theme.BORDER_STRONG};
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {Theme.TEXT_MUTED};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
}}
QScrollBar::handle:horizontal {{
    background: {Theme.BORDER_STRONG};
    border-radius: 5px;
}}

QSplitter::handle {{
    background: {Theme.BORDER};
    height: 3px;
}}
QSplitter::handle:hover {{
    background: {Theme.PRIMARY};
}}

QDialog {{
    background-color: {Theme.SURFACE};
}}
QMessageBox {{
    background-color: {Theme.SURFACE};
}}
QToolTip {{
    background-color: {Theme.SURFACE_RAISED};
    color: {Theme.TEXT};
    border: 1px solid {Theme.BORDER_STRONG};
    padding: 6px 8px;
}}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)

    font = QFont(Theme.FONT_FAMILY, 9)
    font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    app.setFont(font)

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(Theme.BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(Theme.TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(Theme.SURFACE_ALT))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(Theme.SURFACE))
    palette.setColor(QPalette.ColorRole.Text, QColor(Theme.TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(Theme.SURFACE_ALT))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(Theme.TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(Theme.PRIMARY))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#0C0C0E"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(Theme.TEXT_MUTED))
    app.setPalette(palette)


def chart_colors() -> dict[str, str]:
    return {
        "line": Theme.CHART_LINE,
        "grid": Theme.CHART_GRID,
        "label": Theme.TEXT_SECONDARY,
        "background": Theme.SURFACE,
    }
