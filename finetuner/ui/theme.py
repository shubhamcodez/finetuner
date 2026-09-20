"""Design tokens and stylesheet for the research workbench."""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

from finetuner.ui.fonts import application_font, load_application_fonts


class Token:
    BG = "#0B0D10"
    SIDEBAR = "#101319"
    SURFACE = "#12161C"
    SURFACE_2 = "#161B22"
    HOVER = "#1A2029"
    INPUT = "#0F1318"
    BORDER = "#252B35"
    BORDER_SUBTLE = "#1D232C"
    BORDER_INPUT = "#29303A"
    TEXT = "#F4F6F8"
    TEXT_SECONDARY = "#A7AFBC"
    TEXT_TERTIARY = "#707986"
    TEXT_DISABLED = "#555D68"
    ACCENT = "#5B8DEF"
    ACCENT_HOVER = "#6C9AFF"
    ACCENT_PRESSED = "#497BDD"
    ACCENT_SUBTLE = "rgba(91, 141, 239, 0.10)"
    ACCENT_BORDER = "rgba(91, 141, 239, 0.28)"
    SUCCESS = "#36B37E"
    WARNING = "#E7A93B"
    ERROR = "#E05D68"
    RADIUS_INPUT = "8px"
    RADIUS_BUTTON = "8px"
    RADIUS_CARD = "10px"
    RADIUS_PANEL = "12px"
    SPACE_1 = "4px"
    SPACE_2 = "8px"
    SPACE_3 = "12px"
    SPACE_4 = "16px"
    SPACE_5 = "24px"
    SPACE_6 = "32px"
    FONT = "Inter"


# Back-compat names used by charts and older widgets.
class Theme:
    BG = Token.BG
    SURFACE = Token.SURFACE
    SURFACE_ALT = Token.SURFACE_2
    SURFACE_RAISED = Token.HOVER
    BORDER = Token.BORDER
    BORDER_STRONG = Token.BORDER
    TEXT = Token.TEXT
    TEXT_SECONDARY = Token.TEXT_SECONDARY
    TEXT_MUTED = Token.TEXT_TERTIARY
    PRIMARY = Token.ACCENT
    PRIMARY_HOVER = Token.ACCENT_HOVER
    PRIMARY_LIGHT = Token.ACCENT_SUBTLE
    PRIMARY_BORDER = Token.ACCENT_BORDER
    SUCCESS = Token.SUCCESS
    SUCCESS_BG = Token.ACCENT_SUBTLE
    SUCCESS_BORDER = Token.BORDER
    SUCCESS_TEXT = Token.TEXT_SECONDARY
    WARNING = Token.WARNING
    DANGER = Token.ERROR
    ACCENT = Token.ACCENT
    CHART_LINE = Token.ACCENT
    CHART_GRID = Token.BORDER
    LOG_BG = Token.INPUT
    LOG_TEXT = Token.TEXT_SECONDARY
    FONT_FAMILY = Token.FONT


def _sheet(font: str) -> str:
    t = Token
    return f"""
QMainWindow, QDialog, QMessageBox, #AppShell {{
    background-color: {t.BG};
    color: {t.TEXT};
    font-family: "{font}";
    font-size: 14px;
    font-weight: 400;
}}
QLabel {{
    background-color: transparent;
    color: {t.TEXT};
}}
QStackedWidget {{
    background-color: {t.BG};
}}

#AppShell {{
    background-color: {t.BG};
}}
#Sidebar {{
    background-color: {t.SIDEBAR};
    border-right: 1px solid {t.BORDER_SUBTLE};
}}
#SidebarBrand {{
    background-color: transparent;
    color: {t.TEXT};
    font-size: 15px;
    font-weight: 400;
}}
#SidebarCaption {{
    background-color: transparent;
    color: {t.TEXT_TERTIARY};
    font-size: 12px;
}}
QPushButton#NavItem {{
    background: transparent;
    color: {t.TEXT_SECONDARY};
    border: none;
    border-radius: {t.RADIUS_INPUT};
    text-align: left;
    padding: 0 12px;
    min-height: 40px;
    max-height: 40px;
    font-size: 13px;
    font-weight: 400;
}}
QPushButton#NavItem:hover {{
    background-color: {t.HOVER};
    color: {t.TEXT};
}}
QPushButton#NavItem[selected="true"] {{
    background-color: {t.ACCENT_SUBTLE};
    color: {t.TEXT};
    font-weight: 400;
}}
QPushButton#IconButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: {t.RADIUS_INPUT};
    min-width: 36px;
    max-width: 36px;
    min-height: 36px;
    max-height: 36px;
    color: {t.TEXT_SECONDARY};
}}
QPushButton#IconButton:hover {{
    background-color: {t.HOVER};
    border-color: {t.BORDER};
    color: {t.TEXT};
}}

#Eyebrow {{
    background-color: transparent;
    color: {t.TEXT_TERTIARY};
    font-size: 12px;
    font-weight: 400;
    letter-spacing: 0.8px;
}}
#PageTitle {{
    background-color: transparent;
    color: {t.TEXT};
    font-size: 28px;
    font-weight: 400;
}}
#PageSubtitle {{
    background-color: transparent;
    color: {t.TEXT_SECONDARY};
    font-size: 14px;
    font-weight: 400;
}}
#SectionTitle {{
    background-color: transparent;
    color: {t.TEXT};
    font-size: 18px;
    font-weight: 400;
}}
#CardTitle {{
    background-color: transparent;
    color: {t.TEXT};
    font-size: 15px;
    font-weight: 400;
}}
#MetaLabel {{
    background-color: transparent;
    color: {t.TEXT_TERTIARY};
    font-size: 12px;
}}

#SurfaceCard, #ToolCard, #StepCard {{
    background-color: {t.SURFACE};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_CARD};
}}
#ToolCard:hover, #StepCard:hover {{
    background-color: {t.HOVER};
    border-color: {t.ACCENT_BORDER};
}}
#StepCard[current="true"] {{
    border-color: {t.ACCENT_BORDER};
    background-color: {t.ACCENT_SUBTLE};
}}
#StepNumber {{
    color: {t.ACCENT};
    font-size: 12px;
    font-weight: 400;
    background: transparent;
    border: none;
}}

QGroupBox {{
    background-color: {t.SURFACE};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_CARD};
    margin-top: 12px;
    padding: 20px 20px 16px 20px;
    font-size: 15px;
    font-weight: 400;
    color: {t.TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 16px;
    padding: 0 6px;
    background-color: {t.SURFACE};
    color: {t.TEXT_SECONDARY};
    font-size: 13px;
    font-weight: 400;
}}

QPushButton {{
    background-color: {t.SURFACE};
    color: {t.TEXT};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_BUTTON};
    padding: 0 14px;
    min-height: 36px;
    font-size: 13px;
    font-weight: 400;
}}
QPushButton:hover {{
    background-color: {t.HOVER};
    border-color: #303845;
}}
QPushButton:pressed {{
    background-color: {t.SURFACE_2};
}}
QPushButton:disabled {{
    color: {t.TEXT_DISABLED};
    background-color: {t.SURFACE};
    border-color: {t.BORDER_SUBTLE};
}}
QPushButton#PrimaryButton {{
    background-color: {t.ACCENT};
    color: #FFFFFF;
    border: 1px solid {t.ACCENT};
}}
QPushButton#PrimaryButton:hover {{
    background-color: {t.ACCENT_HOVER};
    border-color: {t.ACCENT_HOVER};
    color: #FFFFFF;
}}
QPushButton#PrimaryButton:pressed {{
    background-color: {t.ACCENT_PRESSED};
    border-color: {t.ACCENT_PRESSED};
}}
QPushButton#PrimaryButton:disabled {{
    background-color: {t.ACCENT_BORDER};
    border-color: {t.ACCENT_BORDER};
    color: {t.TEXT_SECONDARY};
}}
QPushButton#SecondaryButton {{
    background-color: {t.SURFACE};
    color: {t.TEXT};
    border: 1px solid {t.BORDER};
}}
QPushButton#GhostButton {{
    background: transparent;
    color: {t.TEXT_SECONDARY};
    border: 1px solid transparent;
}}
QPushButton#GhostButton:hover {{
    background-color: {t.HOVER};
    color: {t.TEXT};
}}
QPushButton#DangerButton {{
    background: transparent;
    color: {t.ERROR};
    border: 1px solid {t.ERROR};
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
    background-color: {t.INPUT};
    border: 1px solid {t.BORDER_INPUT};
    border-radius: {t.RADIUS_INPUT};
    padding: 0 10px;
    color: {t.TEXT};
    min-height: 40px;
    font-size: 14px;
    selection-background-color: {t.ACCENT_SUBTLE};
    selection-color: {t.TEXT};
}}
QPlainTextEdit {{
    padding: 10px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{
    border: 1px solid {t.ACCENT};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    border: none;
    width: 28px;
}}
QComboBox::down-arrow {{
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {t.TEXT_SECONDARY};
}}
#TrendingSelect {{
    background-color: {t.INPUT};
    border: 1px solid {t.BORDER_INPUT};
    border-radius: {t.RADIUS_INPUT};
    min-height: 40px;
    padding: 0 12px;
}}
#TrendingSelect:hover {{
    border-color: {t.BORDER};
}}
#TrendingSelectLabel {{
    background: transparent;
    color: {t.TEXT_TERTIARY};
    font-size: 14px;
    font-weight: 400;
}}
#TrendingSelectLabel[filled="true"] {{
    color: {t.TEXT};
}}
#HubModelCard {{
    background-color: {t.SURFACE};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_CARD};
}}
#HubModelCard:hover {{
    background-color: {t.HOVER};
    border-color: {t.ACCENT_BORDER};
}}
#HubModelCard[selected="true"] {{
    background-color: {t.ACCENT_SUBTLE};
    border-color: {t.ACCENT_BORDER};
}}
#HubModelCardTitle {{
    background: transparent;
    color: {t.TEXT};
    font-size: 14px;
    font-weight: 400;
}}
#HubModelCardRepo {{
    background: transparent;
    color: {t.TEXT_SECONDARY};
    font-size: 12px;
    font-weight: 400;
}}
#HubModelCardBody {{
    background: transparent;
    color: {t.TEXT_SECONDARY};
    font-size: 12px;
    font-weight: 400;
}}
#HubModelCardMeta {{
    background: transparent;
    color: {t.TEXT_TERTIARY};
    font-size: 11px;
    font-weight: 400;
}}
#HubModelCardStat {{
    background: transparent;
    color: {t.TEXT_SECONDARY};
    font-size: 12px;
    font-weight: 400;
}}
#HubModelCardList {{
    background: transparent;
    border: none;
}}
#CapabilityEyebrow {{
    background: transparent;
    color: {t.TEXT_TERTIARY};
    font-size: 11px;
    font-weight: 400;
}}
#CapabilityPill {{
    background-color: {t.SURFACE_2};
    border-radius: 999px;
}}
#CapabilityPillLabel {{
    background: transparent;
    font-size: 12px;
    font-weight: 400;
}}
QComboBox QAbstractItemView {{
    background-color: {t.SURFACE};
    color: {t.TEXT};
    border: 1px solid {t.BORDER};
    selection-background-color: {t.ACCENT_SUBTLE};
    outline: none;
}}

QListWidget {{
    background-color: {t.SURFACE};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_CARD};
    outline: none;
}}
QListWidget::item {{
    padding: 8px 12px;
    color: {t.TEXT};
}}
QListWidget::item:selected {{
    background-color: {t.ACCENT_SUBTLE};
    color: {t.TEXT};
}}

QTableWidget {{
    background-color: {t.SURFACE};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_CARD};
    gridline-color: {t.BORDER_SUBTLE};
    selection-background-color: {t.ACCENT_SUBTLE};
    selection-color: {t.TEXT};
    alternate-background-color: {t.SURFACE};
    outline: none;
    font-size: 13px;
}}
QTableWidget::item {{
    padding: 8px 10px;
    color: {t.TEXT};
    border-bottom: 1px solid {t.BORDER_SUBTLE};
}}
QHeaderView::section {{
    background-color: {t.SURFACE_2};
    color: {t.TEXT_SECONDARY};
    border: none;
    border-bottom: 1px solid {t.BORDER};
    padding: 10px;
    font-size: 12px;
    font-weight: 400;
}}

QCheckBox {{
    background-color: transparent;
    spacing: 8px;
    color: {t.TEXT};
    font-size: 13px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid {t.BORDER_INPUT};
    background: {t.INPUT};
}}
QCheckBox::indicator:checked {{
    background-color: {t.ACCENT};
    border-color: {t.ACCENT};
}}

QProgressBar {{
    background-color: {t.SURFACE_2};
    border: 1px solid {t.BORDER_SUBTLE};
    border-radius: 4px;
    height: 6px;
    max-height: 6px;
    text-align: center;
    color: {t.TEXT_TERTIARY};
    font-size: 11px;
}}
QProgressBar::chunk {{
    background-color: {t.ACCENT};
    border-radius: 4px;
}}

#LogConsole {{
    background-color: {t.INPUT};
    color: {t.TEXT_SECONDARY};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_CARD};
    padding: 10px;
    font-family: "Cascadia Code", "SF Mono", "Consolas", monospace;
    font-size: 12px;
}}
#HintLabel, #MutedLabel {{
    background-color: transparent;
    color: {t.TEXT_SECONDARY};
    font-size: 13px;
    font-weight: 400;
}}
#MetricValue {{
    background-color: transparent;
    font-size: 20px;
    font-weight: 400;
    color: {t.TEXT};
}}
#MetricDetail {{
    background-color: transparent;
    font-size: 12px;
    color: {t.TEXT_SECONDARY};
}}
#SummaryBanner {{
    background-color: {t.SURFACE};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_CARD};
    padding: 16px 20px;
    color: {t.TEXT_SECONDARY};
    font-size: 13px;
}}
#PipelineContext {{
    background: transparent;
    border: none;
}}
#StatusBadge {{
    background-color: {t.ACCENT_SUBTLE};
    border: 1px solid {t.ACCENT_BORDER};
    border-radius: 999px;
    padding: 4px 10px;
    color: {t.TEXT};
    font-size: 12px;
    font-weight: 400;
}}
#AdvancedToggle {{
    background: transparent;
    border: none;
    color: {t.TEXT_SECONDARY};
    font-size: 13px;
    font-weight: 400;
    padding: 4px 0;
}}
QScrollArea {{
    border: none;
    background-color: {t.BG};
}}
QScrollArea > QWidget > QWidget {{
    background-color: {t.BG};
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 4px;
}}
QScrollBar::handle:vertical {{
    background: {t.BORDER};
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QSplitter::handle {{
    background: {t.BORDER_SUBTLE};
}}
QDialog, QMessageBox {{
    background-color: {t.SURFACE};
}}
QToolTip {{
    background-color: {t.SURFACE_2};
    color: {t.TEXT};
    border: 1px solid {t.BORDER};
    padding: 6px 8px;
}}
"""


STYLESHEET = ""


def apply_theme(app: QApplication) -> None:
    family = load_application_fonts()
    Token.FONT = family
    Theme.FONT_FAMILY = family
    global STYLESHEET
    STYLESHEET = _sheet(family)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    app.setFont(application_font(family, 10, QFont.Weight.Normal))

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(Token.BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(Token.TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(Token.INPUT))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(Token.SURFACE))
    palette.setColor(QPalette.ColorRole.Text, QColor(Token.TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(Token.SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(Token.TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(Token.ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(Token.TEXT_TERTIARY))
    app.setPalette(palette)


def chart_colors() -> dict[str, str]:
    return {
        "line": Token.ACCENT,
        "grid": Token.BORDER,
        "label": Token.TEXT_SECONDARY,
        "background": Token.SURFACE,
    }
