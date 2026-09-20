from __future__ import annotations

import pytest

from finetuner.ui.icons import _PHOSPHOR, line_icon


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.mark.ui
def test_phosphor_icons_resolve_for_every_nav_name(app):
    for name in _PHOSPHOR:
        icon = line_icon(name, "#5B8DEF", 18)
        assert not icon.isNull()
        assert not icon.pixmap(18, 18).isNull()
