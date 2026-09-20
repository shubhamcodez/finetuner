from __future__ import annotations

from finetuner.core.paths import logo_path
from finetuner.ui.branding import PRODUCT_NAME


def test_product_name_is_inferna():
    assert PRODUCT_NAME == "Inferna"


def test_inferna_logo_is_present():
    path = logo_path()
    assert path.is_file()
    assert path.name == "inferna-logo.png"
