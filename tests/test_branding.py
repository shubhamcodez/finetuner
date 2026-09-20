from __future__ import annotations

from finetuner.core.paths import bundled_assets_dir, icon_path, logo_path
from finetuner.ui.branding import PRODUCT_NAME


def test_product_name_is_inferna():
    assert PRODUCT_NAME == "Inferna"


def test_inferna_logo_is_used_everywhere():
    path = logo_path()
    assert path.is_file()
    assert path.name == "inferna-logo.png"
    assert not (bundled_assets_dir() / "finetuner-logo.png").exists()
    icon = icon_path()
    assert icon.is_file()
    assert icon.name in {"icon.ico", "inferna-icon.png"}
