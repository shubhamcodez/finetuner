from __future__ import annotations

from finetuner.core.paths import safe_component


def test_untrusted_names_cannot_escape_run_directory():
    component = safe_component("../../outside\\model:C")
    assert "/" not in component
    assert "\\" not in component
    assert ":" not in component
    assert component not in {".", ".."}


def test_path_components_are_stable_and_collision_resistant():
    assert safe_component("a/b") == safe_component("a/b")
    assert safe_component("a/b") != safe_component("a\\b")
