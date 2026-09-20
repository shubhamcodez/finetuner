from __future__ import annotations

from finetuner.reloader import file_snapshot, snapshot_changed, watch_roots


def test_watch_roots_include_package_and_assets():
    roots = watch_roots()
    names = {path.name for path in roots}
    assert "finetuner" in names
    assert any(path.is_dir() for path in roots)


def test_snapshot_changed_detects_new_and_updated_files(tmp_path):
    first = tmp_path / "a.py"
    first.write_text("one", encoding="utf-8")
    before = file_snapshot((tmp_path,))
    first.write_text("two", encoding="utf-8")
    (tmp_path / "b.py").write_text("new", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.cpython-314.pyc").write_bytes(b"x")
    after = file_snapshot((tmp_path,))
    changed = snapshot_changed(before, after)
    assert str(tmp_path / "b.py") in changed
    assert str(first) in changed
    assert not any(path.endswith(".pyc") for path in changed)
