from __future__ import annotations

import json

from finetuner.core.model_catalog import discover_downloaded_models, find_downloaded_model


def _write_model(path, *, metadata: dict | None = None):
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    if metadata is not None:
        (path / ".finetuner-model.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_catalog_discovers_legacy_and_hashed_downloads(tmp_path):
    legacy = tmp_path / "Qwen__Qwen2.5-0.5B-Instruct"
    hashed = tmp_path / "meta-llama__Llama-3.2-1B--0123456789"
    _write_model(legacy)
    _write_model(hashed)
    (tmp_path / "partial-download").mkdir()

    models = discover_downloaded_models(tmp_path)

    assert [model.source_id for model in models] == [
        "meta-llama/Llama-3.2-1B",
        "Qwen/Qwen2.5-0.5B-Instruct",
    ]
    assert find_downloaded_model("qwen/qwen2.5-0.5b-instruct", tmp_path) == str(legacy)


def test_catalog_prefers_download_metadata_for_display_and_lookup(tmp_path):
    path = tmp_path / "custom-cache-name"
    _write_model(path, metadata={"schema_version": 1, "repo_id": "org/source-model"})

    [model] = discover_downloaded_models(tmp_path)

    assert model.name == "org/source-model"
    assert model.path == str(path)
    assert find_downloaded_model("org/source-model", tmp_path) == str(path)
