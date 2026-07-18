from __future__ import annotations

import hashlib
import json

from finetuner.core.artifacts import artifact_metadata, atomic_write_json, stable_digest


def test_atomic_json_and_stable_digest(tmp_path):
    destination = tmp_path / "nested" / "manifest.json"
    atomic_write_json(destination, {"b": 2, "a": 1})
    assert json.loads(destination.read_text("utf-8")) == {"a": 1, "b": 2}
    assert stable_digest({"a": 1, "b": 2}) == stable_digest({"b": 2, "a": 1})
    assert not list(destination.parent.glob("*.tmp"))


def test_small_file_artifact_gets_integrity_hash(tmp_path):
    artifact = tmp_path / "model.json"
    artifact.write_bytes(b"artifact")
    metadata = artifact_metadata(artifact)
    assert metadata["size_bytes"] == 8
    assert metadata["sha256"] == hashlib.sha256(b"artifact").hexdigest()
