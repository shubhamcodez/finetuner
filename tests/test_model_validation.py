from __future__ import annotations

import json

from finetuner.core.model_validation import validate_local_model


def test_local_model_validation_checks_config_semantics_and_weights(tmp_path):
    assert not validate_local_model(tmp_path)[0]
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    ok, message = validate_local_model(tmp_path)
    assert not ok
    assert "model_type" in message
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    assert validate_local_model(tmp_path) == (True, "")
