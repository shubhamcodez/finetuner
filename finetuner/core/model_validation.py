from __future__ import annotations

import json
from pathlib import Path


def validate_local_model(path: Path) -> tuple[bool, str]:
    if not path.is_dir():
        return False, "Path is not a directory"
    config = path / "config.json"
    if not config.is_file():
        return False, "Missing config.json — not a valid Hugging Face model folder"
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "config.json is not valid JSON"
    if not isinstance(payload, dict) or not payload.get("model_type"):
        return False, "config.json does not declare model_type"
    weight_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    if not weight_files:
        index_files = list(path.glob("*.safetensors.index.json")) + list(
            path.glob("*.bin.index.json")
        )
        if not index_files:
            return False, "No model weights or sharded-weight index found"
    return True, ""
