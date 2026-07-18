from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from finetuner.core.model_validation import validate_local_model
from finetuner.core.paths import models_dir

_MODEL_METADATA = ".finetuner-model.json"
_HASH_SUFFIX = re.compile(r"--[0-9a-f]{10}$", re.IGNORECASE)


@dataclass(frozen=True)
class DownloadedModel:
    name: str
    path: str
    source_id: str = ""


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _infer_source_id(path: Path) -> str:
    metadata = _read_json(path / _MODEL_METADATA)
    source_id = str(metadata.get("repo_id", "")).strip()
    if source_id:
        return source_id

    config = _read_json(path / "config.json")
    name_or_path = str(config.get("_name_or_path", "")).strip()
    if name_or_path and "/" in name_or_path and not Path(name_or_path).is_absolute():
        return name_or_path

    decoded = _HASH_SUFFIX.sub("", path.name).replace("__", "/")
    return decoded


def discover_downloaded_models(root: Path | None = None) -> list[DownloadedModel]:
    catalog_root = Path(root) if root is not None else models_dir()
    if not catalog_root.is_dir():
        return []

    discovered: list[DownloadedModel] = []
    for path in sorted(catalog_root.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_dir() or not validate_local_model(path)[0]:
            continue
        source_id = _infer_source_id(path)
        discovered.append(DownloadedModel(source_id or path.name, str(path), source_id))
    return discovered


def find_downloaded_model(source_id: str, root: Path | None = None) -> str:
    wanted = source_id.strip().casefold()
    if not wanted:
        return ""
    for model in discover_downloaded_models(root):
        if model.source_id.casefold() == wanted:
            return model.path
    return ""


def model_metadata_name() -> str:
    return _MODEL_METADATA
