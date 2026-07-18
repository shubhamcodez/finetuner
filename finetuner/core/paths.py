from __future__ import annotations

import os
import hashlib
import re
import sys
from pathlib import Path

APP_NAME = "finetuner"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Open-source ~0.5B instruct models pre-loaded for new projects.
DEFAULT_MODELS: list[tuple[str, str]] = [
    ("Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("Qwen2-0.5B-Instruct", "Qwen/Qwen2-0.5B-Instruct"),
]


def app_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("USERPROFILE", str(Path.home()))
    path = Path(base) / ".finetuner"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = app_data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runs_dir() -> Path:
    path = app_data_dir() / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    path = app_data_dir() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path() -> Path:
    return app_data_dir() / "config.json"


def repo_slug(repo_id: str) -> str:
    return safe_component(repo_id.replace("/", "__"))


def safe_component(value: str, max_length: int = 80) -> str:
    """Turn an untrusted display name/model id into one non-traversing path component."""
    original = str(value).strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", original).strip(" ._")
    cleaned = cleaned or "item"
    if cleaned in {".", ".."} or cleaned.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        cleaned = f"item_{cleaned.lower().strip('.')}"
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[:max_length]}--{digest}"


def model_download_path(repo_id: str) -> Path:
    return models_dir() / repo_slug(repo_id)


def bundled_assets_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "assets"
    return Path(__file__).resolve().parents[2] / "assets"


def logo_path() -> Path:
    bundled = bundled_assets_dir() / "finetuner-logo.png"
    if bundled.exists():
        return bundled
    dev_root = Path(__file__).resolve().parents[2] / "finetuner-logo.png"
    if dev_root.exists():
        return dev_root
    return bundled


def icon_path() -> Path:
    bundled = bundled_assets_dir() / "icon.ico"
    if bundled.exists():
        return bundled
    dev = Path(__file__).resolve().parents[2] / "assets" / "icon.ico"
    return dev if dev.exists() else logo_path()


def sample_dataset_path() -> Path:
    return bundled_assets_dir() / "sample_sft.jsonl"
