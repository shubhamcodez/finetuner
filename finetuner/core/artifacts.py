from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from finetuner import __version__


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def artifact_metadata(uri: str | Path, hash_limit_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    path = Path(uri)
    if not path.exists():
        return {"exists": False}
    if path.is_dir():
        return {"exists": True, "type": "directory"}
    size = path.stat().st_size
    metadata: dict[str, Any] = {"exists": True, "type": "file", "size_bytes": size}
    if size <= hash_limit_bytes:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata["sha256"] = digest.hexdigest()
    else:
        metadata["sha256"] = None
        metadata["hash_skipped"] = f"file exceeds {hash_limit_bytes} bytes"
    return metadata


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON without exposing readers to a partial manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    kind: str
    uri: str
    producer_stage: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "uri": self.uri,
            "producer_stage": self.producer_stage,
            "metadata": self.metadata,
        }


@dataclass
class RunManifest:
    run_id: str
    workflow: dict[str, Any]
    config_digest: str
    started_at: str = field(default_factory=_utc_now)
    finished_at: str | None = None
    status: str = "running"
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    stages: list[dict[str, Any]] = field(default_factory=list)
    environment: dict[str, str] = field(
        default_factory=lambda: {
            "finetuner": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
    )

    def finish(self, status: str) -> None:
        self.status = status
        self.finished_at = _utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "config_digest": self.config_digest,
            "workflow": self.workflow,
            "environment": self.environment,
            "stages": self.stages,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    def save(self, path: Path) -> None:
        atomic_write_json(path, self.to_dict())
