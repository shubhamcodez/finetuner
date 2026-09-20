"""Restart the desktop app when source files change."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_SKIP_DIRS = {
    ".git",
    ".venv",
    ".venv-bench",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "dist",
    "build",
    "runs",
}
_WATCH_SUFFIXES = {".py", ".qss", ".json", ".png", ".ico", ".jsonl"}
_POLL_SECONDS = 0.6
_DEBOUNCE_SECONDS = 0.4
CHILD_ENV = "FINETUNER_RELOADER_CHILD"


def watch_roots() -> tuple[Path, ...]:
    package = Path(__file__).resolve().parent
    assets = package.parent / "assets"
    roots = [package]
    if assets.is_dir():
        roots.append(assets)
    return tuple(roots)


def file_snapshot(roots: tuple[Path, ...] | None = None) -> dict[str, float]:
    stamp: dict[str, float] = {}
    for root in roots or watch_roots():
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _WATCH_SUFFIXES:
                continue
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            stamp[str(path)] = path.stat().st_mtime
    return stamp


def snapshot_changed(before: dict[str, float], after: dict[str, float]) -> tuple[str, ...]:
    changed = [path for path, mtime in after.items() if before.get(path) != mtime]
    changed.extend(path for path in before if path not in after)
    return tuple(sorted(changed))


def _start_child() -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env[CHILD_ENV] = "1"
    return subprocess.Popen([sys.executable, "-m", "finetuner", "--no-reload"], env=env)


def _stop_child(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def run_reloader() -> int:
    print("Finetuner reload: watching for file changes. Ctrl+C stops the app.", flush=True)
    process = _start_child()
    previous = file_snapshot()
    try:
        while True:
            time.sleep(_POLL_SECONDS)
            if process.poll() is not None:
                if process.returncode == 0:
                    return 0
                print("Finetuner exited with an error; restarting.", flush=True)
                time.sleep(0.8)
                process = _start_child()
                previous = file_snapshot()
                continue
            current = file_snapshot()
            changed = snapshot_changed(previous, current)
            if not changed:
                continue
            time.sleep(_DEBOUNCE_SECONDS)
            previous = file_snapshot()
            names = ", ".join(Path(path).name for path in changed[:4])
            extra = f" (+{len(changed) - 4})" if len(changed) > 4 else ""
            print(f"Reloading Finetuner ({names}{extra})", flush=True)
            _stop_child(process)
            process = _start_child()
    except KeyboardInterrupt:
        _stop_child(process)
        return 0
