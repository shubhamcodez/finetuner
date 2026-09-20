from __future__ import annotations

import json

import pytest

from finetuner.core.actions import ActionCancelled, ActionKind, ActionOutput
from finetuner.core.job import ProjectConfig
from finetuner.core.runner import ActionContext, execute_action


def test_execute_action_writes_manifest_and_events(tmp_path, monkeypatch):
    events = []

    def handler(context):
        return ActionOutput(artifacts={"policy_model": str(context.run_dir / "out")})

    monkeypatch.setattr("finetuner.core.runner.run_train", handler)
    result = execute_action(
        ActionContext(
            "run-1",
            tmp_path,
            "/base",
            "/dataset",
            ProjectConfig(),
            ActionKind.TRAIN,
            event_callback=events.append,
            subject="Model A",
        )
    )
    assert result.artifacts["policy_model"].endswith("out")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["action"]["id"] == "train"
    assert [event.status for event in events] == ["running", "completed"]
    assert events[-1].subject == "Model A"
    assert events[-1].artifact_names == ("policy_model",)


def test_execute_action_records_failure(tmp_path, monkeypatch):
    events = []

    def broken(_context):
        raise RuntimeError("boom")

    monkeypatch.setattr("finetuner.core.runner.run_quantize", broken)
    with pytest.raises(RuntimeError, match="boom"):
        execute_action(
            ActionContext(
                "run",
                tmp_path,
                "/base",
                "",
                ProjectConfig(),
                ActionKind.QUANTIZE,
                event_callback=events.append,
            )
        )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["stages"][0]["error"] == "RuntimeError: boom"
    assert [event.status for event in events] == ["running", "failed"]


def test_execute_action_honours_cancellation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "finetuner.core.runner.run_evaluate",
        lambda _context: ActionOutput(),
    )
    with pytest.raises(ActionCancelled):
        execute_action(
            ActionContext(
                "run",
                tmp_path,
                "/base",
                "",
                ProjectConfig(),
                ActionKind.EVALUATE,
                is_cancelled=lambda: True,
            )
        )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"
