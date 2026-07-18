from __future__ import annotations

import json

import pytest

from finetuner.core.job import ProjectConfig
from finetuner.workflows.executor import StageOutput, WorkflowContext, WorkflowExecutor
from finetuner.workflows.schema import (
    StageKind,
    WorkflowSpec,
    WorkflowStage,
    WorkflowValidationError,
)
from finetuner.workflows.templates import get_workflow_template, workflow_templates


def test_all_builtin_templates_are_valid_and_round_trip():
    for template in workflow_templates():
        template.validate()
        restored = WorkflowSpec.from_dict(template.to_dict())
        assert restored == template


def test_topological_sort_is_deterministic_for_parallel_stages():
    workflow = WorkflowSpec(
        "parallel",
        "Parallel",
        (
            WorkflowStage("root", "Root", StageKind.TRAIN),
            WorkflowStage("second", "Second", StageKind.ANALYZE, ("root",)),
            WorkflowStage("first", "First", StageKind.EVALUATE, ("root",)),
            WorkflowStage("end", "End", StageKind.QUANTIZE, ("first", "second")),
        ),
    )
    assert [stage.stage_id for stage in workflow.topological_stages()] == [
        "root",
        "second",
        "first",
        "end",
    ]


def test_cycle_and_missing_dependency_are_rejected():
    cyclic = WorkflowSpec(
        "cycle",
        "Cycle",
        (
            WorkflowStage("a", "A", StageKind.TRAIN, ("b",)),
            WorkflowStage("b", "B", StageKind.EVALUATE, ("a",)),
        ),
    )
    with pytest.raises(WorkflowValidationError, match="cycle"):
        cyclic.validate()

    missing = WorkflowSpec(
        "missing",
        "Missing",
        (WorkflowStage("a", "A", StageKind.TRAIN, ("unknown",)),),
    )
    with pytest.raises(WorkflowValidationError, match="missing dependencies"):
        missing.validate()


def test_executor_passes_only_declared_dependencies_and_writes_manifest(tmp_path):
    calls: list[tuple[str, list[str]]] = []
    events = []

    def handler(stage, _context, dependencies):
        calls.append((stage.stage_id, list(dependencies)))
        return StageOutput(artifacts={"policy_model": f"/{stage.stage_id}"})

    workflow = get_workflow_template("sft")
    result = WorkflowExecutor({StageKind.TRAIN: handler, StageKind.EVALUATE: handler}).execute(
        workflow,
        WorkflowContext(
            "run-1",
            tmp_path,
            "/base",
            "/dataset",
            ProjectConfig(),
            stage_callback=events.append,
            subject="Model A",
        ),
    )
    assert calls == [("sft", []), ("evaluate", ["sft"])]
    assert result.latest_artifact("policy_model") == "/evaluate"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert [stage["status"] for stage in manifest["stages"]] == ["completed", "completed"]
    assert [(event.stage_id, event.status) for event in events] == [
        ("sft", "running"),
        ("sft", "completed"),
        ("evaluate", "running"),
        ("evaluate", "completed"),
    ]
    assert all(event.subject == "Model A" for event in events)
    assert events[-1].artifact_names == ("policy_model",)


def test_executor_records_failure(tmp_path):
    events = []

    def broken(*_args):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        WorkflowExecutor({StageKind.TRAIN: broken}).execute(
            WorkflowSpec(
                "broken",
                "Broken",
                (WorkflowStage("train", "Train", StageKind.TRAIN),),
            ),
            WorkflowContext(
                "run",
                tmp_path,
                "/base",
                "/data",
                ProjectConfig(),
                stage_callback=events.append,
            ),
        )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["stages"][0]["error"] == "RuntimeError: boom"
    assert [event.status for event in events] == ["running", "failed"]
    assert events[-1].message == "boom"
