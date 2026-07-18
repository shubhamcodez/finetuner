from __future__ import annotations

import json

from finetuner.core.job import EvalResult, JobStatus, ModelJob, ModelSource, ProjectConfig
from finetuner.core.queue import JobQueue
from finetuner.workflows.executor import StageOutput
from finetuner.workflows.schema import StageKind, WorkflowSpec, WorkflowStage
from finetuner.workflows.templates import get_workflow_template


def test_queue_executes_workflow_and_persists_result_lineage(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"text":"sample"}\n', encoding="utf-8")
    model = ModelJob("../unsafe model", ModelSource.LOCAL, str(model_dir))
    config = ProjectConfig(
        models=[model],
        workflow=get_workflow_template("sft"),
    )
    config.training.dataset_path = str(dataset)

    def train_handler(_stage, context, _dependencies):
        trained = context.run_dir / "trained"
        trained.mkdir()
        return StageOutput(artifacts={"policy_model": str(trained)})

    def eval_handler(_stage, _context, _dependencies):
        return StageOutput(
            artifacts={"eval_results": [EvalResult("task", "Task", 91.0)]},
            metrics={"task": 91.0},
        )

    monkeypatch.setattr("finetuner.core.queue.runs_dir", lambda: tmp_path / "runs")
    monkeypatch.setattr(
        "finetuner.workflows.runtime.production_handlers",
        lambda: {StageKind.TRAIN: train_handler, StageKind.EVALUATE: eval_handler},
    )
    results = JobQueue(config).run()
    assert len(results) == 1
    assert results[0].eval_results[0].score == 91.0
    assert results[0].manifest_path.endswith("manifest.json")
    assert model.status == JobStatus.COMPLETED
    assert ".." not in results[0].manifest_path
    aggregate = json.loads(next((tmp_path / "runs").glob("*/results.json")).read_text("utf-8"))
    assert aggregate["workflow"]["id"] == "sft"
    assert aggregate["results"][0]["manifest_path"] == results[0].manifest_path


def test_queue_emits_stage_lifecycle_and_allows_deployment_without_dataset(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    config = ProjectConfig(
        models=[ModelJob("Deployable", ModelSource.LOCAL, str(model_dir))],
        workflow=WorkflowSpec(
            "deploy",
            "Deploy only",
            (WorkflowStage("quantize", "Quantize", StageKind.QUANTIZE),),
        ),
    )
    events = []

    def quantize_handler(_stage, context, _dependencies):
        artifact = context.run_dir / "model.gguf"
        artifact.write_bytes(b"gguf")
        assert context.dataset_path == ""
        return StageOutput(artifacts={"deployment_model": str(artifact)})

    monkeypatch.setattr("finetuner.core.queue.runs_dir", lambda: tmp_path / "runs")
    monkeypatch.setattr(
        "finetuner.workflows.runtime.production_handlers",
        lambda: {StageKind.QUANTIZE: quantize_handler},
    )

    results = JobQueue(config, stage_event_callback=events.append).run()

    assert not results[0].training_error
    assert results[0].deployment_path.endswith("model.gguf")
    assert [event.status for event in events] == ["running", "completed"]
    assert events[-1].subject == "Deployable"
