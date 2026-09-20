from __future__ import annotations

import json

from finetuner.core.actions import ActionKind, ActionOutput
from finetuner.core.job import EvalResult, JobStatus, ModelJob, ModelSource, ProjectConfig
from finetuner.core.queue import JobQueue


def test_queue_executes_training_and_persists_result_lineage(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"text":"sample"}\n', encoding="utf-8")
    model = ModelJob("../unsafe model", ModelSource.LOCAL, str(model_dir))
    config = ProjectConfig(models=[model])
    config.training.dataset_path = str(dataset)

    def fake_execute(context):
        trained = context.run_dir / "trained"
        trained.mkdir()
        return ActionOutput(artifacts={"policy_model": str(trained)})

    monkeypatch.setattr("finetuner.core.queue.runs_dir", lambda: tmp_path / "runs")
    monkeypatch.setattr("finetuner.core.queue.execute_action", fake_execute)
    results = JobQueue(config, action=ActionKind.TRAIN).run()
    assert len(results) == 1
    assert results[0].output_path.endswith("trained")
    assert model.status == JobStatus.COMPLETED
    assert ".." not in results[0].output_path
    aggregate = json.loads(next((tmp_path / "runs").glob("*/results.json")).read_text("utf-8"))
    assert aggregate["action"]["id"] == "train"


def test_queue_allows_deployment_without_dataset(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    config = ProjectConfig(models=[ModelJob("Deployable", ModelSource.LOCAL, str(model_dir))])

    def fake_execute(context):
        artifact = context.run_dir / "model.gguf"
        artifact.write_bytes(b"gguf")
        assert context.dataset_path == ""
        return ActionOutput(artifacts={"deployment_model": str(artifact)})

    monkeypatch.setattr("finetuner.core.queue.runs_dir", lambda: tmp_path / "runs")
    monkeypatch.setattr("finetuner.core.queue.execute_action", fake_execute)

    results = JobQueue(config, action=ActionKind.QUANTIZE).run()

    assert not results[0].training_error
    assert results[0].deployment_path.endswith("model.gguf")
    assert results[0].model_name == "Deployable"


def test_queue_records_inference_engine_plan(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    config = ProjectConfig(models=[ModelJob("Servable", ModelSource.LOCAL, str(model_dir))])

    def fake_execute(context):
        return ActionOutput(artifacts={"inference_engine": str(context.run_dir)})

    monkeypatch.setattr("finetuner.core.queue.runs_dir", lambda: tmp_path / "runs")
    monkeypatch.setattr("finetuner.core.queue.execute_action", fake_execute)

    results = JobQueue(config, action=ActionKind.OPTIMIZE).run()

    assert results[0].inference_path
    aggregate = json.loads(next((tmp_path / "runs").glob("*/results.json")).read_text("utf-8"))
    assert aggregate["results"][0]["inference_path"] == results[0].inference_path


def test_queue_evaluate_records_scores(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    config = ProjectConfig(models=[ModelJob("Eval", ModelSource.LOCAL, str(model_dir))])

    def fake_execute(context):
        return ActionOutput(
            artifacts={"eval_results": [EvalResult("task", "Task", 91.0)]},
            metrics={"task": 91.0},
        )

    monkeypatch.setattr("finetuner.core.queue.runs_dir", lambda: tmp_path / "runs")
    monkeypatch.setattr("finetuner.core.queue.execute_action", fake_execute)
    results = JobQueue(config, action=ActionKind.EVALUATE).run()
    assert results[0].eval_results[0].score == 91.0
