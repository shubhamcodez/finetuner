from __future__ import annotations

from pathlib import Path
from typing import Callable

from finetuner.core.artifacts import atomic_write_json
from finetuner.core.job import JobStatus, ModelJob, ModelRunResult, ProjectConfig
from finetuner.core.paths import model_download_path, runs_dir, safe_component
from finetuner.workflows.executor import StageEvent, WorkflowCancelled


class JobQueue:
    def __init__(
        self,
        config: ProjectConfig,
        log_callback: Callable[[str], None] | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
        model_done_callback: Callable[[ModelRunResult], None] | None = None,
        download_progress_callback: Callable[[int, str], None] | None = None,
        stage_event_callback: Callable[[StageEvent], None] | None = None,
    ) -> None:
        self.config = config
        self.log = log_callback or (lambda _msg: None)
        self.on_progress = progress_callback or (lambda _phase, _cur, _total: None)
        self.on_model_done = model_done_callback or (lambda _r: None)
        self.on_download_progress = download_progress_callback or (lambda _pct, _desc: None)
        self.on_stage_event = stage_event_callback or (lambda _event: None)
        self._cancelled = False
        self.results: list[ModelRunResult] = []

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> list[ModelRunResult]:
        import uuid

        from finetuner.workflows.preflight import validate_project_workflow

        validate_project_workflow(self.config)

        run_id = uuid.uuid4().hex[:8]
        run_root = runs_dir() / run_id
        run_root.mkdir(parents=True, exist_ok=True)

        models = [m for m in self.config.models if m.identifier]
        total = len(models)
        self.results = []

        for idx, model in enumerate(models):
            if self._cancelled:
                self.log("Run cancelled by user.")
                break

            self.on_progress("training", idx + 1, total)
            self.log(f"\n=== Model {idx + 1}/{total}: {model.name} ===")

            result = ModelRunResult(
                model_name=model.name,
                model_identifier=model.identifier,
                output_path="",
            )

            try:
                model.status = JobStatus.DOWNLOADING
                model_path = self._ensure_model_ready(model)
                output_path = run_root / safe_component(model.name)
                output_path.mkdir(parents=True, exist_ok=True)

                from finetuner.core.project_state import workflow_requires_dataset

                dataset = self._resolve_dataset(required=workflow_requires_dataset(self.config))
                if dataset:
                    self.log(f"Workflow dataset: {dataset}")

                from finetuner.workflows.executor import WorkflowContext, WorkflowExecutor
                from finetuner.workflows.runtime import production_handlers

                self.config.workflow.validate()
                self.log(f"Workflow: {self.config.workflow.name}")
                model.status = JobStatus.TRAINING
                execution = WorkflowExecutor(production_handlers()).execute(
                    self.config.workflow,
                    WorkflowContext(
                        run_id=f"{run_id}-{idx + 1}",
                        run_dir=output_path,
                        model_path=str(model_path),
                        dataset_path=dataset,
                        project_config=self.config,
                        log=self.log,
                        is_cancelled=lambda: self._cancelled,
                        stage_callback=self.on_stage_event,
                        subject=model.name,
                    ),
                )
                trained_path = execution.latest_artifact("policy_model", str(model_path))
                result.output_path = str(trained_path)
                result.manifest_path = execution.manifest_path
                result.analysis_path = str(execution.latest_artifact("analysis", ""))
                result.deployment_path = str(execution.latest_artifact("deployment_model", ""))
                result.eval_results = execution.latest_artifact("eval_results", [])
                model.output_path = result.output_path
                model.status = JobStatus.COMPLETED

            except WorkflowCancelled:
                model.status = JobStatus.CANCELLED
                self.log("Workflow cancelled by user.")
            except Exception as exc:
                msg = str(exc)
                result.training_error = msg
                model.error = msg
                model.status = JobStatus.CANCELLED if self._cancelled else JobStatus.FAILED
                self.log(f"ERROR: {msg}")

            self.results.append(result)
            self.on_model_done(result)

        self._save_results(run_root)
        return self.results

    def _ensure_model_ready(self, model: ModelJob) -> Path:
        from finetuner.core.job import ModelSource
        from finetuner.core.model_validation import validate_local_model

        if model.source == ModelSource.LOCAL:
            path = Path(model.identifier)
            ok, msg = validate_local_model(path)
            if not ok:
                raise ValueError(msg)
            return path

        dest = model_download_path(model.identifier)
        if dest.exists() and (dest / "config.json").exists():
            self.log(f"Using cached model at {dest}")
            return dest

        self.log(f"Downloading {model.identifier}...")
        from finetuner.core.hf_download import download_hf_model

        download_hf_model(
            repo_id=model.identifier,
            dest=dest,
            token=self.config.hf_token or None,
            on_progress=self.on_download_progress,
            on_log=self.log,
        )
        model.output_path = str(dest)
        return dest

    def _resolve_dataset(self, *, required: bool = True) -> str:
        training = self.config.training
        if training.dataset_path and Path(training.dataset_path).exists():
            return training.dataset_path
        if training.dataset_preset_id:
            return f"preset://{training.dataset_preset_id}"
        if training.dataset_hf_id:
            from finetuner.datasets.hf_datasets import normalize_hf_dataset_id

            return normalize_hf_dataset_id(training.dataset_hf_id)
        if required:
            raise ValueError(
                "No dataset configured. Select a preset, provide a JSONL path, or set an HF dataset ID."
            )
        return ""

    def _save_results(self, run_root: Path) -> None:
        payload = {
            "schema_version": 1,
            "workflow": self.config.workflow.to_dict(),
            "results": [
                {
                    "model_name": r.model_name,
                    "model_identifier": r.model_identifier,
                    "output_path": r.output_path,
                    "training_error": r.training_error,
                    "manifest_path": r.manifest_path,
                    "analysis_path": r.analysis_path,
                    "deployment_path": r.deployment_path,
                    "eval_results": [
                        {
                            "task_id": e.task_id,
                            "task_name": e.task_name,
                            "score": e.score,
                            "metric": e.metric,
                        }
                        for e in r.eval_results
                    ],
                }
                for r in self.results
            ],
        }
        out = run_root / "results.json"
        atomic_write_json(out, payload)
        self.log(f"Results saved to {out}")
