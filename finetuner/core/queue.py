from __future__ import annotations

from pathlib import Path
from typing import Callable

from finetuner.core.actions import ActionCancelled, ActionEvent, ActionKind, get_action
from finetuner.core.artifacts import atomic_write_json
from finetuner.core.job import JobStatus, ModelJob, ModelRunResult, ProjectConfig
from finetuner.core.paths import model_download_path, runs_dir, safe_component
from finetuner.core.preflight import validate_action
from finetuner.core.runner import ActionContext, execute_action


_ACTION_STATUS = {
    ActionKind.TRAIN: JobStatus.TRAINING,
    ActionKind.DISTILL: JobStatus.DISTILLING,
    ActionKind.EVALUATE: JobStatus.EVALUATING,
    ActionKind.ANALYZE: JobStatus.ANALYZING,
    ActionKind.QUANTIZE: JobStatus.QUANTIZING,
    ActionKind.OPTIMIZE: JobStatus.OPTIMIZING,
}


class JobQueue:
    def __init__(
        self,
        config: ProjectConfig,
        action: ActionKind | str = ActionKind.TRAIN,
        log_callback: Callable[[str], None] | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
        model_done_callback: Callable[[ModelRunResult], None] | None = None,
        download_progress_callback: Callable[[int, str], None] | None = None,
        stage_event_callback: Callable[[ActionEvent], None] | None = None,
    ) -> None:
        self.config = config
        self.action = ActionKind(action)
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

        spec = get_action(self.action)
        validate_action(self.config, spec.action)

        run_id = uuid.uuid4().hex[:8]
        run_root = runs_dir() / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        subjects = self._subjects(spec.needs_models)
        total = len(subjects)
        self.results = []
        self.log(f"Starting {spec.title.lower()}")

        for idx, model in enumerate(subjects):
            if self._cancelled:
                self.log("Run cancelled by user.")
                break

            self.on_progress(spec.action.value, idx + 1, total)
            label = model.name if model else spec.title
            self.log(f"\n=== {spec.title} {idx + 1}/{total}: {label} ===")

            result = ModelRunResult(
                model_name=label,
                model_identifier=model.identifier if model else self.config.distillation.student_model,
                output_path="",
            )

            try:
                if model:
                    model.status = JobStatus.DOWNLOADING
                    model_path = self._ensure_model_ready(model)
                    subject = model.name
                else:
                    model_path = self.config.distillation.student_model
                    subject = Path(model_path).name or "student"
                output_path = run_root / safe_component(subject or spec.action.value)
                output_path.mkdir(parents=True, exist_ok=True)

                dataset = self._resolve_dataset(required=spec.needs_dataset)
                if dataset:
                    self.log(f"Dataset: {dataset}")

                if model:
                    model.status = _ACTION_STATUS[spec.action]
                execution = execute_action(
                    ActionContext(
                        run_id=f"{run_id}-{idx + 1}",
                        run_dir=output_path,
                        model_path=str(model_path),
                        dataset_path=dataset,
                        project_config=self.config,
                        action=spec.action,
                        log=self.log,
                        is_cancelled=lambda: self._cancelled,
                        event_callback=self.on_stage_event,
                        subject=subject,
                        index=idx + 1,
                        total=total,
                    )
                )
                trained_path = execution.artifacts.get("policy_model", model_path)
                result.output_path = str(trained_path)
                result.manifest_path = str(execution.artifacts.get("manifest", ""))
                result.analysis_path = str(execution.artifacts.get("analysis", ""))
                result.deployment_path = str(execution.artifacts.get("deployment_model", ""))
                result.inference_path = str(execution.artifacts.get("inference_engine", ""))
                result.eval_results = execution.artifacts.get("eval_results", [])
                if model:
                    model.output_path = result.output_path
                    model.status = JobStatus.COMPLETED

            except ActionCancelled:
                if model:
                    model.status = JobStatus.CANCELLED
                self.log("Run cancelled by user.")
            except Exception as exc:
                msg = str(exc)
                result.training_error = msg
                if model:
                    model.error = msg
                    model.status = JobStatus.CANCELLED if self._cancelled else JobStatus.FAILED
                self.log(f"ERROR: {msg}")

            self.results.append(result)
            self.on_model_done(result)

        self._save_results(run_root)
        return self.results

    def _subjects(self, needs_models: bool) -> list[ModelJob | None]:
        if not needs_models:
            return [None]
        return [model for model in self.config.models if model.identifier]

    def _ensure_model_ready(self, model: ModelJob) -> Path:
        from finetuner.core.job import ModelSource
        from finetuner.core.model_validation import validate_local_model

        if model.source == ModelSource.LOCAL:
            path = Path(model.identifier)
            ok, msg = validate_local_model(path)
            if not ok:
                raise ValueError(msg)
            return path

        from finetuner.core.model_catalog import find_downloaded_model

        cached = find_downloaded_model(model.identifier)
        if cached:
            self.log(f"Using cached model at {cached}")
            return Path(cached)

        dest = model_download_path(model.identifier)

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
        spec = get_action(self.action)
        payload = {
            "schema_version": 1,
            "action": {"id": spec.action.value, "name": spec.title},
            "results": [
                {
                    "model_name": r.model_name,
                    "model_identifier": r.model_identifier,
                    "output_path": r.output_path,
                    "training_error": r.training_error,
                    "manifest_path": r.manifest_path,
                    "analysis_path": r.analysis_path,
                    "deployment_path": r.deployment_path,
                    "inference_path": r.inference_path,
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
