from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from finetuner.core.job import ModelJob, ModelRunResult, ProjectConfig
from finetuner.core.paths import model_download_path, runs_dir


class JobQueue:
    def __init__(
        self,
        config: ProjectConfig,
        log_callback: Callable[[str], None] | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
        model_done_callback: Callable[[ModelRunResult], None] | None = None,
        download_progress_callback: Callable[[int, str], None] | None = None,
    ) -> None:
        self.config = config
        self.log = log_callback or (lambda _msg: None)
        self.on_progress = progress_callback or (lambda _phase, _cur, _total: None)
        self.on_model_done = model_done_callback or (lambda _r: None)
        self.on_download_progress = download_progress_callback or (lambda _pct, _desc: None)
        self._cancelled = False
        self.results: list[ModelRunResult] = []

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> list[ModelRunResult]:
        import uuid

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
                model_path = self._ensure_model_ready(model)
                output_path = run_root / model.name.replace(" ", "_")
                output_path.mkdir(parents=True, exist_ok=True)

                dataset = self._resolve_dataset()
                self.log(f"Training on dataset: {dataset}")

                from finetuner.training.runner import train

                trained_path = train(
                    model_path=str(model_path),
                    output_dir=str(output_path),
                    training=self.config.training,
                    dataset_path=dataset,
                    log_callback=self.log,
                )
                result.output_path = trained_path
                model.output_path = trained_path

                if self._cancelled:
                    break

                self.on_progress("evaluating", idx + 1, total)
                self.log(f"Running evals for {model.name}...")
                from finetuner.eval.runner import run_evals

                eval_results = run_evals(
                    model_path=trained_path,
                    task_ids=self.config.enabled_evals,
                    max_samples=self.config.eval_max_samples,
                    log_callback=self.log,
                )
                result.eval_results = eval_results
                self.log(f"Evals complete for {model.name}")

            except Exception as exc:
                msg = str(exc)
                result.training_error = msg
                model.error = msg
                self.log(f"ERROR: {msg}")

            self.results.append(result)
            self.on_model_done(result)

        self._save_results(run_root)
        return self.results

    def _ensure_model_ready(self, model: ModelJob) -> Path:
        from finetuner.core.job import ModelSource
        from finetuner.ui.models_tab import validate_local_model

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

    def _resolve_dataset(self) -> str:
        training = self.config.training
        if training.dataset_path and Path(training.dataset_path).exists():
            return training.dataset_path
        if training.dataset_preset_id:
            return f"preset://{training.dataset_preset_id}"
        if training.dataset_hf_id:
            from finetuner.datasets.hf_datasets import normalize_hf_dataset_id

            return normalize_hf_dataset_id(training.dataset_hf_id)
        raise ValueError(
            "No dataset configured. Select a preset, provide a JSONL path, or set an HF dataset ID."
        )

    def _save_results(self, run_root: Path) -> None:
        payload = {
            "results": [
                {
                    "model_name": r.model_name,
                    "model_identifier": r.model_identifier,
                    "output_path": r.output_path,
                    "training_error": r.training_error,
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
            ]
        }
        out = run_root / "results.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.log(f"Results saved to {out}")
