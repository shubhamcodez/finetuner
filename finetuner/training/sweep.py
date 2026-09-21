from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from finetuner.core.job import TrainingConfig
from finetuner.core.paths import model_download_path
from finetuner.datasets.presets import get_preset
from finetuner.eval.tasks import EVAL_TASKS
from finetuner.training.efficiency import (
    PAGE_METHODS,
    POLICY_METHODS,
    PREFERENCE_METHODS,
    CellResult,
    SweepCell,
    build_cells,
    directory_bytes,
    finalize_cell,
    parse_trainable_params,
    parse_trainer_state,
    render_markdown,
    run_order,
)

LogFn = Callable[[str], None]


@dataclass
class SweepConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_path: str = ""
    methods: list[str] = field(default_factory=lambda: list(PAGE_METHODS))
    accelerator: str = "cuda"
    npu_artifact_path: str = ""
    datasets: list[str] = field(default_factory=lambda: ["gsm8k", "hellaswag", "arc_challenge"])
    evals: list[str] = field(default_factory=lambda: ["gsm8k", "hellaswag", "arc_challenge"])
    steps_offline: int = 8
    steps_online: int = 4
    eval_samples: int = 10
    max_samples: int = 48
    holdout_samples: int = 8
    bundled_only: bool = False
    matching_eval_only: bool = False
    allow_cpu: bool = False
    learning_rate: float = 2e-4
    lora_rank: int = 8
    lora_alpha: int = 16
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 256
    output_dir: str = ""
    resume: bool = True


class ResourceMonitor:
    def __init__(self) -> None:
        self.peak_ram_gb = 0.0
        self.peak_gpu_gb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float, float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self.peak_ram_gb, self.peak_gpu_gb

    def _run(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        process = psutil.Process()
        while not self._stop.wait(0.5):
            try:
                rss = process.memory_info().rss / (1024**3)
                for child in process.children(recursive=True):
                    try:
                        rss += child.memory_info().rss / (1024**3)
                    except (psutil.Error, OSError):
                        continue
                self.peak_ram_gb = max(self.peak_ram_gb, rss)
            except (psutil.Error, OSError):
                pass
            try:
                import torch

                if torch.cuda.is_available():
                    current = torch.cuda.max_memory_allocated() / (1024**3)
                    self.peak_gpu_gb = max(self.peak_gpu_gb, current)
            except Exception:
                pass


def default_output_dir() -> Path:
    path = Path.home() / ".finetuner" / "bench" / "method-sweep"
    path.mkdir(parents=True, exist_ok=True)
    return path


def related_eval_map(dataset_ids: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for dataset_id in dataset_ids:
        preset = get_preset(dataset_id)
        if preset and preset.related_eval_id:
            mapping[dataset_id] = preset.related_eval_id
    return mapping


def official_eval_ids(dataset_id: str, eval_ids: list[str]) -> tuple[str, ...]:
    related = related_eval_map([dataset_id]).get(dataset_id, "")
    ordered: list[str] = []
    for task_id in (related, *eval_ids):
        if task_id in EVAL_TASKS and task_id not in ordered:
            ordered.append(task_id)
    return tuple(ordered)


def plan_cells(config: SweepConfig) -> list[SweepCell]:
    related = related_eval_map(config.datasets)
    cells: list[SweepCell] = []
    for dataset_id in config.datasets:
        eval_ids = (
            [related.get(dataset_id, dataset_id)]
            if config.matching_eval_only
            else config.evals
        )
        cells.extend(
            build_cells(
                config.methods,
                [dataset_id],
                eval_ids,
                related_evals=related,
                steps_offline=config.steps_offline,
                steps_online=config.steps_online,
            )
        )
    return run_order(cells)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_results(path: Path) -> dict[str, CellResult]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    loaded: dict[str, CellResult] = {}
    for row in payload.get("results", []):
        if not isinstance(row, dict) or "method" not in row:
            continue
        result = CellResult(**{key: row[key] for key in CellResult.__dataclass_fields__ if key in row})
        loaded[f"{result.method}:{result.dataset}"] = result
    return loaded


def _bundled_text_rows(dataset_id: str, limit: int) -> list[dict]:
    import json

    from finetuner.core.paths import bundled_assets_dir
    from finetuner.datasets.presets import FORMATTERS, get_preset
    from finetuner.datasets.rows import row_to_text

    preset = get_preset(dataset_id)
    if preset is None or not preset.bundled_filename:
        raise ValueError(f"Preset {dataset_id} has no bundled sample available.")
    path = bundled_assets_dir() / "datasets" / preset.bundled_filename
    if not path.is_file():
        raise FileNotFoundError(f"Bundled dataset missing: {path}")
    formatter = FORMATTERS.get(dataset_id)
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        text = row_to_text(json.loads(line), formatter)
        if text:
            rows.append({"text": text})
        if limit and len(rows) >= limit:
            break
    return rows


def materialize_dataset(
    dataset_id: str,
    destination: Path,
    *,
    max_samples: int,
    holdout_samples: int,
    bundled_only: bool,
    log: LogFn,
) -> tuple[Path, list[dict]]:
    train_count = max(1, max_samples)
    extra = max(0, holdout_samples)
    try:
        from finetuner.datasets.loader import load_preset_dataset

        raw = load_preset_dataset(
            dataset_id,
            max_samples=train_count + extra,
            log=log,
            bundled_only=bundled_only,
        )
        rows = [dict(row) for row in raw]
    except ImportError:
        rows = _bundled_text_rows(dataset_id, train_count + extra)
        log(f"Loaded bundled {dataset_id} without Hugging Face datasets ({len(rows)} rows)")
    if not rows:
        raise ValueError(f"Preset {dataset_id} produced no rows")
    holdout_n = min(extra, max(0, len(rows) - 1))
    holdout = rows[-holdout_n:] if holdout_n else []
    train_rows = rows[:-holdout_n] if holdout_n else rows
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"Materialized {len(train_rows)} train / {len(holdout)} holdout rows for {dataset_id}")
    return destination, holdout


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {key: value.cuda() for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def score_holdout(model_path: str, holdout: list[dict], log: LogFn) -> float:
    if not holdout:
        return 0.0
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from finetuner.datasets.hf_datasets import extract_gsm8k_answer
    from finetuner.training.dataset_formats import split_instruction_response

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    correct = 0
    for row in holdout:
        text = str(row.get("text", ""))
        prompt, gold = split_instruction_response(text)
        if not gold:
            continue
        generated = _generate(model, tokenizer, prompt)
        expected = extract_gsm8k_answer(gold)
        predicted = extract_gsm8k_answer(generated)
        if expected and predicted and expected.strip() == predicted.strip():
            correct += 1
        elif expected and expected.lower() in generated.lower():
            correct += 1
    del model
    gc.collect()
    score = 100.0 * correct / len(holdout)
    log(f"Holdout: {correct}/{len(holdout)} = {score:.1f}%")
    return score


def score_reward_ranking(model_path: str, dataset_path: str, log: LogFn, max_pairs: int = 16) -> float:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from finetuner.training.common import load_raw_dataset
    from finetuner.training.dataset_formats import prepare_method_dataset

    raw = load_raw_dataset(dataset_path)
    pairs = prepare_method_dataset(raw, "reward", allow_synthetic_preferences=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    wins = 0
    total = min(max_pairs, len(pairs))
    for index in range(total):
        row = pairs[index]
        chosen = tokenizer(
            f"{row['prompt']}{row['chosen']}",
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        rejected = tokenizer(
            f"{row['prompt']}{row['rejected']}",
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        if torch.cuda.is_available():
            chosen = {key: value.cuda() for key, value in chosen.items()}
            rejected = {key: value.cuda() for key, value in rejected.items()}
        with torch.no_grad():
            chosen_score = float(model(**chosen).logits.squeeze())
            rejected_score = float(model(**rejected).logits.squeeze())
        if chosen_score > rejected_score:
            wins += 1
    del model
    gc.collect()
    score = 100.0 * wins / total if total else 0.0
    log(f"Reward ranking: {wins}/{total} = {score:.1f}%")
    return score


def _score_policy(
    model_path: str,
    eval_ids: tuple[str, ...],
    holdout: list[dict],
    eval_samples: int,
    log: LogFn,
    accelerator: str = "cuda",
    npu_artifact_path: str = "",
) -> dict[str, float]:
    scores: dict[str, float] = {}
    if accelerator in {"npu", "tpu", "cpu"}:
        from finetuner.training.accel.evaluate import score_holdout_accel

        adapter_dir = model_path if (Path(model_path) / "npu_adapter.json").is_file() else ""
        if holdout:
            scores["holdout"] = score_holdout_accel(
                holdout,
                artifact_path=npu_artifact_path or model_path,
                adapter_dir=adapter_dir,
                log=log,
            )
        return scores

    from finetuner.eval.runner import run_evals

    official = [task_id for task_id in eval_ids if task_id in EVAL_TASKS]
    if official:
        try:
            results = run_evals(model_path, official, eval_samples, log)
            scores.update({item.task_id: item.score for item in results})
        except Exception as exc:
            log(f"Official eval failed ({exc}); using holdout only.")
    if holdout:
        scores["holdout"] = score_holdout(model_path, holdout, log)
    return scores


def _score_reward(config: SweepConfig, artifact: str, dataset_path: str, log: LogFn) -> float:
    if _uses_accel(config):
        from finetuner.training.accel.evaluate import score_reward_ranking_accel
        from finetuner.training.dataset_formats import prepare_method_dataset

        path = Path(dataset_path)
        raw = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        pairs = prepare_method_dataset(raw, "reward", allow_synthetic_preferences=True)
        return score_reward_ranking_accel(
            list(pairs),
            artifact_path=config.npu_artifact_path or artifact,
            adapter_dir=artifact,
            log=log,
        )
    return score_reward_ranking(artifact, dataset_path, log)


def _uses_accel(config: SweepConfig) -> bool:
    return (config.accelerator or "cuda") in {"npu", "tpu", "cpu"}


def resolve_model_path(config: SweepConfig, log: LogFn) -> str:
    if _uses_accel(config):
        from finetuner.training.npu.runtime import resolve_npu_artifact

        try:
            artifact = resolve_npu_artifact(config.npu_artifact_path or config.model_path)
            log(f"Accel frozen decoder: {artifact}")
            return str(artifact)
        except FileNotFoundError:
            if config.model_path:
                path = Path(config.model_path)
                if path.exists():
                    return str(path)
            raise
    if config.model_path:
        path = Path(config.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Local model path does not exist: {path}")
        return str(path)
    from finetuner.core.hf_download import download_hf_model
    from finetuner.core.model_catalog import find_downloaded_model

    cached = find_downloaded_model(config.model_id)
    if cached:
        log(f"Using cached model {cached}")
        return cached
    dest = model_download_path(config.model_id)
    log(f"Downloading {config.model_id}")
    download_hf_model(config.model_id, dest, on_log=log)
    return str(dest)


def _training_config(config: SweepConfig, cell: SweepCell, reward_model_id: str = "") -> TrainingConfig:
    seq_length = min(128, config.max_seq_length) if cell.is_online else config.max_seq_length
    accum = config.gradient_accumulation_steps
    generations = 2
    if cell.is_online and (config.batch_size * accum) % generations != 0:
        accum = generations
    return TrainingConfig(
        training_method=cell.method,
        reward_function="exact_match",
        reward_model_id=reward_model_id,
        max_steps=cell.max_steps,
        learning_rate=config.learning_rate,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        batch_size=config.batch_size,
        gradient_accumulation_steps=accum,
        max_seq_length=seq_length,
        use_qlora=False,
        allow_synthetic_preferences=cell.method in PREFERENCE_METHODS,
        grpo_num_generations=2,
        accelerator=config.accelerator,
        npu_artifact_path=config.npu_artifact_path,
        seed=42,
    )


def _release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def run_cell(
    cell: SweepCell,
    *,
    config: SweepConfig,
    model_path: str,
    dataset_path: Path,
    holdout: list[dict],
    output_dir: Path,
    baseline_scores: dict[str, float],
    reward_model_id: str,
    log: LogFn,
) -> CellResult:
    from finetuner.training.runner import train

    result = CellResult(
        method=cell.method,
        dataset=cell.dataset,
        status="running",
        baseline_scores=dict(baseline_scores),
    )
    cell_dir = output_dir / cell.dataset / cell.method
    cell_dir.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []

    def captured(message: str) -> None:
        logs.append(message)
        log(message)

    class _Tee:
        def __init__(self, stream) -> None:
            self.stream = stream

        def write(self, chunk: str) -> int:
            if chunk:
                logs.append(chunk)
            return self.stream.write(chunk)

        def flush(self) -> None:
            self.stream.flush()

        def __getattr__(self, name: str):
            return getattr(self.stream, name)

    monitor = ResourceMonitor()
    monitor.start()
    started = time.monotonic()
    stdout = sys.stdout
    try:
        sys.stdout = _Tee(stdout)  # type: ignore[assignment]
        training = _training_config(config, cell, reward_model_id)
        artifact = train(
            model_path=model_path,
            output_dir=str(cell_dir / "train"),
            training=training,
            dataset_path=str(dataset_path),
            log_callback=captured,
        )
        result.output_path = artifact
        result.train_seconds = round(time.monotonic() - started, 3)
        result.trainable_params = parse_trainable_params("\n".join(logs))
        loss, steps = parse_trainer_state(cell_dir / "train")
        result.final_loss = loss
        result.steps = steps or cell.max_steps
        result.examples = training.max_steps * training.batch_size * training.gradient_accumulation_steps
        adapter = Path(artifact).parent / "adapter"
        result.artifact_bytes = directory_bytes(adapter if adapter.is_dir() else Path(artifact))
        eval_started = time.monotonic()
        if cell.method == "reward":
            result.scores = {
                "reward_ranking": _score_reward(config, artifact, str(dataset_path), captured)
            }
        elif cell.method in POLICY_METHODS:
            result.scores = _score_policy(
                artifact,
                cell.eval_ids,
                holdout,
                config.eval_samples,
                captured,
                accelerator=config.accelerator,
                npu_artifact_path=config.npu_artifact_path or model_path,
            )
        result.eval_seconds = round(time.monotonic() - eval_started, 3)
        result.status = "completed"
    except Exception as exc:
        result.train_seconds = round(time.monotonic() - started, 3)
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {exc}"
        captured(f"ERROR {cell.key}: {result.error}")
    finally:
        sys.stdout = stdout
        result.peak_ram_gb, result.peak_gpu_gb = monitor.stop()
        _release_memory()
    return finalize_cell(result)


def run_sweep(config: SweepConfig, log: LogFn | None = None) -> dict:
    if config.allow_cpu:
        os.environ["FINETUNER_ALLOW_CPU_TRAIN"] = "1"

    output_dir = Path(config.output_dir) if config.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    report_path = output_dir / "report.md"
    baseline_path = output_dir / "baselines.json"
    log_path = output_dir / "sweep.log"
    try:
        log_handle = log_path.open("a", encoding="utf-8")
    except OSError:
        log_handle = (output_dir / "sweep-run.log").open("a", encoding="utf-8")

    def logger(message: str) -> None:
        print(message, flush=True)
        try:
            log_handle.write(message + "\n")
            log_handle.flush()
        except OSError:
            pass
        if log:
            try:
                log(message)
            except Exception:
                pass
    cells = plan_cells(config)
    logger(f"Sweep: {len(cells)} cells -> {output_dir}")

    model_path = resolve_model_path(config, logger)
    existing = _load_results(results_path) if config.resume else {}
    baselines: dict[str, dict[str, float]] = {}
    if config.resume and baseline_path.is_file():
        try:
            loaded = json.loads(baseline_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                baselines.update({key: dict(value) for key, value in loaded.items() if isinstance(value, dict)})
        except (OSError, json.JSONDecodeError):
            pass
    reward_by_dataset: dict[str, str] = {}
    holdouts: dict[str, list[dict]] = {}
    dataset_paths: dict[str, Path] = {}
    completed: list[CellResult] = []

    def persist() -> None:
        payload = {
            "model_id": config.model_id,
            "model_path": model_path,
            "device": config.accelerator if _uses_accel(config) else ("cpu" if config.allow_cpu else "cuda"),
            "results": [item.to_dict() for item in completed],
        }
        _write_json(results_path, payload)
        report_path.write_text(
            render_markdown(
                completed,
                title="Finetune method efficiency sweep",
                device=payload["device"],
            ),
            encoding="utf-8",
        )

    for dataset_id in config.datasets:
        data_path, holdout = materialize_dataset(
            dataset_id,
            output_dir / "data" / f"{dataset_id}.jsonl",
            max_samples=config.max_samples,
            holdout_samples=config.holdout_samples,
            bundled_only=config.bundled_only,
            log=logger,
        )
        dataset_paths[dataset_id] = data_path
        holdouts[dataset_id] = holdout
        cached_baseline = baselines.get(dataset_id) or next(
            (
                item.baseline_scores
                for item in existing.values()
                if item.dataset == dataset_id and item.baseline_scores
            ),
            None,
        )
        if cached_baseline:
            baselines[dataset_id] = cached_baseline
            logger(f"Reusing baseline scores for {dataset_id}")
        else:
            logger(f"Evaluating baseline on {dataset_id}")
            baseline_evals = (
                (related_eval_map([dataset_id]).get(dataset_id, ""),)
                if config.matching_eval_only
                else official_eval_ids(dataset_id, config.evals)
            )
            baseline_evals = tuple(task for task in baseline_evals if task in EVAL_TASKS)
            baselines[dataset_id] = _score_policy(
                model_path,
                baseline_evals,
                holdout,
                config.eval_samples,
                logger,
                accelerator=config.accelerator,
                npu_artifact_path=config.npu_artifact_path or model_path,
            )
            _release_memory()
        _write_json(baseline_path, baselines)

    for cell in cells:
        key = cell.key
        prior = existing.get(key)
        if prior and prior.status == "completed" and config.resume:
            logger(f"Skipping completed {key}")
            completed.append(prior)
            if prior.method == "reward" and prior.output_path:
                reward_by_dataset[cell.dataset] = prior.output_path
            persist()
            continue
        logger(f"\n=== {key} ({cell.max_steps} steps) ===")
        result = run_cell(
            cell,
            config=config,
            model_path=model_path,
            dataset_path=dataset_paths[cell.dataset],
            holdout=holdouts[cell.dataset],
            output_dir=output_dir,
            baseline_scores=baselines[cell.dataset],
            reward_model_id=reward_by_dataset.get(cell.dataset, ""),
            log=logger,
        )
        completed.append(result)
        if result.status == "completed" and result.method == "reward" and result.output_path:
            reward_by_dataset[cell.dataset] = result.output_path
        persist()

    payload = {
        "model_id": config.model_id,
        "model_path": model_path,
        "device": config.accelerator if _uses_accel(config) else ("cpu" if config.allow_cpu else "cuda"),
        "output_dir": str(output_dir),
        "results": [item.to_dict() for item in completed],
        "report_path": str(report_path),
    }
    persist()
    logger(f"Wrote {results_path}")
    logger(f"Wrote {report_path}")
    log_handle.close()
    return payload
