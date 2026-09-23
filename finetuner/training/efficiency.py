from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


PAGE_METHODS = ("sft", "dpo", "grpo", "ppo", "kto", "reward", "orpo", "rloo")
ONLINE_METHODS = frozenset({"grpo", "ppo", "rloo"})
PREFERENCE_METHODS = frozenset({"dpo", "kto", "orpo", "reward"})
POLICY_METHODS = frozenset({"sft", "dpo", "grpo", "ppo", "kto", "orpo", "rloo"})


@dataclass(frozen=True)
class SweepCell:
    method: str
    dataset: str
    eval_ids: tuple[str, ...]
    max_steps: int

    @property
    def key(self) -> str:
        return f"{self.method}:{self.dataset}"

    @property
    def is_online(self) -> bool:
        return self.method in ONLINE_METHODS

    @property
    def produces_policy(self) -> bool:
        return self.method in POLICY_METHODS


@dataclass
class CellResult:
    method: str
    dataset: str
    status: str
    error: str = ""
    output_path: str = ""
    train_seconds: float = 0.0
    eval_seconds: float = 0.0
    peak_ram_gb: float = 0.0
    peak_gpu_gb: float = 0.0
    trainable_params: int = 0
    artifact_bytes: int = 0
    steps: int = 0
    examples: int = 0
    steps_per_sec: float = 0.0
    examples_per_sec: float = 0.0
    final_loss: float | None = None
    scores: dict[str, float] = field(default_factory=dict)
    baseline_scores: dict[str, float] = field(default_factory=dict)
    delta_scores: dict[str, float] = field(default_factory=dict)
    mean_delta: float = 0.0
    delta_per_hour: float = 0.0
    delta_per_step: float = 0.0
    delta_per_gb: float = 0.0
    sample_efficiency: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def evals_for_dataset(dataset_id: str, eval_ids: list[str], related_eval_id: str = "") -> tuple[str, ...]:
    ordered: list[str] = []
    if related_eval_id and related_eval_id in eval_ids:
        ordered.append(related_eval_id)
    for task_id in eval_ids:
        if task_id not in ordered:
            ordered.append(task_id)
    return tuple(ordered)


def build_cells(
    methods: list[str],
    datasets: list[str],
    eval_ids: list[str],
    *,
    related_evals: dict[str, str] | None = None,
    steps_offline: int = 8,
    steps_online: int = 4,
) -> list[SweepCell]:
    related = related_evals or {}
    unknown = [method for method in methods if method not in PAGE_METHODS]
    if unknown:
        raise ValueError(f"Unknown training methods: {', '.join(unknown)}")
    cells: list[SweepCell] = []
    for dataset_id in datasets:
        dataset_evals = evals_for_dataset(dataset_id, eval_ids, related.get(dataset_id, ""))
        for method in methods:
            cells.append(
                SweepCell(
                    method=method,
                    dataset=dataset_id,
                    eval_ids=dataset_evals,
                    max_steps=steps_online if method in ONLINE_METHODS else steps_offline,
                )
            )
    return cells


def run_order(cells: list[SweepCell]) -> list[SweepCell]:
    """Keep dataset groups, but train a reward model before PPO on the same data."""
    by_dataset: dict[str, list[SweepCell]] = {}
    for cell in cells:
        by_dataset.setdefault(cell.dataset, []).append(cell)
    ordered: list[SweepCell] = []
    for dataset_cells in by_dataset.values():
        reward = next((cell for cell in dataset_cells if cell.method == "reward"), None)
        has_ppo = any(cell.method == "ppo" for cell in dataset_cells)
        if reward and has_ppo:
            for cell in dataset_cells:
                if cell.method == "reward":
                    continue
                if cell.method == "ppo":
                    ordered.append(reward)
                ordered.append(cell)
        else:
            ordered.extend(dataset_cells)
    return ordered


def parse_trainer_state(run_dir: Path) -> tuple[float | None, int]:
    for candidate in (run_dir, *run_dir.glob("**")):
        state_path = candidate / "trainer_state.json" if candidate.is_dir() else None
        if state_path is None or not state_path.is_file():
            continue
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        loss = None
        for row in reversed(payload.get("log_history") or []):
            if isinstance(row, dict) and "loss" in row:
                try:
                    loss = float(row["loss"])
                except (TypeError, ValueError):
                    loss = None
                break
        try:
            steps = int(payload.get("global_step") or 0)
        except (TypeError, ValueError):
            steps = 0
        return loss, steps
    return None, 0


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def parse_trainable_params(log_text: str) -> int:
    for line in log_text.splitlines():
        lowered = line.lower().replace(",", "")
        if "trainable params" not in lowered:
            continue
        after = lowered.split("trainable params", 1)[-1]
        number = ""
        for ch in after:
            if ch.isdigit():
                number += ch
            elif number:
                break
        if number:
            return int(number)
    return 0


_NON_QUALITY = {"n", "gold_logprob", "token_acc"}


def delta_scores(scores: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        task_id: float(scores[task_id]) - float(baseline.get(task_id, 0.0))
        for task_id in scores
        if task_id != "n"
    }


def finalize_cell(result: CellResult) -> CellResult:
    result.delta_scores = delta_scores(result.scores, result.baseline_scores)
    deltas = [value for key, value in result.delta_scores.items() if key not in _NON_QUALITY]
    result.mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    hours = result.train_seconds / 3600.0
    result.delta_per_hour = result.mean_delta / hours if hours else 0.0
    result.delta_per_step = result.mean_delta / result.steps if result.steps else 0.0
    memory = max(result.peak_gpu_gb, result.peak_ram_gb, 0.01)
    result.delta_per_gb = result.mean_delta / memory
    result.sample_efficiency = (
        result.mean_delta / result.examples if result.examples else 0.0
    )
    result.steps_per_sec = result.steps / result.train_seconds if result.train_seconds else 0.0
    result.examples_per_sec = (
        result.examples / result.train_seconds if result.train_seconds else 0.0
    )
    return result


def rank_cells(results: list[CellResult], key: str = "delta_per_hour") -> list[CellResult]:
    completed = [item for item in results if item.status == "completed"]
    return sorted(completed, key=lambda item: getattr(item, key), reverse=True)


def render_markdown(results: list[CellResult], *, title: str, device: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"Device: `{device}`",
        "",
        "Before = base model. After = the same method trained on that benchmark's data.",
        "Accuracy is generate match (GSM8K exact / HellaSwag+ARC choice). Reward rows use chosen-vs-rejected ranking.",
        "",
        "| Method | Dataset | N | Before acc | After acc | Δ acc | Before logp | After logp | Δ logp | Train s | Loss |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        loss = "" if item.final_loss is None else f"{item.final_loss:.4f}"
        after_acc = item.scores.get("accuracy", item.scores.get("reward_ranking", item.scores.get("holdout", 0.0)))
        before_acc = item.baseline_scores.get(
            "accuracy", item.baseline_scores.get("reward_ranking", item.baseline_scores.get("holdout", 0.0))
        )
        after_lp = item.scores.get("gold_logprob")
        before_lp = item.baseline_scores.get("gold_logprob")
        lines.append(
            "| {method} | {dataset} | {n:.0f} | {before:.1f} | {after:.1f} | {delta:.1f} | {blp} | {alp} | {dlp} | {train:.1f} | {loss} |".format(
                method=item.method,
                dataset=item.dataset,
                n=item.scores.get("n", item.baseline_scores.get("n", 0.0)),
                before=float(before_acc or 0.0),
                after=float(after_acc or 0.0),
                delta=float(after_acc or 0.0) - float(before_acc or 0.0),
                blp="" if before_lp is None else f"{before_lp:.3f}",
                alp="" if after_lp is None else f"{after_lp:.3f}",
                dlp=""
                if before_lp is None or after_lp is None
                else f"{float(after_lp) - float(before_lp):.3f}",
                train=item.train_seconds,
                loss=loss,
            )
        )
    ranked = rank_cells(results)
    if ranked:
        winner = ranked[0]
        lines.extend(
            [
                "",
                "## Most efficient completed run",
                "",
                f"**{winner.method}** on **{winner.dataset}** "
                f"({winner.mean_delta:.2f} Δacc, {winner.delta_per_hour:.2f} Δ/hour).",
            ]
        )
    failed = [item for item in results if item.status == "failed"]
    if failed:
        lines.extend(["", "## Failures", ""])
        for item in failed:
            lines.append(f"- `{item.method}` / `{item.dataset}`: {item.error}")
    return "\n".join(lines) + "\n"
