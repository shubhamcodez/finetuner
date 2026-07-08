from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    name: str
    description: str
    lighteval_task: str
    metric: str = "acc"


EVAL_TASKS: dict[str, EvalTask] = {
    "mmlu": EvalTask(
        task_id="mmlu",
        name="MMLU",
        description="Massive Multitask Language Understanding (5-shot subset)",
        lighteval_task="leaderboard|mmlu|5",
        metric="acc",
    ),
    "gsm8k": EvalTask(
        task_id="gsm8k",
        name="GSM8K",
        description="Grade school math word problems (8-shot)",
        lighteval_task="lighteval|gsm8k|8",
        metric="acc",
    ),
    "hellaswag": EvalTask(
        task_id="hellaswag",
        name="HellaSwag",
        description="Commonsense NLI (10-shot)",
        lighteval_task="leaderboard|hellaswag|10",
        metric="acc",
    ),
    "arc_challenge": EvalTask(
        task_id="arc_challenge",
        name="ARC Challenge",
        description="AI2 Reasoning Challenge — challenge split (25-shot)",
        lighteval_task="leaderboard|arc:challenge|25",
        metric="acc",
    ),
}


def task_list_string(task_ids: list[str]) -> str:
    tasks = [EVAL_TASKS[tid].lighteval_task for tid in task_ids if tid in EVAL_TASKS]
    return ",".join(tasks)
