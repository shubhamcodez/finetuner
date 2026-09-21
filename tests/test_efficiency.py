from __future__ import annotations

import json

from finetuner.training.sweep import SweepConfig, official_eval_ids, plan_cells
from finetuner.training.efficiency import (
    CellResult,
    PAGE_METHODS,
    build_cells,
    delta_scores,
    directory_bytes,
    evals_for_dataset,
    finalize_cell,
    parse_trainable_params,
    parse_trainer_state,
    rank_cells,
    render_markdown,
    run_order,
)


def test_page_methods_match_finetune_tab_order():
    assert PAGE_METHODS == ("sft", "dpo", "grpo", "ppo", "kto", "reward", "orpo", "rloo")


def test_build_cells_uses_shorter_online_budgets():
    cells = build_cells(
        ["sft", "grpo"],
        ["gsm8k"],
        ["gsm8k", "hellaswag"],
        related_evals={"gsm8k": "gsm8k"},
        steps_offline=8,
        steps_online=4,
    )
    by_method = {cell.method: cell for cell in cells}
    assert by_method["sft"].max_steps == 8
    assert by_method["grpo"].max_steps == 4
    assert by_method["sft"].eval_ids[0] == "gsm8k"
    assert "hellaswag" in by_method["sft"].eval_ids


def test_run_order_trains_reward_before_ppo():
    cells = build_cells(
        list(PAGE_METHODS),
        ["gsm8k"],
        ["gsm8k"],
        related_evals={"gsm8k": "gsm8k"},
    )
    ordered = [cell.method for cell in run_order(cells)]
    assert ordered.index("reward") < ordered.index("ppo")
    assert ordered.count("reward") == 1


def test_evals_for_dataset_puts_related_task_first():
    assert evals_for_dataset("hellaswag", ["gsm8k", "hellaswag"], "hellaswag") == (
        "hellaswag",
        "gsm8k",
    )


def test_finalize_cell_tracks_quality_per_compute():
    result = finalize_cell(
        CellResult(
            method="sft",
            dataset="gsm8k",
            status="completed",
            train_seconds=1800,
            peak_ram_gb=4.0,
            steps=10,
            examples=40,
            scores={"gsm8k": 12.0, "holdout": 25.0},
            baseline_scores={"gsm8k": 8.0, "holdout": 15.0},
        )
    )
    assert result.delta_scores == {"gsm8k": 4.0, "holdout": 10.0}
    assert result.mean_delta == 7.0
    assert result.delta_per_hour == 14.0
    assert result.delta_per_step == 0.7
    assert result.sample_efficiency == 0.175
    assert result.steps_per_sec == 10 / 1800


def test_parse_trainer_state_and_trainable_params(tmp_path):
    (tmp_path / "trainer_state.json").write_text(
        json.dumps({"global_step": 8, "log_history": [{"loss": 2.5}, {"loss": 1.25}]}),
        encoding="utf-8",
    )
    loss, steps = parse_trainer_state(tmp_path)
    assert loss == 1.25
    assert steps == 8
    assert parse_trainable_params("trainable params: 12345 || all params: 99999") == 12345
    assert parse_trainable_params("trainable params: 1,234,567 || all params: 9,999,999") == 1234567


def test_directory_bytes_and_ranking(tmp_path):
    blob = tmp_path / "adapter.bin"
    blob.write_bytes(b"abcd")
    assert directory_bytes(tmp_path) == 4
    weak = finalize_cell(
        CellResult("dpo", "gsm8k", "completed", train_seconds=3600, scores={"a": 1}, baseline_scores={"a": 0})
    )
    strong = finalize_cell(
        CellResult("sft", "gsm8k", "completed", train_seconds=1800, scores={"a": 4}, baseline_scores={"a": 0})
    )
    failed = CellResult("ppo", "gsm8k", "failed")
    ranked = rank_cells([failed, weak, strong])
    assert [item.method for item in ranked] == ["sft", "dpo"]
    report = render_markdown(ranked, title="Sweep", device="cpu")
    assert "Most efficient" in report
    assert "sft" in report


def test_delta_scores_ignore_missing_baseline():
    assert delta_scores({"gsm8k": 5.0}, {}) == {"gsm8k": 5.0}
    assert "n" not in delta_scores({"accuracy": 10.0, "n": 120.0}, {"accuracy": 8.0, "n": 120.0})


def test_finalize_ignores_logprob_when_averaging_quality():
    result = finalize_cell(
        CellResult(
            method="sft",
            dataset="gsm8k",
            status="completed",
            train_seconds=3600,
            steps=10,
            examples=10,
            scores={"accuracy": 12.0, "gold_logprob": -1.0, "n": 100.0},
            baseline_scores={"accuracy": 10.0, "gold_logprob": -2.0, "n": 100.0},
        )
    )
    assert result.delta_scores["accuracy"] == 2.0
    assert result.delta_scores["gold_logprob"] == 1.0
    assert result.mean_delta == 2.0


def test_plan_cells_covers_each_method_once_per_dataset():
    cells = plan_cells(
        SweepConfig(methods=list(PAGE_METHODS), datasets=["gsm8k", "hellaswag"], evals=["gsm8k", "hellaswag"])
    )
    keys = [cell.key for cell in cells]
    assert len(keys) == 16
    assert len(set(keys)) == 16
    assert official_eval_ids("gsm8k", ["hellaswag", "gsm8k"])[0] == "gsm8k"
    matching = plan_cells(
        SweepConfig(
            methods=["sft"],
            datasets=["gsm8k", "hellaswag"],
            evals=["gsm8k", "hellaswag", "arc_challenge"],
            matching_eval_only=True,
        )
    )
    assert matching[0].eval_ids == ("gsm8k",)
    assert matching[1].eval_ids == ("hellaswag",)
