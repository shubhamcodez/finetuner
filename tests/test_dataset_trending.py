from __future__ import annotations

from finetuner.datasets.rows import row_to_text
from finetuner.datasets.presets import get_preset, hub_repo_id
from finetuner.datasets.trending import FEATURED_DATASETS, HubDataset, fetch_trending_datasets


def test_featured_finetune_datasets_are_hub_repos():
    assert len(FEATURED_DATASETS) >= 8
    for dataset in FEATURED_DATASETS:
        assert "/" in dataset.repo_id


def test_hub_preset_resolves_for_loader():
    preset = get_preset("hf:tatsu-lab/alpaca")
    assert preset is not None
    assert preset.hf_dataset == "tatsu-lab/alpaca"
    assert hub_repo_id("tatsu-lab/alpaca") == "tatsu-lab/alpaca"


def test_row_to_text_covers_common_finetune_schemas():
    assert "Paris" in (row_to_text({"instruction": "Capital?", "output": "Paris"}) or "")
    assert "hello" in (row_to_text({"prompt": "Say hi", "response": "hello"}) or "")
    assert "user" in (row_to_text({"messages": [{"role": "user", "content": "Hi"}]}) or "")
    assert "assistant" in (
        row_to_text(
            {"conversations": [{"from": "human", "value": "Q"}, {"from": "assistant", "value": "A"}]}
        )
        or ""
    )


def test_fetch_datasets_falls_back_when_hub_fails(monkeypatch):
    def broken(_limit: int, _token: str):
        raise RuntimeError("offline")

    monkeypatch.setattr("finetuner.datasets.trending._query_hub", broken)
    assert fetch_trending_datasets(timeout_s=0.4) == list(FEATURED_DATASETS)


def test_fetch_datasets_returns_live_trending(monkeypatch):
    live = [HubDataset("org/hot-sft", "hot sft")]
    monkeypatch.setattr("finetuner.datasets.trending._query_hub", lambda _limit, _token: live)
    assert fetch_trending_datasets(timeout_s=2) == live
