from __future__ import annotations

from finetuner.core.hf_trending import FEATURED_MODELS, HubModel, fetch_trending_models


def test_featured_models_are_huggingface_repos():
    assert len(FEATURED_MODELS) >= 4
    for model in FEATURED_MODELS:
        assert "/" in model.repo_id
        assert model.name


def test_fetch_falls_back_when_hub_fails(monkeypatch):
    def broken(_limit: int, _token: str):
        raise RuntimeError("offline")

    monkeypatch.setattr("finetuner.core.hf_trending._query_hub", broken)
    models = fetch_trending_models(timeout_s=0.5)
    assert models == list(FEATURED_MODELS)


def test_fetch_returns_live_trending(monkeypatch):
    live = [HubModel("org/hot-model", "hot model")]
    monkeypatch.setattr("finetuner.core.hf_trending._query_hub", lambda _limit, _token: live)
    assert fetch_trending_models(timeout_s=2) == live
