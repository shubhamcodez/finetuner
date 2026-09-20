from __future__ import annotations

from types import SimpleNamespace

from finetuner.core.hf_trending import (
    FEATURED_MODELS,
    HubModel,
    fetch_trending_models,
    format_count,
    hub_download_total,
    infer_capabilities,
    model_from_hub_info,
)


def test_featured_models_are_huggingface_repos():
    assert len(FEATURED_MODELS) >= 4
    for model in FEATURED_MODELS:
        assert "/" in model.repo_id
        assert model.name
        assert model.description
        assert model.params
        assert model.license


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


def test_hub_download_total_prefers_all_time():
    info = SimpleNamespace(downloads=120, downloadsAllTime=1_250_000, downloads_all_time=None)
    assert hub_download_total(info) == 1_250_000
    assert hub_download_total(SimpleNamespace(downloads=88)) == 88


def test_format_count_uses_compact_suffixes():
    assert format_count(0) == ""
    assert format_count(42) == "42"
    assert format_count(1500) == "1.5k"
    assert format_count(2_000_000) == "2M"


def test_model_from_hub_info_reads_card_details():
    info = SimpleNamespace(
        id="org/live-model",
        author="org",
        downloads=12_400,
        likes=880,
        pipeline_tag="text-generation",
        tags=("instruct", "chat"),
        license="",
        card_data={"description": "A live trending checkpoint.", "license": "apache-2.0"},
        lastModified="2026-09-01",
    )
    model = model_from_hub_info(info)
    assert model is not None
    assert model.repo_id == "org/live-model"
    assert model.description.startswith("A live")
    assert model.downloads == 12_400
    assert model.likes == 880
    assert model.license == "apache-2.0"
    assert model.org == "org"


def test_infer_capabilities_from_hub_signals():
    assert infer_capabilities(tags=("image-text-to-text",), repo_id="org/qwen2.5-vl-3b") == (
        "vision",
    )
    assert infer_capabilities(tags=("function-calling",)) == ("tools",)
    assert infer_capabilities(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B") == ("reasoning",)


def test_featured_models_include_capability_pills():
    shown = {model.repo_id: model.shown_capabilities() for model in FEATURED_MODELS}
    assert "vision" in shown["Qwen/Qwen2.5-VL-3B-Instruct"]
    assert "tools" in shown["Qwen/Qwen2.5-VL-3B-Instruct"]
    assert "tools" in shown["Qwen/Qwen2.5-0.5B-Instruct"]
    assert "reasoning" in shown["microsoft/Phi-4-mini-instruct"]
