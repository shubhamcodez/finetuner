"""Trending Hugging Face text-generation models for the Add Model picker."""

from __future__ import annotations

from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class HubModel:
    repo_id: str
    name: str


# Shown immediately, and used when the Hub is unreachable.
FEATURED_MODELS: tuple[HubModel, ...] = (
    HubModel("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5 0.5B Instruct"),
    HubModel("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5 3B Instruct"),
    HubModel("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5 7B Instruct"),
    HubModel("HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM2 1.7B Instruct"),
    HubModel("microsoft/Phi-4-mini-instruct", "Phi-4 mini Instruct"),
    HubModel("google/gemma-3-1b-it", "Gemma 3 1B"),
    HubModel("meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B Instruct"),
    HubModel("mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B Instruct"),
)


def _short_name(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1].replace("-", " ")


def _query_hub(limit: int, token: str) -> list[HubModel]:
    from huggingface_hub import HfApi

    api = HfApi(token=token or None)
    found: list[HubModel] = []
    seen: set[str] = set()
    iterator = api.list_models(
        task="text-generation",
        sort="trendingScore",
        direction=-1,
        limit=limit,
    )
    for info in iterator:
        repo_id = str(getattr(info, "id", "") or getattr(info, "modelId", "")).strip()
        if not repo_id or repo_id in seen:
            continue
        seen.add(repo_id)
        found.append(HubModel(repo_id, _short_name(repo_id)))
        if len(found) >= limit:
            break
    return found


def fetch_trending_models(
    limit: int = 12,
    token: str = "",
    timeout_s: float = 8.0,
) -> list[HubModel]:
    """Return live trending text-generation models, or featured models if the Hub is down."""
    collected: list[HubModel] = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            collected.extend(_query_hub(limit, token))
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if collected:
        return collected
    return list(FEATURED_MODELS)
