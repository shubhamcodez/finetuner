"""Trending Hugging Face datasets that are usable for finetuning."""

from __future__ import annotations

from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class HubDataset:
    repo_id: str
    name: str


# Shown immediately, and used when the Hub is unreachable.
FEATURED_DATASETS: tuple[HubDataset, ...] = (
    HubDataset("tatsu-lab/alpaca", "Alpaca"),
    HubDataset("yahma/alpaca-cleaned", "Alpaca Cleaned"),
    HubDataset("databricks/databricks-dolly-15k", "Dolly 15k"),
    HubDataset("HuggingFaceH4/no_robots", "No Robots"),
    HubDataset("HuggingFaceH4/ultrachat_200k", "UltraChat 200k"),
    HubDataset("HuggingFaceTB/smoltalk", "SmolTalk"),
    HubDataset("Open-Orca/OpenOrca", "OpenOrca"),
    HubDataset("teknium/OpenHermes-2.5", "OpenHermes 2.5"),
    HubDataset("OpenAssistant/oasst1", "OpenAssistant OASST1"),
    HubDataset("HuggingFaceH4/ultrafeedback_binarized", "UltraFeedback"),
    HubDataset("mlabonne/FineTome-100k", "FineTome 100k"),
    HubDataset("nvidia/HelpSteer2", "HelpSteer2"),
)

_SKIP_HINTS = ("image", "audio", "video", "speech", "diffusion", "wav2vec")
_FINETUNE_FILTERS = ("text-generation", "conversational")


def _short_name(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1].replace("_", " ").replace("-", " ")


def _usable(repo_id: str) -> bool:
    lowered = repo_id.casefold()
    return not any(hint in lowered for hint in _SKIP_HINTS)


def _query_hub(limit: int, token: str) -> list[HubDataset]:
    from huggingface_hub import HfApi

    api = HfApi(token=token or None)
    found: list[HubDataset] = []
    seen: set[str] = set()
    for tag in _FINETUNE_FILTERS:
        if len(found) >= limit:
            break
        try:
            iterator = api.list_datasets(
                filter=tag,
                sort="trendingScore",
                direction=-1,
                limit=limit,
            )
        except TypeError:
            iterator = api.list_datasets(sort="trendingScore", direction=-1, limit=limit)
        for info in iterator:
            repo_id = str(getattr(info, "id", "")).strip()
            if not repo_id or repo_id in seen or not _usable(repo_id):
                continue
            seen.add(repo_id)
            found.append(HubDataset(repo_id, _short_name(repo_id)))
            if len(found) >= limit:
                break
    return found


def fetch_trending_datasets(
    limit: int = 16,
    token: str = "",
    timeout_s: float = 8.0,
) -> list[HubDataset]:
    """Return live trending text datasets, or featured finetune sets if the Hub is down."""
    collected: list[HubDataset] = []

    def work() -> None:
        try:
            collected.extend(_query_hub(limit, token))
        except Exception:
            return

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if collected:
        return collected
    return list(FEATURED_DATASETS)


def hub_preset_id(repo_id: str) -> str:
    return f"hf:{repo_id.strip()}"
