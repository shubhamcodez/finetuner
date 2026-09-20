"""Trending Hugging Face text-generation models for the Add Model picker."""

from __future__ import annotations

from dataclasses import dataclass, replace
import threading


@dataclass(frozen=True)
class HubModel:
    repo_id: str
    name: str
    author: str = ""
    description: str = ""
    downloads: int = 0
    likes: int = 0
    license: str = ""
    pipeline: str = "text-generation"
    tags: tuple[str, ...] = ()
    params: str = ""
    updated: str = ""
    capabilities: tuple[str, ...] = ()

    @property
    def org(self) -> str:
        if self.author:
            return self.author
        if "/" in self.repo_id:
            return self.repo_id.split("/", 1)[0]
        return ""

    def shown_capabilities(self) -> tuple[str, ...]:
        if self.capabilities:
            return tuple(name for name in CAPABILITY_ORDER if name in self.capabilities)
        return infer_capabilities(self.tags, self.pipeline, self.repo_id, self.description)


# Shown immediately, and used when the Hub is unreachable.
FEATURED_MODELS: tuple[HubModel, ...] = (
    HubModel(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen2.5 0.5B Instruct",
        author="Qwen",
        description="Instruction-tuned 0.5B chat model. Good first pick for local trials and CPU or NPU smoke tests.",
        license="apache-2.0",
        params="0.5B",
        tags=("instruct", "chat", "function-calling"),
        capabilities=("tools",),
    ),
    HubModel(
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen2.5 3B Instruct",
        author="Qwen",
        description="3B instruction model with stronger reasoning than the 0.5B, still small enough for a workstation.",
        license="apache-2.0",
        params="3B",
        tags=("instruct", "chat", "function-calling"),
        capabilities=("tools",),
    ),
    HubModel(
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5 7B Instruct",
        author="Qwen",
        description="7B instruction model for higher-quality chat and finetune baselines.",
        license="apache-2.0",
        params="7B",
        tags=("instruct", "chat", "function-calling"),
        capabilities=("tools",),
    ),
    HubModel(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen2.5 VL 3B Instruct",
        author="Qwen",
        description="Vision-language 3B instruct model. Image understanding plus chat and tool use.",
        license="apache-2.0",
        params="3B",
        pipeline="image-text-to-text",
        tags=("vision", "multimodal", "instruct", "function-calling"),
        capabilities=("vision", "tools"),
    ),
    HubModel(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "SmolLM2 1.7B Instruct",
        author="HuggingFaceTB",
        description="Compact 1.7B instruct model from Hugging Face SmolLM. Fast to download and finetune.",
        license="apache-2.0",
        params="1.7B",
        tags=("instruct", "smollm"),
    ),
    HubModel(
        "microsoft/Phi-4-mini-instruct",
        "Phi-4 mini Instruct",
        author="microsoft",
        description="Microsoft Phi-4 mini instruct checkpoint. Small multilingual reasoning model.",
        license="MIT",
        params="3.8B",
        tags=("instruct", "phi", "reasoning"),
        capabilities=("reasoning",),
    ),
    HubModel(
        "google/gemma-3-1b-it",
        "Gemma 3 1B",
        author="google",
        description="Gemma 3 1B instruction-tuned model. Lightweight Google checkpoint for on-device work.",
        license="gemma",
        params="1B",
        tags=("instruct", "gemma"),
    ),
    HubModel(
        "meta-llama/Llama-3.2-3B-Instruct",
        "Llama 3.2 3B Instruct",
        author="meta-llama",
        description="Llama 3.2 3B instruct. Compact Meta chat model; Hub access may require a license grant.",
        license="llama3.2",
        params="3B",
        tags=("instruct", "llama", "function-calling"),
        capabilities=("tools",),
    ),
    HubModel(
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Mistral 7B Instruct",
        author="mistralai",
        description="Mistral 7B Instruct v0.3. Established 7B chat baseline for comparison runs.",
        license="apache-2.0",
        params="7B",
        tags=("instruct", "mistral", "function-calling"),
        capabilities=("tools",),
    ),
)


CAPABILITY_ORDER = ("vision", "tools", "reasoning")
_VISION_HINTS = (
    "image-text-to-text",
    "image-to-text",
    "visual-question-answering",
    "multimodal",
    "vision",
    "-vl",
    "llava",
    "pixtral",
    "internvl",
)
_TOOLS_HINTS = ("function-calling", "tool-use", "tool_use", "function_call", "tools")
_REASONING_HINTS = ("reasoning", "thinking", "-r1", "r1-", "qwq")


def infer_capabilities(
    tags: tuple[str, ...] | list[str] = (),
    pipeline: str = "",
    repo_id: str = "",
    description: str = "",
) -> tuple[str, ...]:
    haystack = " ".join((*tags, pipeline, repo_id, description)).casefold()
    found: list[str] = []
    if any(hint in haystack for hint in _VISION_HINTS):
        found.append("vision")
    if any(hint in haystack for hint in _TOOLS_HINTS):
        found.append("tools")
    if any(hint in haystack for hint in _REASONING_HINTS):
        found.append("reasoning")
    return tuple(found)


def _short_name(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1].replace("-", " ")


def format_count(value: int) -> str:
    if value >= 1_000_000:
        text = f"{value / 1_000_000:.1f}M"
        return text.replace(".0M", "M")
    if value >= 1_000:
        text = f"{value / 1_000:.1f}k"
        return text.replace(".0k", "k")
    if value > 0:
        return str(value)
    return ""


def _int_field(info: object, *names: str) -> int:
    for name in names:
        value = getattr(info, name, None)
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def hub_download_total(info: object) -> int:
    """Prefer all-time Hub downloads; fall back to the 30-day count."""
    return _int_field(info, "downloads_all_time", "downloadsAllTime", "downloads")


def model_from_hub_info(info: object) -> HubModel | None:
    repo_id = str(getattr(info, "id", "") or getattr(info, "modelId", "")).strip()
    if not repo_id:
        return None
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
    description = ""
    license_name = ""
    if isinstance(card, dict):
        description = str(card.get("description") or "").strip()
        license_name = str(card.get("license") or "").strip()
    elif card is not None:
        description = str(getattr(card, "description", "") or "").strip()
        license_name = str(getattr(card, "license", "") or "").strip()
    tags = tuple(str(tag) for tag in (getattr(info, "tags", None) or []) if tag)[:8]
    updated = getattr(info, "lastModified", None) or getattr(info, "last_modified", None)
    pipeline = str(getattr(info, "pipeline_tag", "") or "text-generation")
    return HubModel(
        repo_id=repo_id,
        name=_short_name(repo_id),
        author=str(getattr(info, "author", "") or repo_id.split("/", 1)[0]),
        description=description,
        downloads=hub_download_total(info),
        likes=_int_field(info, "likes"),
        license=license_name or str(getattr(info, "license", "") or ""),
        pipeline=pipeline,
        tags=tags,
        updated=str(updated or ""),
        capabilities=infer_capabilities(tags, pipeline, repo_id, description),
    )


_EXPAND_FIELDS = ("downloadsAllTime", "downloads", "likes")


def _list_models(api: object, limit: int):
    kwargs = {
        "task": "text-generation",
        "sort": "trendingScore",
        "direction": -1,
        "limit": limit,
    }
    for extra in ({"expand": list(_EXPAND_FIELDS)}, {"full": True}, {}):
        try:
            return api.list_models(**kwargs, **extra)
        except TypeError:
            continue
    return api.list_models(task="text-generation", sort="trendingScore", direction=-1, limit=limit)


def _apply_hub_stats(model: HubModel, info: object) -> HubModel:
    downloads = hub_download_total(info)
    likes = _int_field(info, "likes")
    if downloads <= 0 and likes <= 0:
        return model
    return replace(
        model,
        downloads=downloads or model.downloads,
        likes=likes or model.likes,
    )


def _fill_missing_stats(models: list[HubModel], token: str) -> list[HubModel]:
    missing = [model for model in models if model.downloads <= 0]
    if not missing:
        return models
    from huggingface_hub import HfApi

    api = HfApi(token=token or None)
    by_id = {model.repo_id: model for model in models}
    for model in missing:
        try:
            try:
                info = api.model_info(model.repo_id, expand=list(_EXPAND_FIELDS))
            except TypeError:
                info = api.model_info(model.repo_id)
        except Exception:
            continue
        by_id[model.repo_id] = _apply_hub_stats(model, info)
    return [by_id[model.repo_id] for model in models]


def _query_hub(limit: int, token: str) -> list[HubModel]:
    from huggingface_hub import HfApi

    api = HfApi(token=token or None)
    found: list[HubModel] = []
    seen: set[str] = set()
    for info in _list_models(api, limit):
        model = model_from_hub_info(info)
        if model is None or model.repo_id in seen:
            continue
        seen.add(model.repo_id)
        found.append(model)
        if len(found) >= limit:
            break
    return _fill_missing_stats(found, token)


def fetch_trending_models(
    limit: int = 12,
    token: str = "",
    timeout_s: float = 8.0,
) -> list[HubModel]:
    """Return live trending text-generation models, or featured models if the Hub is down."""
    collected: list[HubModel] = []

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
    try:
        return _fill_missing_stats(list(FEATURED_MODELS), token)
    except Exception:
        return list(FEATURED_MODELS)
