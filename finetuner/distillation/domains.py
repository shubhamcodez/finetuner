from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from finetuner.distillation.config import DomainSelection


@dataclass(frozen=True)
class DomainPreset:
    domain_id: str
    name: str
    keywords: tuple[str, ...]


DOMAIN_PRESETS: dict[str, DomainPreset] = {
    "computer_science": DomainPreset(
        "computer_science",
        "Computer science",
        (
            "algorithm",
            "programming",
            "software",
            "computer",
            "database",
            "network",
            "compiler",
            "python",
            "code",
        ),
    ),
    "mathematics": DomainPreset(
        "mathematics",
        "Mathematics",
        ("algebra", "geometry", "calculus", "theorem", "proof", "equation", "mathematics"),
    ),
    "optimization": DomainPreset(
        "optimization",
        "Optimization",
        (
            "optimization",
            "gradient",
            "linear programming",
            "convex",
            "objective function",
            "constraint",
            "operations research",
        ),
    ),
    "science": DomainPreset(
        "science",
        "Natural science",
        ("physics", "chemistry", "biology", "astronomy", "scientific"),
    ),
    "reasoning": DomainPreset(
        "reasoning",
        "Reasoning",
        ("reason", "logic", "deduce", "step by step", "puzzle", "inference"),
    ),
    "safety": DomainPreset(
        "safety",
        "Safety and policy",
        ("safety", "policy", "harm", "secure", "privacy", "compliance"),
    ),
}


_METADATA_FIELDS = ("domain", "subject", "category", "topic", "field", "tags")
_TEXT_FIELDS = ("prompt", "instruction", "question", "text", "input")


def _flatten(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        value = " ".join(map(str, value))
    return str(value or "").replace("_", " ").replace("-", " ")


def domain_terms(selection: DomainSelection) -> tuple[str, ...]:
    if selection.mode == "all":
        return ()
    if selection.mode == "custom":
        return tuple(
            term.strip().lower() for term in re.split(r"[,;\n]", selection.custom) if term.strip()
        )
    terms: list[str] = []
    for domain_id in selection.fields:
        try:
            preset = DOMAIN_PRESETS[domain_id]
            terms.extend((domain_id.replace("_", " "), preset.name, *preset.keywords))
        except KeyError as exc:
            raise ValueError(f"Unknown domain preset: {domain_id}") from exc
    return tuple(dict.fromkeys(term.lower() for term in terms))


def row_matches_domain(row: dict[str, Any], selection: DomainSelection) -> bool:
    if selection.mode == "all":
        return True
    terms = domain_terms(selection)
    metadata = " ".join(_flatten(row.get(key)) for key in _METADATA_FIELDS).lower()
    text = " ".join(_flatten(row.get(key)) for key in _TEXT_FIELDS).lower()
    haystack = f"{metadata} {text}"
    return any(term in haystack for term in terms)


def select_domain_rows(
    rows: Iterable[dict[str, Any]], selection: DomainSelection, *, limit: int | None = None
) -> list[dict[str, Any]]:
    errors = selection.validate()
    if errors:
        raise ValueError("; ".join(errors))
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row_matches_domain(row, selection):
            selected.append(dict(row))
            if limit is not None and len(selected) >= limit:
                break
    return selected
