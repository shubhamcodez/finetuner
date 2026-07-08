from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from finetuner.core.paths import bundled_assets_dir
from finetuner.datasets.hf_datasets import load_hf_split, normalize_hf_dataset_id
from finetuner.datasets.presets import FORMATTERS, get_preset


def _messages_to_text(messages: list) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _row_to_text(row: dict, formatter) -> str | None:
    if formatter:
        text = formatter(row)
        if text:
            return text
    if "text" in row:
        return row["text"]
    if "messages" in row:
        return _messages_to_text(row["messages"])
    if "instruction" in row and "output" in row:
        inst = row.get("instruction", "")
        inp = row.get("input", "")
        out = row.get("output", "")
        text = f"### Instruction:\n{inst}\n"
        if inp:
            text += f"### Input:\n{inp}\n"
        text += f"### Response:\n{out}"
        return text
    return None


def _load_jsonl(path: Path, formatter, limit: int | None) -> Dataset:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = _row_to_text(row, formatter)
            if text:
                rows.append({"text": text})
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No training examples found in {path}")
    return Dataset.from_list(rows)


def _resolve_bundled_path(preset) -> Path | None:
    datasets_dir = bundled_assets_dir() / "datasets"
    if preset.bundled_filename:
        candidate = datasets_dir / preset.bundled_filename
        if candidate.exists():
            return candidate
        if preset.preset_id == "sample":
            root_sample = bundled_assets_dir() / preset.bundled_filename
            if root_sample.exists():
                return root_sample
    return None


def load_preset_dataset(
    preset_id: str,
    max_samples: int | None = None,
    log=None,
    bundled_only: bool = False,
) -> Dataset:
    from finetuner.datasets.hf_datasets import load_env_file

    load_env_file()

    preset = get_preset(preset_id)
    if preset is None:
        raise ValueError(f"Unknown dataset preset: {preset_id}")

    def _log(msg: str) -> None:
        if log:
            log(msg)

    limit = max_samples if max_samples is not None else preset.max_samples
    formatter = FORMATTERS.get(preset_id)

    bundled = _resolve_bundled_path(preset)

    if bundled_only:
        if bundled is None:
            raise ValueError(f"Preset {preset_id} has no bundled sample available.")
        _log(f"Loading offline bundled sample: {bundled}")
        return _load_jsonl(bundled, formatter, limit)

    if preset.preset_id == "sample" and bundled is not None:
        _log(f"Loading bundled preset dataset: {bundled}")
        return _load_jsonl(bundled, formatter, limit)

    if preset.hf_dataset:
        repo_id = normalize_hf_dataset_id(preset.hf_dataset)
        _log(
            f"Loading HF dataset {repo_id}"
            + (f" ({preset.hf_config})" if preset.hf_config else "")
            + f" split={preset.split}..."
        )
        try:
            raw = load_hf_split(repo_id, preset.hf_config, preset.split)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Hugging Face dataset {repo_id}: {exc}"
            ) from exc

        if limit and len(raw) > limit:
            raw = raw.select(range(limit))

        rows = []
        for row in raw:
            text = formatter(row) if formatter else _row_to_text(row, None)
            if text:
                rows.append({"text": text})

        if not rows:
            raise ValueError(f"Preset {preset_id} produced no training examples.")

        _log(f"Prepared {len(rows)} training examples for {preset.name}.")
        return Dataset.from_list(rows)

    raise ValueError(
        f"Preset {preset_id} has no dataset source. "
        "Enable 'Use offline sample only' or configure a custom dataset."
    )
