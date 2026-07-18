from __future__ import annotations

import json

from finetuner.core.artifacts import atomic_write_json
from finetuner.core.job import ModelJob, ModelSource, ProjectConfig, TrainingConfig
from finetuner.core.paths import DEFAULT_MODELS, config_path


def load_config() -> ProjectConfig:
    from finetuner.datasets.hf_datasets import load_env_file

    load_env_file()
    path = config_path()
    if not path.exists():
        config = default_config()
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            config = ProjectConfig.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            config = default_config()
    config = _merge_default_models(config)
    config = _normalize_training_dataset_ids(config)
    if not config.hf_token:
        import os

        token = os.environ.get("HF_TOKEN", "")
        if token:
            config.hf_token = token
    return config


def _normalize_training_dataset_ids(config: ProjectConfig) -> ProjectConfig:
    from finetuner.datasets.hf_datasets import normalize_hf_dataset_id

    if config.training.dataset_hf_id:
        config.training.dataset_hf_id = normalize_hf_dataset_id(config.training.dataset_hf_id)
    return config


def _merge_default_models(config: ProjectConfig) -> ProjectConfig:
    """Append any built-in default models missing from an existing config."""
    existing = {m.identifier for m in config.models}
    for name, repo_id in DEFAULT_MODELS:
        if repo_id not in existing:
            config.models.append(
                ModelJob(name=name, source=ModelSource.HUGGINGFACE, identifier=repo_id)
            )
    return config


def save_config(config: ProjectConfig) -> None:
    path = config_path()
    atomic_write_json(path, config.to_dict())


def default_config() -> ProjectConfig:
    return ProjectConfig(
        models=[
            ModelJob(name=name, source=ModelSource.HUGGINGFACE, identifier=repo_id)
            for name, repo_id in DEFAULT_MODELS
        ],
        training=TrainingConfig(dataset_preset_id="gsm8k"),
        enabled_evals=["gsm8k"],
    )
