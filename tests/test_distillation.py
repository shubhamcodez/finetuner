from __future__ import annotations

import json

import pytest

from finetuner.distillation.config import DistillationConfig, DomainSelection
from finetuner.distillation.domains import row_matches_domain, select_domain_rows
from finetuner.distillation.pipeline import build_sequence_distillation_dataset
from finetuner.distillation.runner import run_distillation
from finetuner.core.job import TrainingConfig


def valid_config(**overrides):
    config = DistillationConfig(
        teacher_model="teacher/model",
        student_model="student/model",
        domain=DomainSelection("all"),
        max_samples=10,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_domain_selection_supports_presets_custom_and_all():
    rows = [
        {"prompt": "Implement a compiler optimization pass", "domain": "computer science"},
        {"prompt": "Analyze a sonnet", "domain": "literature"},
    ]
    selected = select_domain_rows(
        rows, DomainSelection("presets", ["computer_science", "optimization"])
    )
    assert selected == [rows[0]]
    assert row_matches_domain(rows[1], DomainSelection("custom", custom="sonnet"))
    assert select_domain_rows(rows, DomainSelection("all")) == rows


def test_sequence_distillation_writes_dataset_and_provenance(tmp_path):
    rows = [{"prompt": "Question one"}, {"instruction": "Question two"}]

    def teacher(prompts, _config):
        return [f"Teacher answer: {prompt}" for prompt in prompts]

    result = build_sequence_distillation_dataset(
        rows, tmp_path, valid_config(), teacher, batch_size=1
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "teacher_sequences.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((tmp_path / "distillation_manifest.json").read_text(encoding="utf-8"))
    assert len(records) == result.selected_count == 2
    assert records[0]["response"].startswith("Teacher answer")
    assert manifest["teacher_model"] == "teacher/model"
    assert len(manifest["dataset_digest"]) == 64


def test_teacher_batch_cardinality_is_checked(tmp_path):
    with pytest.raises(RuntimeError, match="Teacher returned"):
        build_sequence_distillation_dataset(
            [{"prompt": "one"}, {"prompt": "two"}],
            tmp_path,
            valid_config(),
            lambda _prompts, _config: ["only one"],
        )


def test_distillation_rejects_same_teacher_and_student():
    config = valid_config(student_model="teacher/model")
    assert any("must be different" in error for error in config.validate())


def test_logit_gkd_and_minillm_techniques_dispatch_to_distinct_trainers(monkeypatch):
    monkeypatch.setattr(
        "finetuner.distillation.runner._run_gkd", lambda *_args: ("gkd", "gkd-manifest")
    )
    monkeypatch.setattr(
        "finetuner.distillation.runner._run_minillm",
        lambda *_args: ("minillm", "minillm-manifest"),
    )
    logit = valid_config(technique="logit_kl")
    reverse = valid_config(technique="reverse_kl")
    assert run_distillation("data", "out", logit, TrainingConfig())[0] == "gkd"
    assert run_distillation("data", "out", reverse, TrainingConfig())[0] == "minillm"
