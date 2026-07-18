from __future__ import annotations

import pytest
from datasets import Dataset

from finetuner.training.dataset_formats import prepare_method_dataset, sft_to_dpo_rows


def test_real_preference_rows_are_preserved():
    dataset = Dataset.from_list(
        [{"prompt": "p", "chosen": "good", "rejected": "bad", "source": "human"}]
    )
    assert prepare_method_dataset(dataset, "dpo") is dataset


def test_preference_rows_can_feed_the_sft_stage_without_training_on_prompts_only():
    dataset = Dataset.from_list([{"prompt": "p", "chosen": "good", "rejected": "bad"}])
    sft = prepare_method_dataset(dataset, "sft")
    assert sft.column_names == ["text"]
    assert sft[0]["text"] == "p\ngood"


def test_synthetic_preferences_are_opt_in():
    dataset = Dataset.from_list([{"text": "Prompt\n### Response:\nAnswer #### 4"}])
    with pytest.raises(ValueError, match="disabled by default"):
        prepare_method_dataset(dataset, "dpo")
    converted = prepare_method_dataset(dataset, "dpo", allow_synthetic_preferences=True, seed=7)
    assert converted[0]["chosen"] != converted[0]["rejected"]


def test_synthetic_negative_generation_is_reproducible():
    dataset = Dataset.from_list([{"text": "Prompt\n### Response:\nAnswer #### 10"}])
    assert sft_to_dpo_rows(dataset, seed=123) == sft_to_dpo_rows(dataset, seed=123)
