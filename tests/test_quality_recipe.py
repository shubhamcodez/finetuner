from __future__ import annotations

from finetuner.core.job import TrainingConfig
from finetuner.training.accel.detect import uses_accel_engine
from finetuner.training.chat_format import conversational_rows, row_to_messages, user_and_assistant
from finetuner.training.recipe import QUALITY_LORA_TARGETS, apply_quality_recipe


def test_chat_format_prefers_the_raw_gsm8k_question():
    user, assistant = user_and_assistant(
        {
            "question": "What is 2+2?",
            "answer": "Adding 2 and 2 gives 4.\n#### 4",
        }
    )
    assert user == "What is 2+2?"
    assert assistant.endswith("#### 4")
    messages = row_to_messages({"question": "What is 2+2?", "answer": "4"})
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_conversational_rows_drop_leftover_prompt_columns():
    rows = conversational_rows(
        [
            {
                "prompt": "### Instruction:\nSolve this grade-school math word problem.\n### Input:\nWhat is 2+2?",
                "question": "What is 2+2?",
                "answer": "#### 4",
                "text": "ignore",
            }
        ]
    )
    assert list(rows[0]) == ["messages"]
    assert rows[0]["messages"][0]["content"] == "What is 2+2?"
    assert rows[0]["messages"][1]["content"] == "#### 4"


def test_quality_recipe_defaults_to_full_sft_not_lora():
    config = apply_quality_recipe(
        TrainingConfig(
            training_method="sft",
            accelerator="npu",
            max_steps=40,
            max_seq_length=2048,
            quality_recipe=True,
        )
    )
    assert config.use_chat_template
    assert not config.uses_lora()
    assert config.learning_rate <= 5e-5
    assert config.max_seq_length < 2048
    assert config.max_steps == 40
    assert not uses_accel_engine(config)


def test_quality_recipe_keeps_lora_only_when_toggled():
    config = apply_quality_recipe(
        TrainingConfig(
            training_method="sft",
            use_lora=True,
            lora_rank=8,
            quality_recipe=True,
        )
    )
    assert config.uses_lora()
    assert config.lora_rank >= 32
    assert "q_proj" in (config.lora_target_modules or QUALITY_LORA_TARGETS)


def test_npu_alias_still_uses_the_accel_engine():
    assert uses_accel_engine(TrainingConfig(training_method="npu", quality_recipe=True))
