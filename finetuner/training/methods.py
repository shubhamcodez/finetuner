from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingMethodSpec:
    method_id: str
    name: str
    description: str
    uses_reward: bool = False
    uses_reward_model: bool = False
    uses_preference_data: bool = False
    family: str = "offline"
    maturity: str = "stable"


TRAINING_METHODS: dict[str, TrainingMethodSpec] = {
    "sft": TrainingMethodSpec(
        method_id="sft",
        name="SFT",
        description="Supervised fine-tuning on instruction–response pairs",
    ),
    "dpo": TrainingMethodSpec(
        method_id="dpo",
        name="DPO",
        description="Direct Preference Optimization (chosen vs rejected)",
        uses_preference_data=True,
    ),
    "grpo": TrainingMethodSpec(
        method_id="grpo",
        name="GRPO",
        description="Group Relative Policy Optimization with on-policy rollouts",
        uses_reward=True,
    ),
    "ppo": TrainingMethodSpec(
        method_id="ppo",
        name="PPO",
        description="Proximal Policy Optimization with a reward model",
        uses_reward=True,
        uses_reward_model=True,
    ),
    "kto": TrainingMethodSpec(
        method_id="kto",
        name="KTO",
        description="Kahneman–Tversky Optimization from binary feedback",
    ),
    "reward": TrainingMethodSpec(
        method_id="reward",
        name="Reward Model",
        description="Train a Bradley–Terry reward model on preferences",
        uses_preference_data=True,
    ),
    "orpo": TrainingMethodSpec(
        method_id="orpo",
        name="ORPO",
        description="Reference-free odds-ratio preference optimization (experimental in TRL)",
        uses_preference_data=True,
        maturity="experimental",
    ),
    "rloo": TrainingMethodSpec(
        method_id="rloo",
        name="RLOO",
        description="REINFORCE Leave-One-Out online policy optimization",
        uses_reward=True,
        family="online",
    ),
}


def method_list() -> list[TrainingMethodSpec]:
    return list(TRAINING_METHODS.values())


def get_method(method_id: str) -> TrainingMethodSpec | None:
    return TRAINING_METHODS.get(method_id)
