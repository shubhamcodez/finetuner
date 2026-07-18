from __future__ import annotations

from finetuner.workflows.schema import StageKind, WorkflowSpec, WorkflowStage


def _stage(
    stage_id: str,
    name: str,
    kind: StageKind,
    *depends_on: str,
    **parameters,
) -> WorkflowStage:
    return WorkflowStage(stage_id, name, kind, tuple(depends_on), parameters)


_TEMPLATES: dict[str, WorkflowSpec] = {
    "sft": WorkflowSpec(
        "sft",
        "Supervised fine-tuning",
        (
            _stage("sft", "Instruction tuning", StageKind.TRAIN, method="sft"),
            _stage("evaluate", "Evaluate", StageKind.EVALUATE, "sft"),
        ),
        "Instruction tuning followed by held-out benchmark evaluation.",
    ),
    "dpo": WorkflowSpec(
        "dpo",
        "SFT + DPO alignment",
        (
            _stage("sft", "Instruction tuning", StageKind.TRAIN, method="sft"),
            _stage("dpo", "Preference optimization", StageKind.TRAIN, "sft", method="dpo"),
            _stage("evaluate", "Evaluate", StageKind.EVALUATE, "dpo"),
            _stage("analyze", "Representations", StageKind.ANALYZE, "dpo"),
        ),
        "Offline chosen/rejected preference optimization after SFT.",
    ),
    "rlhf": WorkflowSpec(
        "rlhf",
        "Classic RLHF",
        (
            _stage("sft", "Instruction tuning", StageKind.TRAIN, method="sft"),
            _stage("reward", "Reward model", StageKind.TRAIN, method="reward"),
            _stage(
                "ppo",
                "Policy optimization",
                StageKind.TRAIN,
                "sft",
                "reward",
                method="ppo",
            ),
            _stage("evaluate", "Evaluate", StageKind.EVALUATE, "ppo"),
            _stage("analyze", "Representations", StageKind.ANALYZE, "ppo"),
        ),
        "SFT, preference reward-model training, then PPO policy optimization.",
    ),
    "reasoning_rl": WorkflowSpec(
        "reasoning_rl",
        "Reasoning RL (GRPO)",
        (
            _stage("sft", "Instruction tuning", StageKind.TRAIN, method="sft"),
            _stage("grpo", "Verifiable-reward RL", StageKind.TRAIN, "sft", method="grpo"),
            _stage("evaluate", "Evaluate", StageKind.EVALUATE, "grpo"),
            _stage("analyze", "Representations", StageKind.ANALYZE, "grpo"),
        ),
        "Online group-relative RL with a configurable verifier or reward model.",
    ),
    "kto": WorkflowSpec(
        "kto",
        "Binary-feedback alignment (KTO)",
        (
            _stage("sft", "Instruction tuning", StageKind.TRAIN, method="sft"),
            _stage("kto", "Binary-feedback optimization", StageKind.TRAIN, "sft", method="kto"),
            _stage("evaluate", "Evaluate", StageKind.EVALUATE, "kto"),
        ),
        "Preference alignment when feedback is desirable/undesirable rather than paired.",
    ),
    "distill_deploy": WorkflowSpec(
        "distill_deploy",
        "Distill, evaluate, and deploy",
        (
            _stage("distill", "Teacher-to-student distillation", StageKind.DISTILL),
            _stage("evaluate", "Evaluate student", StageKind.EVALUATE, "distill"),
            _stage("analyze", "Representations", StageKind.ANALYZE, "distill"),
            _stage("quantize", "Deployment artifact", StageKind.QUANTIZE, "distill"),
        ),
        "Domain-selective distillation with evaluation, interpretability, and quantization.",
    ),
}


def workflow_templates() -> list[WorkflowSpec]:
    return list(_TEMPLATES.values())


def get_workflow_template(template_id: str) -> WorkflowSpec:
    try:
        return _TEMPLATES[template_id]
    except KeyError as exc:
        raise KeyError(f"Unknown workflow template: {template_id}") from exc


def training_method_workflow(method: str) -> WorkflowSpec:
    if method in _TEMPLATES:
        return _TEMPLATES[method]
    stage = _stage("train", f"{method.upper()} training", StageKind.TRAIN, method=method)
    return WorkflowSpec(
        f"{method}_workflow",
        f"{method.upper()} workflow",
        (stage, _stage("evaluate", "Evaluate", StageKind.EVALUATE, "train")),
    )
