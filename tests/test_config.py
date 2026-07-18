from __future__ import annotations

from finetuner.core.job import ProjectConfig
from finetuner.workflows.templates import get_workflow_template


def test_config_round_trip_includes_product_pipelines():
    config = ProjectConfig(workflow=get_workflow_template("rlhf"))
    config.quantization.backend = "openvino"
    config.quantization.target = "intel_npu"
    config.distillation.teacher_model = "teacher/model"
    config.distillation.student_model = "student/model"
    config.analysis.reducer = "tsne"
    config.training.lora_target_modules = ["q_proj", "v_proj"]
    restored = ProjectConfig.from_dict(config.to_dict())
    assert restored.workflow.workflow_id == "rlhf"
    assert restored.quantization.target == "intel_npu"
    assert restored.distillation.teacher_model == "teacher/model"
    assert restored.analysis.reducer == "tsne"
    assert restored.training.lora_target_modules == ["q_proj", "v_proj"]


def test_hugging_face_token_is_never_serialized():
    config = ProjectConfig(hf_token="hf_secret")
    payload = config.to_dict()
    assert "hf_token" not in payload
    assert "hf_secret" not in repr(payload)


def test_legacy_config_migrates_training_method_to_workflow():
    restored = ProjectConfig.from_dict(
        {"training": {"training_method": "dpo"}, "hf_token": "legacy"}
    )
    assert restored.workflow.workflow_id == "dpo"
    assert restored.hf_token == "legacy"
    assert "hf_token" not in restored.to_dict()
