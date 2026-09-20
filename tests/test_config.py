from __future__ import annotations

from finetuner.core.job import ProjectConfig


def test_config_round_trip_includes_product_pipelines():
    config = ProjectConfig()
    config.quantization.backend = "openvino"
    config.quantization.target = "intel_npu"
    config.distillation.teacher_model = "teacher/model"
    config.distillation.student_model = "student/model"
    config.analysis.reducer = "tsne"
    config.inference.engine = "vllm"
    config.inference.target = "nvidia_gpu"
    config.inference.serve_port = 1234
    config.training.lora_target_modules = ["q_proj", "v_proj"]
    restored = ProjectConfig.from_dict(config.to_dict())
    assert restored.quantization.target == "intel_npu"
    assert restored.distillation.teacher_model == "teacher/model"
    assert restored.analysis.reducer == "tsne"
    assert restored.inference.engine == "vllm"
    assert restored.inference.serve_port == 1234
    assert restored.training.lora_target_modules == ["q_proj", "v_proj"]
    assert "workflow" not in config.to_dict()


def test_hugging_face_token_is_never_serialized():
    config = ProjectConfig(hf_token="hf_secret")
    payload = config.to_dict()
    assert "hf_token" not in payload
    assert "hf_secret" not in repr(payload)


def test_legacy_workflow_key_is_ignored():
    restored = ProjectConfig.from_dict(
        {
            "training": {"training_method": "dpo"},
            "hf_token": "legacy",
            "workflow": {"id": "dpo", "name": "SFT + DPO alignment", "stages": []},
        }
    )
    assert restored.training.training_method == "dpo"
    assert restored.hf_token == "legacy"
    assert "workflow" not in restored.to_dict()
    assert "hf_token" not in restored.to_dict()
