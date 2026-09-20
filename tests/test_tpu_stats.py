from __future__ import annotations

from finetuner.monitor.tpu import TpuCollector, name_from_local_devices, snapshot_from_env


def test_local_tpu_name_ignores_hexagon_npu():
    assert name_from_local_devices(
        [
            "Snapdragon(R) X Elite - Qualcomm(R) Hexagon(TM) NPU",
            "Google Coral Edge TPU",
        ]
    ) == "Google Coral Edge TPU"
    assert name_from_local_devices(["Intel(R) AI Boost NPU"]) == ""


def test_cloud_tpu_env_builds_a_label():
    snap = snapshot_from_env(
        {"TPU_NAME": "node-1", "TPU_ACCELERATOR_TYPE": "v5litepod-4"}
    )
    assert snap is not None
    assert snap.available
    assert snap.name == "node-1 (v5litepod-4)"
    assert snapshot_from_env({}) is None


def test_tpu_collect_does_not_raise():
    snap = TpuCollector.collect()
    assert snap.util_percent >= 0
    TpuCollector.shutdown()
