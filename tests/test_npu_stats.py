from __future__ import annotations

import sys

import pytest

from finetuner.monitor.npu import (
    aggregate_npu_util,
    is_npu_adapter,
    memory_for_luid,
    npu_luid_from_instances,
    parse_adapter_luid,
    parse_gpu_engine_instance,
)


NPU = "pid_10_luid_0x00000000_0x000125BD_phys_0_eng_0_engtype_Compute"
GPU = "pid_20_luid_0x00000000_0x00012066_phys_0_eng_0_engtype_3D"
GPU_COMPUTE = "pid_20_luid_0x00000000_0x00012066_phys_0_eng_4_engtype_Compute"


def test_parse_gpu_engine_instance_reads_luid_and_type():
    parsed = parse_gpu_engine_instance(NPU)
    assert parsed is not None
    assert parsed.luid == "0x00000000_0x000125BD"
    assert parsed.engine == 0
    assert parsed.engine_type == "Compute"


def test_parse_adapter_luid_from_memory_instance():
    assert parse_adapter_luid("luid_0x00000000_0x000125BD_phys_0") == "0x00000000_0x000125BD"


def test_npu_luid_is_compute_only_adapter():
    assert npu_luid_from_instances([NPU, GPU, GPU_COMPUTE]) == "0x00000000_0x000125BD"
    assert is_npu_adapter({"Compute"})
    assert not is_npu_adapter({"3D", "Compute"})


def test_aggregate_npu_util_sums_pids_then_caps_busiest_engine():
    samples = [
        (NPU, 40.0),
        ("pid_11_luid_0x00000000_0x000125BD_phys_0_eng_0_engtype_Compute", 30.0),
        ("pid_12_luid_0x00000000_0x000125BD_phys_0_eng_1_engtype_Compute", 12.0),
        (GPU_COMPUTE, 90.0),
    ]
    assert aggregate_npu_util(samples, "0x00000000_0x000125BD") == 70.0


def test_memory_for_luid_converts_bytes():
    committed = [("luid_0x00000000_0x000125BD_phys_0", 2 * 1024**3)]
    shared = [("luid_0x00000000_0x000125BD_phys_0", 0.5 * 1024**3)]
    used, shared_gb = memory_for_luid(committed, shared, "0x00000000_0x000125BD")
    assert used == pytest.approx(2.0)
    assert shared_gb == pytest.approx(0.5)


@pytest.mark.skipif(sys.platform != "win32", reason="PDH is Windows-only")
def test_npu_collect_does_not_raise():
    from finetuner.monitor.npu import NpuCollector

    snap = NpuCollector.collect()
    assert snap.util_percent >= 0
    NpuCollector.shutdown()
