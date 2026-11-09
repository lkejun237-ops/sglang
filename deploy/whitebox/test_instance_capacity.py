"""Admission happens before a worker command or GPU allocation is started."""

import json
import sys

import pytest
from instance_capacity import admit, supervise

from sglang.multimodal_gen.runtime.utils.shared_capacity import CapacityUnavailable


def plan(tmp_path):
    return {
        "capacity_dir": str(tmp_path),
        "host_cache_bytes": 100,
        "gpu_bytes": {"GPU-a": 100},
        "instances": {
            "one": {"gpu_bytes": {"GPU-a": 60}, "host_cache_bytes": 50},
            "two": {"gpu_bytes": {"GPU-a": 60}, "host_cache_bytes": 50},
        },
    }


def test_multiple_instances_cannot_overbook_same_gpu(tmp_path):
    policy = plan(tmp_path)
    inventory = {"GPU-a": {"total": 120, "free": 120}}
    first = admit(policy, "one", inventory)
    try:
        with pytest.raises(CapacityUnavailable):
            admit(policy, "two", inventory)
    finally:
        for lease in first:
            lease.close()
    second = admit(policy, "two", inventory)
    for lease in second:
        lease.close()


def test_driver_free_memory_and_machine_host_plan_are_checked(tmp_path):
    policy = plan(tmp_path)
    with pytest.raises(ValueError, match="available memory"):
        admit(policy, "one", {"GPU-a": {"total": 120, "free": 40}})
    policy["instances"]["two"]["host_cache_bytes"] = 60
    with pytest.raises(ValueError, match="machine total"):
        admit(policy, "one", {"GPU-a": {"total": 120, "free": 120}})


def test_partial_gpu_admission_failure_rolls_back_only_own_leases(tmp_path):
    policy = plan(tmp_path)
    policy["gpu_bytes"]["GPU-b"] = 100
    policy["instances"]["one"]["gpu_bytes"]["GPU-b"] = 60
    inventory = {
        "GPU-a": {"total": 120, "free": 120},
        "GPU-b": {"total": 120, "free": 0},
    }
    with pytest.raises(ValueError):
        admit(policy, "one", inventory)
    second = admit(policy, "two", inventory)
    for lease in second:
        lease.close()


def test_unclean_supervisor_marker_quarantines_gpu_capacity(tmp_path):
    policy = plan(tmp_path)
    (tmp_path / "orphan.instance").write_text(
        json.dumps({"instance": "one", "gpus": {"GPU-a": 60}})
    )
    with pytest.raises(CapacityUnavailable, match="quarantine"):
        admit(policy, "two", {"GPU-a": {"total": 120, "free": 120}})


def test_foreground_worker_exit_releases_quota_and_marker(tmp_path):
    policy = plan(tmp_path)
    inventory = {"GPU-a": {"total": 120, "free": 120}}
    leases = admit(policy, "one", inventory)
    assert supervise(policy, "one", [sys.executable, "-c", "pass"], leases) == 0
    assert not list(tmp_path.glob("*.instance"))
    next_leases = admit(policy, "two", inventory)
    for lease in next_leases:
        lease.close()
