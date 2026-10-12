# SPDX-License-Identifier: Apache-2.0
import importlib.util

import torch


def memory_api():
    name = "sglang.multimodal_gen.runtime.conditioning.memory"
    assert importlib.util.find_spec(name) is not None, (
        "Conditioning memory storage ledger is not implemented"
    )
    from sglang.multimodal_gen.runtime.conditioning.memory import memory_receipt

    return memory_receipt


def test_shared_storage_views_have_one_global_physical_owner_union():
    receipt = memory_api()
    base = torch.arange(32, dtype=torch.float32)
    a = torch.nn.Module()
    a.register_buffer("all", base)
    b = torch.nn.Module()
    b.register_buffer("slice", base[8:12])
    record = receipt(
        {
            "video_vae": a,
            "transformer": b,
            "condition_current": {"context": base.view(4, 8)},
        },
        phase="steady",
        rank=1,
    )
    assert record["storage_union_bytes"] == base.untyped_storage().nbytes()
    assert record["device_bytes"] == {"cpu": 128}
    assert len(record["storages"]) == 1
    assert record["storages"][0]["owners"] == [
        "condition_current",
        "transformer",
        "video_vae",
    ]
    assert record["owner_bytes"] == {
        "condition_current": 128,
        "transformer": 128,
        "video_vae": 128,
    }
    assert record["phase"] == "steady"
    assert record["rank"] == 1


def test_allocated_reserved_and_storage_are_distinct_measures():
    receipt = memory_api()
    record = receipt(
        {"condition_staging": torch.zeros(3, dtype=torch.bfloat16)},
        phase="condition_ready",
        rank=0,
        allocator={
            "cuda:0": {
                "allocated": 100,
                "reserved": 256,
                "peak_allocated": 120,
                "peak_reserved": 300,
            }
        },
    )
    assert record["storage_union_bytes"] == 6
    assert record["allocator"]["cuda:0"]["allocated"] == 100
    assert record["allocator"]["cuda:0"]["reserved"] == 256
    assert record["allocator"]["cuda:0"]["unattributed_allocated"] == 100
    assert "allocated_plus_reserved" not in record


def test_cpu_clone_is_a_distinct_storage_and_cleanup_removes_owner():
    receipt = memory_api()
    condition = torch.zeros(10, dtype=torch.bfloat16)
    cloned = condition.clone()
    before = receipt(
        {"condition_current": condition, "condition_pending": cloned},
        phase="prompt_update",
        rank=0,
    )
    after = receipt({}, phase="after_cleanup", rank=0)
    assert before["storage_union_bytes"] == 40
    assert len(before["storages"]) == 2
    assert after["storage_union_bytes"] == 0
    assert after["owner_bytes"] == {}


def test_slotted_cache_and_deque_views_are_included_in_physical_union():
    from collections import deque
    from dataclasses import dataclass

    @dataclass(slots=True)
    class Cache:
        tensor: torch.Tensor

    value = torch.zeros(8, dtype=torch.float32)
    record = memory_api()(
        {"kv_slot": Cache(value), "codec_pending": deque([value[2:3]])},
        phase="steady",
        rank=0,
    )
    assert record["storage_union_bytes"] == 32
    assert record["storages"][0]["owners"] == ["codec_pending", "kv_slot"]
