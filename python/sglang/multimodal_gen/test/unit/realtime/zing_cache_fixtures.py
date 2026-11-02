# SPDX-License-Identifier: Apache-2.0
"""Small CPU cache fixtures without model-accuracy test imports."""

import os
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def cpu_dma(monkeypatch):
    """Run real tensor copies/KV commits, replacing only CUDA transport on CPU."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        preparation_cache,
    )

    stream = NS(
        device=torch.device("cuda:0"),
        wait_stream=Mock(),
        wait_event=Mock(),
        synchronize=Mock(),
    )
    events = []

    class Event:
        def __init__(self, **kwargs):
            self.synchronized = False
            events.append(self)

        def record(self):
            pass

        def synchronize(self):
            self.synchronized = True

        def query(self):
            return self.synchronized

    empty = torch.empty

    def allocate(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(
        preparation_cache, "coordinate_validation", lambda value: [value]
    )
    return events, stream


def cpu_cache(storage, *, entries=6, rows=32):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        Block0PreparationCache,
    )

    pool = NS(
        _pool=NS(_slots=[NS(kv_cache=storage)]),
        plan=NS(config=NS(max_prefix_rows=rows)),
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache import (
        Block0DiskCache,
    )

    identity = {"model": {"fixture": "model"}, "layout": {"sp": 1}}
    root = os.environ.get("SGLANG_ZING_DISK_CACHE_DIR")
    disk = None
    if root:
        from pathlib import Path

        Path(root).mkdir(parents=True, exist_ok=True)
        disk = Block0DiskCache(root, max_bytes=1 << 30, identity=identity)
    first = storage[0].k
    slot_bytes = len(storage) * 2 * rows * first[0].numel() * first.element_size()
    return Block0PreparationCache(
        pool, max_bytes=entries * slot_bytes, identity=identity, disk_cache=disk
    )


def disk_commit(
    cache, key, value=1.0, rows=3, layers=2, rank=0, *, generation=None, force=False
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache import (
        Block0DiskCache,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import ZingKVRequest

    assert isinstance(cache, Block0DiskCache)
    kv = torch.full((layers, 2, rows, 2, 2), value, dtype=torch.bfloat16)
    tmp = cache.new_tmp_dir(key)
    path = cache.save_rank_kv(key, rank, kv, tmp)
    from sglang.multimodal_gen.test.unit.realtime.test_zing_preparation_cache import (
        inputs,
    )

    prepared = inputs()
    prepared.positive["context"] = torch.full((rows, 4), value, dtype=torch.bfloat16)
    conditions = {
        "conditioning_identity": f"{key}-condition",
        "inputs": prepared,
        "request": ZingKVRequest(
            target_height=32,
            target_width=32,
            source_shape=(32, 32),
            prefix_rows=rows,
            window_chunks=6,
            sink_chunks=3,
            rope_position_mode="block_relative",
            rope_max_frame_gap=12,
        ),
        "positions": torch.arange(rows).reshape(rows, 1).repeat(1, 3),
    }
    manifest = {
        "identity": cache.identity,
        "generation": generation or "fixture-" + key,
        "sp_world_size": cache.sp_world_size,
        "geometry": {"prefix_rows": rows},
        "prompt_preview": f"prompt-{key}",
    }
    entry = cache.commit_entry(
        key, conditions, manifest, {rank: path}, tmp, force=force
    )
    assert entry is not None
    return kv, entry
