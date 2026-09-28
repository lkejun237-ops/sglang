# SPDX-License-Identifier: Apache-2.0
"""Run one existing tiny lifecycle oracle with real CUDA BF16 grouped RoPE."""

import json

import pytest
import torch

import sglang.multimodal_gen.runtime.models.dits.zing_h3  # noqa: F401
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.models.dits.zing_h3_rope import HISTORY_ROPE_ROUTES
from sglang.multimodal_gen.test.unit.test_zing_history_rope import (
    test_cache_transactions_eviction_and_failure,
)


def main():
    assert torch.cuda.is_available(), "Real CUDA is required; this test cannot skip"
    before = HISTORY_ROPE_ROUTES.copy()
    snapshots = []
    clear = ZingH3KVCache.clear

    def record_clear(cache):
        if cache.stats["store_allocations"]:
            snapshots.append(
                dict(stats=dict(cache.stats), reuse=dict(cache.rope_reuse_stats))
            )
        return clear(cache)

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SGLANG_ZING_CHUNK_ROPE_REUSE", "1")
        patch.setattr(ZingH3KVCache, "clear", record_clear)
        test_cache_transactions_eviction_and_failure(
            device="cuda",
            window=3,
            sink=1,
            mode="block_relative",
            grouped=True,
            planned=False,
            monkeypatch=patch,
        )
    torch.cuda.synchronize()
    routes = dict(HISTORY_ROPE_ROUTES - before)
    assert routes.get("grouped", 0) > 0
    assert routes["grouped_chunks"] > routes["grouped"]
    assert not any(k.startswith("grouped_fallback") and v for k, v in routes.items())
    assert len(snapshots) == 2
    assert all(s["stats"]["compactions"] > 0 for s in snapshots)
    assert all(s["stats"]["store_allocations"] > 1 for s in snapshots)
    print(
        json.dumps(
            dict(
                result="PASS",
                oracle="test_cache_transactions_eviction_and_failure",
                device=torch.cuda.get_device_name(),
                capability=torch.cuda.get_device_capability(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                window=3,
                sink=1,
                mode="block_relative",
                planned=False,
                dtype="bfloat16",
                heads=14,
                dim=128,
                transactions=28,
                prefix_rows=[5, 6],
                source_rows=7,
                target_rows=9,
                routes=routes,
                cache_snapshots_before_clear=snapshots,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
