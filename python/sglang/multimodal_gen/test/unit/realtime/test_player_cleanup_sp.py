# SPDX-License-Identifier: Apache-2.0
"""Two real CPU/Gloo ranks, no GPU or production fault injection."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp


def _release_rank(rank, address, results):
    from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=address,
        timeout=timedelta(seconds=30),
    )
    try:
        worker = GPUWorker.__new__(GPUWorker)
        worker._async_raw_frame_materializer = None
        worker.sp_cpu_group = dist.group.WORLD
        worker.cfg_cpu_group = worker.tp_cpu_group = None
        worker.pipeline = NS(stages=[])

        def release(_):
            if rank == 1:
                raise RuntimeError("decoder abort unconfirmed")
            return True

        worker._realtime_sessions = NS(release=release)
        result = worker.release_realtime_session("same-player")
        Path(results, f"{rank}.json").write_text(
            json.dumps(dict(error=result.error, output=result.output))
        )
        # Both peers must finish the observed collective and persist their
        # receipts before either process tears down its Gloo connections.
        # The production worker stays alive after releasing a player.
        dist.barrier()
        worker.sp_cpu_group = None
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires real CPU Gloo")
def test_sp_peer_cleanup_failure_is_visible_to_both_ranks(tmp_path):
    address = "file://" + str(tmp_path / "rendezvous")
    mp.spawn(_release_rank, args=(address, str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        result = json.loads((tmp_path / f"{rank}.json").read_text())
        assert result["error"] and result["output"]["cleanup_complete"] is False
        peers = {row["rank"]: row for row in result["output"]["ranks"]}
        assert set(peers) == {0, 1}
        assert peers[0]["cleaned"] is True and peers[1]["cleaned"] is False
