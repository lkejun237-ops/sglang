# SPDX-License-Identifier: Apache-2.0
"""Exercise the real two-rank startup failure path without CUDA or models."""

import json
import socket
from datetime import timedelta
from pathlib import Path

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp


def _rank(rank, address, output, fail):
    from sglang.multimodal_gen.runtime.realtime.worker_connection_ports import (
        ConnectionPorts,
        coordinated_descriptor,
    )

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=address,
        timeout=timedelta(seconds=15),
    )

    class Peers:
        rank_in_group = rank

        def broadcast_object(self, value):
            values = [value]
            dist.broadcast_object_list(values, src=0)
            return values[0]

    try:
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()

            def prepare():
                assert rank == 0, "Only the leader may reserve endpoints"
                if fail:
                    # Reproduce a real bind failure after all peers joined.
                    ConnectionPorts(
                        "dit",
                        1,
                        worker_id="startup-test",
                        host="127.0.0.1",
                        bind_host="127.0.0.1",
                        rendezvous=occupied.getsockname()[1],
                    )
                    raise AssertionError("Expected the bind to fail")
                return {"worker_id": "startup-test", "service_id": "startup-test/0"}

            try:
                descriptor = coordinated_descriptor(Peers(), prepare)
                result = {"descriptor": descriptor}
            except RuntimeError as error:
                result = {"error": str(error)}
            Path(output, str(rank) + ".json").write_text(json.dumps(result))
            # Both peers reach teardown instead of one waiting for a descriptor.
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires real CPU Gloo")
@pytest.mark.parametrize("fail", [False, True])
def test_endpoint_setup_result_reaches_both_sp_ranks(tmp_path, fail):
    mp.spawn(
        _rank,
        args=("file://" + str(tmp_path / "rendezvous"), str(tmp_path), fail),
        nprocs=2,
        join=True,
    )
    results = [
        json.loads((tmp_path / (str(i) + ".json")).read_text()) for i in range(2)
    ]
    assert results[0] == results[1]
    if fail:
        assert "Worker endpoint setup failed: OSError" in results[0]["error"]
    else:
        assert results[0]["descriptor"]["service_id"] == "startup-test/0"
