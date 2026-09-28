# SPDX-License-Identifier: Apache-2.0
"""Actual multi-process Gloo/CNN regression; not GPU/learned-weight acceptance."""

import os
import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_zing_h3_progressive_kernel import configure_tiling, make_codecs

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import klvae
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
    create_session,
    expected_progressive_slices,
)


# Workers fork from a clean server process, not from the pytest process whose
# shared model fixture leaves a default process group initialized; preloading
# the heavy dependencies pays their import once instead of once per worker.
mp.get_context("forkserver").set_forkserver_preload(
    [
        "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae",
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder",
    ]
)


# Representative cases run by default; the full matrix needs this opt-in.
_FULL = pytest.mark.skipif(
    os.environ.get("ZING_FULL_TILE_TESTS") != "1",
    reason="set ZING_FULL_TILE_TESTS=1 for the full tile matrix",
)


def _worker(
    rank, size, path, stack, canvas, failure, mode="parallel", sp1_reference=None
):
    torch.set_num_threads(1)
    codecs = make_codecs()
    torch.manual_seed(7)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=45),
    )
    try:
        group = SimpleNamespace(
            rank_in_group=rank,
            world_size=size,
            ranks=list(range(size)),
            cpu_group=dist.group.WORLD,
            device_group=dist.group.WORLD,
        )
        klvae.get_tile_parallel_state = lambda: (rank, size)
        klvae.get_decode_parallel_group_coordinator = lambda: group
        model = configure_tiling(codecs, stack)
        if mode == "auto":
            model.to(torch.bfloat16)
            codecs.runtime["vae_config"].update(
                encode_mode="deterministic", encode_dtype="bfloat16"
            )
        torch.manual_seed(23)
        pixels = torch.randint(256, (3, 34, canvas, canvas), dtype=torch.uint8)
        os.environ["MINIMAX_H3_PROGRESSIVE_ENCODER_TILING"] = "replicated"
        baseline = create_session(codecs, capacity_frames=34)
        if mode == "auto":
            os.environ.pop("MINIMAX_H3_PROGRESSIVE_ENCODER_TILING", None)
        else:
            os.environ["MINIMAX_H3_PROGRESSIVE_ENCODER_TILING"] = "parallel"
        stream = create_session(codecs, capacity_frames=34)
        tile_count = len(model.split_tiles(canvas)[0]) ** 2
        parallel = size > 1 and tile_count >= size
        ids = (
            list(range(rank, tile_count, size)) if parallel else list(range(tile_count))
        )
        if failure:
            count = 5 if failure == "ended" else 9
            stream.append(pixels[:, :count], frame_start=0)
            stream.pop()
            previous = stream.state_usage()
            if failure == "kernel" and rank == size - 1:

                def fail(*args):
                    raise RuntimeError("injected kernel failure")

                model.quant_conv.forward = fail
            elif failure == "state" and rank == size - 1:
                stream.reset()
            elif failure == "ended" and rank == size - 1:
                stream.finish()
            elif failure == "dispose" and rank == size - 1:
                stream.dispose()
            elif failure == "dtype" and rank == size - 1:
                codecs.runtime["vae_config"]["encode_dtype"] = "float32"
            with pytest.raises(RuntimeError, match="Progressive encoder"):
                stream.pop()
            if failure in ("kernel", "dtype"):
                assert stream.state_usage() == previous
            stream.dispose()
            assert stream.state_usage().retained_bytes == 0
            return
        # Network chunking differs between independent baseline and candidate.
        baseline.append(pixels, frame_start=0)
        expected = []
        while baseline.ready:
            expected.append(baseline.pop())
        baseline.finish()
        if sp1_reference is not None:
            if size == 1:
                torch.save([item.values for item in expected], sp1_reference)
            sp1 = torch.load(sp1_reference, weights_only=True)
        indices = []
        total_work = 0
        for i in range(pixels.shape[1]):
            stream.append(pixels[:, i : i + 1], frame_start=i)
            while stream.ready:
                actual = stream.pop()
                reference = expected[len(indices)]
                references = [reference.values]
                if sp1_reference is not None:
                    references.append(sp1[len(indices)])
                for tensor in references:
                    assert actual.values.shape == tensor.shape
                    assert actual.values.dtype == tensor.dtype
                    assert torch.equal(
                        actual.values.contiguous().view(torch.uint8),
                        tensor.contiguous().view(torch.uint8),
                    )
                assert torch.isfinite(actual.values).all()
                indices.append((actual.start, actual.end, actual.source_dependency_end))
                receipt = stream.cache.last_execution
                assert receipt["requested_mode"] == mode
                assert receipt["local_tile_ids"] == ids
                assert receipt["gather_calls"] == int(parallel)
                assert receipt["encoder_ranks"] == (size if parallel else 1)
                total_work += receipt["temporal_positions"]
                if stream.cache.states:
                    cache_ids = {key[0] for key in stream.cache.states}
                    assert cache_ids == ({tuple(ids)} if stack else set(ids))
        stream.finish()
        assert indices == expected_progressive_slices(34)
        assert stream.state_usage().retained_bytes == 0
        count = torch.tensor(total_work)
        dist.all_reduce(count)
        expected_work = baseline.state_usage().temporal_positions
        assert count.item() == expected_work * (1 if parallel else size)
        stream.dispose()
        baseline.dispose()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "stack,size,canvas",
    [
        (False, 2, 32),
        (True, 4, 32),
        (False, 8, 24),
        *(
            pytest.param(*case, marks=_FULL)
            for case in (
                (False, 1, 24),
                (False, 4, 32),
                (False, 8, 32),
                (True, 1, 24),
                (True, 2, 32),
                (True, 8, 24),
                (True, 8, 32),
            )
        ),
    ],
)
def test_distributed_tile_ownership_and_exact_moments(tmp_path, stack, size, canvas):
    torch.set_num_threads(1)
    if sys.platform == "darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    mp.start_processes(
        _worker,
        args=(size, str(tmp_path / "rendezvous"), stack, canvas, None),
        nprocs=size,
        join=True,
        start_method="forkserver",
    )


@pytest.mark.parametrize(
    "failure",
    [
        "kernel",
        "state",
        *(pytest.param(f, marks=_FULL) for f in ("ended", "dispose", "dtype")),
    ],
)
def test_distributed_failure_is_consistent(tmp_path, failure):
    torch.set_num_threads(1)
    if sys.platform == "darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    mp.start_processes(
        _worker,
        args=(2, str(tmp_path / "rendezvous"), False, 24, failure),
        nprocs=2,
        join=True,
        start_method="forkserver",
    )


def test_default_sp1_sp2_sp4_bitwise_moments(tmp_path):
    torch.set_num_threads(1)
    if sys.platform == "darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    reference = str(tmp_path / "sp1.pt")
    for size in (1, 2, 4):
        mp.start_processes(
            _worker,
            args=(
                size,
                str(tmp_path / f"rendezvous-{size}"),
                False,
                32,
                None,
                "auto",
                reference,
            ),
            nprocs=size,
            join=True,
            start_method="forkserver",
        )
