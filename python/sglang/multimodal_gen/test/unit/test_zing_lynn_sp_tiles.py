# SPDX-License-Identifier: Apache-2.0
"""Lynn SP tile parallelism: real multi-process Gloo groups on a tiny FP32 CPU decoder.

Every rank also decodes the SP1 baseline itself and requires bitwise equality;
this is not GPU or learned-weight acceptance.
"""

import gc
import os
import sys
import time
import weakref
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from msgspec.structs import replace
from test_zing_lynn_stream import ARCHITECTURE, CONFIG, _provenance

from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.geometry import (
    split_tiles,
    tile_shard,
)
from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.model import (
    StreamingDecoder,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    decoder as decoder_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_decoder import (
    SourceLynnDecoder,
)

mp.get_context("forkserver").set_forkserver_preload(
    ["sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.model"]
)

ADAPTER = decoder_module.DECODERS["lynn_streaming"]
# Latent (height, width) -> tile count with 8px tiles, 4px overlap, 4x ratio:
# 2x2 -> 1 tile, 3x3 -> 4 tiles, 3x5 -> 8 tiles.
GEOMETRIES = {1: (2, 2), 4: (3, 3), 8: (3, 5)}


@pytest.mark.parametrize(
    "count,world_size,expected",
    [
        (8, 1, [(0, 8)]),
        (8, 2, [(0, 4), (4, 8)]),
        (8, 4, [(0, 2), (2, 4), (4, 6), (6, 8)]),
        (8, 3, [(0, 3), (3, 6), (6, 8)]),
        (5, 4, [(0, 2), (2, 3), (3, 4), (4, 5)]),
        (2, 4, [(0, 1), (1, 2), (2, 2), (2, 2)]),
        (1, 2, [(0, 1), (1, 1)]),
        (60, 4, [(0, 15), (15, 30), (30, 45), (45, 60)]),
        (54, 4, [(0, 14), (14, 28), (28, 41), (41, 54)]),
    ],
)
def test_tile_shard_is_contiguous_with_leading_remainder(count, world_size, expected):
    bounds = [tile_shard(count, rank, world_size) for rank in range(world_size)]
    assert bounds == expected


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("count", range(0, 70))
def test_tile_shard_covers_every_tile_once(count, world_size):
    bounds = [tile_shard(count, rank, world_size) for rank in range(world_size)]
    sizes = [end - start for start, end in bounds]
    assert bounds[0][0] == 0 and bounds[-1][1] == count
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))
    assert max(sizes) - min(sizes) <= 1 and sizes == sorted(sizes, reverse=True)


@pytest.mark.parametrize("tiles,geometry", GEOMETRIES.items())
def test_fixture_geometries_have_expected_tile_counts(tiles, geometry):
    counts = [
        len(
            split_tiles(
                n * 4, size=CONFIG.tile_size, overlap=CONFIG.tile_overlap, ratio=4
            )[0]
        )
        for n in geometry
    ]
    assert counts[0] * counts[1] == tiles


def test_sp1_never_gathers():
    class NoCollective:
        rank_in_group, world_size = 0, 1

        def all_gather(self, *args, **kwargs):
            raise AssertionError("SP1 must not communicate")

    model = _model()
    latent = _latent(3, (3, 5))
    expected = model.decode(latent, model.protocol.frame_count(3))
    actual = model.decode(latent, model.protocol.frame_count(3), group=NoCollective())
    assert torch.equal(actual, expected)


def test_sp_tile_group_is_none_without_distributed_sp():
    assert decoder_module.sp_tile_group() is None


class _GlooGroup:
    """The GroupCoordinator surface Lynn uses, over the default Gloo group."""

    def __init__(self, rank, world_size):
        self.rank_in_group, self.world_size = rank, world_size
        self.gathers = 0
        self.transfers = []
        self.cpu_group = self.device_group = dist.group.WORLD

    def all_gather(self, tensor, dim=0):
        parts = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(parts, tensor)
        self.gathers += 1
        return torch.cat(parts, dim)

    def all_gather_into(self, output, tensor):
        from sglang.multimodal_gen.runtime.distributed.group_coordinator import (
            GroupCoordinator,
        )

        assert callable(getattr(GroupCoordinator, "all_gather_into", None)), (
            "GroupCoordinator must support caller-owned gather storage"
        )
        assert tensor.ndim == output.ndim == 1
        assert tensor.is_contiguous() and output.is_contiguous()
        assert output.numel() == tensor.numel() * self.world_size
        self.transfers.append((tensor.data_ptr(), output.data_ptr(), tensor.numel()))
        result = GroupCoordinator.all_gather_into(self, output, tensor)
        assert result is output
        self.gathers += 1
        return result

    def all_reduce(self, tensor, op=dist.ReduceOp.SUM):
        dist.all_reduce(tensor, op=op)
        return tensor


def _model(tile_batch_size=0):
    # FP32 compute: single-thread CPU BF16 kernels are not run-to-run exact for
    # small tile batches under load (seen at SP1 with tile_batch_size=1), which
    # would mask the ownership/gather/stitch checks. BF16 exactness is a GPU gate.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = StreamingDecoder(
            replace(CONFIG, tile_batch_size=tile_batch_size), ARCHITECTURE
        )
        with torch.no_grad():
            for parameter in model.parameters():
                if not parameter.any():
                    parameter.normal_(std=0.5)
    return model.to_compute("cpu", torch.float32).eval().requires_grad_(False)


def _latent(count, geometry, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 24, count, *geometry, generator=generator)


def _push_chunks(stream, latent, sizes):
    outputs, cursor = [], 0
    for size in sizes:
        outputs.append(stream.push(latent[:, :, cursor : cursor + size]))
        cursor += size
    return outputs


def _assert_equal(actual, expected, label):
    if not torch.equal(actual, expected):
        diff = (actual - expected).abs()
        frames = diff.amax(dim=(0, 1, 3, 4)).nonzero().flatten().tolist()
        raise AssertionError(
            f"{label}: max_abs={diff.max().item():.3g} frames={frames}"
        )


def _state_bytes(state):
    return sum(
        k.numel() * k.element_size() + v.numel() * v.element_size()
        for layers in state.tiles.values()
        for k, v in layers
    )


def _setup(rank, world_size, path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    group = _GlooGroup(rank, world_size)
    decoder_module.sp_tile_group = lambda: group
    return group


def _exact_worker(rank, world_size, path, tiles, tile_batch_size):
    group = _setup(rank, world_size, path)
    try:
        model, geometry = _model(tile_batch_size), GEOMETRIES[tiles]
        # 12 latents wrap the five-phase protocol twice (F(12)=39); 37 frames trims the tail.
        latent, frames = _latent(12, geometry), 37
        baseline = model.decode(latent, frames)

        # Offline: the registry adapter's decode is the codecs.decode_visual path.
        gathers = group.gathers
        _assert_equal(
            ADAPTER.decode(model, latent, frames), baseline, f"rank {rank} offline"
        )
        assert group.gathers - gathers == 12

        # Finite realtime: business chunks 1,2,2,3,2,2 with the request frame limit.
        stream = ADAPTER.new_stream(model, frame_limit=frames)
        assert stream.group is group
        outputs = _push_chunks(stream, latent, (1, 2, 2, 3, 2, 2))
        assert [o.shape[2] for o in outputs] == [1, 8, 8, 9, 8, 3]
        start, end = tile_shard(tiles, rank, world_size)
        owned = set(stream.state.tiles)
        assert len(owned) == end - start
        # Only owned tiles keep history; each holds W-1 = 2 latents per layer.
        single = model.new_stream()
        single.push(latent)
        layout = sorted(single.state.tiles, key=lambda t: (t[0], t[1]))
        assert owned == set(layout[start:end])
        if end > start:
            assert all(
                k.shape[1] == v.shape[1] == 2 * 4
                for layers in stream.state.tiles.values()
                for k, v in layers
            )
        assert _state_bytes(stream.state) * tiles == _state_bytes(single.state) * (
            end - start
        )
        stream.finish(frames)
        _assert_equal(torch.cat(outputs, dim=2), baseline, f"rank {rank} finite")

        # Continuous source: irregular appends, held tail until EOS proves it valid.
        codecs = SimpleNamespace(modules={ADAPTER.module_key: model}, decoder=ADAPTER)
        source = SourceLynnDecoder(
            codecs, capacity_latents=20, provenance_for=_provenance
        )
        assert source.stream.group is group
        slices, cursor = [], 0
        for size in (1, 3, 1, 4, 3):
            dependency = min(frames, model.protocol.frame_count(cursor + size))
            source.append(
                latent[:, :, cursor : cursor + size],
                latent_start=cursor,
                source_dependency_end=dependency,
            )
            cursor += size
            while source.ready:
                slices.append(source.pop())
        source.finish(total_frames=frames)
        while source.ready:
            slices.append(source.pop())
        assert [(s.start, s.end) for s in slices][-1][1] == frames
        assert all(a.end == b.start for a, b in zip(slices, slices[1:]))
        _assert_equal(
            torch.cat([s.values for s in slices], dim=2),
            baseline,
            f"rank {rank} source",
        )
        source.dispose()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size,tiles,tile_batch_size",
    [
        (2, 4, 0),  # divisible
        (3, 4, 0),  # remainder to rank 0
        (2, 8, 0),
        (4, 8, 0),
        (3, 8, 2),  # per-rank batches of the configured tile_batch_size
        (2, 1, 0),  # fewer tiles than ranks
        (4, 1, 0),
    ],
)
def test_sp_tiles_match_sp1_on_every_path(tmp_path, world_size, tiles, tile_batch_size):
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    mp.start_processes(
        _exact_worker,
        args=(world_size, str(tmp_path / "rendezvous"), tiles, tile_batch_size),
        nprocs=world_size,
        join=True,
        start_method="forkserver",
    )


def _failure_worker(rank, world_size, path, failing_rank, tiles):
    _setup(rank, world_size, path)
    try:
        model = _model()
        latent = _latent(6, GEOMETRIES[tiles])
        stream = ADAPTER.new_stream(model, frame_limit=None)
        stream.push(latent[:, :, :3])
        if rank == failing_rank:
            original = model._decode_tiles

            def fail(*args, **kwargs):
                model._decode_tiles = original
                raise MemoryError("injected tile failure")

            model._decode_tiles = fail
        expected = (
            (MemoryError, "injected")
            if rank == failing_rank
            else (RuntimeError, rf"SP rank\(s\) \[{failing_rank}\]")
        )
        with pytest.raises(expected[0], match=expected[1]):
            stream.push(latent[:, :, 3:])
        # The whole group terminates: every stream is closed and holds no state.
        assert stream.closed and stream.state is None
        with pytest.raises(ValueError, match="closed"):
            stream.push(latent[:, :, 5:])
        # The next session starts at phase 0 on every rank and still matches SP1.
        assert torch.equal(ADAPTER.decode(model, latent, 18), model.decode(latent, 18))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size,failing_rank,tiles",
    [
        (2, 1, 4),
        (2, 0, 4),
        (4, 3, 8),
        (4, 2, 1),
    ],  # last case: the failing rank owns no tile
)
def test_single_rank_failure_terminates_the_group(
    tmp_path, world_size, failing_rank, tiles
):
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    mp.start_processes(
        _failure_worker,
        args=(world_size, str(tmp_path / "rendezvous"), failing_rank, tiles),
        nprocs=world_size,
        join=True,
        start_method="forkserver",
    )


# Reusable pixel workspace contract. No learned weights or accelerator required.
def test_sp1_has_no_pixel_workspace():
    class NoCollective:
        rank_in_group, world_size = 0, 1

        def all_gather(self, *args, **kwargs):
            raise AssertionError("SP1 must not communicate")

        all_gather_into = all_reduce = all_gather

    stream = _model().new_stream(group=NoCollective())
    stream.push(_latent(2, GEOMETRIES[4]))
    assert getattr(stream.state, "pixel_send", None) is None
    assert getattr(stream.state, "pixel_recv", None) is None
    stream.close()


def _assert_storage_released(handles, message):
    # Gloo may fulfill its synchronous collective future before the worker
    # drops its final output view. Bound that backend lifetime, but require
    # session-owned references to be cleared immediately at each call site.
    deadline = time.monotonic() + 1.0
    while True:
        gc.collect()
        if all(torch.UntypedStorage._expired(handle) for handle in handles):
            return
        assert time.monotonic() < deadline, message
        time.sleep(0.001)


def _workspace_worker(rank, world_size, path, tiles, geometry, batch_size):
    group = _setup(rank, world_size, path)
    storage_handles = []
    try:
        model = _model()
        latent = _latent(7, geometry)
        if batch_size > 1:
            latent = torch.cat([latent + i for i in range(batch_size)], dim=0)
        _, hs, _, _, ws, _ = model._tile_layout(latent[:, :, :1])
        tile_pixels = batch_size * model.decoder.out_channels * hs[0] * ws[0]
        baseline = model.decode(latent, model.protocol.frame_count(7))
        stream = model.new_stream(group=group)
        original_decode = model._decode_tiles

        def check_preallocated(z_t, tile_ids, state, phase):
            assert getattr(state, "pixel_send", None) is not None, (
                "SP must preallocate reusable send workspace before tile decode"
            )
            assert getattr(state, "pixel_recv", None) is not None, (
                "SP must preallocate reusable receive workspace before tile decode"
            )
            return original_decode(z_t, tile_ids, state, phase)

        model._decode_tiles = check_preallocated
        original_stack = torch.stack

        def reject_pixel_stack(values, *args, **kwargs):
            assert not (values and values[0].ndim == 5), (
                "pixel packing must copy tiles directly without torch.stack"
            )
            return original_stack(values, *args, **kwargs)

        outputs, snapshots = [], []
        send_ptr = recv_ptr = None
        with patch.object(torch, "stack", reject_pixel_stack):
            for index in range(7):
                output = stream.push(latent[:, :, index : index + 1])
                outputs.append(output)
                snapshots.append(output.clone())
                send, recv = stream.state.pixel_send, stream.state.pixel_recv
                assert send.ndim == recv.ndim == 1
                assert send.dtype == recv.dtype == torch.float32
                max_tiles = (tiles + world_size - 1) // world_size
                capacity = max_tiles * tile_pixels * model.protocol.ratio + 1
                assert send.numel() == capacity
                assert recv.numel() == world_size * capacity
                pointers = send.data_ptr(), recv.data_ptr()
                if send_ptr is None:
                    send_ptr, recv_ptr = pointers
                assert pointers == (send_ptr, recv_ptr)
                frames = 1 if index % model.protocol.phases == 0 else 4
                assert group.transfers[-1] == (
                    send_ptr,
                    recv_ptr,
                    max_tiles * tile_pixels * frames + 1,
                )
                assert (
                    output.untyped_storage().data_ptr()
                    != recv.untyped_storage().data_ptr()
                )
                for previous, saved in zip(outputs, snapshots):
                    assert torch.equal(previous, saved), (
                        "next push overwrote emitted RGB"
                    )
        _assert_equal(torch.cat(outputs, dim=2), baseline, "workspace phases")
        storage_handles = [
            send.untyped_storage()._weak_ref(),
            recv.untyped_storage()._weak_ref(),
        ]
        state = stream.state
        del send, recv
        stream.close()
        assert stream.closed and stream.state is None
        assert state.pixel_send is None and state.pixel_recv is None
        _assert_storage_released(storage_handles, "closed stream retains pixel storage")
    finally:
        for handle in storage_handles:
            torch.UntypedStorage._free_weak_ref(handle)
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size,tiles,geometry,batch_size",
    [
        (2, 4, (3, 3), 1),
        (3, 4, (3, 3), 1),
        (4, 8, (3, 5), 1),
        (2, 1, (2, 2), 1),
        (3, 1, (2, 2), 1),
        (4, 1, (2, 2), 1),
        (3, 1, (1, 2), 2),
    ],
)
def test_reusable_pixel_workspace_across_phases(
    tmp_path, world_size, tiles, geometry, batch_size
):
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    mp.start_processes(
        _workspace_worker,
        args=(
            world_size,
            str(tmp_path / "workspace-rendezvous"),
            tiles,
            geometry,
            batch_size,
        ),
        nprocs=world_size,
        join=True,
        start_method="forkserver",
    )


def _allocation_failure_worker(rank, world_size, path, failing_rank, tiles, fail_call):
    group = _setup(rank, world_size, path)
    try:
        model = _model()
        latent = _latent(2, GEOMETRIES[tiles])
        stream = model.new_stream(group=group)
        original_empty = torch.empty
        workspace_refs, allocation_calls = [], 0

        def fail_workspace(*args, **kwargs):
            nonlocal allocation_calls
            shape = args[0] if args else kwargs.get("size", ())
            shape = (shape,) if isinstance(shape, int) else tuple(shape)
            if (
                len(shape) == 1
                and shape[0] > 100
                and kwargs.get("dtype") == torch.float32
            ):
                allocation_calls += 1
                if rank == failing_rank and allocation_calls == fail_call:
                    raise MemoryError("injected pixel workspace allocation failure")
                result = original_empty(*args, **kwargs)
                workspace_refs.append(weakref.ref(result))
                return result
            return original_empty(*args, **kwargs)

        original_reduce = dist.all_reduce

        def check_failure_consensus(tensor, *args, **kwargs):
            assert tensor.device.type == "cpu" and tensor.numel() == 1
            if rank == failing_rank:
                assert all(ref() is None for ref in workspace_refs), (
                    "partial workspace must be released before failure consensus"
                )
            return original_reduce(tensor, *args, **kwargs)

        with (
            patch.object(torch, "empty", fail_workspace),
            patch.object(dist, "all_reduce", check_failure_consensus),
        ):
            with pytest.raises((MemoryError, RuntimeError), match="workspace|SP rank"):
                stream.push(latent[:, :, :1])
        assert stream.closed and stream.state is None
        assert group.gathers == 0, "peers entered pixel gather after failed allocation"
        gc.collect()
        assert all(ref() is None for ref in workspace_refs), (
            "partial workspace was retained"
        )
        with pytest.raises(ValueError, match="closed"):
            stream.push(latent[:, :, :1])
        recovered = model.new_stream(group=group)
        actual = recovered.push(latent)
        _assert_equal(
            actual,
            model.decode(latent, model.protocol.frame_count(2)),
            "recovered session",
        )
        recovered.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size,failing_rank,tiles,fail_call",
    [(2, 1, 4, 1), (3, 0, 4, 2), (4, 3, 1, 2)],
)
def test_workspace_allocation_failure_synchronizes_and_recovers(
    tmp_path, world_size, failing_rank, tiles, fail_call
):
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    mp.start_processes(
        _allocation_failure_worker,
        args=(
            world_size,
            str(tmp_path / "allocation-rendezvous"),
            failing_rank,
            tiles,
            fail_call,
        ),
        nprocs=world_size,
        join=True,
        start_method="forkserver",
    )


def _retained_exception_worker(rank, world_size, path, failure_point):
    group = _setup(rank, world_size, path)
    storage_handles = []
    try:
        model = _model()
        latent = _latent(2, GEOMETRIES[4])
        stream = model.new_stream(group=group)
        stream.push(latent[:, :, :1])
        # Storage weak handles observe the C++ allocation, including ownership
        # through a view whose Python Tensor wrapper differs from the workspace.
        for tensor in (stream.state.pixel_send, stream.state.pixel_recv):
            storage_handles.append(tensor.untyped_storage()._weak_ref())
        del tensor

        def lower_gather(received, local):
            assert received._base is not None and local._base is not None
            raise ValueError("inner gather failure")

        def fail_gather(received, local):
            assert received._base is not None and local._base is not None
            if failure_point == "gather":
                raise RuntimeError("injected synchronous pixel gather failure")
            try:
                lower_gather(received, local)
            except ValueError as inner:
                outer = RuntimeError(
                    f"injected synchronous pixel {failure_point} failure"
                )
                if failure_point == "gather_context":
                    raise outer
                if failure_point == "gather_cycle":
                    inner.__cause__ = outer
                raise outer from inner

        def fail_blend(a, b, extent, dim):
            assert a.untyped_storage()._cdata == b.untyped_storage()._cdata
            raise RuntimeError("injected synchronous pixel stitch failure")

        target, name, replacement = (
            (group, "all_gather_into", fail_gather)
            if failure_point.startswith("gather")
            else (model, "_blend", fail_blend)
        )
        armed = False
        original_step = model.decode_step
        failing_latent = latent[:, :, 1:2]

        def arm_output_failure(*args, **kwargs):
            nonlocal armed
            result = original_step(*args, **kwargs)
            armed = True
            return result

        if failure_point in ("output_cat", "output_empty", "latent_transfer"):
            target, name = {
                "output_cat": (torch, "cat"),
                "output_empty": (torch.Tensor, "new_empty"),
                "latent_transfer": (torch.Tensor, "to"),
            }[failure_point]
            original_operation = getattr(target, name)

            def fail_output(*args, **kwargs):
                should_fail = (
                    args[0] is failing_latent
                    if failure_point == "latent_transfer"
                    else armed
                )
                if should_fail:
                    raise MemoryError(
                        f"injected synchronous pixel {failure_point} failure"
                    )
                return original_operation(*args, **kwargs)

            replacement = fail_output
        retained_error = None
        state = stream.state
        with (
            patch.object(target, name, replacement),
            patch.object(model, "decode_step", arm_output_failure),
        ):
            try:
                stream.push(failing_latent)
            except (RuntimeError, MemoryError) as exc:
                retained_error = exc
        assert retained_error is not None
        if failure_point in ("output_cat", "output_empty", "latent_transfer"):
            assert isinstance(retained_error, MemoryError)
        assert str(retained_error) == (
            f"injected synchronous pixel {failure_point} failure"
        )
        assert retained_error.__traceback__ is not None
        if failure_point in ("gather_cause", "gather_cycle"):
            assert isinstance(retained_error.__cause__, ValueError)
            assert str(retained_error.__cause__) == "inner gather failure"
            assert retained_error.__cause__.__traceback__ is not None
        if failure_point == "gather_context":
            assert retained_error.__cause__ is None
            assert isinstance(retained_error.__context__, ValueError)
            assert str(retained_error.__context__) == "inner gather failure"
            assert retained_error.__context__.__traceback__ is not None
        if failure_point == "gather_cycle":
            assert retained_error.__cause__.__cause__ is retained_error
        assert stream.closed and stream.state is None
        assert state.pixel_send is None and state.pixel_recv is None
        _assert_storage_released(
            storage_handles,
            f"retained {failure_point} traceback owns pixel workspace storage",
        )
        with pytest.raises(ValueError, match="closed"):
            stream.push(latent[:, :, :1])
        recovered = model.new_stream(group=group)
        actual = recovered.push(latent)
        _assert_equal(
            actual,
            model.decode(latent, model.protocol.frame_count(2)),
            "session after retained synchronous exception",
        )
        recovered.close()
        # Keep the exception alive across recovery as a caller/logging system may.
        assert retained_error.__traceback__ is not None
    finally:
        for handle in storage_handles:
            torch.UntypedStorage._free_weak_ref(handle)
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "failure_point",
    [
        "gather",
        "stitch",
        "gather_cause",
        "gather_context",
        "gather_cycle",
        "output_cat",
        "output_empty",
        "latent_transfer",
    ],
)
def test_retained_synchronous_exception_releases_workspace(tmp_path, failure_point):
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    # Both ranks inject the same local synchronous failure, so this does not
    # claim recovery from an asymmetric or asynchronous collective failure.
    mp.start_processes(
        _retained_exception_worker,
        args=(2, str(tmp_path / "retained-exception-rendezvous"), failure_point),
        nprocs=2,
        join=True,
        start_method="forkserver",
    )
