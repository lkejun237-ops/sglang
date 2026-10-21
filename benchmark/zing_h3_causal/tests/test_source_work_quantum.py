# SPDX-License-Identifier: Apache-2.0
"""CPU wiring tests of actual methods without importing unrelated GPU registries."""

import ast
import asyncio
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/multimodal_gen/runtime"


def methods(sampling_schedule=None):
    path = ROOT / "entrypoints/openai/realtime/adapters/zing_h3_source_adapter.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "ZingH3SourceStreamAdapter"
    )
    selected = {"wait_for_next_chunk", "prepare_next_request", "on_chunk_complete"}
    cls.body = [n for n in cls.body if getattr(n, "name", None) in selected]
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]

    class Base:
        def prepare_next_request(self, *a):
            return NS(extra={}, block_idx=1)

        def on_chunk_complete(self, *a):
            pass

    def require(ok, message):
        if not ok:
            raise ValueError(message)

    ns = dict(Base=Base, asyncio=asyncio, time=time, require=require)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            str(path),
            "exec",
        ),
        ns,
    )
    adapter = ns["ZingH3SourceStreamAdapter"]()
    adapter.__dict__.update(
        pending=deque(),
        pending_frames=0,
        _source_reservations={},
        _inflight_reservations=[],
        _selected_prompt_command=None,
        _inflight_reference_uploads=[],
        dispatched_source_frames=0,
        end=None,
        end_dispatched=False,
        more_work=False,
        source_work_quantum=1,
        clip=17,
        auto_source_preview=False,
        source_encoder={"mode": "progressive"},
        available=asyncio.Event(),
        ledger=NS(closed=False),
        config={},
        sampling_schedule=(
            sampling_schedule
            if sampling_schedule is not None
            else dict(
                solver="few_step",
                denoising_step_list=[1000, 750, 500, 250],
                initial_block_count=0,
                initial_denoising_step_list=None,
            )
        ),
        reference_ids=[],
        first_source_received_unix_ns=None,
        source_preview_sha256=None,
        profile={},
        joint_wait={},
    )
    return adapter


@pytest.mark.parametrize(
    "sampling_schedule",
    [
        pytest.param(
            dict(
                solver="few_step",
                denoising_step_list=[1000, 750, 500, 250],
                initial_block_count=0,
                initial_denoising_step_list=None,
            ),
            id="four-step",
        ),
        pytest.param(
            dict(
                solver="few_step",
                denoising_step_list=[1000, 750],
                initial_block_count=3,
                initial_denoising_step_list=[1000, 750, 500, 250],
            ),
            id="mixed-four-then-two",
        ),
    ],
)
def test_continuation_keeps_new_source_queued_and_never_repeats_eos(sampling_schedule):
    adapter = methods(sampling_schedule)
    adapter.more_work = True
    adapter.end = {"frame_end": 17, "event_id": 8}
    adapter.end_dispatched = True
    row = {"payload": {"num_frames": 1}}
    adapter.pending.append(row)
    adapter.pending_frames = 1
    batch = adapter.prepare_next_request(None, None, None)
    assert batch.extra["zing_source_command"]["appends"] == []
    assert batch.extra["zing_source_command"]["end"] is None
    assert batch.extra["zing_source_command"]["sampling_schedule"] == sampling_schedule
    assert list(adapter.pending) == [row] and adapter.pending_frames == 1


def test_zero_rgb_more_work_wakes_but_needs_source_blocks_and_cancels():
    async def run():
        adapter = methods()
        session = NS(request=NS(max_chunks=None), generate_chunk_cnt=3)
        released = []
        adapter._inflight_reservations.append(NS(release=lambda: released.append(True)))
        adapter.on_chunk_complete(session, NS(usage={"zing_source_more_work": True}))
        assert released == [True] and adapter._inflight_reservations == []
        await asyncio.wait_for(adapter.wait_for_next_chunk(session), 0.1)
        adapter.on_chunk_complete(session, NS(usage={"zing_source_more_work": False}))
        task = asyncio.create_task(adapter.wait_for_next_chunk(session))
        await asyncio.sleep(0)
        assert not task.done()
        adapter.ledger.closed = True
        adapter.available.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        adapter.on_chunk_complete(session, NS(usage={"zing_source_complete": True}))
        assert session.request.max_chunks == 3

    asyncio.run(run())


def send_wait():
    path = ROOT / "entrypoints/openai/realtime/realtime_video_api.py"
    fn = next(
        n
        for n in ast.parse(path.read_text()).body
        if getattr(n, "name", None) == "_wait_for_source_chunk_output"
    )
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns[fn.name]


@pytest.mark.parametrize("failure", [False, True])
def test_slow_ws_backpressure_and_error_before_next_work(failure):
    async def run():
        release = asyncio.Event()

        async def send():
            await release.wait()
            if failure:
                raise RuntimeError("injected send failure")

        pending = asyncio.create_task(send())
        wait = asyncio.create_task(send_wait()(NS(source_work_quantum=1), pending))
        await asyncio.sleep(0)
        assert not wait.done()
        release.set()
        if failure:
            with pytest.raises(RuntimeError, match="send failure"):
                await wait
        else:
            assert await wait is None

    asyncio.run(run())


def test_default_send_policy_does_not_gain_extra_wait():
    async def run():
        pending = asyncio.create_task(asyncio.sleep(10))
        assert await send_wait()(NS(source_work_quantum=0), pending) is pending
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(run())


def test_remote_readiness_is_identical_despite_different_producer_queue_states():
    path = (
        ROOT / "pipelines_core/stages/model_specific_stages/zing_h3/remote_encoder.py"
    )
    tree = ast.parse(path.read_text())
    available = next(
        n for n in tree.body if getattr(n, "name", None) == "available_latents"
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "RemoteEncoderSession"
    )
    cls.body = [n for n in cls.body if getattr(n, "name", None) == "ready"]
    ns = {}
    exec(
        compile(ast.Module(body=[available, cls], type_ignores=[]), str(path), "exec"),
        ns,
    )
    replicas = [ns["RemoteEncoderSession"](), ns["RemoteEncoderSession"]()]
    for i, replica in enumerate(replicas):
        replica.disposed = False
        replica.accepted_frames = 17
        replica.emitted_latents = 1
        replica.ended = False
        replica.results = NS(qsize=lambda: i)
    assert all(r.ready for r in replicas)
    for replica in replicas:
        replica.emitted_latents = 5
    assert not any(r.ready for r in replicas)
