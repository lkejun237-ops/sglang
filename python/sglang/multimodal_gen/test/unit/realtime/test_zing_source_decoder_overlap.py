# SPDX-License-Identifier: Apache-2.0
"""Bounded source decode/output overlap through the real API scheduling loop."""

import asyncio
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import realtime_video_api
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


@pytest.fixture
def pipeline(monkeypatch):
    """Run the actual API loop with controllable worker and output completion."""
    state = SimpleNamespace(
        started=[],
        sent=[],
        cancelled=[],
        results=[],
        errors=[],
        output_started=asyncio.Event(),
        output_release=asyncio.Event(),
        second_started=asyncio.Event(),
        second_completed=asyncio.Event(),
        input_waiting=asyncio.Event(),
        input_cancelled=asyncio.Event(),
        worker_release=asyncio.Event(),
        last_output_started=asyncio.Event(),
        last_output_release=asyncio.Event(),
        fail_output=False,
        block_input=False,
        block_worker=False,
        block_last_output=False,
    )

    class Adapter:
        source_work_quantum = 1
        source_decoder_overlap = True

        async def wait_for_next_chunk(self, session):
            if session.next_chunk_index and state.block_input:
                state.input_waiting.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    state.input_cancelled.set()
            await asyncio.sleep(0)

        async def prepare_conditioning(self, session, args):
            pass

        def prepare_next_request(self, session, args, chunk):
            return SimpleNamespace(
                block_idx=chunk.index,
                condition_inputs={},
                realtime_event_id=session.interactive_event_version,
                request_id=chunk.request_id,
            )

        def on_chunk_complete(self, session, result):
            session.generate_chunk_completed()
            if session.generate_chunk_cnt == 2:
                state.second_completed.set()

        def session_ready_message(self, session):
            return {}

    async def worker(_client, batch):
        state.started.append(batch.block_idx)
        if batch.block_idx == 1:
            state.second_started.set()
            if state.block_worker:
                await state.worker_release.wait()
        result = OutputBatch(
            raw_frame_shared_memory_ref={"index": batch.block_idx},
            realtime_request_metadata={"event_id": batch.realtime_event_id},
        )
        state.results.append(result)
        return None, result

    async def output(_ws, session, chunk, batch, result, *_args):
        if batch.block_idx == 0:
            state.output_started.set()
            await state.output_release.wait()
            if state.fail_output:
                raise RuntimeError("Output failed")
        if batch.block_idx == 2 and state.block_last_output:
            state.last_output_started.set()
            await state.last_output_release.wait()
        state.sent.append((batch.block_idx, batch.realtime_event_id))
        # Successful send has consumed ownership before the done callback.
        result.raw_frame_shared_memory_ref = None

    async def error(message, _ws, **kwargs):
        state.errors.append(message)

    for name in (
        "_observe_realtime_result_stage_metrics",
        "_emit_realtime_result_stage_traces",
        "log_realtime_trace",
        "observe_stage_ms",
    ):
        monkeypatch.setattr(realtime_video_api, name, lambda *_a, **_k: None)
    monkeypatch.setattr(
        realtime_video_api, "_collect_realtime_result_stage_metrics", lambda *_a: {}
    )
    monkeypatch.setattr(
        realtime_video_api,
        "get_global_server_args",
        lambda: SimpleNamespace(model_id="test"),
    )
    monkeypatch.setattr(realtime_video_api, "process_generation_batch", worker)
    monkeypatch.setattr(realtime_video_api, "_send_output_and_log", output)
    monkeypatch.setattr(realtime_video_api, "write_error_msg", error)
    monkeypatch.setattr(
        realtime_video_api,
        "cancel_async_raw_rgb_frame_reference",
        lambda ref: state.cancelled.append(ref["index"]),
    )
    session = GenerateSession()
    session.adapter = Adapter()
    session.request = SimpleNamespace(
        max_chunks=3, realtime_interactive_event_grace_ms=0
    )
    state.session = session
    return state


@pytest.mark.parametrize("overlap", [False, True])
def test_decoder_output_overlap_is_bounded_and_preserves_event_order(pipeline, overlap):
    async def run():
        p = pipeline
        p.session.adapter.source_decoder_overlap = overlap
        task = asyncio.create_task(
            realtime_video_api._generate_loop_local(SimpleNamespace(), p.session)
        )
        try:
            await asyncio.wait_for(p.output_started.wait(), 2)
            p.session.mark_event_version("prompt")
            if overlap:
                await asyncio.wait_for(p.second_started.wait(), 2)
                await asyncio.wait_for(p.second_completed.wait(), 2)
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                # The next DiT ran while output 0 was blocked, but there is no
                # unbounded dispatch-ahead when the client cannot consume it.
                assert p.started == [0, 1] and p.sent == []
            else:
                await asyncio.sleep(0)
                assert p.started == [0] and p.sent == []
            p.output_release.set()
            await asyncio.wait_for(task, 2)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert p.started == [0, 1, 2]
        assert p.sent == [(0, 0), (1, 1), (2, 1)]
        assert not p.cancelled and not p.errors
        assert not p.session.active_chunks

    asyncio.run(run())


def test_output_failure_wakes_source_credit_wait(pipeline):
    async def run():
        p = pipeline
        p.block_input = p.fail_output = True
        task = asyncio.create_task(
            realtime_video_api._generate_loop_local(SimpleNamespace(), p.session)
        )
        await asyncio.wait_for(p.input_waiting.wait(), 2)
        p.output_release.set()
        await asyncio.wait_for(task, 2)
        assert p.input_cancelled.is_set()
        assert p.started == [0] and p.cancelled == [0]
        assert len(p.errors) == 1 and not p.sent
        # A failed session must not leave a pending input/output task that
        # interferes with the next session on the same deployment.
        p.block_input = p.fail_output = False
        next_session = GenerateSession()
        next_session.adapter = p.session.adapter
        next_session.request = SimpleNamespace(
            max_chunks=1, realtime_interactive_event_grace_ms=0
        )
        await asyncio.wait_for(
            realtime_video_api._generate_loop_local(SimpleNamespace(), next_session), 2
        )
        assert p.sent == [(0, 0)]

    asyncio.run(run())


def test_successful_output_still_waits_for_source_input(pipeline):
    async def run():
        p = pipeline
        p.block_input = True
        task = asyncio.create_task(
            realtime_video_api._generate_loop_local(SimpleNamespace(), p.session)
        )
        await asyncio.wait_for(p.input_waiting.wait(), 2)
        p.output_release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not task.done() and p.started == [0]
        task.cancel()
        await asyncio.wait_for(task, 2)
        assert p.input_cancelled.is_set()
        assert not p.errors

    asyncio.run(run())


def test_last_output_is_drained_before_loop_finishes(pipeline):
    async def run():
        p = pipeline
        p.output_release.set()
        p.block_last_output = True
        task = asyncio.create_task(
            realtime_video_api._generate_loop_local(SimpleNamespace(), p.session)
        )
        await asyncio.wait_for(p.last_output_started.wait(), 2)
        assert p.sent == [(0, 0), (1, 0)] and not task.done()
        p.last_output_release.set()
        await asyncio.wait_for(task, 2)
        assert p.sent == [(0, 0), (1, 0), (2, 0)] and not p.cancelled

    asyncio.run(run())


def test_cancel_drains_inflight_dit_and_releases_both_outputs(pipeline):
    async def run():
        p = pipeline
        p.block_worker = True
        task = asyncio.create_task(
            realtime_video_api._generate_loop_local(SimpleNamespace(), p.session)
        )
        await asyncio.wait_for(p.second_started.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()  # Cancelling the waiter does not cancel GPU work.
        p.worker_release.set()
        await asyncio.wait_for(task, 2)
        assert sorted(p.cancelled) == [0, 1]
        assert all(r.raw_frame_shared_memory_ref is None for r in p.results)
        assert not p.sent

    asyncio.run(run())
