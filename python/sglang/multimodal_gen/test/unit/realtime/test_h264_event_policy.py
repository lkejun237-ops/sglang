# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
import sglang.multimodal_gen.runtime.realtime.h264_media_pipeline as pipeline_module
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import decode_message
from sglang.multimodal_gen.runtime.realtime.h264_media_pipeline import (
    H264MediaPipeline,
    H264PipelineConfig,
)
from sglang.multimodal_gen.test.unit.realtime.test_h264_media_pipeline import (
    _batch,
    _FakeProcess,
    _header,
)


@pytest.fixture
def encoder(monkeypatch):
    process = _FakeProcess()

    async def create_subprocess(*_args, **_kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    return process


def _frame(value):
    return bytes([value, 0, 0]) * 4


def _pipeline(wires, **options):
    async def send(wire):
        wires.append(decode_message(wire))

    return H264MediaPipeline(
        session_id="session",
        generation_id="generation",
        send=send,
        config=H264PipelineConfig(enabled=True, **options),
    )


@pytest.mark.parametrize("backpressure", [False, True])
@pytest.mark.parametrize("events", [(100, 101, 102, 103), (100, 100, 101, 100)])
def test_fresh_frames_survive_input_updates_and_finalization(
    encoder, backpressure, events
):
    async def run():
        wires = []
        pipeline = _pipeline(wires, gateway_backpressure=backpressure)
        expected = []
        for index, event_id in enumerate(events):
            header = replace(_header(), event_id=event_id, chunk_index=index)
            frames = [_frame(index * 2 + 1), _frame(index * 2 + 2)]
            expected.extend(frames)
            pipeline.enqueue(header, _batch(*frames))
            pipeline.enqueue_completion(
                replace(header, is_final_chunk=index == len(events) - 1), num_frames=2
            )
        assert pipeline._queued_frames == len(expected)
        assert pipeline.dropped_frames == 0
        await asyncio.wait_for(pipeline.close(), 2)
        assert encoder.stdin.frames == expected
        batches = [m for m in wires if m["type"] == "media_batch"]
        assert [m["event_id"] for m in batches] == [e for e in events for _ in range(2)]
        assert [m["first_frame_index"] for m in batches] == list(range(len(expected)))
        assert len([m for m in wires if m["type"] == "media_chunk_complete"]) == len(
            events
        )
        assert wires[-1]["type"] == "media_chunk_complete"
        assert wires[-1]["is_final_chunk"]
        assert pipeline.dropped_frames == 0
        assert all("queue_full_dropped_frames" not in m for m in batches)

    asyncio.run(run())


def test_capacity_trims_oldest_rgb_but_keeps_completion_markers(encoder, caplog):
    async def run():
        wires = []
        pipeline = _pipeline(wires, max_queued_frames=3, live_edge_frames=0)
        for index in range(5):
            header = replace(_header(), event_id=100 + index, chunk_index=index)
            pipeline.enqueue(header, _batch(_frame(index)))
            pipeline.enqueue_completion(header, num_frames=1)
            assert pipeline._queued_frames <= 3
        assert pipeline.queue_full_dropped_frames == 2
        assert pipeline.latency_dropped_frames == 0
        await asyncio.wait_for(pipeline.close(), 2)
        assert encoder.stdin.frames == [_frame(i) for i in range(2, 5)]
        assert len([m for m in wires if m["type"] == "media_chunk_complete"]) == 5
        assert (
            "total=2 queue_full=2 latency_expired=0 startup_trim=0 final=True"
            in caplog.text
        )

    caplog.set_level(logging.INFO, logger=pipeline_module.__name__)
    asyncio.run(run())


@pytest.mark.parametrize("new_frame", [False, True])
def test_age_expiry_is_independent_of_event_id_and_retains_latest(
    encoder, monkeypatch, new_frame
):
    async def run():
        clock = [100.0]
        monkeypatch.setattr(
            pipeline_module,
            "time",
            SimpleNamespace(
                time=lambda: clock[0],
                monotonic=time.monotonic,
                perf_counter=time.perf_counter,
            ),
        )
        wires = []
        pipeline = _pipeline(wires, gateway_backpressure=True)
        pipeline.enqueue(replace(_header(), event_id=101), _batch(_frame(1), _frame(2)))
        clock[0] += 1
        if new_frame:
            pipeline.enqueue(replace(_header(), event_id=100), _batch(_frame(3)))
        assert pipeline.dropped_frames == 0
        await asyncio.wait_for(pipeline.close(), 2)
        assert encoder.stdin.frames == [_frame(3 if new_frame else 2)]
        assert pipeline.latency_dropped_frames == (2 if new_frame else 1)
        assert pipeline.queue_full_dropped_frames == 0
        assert pipeline.dropped_frames == pipeline.latency_dropped_frames

    asyncio.run(run())


def test_startup_trim_is_counted_separately(encoder, caplog):
    async def run():
        wires = []
        pipeline = _pipeline(wires, startup_drop_frames=2)
        pipeline.enqueue(_header(), _batch(_frame(1), _frame(2), _frame(3)))
        await asyncio.wait_for(pipeline.close(), 2)
        assert encoder.stdin.frames == [_frame(3)]
        assert pipeline.dropped_frames == pipeline.startup_dropped_frames == 2
        assert (
            pipeline.queue_full_dropped_frames == pipeline.latency_dropped_frames == 0
        )
        assert (
            "total=2 queue_full=0 latency_expired=0 startup_trim=2 final=True"
            in caplog.text
        )

    caplog.set_level(logging.INFO, logger=pipeline_module.__name__)
    asyncio.run(run())
