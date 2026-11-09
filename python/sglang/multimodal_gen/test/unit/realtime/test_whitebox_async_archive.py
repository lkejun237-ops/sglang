# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import time

import av
import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
    AsyncArchiveManager,
    AsyncRecording,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import (
    EngineFrame,
    VideoSpec,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.playback import H264PlaybackEncoder


async def make_recording(tmp_path, **limits):
    manager = AsyncArchiveManager(**limits)
    spec = VideoSpec(32, 32)
    recording = await manager.create_recording(
        directory=tmp_path / ("a" * 32),
        spec=spec,
        world_id="world",
        trace_id="offline-room-sidecar",
        inference={},
        ui={"mode": "browser"},
        references=[],
        object_prefix="offline/game/stream",
    )
    return manager, recording, spec


def test_archive_reply_wakes_on_ipc_without_blocking_the_event_loop():
    import queue
    from types import SimpleNamespace

    async def run():
        loop = asyncio.get_running_loop()
        waiting = asyncio.Event()

        class Replies(queue.Queue):
            def get(self, block=True, timeout=None):
                if block:
                    assert 0 < timeout <= 0.05
                    loop.call_soon_threadsafe(waiting.set)
                return super().get(block, timeout)

        replies = Replies()
        owner = SimpleNamespace(
            failure=None, process=SimpleNamespace(is_alive=lambda: True)
        )
        task = asyncio.create_task(AsyncRecording._reply(owner, replies, 1))
        # This producer must run while the worker reply is being awaited; a
        # blocking Queue.get on the event loop would prevent it from running.
        await asyncio.wait_for(waiting.wait(), 1)
        replies.put_nowait(("ack", {"last_frame": 7}))
        assert await task == ("ack", {"last_frame": 7})
        owner.failure = "QUEUE_FULL"
        with pytest.raises(RuntimeError, match="QUEUE_FULL"):
            await AsyncRecording._reply(owner, replies, 1)

    asyncio.run(run())


def test_sidecar_full_queue_never_waits_and_stops_archive_intake(tmp_path):
    async def run():
        manager, recording, _ = await make_recording(tmp_path, queue_depth=1)
        start = time.perf_counter()
        await recording.configure()
        await recording.write_source_state({"states": {"opaque": 1}})
        assert time.perf_counter() - start < 0.1
        assert recording.failure == "QUEUE_FULL"
        assert recording.manifest["recording_status"] == "failed"
        assert recording.manifest["upload_status"] == "failed"
        assert recording.manifest["upload_failure"] == "ARCHIVE_FAILED"
        assert recording.offer("state", {}) is False
        assert recording.pair(EngineFrame(7, 0, b"", {}), b"", b"", "world") is False
        await recording.finish("completed")
        assert recording.pending_bytes == 0
        saved = json.loads((recording.directory / "manifest.json").read_text())
        assert saved["upload_status"] == "failed"
        assert saved["missing_frame_range"] == {"start": 0, "end": 8}
        await manager.close()

    asyncio.run(run())


@pytest.mark.parametrize("offset", [0, 3229875])
def test_sidecar_remuxes_exact_source_and_model_access_units_with_vfr_timestamps(
    tmp_path,
    offset,
):
    async def run():
        manager, recording, spec = await make_recording(tmp_path, max_age=10)
        encoder = H264PlaybackEncoder(spec, gop=6)
        source_encoder = H264PlaybackEncoder(spec, max_bitrate=800000, gop=6)
        source_aus, model_aus = [], []
        recording.manifest["playback"] = encoder.stream
        recording.manifest["video_streams"] = {
            "whitebox": source_encoder.stream,
            "model": encoder.stream,
        }
        await recording.configure()
        for i, pts in enumerate([0, 85000, 190000]):
            pts += offset
            rgb = bytes([50 + i * 20]) * spec.frame_bytes
            frame = EngineFrame(
                i,
                spec.pts(i),
                rgb,
                {"opaque": i},
                effective_input={"held_keys": ["KeyW"]},
                provenance={"capture_pts_us": pts},
            )
            source_aus.append(source_encoder.encode(i, rgb))
            model_aus.append(
                encoder.encode(i, bytes([120 + i * 20]) * spec.frame_bytes)
            )
            assert recording.pair(frame, source_aus[-1], model_aus[-1], "world")
        await recording.finish("completed")
        assert recording.failure is None
        assert recording.manifest["recording_status"] == "complete"
        stamps = []
        for name in ("whitebox", "model"):
            with av.open(str(recording.directory / (name + ".mp4"))) as video:
                decoded = list(video.decode(video=0))
                assert len(decoded) == 3
                stamps.append(
                    [round(float(f.pts * f.time_base) * 1e6) for f in decoded]
                )
        assert stamps == [[offset, offset + 85000, offset + 190000]] * 2
        # Compare decoded pixels with the same decoder, not MP4 vs Annex B bytes.
        for name, units in (("whitebox", source_aus), ("model", model_aus)):
            decoder = av.CodecContext.create("h264", "r")
            live = [
                f.to_ndarray(format="rgb24")
                for au in units
                for f in decoder.decode(av.Packet(au["payload"]))
            ]
            with av.open(str(recording.directory / (name + ".mp4"))) as video:
                archived = [f.to_ndarray(format="rgb24") for f in video.decode(video=0)]
            assert all((a == b).all() for a, b in zip(live, archived))
            meta = recording.manifest["videos"][name]["encoder"]
            assert meta["gop"] == 6 and meta["keyframes"] == 1
            assert meta["reused_playback_bitstream"]
        source_encoder.close()
        assert recording.last_frame == 2
        encoder.close()
        await manager.close()

    asyncio.run(run())


def test_dead_recorder_is_reported_without_poisoning_next_admission(tmp_path):
    async def run():
        manager, recording, _ = await make_recording(tmp_path)
        await recording.configure()
        async with asyncio.timeout(15):
            while (
                recording.process is None
                or not (recording.directory / "manifest.json").exists()
            ):
                await asyncio.sleep(0.02)
        recording.process.kill()
        async with asyncio.timeout(10):
            while not recording.closed:
                await asyncio.sleep(0.02)
        start = time.perf_counter()
        for _ in range(100):
            await recording.write_source_state({"states": {}})
        assert time.perf_counter() - start < 0.1
        assert recording.failure
        assert recording.manifest["timeline"]["alignment"] == "unverified"
        assert manager.recorders._value == 2
        await recording.finish("completed")
        assert (
            json.loads((recording.directory / "manifest.json").read_text())[
                "recording_status"
            ]
            == "failed"
        )
        await manager.close()

    asyncio.run(run())


def test_stalled_child_does_not_block_live_offers_and_is_killed(tmp_path):
    import os
    import signal

    async def run():
        manager, recording, _ = await make_recording(tmp_path, max_age=0.15)
        async with asyncio.timeout(10):
            while not recording.worker_ready:
                await asyncio.sleep(0.01)
        os.kill(recording.process.pid, signal.SIGSTOP)
        started = time.perf_counter()
        await recording.write_source_state({"states": {"opaque": 1}})
        assert time.perf_counter() - started < 0.1
        async with asyncio.timeout(8):
            while not recording.closed:
                await asyncio.sleep(0.01)
        assert recording.failure and not recording.process.is_alive()
        await manager.close()

    asyncio.run(run())


def test_archive_disk_start_failure_and_upload_timeout_are_separate_from_live(tmp_path):
    async def run():
        # Require more free bytes than any supported test machine has.
        manager, recording, _ = await make_recording(tmp_path, reserve_bytes=10**20)
        await recording.configure()
        await recording.finish("completed")
        assert recording.failure and recording.manifest["recording_status"] == "failed"
        await manager.close()

        class Store:
            async def upload(self, *_):
                await asyncio.sleep(100)

        manager, recording, spec = await make_recording(tmp_path / "upload")
        encoder = H264PlaybackEncoder(spec)
        rgb = bytes(spec.frame_bytes)
        au = encoder.encode(0, rgb)
        recording.pair(
            EngineFrame(
                0, 0, rgb, {}, effective_input={}, provenance={"capture_pts_us": 0}
            ),
            au,
            au,
            "p",
        )
        await recording.finish("completed")
        manager.store = Store()
        manager.upload_timeout = 0.1
        started = time.perf_counter()
        await manager._publish_bounded(recording)
        assert time.perf_counter() - started < 1
        assert recording.manifest["recording_status"] == "complete"
        assert recording.manifest["upload_status"] == "failed"
        assert manager.upload_pending == 0
        encoder.close()
        await manager.close()

    asyncio.run(run())


def test_recording_admission_waits_for_boot_before_accepting_live_frames(
    tmp_path, monkeypatch
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
        AsyncRecording,
    )

    async def run():
        entered, boot = asyncio.Event(), asyncio.Event()
        original = AsyncRecording._reply
        seen = []

        async def delayed_boot(self, replies, timeout):
            if not self.worker_ready:
                seen.append(self)
                entered.set()
                await boot.wait()
            return await original(self, replies, timeout)

        monkeypatch.setattr(AsyncRecording, "_reply", delayed_boot)
        pending = asyncio.create_task(make_recording(tmp_path))
        await asyncio.wait_for(entered.wait(), 10)
        assert not pending.done(), "Live intake must not race a cold archive worker"
        assert seen[0].process.is_alive() and not seen[0].worker_ready
        boot.set()
        manager, recording, spec = await asyncio.wait_for(pending, 10)
        encoder = H264PlaybackEncoder(spec)
        try:
            assert recording.worker_ready and recording.failure is None
            await recording.configure()
            await recording.write_source_state({"states": {"opaque": 1}})
            rgb = bytes(spec.frame_bytes)
            encoded = encoder.encode(0, rgb)
            assert recording.pair(
                EngineFrame(
                    0,
                    0,
                    rgb,
                    {},
                    effective_input={},
                    provenance={"capture_pts_us": 0},
                ),
                encoded,
                encoded,
                "world",
            )
            assert recording.failure is None
            await recording.finish("completed")
            assert recording.manifest["recording_status"] == "complete"
        finally:
            encoder.close()
            await manager.close()

    asyncio.run(run())


def test_cancelled_archive_admission_closes_started_worker_and_releases_capacity(
    tmp_path, monkeypatch
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
        AsyncRecording,
    )

    async def run():
        entered = asyncio.Event()
        seen = []

        async def held_boot(self, replies, timeout):
            seen.append(self)
            entered.set()
            await asyncio.Future()

        monkeypatch.setattr(AsyncRecording, "_reply", held_boot)
        pending = asyncio.create_task(make_recording(tmp_path))
        await asyncio.wait_for(entered.wait(), 10)
        assert not pending.done()
        recording = seen[0]
        assert recording.process.is_alive()
        pending.cancel()
        result = await asyncio.gather(pending, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        assert recording.task.done() and recording.closed
        assert not recording.process.is_alive()
        assert recording.manager.recorders._value == 2
        assert recording.failure == "ARCHIVE_CANCELLED"
        await recording.manager.close()

    asyncio.run(run())
