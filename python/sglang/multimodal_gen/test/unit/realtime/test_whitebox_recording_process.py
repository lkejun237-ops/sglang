# SPDX-License-Identifier: Apache-2.0
"""Real CPU codecs/processes, local temporary files; no cloud writes."""

import asyncio
import json
import os
import signal
import time
from fractions import Fraction

import av
import numpy as np
import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import (
    EngineFrame,
    VideoSpec,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.playback import H264PlaybackEncoder
from sglang.multimodal_gen.runtime.realtime.whitebox.recording_process import (
    ProcessRecording,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.telemetry import (
    CHUNK_FIELDS,
    h3_host_durations,
    numeric_durations,
)


def arguments(path):
    return dict(
        directory=path,
        spec=VideoSpec(32, 32),
        world_id="world",
        trace_id="process-test",
        inference={},
        ui={},
        references=[],
        object_prefix="test/local",
    )


def test_three_files_align_and_ui_pixels_match_exact_playback(tmp_path):
    async def run():
        record = await ProcessRecording.create(
            **arguments(tmp_path / "record"), queue_depth=3
        )
        assert record.worker_pid != os.getpid()
        encoder = H264PlaybackEncoder(record.spec)
        decoder = av.CodecContext.create("h264", "r")
        expected = []
        record.manifest["engine"] = {"test_identity": True}
        await record.configure()
        for i in range(50):  # Cross multiple IDRs; test nonconstant frames.
            pixels = (
                np.random.default_rng(i)
                .integers(0, 256, (32, 32, 3), dtype=np.uint8)
                .tobytes()
            )
            encoded = encoder.encode(i, pixels)
            expected.extend(
                f.to_ndarray(format="rgb24")
                for f in decoder.decode(av.Packet(encoded["payload"]))
            )
            await record.video("whitebox", i, pixels)
            await record.video("model", i, pixels)
            await record.encoded_video(i, encoded)
            states = {"opaque": [i, {"nested": True}]}
            frame = EngineFrame(
                i, record.spec.pts(i), pixels, states, {"held_keys": ["KeyW"]}
            )
            await record.write_metadata(frame, "prompt")
            states["opaque"][0] = -1  # Mutable producer data must be frozen.
            assert record.statistics()["pending_bytes"] <= record.byte_limit
        expected.extend(f.to_ndarray(format="rgb24") for f in decoder.decode(None))
        encoder.finish()
        await record.finish("completed")
        manifest = json.loads((record.directory / "manifest.json").read_text())
        assert manifest["recording_status"] == "complete"
        assert manifest["engine"] == {"test_identity": True}
        assert manifest["videos"]["ui_rendered"]["encoder"]["reused_playback_bitstream"]
        for name in ("whitebox", "model", "ui_rendered"):
            with av.open(str(record.directory / (name + ".mp4"))) as video:
                frames = list(video.decode(video=0))
            assert [f.pts * f.time_base for f in frames] == [
                Fraction(i, 24) for i in range(50)
            ]
            if name == "ui_rendered":
                assert all(
                    np.array_equal(f.to_ndarray(format="rgb24"), e)
                    for f, e in zip(frames, expected, strict=True)
                )
        rows = [
            json.loads(line)
            for line in (record.directory / "states.jsonl").read_text().splitlines()
        ]
        assert [r["states"]["opaque"][0] for r in rows] == list(range(50))

    asyncio.run(run())


def test_slow_worker_bounds_queue_and_exposes_real_backpressure(tmp_path):
    async def run():
        record = await ProcessRecording.create(
            **arguments(tmp_path / "slow"), queue_depth=1
        )
        busy = asyncio.create_task(record._call(time.sleep, 0.3))
        await asyncio.sleep(0.02)
        payload = bytes(record.spec.frame_bytes)
        await record.video("whitebox", 0, payload)
        await asyncio.sleep(0.01)
        await record.video("whitebox", 1, payload)
        blocked = asyncio.create_task(record.video("whitebox", 2, payload))
        await asyncio.sleep(0.03)
        assert not blocked.done()
        assert record.statistics()["pending_operations"] == 2
        assert record.statistics()["oldest_wait_ms"] >= 30
        await busy
        await asyncio.wait_for(blocked, 5)
        await record.finish("aborted")
        assert record.manifest["recording_status"] == "partial"

    asyncio.run(run())


def test_worker_crash_releases_capacity_without_deadlocking_or_claiming_success(
    tmp_path,
):
    async def run():
        released = []
        record = await ProcessRecording.create(
            **arguments(tmp_path / "crash"), release=lambda: released.append(True)
        )
        os.kill(record.worker_pid, signal.SIGTERM)
        await record.video("whitebox", 0, bytes(record.spec.frame_bytes))
        with pytest.raises(Exception):
            await asyncio.wait_for(record.finish("completed"), 10)
        assert released == [True] and record.closed
        manifest = json.loads((record.directory / "manifest.json").read_text())
        assert manifest["recording_status"] == "failed"
        assert manifest["timeline"]["alignment"] == "unverified"

    asyncio.run(run())


def test_h3_telemetry_is_finite_allowlisted_and_distinguishes_host_time():
    assert h3_host_durations(
        {
            "zing_source_stage_host_ms": {
                "dit_host_ms": 12.34,
                "qwen_init_host_ms": 0,
                "prompt": "secret",
                "encoder_host_ms": float("nan"),
                "decoder_host_ms": True,
            }
        }
    ) == {"h3_dit_host_ms": 12.34, "h3_qwen_init_host_ms": 0}
    assert numeric_durations(
        {"scheduler_forward_ms": 12, "prompt_ms": 99, "chunk_total_ms": -1},
        CHUNK_FIELDS,
    ) == {"scheduler_forward_ms": 12}


def test_upload_uses_separate_process_and_publishes_manifest_last(
    tmp_path, monkeypatch
):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from sglang.multimodal_gen.runtime.realtime.whitebox.recording import (
        ArchiveManager,
        S3ArchiveStore,
    )

    # Endpoint and dummy credentials are confined to a local HTTP fixture.
    for name, value in {
        "AWS_ACCESS_KEY_ID": "local",
        "AWS_SECRET_ACCESS_KEY": "local",
        "AWS_EC2_METADATA_DISABLED": "true",
        "NO_PROXY": "*",
    }.items():
        monkeypatch.setenv(name, value)
    uploads = []

    class Handler(BaseHTTPRequestHandler):
        def do_PUT(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            uploads.append((self.path, body))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def run():
        store = S3ArchiveStore(
            "local-test", f"http://127.0.0.1:{server.server_port}", "us-east-1"
        )
        manager = ArchiveManager(store, processes=True)
        record = await manager.create_recording(**arguments(tmp_path / "upload"))
        worker_pid = record.worker_pid
        encoder = H264PlaybackEncoder(record.spec)
        rgb = bytes(record.spec.frame_bytes)
        await record.video("whitebox", 0, rgb)
        await record.video("model", 0, rgb)
        await record.encoded_video(0, encoder.encode(0, rgb))
        await record.write_metadata(EngineFrame(0, 0, rgb, {}, {}), "p")
        encoder.finish()
        await record.finish("completed")
        from concurrent.futures.process import BrokenProcessPool

        from sglang.multimodal_gen.runtime.realtime.whitebox.recording_process import (
            process_pool,
        )

        manager.upload_pool = process_pool(manager.upload_cpu)
        dead_pid = await asyncio.get_running_loop().run_in_executor(
            manager.upload_pool, os.getpid
        )
        os.kill(dead_pid, signal.SIGTERM)
        with pytest.raises(BrokenProcessPool):
            await manager._publish(record.directory)
        assert manager.upload_pool is None and manager.upload_pending == 0
        await manager._publish(record.directory)
        upload_pid = await asyncio.get_running_loop().run_in_executor(
            manager.upload_pool, os.getpid
        )
        assert upload_pid not in (os.getpid(), worker_pid)
        await manager.close()
        assert manager.upload_pending == 0
        assert uploads[-1][0].endswith("/manifest.json")
        assert json.loads(uploads[-1][1])["upload_status"] == "complete"
        assert len(uploads) == 6

    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
