"""Decode real independently encoded archives, including asymmetric EOS tails."""

import asyncio
import copy
import io
import threading

import av
import numpy as np
import pytest
from test_whitebox_stream_archive import MemoryS3
from test_whitebox_stream_archive import local_s3 as local_s3
from test_whitebox_stream_archive import prototype

from sglang.multimodal_gen.runtime.realtime.whitebox.archive_encoder import (
    ArchiveVideoEncoder,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.archive_rgb import (
    RGBBudget,
    RGBReader,
    RGBSlots,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import (
    EngineFrame,
    VideoSpec,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig
from sglang.multimodal_gen.runtime.realtime.whitebox.stream_recording import (
    StreamingData,
)

CONFIG = dict(
    source_bitrate=4_000_000,
    paired_source_bitrate=4_000_000,
    model_bitrate=4_000_000,
    vbv_ms=2000,
    keyframe_ms=1000,
    pair_frames=21,
    pair_age=120,
)


@pytest.mark.parametrize("width,height", [(832, 480), (1280, 736)])
def test_archive_bitrate_uses_real_capture_clock_and_decodes_every_frame(width, height):
    spec = VideoSpec(width, height)
    encoder = ArchiveVideoEncoder(spec)
    rng = np.random.default_rng(42)
    output = []
    fps, seconds = 30, 5
    origin = 9_000_000
    try:
        for i in range(fps * seconds):
            rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8).tobytes()
            output.extend(encoder.push(i, rgb, origin + round(i * 1e6 / fps)))
            assert encoder.count - encoder.output_count <= 1
        output.extend(encoder.drain())
        assert encoder.drain() == []
        decoder = av.CodecContext.create("h264", "r")
        assert [i for i, _ in output] == list(range(fps * seconds))
        for i, encoded in output:
            assert encoded["pts_us"] == origin + round(i * 1e6 / fps)
            frames = decoder.decode(av.Packet(encoded["payload"]))
            assert len(frames) == 1
            assert (frames[0].width, frames[0].height) == (width, height)
        assert output[0][1]["keyframe"]
        # A two-second VBV may burst at startup; steady state must not scale
        # a 24 FPS model budget by the actual 30 FPS source capture rate.
        steady = (
            sum(len(x["payload"]) for _, x in output[2 * fps :]) * 8 / (seconds - 2)
        )
        assert 3_500_000 < steady < 4_400_000
        assert encoder.stream["target_bitrate"] == 4_000_000
        assert encoder.stream["rate_control"] == "abr"
        assert encoder.stream["network_adaptive"] is False
        assert not hasattr(encoder, "set_target_bitrate")
    finally:
        encoder.close()


def test_rgb_slots_bound_memory_and_reject_late_receipts():
    budget = RGBBudget(96)
    pool = RGBSlots(12, budget, frames=2, max_bytes=24)
    reader = RGBReader(12)
    try:
        first, second = pool.offer(b"a" * 12), pool.offer(b"b" * 12)
        with pytest.raises(BufferError):
            pool.offer(b"c" * 12)
        assert reader.read(first) == b"a" * 12
        pool.release(first)
        third = pool.offer(b"c" * 12)
        with pytest.raises(ValueError, match="Stale"):
            pool.release(first)
        assert reader.read(second) == b"b" * 12
        assert reader.read(third) == b"c" * 12
        pool.release(second)
        pool.release(third)
        assert budget.used == 24
    finally:
        reader.close()
        pool.close()
    assert budget.used == 0


def test_archive_empty_single_frame_and_invalid_pts():
    spec = VideoSpec(32, 32)
    empty = ArchiveVideoEncoder(spec)
    assert empty.drain() == []
    empty.close()
    codec = ArchiveVideoEncoder(spec)
    try:
        assert codec.push(0, bytes(spec.frame_bytes), 900_000) == []
        with pytest.raises(ValueError, match="timestamps"):
            codec.push(1, bytes(spec.frame_bytes), 900_000)
        output = codec.drain()
        assert len(output) == 1 and output[0][0] == 0 and output[0][1]["keyframe"]
        assert output[0][1]["pts_us"] == 900_000
        with pytest.raises(ValueError, match="contiguous"):
            codec.push(1, bytes(spec.frame_bytes), 1_000_000)
    finally:
        codec.close()


def make_data(tmp_path, source=False, encoding=None):
    kwargs, manifest = asyncio.run(prototype(tmp_path, source=source))
    kwargs["archive_encoding"] = copy.deepcopy(CONFIG if encoding is None else encoding)
    manifest["recording_kind"] = "source" if source else "render_segment"
    manifest["rendering"] = {}
    client = MemoryS3()
    settings = dict(
        upload_bytes=32 << 20,
        archive_bytes=2 << 30,
        part_bytes=65536,
        finish_seconds=10,
        index_key="test/.archive-index/v1/" + "a" * 32 + ".json",
    )
    return StreamingData(client, {"bucket": "b"}, kwargs, manifest, settings), client


def test_archive_accepts_conservative_input_window_and_codec_tail(tmp_path):
    config = WhiteboxConfig(None, None, tmp_path)
    data, client = make_data(tmp_path, encoding=config.archive_encoding)
    spec = data.spec
    pool = RGBSlots(spec.frame_bytes, RGBBudget())
    # Fill the negotiated legacy input window, the output reservations, and
    # the one independent VFR lookahead frame. The previous limit failed here.
    count = config.model_input_window() + config.model_output_reserve_frames + 1
    try:
        for i in range(count):
            descriptor = pool.offer(bytes([70]) * spec.frame_bytes)
            data.operation("selected_source_rgb", (i, spec.pts(i), i * 2, descriptor))
            pool.release(descriptor)
        for i in range(count):
            descriptor = pool.offer(bytes([170]) * spec.frame_bytes)
            data.operation("model_rgb", (frame(i, spec.pts(i), spec), descriptor, "p"))
            pool.release(descriptor)
        manifest = data.finish("completed", data.manifest)
        a, ad = decode(client, manifest["object_prefix"], "whitebox")
        b, bd = decode(client, manifest["object_prefix"], "model")
        assert len(a) == len(b) == count
        assert ad == bd
        assert manifest["recording_status"] == "complete"
    finally:
        data.close()
        pool.close()


def test_archive_still_rejects_excess_conservative_input(tmp_path):
    config = WhiteboxConfig(None, None, tmp_path)
    data, _ = make_data(tmp_path, encoding=config.archive_encoding)
    pool = RGBSlots(data.spec.frame_bytes, RGBBudget())
    try:
        for i in range(config.archive_encoding["pair_frames"] + 1):
            descriptor = pool.offer(bytes(data.spec.frame_bytes))
            try:
                if i == config.archive_encoding["pair_frames"]:
                    with pytest.raises(BufferError, match="frame budget"):
                        data.operation(
                            "selected_source_rgb",
                            (i, data.spec.pts(i), i * 2, descriptor),
                        )
                else:
                    data.operation(
                        "selected_source_rgb", (i, data.spec.pts(i), i * 2, descriptor)
                    )
            finally:
                pool.release(descriptor)
    finally:
        data.close()
        pool.close()


def frame(index, pts, spec):
    return EngineFrame(
        index,
        spec.pts(index),
        b"",
        {"frame": index},
        effective_input={"held_keys": []},
        provenance={"capture_pts_us": pts, "source_frame_index": index * 2},
    )


def decode(client, prefix, name):
    raw = client.objects[("b", prefix + name + ".mp4")][0]
    with av.open(io.BytesIO(raw)) as movie:
        stream = movie.streams.video[0]
        duration = stream.duration * stream.time_base
        frames = list(movie.decode(video=0))
        return [
            (f.pts * f.time_base, f.to_ndarray(format="rgb24").mean()) for f in frames
        ], duration


def test_independent_pair_uses_common_duration_when_selected_tail_has_no_model(
    tmp_path,
):
    data, client = make_data(tmp_path)
    spec, budget = data.spec, RGBBudget()
    source, model = (
        RGBSlots(spec.frame_bytes, budget),
        RGBSlots(spec.frame_bytes, budget),
    )
    try:
        times = [2_000_000, 2_040_000, 2_950_000]
        for i, pts in enumerate(times):
            d = source.offer(bytes([70 + i]) * spec.frame_bytes)
            data.operation("selected_source_rgb", (i, pts, i * 2, d))
            source.release(d)
        for i, pts in enumerate(times[:2]):
            d = model.offer(bytes([170 + i]) * spec.frame_bytes)
            data.operation("model_rgb", (frame(i, pts, spec), d, "condition"))
            model.release(d)
        manifest = data.finish("cancelled", data.manifest)
        a, ad = decode(client, manifest["object_prefix"], "whitebox")
        b, bd = decode(client, manifest["object_prefix"], "model")
        assert len(a) == len(b) == 2
        assert [p for p, _ in a] == [p for p, _ in b]
        assert ad == bd
        assert manifest["timeline"]["end_time_us"] == 2_080_000
        assert all(x < 90 for _, x in a) and all(x > 150 for _, x in b)
        assert manifest["recording_status"] == "partial"
        assert manifest["rendering"]["unpaired_source_frames"] == 1
        for video in manifest["videos"].values():
            assert video["encoder"]["target_bitrate"] == 4_000_000
            assert video["encoder"]["reused_playback_bitstream"] is False
            assert video["encoder"]["encoder_parameters_verified"] is True
            assert video["encoder"]["x264_rate_control"] == "cbr"
            assert "crf" not in video["encoder"]
    finally:
        data.close()
        source.close()
        model.close()
    assert budget.used == 0


def test_full_source_keeps_every_frame_and_absolute_capture_time(tmp_path):
    data, client = make_data(tmp_path, source=True)
    pool = RGBSlots(data.spec.frame_bytes, RGBBudget())
    try:
        times = [8_000_000, 8_030_000, 8_120_000, 9_000_000]
        for i, pts in enumerate(times):
            descriptor = pool.offer(bytes([30 + 20 * i]) * data.spec.frame_bytes)
            data.operation("source_rgb", (frame(i, pts, data.spec), descriptor))
            pool.release(descriptor)
        manifest = data.finish("completed", data.manifest)
        decoded, _ = decode(client, manifest["object_prefix"], "whitebox")
        assert [round(pts * 1_000_000) for pts, _ in decoded] == times
        assert manifest["videos"]["whitebox"]["frames"] == len(times)
        assert manifest["recording_status"] == "complete"
        assert manifest["timeline"]["alignment"] == "source_timeline"
    finally:
        data.close()
        pool.close()


def test_source_identity_mismatch_fails_before_committing_pair(tmp_path):
    data, _ = make_data(tmp_path)
    pool = RGBSlots(data.spec.frame_bytes, RGBBudget())
    try:
        d = pool.offer(bytes(data.spec.frame_bytes))
        data.operation("selected_source_rgb", (0, 1_000_000, 17, d))
        pool.release(d)
        d = pool.offer(bytes(data.spec.frame_bytes))
        with pytest.raises(ValueError, match="identity"):
            data.operation("model_rgb", (frame(0, 1_000_000, data.spec), d, "prompt"))
        pool.release(d)
        assert all(mux.count == 0 for mux in data.encoders.values())
    finally:
        data.close()
        pool.close()


def test_finalization_releases_shared_rgb_before_waiting_for_reference_upload(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    data, client = make_data(tmp_path, source=True)
    budget = RGBBudget()
    pool = RGBSlots(data.spec.frame_bytes, budget)
    waiting, release, rgb_released = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original = client.put_object

    def delayed_reference(**kwargs):
        if "/references/" in kwargs["Key"]:
            waiting.set()
            assert release.wait(5)
        return original(**kwargs)

    monkeypatch.setattr(client, "put_object", delayed_reference)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        descriptor = pool.offer(bytes([70]) * data.spec.frame_bytes)
        data.operation("source_rgb", (frame(0, 500_000, data.spec), descriptor))
        pool.release(descriptor)
        data.operation(
            "reference_prepare",
            (
                "ending-reference",
                {
                    "id": "new",
                    "format": "png",
                    "payload": b"pending-reference",
                },
            ),
        )
        assert waiting.wait(2)
        done = executor.submit(
            data.finish, "completed", data.manifest, rgb_released.set
        )
        assert rgb_released.wait(1)
        assert not done.done()
        pool.close()
        assert budget.used == 0
        release.set()
        manifest = done.result(timeout=5)
        decoded, _ = decode(client, manifest["object_prefix"], "whitebox")
        assert len(decoded) == 1
        assert manifest["recording_status"] == "complete"
    finally:
        release.set()
        executor.shutdown(wait=True)
        data.close()
        pool.close()


def test_full_raw_queue_fails_open_and_next_recording_reclaims_slots(
    tmp_path, local_s3
):
    from dataclasses import replace

    from sglang.multimodal_gen.runtime.realtime.whitebox.recording import S3ArchiveStore
    from sglang.multimodal_gen.runtime.realtime.whitebox.stream_archive import (
        StreamingArchiveManager,
    )

    _, endpoint = local_s3

    async def run():
        manager = StreamingArchiveManager(
            S3ArchiveStore("bucket", endpoint, "ap-southeast-7"),
            index_prefix="test",
            recorder_slots=1,
        )
        spec = VideoSpec(64, 64)

        async def new_record(name):
            r = await manager.create_recording(
                directory=tmp_path / name,
                spec=spec,
                world_id="world",
                trace_id="raw-overflow",
                inference={},
                ui={"mode": "browser", "files": {}},
                references=[],
                object_prefix="test/" + name + "/",
                source_only=True,
                archive_encoding=CONFIG,
            )
            await r.prepare_raw()
            r.manifest.update(recording_kind="source", rendering={})
            return r

        try:
            first = await new_record("1" * 32)
            # No event-loop turn lets the parent consume worker ACKs. Overflow
            # must return false, never block the live source or overwrite RGB.
            results = [
                first.source_rgb(
                    replace(frame(i, i * 40_000, spec), rgb=bytes(spec.frame_bytes))
                )
                for i in range(9)
            ]
            assert results == [True] * 8 + [False]
            assert first.failure == "ARCHIVE_RGB_CAPACITY_EXHAUSTED"
            await asyncio.wait_for(first.task, 5)
            assert manager.rgb_budget.used == 0
            assert not first.process.is_alive()
            second = await new_record("2" * 32)
            assert second.source_rgb(
                replace(frame(0, 0, spec), rgb=bytes(spec.frame_bytes))
            )
            await second.finish("completed")
            assert second.manifest["recording_status"] == "complete", second.manifest
            assert manager.rgb_budget.used == 0
        finally:
            await manager.close()

    asyncio.run(run())


def test_spawned_independent_archive_keeps_recording_during_reference_confirmation(
    tmp_path, local_s3, monkeypatch
):
    from dataclasses import replace

    from sglang.multimodal_gen.runtime.realtime.whitebox.recording import S3ArchiveStore
    from sglang.multimodal_gen.runtime.realtime.whitebox.stream_archive import (
        StreamingArchiveManager,
    )

    fake, endpoint = local_s3
    original_put = fake.put_object
    waiting, release = threading.Event(), threading.Event()

    def delayed_reference(**kwargs):
        if "/references/" in kwargs["Key"]:
            waiting.set()
            if not release.wait(5):
                raise TimeoutError("Test reference confirmation timed out")
        return original_put(**kwargs)

    monkeypatch.setattr(fake, "put_object", delayed_reference)

    async def run():
        manager = StreamingArchiveManager(
            S3ArchiveStore("bucket", endpoint, "ap-southeast-7"),
            index_prefix="test",
            finish_timeout=15,
        )
        spec = VideoSpec(64, 64)
        record = await manager.create_recording(
            directory=tmp_path / ("f" * 32),
            spec=spec,
            world_id="world",
            trace_id="independent-rgb",
            inference={},
            ui={"mode": "browser", "files": {}},
            references=[],
            object_prefix="test/" + "f" * 32 + "/",
            archive_encoding=CONFIG,
        )
        reference_task = None
        try:
            await record.prepare_raw()
            record.manifest.update(
                recording_kind="render_segment", rendering={"model_frames": 20}
            )
            await record.configure()
            reference_task = asyncio.create_task(
                record.add_references(
                    [{"id": "new", "format": "png", "payload": b"new-reference"}],
                    wait=True,
                )
            )
            assert await asyncio.to_thread(waiting.wait, 3)
            for i in range(20):
                f = replace(
                    frame(i * 2, 3_000_000 + i * 40_000, spec),
                    rgb=bytes([60 + i]) * spec.frame_bytes,
                )
                assert record.selected_source_rgb(i, f)
                assert record.model_rgb(
                    i, f, bytes([160 + i]) * spec.frame_bytes, "prompt"
                )
                await asyncio.sleep(0.025)
            async with asyncio.timeout(3):
                while record.last_frame < 18:
                    assert not record.failure, record.failure
                    await asyncio.sleep(0.01)
            assert not reference_task.done()
            release.set()
            descriptors = await asyncio.wait_for(reference_task, 3)
            assert (
                fake.objects[
                    (
                        "bucket",
                        record.manifest["object_prefix"] + descriptors[0]["path"],
                    )
                ][0]
                == b"new-reference"
            )
            await record.finish("completed")
            assert record.manifest["recording_status"] == "complete", record.manifest
            assert record.manifest["upload_status"] == "complete"
            assert record.manifest["videos"]["whitebox"]["frames"] == 20
            assert record.manifest["videos"]["model"]["frames"] == 20
            assert manager.rgb_budget.used == 0
            assert record.closed and not record.process.is_alive()
        finally:
            release.set()
            if reference_task:
                await asyncio.gather(reference_task, return_exceptions=True)
            await manager.close()

    asyncio.run(run())


def test_render_source_and_four_segments_use_independent_archive_workers(
    tmp_path, local_s3, monkeypatch
):
    import test_whitebox_render_lifecycle as lifecycle

    from sglang.multimodal_gen.runtime.realtime.whitebox.recording import S3ArchiveStore
    from sglang.multimodal_gen.runtime.realtime.whitebox.stream_archive import (
        StreamingArchiveManager,
    )

    fake, endpoint = local_s3

    class CreditedEngine(lifecycle.Engine):
        def __init__(self, spec):
            super().__init__(spec)
            self.queue = asyncio.Queue(maxsize=2)
            self.capture_started = asyncio.Event()

        async def frames(self):
            # Native CLI grants its two-frame window only when frames() starts.
            # Do not manufacture an unbounded pre-start backlog while the real
            # archive process is importing its codecs during startup.
            self.capture_started.set()
            async for value in super().frames():
                yield value

        async def push(self):
            await self.capture_started.wait()
            await super().push()

    monkeypatch.setattr(lifecycle, "Engine", CreditedEngine)
    monkeypatch.setattr(
        lifecycle,
        "AsyncArchiveManager",
        lambda **kw: StreamingArchiveManager(
            S3ArchiveStore("bucket", endpoint, "ap-southeast-7"),
            index_prefix="test",
            **kw,
        ),
    )
    lifecycle.test_source_survives_four_segments_and_ordered_prompt_updates(
        tmp_path, "valid-source-frames-v1", True
    )
    import json

    manifests = [
        json.loads(value[0])
        for (_, key), value in fake.objects.items()
        if key.endswith("/manifest.json")
    ]
    source = [m for m in manifests if m.get("recording_kind") == "source"]
    segments = [m for m in manifests if m.get("recording_kind") == "render_segment"]
    assert len(source) == 1 and len(segments) == 4
    for manifest in source + segments:
        assert manifest["archive_encoding_version"] == 2
        assert manifest["upload_status"] == "complete", {
            k: manifest.get(k)
            for k in (
                "recording_kind",
                "recording_id",
                "archive_failure",
                "upload_failure",
                "rendering",
            )
        }
        assert manifest.get("archive_failure") is None, manifest
        assert manifest["videos"], manifest
        for video in manifest["videos"].values():
            assert video["encoder"]["target_bitrate"] == 4_000_000
            assert video["encoder"]["network_adaptive"] is False


def test_prefill_pair_playback_omits_hold_without_rewriting_source_metadata(tmp_path):
    from dataclasses import replace

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import (
        capture_pts,
        presentation_pts,
    )

    data, client = make_data(tmp_path)
    budget = RGBBudget()
    source = RGBSlots(data.spec.frame_bytes, budget)
    model = RGBSlots(data.spec.frame_bytes, budget)
    try:
        real = [1_000_000, 6_033_333, 6_066_666]
        records = []
        for i, pts in enumerate(real):
            value = frame(i, pts, data.spec)
            if i == 0:
                value = replace(
                    value,
                    provenance={
                        **value.provenance,
                        "model_presentation_pts_us": 5_958_334,
                        "prefill_hold_us": 5_000_000,
                    },
                )
            records.append(value)
            d = source.offer(bytes([70 + i]) * data.spec.frame_bytes)
            data.operation(
                "selected_source_rgb", (i, presentation_pts(value), i * 2, d)
            )
            source.release(d)
        for i, value in enumerate(records):
            d = model.offer(bytes([170 + i]) * data.spec.frame_bytes)
            data.operation("model_rgb", (value, d, "condition"))
            model.release(d)
        manifest = data.finish("completed", data.manifest)
        a, ad = decode(client, manifest["object_prefix"], "whitebox")
        b, bd = decode(client, manifest["object_prefix"], "model")
        assert len(a) == len(b) == 3 and ad == bd
        assert [p for p, _ in a] == [p for p, _ in b]
        assert float(a[1][0] - a[0][0]) < 0.1
        assert [capture_pts(x) for x in records] == real
        assert manifest["videos"]["model"]["duration_us"] < 250_000
    finally:
        data.close()
        source.close()
        model.close()


def test_independent_archive_preserves_different_source_and_model_dimensions(tmp_path):
    kwargs, manifest = asyncio.run(prototype(tmp_path))
    kwargs.update(model_spec=VideoSpec(96, 64), archive_encoding=copy.deepcopy(CONFIG))
    manifest.update(recording_kind="render_segment", rendering={})
    client = MemoryS3()
    settings = dict(
        upload_bytes=32 << 20,
        archive_bytes=2 << 30,
        part_bytes=65536,
        finish_seconds=10,
        index_key="test/.archive-index/v1/" + "a" * 32 + ".json",
    )
    data = StreamingData(client, {"bucket": "b"}, kwargs, manifest, settings)
    budget = RGBBudget()
    source = RGBSlots(data.spec.frame_bytes, budget)
    model = RGBSlots(data.model_spec.frame_bytes, budget)
    try:
        for i in range(4):
            pts = 1_000_000 + i * 50_000
            lease = source.offer(bytes([30 + i]) * data.spec.frame_bytes)
            data.operation("selected_source_rgb", (i, pts, i * 2, lease))
            source.release(lease)
            lease = model.offer(bytes([150 + i]) * data.model_spec.frame_bytes)
            data.operation("model_rgb", (frame(i, pts, data.spec), lease, "condition"))
            model.release(lease)
        manifest = data.finish("completed", data.manifest)
        decoded = {}
        for name, expected in [("whitebox", (64, 64)), ("model", (96, 64))]:
            raw = client.objects[("b", manifest["object_prefix"] + name + ".mp4")][0]
            with av.open(io.BytesIO(raw)) as video:
                frames = list(video.decode(video=0))
                assert len(frames) == 4
                assert all((f.width, f.height) == expected for f in frames)
                decoded[name] = [f.pts * f.time_base for f in frames]
            assert (
                manifest["videos"][name]["width"],
                manifest["videos"][name]["height"],
            ) == expected
        assert decoded["whitebox"] == decoded["model"]
        assert manifest["recording_status"] == "complete"
    finally:
        data.close()
        source.close()
        model.close()
    assert budget.used == 0
