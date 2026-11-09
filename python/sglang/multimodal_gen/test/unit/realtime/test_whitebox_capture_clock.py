"""Real-codec gates for rate control, identity and the retained VFR tail."""

import hashlib
import re

import av
import numpy as np
import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.playback import H264PlaybackEncoder
from sglang.multimodal_gen.runtime.realtime.whitebox.recording import H264MuxFile


def _nal_types(payload):
    return [unit[0] & 31 for unit in re.split(b"\x00\x00\x00?\x01", payload) if unit]


def test_intra_refresh_default_preserves_baseline_bytes_and_rejects_other_paths():
    spec = VideoSpec(64, 64)
    default = H264PlaybackEncoder(spec, capture_clock=True, gop=6)
    explicit = H264PlaybackEncoder(spec, capture_clock=True, gop=6, intra_refresh=False)
    try:
        assert default.stream == explicit.stream
        assert "intra_refresh" not in default.stream
        left, right = [], []
        for index in range(16):
            rgb = bytes([index * 7]) * spec.frame_bytes
            left.extend(default.push(index, rgb, index * 41667))
            right.extend(explicit.push(index, rgb, index * 41667))
        left.extend(default.drain())
        right.extend(explicit.drain())
        assert left == right
    finally:
        default.close()
        explicit.close()
    for options in (
        {},
        {"capture_clock": True, "immediate_capture": True},
        {"capture_clock": True, "intra_refresh": 1},
    ):
        options = {"intra_refresh": True, **options}
        with pytest.raises(ValueError, match="paired capture-clock"):
            H264PlaybackEncoder(spec, **options)


def test_real_intra_refresh_recovery_points_do_not_reopen_rtc_reference_chain():
    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_transport import AUQueue

    spec = VideoSpec(64, 64)
    encoder = H264PlaybackEncoder(spec, capture_clock=True, gop=6, intra_refresh=True)
    queue = AUQueue(encoder.request_keyframe, clock=lambda: 0.0)
    reference = av.CodecContext.create("h264", "r")
    recovered = av.CodecContext.create("h264", "r")
    reference_hashes, resumed, outputs = {}, [], []
    rng = np.random.default_rng(155)
    rgb = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8).tobytes()

    def emit(values, live=False):
        for index, packet in values:
            outputs.append((index, packet))
            assert packet["keyframe"] == (5 in _nal_types(packet["payload"]))
            frame = reference.decode(av.Packet(packet["payload"]))
            assert len(frame) == 1
            reference_hashes[index] = hashlib.sha256(
                frame[0].to_ndarray(format="rgb24").tobytes()
            ).hexdigest()
            if live:
                queue.offer({**packet, "output_frame_start": index})
                message = queue.take()
                if message is not None:
                    frames = recovered.decode(av.Packet(message["payload"]))
                    assert len(frames) == 1
                    assert (
                        hashlib.sha256(
                            frames[0].to_ndarray(format="rgb24").tobytes()
                        ).hexdigest()
                        == reference_hashes[index]
                    )
                    resumed.append(index)

    try:
        for index in range(48):
            emit(encoder.push(index, rgb, index * 41667))
        # Natural cyclic refresh supplies recovery-point SEI, not an IDR.
        natural = outputs[6][1]
        assert any(
            unit and unit[0] & 31 == 6 and len(unit) > 1 and unit[1] == 6
            for unit in re.split(b"\x00\x00\x00?\x01", natural["payload"])
        )
        assert not natural["keyframe"]
        queue.clear("expired")
        queue.offer({**natural, "output_frame_start": 6})
        assert queue.needs_key and queue.take() is None
        assert queue.drops["awaiting_keyframe"] == 1
        # Queue requested a real keyframe. The retained AU may precede it, but
        # only the actual IDR may reopen delivery to a fresh decoder.
        for index in range(48, 56):
            emit(encoder.push(index, rgb, index * 41667), live=True)
        emit(encoder.drain(), live=True)
        assert encoder.stream["intra_refresh"] is True
        assert resumed == list(range(48, 56))
        assert [i for i, packet in outputs if packet["keyframe"]] == [0, 48]
    finally:
        encoder.close()


def test_intra_refresh_survives_live_bitrate_rollover():
    spec = VideoSpec(64, 64)
    encoder = H264PlaybackEncoder(spec, capture_clock=True, gop=6, intra_refresh=True)
    values = []
    try:
        for index in range(30):
            if index == 12:
                encoder.set_target_bitrate(800000)
            values.extend(encoder.push(index, bytes(spec.frame_bytes), index * 41667))
        values.extend(encoder.drain())
        assert encoder.context_bitrate == 800000 and encoder.intra_refresh
        assert [i for i, packet in values if packet["keyframe"]] == [0, 12]
        assert all(
            packet["keyframe"] == (5 in _nal_types(packet["payload"]))
            for _, packet in values
        )
    finally:
        encoder.close()


def test_source_intra_refresh_config_and_environment_are_explicit(
    monkeypatch, tmp_path
):
    import os

    from sglang.multimodal_gen.runtime.realtime.whitebox import session

    base = dict(engine_factory=None, ui_factory=None, spool=tmp_path)
    assert session.WhiteboxConfig(**base).source_intra_refresh is False
    for fields in ({}, {"room_owned": True}, {"capture_clock": True}):
        with pytest.raises(ValueError, match="capture-clock Room"):
            session.WhiteboxConfig(**base, **fields, source_intra_refresh=True)
    for value in (1, "true", None):
        with pytest.raises(ValueError, match="Source intra refresh"):
            session.WhiteboxConfig(
                **base, room_owned=True, capture_clock=True, source_intra_refresh=value
            )
    for key in list(os.environ):
        if key.startswith("WHITEBOX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("WHITEBOX_CAPTURE_CLOCK", "1")
    monkeypatch.setattr(session, "load_plugin", lambda _: lambda **kwargs: None)
    config, _ = session.from_environment(room=True)
    assert config.source_intra_refresh is False
    monkeypatch.setenv("WHITEBOX_SOURCE_INTRA_REFRESH", "1")
    config, _ = session.from_environment(room=True)
    assert config.source_intra_refresh is True
    monkeypatch.setenv("WHITEBOX_SOURCE_INTRA_REFRESH", "true")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        session.from_environment(room=True)


@pytest.mark.parametrize("fps", [24, 35, 48])
def test_capture_clock_caps_real_rate_and_remuxes_every_frame(tmp_path, fps):
    spec = VideoSpec(832, 480)
    encoder = H264PlaybackEncoder(
        spec, max_bitrate=4_000_000, gop=6, capture_clock=True
    )
    decoder = av.CodecContext.create("h264", "r")
    mux = H264MuxFile(tmp_path / "capture.mp4", spec)
    rng = np.random.default_rng(42)
    count = fps * 4
    sizes, hashes, indices = [], [], []
    origin = 91_123_456

    def emit(outputs):
        for index, encoded in outputs:
            assert index == len(indices)
            indices.append(index)
            sizes.append(len(encoded["payload"]))
            frames = decoder.decode(av.Packet(encoded["payload"]))
            assert len(frames) == 1
            hashes.append(
                hashlib.sha256(
                    frames[0].to_ndarray(format="rgb24").tobytes()
                ).hexdigest()
            )
            mux.append(index, encoded, pts_us=origin + round(index * 1e6 / fps))

    try:
        for index in range(count):
            rgb = rng.integers(0, 256, (480, 832, 3), dtype=np.uint8).tobytes()
            emit(encoder.push(index, rgb, origin + round(index * 1e6 / fps)))
            assert encoder.count - encoder.output_count <= 1
        assert len(indices) == count - 1
        emit(encoder.drain())
        assert encoder.drain() == []
        assert indices == list(range(count))
        # A 250ms VBV permits a bounded initial burst, not FPS-scaled sustained rate.
        assert sum(sizes[:fps]) * 8 <= 5_100_000
        assert sum(sizes[fps:]) * 8 / 3 <= 4_250_000
        assert encoder.stream["rate_control_clock"] == "source_capture_vfr"
        mux.close()
        with av.open(str(tmp_path / "capture.mp4")) as movie:
            archived = [
                hashlib.sha256(f.to_ndarray(format="rgb24").tobytes()).hexdigest()
                for f in movie.decode(video=0)
            ]
        assert archived == hashes
    finally:
        encoder.close()


def test_capture_clock_empty_single_frame_and_invalid_timestamp(tmp_path):
    spec = VideoSpec(32, 32)
    empty = H264PlaybackEncoder(spec, capture_clock=True)
    assert empty.drain() == []
    empty.close()
    encoder = H264PlaybackEncoder(spec, capture_clock=True)
    try:
        assert encoder.push(0, bytes(spec.frame_bytes), 9_000_000) == []
        with pytest.raises(ValueError, match="timestamps"):
            encoder.push(1, bytes(spec.frame_bytes), 9_000_000)
        output = encoder.drain()
        assert len(output) == 1 and output[0][0] == 0 and output[0][1]["keyframe"]
        assert encoder.count == encoder.output_count == 1
        with pytest.raises(ValueError, match="contiguous"):
            encoder.push(1, bytes(spec.frame_bytes), 10_000_000)
    finally:
        encoder.close()


def test_vfr_admission_requires_a_clip_plus_its_retained_au(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig

    with pytest.raises(ValueError, match="between 18"):
        WhiteboxConfig(
            engine_factory=None,
            ui_factory=None,
            spool=tmp_path,
            room_owned=True,
            capture_clock=True,
            model_input_frames=17,
        )
    assert (
        WhiteboxConfig(
            engine_factory=None,
            ui_factory=None,
            spool=tmp_path,
            room_owned=True,
            capture_clock=True,
            model_input_frames=18,
        ).model_input_frames
        == 18
    )


def test_subclip_window_requires_explicit_overlap_and_separate_output_budget(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig

    base = dict(
        engine_factory=None,
        ui_factory=None,
        spool=tmp_path,
        room_owned=True,
        capture_clock=True,
        model_input_frames=9,
    )
    with pytest.raises(ValueError, match="between 18"):
        WhiteboxConfig(**base, pipeline_overlap=True)
    with pytest.raises(ValueError, match="overlapping"):
        WhiteboxConfig(**base, model_subclip_admission=True)
    config = WhiteboxConfig(**base, pipeline_overlap=True, model_subclip_admission=True)
    assert config.minimum_model_input_frames == 9
    assert config.model_output_reserve_frames == 4
    with pytest.raises(ValueError, match="between 9"):
        WhiteboxConfig(
            **{**base, "model_input_frames": 8},
            pipeline_overlap=True,
            model_subclip_admission=True,
        )


def test_live_expiry_recovers_real_h264_at_idr_without_dropping_archive_frames(
    tmp_path, monkeypatch
):
    import asyncio
    from types import SimpleNamespace

    import msgspec

    from sglang.multimodal_gen.runtime.realtime.whitebox import room_pipeline

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        room_pipeline, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    spec = VideoSpec(64, 64)
    encoder = H264PlaybackEncoder(
        spec, max_bitrate=4_000_000, gop=6, capture_clock=True
    )
    source_decoder = av.CodecContext.create("h264", "r")
    live_decoder = av.CodecContext.create("h264", "r")
    mux = H264MuxFile(tmp_path / "complete.mp4", spec)
    sub = room_pipeline.Subscription(
        SimpleNamespace(
            terminal=None, request_keyframe=lambda _: encoder.request_keyframe()
        ),
        1,
    )
    sub.ack(dict(protocol="whitebox-dual-v2", stream="model", frame=-1, window=32))
    hashes, shown = {}, []
    rng = np.random.default_rng(71)

    async def emit(outputs, consume):
        for index, encoded in outputs:
            mux.append(index, encoded, pts_us=round(index * 1e6 / 24))
            decoded = source_decoder.decode(av.Packet(encoded["payload"]))
            assert len(decoded) == 1
            hashes[index] = hashlib.sha256(
                decoded[0].to_ndarray(format="rgb24").tobytes()
            ).hexdigest()
            sub.offer_media(dict(encoded, stream="model", output_frame_start=index))
            if consume:
                while sub.media["model"]:
                    message = msgspec.msgpack.decode(await sub.recv())
                    frames = live_decoder.decode(av.Packet(message["payload"]))
                    assert len(frames) == 1
                    digest = hashlib.sha256(
                        frames[0].to_ndarray(format="rgb24").tobytes()
                    ).hexdigest()
                    assert digest == hashes[message["media_sequence"]]
                    shown.append(message["media_sequence"])
                    sub.ack(
                        dict(
                            protocol="whitebox-dual-v2",
                            stream="model",
                            frame=message["output_frame_start"],
                            window=32,
                        )
                    )

    async def run():
        for index in range(96):
            clock.now = index / 24
            rgb = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8).tobytes()
            await emit(
                encoder.push(index, rgb, round(index * 1e6 / 24)),
                index < 12 or index >= 48,
            )
        await emit(encoder.drain(), True)

    try:
        asyncio.run(run())
        mux.close()
        assert 0 < len(shown) < 96
        assert shown[-1] == 95
        assert sub.delivery["streams"]["model"]["dropped_by_reason"]["queue_age"] > 0
        with av.open(str(tmp_path / "complete.mp4")) as movie:
            archived = [
                hashlib.sha256(frame.to_ndarray(format="rgb24").tobytes()).hexdigest()
                for frame in movie.decode(video=0)
            ]
        assert archived == [hashes[index] for index in range(96)]
    finally:
        sub.stop("DONE")
        encoder.close()
        mux.close()
