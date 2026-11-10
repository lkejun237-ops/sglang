"""Real elementary-AU codec compatibility, VFR tail and recovery gates."""

import av
import numpy as np
import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.playback import (
    H264PlaybackEncoder,
    create_source_playback_encoder,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.source_encoding import (
    default_preset,
    default_quality,
    encoder_implementation,
)


@pytest.mark.parametrize("width,height", [(832, 480), (1280, 720)])
@pytest.mark.parametrize(
    "codec,min_qp",
    [("h264", 0), ("vp9", 0), ("vp9", 48), ("av1", 0), ("av1", 32)],
)
def test_source_codecs_are_elementary_decodable_and_recover_after_forced_key(
    codec, min_qp, width, height
):
    if encoder_implementation(codec) not in av.codecs_available:
        pytest.skip("Configured software codec is unavailable")
    spec = VideoSpec(width, height)
    encoder = create_source_playback_encoder(
        spec,
        settings=dict(
            codec=codec,
            bitrate=1000000,
            gop=6,
            vbvMs=500,
            preset=default_preset(codec),
            quality=default_quality(codec),
            minQp=min_qp,
            intraRefresh=False,
        ),
        capture_clock=True,
        capture_fps=30,
    )
    model = H264PlaybackEncoder(spec)
    assert model.stream["format"] == "h264"
    model.close()
    decoder = av.CodecContext.create(
        "libdav1d" if codec == "av1" and "libdav1d" in av.codecs_available else codec,
        "r",
    )
    fresh = av.CodecContext.create(
        "libdav1d" if codec == "av1" and "libdav1d" in av.codecs_available else codec,
        "r",
    )
    decoder.thread_count = fresh.thread_count = 1
    outputs = []
    frames = []
    for i in range(14):
        if i == 5:
            encoder.set_target_bitrate(700000)
        if i == 9:
            encoder.request_keyframe()
        rgb = np.full(
            (height, width, 3), [10 + i * 5, 100, 180], dtype=np.uint8
        ).tobytes()
        outputs.extend(encoder.push(i, rgb, i * 33333))
    outputs.extend(encoder.drain())
    assert [i for i, _ in outputs] == list(range(14))
    assert encoder.context_bitrate == 700000
    assert encoder.encoding_settings["vbvMs"] == 500
    assert encoder.stream["encoder"] == encoder_implementation(codec)
    assert encoder.stream["format"] == codec
    if codec != "h264":
        assert encoder.stream["encoding_policy"] == "source-cbr-low-delay-v2"
        assert encoder.stream["quality_semantics"] == "cbr-max-qp"
        assert encoder.stream["qmax"] == default_quality(codec)
        assert encoder.stream["rate_control_target_bitrate"] == 700000
        assert encoder.stream["qmin"] == min_qp
        assert encoder.encoding_settings["minQp"] == min_qp
        if codec == "vp9":
            assert encoder.stream["min_bitrate"] == 700000
    assert all(
        m["encoding"] == codec and not m["payload"].startswith(b"\x1aE\xdf\xa3")
        for _, m in outputs
    )
    forced = next(m for i, m in outputs if i == 9)
    assert forced["keyframe"]
    assert len(fresh.decode(av.Packet(forced["payload"]))) == 1
    for _, m in outputs:
        frames.extend(decoder.decode(av.Packet(m["payload"])))
    frames.extend(decoder.decode(None))
    assert len(frames) == 14
    assert all((f.width, f.height) == (width, height) for f in frames)
    encoder.close()


@pytest.mark.parametrize("codec", ["h264", "vp9", "av1"])
def test_real_source_au_crosses_render_envelope_and_rtc_handoff(codec):
    """Exercise the actual source/frame publishing path before native RTP IPC.

    A browser rejecting session.ready closes before this handoff. This gate
    distinguishes that control failure from an encoder/envelope codec rejection.
    """
    import asyncio
    import time
    from types import SimpleNamespace

    import msgspec

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import EngineFrame
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        InputReceiptWire,
        RoomPipeline,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_transport import (
        AUQueue,
        RTCSubscription,
    )

    if encoder_implementation(codec) not in av.codecs_available:
        pytest.skip("Configured software codec is unavailable")

    async def run():
        spec = VideoSpec(832, 480)
        encoder = create_source_playback_encoder(
            spec,
            settings=dict(
                codec=codec,
                bitrate=1000000,
                gop=6,
                vbvMs=250,
                preset=default_preset(codec),
                quality=default_quality(codec),
                intraRefresh=False,
            ),
            capture_clock=True,
            capture_fps=30,
        )
        source = object.__new__(SourceSession)
        source.config = SimpleNamespace(spec=spec)
        source.recording = SimpleNamespace(raw_archive=True)
        source.binding_ended = False
        source.session_id, source.source_epoch = "source-test", "source-epoch"
        source.engine = SimpleNamespace(metadata={}, stream_id="view-test")
        source.outgoing = asyncio.Queue()
        source.first_media = asyncio.Event()
        sub = object.__new__(RTCSubscription)
        sub.closed, sub.model_clock = False, None
        sub.stream_ids = {"whitebox": source.source_epoch, "model": None}
        sub.media = {
            name: AUQueue(encoder.request_keyframe, clock=lambda: 0)
            for name in ("whitebox", "model")
        }
        sub.last_offered = {name: -1 for name in sub.media}
        sub.send_service = {name: 0 for name in sub.media}
        sub.handoffs, sub.wake = {}, asyncio.Event()
        pipeline = object.__new__(RoomPipeline)
        pipeline.member_source = None
        pipeline.subscription = sub
        pipeline.receipt_wire = InputReceiptWire({})
        pipeline.first_media_at, pipeline.session = None, source
        decoder = av.CodecContext.create(
            "libdav1d"
            if codec == "av1" and "libdav1d" in av.codecs_available
            else codec,
            "r",
        )
        decoder.thread_count = 1
        frames, delivered = {}, []

        async def publish(values):
            for index, encoded in values:
                frame = frames[index]
                await source._media(
                    "whitebox", index, frame, encoded, time.perf_counter()
                )
                message = msgspec.msgpack.decode(await source.outgoing.get())
                pipeline.publish(message)
                handoff = sub.media["whitebox"].take()
                assert handoff is not None
                assert handoff["encoding"] == codec
                assert handoff["codec"] == encoder.stream["codec"]
                assert handoff["bitstream_format"] == encoder.stream["bitstream_format"]
                assert handoff["media_stream_id"] == source.source_epoch
                assert handoff["output_frame_start"] == index
                assert handoff["pts_us"] == frame.pts_us
                delivered.extend(decoder.decode(av.Packet(handoff["payload"])))

        try:
            for index in range(4):
                frame = EngineFrame(
                    index, index * 33333, bytes([index * 30]) * spec.frame_bytes, {}
                )
                frames[index] = frame
                await publish(encoder.push(index, frame.rgb, frame.pts_us))
            await publish(encoder.drain())
            delivered.extend(decoder.decode(None))
            assert len(delivered) == 4
            assert source.first_media.is_set()
            assert sub.last_offered["whitebox"] == 3
            assert sub.media["whitebox"].offered == 4
            assert sub.media["whitebox"].taken == 4
        finally:
            encoder.close()

    asyncio.run(run())


@pytest.mark.parametrize("codec", ["vp9", "av1"])
def test_non_h264_uses_source_cadence_and_real_vfr_durations(codec, monkeypatch):
    """A legacy H.264 rate hint must not become a CBR source-frame budget."""
    if encoder_implementation(codec) not in av.codecs_available:
        pytest.skip("Configured software codec is unavailable")
    codec_class = av.CodecContext
    submitted = []

    class RecordingContext:
        def __init__(self, context):
            object.__setattr__(self, "context", context)

        def __getattr__(self, name):
            return getattr(self.context, name)

        def __setattr__(self, name, value):
            setattr(self.context, name, value)

        def encode(self, frame):
            if frame is not None:
                submitted.append((frame.pts, frame.duration))
            return self.context.encode(frame)

    class Factory:
        @staticmethod
        def create(name, mode):
            return RecordingContext(codec_class.create(name, mode))

    monkeypatch.setattr(av, "CodecContext", Factory)
    spec = VideoSpec(832, 480, 24)
    encoder = H264PlaybackEncoder(spec, codec=codec, capture_clock=True, capture_fps=30)
    values = []
    times = [0, 45000, 96000, 136000]
    try:
        assert encoder.context.framerate == 30
        for i, pts in enumerate(times):
            values.extend(
                encoder.push(
                    i,
                    bytes([50 + i]) * spec.frame_bytes,
                    pts,
                    **({"duration_hint_us": 47000} if i == 2 else {}),
                )
            )
            assert not encoder.pending_timing
        values.extend(encoder.drain())
        assert submitted == [
            (0, 33333),
            (45000, 45000),
            (96000, 47000),
            (136000, 40000),
        ]
        assert [i for i, _ in values] == list(range(4))
        assert encoder.stream["max_encoding_fps"] == 30
        assert encoder.stream["fps"] == 30 and spec.fps == 24
        assert encoder.context.options == {}
    finally:
        encoder.close()


@pytest.mark.parametrize("codec", ["vp9", "av1"])
def test_source_capture_hint_follows_cli_descriptor_without_changing_model_clock(
    codec, monkeypatch
):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxUpstream

    session = object.__new__(WhiteboxUpstream)
    session.config = SimpleNamespace(spec=VideoSpec(832, 480, 24))
    session.engine = SimpleNamespace(metadata={"capture_fps": 30})
    session.rtc_level31 = True
    assert session.capture_rate_hint(codec) == 30
    assert session.capture_rate_hint() == 60
    assert session.config.spec.fps == 24
    # The same resolver used to launch CLI owns fallback env/request precedence.
    session.engine.metadata = {}
    monkeypatch.setenv("WHITEBOX_CLI_FPS", "30")
    assert session.capture_rate_hint(codec) == 30
    session.requested_capture_fps = 25
    assert session.capture_rate_hint(codec) == 25
    session.engine.metadata = {"capture_fps": 30}
    assert session.capture_rate_hint(codec) == 30  # Actual descriptor is authoritative.
