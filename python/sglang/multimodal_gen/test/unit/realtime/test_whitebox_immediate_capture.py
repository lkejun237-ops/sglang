"""Immediate H.264 output, source-clock rate control and archive timestamps."""

import io
import os

import av
import numpy as np
import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.playback import H264PlaybackEncoder
from sglang.multimodal_gen.runtime.realtime.whitebox.recording import H264MuxFile


@pytest.fixture(autouse=True)
def native_available():
    from sglang.multimodal_gen.runtime.realtime.whitebox.native_capture import library

    try:
        library()
    except RuntimeError:
        if os.environ.get("WHITEBOX_REQUIRE_NATIVE_X264") == "1":
            raise
        pytest.skip("Build the native x264 bridge to run immediate-capture tests")


def test_current_frame_returns_immediately_with_inter_prediction_and_no_tail_loss():
    spec = VideoSpec(64, 64)
    encoder = H264PlaybackEncoder(
        spec, capture_clock=True, immediate_capture=True, gop=12, max_bitrate=4_000_000
    )
    decoder = av.CodecContext.create("h264", "r")
    keyframes = []
    try:
        for i in range(80):
            if i == 17:
                encoder.request_keyframe()
            rgb = np.full((64, 64, 3), (i * 2) % 255, np.uint8)
            packets = encoder.push(i, rgb.tobytes(), 2_000_000 + i * 33333)
            assert [index for index, _ in packets] == [i]
            packet = packets[0][1]
            frames = decoder.decode(av.Packet(packet["payload"]))
            assert len(frames) == 1 and (frames[0].width, frames[0].height) == (64, 64)
            assert packet["duration_provisional"]
            keyframes.append(packet["keyframe"])
            assert not encoder.pending_timing
        assert keyframes[0] and keyframes[17] and sum(keyframes) < 12
        assert encoder.stream["encoder_retained_frames"] == 0
        assert encoder.drain() == [] and encoder.drain() == []
        assert encoder.output_count == encoder.count == 80
        with pytest.raises(ValueError):
            encoder.push(80, rgb.tobytes(), 5_000_000)
    finally:
        encoder.close()
        encoder.close()


@pytest.mark.parametrize("fps", [30, 45, 60])
@pytest.mark.parametrize("known_duration", [False, True])
def test_vbv_budget_uses_source_elapsed_time_instead_of_the_fixed_codec_tick(
    fps, known_duration
):
    spec = VideoSpec(832, 480)
    encoder = H264PlaybackEncoder(
        spec, capture_clock=True, immediate_capture=True, gop=12, max_bitrate=4_000_000
    )
    rng = np.random.default_rng(57)
    rows = []
    pts = 0
    try:
        for i in range(120):
            if i:
                pts += round(1e6 / fps) * (5 if i % 17 == 0 else 1)
            rgb = rng.integers(0, 256, (480, 832, 3), np.uint8)
            hint = (
                round(1e6 / fps) * (5 if (i + 1) % 17 == 0 else 1)
                if known_duration
                else None
            )
            [(index, packet)] = encoder.push(
                i, rgb.tobytes(), pts, duration_hint_us=hint
            )
            assert index == i
            rows.append((pts, len(packet["payload"]) * 8))
        # Standard leaky-bucket allowance includes the existing 250ms VBV.
        # Check every interval, not just a low average over the entire clip.
        for begin in range(len(rows)):
            bits = 0
            for end in range(begin, len(rows)):
                bits += rows[end][1]
                elapsed = (rows[end][0] - rows[begin][0]) / 1e6
                assert bits <= 4_000_000 * elapsed + 1_000_000
        assert sum(bits for _, bits in rows) > 0.7 * 4_000_000 * pts / 1e6
        assert encoder.drain() == []
    finally:
        encoder.close()


class ForwardSink(io.RawIOBase):
    def __init__(self):
        self.data = bytearray()

    def writable(self):
        return True

    def seekable(self):
        return False

    def tell(self):
        return len(self.data)

    def write(self, data):
        self.data.extend(data)
        return len(data)


def test_archive_corrects_provisional_durations_without_delaying_live_output():
    spec = VideoSpec(64, 64)
    encoder = H264PlaybackEncoder(
        spec, capture_clock=True, immediate_capture=True, gop=6, max_bitrate=4_000_000
    )
    sink = ForwardSink()
    mux = H264MuxFile(sink, spec, streaming=True)
    pts = [2_000_000]
    for i in range(1, 60):
        pts.append(pts[-1] + (133333 if i % 8 == 0 else 33333))
    expected_pixels = []
    decoder = av.CodecContext.create("h264", "r")
    try:
        for i, timestamp in enumerate(pts):
            rgb = np.full((64, 64, 3), i * 3, np.uint8)
            [(index, packet)] = encoder.push(i, rgb.tobytes(), timestamp)
            expected_pixels.append(
                decoder.decode(av.Packet(packet["payload"]))[0].to_ndarray(
                    format="rgb24"
                )
            )
            mux.append(index, packet, timestamp)
        assert encoder.drain() == []
        mux.close()
        mux.close()
        with av.open(io.BytesIO(sink.data), format="mp4") as video:
            frames = list(video.decode(video=0))
        actual_pts = [
            round(float(frame.pts * frame.time_base) * 1e6) for frame in frames
        ]
        actual_durations = [
            round(float(frame.duration * frame.time_base) * 1e6) for frame in frames
        ]
        assert actual_pts == pts
        assert actual_durations[:-1] == [b - a for a, b in zip(pts, pts[1:])]
        assert actual_durations[-1] == pts[-1] - pts[-2]
        assert all(
            np.array_equal(a, b.to_ndarray(format="rgb24"))
            for a, b in zip(expected_pixels, frames)
        )
    finally:
        mux.close()
        encoder.close()


def test_quality_does_not_trade_away_the_removed_output_wait():
    from PIL import Image

    spec = VideoSpec(832, 480)
    rng = np.random.default_rng(29)
    small = rng.integers(0, 256, (48, 84, 3), np.uint8)
    texture = np.asarray(
        Image.fromarray(small).resize((832, 480), Image.Resampling.BILINEAR)
    )
    codecs = [
        H264PlaybackEncoder(
            spec,
            capture_clock=True,
            immediate_capture=mode,
            gop=6,
            max_bitrate=4_000_000,
        )
        for mode in [False, True]
    ]
    decoders = [av.CodecContext.create("h264", "r") for _ in codecs]
    originals, scores = {}, [[], []]
    pts = 0

    def accept(which, packets):
        for index, packet in packets:
            pixels = (
                decoders[which]
                .decode(av.Packet(packet["payload"]))[0]
                .to_ndarray(format="rgb24")
            )
            error = np.mean(
                (pixels.astype(np.float32) - originals[index].astype(np.float32)) ** 2
            )
            scores[which].append(float(10 * np.log10(255**2 / max(error, 1e-9))))
            if which == 0:
                del originals[index]

    try:
        for i in range(90):
            if i:
                pts += 133333 if i % 17 == 0 else 33333
            pixels = np.roll(texture, i * 3, axis=1).copy()
            pixels[140:260, (i * 3) % 680 : (i * 3) % 680 + 120] = [230, 30, 60]
            originals[i] = pixels
            hint = 133333 if (i + 1) % 17 == 0 else 33333
            accept(1, codecs[1].push(i, pixels.tobytes(), pts, duration_hint_us=hint))
            accept(0, codecs[0].push(i, pixels.tobytes(), pts))
        accept(0, codecs[0].drain())
        assert codecs[1].drain() == []
        differences = np.array(scores[1]) - np.array(scores[0])
        assert len(differences) == 90
        assert np.mean(differences) >= -0.2
        assert np.percentile(differences, 5) >= -0.5
    finally:
        for codec in codecs:
            codec.close()


def test_four_model_segments_keep_working_with_immediate_codecs(tmp_path, monkeypatch):
    import test_whitebox_render_lifecycle as fixture

    factory = fixture.WhiteboxConfig

    def immediate_config(**kwargs):
        return factory(**kwargs, immediate_capture=True, playback_gop=6)

    monkeypatch.setattr(fixture, "WhiteboxConfig", immediate_config)
    fixture.test_source_survives_four_segments_and_ordered_prompt_updates(
        tmp_path, "valid-source-frames-v1", True, "adaptive"
    )
