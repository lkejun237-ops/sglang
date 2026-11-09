#!/usr/bin/env python3
"""Offline, identical-frame H.264 comparison. No GPU/network/store writes."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import av
import numpy as np

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.playback import H264PlaybackEncoder


def compare(path, limit, bitrate):
    with av.open(str(path)) as source:
        frames = []
        for f in source.decode(video=0):
            frames.append(f.to_ndarray(format="rgb24"))
            if len(frames) >= limit:
                break
    if len(frames) < 24:
        raise ValueError("At least 24 frames required")
    spec = VideoSpec(frames[0].shape[1], frames[0].shape[0])
    results = []
    for gop in (1, 6, 12):
        codec = H264PlaybackEncoder(spec, max_bitrate=bitrate, gop=gop)
        latencies = []
        encoded = []
        errors = []
        decoder = av.CodecContext.create("h264", "r")
        for i, pixels in enumerate(frames):
            rgb = pixels.tobytes()
            start = time.perf_counter()
            packet = codec.encode(i, rgb)
            latencies.append((time.perf_counter() - start) * 1000)
            encoded.append(packet)
            decoded = decoder.decode(av.Packet(packet["payload"]))
            assert len(decoded) == 1
            error = np.mean(
                (decoded[0].to_ndarray(format="rgb24").astype(np.float32) - pixels) ** 2
            )
            errors.append(float(error))
        codec.finish()
        # Loss of an AU: discard all dependent AUs and resume from an actual IDR.
        recover = av.CodecContext.create("h264", "r")
        waiting = True
        recovered = 0
        gap = 0
        max_gap = 0
        for i, packet in enumerate(encoded):
            if i % 37 == 9:
                waiting = True
                gap = 1
                continue
            if waiting and not packet["keyframe"]:
                gap += 1
                continue
            if packet["keyframe"]:
                waiting = False
                max_gap = max(max_gap, gap)
                gap = 0
            assert len(recover.decode(av.Packet(packet["payload"]))) == 1
            recovered += 1
        sizes = [len(packet["payload"]) for packet in encoded]
        mse = float(np.mean(errors))
        results.append(
            dict(
                gop=gop,
                frames=len(frames),
                keyframes=sum(p["keyframe"] for p in encoded),
                bitrate_at_24fps_bps=sum(sizes) * 8 * 24 / len(frames),
                peak_access_unit_bytes=max(sizes),
                encode_p50_ms=float(np.percentile(latencies, 50)),
                encode_p95_ms=float(np.percentile(latencies, 95)),
                encode_p99_ms=float(np.percentile(latencies, 99)),
                rgb_psnr_db=10 * np.log10(255**2 / mse) if mse else None,
                recovered_frames=recovered,
                max_dependency_recovery_frames=max_gap,
            )
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(
        source_sha256=digest,
        width=spec.width,
        height=spec.height,
        frames=len(frames),
        fps=24,
        max_bitrate=bitrate,
        methodology="Local CPU, re-encode identical decoded source frames; PSNR is relative to this source, not original world truth. Recovery frames are synthetic packet-loss recovery, not online RTT.",
        results=results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--bitrate", type=int, default=1500000)
    args = parser.parse_args()
    result = compare(args.video, args.frames, args.bitrate)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
