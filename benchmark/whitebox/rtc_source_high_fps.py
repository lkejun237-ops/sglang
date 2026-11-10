"""Generate small local RTC fixtures from real 720p high-rate software encoders.

No GPU, service, cloud or native listener is started. Output stays outside Git.
The Go TCP test consumes each generated JSON via WHITEBOX_SOURCE_CODEC_FIXTURE.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import av

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.playback import (
    create_source_playback_encoder,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig
from sglang.multimodal_gen.runtime.realtime.whitebox.session_encoding import (
    default_source_encoding,
)


def generate(fps):
    units, evidence = {}, []
    profile = None
    for codec in ("h264", "vp9", "av1"):
        settings = default_source_encoding(
            WhiteboxConfig(engine_factory=None, ui_factory=None, spool=None)
        )
        if codec != "h264":
            settings.update(
                codec=codec,
                preset="6" if codec == "vp9" else "10",
                quality=56,
                vbvMs=250 if codec == "vp9" else 100,
            )
        encoder = create_source_playback_encoder(
            VideoSpec(1280, 720),
            settings=settings,
            capture_clock=True,
            capture_fps=fps,
        )
        try:
            packets = []
            for i in range(3):
                packets.extend(
                    encoder.push(i, bytes(1280 * 720 * 3), i * 1_000_000 // fps)
                )
            packets.extend(encoder.drain())
            decoder = av.CodecContext.create(
                "libdav1d" if codec == "av1" else codec, "r"
            )
            decoder.thread_count = 1
            decoded = []
            for _, packet in packets:
                decoded.extend(decoder.decode(av.Packet(packet["payload"])))
            decoded.extend(decoder.decode(None))
            assert len(decoded) == 3
            assert all((f.width, f.height) == (1280, 720) for f in decoded)
            payload = packets[0][1]["payload"]
            units[codec] = base64.b64encode(payload).decode()
            if codec == "h264":
                profile = encoder.stream["codec"][5:]
                assert int(profile[-2:], 16) > 31
            evidence.append(
                dict(
                    codec=codec,
                    fps=fps,
                    decoded_frames=len(decoded),
                    stream=encoder.stream,
                    settings=settings,
                    first_au_sha256=hashlib.sha256(payload).hexdigest(),
                )
            )
        finally:
            encoder.close()
    return dict(accessUnits=units, h264Profile=profile), evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for fps in (36, 60):
        fixture, evidence = generate(fps)
        (args.output_dir / f"source-720p-{fps}.json").write_text(json.dumps(fixture))
        reports.extend(evidence)
    report = dict(pyav=av.__version__, libraries=av.library_versions, cases=reports)
    (args.output_dir / "encode-decode.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(cases=len(reports), output_dir=str(args.output_dir))))


if __name__ == "__main__":
    main()
