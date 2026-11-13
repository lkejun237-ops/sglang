"""Full visual-review artifacts preserve every frame and remove only their spool."""

import shutil
import subprocess

import pytest
from benchmark_artifacts import MediaCapture


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="Requires FFmpeg")
def test_full_video_preserves_frames_and_cleans_spool(tmp_path):
    capture = MediaCapture(tmp_path, 16, 16, 9, mode="full")
    payload = b"".join(bytes([i * 20, 30, 80]) * (16 * 16) for i in range(9))
    capture.write(payload[: 16 * 16 * 3 * 4])
    capture.write(payload[16 * 16 * 3 * 4 :])
    receipt = capture.finish()
    assert receipt["output_frames"] == 9
    assert not capture.spool_path.exists()
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "info",
            "-i",
            str(capture.path),
            "-vf",
            "showinfo",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    assert len(decoded.stdout) == 9 * 16 * 16 * 3
    assert b"s:16x16" in decoded.stderr
    assert b"frame_rate: 24/1" in decoded.stderr
