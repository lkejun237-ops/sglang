# SPDX-License-Identifier: Apache-2.0
"""Bounded benchmark media and explicit, owned experiment cache teardown."""

import fcntl
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

DEFAULT_RETENTION_BYTES = 2 << 30
TIMING_SCOPE = (
    "client_delivery; artifact_mode_in_media_receipt; video_encode_after_request"
)


def media_timing_scope(mode):
    return {
        "raw": "client_delivery_with_full_rgb_hash_and_rgb_file_backpressure",
        "full": "client_delivery_with_full_rgb_hash_and_rgb_file_backpressure; full_video_encode_after_request",
        "video": "client_delivery_with_full_rgb_hash_and_rgb_file_backpressure; full_video_encode_after_request",
        "metrics": "client_delivery_with_full_rgb_hash; no_media_write",
        "representative": "client_delivery_with_full_rgb_hash_and_bounded_sampling; video_encode_after_request",
    }[mode]


def retained_bytes(root):
    """Bound logical and allocated file bytes; never follow shared asset links."""
    total = 0
    for p in Path(root).rglob("*"):
        if p.is_symlink() or p.is_file():
            stat = p.lstat()
            total += max(stat.st_size, getattr(stat, "st_blocks", 0) * 512)
    return total


def check_retention(root, budget=DEFAULT_RETENTION_BYTES):
    root = Path(root)
    receipt = root / "retention.json"
    # Include the receipt itself in the final budget check.
    result = {"budget_bytes": budget, "retained_bytes": 0, "within_budget": False}
    for _ in range(3):
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        result["retained_bytes"] = retained_bytes(root)
        result["within_budget"] = result["retained_bytes"] <= budget
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    if retained_bytes(root) > budget:
        raise RuntimeError(
            "Experiment exceeds retention budget; inspect retention.json"
        )
    return result


class MediaCapture:
    """Hash every byte; keep only uniformly sampled frames unless raw is explicit.

    Preview is a labelled time-lapse, not a full video or a pixel-parity artifact.
    Sampling stays below 128 MiB RAM; FFmpeg runs only after request completion.
    """

    def __init__(
        self,
        output,
        width,
        height,
        expected_frames,
        mode="representative",
        ffmpeg="ffmpeg",
        *,
        retention_budget_bytes=DEFAULT_RETENTION_BYTES,
    ):
        if mode not in ("metrics", "representative", "raw", "full", "video"):
            raise ValueError("Invalid media mode")
        if min(width, height) < 1 or (
            expected_frames is not None and expected_frames < 1
        ):
            raise ValueError("Media dimensions and frame count must be positive")
        if (
            type(retention_budget_bytes) is not int
            or retention_budget_bytes <= 64 << 20
        ):
            raise ValueError("Media retention budget must exceed 64 MiB")
        self.retention_budget_bytes = retention_budget_bytes
        self.output = Path(output)
        self.width, self.height = width, height
        self.expected_frames = expected_frames
        self.frame_bytes = width * height * 3
        self.mode, self.ffmpeg = mode, ffmpeg
        self.timing_scope = media_timing_scope(mode)
        self.hash = hashlib.sha256()
        self.bytes = 0
        self.frames = []
        count = min(120, expected_frames or 120, (128 << 20) // self.frame_bytes)
        if mode == "representative" and not count:
            raise ValueError(
                "A representative frame exceeds the sampling memory budget"
            )
        self.indices = [
            i * ((expected_frames or count) - 1) // max(1, count - 1)
            for i in range(count)
        ]
        self.saved_indices = []
        self.raw = None
        self.path = self.output / (
            "output.rgb"
            if mode == "raw"
            else "output.mp4"
            if mode in ("full", "video")
            else "representative.mp4"
        )
        self.output.mkdir(parents=True, exist_ok=True)
        self.spool_path = self.output / (
            "video-spool.rgb" if mode == "video" else ".full-video.rgb"
        )
        if mode == "raw":
            self.raw = self.path.open("xb")
        elif mode in ("full", "video"):
            if mode == "video" and expected_frames is None:
                raise ValueError("Full video requires a frozen expected frame count")
            needed = (expected_frames or 1) * self.frame_bytes + (
                (256 << 20) if mode == "video" else (1 << 30)
            )
            if shutil.disk_usage(self.output).free < needed:
                raise RuntimeError("Insufficient scratch space for full video capture")
            self.raw = self.spool_path.open("xb")
        # File-like name retained for callers that instrument disk writes.
        self.name = str(self.path)

    def write(self, payload):
        if len(payload) % self.frame_bytes:
            raise ValueError("Incomplete RGB frame in media capture")
        if self.mode == "raw" and self.bytes + len(
            payload
        ) > self.retention_budget_bytes - (64 << 20):
            raise RuntimeError("Raw output exceeds the experiment retention budget")
        if (
            self.mode == "video"
            and self.bytes + len(payload) > self.expected_frames * self.frame_bytes
        ):
            raise RuntimeError("Full video exceeds its frozen frame count")
        start = self.bytes // self.frame_bytes
        end = start + len(payload) // self.frame_bytes
        self.hash.update(payload)
        if self.mode == "representative":
            for index in self.indices:
                if start <= index < end:
                    offset = (index - start) * self.frame_bytes
                    self.frames.append(
                        bytes(payload[offset : offset + self.frame_bytes])
                    )
                    self.saved_indices.append(index)
        if self.raw is not None and self.raw.write(payload) != len(payload):
            raise OSError("Incomplete RGB output write")
        self.bytes += len(payload)
        return len(payload)

    def finish(self):
        error = None
        try:
            if self.raw is not None:
                self.raw.close()
            if self.mode in ("full", "video") and self.bytes:
                # Scratch RGB may exceed the final retention budget; encode only
                # after request timing, then discard this capture's exact spool.
                subprocess.run(
                    [
                        self.ffmpeg,
                        "-v",
                        "error",
                        "-nostdin",
                        "-n",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        f"{self.width}x{self.height}",
                        "-r",
                        "24",
                        "-i",
                        str(self.spool_path),
                        "-an",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-threads",
                        "2",
                        "-preset",
                        "fast",
                        "-crf",
                        "10" if self.mode == "full" else "18",
                        "-movflags",
                        "+faststart",
                        str(self.path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=1800,
                    check=True,
                )
            if (
                self.mode == "video"
                and self.path.is_file()
                and self.path.stat().st_size > self.retention_budget_bytes - (64 << 20)
            ):
                raise RuntimeError("Full video exceeds the experiment retention budget")
            if self.mode == "representative" and self.frames:
                width = max(2, min(640, self.width) // 2 * 2)
                height = max(2, round(self.height * width / self.width / 2) * 2)
                subprocess.run(
                    [
                        self.ffmpeg,
                        "-v",
                        "error",
                        "-nostdin",
                        "-n",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        f"{self.width}x{self.height}",
                        "-r",
                        "12",
                        "-i",
                        "pipe:0",
                        "-an",
                        "-vf",
                        f"scale={width}:{height}",
                        "-c:v",
                        "libx264",
                        "-threads",
                        "1",
                        "-preset",
                        "fast",
                        "-crf",
                        "23",
                        "-movflags",
                        "+faststart",
                        str(self.path),
                    ],
                    input=b"".join(self.frames),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=120,
                    check=True,
                )
                if self.path.stat().st_size > 64 << 20:
                    self.path.unlink()
                    raise RuntimeError("Representative video exceeds 64 MiB")
        except BaseException as exc:
            error = type(exc).__name__
            # A failed encoder must not leave a file that looks like valid media.
            if self.mode in ("representative", "full", "video"):
                self.path.unlink(missing_ok=True)
            raise
        finally:
            if self.mode in ("full", "video"):
                self.spool_path.unlink(missing_ok=True)
            self.frames.clear()
            self.receipt = {
                "mode": self.mode,
                "retention_budget_bytes": self.retention_budget_bytes,
                "artifact_helper_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "output_sha256": self.hash.hexdigest(),
                "output_rgb_bytes": self.bytes,
                "output_frames": self.bytes // self.frame_bytes,
                "media_path": str(self.path) if self.path.is_file() else None,
                "sampled_frame_indices": self.saved_indices,
                "representative_scope": (
                    (
                        "uniform_time_lapse_not_full_video_or_pixel_parity"
                        if self.expected_frames
                        else "initial_frames_only_not_full_stream"
                    )
                    if self.mode == "representative"
                    else None
                ),
                "encoding_after_request": True,
                "full_video_fps": 24 if self.mode in ("full", "video") else None,
                "full_video_encoding": (
                    "H264 CRF"
                    + ("10" if self.mode == "full" else "18")
                    + " yuv420p 24fps; visualization, not pixel parity"
                )
                if self.mode in ("full", "video")
                else None,
                "error": error,
                "timing_scope": self.timing_scope,
            }
            (self.output / "media.json").write_text(
                json.dumps(self.receipt, indent=2) + "\n"
            )
        return self.receipt


class ExperimentRetention:
    """Hold a per-experiment lease; remove only caches created by this owner.

    Call finish only after owned workers are stopped and verified. No root scans,
    TTL deletion, recovery of another owner's cache, or process signalling.
    """

    def __init__(self, root):
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or self.root.resolve() != self.root:
            raise ValueError("Experiment root must not traverse a symlink")
        lock_path = self.root / ".retention.lock"
        if lock_path.is_symlink():
            raise ValueError("Experiment lease must not be a symlink")
        self.lock = lock_path.open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise
        self.caches = []

    def create_cache(self, relative):
        relative = Path(relative)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(p in (".", "..") for p in relative.parts)
        ):
            raise ValueError("Cache must be a relative experiment child")
        path = self.root / relative
        if any(
            path == p or p in path.parents or path in p.parents
            for p, _, _ in self.caches
        ):
            raise ValueError("Owned cache directories must not overlap")
        if path.resolve() != path:
            raise ValueError("Cache path must not traverse a symlink")
        # Never adopt an old or shared cache, even when it looks inactive.
        path.mkdir(parents=True, exist_ok=False)
        self.caches.append((path, path.stat().st_dev, path.stat().st_ino))
        return path

    def finish(self, *, workers_stopped=False):
        if self.lock.closed:
            raise RuntimeError("Experiment retention lease is closed")
        if not workers_stopped:
            raise RuntimeError("Workers must be stopped before cache cleanup")
        # Validate the entire allowlist before removing anything.
        for path, device, inode in self.caches:
            if (
                path.is_symlink()
                or path.resolve() != path
                or (path.stat().st_dev, path.stat().st_ino) != (device, inode)
            ):
                raise RuntimeError("Owned cache directory identity changed")
            if (
                any(p.is_symlink() or p.is_mount() for p in path.rglob("*"))
                or path.is_mount()
            ):
                raise RuntimeError(
                    "Owned cache contains a link or mount; refusing cleanup"
                )
        removed, metadata, metadata_bytes = [], [], 0
        for path, _, _ in self.caches:
            removed.append(
                {
                    "path": str(path.relative_to(self.root)),
                    "bytes": retained_bytes(path),
                }
            )
            for file in path.rglob("*.json"):
                metadata_bytes += file.stat().st_size
                if metadata_bytes > 64 << 20:
                    raise RuntimeError(
                        "Cache metadata exceeds the 64 MiB evidence budget"
                    )
                data = file.read_bytes()
                metadata.append(
                    {
                        "path": str(file.relative_to(self.root)),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "content": data.decode("utf-8"),
                    }
                )
        receipt = self.root / "cache-cleanup.json"
        evidence = {
            "status": "prepared",
            "removed": removed,
            "metadata": metadata,
            "workers_stopped": True,
        }
        # Persist evidence before removing any payload, including failure/OOM remnants.
        receipt.write_text(json.dumps(evidence, indent=2) + "\n")
        for path, _, _ in self.caches:
            shutil.rmtree(path)
        evidence["status"] = "complete"
        receipt.write_text(json.dumps(evidence, indent=2) + "\n")
        return check_retention(self.root)

    def close(self):
        self.lock.close()
