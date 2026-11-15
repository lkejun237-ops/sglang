#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify live Coordinator traffic readiness, startup warmup, and frame-level visual quality."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import io
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
import zlib
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CASES_JSON = (
    REPO / "benchmark/whitebox/cases/interactive_production_four_cases.json"
)


def _http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    token: str | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def decode_mp4_to_rgb(
    mp4_path: Path,
    expected_width: int = 832,
    expected_height: int = 480,
) -> list[bytes]:
    """Decode a recorded MP4 file (e.g. SourceSession spool/*/model.mp4) into RGB24 frame bytes."""
    frame_bytes = expected_width * expected_height * 3
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        proc = subprocess.run(
            [
                ffmpeg_bin,
                "-v",
                "error",
                "-i",
                str(mp4_path),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            capture_output=True,
            timeout=60,
        )
        if proc.returncode == 0 and proc.stdout and len(proc.stdout) % frame_bytes == 0:
            raw_all = proc.stdout
            return [
                raw_all[i : i + frame_bytes]
                for i in range(0, len(raw_all), frame_bytes)
            ]
    import av  # type: ignore[import-not-found]

    container = av.open(str(mp4_path), mode="r")
    try:
        out: list[bytes] = []
        for frame in container.decode(video=0):
            rgb_arr = frame.to_ndarray(format="rgb24")
            if rgb_arr.shape[:2] != (expected_height, expected_width):
                raise ValueError(
                    f"Decoded MP4 frame shape {rgb_arr.shape} != {(expected_height, expected_width, 3)}"
                )
            out.append(rgb_arr.tobytes())
        return out
    finally:
        container.close()


def decode_frame_batches_to_rgb(
    batches: list[dict[str, Any]],
    expected_width: int = 832,
    expected_height: int = 480,
) -> list[bytes]:
    """Decode ordered frame_batch messages (raw, jpeg/png/webp, or h264) into RGB24 frame bytes."""
    if not batches:
        return []
    ordered = sorted(batches, key=lambda m: int(m.get("output_frame_start", 0)))
    frame_bytes = expected_width * expected_height * 3
    encoding = str(ordered[0].get("encoding") or "raw").lower()

    if encoding == "raw":
        frames: list[bytes] = []
        for msg in ordered:
            w = int(msg.get("width", expected_width))
            h = int(msg.get("height", expected_height))
            if (w, h) != (expected_width, expected_height):
                raise ValueError(
                    f"Frame geometry mismatch: {w}x{h} != {expected_width}x{expected_height}"
                )
            raw = bytes(msg["payload"])
            count = int(msg.get("num_frames", max(1, len(raw) // frame_bytes)))
            if len(raw) < count * frame_bytes:
                raise ValueError(
                    f"Truncated raw frame_batch payload: {len(raw)} < {count * frame_bytes}"
                )
            for idx in range(count):
                frames.append(raw[idx * frame_bytes : (idx + 1) * frame_bytes])
        return frames

    if encoding in {"jpeg", "jpg", "png", "webp"}:
        frames = []
        for msg in ordered:
            payload_bytes = bytes(msg["payload"])
            if encoding == "png":
                rgb_bytes, w, h = _decode_rgb_png(payload_bytes)
                if (w, h) != (expected_width, expected_height):
                    raise ValueError(
                        f"Decoded PNG size {(w, h)} != {(expected_width, expected_height)}"
                    )
                frames.append(rgb_bytes)
            else:
                from PIL import Image

                img = Image.open(io.BytesIO(payload_bytes)).convert("RGB")
                if img.size != (expected_width, expected_height):
                    raise ValueError(
                        f"Decoded image size {img.size} != {(expected_width, expected_height)}"
                    )
                frames.append(img.tobytes())
        return frames

    if encoding == "h264":
        stream_bytes = b"".join(bytes(msg["payload"]) for msg in ordered)
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            proc = subprocess.run(
                [
                    ffmpeg_bin,
                    "-v",
                    "error",
                    "-f",
                    "h264",
                    "-i",
                    "pipe:0",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ],
                input=stream_bytes,
                capture_output=True,
                timeout=60,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg h264 decode failed: {proc.stderr.decode(errors='replace')}"
                )
            raw_all = proc.stdout
            if len(raw_all) % frame_bytes != 0:
                raise ValueError(
                    f"Decoded h264 byte length {len(raw_all)} not divisible by frame_bytes {frame_bytes}"
                )
            return [
                raw_all[i : i + frame_bytes]
                for i in range(0, len(raw_all), frame_bytes)
            ]
        import av  # type: ignore[import-not-found]

        container = av.open(io.BytesIO(stream_bytes), format="h264", mode="r")
        try:
            out: list[bytes] = []
            for frame in container.decode(video=0):
                rgb_arr = frame.to_ndarray(format="rgb24")
                if rgb_arr.shape[:2] != (expected_height, expected_width):
                    raise ValueError(
                        f"Decoded av frame shape {rgb_arr.shape} != {(expected_height, expected_width, 3)}"
                    )
                out.append(rgb_arr.tobytes())
            return out
        finally:
            container.close()

    raise ValueError(f"Unsupported frame_batch encoding: {encoding!r}")


def _luma_stats_and_blocks(
    rgb: bytes, width: int, height: int, block: int = 8
) -> tuple[float, float, list[tuple[float, float]]]:
    """Compute BT.601 luma mean, std, and 8x8 block (mean, var) without requiring GPU libs."""
    try:
        import numpy as np

        arr = (
            np.frombuffer(rgb, dtype=np.uint8)
            .reshape(height, width, 3)
            .astype(np.float64)
        )
        y = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        mean_val = float(y.mean())
        std_val = float(y.std())
        bh = (height // block) * block
        bw = (width // block) * block
        patches = y[:bh, :bw].reshape(bh // block, block, bw // block, block)
        patches = patches.transpose(0, 2, 1, 3).reshape(-1, block * block)
        p_means = patches.mean(axis=1)
        p_vars = patches.var(axis=1)
        blocks = [(float(m), float(v)) for m, v in zip(p_means, p_vars)]
        return mean_val, std_val, blocks
    except ImportError:
        pixels = [
            0.299 * rgb[i] + 0.587 * rgb[i + 1] + 0.114 * rgb[i + 2]
            for i in range(0, len(rgb), 3)
        ]
        n = len(pixels)
        mean_val = sum(pixels) / n
        var_val = sum((p - mean_val) ** 2 for p in pixels) / n
        blocks = []
        for by in range(0, (height // block) * block, block):
            for bx in range(0, (width // block) * block, block):
                vals = [
                    pixels[(by + dy) * width + (bx + dx)]
                    for dy in range(block)
                    for dx in range(block)
                ]
                bm = sum(vals) / len(vals)
                bv = sum((v - bm) ** 2 for v in vals) / len(vals)
                blocks.append((bm, bv))
        return mean_val, math.sqrt(var_val), blocks


def compute_pair_psnr_ssim_mae(
    rgb_a: bytes, rgb_b: bytes, width: int, height: int, block: int = 8
) -> dict[str, float]:
    """Compute RGB PSNR (dB), RGB MAE, and 8x8 block-averaged Luma SSIM between two frames."""
    if len(rgb_a) != len(rgb_b) or len(rgb_a) != width * height * 3:
        raise ValueError("RGB frame byte sizes do not match expected geometry")
    if rgb_a == rgb_b:
        return {"psnr_db": 99.0, "ssim": 1.0, "mae": 0.0, "exact_match": 1.0}

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    try:
        import numpy as np

        a = np.frombuffer(rgb_a, dtype=np.uint8).astype(np.float64)
        b = np.frombuffer(rgb_b, dtype=np.uint8).astype(np.float64)
        diff = a - b
        mse = float(np.mean(diff * diff))
        mae = float(np.mean(np.abs(diff)))
        psnr = (
            99.0
            if mse <= 1e-12
            else float(min(99.0, 10.0 * math.log10((255.0**2) / mse)))
        )

        ya = (0.299 * a[0::3] + 0.587 * a[1::3] + 0.114 * a[2::3]).reshape(
            height, width
        )
        yb = (0.299 * b[0::3] + 0.587 * b[1::3] + 0.114 * b[2::3]).reshape(
            height, width
        )
        bh = (height // block) * block
        bw = (width // block) * block
        pa = (
            ya[:bh, :bw]
            .reshape(bh // block, block, bw // block, block)
            .transpose(0, 2, 1, 3)
            .reshape(-1, block * block)
        )
        pb = (
            yb[:bh, :bw]
            .reshape(bh // block, block, bw // block, block)
            .transpose(0, 2, 1, 3)
            .reshape(-1, block * block)
        )
        mu_a = pa.mean(axis=1)
        mu_b = pb.mean(axis=1)
        var_a = pa.var(axis=1)
        var_b = pb.var(axis=1)
        cov_ab = ((pa - mu_a[:, None]) * (pb - mu_b[:, None])).mean(axis=1)
        num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov_ab + c2)
        den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
        ssim = float(np.mean(num / den))
        return {
            "psnr_db": round(psnr, 4),
            "ssim": round(ssim, 6),
            "mae": round(mae, 4),
            "exact_match": 0.0,
        }
    except ImportError:
        n = len(rgb_a)
        sq = 0.0
        ab = 0.0
        for x, y in zip(rgb_a, rgb_b):
            d = float(x) - float(y)
            sq += d * d
            ab += abs(d)
        mse = sq / n
        mae = ab / n
        psnr = 99.0 if mse <= 1e-12 else min(99.0, 10.0 * math.log10((255.0**2) / mse))
        la = [
            0.299 * rgb_a[i] + 0.587 * rgb_a[i + 1] + 0.114 * rgb_a[i + 2]
            for i in range(0, len(rgb_a), 3)
        ]
        lb = [
            0.299 * rgb_b[i] + 0.587 * rgb_b[i + 1] + 0.114 * rgb_b[i + 2]
            for i in range(0, len(rgb_b), 3)
        ]
        ssim_vals = []
        for by in range(0, (height // block) * block, block):
            for bx in range(0, (width // block) * block, block):
                va = [
                    float(la[(by + dy) * width + (bx + dx)])
                    for dy in range(block)
                    for dx in range(block)
                ]
                vb = [
                    float(lb[(by + dy) * width + (bx + dx)])
                    for dy in range(block)
                    for dx in range(block)
                ]
                k = len(va)
                ma = sum(va) / k
                mb = sum(vb) / k
                sa = sum((u - ma) ** 2 for u in va) / k
                sb = sum((v - mb) ** 2 for v in vb) / k
                sab = sum((u - ma) * (v - mb) for u, v in zip(va, vb)) / k
                ssim_vals.append(
                    ((2.0 * ma * mb + c1) * (2.0 * sab + c2))
                    / ((ma * ma + mb * mb + c1) * (sa + sb + c2))
                )
        ssim = sum(ssim_vals) / max(1, len(ssim_vals))
        return {
            "psnr_db": round(psnr, 4),
            "ssim": round(ssim, 6),
            "mae": round(mae, 4),
            "exact_match": 0.0,
        }


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _encode_rgb_png(rgb: bytes, width: int, height: int) -> bytes:
    """Encode 8-bit RGB24 bytes into a valid PNG using only the Python standard library."""
    stride = width * 3
    if len(rgb) != stride * height:
        raise ValueError(f"RGB buffer size {len(rgb)} != {stride * height}")
    raw_scanlines = b"".join(
        b"\x00" + rgb[y * stride : (y + 1) * stride] for y in range(height)
    )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw_scanlines, level=6))
        + _png_chunk(b"IEND", b"")
    )


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _decode_rgb_png(png_bytes: bytes) -> tuple[bytes, int, int]:
    """Decode an 8-bit RGB/RGBA PNG into RGB24 bytes without requiring Pillow."""
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        return img.tobytes(), img.size[0], img.size[1]
    except ImportError:
        pass
    if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Invalid PNG signature")
    pos = 8
    width = height = bit_depth = color_type = interlace = 0
    idat_parts: list[bytes] = []
    while pos + 8 <= len(png_bytes):
        length = struct.unpack(">I", png_bytes[pos : pos + 4])[0]
        ctype = png_bytes[pos + 4 : pos + 8]
        cdata = png_bytes[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", cdata
            )
        elif ctype == b"IDAT":
            idat_parts.append(cdata)
        elif ctype == b"IEND":
            break
    if bit_depth != 8 or color_type not in (2, 6) or interlace != 0:
        raise ValueError(
            f"Stdlib PNG decoder requires non-interlaced 8-bit RGB/RGBA (got depth={bit_depth}, type={color_type})"
        )
    bpp = 3 if color_type == 2 else 4
    stride = width * bpp
    raw = zlib.decompress(b"".join(idat_parts))
    if len(raw) < height * (1 + stride):
        raise ValueError("Truncated PNG IDAT stream")
    out_rgb = bytearray()
    prev = bytearray(stride)
    offset = 0
    for _ in range(height):
        ftype = raw[offset]
        scan = bytearray(raw[offset + 1 : offset + 1 + stride])
        offset += 1 + stride
        recon = bytearray(stride)
        for i in range(stride):
            left = recon[i - bpp] if i >= bpp else 0
            up = prev[i]
            up_left = prev[i - bpp] if i >= bpp else 0
            val = scan[i]
            if ftype == 0:
                recon[i] = val
            elif ftype == 1:
                recon[i] = (val + left) & 0xFF
            elif ftype == 2:
                recon[i] = (val + up) & 0xFF
            elif ftype == 3:
                recon[i] = (val + ((left + up) >> 1)) & 0xFF
            elif ftype == 4:
                recon[i] = (val + _paeth_predictor(left, up, up_left)) & 0xFF
            else:
                raise ValueError(f"Unsupported PNG filter type: {ftype}")
        if bpp == 3:
            out_rgb.extend(recon)
        else:
            for px in range(0, stride, 4):
                out_rgb.extend(recon[px : px + 3])
        prev = recon
    return bytes(out_rgb), width, height


def save_frames_png(
    frames: list[bytes],
    frames_dir: Path,
    width: int = 832,
    height: int = 480,
) -> list[str]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for idx, rgb in enumerate(frames):
        out = frames_dir / f"frame_{idx:04d}.png"
        out.write_bytes(_encode_rgb_png(rgb, width, height))
        saved.append(out.name)
    return saved


def load_frames_png(frames_dir: Path) -> tuple[list[bytes], int, int]:
    pngs = sorted(frames_dir.glob("frame_*.png"))
    if not pngs:
        raise FileNotFoundError(f"No frame_*.png files found under {frames_dir}")
    frames: list[bytes] = []
    width = height = 0
    for p in pngs:
        rgb, w, h = _decode_rgb_png(p.read_bytes())
        if not frames:
            width, height = w, h
        elif (w, h) != (width, height):
            raise ValueError(f"Inconsistent PNG geometry in {p}: {(w, h)}")
        frames.append(rgb)
    return frames, width, height


BASELINE_MANIFEST_FILENAME = "baseline_manifest.json"


def compute_frames_sha256(frames: list[bytes]) -> str:
    """Compute a deterministic aggregate SHA256 digest over ordered RGB24 frame bytes."""
    h = hashlib.sha256()
    for idx, rgb in enumerate(frames):
        frame_digest = hashlib.sha256(rgb).hexdigest()
        h.update(f"{idx}:{len(rgb)}:{frame_digest}\n".encode("ascii"))
    return h.hexdigest()


def write_baseline_manifest(
    baseline_dir: Path,
    *,
    frames: list[bytes],
    width: int,
    height: int,
    case_id: str,
    seed: int,
    inference_steps: int | None = None,
    source_deployment_id: str | None = None,
    routing_revision: str | None = None,
) -> dict[str, Any]:
    """Write a pinned baseline identity manifest alongside saved baseline PNG frames."""
    if not frames:
        raise ValueError("Cannot write baseline manifest for empty frame sequence")
    if not str(case_id).strip():
        raise ValueError("Baseline manifest requires a non-empty case_id")
    manifest: dict[str, Any] = {
        "schema": 1,
        "case_id": str(case_id).strip(),
        "seed": int(seed),
        "width": int(width),
        "height": int(height),
        "frame_count": len(frames),
        "frames_sha256": compute_frames_sha256(frames),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if inference_steps is not None:
        manifest["inference_steps"] = int(inference_steps)
    if source_deployment_id:
        manifest["source_deployment_id"] = str(source_deployment_id)
    if routing_revision:
        manifest["routing_revision"] = str(routing_revision)
    baseline_dir = Path(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / BASELINE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _is_valid_baseline_identity(
    identity: dict[str, Any] | None,
    baseline_frames: list[bytes] | None = None,
) -> bool:
    if not isinstance(identity, dict) or identity.get("schema") != 1:
        return False
    if not str(identity.get("case_id") or "").strip():
        return False
    if type(identity.get("seed")) is not int:
        return False
    expected_sha = str(identity.get("frames_sha256") or "")
    if len(expected_sha) != 64:
        return False
    if baseline_frames is not None:
        if identity.get("frame_count") != len(baseline_frames):
            return False
        if compute_frames_sha256(baseline_frames) != expected_sha:
            return False
    return True


def load_verified_baseline(
    baseline_dir: Path,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_case_id: str | None = None,
    expected_seed: int | None = None,
    expected_inference_steps: int | None = None,
) -> tuple[list[bytes], int, int, dict[str, Any]]:
    """Load baseline PNG frames and fail-closed on missing or mismatched baseline identity."""
    baseline_dir = Path(baseline_dir)
    frames_dir = (
        baseline_dir / "frames" if (baseline_dir / "frames").is_dir() else baseline_dir
    )
    manifest_candidates = [
        baseline_dir / BASELINE_MANIFEST_FILENAME,
        frames_dir / BASELINE_MANIFEST_FILENAME,
        frames_dir.parent / BASELINE_MANIFEST_FILENAME,
    ]
    manifest_path = next((p for p in manifest_candidates if p.is_file()), None)
    if manifest_path is None:
        raise ValueError(
            f"Baseline directory {baseline_dir} is missing required {BASELINE_MANIFEST_FILENAME}; "
            "record a pinned baseline via `record-baseline` first"
        )
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise ValueError(f"Invalid baseline manifest schema in {manifest_path}")
    frames, bw, bh = load_frames_png(frames_dir)
    if expected_width is not None and expected_height is not None:
        if (bw, bh) != (expected_width, expected_height):
            raise ValueError(
                f"Baseline geometry {(bw, bh)} != expected {(expected_width, expected_height)}"
            )
    if (manifest.get("width"), manifest.get("height")) != (bw, bh):
        raise ValueError(
            f"Baseline manifest geometry {(manifest.get('width'), manifest.get('height'))} != decoded {(bw, bh)}"
        )
    if not _is_valid_baseline_identity(manifest, frames):
        raise ValueError(
            f"Baseline frames under {frames_dir} do not match pinned {BASELINE_MANIFEST_FILENAME} digest/count"
        )
    if expected_case_id is not None and manifest.get("case_id") != expected_case_id:
        raise ValueError(
            f"Baseline case_id {manifest.get('case_id')!r} != candidate {expected_case_id!r}"
        )
    if expected_seed is not None and manifest.get("seed") != int(expected_seed):
        raise ValueError(
            f"Baseline seed {manifest.get('seed')!r} != candidate {int(expected_seed)!r}"
        )
    if (
        expected_inference_steps is not None
        and "inference_steps" in manifest
        and int(manifest["inference_steps"]) != int(expected_inference_steps)
    ):
        raise ValueError(
            f"Baseline inference_steps {manifest.get('inference_steps')!r} != candidate {int(expected_inference_steps)!r}"
        )
    return frames, bw, bh, manifest


def evaluate_visual_quality(
    frames: list[bytes],
    *,
    width: int = 832,
    height: int = 480,
    min_frames: int = 16,
    baseline_frames: list[bytes] | None = None,
    baseline_identity: dict[str, Any] | None = None,
    min_psnr_db: float = 25.0,
    min_ssim: float = 0.85,
    min_first_frame_psnr_db: float = 28.0,
    min_first_frame_ssim: float = 0.90,
    max_luma_mean_diff: float = 5.0,
) -> dict[str, Any]:
    """Evaluate intrinsic non-degradation gates and fail-closed causal AR+DMD baseline parity."""
    per_frame: list[dict[str, Any]] = []
    luma_means: list[float] = []
    luma_stds: list[float] = []
    temporal_diffs: list[float] = []

    for idx, rgb in enumerate(frames):
        if len(rgb) != width * height * 3:
            raise ValueError(f"Frame {idx} byte length mismatch")
        mean_y, std_y, _ = _luma_stats_and_blocks(rgb, width, height)
        luma_means.append(mean_y)
        luma_stds.append(std_y)
        if idx > 0:
            prev = frames[idx - 1]
            step = max(1, len(rgb) // 4096)
            sampled = [abs(a - b) for a, b in zip(rgb[::step], prev[::step])]
            temporal_diffs.append(sum(sampled) / max(1, len(sampled)))
        per_frame.append(
            {
                "frame_index": idx,
                "sha256": hashlib.sha256(rgb).hexdigest(),
                "luma_mean": round(mean_y, 3),
                "luma_std": round(std_y, 3),
            }
        )

    decoded_count = len(frames)
    mean_luma = (
        round(sum(luma_means) / max(1, decoded_count), 3) if decoded_count else 0.0
    )
    min_std = round(min(luma_stds), 3) if luma_stds else 0.0
    motion_mae = (
        round(sum(temporal_diffs) / len(temporal_diffs), 4) if temporal_diffs else 0.0
    )

    intrinsic_passed = bool(
        decoded_count >= min_frames
        and 5.0 <= mean_luma <= 250.0
        and min_std >= 5.0
        and (decoded_count < 4 or motion_mae >= 0.05)
    )
    intrinsic = {
        "passed": intrinsic_passed,
        "decoded_frames": decoded_count,
        "min_frames_required": min_frames,
        "resolution": [width, height],
        "mean_luma": mean_luma,
        "min_frame_luma_std": min_std,
        "temporal_motion_mae": motion_mae,
    }

    baseline_cmp: dict[str, Any] | None = None
    if baseline_frames is not None:
        compared = min(len(frames), len(baseline_frames))
        pair_metrics = [
            compute_pair_psnr_ssim_mae(frames[i], baseline_frames[i], width, height)
            for i in range(compared)
        ]
        for i, pm in enumerate(pair_metrics):
            per_frame[i]["baseline_psnr_db"] = pm["psnr_db"]
            per_frame[i]["baseline_ssim"] = pm["ssim"]
            per_frame[i]["baseline_mae"] = pm["mae"]
        mean_psnr = (
            round(sum(m["psnr_db"] for m in pair_metrics) / compared, 4)
            if compared
            else 0.0
        )
        min_pair_psnr = (
            round(min(m["psnr_db"] for m in pair_metrics), 4) if compared else 0.0
        )
        mean_ssim_val = (
            round(sum(m["ssim"] for m in pair_metrics) / compared, 6)
            if compared
            else 0.0
        )
        min_pair_ssim = (
            round(min(m["ssim"] for m in pair_metrics), 6) if compared else 0.0
        )
        mean_mae_val = (
            round(sum(m["mae"] for m in pair_metrics) / compared, 4)
            if compared
            else 999.0
        )
        exact_matches = sum(int(m["exact_match"]) for m in pair_metrics)
        first_psnr = pair_metrics[0]["psnr_db"] if pair_metrics else 0.0
        first_ssim = pair_metrics[0]["ssim"] if pair_metrics else 0.0
        base_lumas = [
            _luma_stats_and_blocks(bf, width, height)[0]
            for bf in baseline_frames[:compared]
        ]
        base_mean_luma = (
            round(sum(base_lumas) / len(base_lumas), 3) if base_lumas else 0.0
        )
        luma_diff = round(abs(mean_luma - base_mean_luma), 3)
        cmp_passed = bool(
            compared >= min_frames
            and mean_psnr >= min_psnr_db
            and mean_ssim_val >= min_ssim
            and first_psnr >= min_first_frame_psnr_db
            and first_ssim >= min_first_frame_ssim
            and luma_diff <= max_luma_mean_diff
        )
        baseline_cmp = {
            "passed": cmp_passed,
            "compared_frames": compared,
            "baseline_total_frames": len(baseline_frames),
            "first_frame_psnr_db": first_psnr,
            "first_frame_ssim": first_ssim,
            "mean_psnr_db": mean_psnr,
            "min_psnr_db": min_pair_psnr,
            "mean_ssim": mean_ssim_val,
            "min_ssim": min_pair_ssim,
            "mean_mae": mean_mae_val,
            "luma_mean_diff": luma_diff,
            "exact_frame_matches": exact_matches,
            "gates": {
                "min_psnr_db": min_psnr_db,
                "min_ssim": min_ssim,
                "min_first_frame_psnr_db": min_first_frame_psnr_db,
                "min_first_frame_ssim": min_first_frame_ssim,
                "max_luma_mean_diff": max_luma_mean_diff,
            },
        }
        has_bound_identity = _is_valid_baseline_identity(
            baseline_identity, baseline_frames
        )
        quality_unchanged = bool(intrinsic_passed and cmp_passed and has_bound_identity)
    else:
        # Fail-closed: without a pinned baseline comparison, candidate quality_unchanged is NEVER True.
        quality_unchanged = False

    return {
        "quality_unchanged": quality_unchanged,
        "is_baseline": baseline_frames is None,
        "baseline_identity": baseline_identity,
        "intrinsic_quality": intrinsic,
        "baseline_comparison": baseline_cmp,
        "per_frame": per_frame,
    }


ALL_ROLES = ("encoder", "dit", "decoder")


def check_role_warmup_and_pool_capacity(
    *,
    role_urls: dict[str, str],
    coordinator_url: str | None,
    routing_revision: str | None,
    compatibility: str | None,
    profile: str,
    network_domain: str,
    room_token: str | None,
    require_warmup_roles: set[str] | None = None,
) -> dict[str, Any]:
    """Query /health on each role and fail-closed unless all required roles explicitly report warmed_up=true."""
    warmup_report: dict[str, Any] = {}
    required = (
        set(require_warmup_roles)
        if require_warmup_roles is not None
        else set(ALL_ROLES)
    )
    missing_roles = sorted(required - set(role_urls))
    all_warmed = len(missing_roles) == 0 and len(role_urls) > 0
    for role in missing_roles:
        warmup_report[role] = {
            "http": None,
            "ready": False,
            "warmed_up": False,
            "explicit_warmup_flag": False,
            "warmup": None,
            "missing_endpoint": True,
        }
    for role, base_url in role_urls.items():
        health = _http_json(base_url.rstrip("/") + "/health")
        ready = health.get("ready") is True and health.get("role") == role
        if not ready:
            raise RuntimeError(f"Role {role} at {base_url} is not ready: {health}")
        warmed = health.get("warmed_up")
        # Fail-closed across both owned and reused roles: missing warmed_up or False fails the gate.
        explicit_warmed = warmed is True
        if not explicit_warmed:
            all_warmed = False
        warmup_report[role] = {
            "http": base_url,
            "ready": ready,
            "warmed_up": explicit_warmed,
            "explicit_warmup_flag": explicit_warmed,
            "warmup": health.get("warmup"),
        }

    capacity_report: dict[str, Any] = {}
    if coordinator_url and routing_revision:
        pub = _http_json(
            f"{coordinator_url.rstrip('/')}/v1/capacity?model_revision={routing_revision}"
        )
        capacity_report["route_capacity"] = pub
        if pub.get("backend") != "role_pool":
            raise RuntimeError(f"Coordinator route backend is not role_pool: {pub}")
        if room_token and compatibility:
            exact = _http_json(
                f"{coordinator_url.rstrip('/')}/v1/role-pool/capacity",
                payload={
                    "profile": profile,
                    "compatibility": compatibility,
                    "network_domain": network_domain,
                },
                token=room_token,
            )
            capacity_report["exact_group_capacity"] = exact
            mg_free = (
                exact.get("roles", {})
                .get("model_group", {})
                .get("free_slots", exact.get("free_slots", 0))
            )
            if int(mg_free) < 1:
                raise RuntimeError(
                    f"Compatibility group {compatibility!r} has no free slots: {exact}"
                )
    return {
        "all_roles_warmed_up": all_warmed,
        "roles": warmup_report,
        "capacity": capacity_report,
    }


def infer_denoising_steps_from_state(st: dict[str, Any] | None) -> int | None:
    """Derive regular denoising step count from deployment state dit_options if present."""
    if not isinstance(st, dict):
        return None
    opts = st.get("spec", {}).get("roles", {}).get("dit", {}).get("dit_options", {})
    raw = opts.get("zing_denoising_step_list")
    if isinstance(raw, list) and raw:
        return len(raw)
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if parts:
            return len(parts)
    return None


async def run_live_replay_capture(
    *,
    cases_json: Path,
    case_id: str,
    media_dir: Path,
    mode: str,
    coordinator_url: str | None,
    routing_revision: str | None,
    compatibility: str | None,
    profile: str,
    network_domain: str,
    seed: int,
    target_frames: int,
    work_dir: Path,
    inference_steps: int = 2,
) -> tuple[dict[str, Any], list[bytes]]:
    """Run a real SourceSession generation through Coordinator and capture decoded RGB frames."""
    wb_dir = REPO / "benchmark/whitebox"
    if str(wb_dir) not in sys.path:
        sys.path.insert(0, str(wb_dir))
    if str(REPO / "python") not in sys.path:
        sys.path.insert(0, str(REPO / "python"))

    import msgspec
    import run_room_sglang_source_replay as replay
    from source_replay_role_pool import SuiteRolePool

    suite_data = json.loads(cases_json.read_text())
    cases_list = suite_data.get("cases") or suite_data.get("recordings") or []
    rec_map = {r["case_id"]: r for r in cases_list}
    if case_id not in rec_map:
        raise KeyError(f"Unknown case_id {case_id!r}; available: {sorted(rec_map)}")
    recording_meta = copy.deepcopy(rec_map[case_id])
    # Disable in-segment condition switches for deterministic single-prompt visual baseline
    # while still exercising full ControlPrefill + RenderSegment steady-state generation.
    recording_meta["condition_switches"] = []

    spec = replay.VideoSpec(width=832, height=480, fps=24)
    work_dir.mkdir(parents=True, exist_ok=True)
    spool_dir = work_dir / "spool"
    spool_dir.mkdir(parents=True, exist_ok=True)

    captured_batches: list[dict[str, Any]] = []
    orig_decode = msgspec.msgpack.decode

    def _sniff_decode(raw: bytes, *args: Any, **kwargs: Any) -> Any:
        msg = orig_decode(raw, *args, **kwargs)
        if isinstance(msg, dict) and msg.get("type") == "frame_batch":
            captured_batches.append(msg)
        return msg

    async def _execute_case() -> dict[str, Any]:
        cache_store = replay.MockSglangCacheStore()
        orig_msgpack_decode = replay.msgspec.msgpack.decode
        prev_steps_env = os.environ.get("WHITEBOX_INFERENCE_STEPS")
        os.environ["WHITEBOX_INFERENCE_STEPS"] = str(int(inference_steps))
        replay.msgspec.msgpack.decode = _sniff_decode
        try:
            return await replay._run_single_segment_flow(
                case_id=case_id,
                recording_meta=recording_meta,
                media_root=media_dir,
                spool_dir=spool_dir,
                spec=spec,
                mode=mode,
                cache_store=cache_store,
                sglang_url=None,
                coordinator_url=coordinator_url,
                max_frames_after_last_event=target_frames,
                full_frames=False,
                preserve_switch_boundaries=False,
                flow_mode="deterministic",
                realtime_pacing=(mode == "real-sglang"),
                pacing_speed=1.0,
                capture_fps=24,
                repeat_warm_prefill=False,
                prewarm_switch_conditions=False,
                seed_override=seed,
            )
        finally:
            replay.msgspec.msgpack.decode = orig_msgpack_decode
            if prev_steps_env is None:
                os.environ.pop("WHITEBOX_INFERENCE_STEPS", None)
            else:
                os.environ["WHITEBOX_INFERENCE_STEPS"] = prev_steps_env

    if mode == "real-sglang":
        async with SuiteRolePool(
            coordinator_url=coordinator_url,
            revision=routing_revision,
            compatibility=compatibility,
            profile=profile,
            network_domain=network_domain,
            room_token_env="WHITEBOX_ROOM_TOKEN",
            signing_secret_env="ROLE_POOL_SIGNING_KEY",
        ) as pool:
            await pool.wait_idle()
            session_result = await _execute_case()
            post_cap = await pool.wait_idle()
            session_result["post_session_capacity"] = post_cap
    else:
        if not (media_dir / recording_meta["recording_id"] / "whitebox.mp4").exists():
            replay.prepare_recording_media(
                recording_meta,
                media_dir,
                oss_bucket="unused",
                oss_endpoint="https://oss-cn-shanghai.aliyuncs.com",
                synthesize_fixtures=True,
                synthetic_frames=max(32, target_frames + 8),
            )
        session_result = await _execute_case()

    spool_mp4s = sorted(spool_dir.glob("*/model.mp4"), key=lambda p: p.stat().st_mtime)
    if spool_mp4s:
        frames = decode_mp4_to_rgb(
            spool_mp4s[-1], expected_width=spec.width, expected_height=spec.height
        )
    else:
        frames = decode_frame_batches_to_rgb(
            captured_batches, expected_width=spec.width, expected_height=spec.height
        )
    return session_result, frames


def attach_verification_to_deployment(
    state_path: Path,
    verification: dict[str, Any],
    *,
    sync_oss: bool = False,
    local_history_dir: Path | None = None,
    aliyun_profile: str | None = None,
) -> dict[str, Any]:
    """Persist traffic_verification.json into a deployment state directory and refresh credentials."""
    if (
        verification.get("is_baseline") is True
        or verification.get("mode_kind") == "baseline_recording"
    ):
        raise ValueError(
            "Baseline recording output cannot be attached as candidate deployment verification"
        )
    deploy_dir = Path(__file__).resolve().parent
    if str(deploy_dir) not in sys.path:
        sys.path.insert(0, str(deploy_dir))
    import orchestrate as orch

    sanitized = copy.deepcopy(verification)
    b_cmp = sanitized.get("baseline_comparison")
    b_id = sanitized.get("baseline_identity")
    if not (
        _is_valid_baseline_identity(b_id)
        and isinstance(b_cmp, dict)
        and b_cmp.get("passed") is True
        and (sanitized.get("intrinsic_quality") or {}).get("passed") is True
    ):
        sanitized["quality_unchanged"] = False

    run_dir = state_path.parent
    orch.save(run_dir / "traffic_verification.json", sanitized)
    st = orch.load_state(state_path)
    st["traffic_verification"] = {
        k: v for k, v in sanitized.items() if k != "per_frame"
    }
    orch.save(state_path, st)
    summary = orch.write_deployment_summary(st, run_dir)
    orch.save_history_record(st, run_dir, history_dir=local_history_dir)
    if sync_oss:
        cloud = orch.Cloud(target=st.get("target"), aliyun_profile=aliyun_profile)
        staging_store = local_history_dir or (run_dir / "history")
        cloud.sync_history_artifacts(
            staging_store, st["deployment_id"], summary_dir=run_dir
        )
    return summary


def _add_live_session_args(sp: argparse.ArgumentParser, *, allow_state: bool) -> None:
    if allow_state:
        sp.add_argument(
            "--state", type=Path, help="Optional orchestrate state.json path"
        )
    sp.add_argument("--coordinator-url", default=None)
    sp.add_argument("--routing-revision", default=None)
    sp.add_argument("--compatibility", default=None)
    sp.add_argument("--profile", default="480")
    sp.add_argument("--network-domain", default="shanghai-vpc-h100-h200")
    for r in ALL_ROLES:
        sp.add_argument(f"--{r}-url", default=None)
    sp.add_argument("--cases-json", type=Path, default=DEFAULT_CASES_JSON)
    sp.add_argument("--case-id", default="shanghai_drive_baseline")
    sp.add_argument(
        "--media-dir", type=Path, default=Path("/tmp/whitebox-four-cases-media")
    )
    sp.add_argument("--seed", type=int, default=886410)
    sp.add_argument(
        "--inference-steps",
        type=int,
        default=None,
        help="Regular denoising steps (auto-inferred from state dit_options or defaults to 2 for DMD few-step)",
    )
    sp.add_argument("--target-frames", type=int, default=20)
    sp.add_argument(
        "--mode", choices=("real-sglang", "mock-worker"), default="real-sglang"
    )
    sp.add_argument("--output-dir", type=Path, required=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)

    run_p = sub.add_parser(
        "run",
        help="Execute candidate deployment live traffic session, 3-role warmup gate, and baseline quality gate",
    )
    _add_live_session_args(run_p, allow_state=True)
    run_p.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="Required directory containing pinned baseline PNG frames and baseline_manifest.json",
    )
    run_p.add_argument("--min-psnr-db", type=float, default=25.0)
    run_p.add_argument("--min-ssim", type=float, default=0.85)
    run_p.add_argument(
        "--sync-oss",
        dest="sync_oss",
        action="store_true",
        default=True,
        help="Archive verification receipt to fixed environment OSS prefix (enabled by default)",
    )
    run_p.add_argument(
        "--no-sync-oss",
        dest="sync_oss",
        action="store_false",
        help="Skip archiving verification receipt to OSS",
    )
    run_p.add_argument(
        "--local-history-dir",
        type=Path,
        default=None,
        help="Optional extra local directory to mirror deployment history",
    )
    run_p.add_argument("--aliyun-profile", default=os.environ.get("ALIYUN_PROFILE"))

    rec_p = sub.add_parser(
        "record-baseline",
        help="Explicitly record a pinned reference baseline (never marks a candidate deployment as passed)",
    )
    _add_live_session_args(rec_p, allow_state=False)
    rec_p.add_argument(
        "--source-state",
        type=Path,
        default=None,
        help="Optional reference deployment state.json to read endpoints/routing without mutating state",
    )

    cmp_p = sub.add_parser(
        "compare-dirs",
        help="Compare saved candidate PNG frame directory against a pinned baseline directory",
    )
    cmp_p.add_argument("--candidate-dir", type=Path, required=True)
    cmp_p.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="Required directory containing pinned baseline PNG frames and baseline_manifest.json",
    )
    cmp_p.add_argument("--case-id", default=None)
    cmp_p.add_argument("--seed", type=int, default=None)
    cmp_p.add_argument("--min-frames", type=int, default=16)
    cmp_p.add_argument("--min-psnr-db", type=float, default=25.0)
    cmp_p.add_argument("--min-ssim", type=float, default=0.85)
    cmp_p.add_argument("--output", type=Path, required=True)
    cmp_p.add_argument("--state", type=Path, default=None)
    cmp_p.add_argument(
        "--sync-oss",
        dest="sync_oss",
        action="store_true",
        default=True,
        help="Archive verification receipt to fixed environment OSS prefix (enabled by default)",
    )
    cmp_p.add_argument(
        "--no-sync-oss",
        dest="sync_oss",
        action="store_false",
        help="Skip archiving verification receipt to OSS",
    )
    cmp_p.add_argument(
        "--local-history-dir",
        type=Path,
        default=None,
        help="Optional extra local directory to mirror deployment history",
    )
    cmp_p.add_argument("--aliyun-profile", default=os.environ.get("ALIYUN_PROFILE"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.action == "compare-dirs":
        cand_frames_dir = (
            args.candidate_dir / "frames"
            if (args.candidate_dir / "frames").is_dir()
            else args.candidate_dir
        )
        frames, w, h = load_frames_png(cand_frames_dir)
        base_frames, _, _, base_manifest = load_verified_baseline(
            args.baseline_dir,
            expected_width=w,
            expected_height=h,
            expected_case_id=args.case_id,
            expected_seed=args.seed,
        )
        report = evaluate_visual_quality(
            frames,
            width=w,
            height=h,
            min_frames=args.min_frames,
            baseline_frames=base_frames,
            baseline_identity=base_manifest,
            min_psnr_db=args.min_psnr_db,
            min_ssim=args.min_ssim,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if args.state:
            attach_verification_to_deployment(
                args.state,
                report,
                sync_oss=args.sync_oss,
                local_history_dir=args.local_history_dir,
                aliyun_profile=args.aliyun_profile,
            )
        print(
            json.dumps(
                {k: v for k, v in report.items() if k != "per_frame"}, sort_keys=True
            )
        )
        return 0 if report["quality_unchanged"] else 2

    t0 = time.monotonic()
    role_urls: dict[str, str] = {}
    deployment_id = None
    coordinator_url = args.coordinator_url
    routing_revision = args.routing_revision
    compatibility = args.compatibility
    profile = args.profile
    network_domain = args.network_domain

    state_for_endpoints: Path | None = (
        getattr(args, "state", None)
        if args.action == "run"
        else getattr(args, "source_state", None)
    )
    st_obj: dict[str, Any] | None = None
    if state_for_endpoints:
        st = json.loads(state_for_endpoints.read_text())
        st_obj = st
        deployment_id = st.get("deployment_id")
        pool_cfg = st.get("spec", {}).get("pool", {})
        coordinator_url = (
            coordinator_url
            or st.get("target", {}).get("coordinator_url")
            or st.get("spec", {}).get("target", {}).get("coordinator_url")
        )
        routing_revision = (
            routing_revision or pool_cfg.get("route") or st.get("spec", {}).get("route")
        )
        compatibility = compatibility or pool_cfg.get("compatibility")
        profile = pool_cfg.get("profile") or profile
        network_domain = pool_cfg.get("network_domain") or network_domain
        for r in ALL_ROLES:
            res = st.get("resources", {}).get(r, {})
            http_ep = (res.get("receipt") or {}).get("http") or (
                st.get("spec", {}).get("existing_services", {}).get(r, {}).get("http")
            )
            if http_ep:
                role_urls[r] = http_ep

    for r in ALL_ROLES:
        cli_ep = getattr(args, f"{r}_url")
        if cli_ep:
            role_urls[r] = cli_ep

    effective_steps = (
        args.inference_steps
        or infer_denoising_steps_from_state(st_obj)
        or int(os.environ.get("WHITEBOX_INFERENCE_STEPS", "2"))
    )

    # For candidate `run`, fail-closed on baseline manifest BEFORE running live traffic!
    base_frames: list[bytes] | None = None
    base_manifest: dict[str, Any] | None = None
    if args.action == "run":
        base_frames, _, _, base_manifest = load_verified_baseline(
            args.baseline_dir,
            expected_width=832,
            expected_height=480,
            expected_case_id=args.case_id,
            expected_seed=args.seed,
            expected_inference_steps=effective_steps,
        )

    precheck: dict[str, Any] = {
        "all_roles_warmed_up": True,
        "roles": {},
        "capacity": {},
    }
    if args.mode == "real-sglang":
        precheck = check_role_warmup_and_pool_capacity(
            role_urls=role_urls,
            coordinator_url=coordinator_url,
            routing_revision=routing_revision,
            compatibility=compatibility,
            profile=profile,
            network_domain=network_domain,
            room_token=os.environ.get("WHITEBOX_ROOM_TOKEN"),
            require_warmup_roles=set(ALL_ROLES),
        )
        if not precheck["all_roles_warmed_up"]:
            if args.action == "run" and getattr(args, "state", None):
                failed_warmup_report = {
                    "schema": 1,
                    "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "deployment_id": deployment_id,
                    "mode": args.mode,
                    "case_id": args.case_id,
                    "seed": args.seed,
                    "traffic_accepted": False,
                    "warmup_verified": False,
                    "warmup_roles": precheck["roles"],
                    "capacity_precheck": precheck["capacity"],
                    "quality_unchanged": False,
                    "is_baseline": False,
                    "baseline_identity": base_manifest,
                    "intrinsic_quality": {"passed": False},
                    "baseline_comparison": None,
                }
                attach_verification_to_deployment(
                    args.state,
                    failed_warmup_report,
                    sync_oss=args.sync_oss,
                    local_history_dir=args.local_history_dir,
                    aliyun_profile=args.aliyun_profile,
                )
            raise RuntimeError(
                f"Warmup gate failed across required roles {ALL_ROLES}: {precheck}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_res, rgb_frames = asyncio.run(
        run_live_replay_capture(
            cases_json=args.cases_json,
            case_id=args.case_id,
            media_dir=args.media_dir,
            mode=args.mode,
            coordinator_url=coordinator_url,
            routing_revision=routing_revision,
            compatibility=compatibility,
            profile=profile,
            network_domain=network_domain,
            seed=args.seed,
            target_frames=args.target_frames,
            work_dir=args.output_dir / "work",
            inference_steps=effective_steps,
        )
    )
    frames_dir = args.output_dir / "frames"
    saved_pngs = save_frames_png(rgb_frames, frames_dir)
    traffic_accepted = bool(
        session_res.get("steady_state_flow", {}).get("received_model_frames", 0)
        >= args.target_frames
        and session_res.get("lease_release_verified") is True
    )

    if args.action == "record-baseline":
        quality = evaluate_visual_quality(
            rgb_frames,
            width=832,
            height=480,
            min_frames=args.target_frames,
            baseline_frames=None,
            baseline_identity=None,
        )
        intrinsic_ok = bool(quality["intrinsic_quality"]["passed"])
        manifest = None
        if traffic_accepted and precheck["all_roles_warmed_up"] and intrinsic_ok:
            manifest = write_baseline_manifest(
                args.output_dir,
                frames=rgb_frames,
                width=832,
                height=480,
                case_id=args.case_id,
                seed=args.seed,
                inference_steps=effective_steps,
                source_deployment_id=deployment_id,
                routing_revision=routing_revision,
            )
        record_out = {
            "schema": 1,
            "mode_kind": "baseline_recording",
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_deployment_id": deployment_id,
            "case_id": args.case_id,
            "seed": args.seed,
            "inference_steps": effective_steps,
            "traffic_accepted": traffic_accepted,
            "warmup_verified": precheck["all_roles_warmed_up"],
            "intrinsic_quality": quality["intrinsic_quality"],
            "baseline_manifest": manifest,
            "frames_dir": str(frames_dir),
            "saved_frame_count": len(saved_pngs),
            "elapsed_sec": round(time.monotonic() - t0, 3),
        }
        (args.output_dir / "baseline_record.json").write_text(
            json.dumps(record_out, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(record_out, sort_keys=True))
        return 0 if manifest is not None else 2

    quality = evaluate_visual_quality(
        rgb_frames,
        width=832,
        height=480,
        min_frames=args.target_frames,
        baseline_frames=base_frames,
        baseline_identity=base_manifest,
        min_psnr_db=args.min_psnr_db,
        min_ssim=args.min_ssim,
    )
    elapsed_s = round(time.monotonic() - t0, 3)
    verification = {
        "schema": 1,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deployment_id": deployment_id,
        "mode": args.mode,
        "case_id": args.case_id,
        "seed": args.seed,
        "inference_steps": effective_steps,
        "routing": {
            "coordinator_url": coordinator_url,
            "routing_revision": routing_revision,
            "compatibility": compatibility,
            "profile": profile,
            "network_domain": network_domain,
        },
        "traffic_accepted": traffic_accepted,
        "warmup_verified": precheck["all_roles_warmed_up"],
        "warmup_roles": precheck["roles"],
        "capacity_precheck": precheck["capacity"],
        "session_summary": {
            "whitebox_session_ttff_ms": session_res.get("whitebox_session_ttff_ms"),
            "prepare": session_res.get("prepare"),
            "steady_state_flow": session_res.get("steady_state_flow"),
            "lease_release_verified": session_res.get("lease_release_verified"),
            "negotiated_protocol": session_res.get("negotiated_protocol"),
        },
        "frames_dir": str(frames_dir),
        "saved_frame_count": len(saved_pngs),
        "elapsed_sec": elapsed_s,
        **quality,
    }
    out_file = args.output_dir / "traffic_verification.json"
    out_file.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    if args.state:
        attach_verification_to_deployment(
            args.state,
            verification,
            sync_oss=args.sync_oss,
            local_history_dir=args.local_history_dir,
            aliyun_profile=args.aliyun_profile,
        )
    compact_out = {k: v for k, v in verification.items() if k != "per_frame"}
    print(json.dumps(compact_out, sort_keys=True))
    if not (
        verification["traffic_accepted"]
        and verification["warmup_verified"]
        and verification["quality_unchanged"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
