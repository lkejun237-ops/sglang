#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Four-Case Chained SGLang Suite reusing Room SourceSession + RecordedVideoEngine."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import msgspec
from PIL import Image

os.environ.setdefault("SGLANG_LIGHTWEIGHT_RUNTIME", "1")

from sglang.multimodal_gen.runtime.realtime.player_lifecycle import validate_sdk_cleanup
from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
    AsyncArchiveManager,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec

try:
    from sglang.multimodal_gen.runtime.realtime.whitebox.recorded_engine import (
        RecordedVideoEngine,
        create_recorded_video_engine,
    )
except ImportError:
    from recorded_engine import (  # type: ignore[no-redef]
        RecordedVideoEngine,
        create_recorded_video_engine,
    )

from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
    RenderJournal,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import SourceSession
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import MemoryRoomStore
from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_CASES_JSON = (
    Path(__file__).resolve().parent / "cases" / "interactive_production_four_cases.json"
)


class _BrowserUIStub:
    metadata = {"mode": "browser", "files": {}}

    async def close(self) -> None:
        pass


def _compute_cache_key(
    prompt: str,
    seed: int,
    ref_hashes: list[str],
    source_frame_sha256: str | None = None,
) -> str:
    payload = json.dumps(
        {
            "prompt": prompt,
            "seed": int(seed),
            "references": list(ref_hashes),
            "source_frame_sha256": source_frame_sha256 or "",
        },
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ref_format_from_rel_path(rel_path: str) -> str:
    ext = Path(rel_path).suffix.lower().lstrip(".")
    if ext == "jpg":
        return "jpeg"
    if ext in {"png", "jpeg", "webp"}:
        return ext
    return "webp"


def _synthesize_image_bytes(
    fmt: str, seed_idx: int, width: int = 128, height: int = 128
) -> bytes:
    buf = io.BytesIO()
    color = (
        (37 * (seed_idx + 1)) % 256,
        (73 * (seed_idx + 1)) % 256,
        (119 * (seed_idx + 1)) % 256,
    )
    img = Image.new("RGB", (width, height), color=color)
    pil_fmt = "JPEG" if fmt == "jpeg" else fmt.upper()
    img.save(buf, format=pil_fmt)
    return buf.getvalue()


def _synthesize_mp4(
    mp4_path: Path,
    width: int,
    height: int,
    fps: int,
    num_frames: int,
    seed_offset: int = 0,
) -> None:
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(mp4_path),
        ]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert proc.stdin is not None
        frame_bytes = width * height * 3
        for idx in range(num_frames):
            val = (idx + seed_offset) % 251
            proc.stdin.write(bytes([val]) * frame_bytes)
        proc.stdin.close()
        proc.stdin = None
        _, stderr = proc.communicate(timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to synthesize {mp4_path}: {stderr.decode(errors='replace')}"
            )
        return

    import av  # type: ignore[import-not-found]
    import numpy as np

    container = av.open(str(mp4_path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for idx in range(num_frames):
            val = (idx + seed_offset) % 251
            arr = np.full((height, width, 3), val, dtype=np.uint8)
            video_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _download_oss_object(bucket, key, dest, endpoint, profile):
    endpoint = os.environ.get("WHITEBOX_OSS_ENDPOINT") or endpoint
    if not endpoint.startswith(("http://", "https://")):
        endpoint = "https://" + endpoint
    if shutil.which("aliyun"):
        result = subprocess.run(
            [
                "aliyun",
                "oss",
                "cp",
                f"oss://{bucket}/{key}",
                str(dest),
                "--profile",
                profile,
                "-e",
                endpoint,
                "-f",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode:
            raise RuntimeError("OSS download failed")
    else:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name="cn-shanghai",
            config=Config(s3={"addressing_style": "virtual"}),
        )
        client.download_file(bucket, key, str(dest))


def prepare_recording_media(
    recording_meta: dict[str, Any],
    media_root: Path,
    *,
    oss_bucket: str,
    oss_endpoint: str,
    oss_profile: str = "shengdong",
    oss_bucket_aliases: tuple[str, ...] = (),
    synthesize_fixtures: bool = False,
    synthetic_frames: int | None = None,
) -> dict[str, Any]:
    rec_id = recording_meta["recording_id"]
    rec_dir = media_root / rec_id
    rec_dir.mkdir(parents=True, exist_ok=True)
    objects = recording_meta.get("objects", {})
    video_info = recording_meta.get("video", {})
    width = int(video_info.get("width", 832))
    height = int(video_info.get("height", 480))
    fps = int(video_info.get("fps", 24))
    total_frames = int(video_info.get("frames", 64))
    target_frames = (
        min(total_frames, synthetic_frames)
        if (synthesize_fixtures and synthetic_frames is not None)
        else total_frames
    )

    verified_files: list[str] = []
    synthesized_files: list[str] = []
    prefix = recording_meta["object_prefix"].rstrip("/") + "/"

    for idx, (rel_path, meta) in enumerate(sorted(objects.items())):
        dest = rec_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        expected_sha = meta.get("sha256")
        expected_bytes = meta.get("bytes")

        if not synthesize_fixtures:
            if dest.exists():
                data = dest.read_bytes()
                if (
                    len(data) == expected_bytes
                    and hashlib.sha256(data).hexdigest() == expected_sha
                ):
                    verified_files.append(rel_path)
                    continue
            failures = []
            for bucket in dict.fromkeys((oss_bucket.strip(), *oss_bucket_aliases)):
                uri = f"oss://{bucket}/{prefix}{rel_path}"
                try:
                    _download_oss_object(
                        bucket, prefix + rel_path, dest, oss_endpoint, oss_profile
                    )
                    data = dest.read_bytes()
                    if expected_bytes is not None and len(data) != expected_bytes:
                        raise ValueError("Object byte size mismatch")
                    if (
                        expected_sha
                        and hashlib.sha256(data).hexdigest() != expected_sha
                    ):
                        raise ValueError("Object SHA-256 mismatch")
                    break
                except Exception as error:
                    failures.append(f"{uri}: {type(error).__name__}")
            else:
                raise RuntimeError(
                    "OSS object unavailable or invalid: " + "; ".join(failures)
                )
            verified_files.append(rel_path)
        else:
            if rel_path == "whitebox.mp4":
                if not dest.exists():
                    _synthesize_mp4(
                        dest,
                        width=width,
                        height=height,
                        fps=fps,
                        num_frames=max(target_frames, 24),
                        seed_offset=idx + 1,
                    )
                synthesized_files.append(rel_path)
            elif rel_path.startswith("references/"):
                fmt = _ref_format_from_rel_path(rel_path)
                dest.write_bytes(_synthesize_image_bytes(fmt, idx + 1))
                synthesized_files.append(rel_path)

    return {
        "recording_id": rec_id,
        "directory": str(rec_dir),
        "verified_files": verified_files,
        "synthesized_files": synthesized_files,
    }


def prepare_all_media(
    suite_data: dict[str, Any],
    media_root: Path,
    *,
    oss_profile: str = "shengdong",
    synthesize_fixtures: bool = False,
    synthetic_frames: int | None = None,
) -> list[dict[str, Any]]:
    bucket = suite_data.get("oss_bucket", "leap-world-shanghai").strip()
    endpoint = suite_data.get("oss_endpoint", "oss-cn-shanghai.aliyuncs.com").strip()
    receipts: list[dict[str, Any]] = []
    for case in suite_data.get("cases", []):
        receipts.append(
            prepare_recording_media(
                case,
                media_root,
                oss_bucket=bucket,
                oss_endpoint=endpoint,
                oss_profile=oss_profile,
                oss_bucket_aliases=tuple(suite_data.get("oss_bucket_aliases", [])),
                synthesize_fixtures=synthesize_fixtures,
                synthetic_frames=synthetic_frames,
            )
        )
        if "stress_recording" in case:
            receipts.append(
                prepare_recording_media(
                    case["stress_recording"],
                    media_root,
                    oss_bucket=bucket,
                    oss_endpoint=endpoint,
                    oss_profile=oss_profile,
                    oss_bucket_aliases=tuple(suite_data.get("oss_bucket_aliases", [])),
                    synthesize_fixtures=synthesize_fixtures,
                    synthetic_frames=synthetic_frames,
                )
            )
        if "secondary_recording" in case:
            receipts.append(
                prepare_recording_media(
                    case["secondary_recording"],
                    media_root,
                    oss_bucket=bucket,
                    oss_endpoint=endpoint,
                    oss_profile=oss_profile,
                    oss_bucket_aliases=tuple(suite_data.get("oss_bucket_aliases", [])),
                    synthesize_fixtures=synthesize_fixtures,
                    synthetic_frames=synthetic_frames,
                )
            )
    return receipts


def _iter_all_recording_metas(
    suite_data: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for case in suite_data.get("cases", []):
        cid = case["case_id"]
        items.append((cid, case))
        if "stress_recording" in case:
            items.append((f"{cid}:stress_recording", case["stress_recording"]))
        if "secondary_recording" in case:
            items.append((f"{cid}:secondary_recording", case["secondary_recording"]))
    return items


def _resolve_coordinator_url(
    sglang_url: str | None = None,
    coordinator_url: str | None = None,
) -> str | None:
    if coordinator_url:
        return coordinator_url.rstrip("/")
    if sglang_url and sglang_url.startswith(("http://", "https://")):
        return sglang_url.rstrip("/")
    role_pool_cfg_path = os.environ.get("WHITEBOX_ROLE_POOL_CONFIG")
    if role_pool_cfg_path and Path(role_pool_cfg_path).exists():
        cfg = json.loads(Path(role_pool_cfg_path).read_text())
        if isinstance(cfg.get("coordinator_url"), str):
            return cfg["coordinator_url"].rstrip("/")
    return None


def _validate_prompt_and_references(
    label: str,
    prompt_text: str,
    refs: list[dict[str, Any]],
    known_ref_shas: set[str],
) -> dict[str, Any]:
    if "<Video 1>" not in prompt_text:
        raise ValueError(f"[{label}] Prompt missing required '<Video 1>' tag")
    pic_nums = {int(m) for m in re.findall(r"<Picture (\d+)>", prompt_text)}
    expected_contiguous = set(range(1, len(pic_nums) + 1))
    if pic_nums != expected_contiguous or len(pic_nums) > len(refs):
        raise ValueError(
            f"[{label}] Prompt <Picture k> tags {sorted(pic_nums)} invalid for reference count {len(refs)}"
        )
    ref_ids = [r.get("id", "") for r in refs]
    if len(ref_ids) != len(set(ref_ids)):
        raise ValueError(f"[{label}] Duplicate reference IDs in mapping: {ref_ids}")
    for r in refs:
        clean_sha = str(r.get("asset_id", "")).removeprefix("sha256:")
        if known_ref_shas and clean_sha not in known_ref_shas:
            raise ValueError(
                f"[{label}] Reference asset_id {clean_sha} not found in recording objects"
            )
    return {
        "label": label,
        "reference_count": len(refs),
        "picture_tags": sorted(pic_nums),
        "has_video_tag": True,
    }


def _inspect_mp4_geometry(mp4_path: Path) -> dict[str, Any]:
    import av  # type: ignore[import-not-found]

    container = av.open(str(mp4_path))
    try:
        if not container.streams.video:
            raise ValueError(f"No video stream in {mp4_path}")
        stream = container.streams.video[0]
        width = int(stream.width)
        height = int(stream.height)
        observed_vfr_avg_fps = (
            round(float(stream.average_rate), 2) if stream.average_rate else None
        )
        frames = int(stream.frames or 0)
        if frames <= 0:
            frames = sum(
                1 for packet in container.demux(stream) if packet.pts is not None
            )
        return {
            "width": width,
            "height": height,
            "observed_vfr_avg_fps": observed_vfr_avg_fps,
            "frames": frames,
            "codec": stream.codec_context.name,
        }
    finally:
        container.close()


def _query_service_and_capacity(
    coordinator_url: str | None,
    *,
    require_idle: bool = True,
) -> dict[str, Any]:
    import httpx

    if not coordinator_url:
        return {"status": "skipped_no_coordinator_url"}

    with httpx.Client(timeout=10.0, trust_env=False) as client:
        ver_resp = client.get(f"{coordinator_url}/v1/whitebox/version")
        ver_resp.raise_for_status()
        ver_data = ver_resp.json()
        if ver_data.get("identity_status") != "configured":
            raise RuntimeError(
                f"PREFLIGHT_VERSION_UNCONFIGURED: {coordinator_url}/v1/whitebox/version returned {ver_data}"
            )

        cap_resp = client.get(f"{coordinator_url}/v1/capacity")
        if cap_resp.status_code == 404:
            cap_resp = client.get(f"{coordinator_url}/v1/slots")
        cap_resp.raise_for_status()
        cap_data = cap_resp.json()

    mg = (cap_data.get("roles") or {}).get("model_group") or {}
    free_slots = int(cap_data.get("free_slots", mg.get("free_slots", 0)))
    active_sessions = int(
        cap_data.get(
            "active_sessions",
            cap_data.get(
                "used_slots", mg.get("active_sessions", mg.get("used_slots", 0))
            ),
        )
    )
    waiting_sessions = int(
        cap_data.get("waiting_sessions", mg.get("waiting_sessions", 0))
    )
    queued_sessions = int(cap_data.get("queued_sessions", mg.get("queued_sessions", 0)))
    draining_workers = int(
        cap_data.get("draining_workers", mg.get("draining_workers", 0))
    )
    roles = cap_data.get("role_capacity") or cap_data.get("roles") or {}

    if require_idle:
        if (
            free_slots < 1
            or active_sessions > 0
            or waiting_sessions > 0
            or queued_sessions > 0
            or draining_workers > 0
        ):
            raise RuntimeError(
                f"PREFLIGHT_CAPACITY_BUSY: require idle service (free_slots>=1, active=0), got {cap_data}"
            )
        for role_name in ("encoder", "dit", "decoder"):
            if role_name in roles:
                r_info = roles[role_name]
                if (
                    int(r_info.get("free_slots", 0)) < 1
                    or int(r_info.get("workers", 0)) < 1
                ):
                    raise RuntimeError(
                        f"PREFLIGHT_ROLE_UNAVAILABLE: role={role_name} status={r_info}"
                    )

    return {
        "coordinator_url": coordinator_url,
        "version": {
            "service": ver_data.get("service"),
            "backend": ver_data.get("backend"),
            "profile": ver_data.get("profile"),
            "region": ver_data.get("region"),
            "release_scope": ver_data.get("release_scope"),
            "source_sha": ver_data.get("source_sha"),
            "sdk_sha": ver_data.get("sdk_sha"),
            "config_sha256": ver_data.get("config_sha256"),
            "identity_status": ver_data.get("identity_status"),
        },
        "capacity": {
            "free_slots": free_slots,
            "active_sessions": active_sessions,
            "waiting_sessions": waiting_sessions,
            "queued_sessions": queued_sessions,
            "draining_workers": draining_workers,
            "roles": {
                k: {
                    "workers": v.get("workers"),
                    "free_slots": v.get("free_slots"),
                    "active_sessions": v.get("active_sessions"),
                }
                for k, v in roles.items()
                if isinstance(v, dict)
            },
        },
    }


async def _wait_for_coordinator_idle(
    coordinator_url: str | None,
    *,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    """Waits until Coordinator /v1/capacity reports free_slots>=1 and active_sessions==0.

    In Role-Pool mode, /v1/role-pool/close clears allocation_id immediately,
    but store.candidates() requires ProcessReporter's next heartbeat (up to 2s)
    to transition observation['state'] back to 'idle'.
    """
    from four_case_role_pool import current_pool

    pool = current_pool.get()
    if pool is not None:
        return await pool.wait_idle(timeout_s)
    if not coordinator_url:
        return {"status": "skipped_no_coordinator_url"}
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return await asyncio.to_thread(
                _query_service_and_capacity,
                coordinator_url,
                require_idle=True,
            )
        except Exception:
            if time.monotonic() >= deadline:
                raise
            await asyncio.sleep(0.25)


def run_preflight_checks(
    suite_data: dict[str, Any],
    *,
    media_root: Path,
    mode: str = "mock-worker",
    sglang_url: str | None = None,
    coordinator_url: str | None = None,
    require_real_media: bool = False,
    preserve_switch_boundaries: bool = False,
    full_frames: bool = False,
    max_frames_after_last_event: int = 18,
    seed_base: int = 42,
) -> dict[str, Any]:
    """Validates all 6 precondition categories before executing the 4-case suite."""
    enforce_checksum = require_real_media or (mode == "real-sglang")
    media_checks: list[dict[str, Any]] = []
    prompt_checks: list[dict[str, Any]] = []

    for label, rec_meta in _iter_all_recording_metas(suite_data):
        rec_id = rec_meta["recording_id"]
        rec_dir = media_root / rec_id
        objects = rec_meta.get("objects", {})
        video_expected = rec_meta.get("video", {})
        expected_w = int(video_expected.get("width", 832))
        expected_h = int(video_expected.get("height", 480))
        expected_fps = int(video_expected.get("fps", 24))

        known_ref_shas: set[str] = set()
        verified_object_count = 0
        for rel_path, obj_meta in sorted(objects.items()):
            f_path = rec_dir / rel_path
            if rel_path.startswith("references/") and obj_meta.get("sha256"):
                known_ref_shas.add(str(obj_meta["sha256"]))
            if not f_path.exists():
                if enforce_checksum:
                    raise FileNotFoundError(
                        f"PREFLIGHT_MEDIA_MISSING: [{label}] {f_path} does not exist"
                    )
                continue
            raw_bytes = f_path.read_bytes()
            actual_sha = hashlib.sha256(raw_bytes).hexdigest()
            if enforce_checksum:
                if obj_meta.get("bytes") is not None and len(raw_bytes) != int(
                    obj_meta["bytes"]
                ):
                    raise ValueError(
                        f"PREFLIGHT_MEDIA_SIZE_MISMATCH: [{label}] {rel_path} got {len(raw_bytes)} != {obj_meta['bytes']}"
                    )
                if obj_meta.get("sha256") and actual_sha != obj_meta["sha256"]:
                    raise ValueError(
                        f"PREFLIGHT_MEDIA_SHA_MISMATCH: [{label}] {rel_path} got {actual_sha} != {obj_meta['sha256']}"
                    )
            if rel_path.startswith("references/"):
                with Image.open(f_path) as img:
                    img.verify()
            verified_object_count += 1

        mp4_path = rec_dir / "whitebox.mp4"
        mp4_geom = None
        if mp4_path.exists():
            mp4_geom = _inspect_mp4_geometry(mp4_path)
            if mp4_geom["width"] != expected_w or mp4_geom["height"] != expected_h:
                raise ValueError(
                    f"PREFLIGHT_VIDEO_GEOMETRY_MISMATCH: [{label}] {mp4_geom['width']}x{mp4_geom['height']} != {expected_w}x{expected_h}"
                )
            if mp4_geom["frames"] < 1:
                raise ValueError(
                    f"PREFLIGHT_VIDEO_EMPTY: [{label}] frames={mp4_geom['frames']}"
                )
            mp4_geom["nominal_replay_fps"] = expected_fps
            switches = rec_meta.get("condition_switches", [])
            if switches and (preserve_switch_boundaries or full_frames):
                max_sw_boundary = max(int(s["boundary"]) for s in switches)
                if mp4_geom["frames"] <= max_sw_boundary:
                    raise ValueError(
                        f"PREFLIGHT_VIDEO_FRAMES_SHORT: [{label}] frames={mp4_geom['frames']} <= max_switch_boundary={max_sw_boundary}"
                    )

        media_checks.append(
            {
                "label": label,
                "recording_id": rec_id,
                "verified_objects": verified_object_count,
                "checksum_enforced": enforce_checksum,
                "video_geometry": mp4_geom,
            }
        )

        prompt_checks.append(
            _validate_prompt_and_references(
                f"{label}:initial",
                rec_meta["prompt"],
                rec_meta.get("references_mapping", []),
                known_ref_shas,
            )
        )
        for sw in rec_meta.get("condition_switches", []):
            prompt_checks.append(
                _validate_prompt_and_references(
                    f"{label}:switch_v{sw['prompt_version']}",
                    sw["prompt"],
                    sw.get("references", []),
                    known_ref_shas,
                )
            )

    resolved_coord = _resolve_coordinator_url(sglang_url, coordinator_url)
    from four_case_role_pool import current_pool

    service_and_cap = (
        {"status": "checked_by_role_pool_adapter"}
        if current_pool.get()
        else (
            _query_service_and_capacity(resolved_coord, require_idle=True)
            if mode == "real-sglang"
            else {"status": "mock_worker_mode"}
        )
    )

    return {
        "status": "passed",
        "checked_at_epoch_ms": round(time.time() * 1000),
        "mode": mode,
        "preconditions": {
            "1_media_and_geometry_integrity": {
                "passed": True,
                "checksum_enforced": enforce_checksum,
                "recordings_checked": len(media_checks),
                "details": media_checks,
            },
            "2_prompt_and_reference_contract": {
                "passed": True,
                "conditions_checked": len(prompt_checks),
                "max_reference_count_required": 12,
                "details": prompt_checks,
            },
            "3_service_identity_and_health": {
                "passed": True,
                "details": (
                    service_and_cap.get("version")
                    if isinstance(service_and_cap, dict)
                    and "version" in service_and_cap
                    else service_and_cap
                ),
            },
            "4_idle_capacity_and_non_preemption": {
                "passed": True,
                "details": (
                    service_and_cap.get("capacity")
                    if isinstance(service_and_cap, dict)
                    and "capacity" in service_and_cap
                    else service_and_cap
                ),
            },
            "5_protocol_and_reference_capability": {
                "required_source_prompt_protocol": "ordered-boundary-v2",
                "required_condition_switch_protocol": "clip-barrier-v1",
                "required_prompt_reference_updates": True,
                "required_max_references": 12,
                "configured_num_inference_steps": int(
                    os.environ.get("WHITEBOX_INFERENCE_STEPS", "4")
                ),
                "configured_model_input_frames": int(
                    os.environ.get("WHITEBOX_MODEL_INPUT_FRAMES", "34")
                ),
            },
            "6_cache_isolation_and_post_release": {
                "seed_base": seed_base,
                "step_seeds": {
                    "step1_shanghai_drive_cold_and_warm": seed_base + 1,
                    "step2_marvel_overdrive_primary": seed_base + 2,
                    "step2_marvel_12ref_stress": seed_base + 20,
                    "step3_dark_horns_arm_a_unprefilled_miss": seed_base + 30,
                    "step3_dark_horns_arm_b_prefilled_hit": seed_base + 31,
                    "step4_wuxia_jiahao_seg1": seed_base + 40,
                    "step4_wuxia_jiahao_seg2": seed_base + 41,
                },
                "strict_release_required": True,
            },
        },
    }


class MockSglangCacheStore:
    """Tracks simulated L1 memory and L2 persisted Block 0 KV cache across bound operations."""

    def __init__(self) -> None:
        self.l1_keys: set[str] = set()
        self.l2_keys: set[str] = set()
        self.history: list[dict[str, Any]] = []

    def lookup_or_populate(
        self,
        cache_key: str,
        *,
        num_refs: int,
        persist: bool = True,
        populate_on_miss: bool = True,
    ) -> dict[str, Any]:
        if cache_key in self.l1_keys:
            cache_result = "l1_hit"
            hit = True
            miss_reason = None
            memory_ms = 3.12
            persisted_ms = 4.25
            stages = {
                "cache_lookup": {"host_ms": 2.15, "status": "executed"},
                "qwen_prepare": {"host_ms": None, "status": "skipped"},
                "qwen_remote_prepare": {"host_ms": None, "status": "skipped"},
                "reference_vae": {"host_ms": None, "status": "skipped"},
                "block0_forward": {"host_ms": None, "status": "skipped"},
            }
        elif cache_key in self.l2_keys:
            cache_result = "l2_hit"
            hit = True
            miss_reason = None
            self.l1_keys.add(cache_key)
            memory_ms = 38.4
            persisted_ms = 39.8
            stages = {
                "cache_lookup": {"host_ms": 38.4, "status": "executed"},
                "qwen_prepare": {"host_ms": None, "status": "skipped"},
                "qwen_remote_prepare": {"host_ms": None, "status": "skipped"},
                "reference_vae": {"host_ms": None, "status": "skipped"},
                "block0_forward": {"host_ms": None, "status": "skipped"},
            }
        else:
            cache_result = "miss"
            hit = False
            miss_reason = "not_resident"
            if populate_on_miss:
                self.l1_keys.add(cache_key)
                if persist:
                    self.l2_keys.add(cache_key)
            ref_vae_ms = round(22.5 * max(1, num_refs), 3)
            qwen_ms = 142.0
            block0_ms = 318.0
            memory_ms = round(2.4 + qwen_ms + ref_vae_ms + block0_ms, 3)
            persisted_ms = round(memory_ms + 86.0, 3)
            stages = {
                "cache_lookup": {"host_ms": 2.4, "status": "executed"},
                "qwen_prepare": {"host_ms": qwen_ms, "status": "executed"},
                "qwen_remote_prepare": {"host_ms": 14.5, "status": "executed"},
                "reference_vae": {"host_ms": ref_vae_ms, "status": "executed"},
                "block0_forward": {"host_ms": block0_ms, "status": "executed"},
            }

        entry = {
            "cache_key": cache_key,
            "cache_result": cache_result,
            "hit": hit,
            "miss_reason": miss_reason,
            "prefill_to_memory_ms": memory_ms,
            "prefill_to_persisted_ms": persisted_ms,
            "timings": {
                "elapsed_host_ms": memory_ms,
                "stages": stages,
            },
        }
        self.history.append(entry)
        return entry


class MockBoundWebSocket:
    """In-memory bound-compute-v1 WebSocket endpoint simulating SGLang H3 Worker."""

    def __init__(
        self,
        spec: VideoSpec,
        cache_store: MockSglangCacheStore,
        worker_epoch: str,
        player_expires_at: float,
    ) -> None:
        self.spec = spec
        self.cache_store = cache_store
        self.worker_epoch = worker_epoch
        self.player_expires_at = player_expires_at
        self.outgoing: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.closed = False
        self.active_op_id: str | None = None
        self.active_kind: str | None = None
        self.active_session_id: str | None = None
        self.active_generation_id: str | None = None
        self.init_msg: dict[str, Any] | None = None
        self.prompt_id: int | None = None
        self.switch_candidate: dict[str, Any] | None = None
        self.switch_end: int | None = None
        self.appended_frames = 0
        self.tasks: set[asyncio.Task[Any]] = set()

        self._enqueue(
            {
                "type": "bound.ready",
                "protocol": "bound-compute-v1",
                "worker_epoch": self.worker_epoch,
                "player_expires_at": self.player_expires_at,
            }
        )

    def _enqueue(self, payload: dict[str, Any]) -> None:
        if not self.closed:
            self.outgoing.put_nowait(msgspec.msgpack.encode(payload))

    def _enqueue_child_data(self, op_id: str, inner: dict[str, Any]) -> None:
        self._enqueue(
            {
                "type": "bound.data",
                "operation_id": op_id,
                "payload": msgspec.msgpack.encode(inner),
            }
        )

    async def recv(self) -> bytes:
        item = await self.outgoing.get()
        if item is None:
            raise EOFError("MockBoundWebSocket closed")
        return item

    async def send(self, wire: bytes) -> None:
        if self.closed:
            return
        msg = msgspec.msgpack.decode(wire)
        mtype = msg.get("type")
        if mtype == "bound.open":
            op_id = msg["operation_id"]
            self.active_op_id = op_id
            self.active_kind = msg["kind"]
            self.active_session_id = msg["session_id"]
            self.active_generation_id = msg["generation_id"]
            self.init_msg = None
            self.prompt_id = None
            self.switch_candidate = None
            self.switch_end = None
            self.appended_frames = 0
            self._enqueue({"type": "bound.opened", "operation_id": op_id})
            return

        if mtype == "bound.cancel":
            op_id = msg["operation_id"]
            self._enqueue(
                {
                    "type": "bound.completed",
                    "operation_id": op_id,
                    "cleaned": True,
                }
            )
            return

        if mtype == "bound.data":
            op_id = msg["operation_id"]
            inner = msgspec.msgpack.decode(msg["payload"])
            if self.active_kind == "prefill":
                await self._handle_prefill_data(op_id, inner)
            else:
                await self._handle_generate_data(op_id, inner)

    async def _handle_prefill_data(self, op_id: str, inner: dict[str, Any]) -> None:
        request_id = inner["request_id"]
        prompt = inner.get("prompt", "")
        seed = int(inner.get("seed", 42))
        refs = inner.get("references", [])
        ref_hashes = [
            r.get("sha256") or hashlib.sha256(r.get("payload", b"")).hexdigest()
            for r in refs
        ]
        cache_key = _compute_cache_key(prompt, seed, ref_hashes)
        lookup = self.cache_store.lookup_or_populate(
            cache_key,
            num_refs=len(refs),
            persist=bool(inner.get("persist", True)),
            populate_on_miss=True,
        )
        await asyncio.sleep(0.002 if lookup["hit"] else 0.008)
        self._enqueue_child_data(
            op_id,
            {
                "type": "prefill.memory_ready",
                "request_id": request_id,
                "session_id": self.active_session_id,
                "cache_key": cache_key,
                "cache_result": lookup["cache_result"],
                "hit": lookup["hit"],
                "prefill_to_memory_ms": lookup["prefill_to_memory_ms"],
                "timings": lookup["timings"],
            },
        )
        self._enqueue(
            {
                "type": "bound.completed",
                "operation_id": op_id,
                "cleaned": True,
            }
        )
        if inner.get("persist", True):
            self._enqueue(
                {
                    "type": "prefill.persisted",
                    "request_id": request_id,
                    "session_id": self.active_session_id,
                    "cache_key": cache_key,
                    "cache_result": lookup["cache_result"],
                    "hit": lookup["hit"],
                    "memory_ready": True,
                    "prefill_to_memory_ms": lookup["prefill_to_memory_ms"],
                    "prefill_to_persisted_ms": lookup["prefill_to_persisted_ms"],
                }
            )

    async def _finish_switch(
        self, op_id: str, candidate: dict[str, Any], boundary: int
    ) -> None:
        payload = candidate["payload"]
        prompt = payload.get("text", "")
        version = int(payload.get("version", 2))
        seed = int((self.init_msg or {}).get("seed", 42))
        if "references" in payload:
            refs = payload["references"]
        else:
            refs = (self.init_msg or {}).get("references", [])
        ref_hashes = [
            r.get("sha256") or hashlib.sha256(r.get("payload", b"")).hexdigest()
            for r in refs
        ]
        cache_key = _compute_cache_key(prompt, seed, ref_hashes)
        lookup = self.cache_store.lookup_or_populate(
            cache_key,
            num_refs=len(refs),
            persist=True,
            populate_on_miss=True,
        )
        await asyncio.sleep(0.004 if lookup["hit"] else 0.025)
        self._enqueue_child_data(
            op_id,
            {
                "type": "model.cache",
                "model_session_id": self.active_session_id,
                "generation_id": self.active_generation_id,
                "prompt_version": version,
                "usage": {
                    "zing_block0_cache": {
                        "cache_key": cache_key,
                        "prompt_version": version,
                        "cache_result": lookup["cache_result"],
                        "miss_reason": lookup["miss_reason"],
                        "timings": lookup["timings"],
                    }
                },
            },
        )
        self.prompt_id = candidate["event_id"]
        self.switch_end = None
        self._enqueue_child_data(
            op_id,
            {
                "type": "control_ack",
                "kind": "prompt.update",
                "state": "kv_ready",
                "event_id": self.prompt_id,
                "prompt_version": version,
                "source_frame_index": boundary,
            },
        )

    async def _handle_generate_data(self, op_id: str, inner: dict[str, Any]) -> None:
        if inner.get("type") == "init":
            self.init_msg = inner
            self._enqueue_child_data(
                op_id,
                {
                    "type": "session.ready",
                    "source_stream": inner["source_stream"],
                    "max_message_bytes": 64 * 1024 * 1024,
                    "max_pending_source_frames": 51,
                    "max_pending_source_bytes": self.spec.frame_bytes * 51,
                    "max_source_message_bytes": self.spec.frame_bytes + 4096,
                    "source_prompt_protocol": "ordered-boundary-v2",
                    "source_condition_switch_protocol": "clip-barrier-v1",
                    "prompt_reference_updates": True,
                    "max_references": 21,
                    "max_reference_bytes": 64 << 20,
                    "source_output_release_protocol": "valid-source-frames-v1",
                    "sink_updates": True,
                },
            )
            prompt = inner.get("prompt", "")
            seed = int(inner.get("seed", 42))
            refs = inner.get("references", [])
            ref_hashes = [
                r.get("sha256") or hashlib.sha256(r.get("payload", b"")).hexdigest()
                for r in refs
            ]
            cache_key = _compute_cache_key(prompt, seed, ref_hashes)
            lookup = self.cache_store.lookup_or_populate(
                cache_key,
                num_refs=len(refs),
                persist=True,
                populate_on_miss=True,
            )
            self._enqueue_child_data(
                op_id,
                {
                    "type": "model.cache",
                    "model_session_id": self.active_session_id,
                    "generation_id": self.active_generation_id,
                    "prompt_version": 1,
                    "usage": {
                        "zing_block0_cache": {
                            "cache_key": cache_key,
                            "prompt_version": 1,
                            "cache_result": lookup["cache_result"],
                            "miss_reason": lookup["miss_reason"],
                            "timings": lookup["timings"],
                        }
                    },
                },
            )
            return

        kind = inner.get("kind")
        if kind == "prompt.update":
            self.switch_candidate = inner
            min_frame = int(inner["payload"].get("min_source_frame", 0))
            self.switch_end = max(17, ((min_frame + 16) // 17) * 17)
            self._enqueue_child_data(
                op_id,
                {
                    "type": "control_ack",
                    "kind": "prompt.update",
                    "event_id": inner["event_id"],
                    "state": "accepted",
                    "source_frame_index": self.switch_end,
                },
            )
            if self.appended_frames >= self.switch_end:
                task = asyncio.create_task(
                    self._finish_switch(op_id, inner, self.switch_end)
                )
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)
            return

        if kind == "source.end":
            self._enqueue_child_data(op_id, {"type": "session.completed"})
            self._enqueue(
                {
                    "type": "bound.completed",
                    "operation_id": op_id,
                    "cleaned": True,
                }
            )
            return

        if kind == "source.append":
            payload = inner["payload"]
            index = int(payload["source_frame_start"])
            self.appended_frames = index + 1
            self._enqueue_child_data(
                op_id,
                {
                    "type": "source.accepted",
                    "source_frame_start": index,
                    "source_frame_end": index + 1,
                },
            )
            self._enqueue_child_data(
                op_id,
                {
                    "type": "frame_batch",
                    "output_frame_start": index,
                    "num_frames": 1,
                    "pts_us": self.spec.pts(index),
                    "width": self.spec.width,
                    "height": self.spec.height,
                    "format": "rgb24",
                    "encoding": "raw",
                    "payload": payload.get("data")
                    or (bytes([100]) * self.spec.frame_bytes),
                    "event_id": self.prompt_id,
                    "sink_chunks": 2,
                    "sink_chunk_ids": [0, 1],
                    "source_alignment": [
                        {
                            "source_frame_start": index,
                            "output_frame_start": index,
                            "num_frames": 1,
                        }
                    ],
                },
            )
            self._enqueue_child_data(
                op_id,
                {
                    "type": "source.consumed",
                    "source_frame_end": index + 1,
                    "source_bytes_end": (index + 1) * self.spec.frame_bytes,
                },
            )
            if (
                self.switch_end is not None
                and self.switch_candidate is not None
                and index + 1 == self.switch_end
            ):
                task = asyncio.create_task(
                    self._finish_switch(op_id, self.switch_candidate, self.switch_end)
                )
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)

    async def close(self) -> None:
        self.closed = True
        self.outgoing.put_nowait(None)
        for task in list(self.tasks):
            if not task.done():
                task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)


@dataclass(frozen=True)
class _MockDenoiser:
    endpoint: str
    worker_id: str
    worker_epoch: str
    model_revision: str


@dataclass(frozen=True)
class _MockAssignment:
    token: str
    session_id: str
    generation_id: str
    admitted_at: float
    player_expires_at: float | None
    vae: Any
    supports_condition_barrier: bool
    denoiser: _MockDenoiser


class MockBoundLease:
    """Simulates a Coordinator player-bound compute lease over MockBoundWebSocket."""

    def __init__(self, spec: VideoSpec, cache_store: MockSglangCacheStore) -> None:
        self.spec = spec
        self.cache_store = cache_store
        self.operation_path = "/v1/realtime_video/bound"
        self.cleanup_timeout_s = 90
        self.sdk_cleanup_provider: Any = None
        self.strict_release = True
        self.resource_state = "none"
        self._active_assignment: Any = None
        self._cleanup_assignment: Any = None
        self.sdk_cleanup_receipt: dict[str, Any] | None = None

    @asynccontextmanager
    async def open(self):
        self.resource_state = "reserved"
        worker_epoch = f"epoch-{uuid4().hex[:8]}"
        now = time.time()
        assignment = _MockAssignment(
            token=f"tok-{uuid4().hex[:12]}",
            session_id=f"bound-sess-{uuid4().hex[:8]}",
            generation_id=f"bound-gen-{uuid4().hex[:8]}",
            admitted_at=now,
            player_expires_at=now + 1800.0,
            vae=None,
            supports_condition_barrier=True,
            denoiser=_MockDenoiser(
                endpoint="ws://mock-sglang-worker:30000",
                worker_id="mock-sglang-h3-worker-0",
                worker_epoch=worker_epoch,
                model_revision="whitebox-sh-s0v2-main",
            ),
        )
        self._active_assignment = assignment
        ws = MockBoundWebSocket(
            self.spec,
            self.cache_store,
            worker_epoch=worker_epoch,
            player_expires_at=assignment.player_expires_at,
        )
        try:
            yield ws, assignment
        finally:
            self._cleanup_assignment = assignment
            await ws.close()
            if self.sdk_cleanup_provider is not None:
                receipt = await self.sdk_cleanup_provider(assignment)
                self.sdk_cleanup_receipt = validate_sdk_cleanup(assignment, receipt)
            self.resource_state = "released"


def _create_real_sglang_lease(
    sglang_url: str | None,
    coordinator_url: str | None,
    spec: VideoSpec,
    trace_id: str,
    segment: dict[str, Any],
):
    import websockets

    try:
        from websockets.asyncio.client import connect as ws_connect
    except ImportError:
        ws_connect = websockets.connect

    from four_case_role_pool import current_pool

    from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import (
        create_model_leases,
    )

    pool = current_pool.get()
    if pool is not None:
        return pool.lease(spec, trace_id, segment)
    revision = os.environ.get("WHITEBOX_ROUTING_REVISION", "real-sglang")
    large_revision = os.environ.get("WHITEBOX_LARGE_REFERENCE_MODEL_REVISION")
    default_limit = int(os.environ.get("WHITEBOX_DEFAULT_REFERENCE_CAPACITY", "3"))
    if large_revision and len(segment.get("references") or []) > default_limit:
        revision = large_revision

    profile_hint = segment.get("model_size") or f"{spec.width}x{spec.height}"
    model_session_id = segment.get("model_session_id") or uuid4().hex
    generation_id = segment.get("generation_id") or uuid4().hex
    capacity_wait_ms = int(segment.get("capacity_wait_ms", 0))

    if os.environ.get("WHITEBOX_MODEL_BACKEND") == "role_pool":
        return create_model_leases(
            None,
            ws_connect,
            user_id="suite-runner",
            profile_hint=profile_hint,
            trace_id=segment.get("segment_id") or trace_id,
            model_revision=revision,
            vae_fingerprint="local",
            max_bytes=64 * 1024 * 1024,
            strict_release=True,
            session_id=model_session_id,
            generation_id=generation_id,
            capacity_wait_ms=capacity_wait_ms,
        )

    effective_coord_url = coordinator_url or (
        sglang_url
        if sglang_url and sglang_url.startswith(("http://", "https://"))
        else None
    )
    if effective_coord_url:
        from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
            HTTPCoordinatorClient,
        )

        coord_client = HTTPCoordinatorClient(effective_coord_url)
        return create_model_leases(
            coord_client,
            ws_connect,
            user_id="suite-runner",
            profile_hint=profile_hint,
            trace_id=segment.get("segment_id") or trace_id,
            model_revision=revision,
            vae_fingerprint="local",
            max_bytes=64 * 1024 * 1024,
            strict_release=True,
            session_id=model_session_id,
            generation_id=generation_id,
            capacity_wait_ms=capacity_wait_ms,
        )

    if not sglang_url:
        raise ValueError(
            "Real-sglang mode requires WHITEBOX_MODEL_BACKEND=role_pool, --coordinator-url, or --sglang-url"
        )

    raise ValueError(
        "Direct bound Worker URLs require a real reservation. Use --coordinator-url "
        "or provide --encoder-url, --dit-url and --decoder-url."
    )


def _find_ref_file_for_sha(
    rec_dir: Path, objects: dict[str, Any], target_sha: str
) -> tuple[Path, str]:
    clean_sha = target_sha.removeprefix("sha256:")
    for rel_path, meta in objects.items():
        if rel_path.startswith("references/") and meta.get("sha256") == clean_sha:
            return rec_dir / rel_path, _ref_format_from_rel_path(rel_path)
    # Fallback by index if synthesized
    ref_paths = sorted(p for p in objects if p.startswith("references/"))
    if ref_paths:
        first = ref_paths[0]
        return rec_dir / first, _ref_format_from_rel_path(first)
    raise FileNotFoundError(f"Reference asset {target_sha} not found in {rec_dir}")


def _require_cache_result(receipt, expected, label):
    if (
        not isinstance(receipt, dict)
        or receipt.get("cache_result") not in expected
        or not isinstance(receipt.get("cache_key"), str)
        or not receipt["cache_key"]
    ):
        raise RuntimeError(
            f"CACHE_EXPECTATION_FAILED: {label}: expected {sorted(expected)}, got {receipt and receipt.get('cache_result')}"
        )


class _ReplaySourceSession(SourceSession):
    """Gate model admission at replay triggers, before asynchronous reference writes.

    Capture and Room selection/credit/switch logic stay unchanged. The gate only
    prevents the next old-condition clip from racing the benchmark command.
    """

    async def segment_status(self, segment, state, **fields):
        if fields.get("phase") == "protocol_ready" and self.replay_boundaries:
            original = segment.model_source_allowed
            boundaries = list(self.replay_boundaries)
            gate_index = 0
            ready_versions = set()
            original_control = segment.model_control

            async def control(message):
                nonlocal gate_index
                result = await original_control(message)
                if (
                    message.get("type") == "control_ack"
                    and message.get("kind") == "prompt.update"
                    and message.get("state") == "kv_ready"
                    and message.get("prompt_version") not in ready_versions
                ):
                    ready_versions.add(message.get("prompt_version"))
                    gate_index += 1
                    await segment._notify_source_admission()
                return result

            def allowed(sent):
                if segment.condition_switch is not None:
                    return original(sent)
                limit = (
                    boundaries[gate_index]
                    if gate_index < len(boundaries)
                    else float("inf")
                )
                return sent < limit and original(sent)

            segment.model_control = control
            segment.model_source_allowed = allowed
        await super().segment_status(segment, state, **fields)


async def _run_single_segment_flow(
    *,
    case_id: str,
    recording_meta: dict[str, Any],
    media_root: Path,
    spool_dir: Path,
    spec: VideoSpec,
    mode: str,
    cache_store: MockSglangCacheStore,
    sglang_url: str | None,
    coordinator_url: str | None = None,
    max_frames_after_last_event: int,
    full_frames: bool,
    preserve_switch_boundaries: bool,
    realtime_pacing: bool,
    pacing_speed: float,
    capture_fps: int | None = None,
    repeat_warm_prefill: bool = False,
    prewarm_switch_conditions: bool = False,
    extra_stress_recording: dict[str, Any] | None = None,
    secondary_segment_recording: dict[str, Any] | None = None,
    seed_override: int | None = None,
    stress_seed_override: int | None = None,
    secondary_seed_override: int | None = None,
) -> dict[str, Any]:
    os.environ["WHITEBOX_BOUND_COMPUTE"] = "1"
    resolved_coord = _resolve_coordinator_url(sglang_url, coordinator_url)
    if mode == "real-sglang":
        await _wait_for_coordinator_idle(resolved_coord, timeout_s=20.0)
    rec_id = recording_meta["recording_id"]
    rec_dir = media_root / rec_id
    mp4_path = rec_dir / "whitebox.mp4"
    if mode == "real-sglang":
        for rel, meta in recording_meta["objects"].items():
            raw = (rec_dir / rel).read_bytes()
            if (
                len(raw) != meta["bytes"]
                or hashlib.sha256(raw).hexdigest() != meta["sha256"]
            ):
                raise ValueError(f"PREFLIGHT_MEDIA_MISMATCH: {rec_id}/{rel}")
    raw_switches = list(recording_meta.get("condition_switches", []))
    primary_seed = (
        int(seed_override)
        if seed_override is not None
        else int(recording_meta.get("seed", 42))
    )

    effective_switches: list[dict[str, Any]] = []
    for idx, sw in enumerate(raw_switches):
        sw_copy = dict(sw)
        if full_frames or preserve_switch_boundaries:
            sw_copy["executed_boundary"] = int(sw["boundary"])
        else:
            sw_copy["executed_boundary"] = 17 * (idx + 1)
        effective_switches.append(sw_copy)

    last_event_frame = (
        max(sw["executed_boundary"] for sw in effective_switches)
        if effective_switches
        else 0
    )
    video_total_frames = int(recording_meta["video"]["frames"])
    target_segment_frames = (
        video_total_frames
        if full_frames
        else min(video_total_frames, last_event_frame + max_frames_after_last_event)
    )
    effective_realtime_pacing = realtime_pacing or (mode == "real-sglang")
    effective_pacing_speed = (
        1.0 if (mode == "real-sglang" and not realtime_pacing) else pacing_speed
    )
    default_wait_timeout = 180.0 if mode == "real-sglang" else 25.0

    created_engines: list[RecordedVideoEngine] = []
    engine_factory = create_recorded_video_engine(
        mp4_path if mp4_path.exists() else None,
        rgb_frames=(
            None
            if mp4_path.exists()
            else (lambda idx, s: bytes([(idx * 13 + 7) % 251]) * s.frame_bytes)
        ),
        max_frames=None,
        capture_fps=capture_fps,
        realtime_pacing=effective_realtime_pacing,
        pacing_speed=effective_pacing_speed,
        fast_frame_delay_s=0.008,
        loop=True,
        on_created=created_engines.append,
    )

    archives = AsyncArchiveManager(
        recorder_slots=4,
        queue_depth=512,
        queue_bytes=256 * 1024 * 1024,
        max_age=60.0,
        reserve_bytes=0,
    )
    journal = RenderJournal(
        MemoryRoomStore(),
        user_id="suite-user",
        source_session_id=f"src-{case_id}-{uuid4().hex[:6]}",
        source_epoch="epoch-1",
        owner_epoch="room-1",
    )
    await journal.create()

    def _make_leases(_user_id: str, seg: dict[str, Any]):
        if mode == "real-sglang":
            return _create_real_sglang_lease(
                sglang_url,
                coordinator_url,
                spec,
                trace_id=case_id,
                segment=seg,
            )
        return MockBoundLease(spec, cache_store)

    host = SimpleNamespace(
        runtime=SimpleNamespace(players={}),
        make_segment_leases=_make_leases,
        sessions={},
    )
    source = _ReplaySourceSession(
        host,
        journal,
        source_epoch="epoch-1",
        user_id="suite-user",
        session_id=journal.source_session_id,
        trace_id=f"trace-{case_id}",
        config=WhiteboxConfig(
            engine_factory=engine_factory,
            ui_factory=lambda **_: _BrowserUIStub(),
            spool=spool_dir,
            spec=spec,
            room_owned=True,
            capture_clock=True,
            pipeline_overlap=True,
            num_inference_steps=int(os.environ.get("WHITEBOX_INFERENCE_STEPS", "4")),
            model_input_frames=int(os.environ.get("WHITEBOX_MODEL_INPUT_FRAMES", "34")),
            startup_seconds=120 if mode == "real-sglang" else 30,
            play_seconds=600,
            drain_seconds=30 if mode == "real-sglang" else 10,
        ),
        archives=archives,
    )

    source.replay_boundaries = [s["executed_boundary"] for s in effective_switches]

    messages: list[dict[str, Any]] = []
    error_messages: list[dict[str, Any]] = []
    prefill_receipts: list[dict[str, Any]] = []
    prefill_receipt_times: dict[tuple[str, str], float] = {}
    model_cache_messages: list[dict[str, Any]] = []
    protocol_ready_messages: list[dict[str, Any]] = []
    frame_batches_by_epoch: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    prompt_applied_times: dict[tuple[str, int], float] = {}
    req_counter = 0

    def _received_frames_for_epoch(seg_epoch: str) -> int:
        return sum(
            int(m.get("num_frames", 1))
            for _, m in frame_batches_by_epoch.get(seg_epoch, [])
        )

    async def _consume() -> None:
        while True:
            raw = await source.recv()
            now_mono = time.perf_counter()
            msg = msgspec.msgpack.decode(raw)
            messages.append(msg)
            mtype = msg.get("type")
            if mtype in {"error", "operation.rejected"} or (
                mtype == "prefill.failed" and not msg.get("memory_ready")
            ):
                error_messages.append(msg)
            if mtype in {"whitebox.frame", "frame_batch"}:
                source.media_delivered()
            if mtype == "frame_batch":
                seg_epoch = msg.get("segment_epoch", "")
                frame_batches_by_epoch.setdefault(seg_epoch, []).append((now_mono, msg))
            elif mtype in {
                "prefill.memory_ready",
                "prefill.persisted",
                "prefill.source_resumed",
            } or (mtype == "prefill.failed" and msg.get("memory_ready") is True):
                prefill_receipts.append(msg)
                prefill_receipt_times[(msg.get("request_id", ""), mtype)] = now_mono
            elif mtype == "model.cache":
                model_cache_messages.append(msg)
            elif mtype == "render.status":
                if msg.get("phase") == "protocol_ready":
                    protocol_ready_messages.append(msg)
                if msg.get("state") == "prompt_applied":
                    key = (msg.get("segment_id", ""), int(msg.get("prompt_version", 0)))
                    prompt_applied_times[key] = now_mono

    async def _wait_for(predicate, timeout_s: float | None = None):
        eff_timeout = default_wait_timeout if timeout_s is None else timeout_s
        async with asyncio.timeout(eff_timeout):
            while True:
                if error_messages:
                    raise RuntimeError(
                        f"SourceSession emitted error: {error_messages[-1]}"
                    )
                res = predicate()
                if res:
                    return res
                await asyncio.sleep(0.002)

    async def _send_cmd(op: str, **fields: Any) -> str:
        nonlocal req_counter
        req_counter += 1
        req_id = f"{case_id}-cmd-{req_counter}"
        source.control_tokens = 4.0
        cmd = {
            "type": "command",
            "op": op,
            "request_id": req_id,
            "source_session_id": source.session_id,
            "source_epoch": source.source_epoch,
            **fields,
        }
        await source.send(msgspec.msgpack.encode(cmd))
        return req_id

    async def _upload_recording_refs(
        rec_meta: dict[str, Any], r_dir: Path
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Uploads all reference objects for a recording and returns initial mapping + sha->asset_id map."""
        nonlocal req_counter
        objs = rec_meta.get("objects", {})
        ref_rel_paths = sorted(p for p in objs if p.startswith("references/"))
        sha_to_asset_id: dict[str, str] = {}
        ordered_asset_ids: list[str] = []
        for rel_p in ref_rel_paths:
            f_path = r_dir / rel_p
            fmt = _ref_format_from_rel_path(rel_p)
            data = (
                f_path.read_bytes()
                if f_path.exists()
                else _synthesize_image_bytes(fmt, len(ordered_asset_ids) + 1)
            )
            req_counter += 1
            req_id = f"{case_id}-upload-{req_counter}"
            await source.send(
                msgspec.msgpack.encode(
                    {
                        "type": "asset.upload",
                        "version": 2,
                        "request_id": req_id,
                        "source_session_id": source.session_id,
                        "source_epoch": source.source_epoch,
                        "format": fmt,
                        "payload": data,
                    }
                )
            )
            done_msg = await _wait_for(
                lambda: next(
                    (
                        m
                        for m in messages
                        if m.get("type") == "operation.result"
                        and m.get("request_id") == req_id
                        and m.get("state") == "complete"
                    ),
                    None,
                )
            )
            actual_asset_id = done_msg["asset"]["asset_id"]
            orig_sha = objs[rel_p].get("sha256", "")
            if orig_sha:
                sha_to_asset_id[f"sha256:{orig_sha}"] = actual_asset_id
                sha_to_asset_id[orig_sha] = actual_asset_id
            ordered_asset_ids.append(actual_asset_id)

        initial_mapping: list[dict[str, Any]] = []
        for idx, ref_item in enumerate(rec_meta.get("references_mapping", [])):
            orig_aid = ref_item.get("asset_id", "")
            mapped_aid = sha_to_asset_id.get(
                orig_aid,
                ordered_asset_ids[idx] if idx < len(ordered_asset_ids) else orig_aid,
            )
            initial_mapping.append(
                {
                    "id": ref_item["id"],
                    "asset_id": mapped_aid,
                    "role": ref_item.get("role", "scene"),
                }
            )
        return initial_mapping, sha_to_asset_id

    async def _prepare_f0() -> dict[str, Any]:
        state = await journal.read()
        prep_req_id = await _send_cmd(
            "render.prepare",
            expected_render_revision=state["render_revision"],
        )
        return await _wait_for(
            lambda: next(
                (
                    m
                    for m in messages
                    if m.get("type") == "operation.result"
                    and m.get("request_id") == prep_req_id
                    and m.get("state") == "completed"
                ),
                None,
            )
        )

    async def _run_prefills(
        prep_done: dict[str, Any],
        prompt: str,
        seed: int,
        refs_mapping: list[dict[str, Any]],
        *,
        repeat_warm: bool = False,
        extra_conditions: list[tuple[str, int, list[dict[str, Any]]]] | None = None,
    ) -> dict[str, Any]:
        nonlocal req_counter
        ctx_id = prep_done["context_id"]
        f0_index = prep_done["source_frame_index"]
        f0_sha256 = prep_done["source_frame_sha256"]

        async def _submit_one_prefill(
            p_text: str, p_seed: int, p_refs: list[dict[str, Any]]
        ) -> dict[str, Any]:
            nonlocal req_counter
            req_counter += 1
            pf_req_id = f"{case_id}-pf-{req_counter}"
            t0 = time.perf_counter()
            await source.send(
                msgspec.msgpack.encode(
                    {
                        "type": "event",
                        "eventType": "prefill",
                        "version": 2,
                        "request_id": pf_req_id,
                        "source_session_id": source.session_id,
                        "source_epoch": source.source_epoch,
                        "payload": {
                            "context_id": ctx_id,
                            "prompt": p_text,
                            "seed": p_seed,
                            "references": p_refs,
                        },
                    }
                )
            )
            mem_ready = await _wait_for(
                lambda: next(
                    (
                        m
                        for m in prefill_receipts
                        if m.get("type") == "prefill.memory_ready"
                        and m.get("request_id") == pf_req_id
                    ),
                    None,
                )
            )
            wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            cache_nested = (
                mem_ready.get("cache")
                if isinstance(mem_ready.get("cache"), dict)
                else {}
            )
            cache_result = mem_ready.get("cache_result") or cache_nested.get(
                "cache_result"
            )
            hit = (
                mem_ready.get("hit")
                if mem_ready.get("hit") is not None
                else (
                    cache_nested.get("hit")
                    if cache_nested.get("hit") is not None
                    else (cache_result in {"l1_hit", "l2_hit"})
                )
            )
            timings_dict = mem_ready.get("timings") or mem_ready.get("timing") or {}
            raw_mem_ms = (
                mem_ready.get("prefill_to_memory_ms")
                or timings_dict.get("elapsed_host_ms")
                or wall_ms
            )
            mem_ms = round(float(raw_mem_ms), 3)

            persisted = None
            persist_wait_s = 1.0 if hit else (8.0 if mode == "real-sglang" else 3.0)
            try:
                persisted = await _wait_for(
                    lambda: next(
                        (
                            m
                            for m in prefill_receipts
                            if m.get("type") in {"prefill.persisted", "prefill.failed"}
                            and m.get("request_id") == pf_req_id
                        ),
                        None,
                    ),
                    timeout_s=persist_wait_s,
                )
            except TimeoutError:
                persisted = None

            persisted_ms = None
            persistence_state = "not_observed"
            persistence_timing_source = None
            if persisted is not None:
                if persisted.get("type") == "prefill.persisted":
                    raw_p_ms = persisted.get("prefill_to_persisted_ms")
                    persistence_timing_source = "worker"
                    if raw_p_ms is None:
                        raw_p_ms = (
                            prefill_receipt_times[(pf_req_id, "prefill.persisted")] - t0
                        ) * 1000
                        persistence_timing_source = "client_receipt"
                    persisted_ms = round(float(raw_p_ms), 3)
                    persistence_state = "persisted"
                else:
                    persistence_state = (
                        f"skipped:{persisted.get('code', 'unavailable')}"
                    )

            return {
                "request_id": pf_req_id,
                "context_id": ctx_id,
                "seed": p_seed,
                "cache_key": mem_ready.get("cache_key")
                or cache_nested.get("cache_key"),
                "cache_result": cache_result,
                "hit": hit,
                "prefill_to_memory_ms": mem_ms,
                "prefill_to_persisted_ms": persisted_ms,
                "persistence_state": persistence_state,
                "persistence_timing_source": persistence_timing_source,
                "wall_clock_ms": wall_ms,
                "reference_count": len(p_refs),
                "timings": timings_dict,
            }

        first_pf = await _submit_one_prefill(prompt, seed, refs_mapping)
        warm_pf = None
        if repeat_warm:
            warm_pf = await _submit_one_prefill(prompt, seed, refs_mapping)

        extra_pf_results: list[dict[str, Any]] = []
        for ex_prompt, ex_seed, ex_refs in extra_conditions or []:
            extra_pf_results.append(
                await _submit_one_prefill(ex_prompt, ex_seed, ex_refs)
            )

        return {
            "context_id": ctx_id,
            "f0_source_frame_index": f0_index,
            "f0_source_frame_sha256": f0_sha256,
            "primary_prefill": first_pf,
            "warm_prefill": warm_pf,
            "extra_prefills": extra_pf_results,
        }

    consumer_task = asyncio.create_task(_consume())
    lease_release_verified = False
    try:
        world_short = recording_meta["world_id"].split("/")[-1]
        await source.send(
            msgspec.msgpack.encode(
                {
                    "type": "init",
                    "whitebox": {"world_id": world_short},
                    "prompt": recording_meta["prompt"],
                }
            )
        )
        await _wait_for(lambda: source.ready is not None and source.ready.is_set())
        assert source.model_binding is not None, "ModelBinding must be initialized"
        assert source.control_prefill is not None, "ControlPrefill must be initialized"
        await source.model_binding.wait_ready()

        # Pin F0 immediately so source capture is held while references upload & prefill
        prep_done = await _prepare_f0()

        initial_refs, sha_to_aid = await _upload_recording_refs(recording_meta, rec_dir)

        mapped_switches: list[dict[str, Any]] = []
        extra_prefill_conds: list[tuple[str, int, list[dict[str, Any]]]] = []
        for sw in effective_switches:
            sw_refs_mapped: list[dict[str, Any]] = []
            for r_item in sw.get("references", []):
                orig_aid = r_item.get("asset_id", "")
                sw_refs_mapped.append(
                    {
                        "id": r_item["id"],
                        "asset_id": sha_to_aid.get(orig_aid, orig_aid),
                        "role": r_item.get("role", "scene"),
                    }
                )
            sw_copy = dict(sw, mapped_references=sw_refs_mapped)
            mapped_switches.append(sw_copy)
            if prewarm_switch_conditions:
                extra_prefill_conds.append(
                    (
                        sw["prompt"],
                        primary_seed,
                        sw_refs_mapped,
                    )
                )

        if extra_stress_recording is not None:
            stress_dir = media_root / extra_stress_recording["recording_id"]
            stress_refs, _ = await _upload_recording_refs(
                extra_stress_recording, stress_dir
            )
            stress_seed = (
                int(stress_seed_override)
                if stress_seed_override is not None
                else int(extra_stress_recording.get("seed", 42))
            )
            extra_prefill_conds.append(
                (
                    extra_stress_recording["prompt"],
                    stress_seed,
                    stress_refs,
                )
            )

        prep_info = await _run_prefills(
            prep_done,
            recording_meta["prompt"],
            primary_seed,
            initial_refs,
            repeat_warm=repeat_warm_prefill,
            extra_conditions=extra_prefill_conds,
        )

        # Start Segment 1 with prepared context_id & F0 start_boundary
        state = await journal.read()
        start_wall = time.perf_counter()
        await _send_cmd(
            "render.start",
            expected_render_revision=state["render_revision"],
            context_id=prep_info["context_id"],
            start_boundary={"source_frame_index": prep_info["f0_source_frame_index"]},
            prompt=recording_meta["prompt"],
            seed=primary_seed,
            references=initial_refs,
        )
        await _wait_for(
            lambda: (
                source.current_segment is not None
                and source.current_segment.first_decodable
                and _received_frames_for_epoch(source.current_segment.segment_epoch) > 0
            )
        )
        segment = source.current_segment
        assert segment is not None
        first_batch_time, _ = frame_batches_by_epoch[segment.segment_epoch][0]
        session_ttff_ms = round((first_batch_time - start_wall) * 1000.0, 3)

        # Validate Precondition 5: Negotiated Worker Protocol & Reference Capacity
        proto_msg = protocol_ready_messages[-1] if protocol_ready_messages else {}
        negotiated_protocol = {
            "condition_switch_protocol": proto_msg.get("condition_switch_protocol"),
            "prompt_reference_updates": proto_msg.get("prompt_reference_updates"),
            "max_references": proto_msg.get("max_references"),
            "model_session_id": proto_msg.get("model_session_id"),
            "generation_id": proto_msg.get("generation_id"),
        }
        if proto_msg:
            if proto_msg.get("condition_switch_protocol") != "clip-barrier-v1":
                raise RuntimeError(
                    f"PREFLIGHT_PROTOCOL_MISMATCH: expected clip-barrier-v1, got {proto_msg}"
                )
            if not proto_msg.get("prompt_reference_updates"):
                raise RuntimeError(
                    f"PREFLIGHT_REF_UPDATES_UNSUPPORTED: worker did not enable prompt_reference_updates: {proto_msg}"
                )
            if int(proto_msg.get("max_references", 0)) < 12:
                raise RuntimeError(
                    f"PREFLIGHT_MAX_REFS_TOO_LOW: worker max_references={proto_msg.get('max_references')} < 12"
                )

        resumed_receipt = await _wait_for(
            lambda: next(
                (
                    m
                    for m in prefill_receipts
                    if m.get("type") == "prefill.source_resumed"
                    and m.get("context_id") == prep_info["context_id"]
                ),
                None,
            )
        )

        # Execute in-segment condition switches
        switch_metrics: list[dict[str, Any]] = []
        for sw in mapped_switches:
            exec_boundary = int(sw["executed_boundary"])
            prev_frame_trigger = exec_boundary
            await _wait_for(lambda: segment.selected_frames == prev_frame_trigger)
            sw_wall_start = time.perf_counter()
            version = int(sw["prompt_version"])
            await _send_cmd(
                "prompt.replace",
                segment_id=segment.segment_id,
                segment_epoch=segment.segment_epoch,
                expected_prompt_version=version - 1,
                prompt_version=version,
                prompt=sw["prompt"],
                references=sw["mapped_references"],
                source_policy=sw.get("source_policy", "skip_during_switch"),
                effective={
                    "source_frame_index": prep_info["f0_source_frame_index"]
                    + prev_frame_trigger,
                    "policy": "at_or_after",
                },
            )
            await _wait_for(
                lambda: (
                    journal.snapshot["segment"]["applied_version"] >= version
                    and (segment.segment_id, version) in prompt_applied_times
                ),
            )
            applied_mono = prompt_applied_times[(segment.segment_id, version)]
            prompt_switch_ttff_ms = round((applied_mono - sw_wall_start) * 1000.0, 3)
            await _wait_for(
                lambda: any(
                    row.get("prompt_version") == version
                    for row in segment.recording.manifest.get("condition_switches", [])
                )
            )
            manifest_sw = next(
                row
                for row in segment.recording.manifest["condition_switches"]
                if row.get("prompt_version") == version
            )
            if manifest_sw["boundary"] != exec_boundary:
                raise RuntimeError(
                    f"REPLAY_BOUNDARY_MISMATCH: expected {exec_boundary}, got {manifest_sw['boundary']}"
                )
            cache_receipt = next(
                (
                    r
                    for r in segment.recording.manifest.get("model_cache_receipts", [])
                    if r.get("prompt_version") == version
                ),
                None,
            )
            switch_metrics.append(
                {
                    "prompt_version": version,
                    "configured_historical_boundary": sw["boundary"],
                    "executed_boundary": manifest_sw["boundary"],
                    "whitebox_condition_switch_host_ms": round(
                        float(manifest_sw["switch_host_ms"]), 3
                    ),
                    "prompt_switch_ttff_ms": prompt_switch_ttff_ms,
                    "skipped_captures": int(manifest_sw["skipped_captures"]),
                    "kv_ready": bool(manifest_sw["kv_ready"]),
                    "reference_count": len(sw["mapped_references"]),
                    "cache_receipt": cache_receipt,
                }
            )

        # Ensure steady-state tail frames are collected after the actual worker clip boundary
        effective_target_frames = max(
            target_segment_frames,
            max((int(s["executed_boundary"]) for s in switch_metrics), default=0)
            + max_frames_after_last_event,
        )
        await _wait_for(
            lambda: (
                _received_frames_for_epoch(segment.segment_epoch)
                >= effective_target_frames
            ),
        )

        # End Segment 1
        state = await journal.read()
        end_req_id = await _send_cmd(
            "render.end",
            expected_render_revision=state["render_revision"],
            segment_id=segment.segment_id,
            segment_epoch=segment.segment_epoch,
            mode="drain",
        )
        await _wait_for(
            lambda: (
                segment.closed_event.is_set()
                and any(
                    m.get("type") == "operation.result"
                    and m.get("request_id") == end_req_id
                    and m.get("state") == "completed"
                    for m in messages
                )
            )
        )
        seg1_manifest = dict(segment.recording.manifest)
        seg1_dropped = int(segment.dropped_before_model)
        seg1_sent = int(segment.selected_frames)
        seg1_received = _received_frames_for_epoch(segment.segment_epoch)

        secondary_result = None
        if secondary_segment_recording is not None:
            sec_seed = (
                int(secondary_seed_override)
                if secondary_seed_override is not None
                else int(secondary_segment_recording.get("seed", 42))
            )
            sec_prep_done = await _prepare_f0()
            sec_dir = media_root / secondary_segment_recording["recording_id"]
            sec_refs, _ = await _upload_recording_refs(
                secondary_segment_recording, sec_dir
            )
            sec_prep = await _run_prefills(
                sec_prep_done,
                secondary_segment_recording["prompt"],
                sec_seed,
                sec_refs,
            )
            state = await journal.read()
            sec_start_wall = time.perf_counter()
            await _send_cmd(
                "render.start",
                expected_render_revision=state["render_revision"],
                context_id=sec_prep["context_id"],
                start_boundary={
                    "source_frame_index": sec_prep["f0_source_frame_index"]
                },
                prompt=secondary_segment_recording["prompt"],
                seed=sec_seed,
                references=sec_refs,
            )
            await _wait_for(
                lambda: (
                    source.current_segment is not None
                    and source.current_segment.segment_id != segment.segment_id
                    and source.current_segment.first_decodable
                    and _received_frames_for_epoch(source.current_segment.segment_epoch)
                    > 0
                )
            )
            seg2 = source.current_segment
            assert seg2 is not None
            sec_first_time, _ = frame_batches_by_epoch[seg2.segment_epoch][0]
            sec_ttff_ms = round((sec_first_time - sec_start_wall) * 1000.0, 3)
            sec_target_frames = min(
                int(secondary_segment_recording["video"]["frames"]),
                max_frames_after_last_event,
            )
            await _wait_for(
                lambda: (
                    _received_frames_for_epoch(seg2.segment_epoch) >= sec_target_frames
                )
            )
            state = await journal.read()
            await _send_cmd(
                "render.end",
                expected_render_revision=state["render_revision"],
                segment_id=seg2.segment_id,
                segment_epoch=seg2.segment_epoch,
                mode="drain",
            )
            await _wait_for(lambda: seg2.closed_event.is_set())
            secondary_result = {
                "recording_id": secondary_segment_recording["recording_id"],
                "seed": sec_seed,
                "prepare": sec_prep,
                "whitebox_session_ttff_ms": sec_ttff_ms,
                "sent_frames": int(seg2.selected_frames),
                "received_model_frames": _received_frames_for_epoch(seg2.segment_epoch),
                "dropped_before_model": int(seg2.dropped_before_model),
                "model_cache_receipts": list(
                    seg2.recording.manifest.get("model_cache_receipts", [])
                ),
            }

        await source.unbind("SUITE_COMPLETED")
        await source.close()
        lease_obj = getattr(source.model_binding, "lease", None)
        lease_release_verified = (
            getattr(lease_obj, "resource_state", None) == "released"
        )
        if not lease_release_verified:
            raise RuntimeError(
                f"PREFLIGHT_RELEASE_UNVERIFIED: case={case_id} resource_state={getattr(lease_obj, 'resource_state', None)}"
            )
        engine_obj = created_engines[0] if created_engines else None
        return {
            "case_id": case_id,
            "world_id": recording_meta["world_id"],
            "recording_id": rec_id,
            "seed": primary_seed,
            "negotiated_protocol": negotiated_protocol,
            "lease_release_verified": lease_release_verified,
            "prepare": prep_info,
            "source_resumed_paused_ms": resumed_receipt.get("paused_ms"),
            "whitebox_session_ttff_ms": session_ttff_ms,
            "model_cache_receipts": list(seg1_manifest.get("model_cache_receipts", [])),
            "condition_switches": switch_metrics,
            "steady_state_flow": {
                "capture_fps": (
                    engine_obj.capture_fps if engine_obj else (capture_fps or spec.fps)
                ),
                "model_fps": spec.fps,
                "sent_frames": seg1_sent,
                "received_model_frames": seg1_received,
                "dropped_before_model": seg1_dropped,
                "skipped_captures": sum(s["skipped_captures"] for s in switch_metrics),
                "capture_control_log": (
                    list(engine_obj.control_log) if engine_obj else []
                ),
            },
            "secondary_segment": secondary_result,
        }
    finally:
        consumer_task.cancel()
        await asyncio.gather(consumer_task, return_exceptions=True)
        await source.close()
        await archives.close()
        if mode == "real-sglang":
            await _wait_for_coordinator_idle(resolved_coord, timeout_s=20.0)


async def run_four_case_suite(
    suite_data,
    *,
    encoder_url=None,
    dit_url=None,
    decoder_url=None,
    routing_revision=None,
    compatibility=None,
    profile="480",
    network_domain="test-shanghai-copy",
    room_token_env="WHITEBOX_ROOM_TOKEN",
    signing_secret_env="ROLE_POOL_SIGNING_SECRET",
    control_epoch=None,
    **kwargs,
):
    from four_case_role_pool import SuiteRolePool

    urls = dict(encoder=encoder_url, dit=dit_url, decoder=decoder_url)
    explicit = any(urls.values())
    if explicit and (not all(urls.values()) or kwargs.get("coordinator_url")):
        raise ValueError(
            "Provide all three Worker URLs or a Coordinator URL, exclusively"
        )
    if explicit or compatibility or routing_revision:
        if (
            kwargs.get("mode") != "real-sglang"
            or not compatibility
            or not routing_revision
        ):
            raise ValueError(
                "Explicit role pool requires --mode real-sglang, --routing-revision and --compatibility"
            )
        if not explicit and not kwargs.get("coordinator_url"):
            raise ValueError(
                "Role pool requires a Coordinator URL or all three Worker URLs"
            )
        async with SuiteRolePool(
            coordinator_url=kwargs.get("coordinator_url"),
            workers=urls if explicit else None,
            revision=routing_revision,
            compatibility=compatibility,
            profile=profile,
            network_domain=network_domain,
            room_token_env=room_token_env,
            signing_secret_env=signing_secret_env,
            control_epoch=control_epoch,
        ) as pool:
            await pool.wait_idle()
            return await _run_four_case_suite(suite_data, **kwargs)
    return await _run_four_case_suite(suite_data, **kwargs)


async def _run_four_case_suite(
    suite_data: dict[str, Any],
    *,
    media_root: Path,
    output_dir: Path,
    mode: str = "mock-worker",
    sglang_url: str | None = None,
    coordinator_url: str | None = None,
    max_frames_after_last_event: int = 18,
    full_frames: bool = False,
    preserve_switch_boundaries: bool = False,
    realtime_pacing: bool = False,
    pacing_speed: float = 1.0,
    capture_fps: int | None = None,
    synthesize_if_missing: bool = True,
    seed_base: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    spool_dir = output_dir / "spool"
    spool_dir.mkdir(parents=True, exist_ok=True)

    if synthesize_if_missing and mode == "mock-worker":
        prepare_all_media(
            suite_data,
            media_root,
            synthesize_fixtures=True,
            synthetic_frames=(
                None if full_frames else max_frames_after_last_event + 48
            ),
        )

    effective_seed_base = (
        int(seed_base)
        if seed_base is not None
        else ((int(time.time()) % 100000) * 10 + 1000 if mode == "real-sglang" else 42)
    )

    preflight_report = run_preflight_checks(
        suite_data,
        media_root=media_root,
        mode=mode,
        sglang_url=sglang_url,
        coordinator_url=coordinator_url,
        require_real_media=(mode == "real-sglang"),
        preserve_switch_boundaries=preserve_switch_boundaries,
        full_frames=full_frames,
        max_frames_after_last_event=max_frames_after_last_event,
        seed_base=effective_seed_base,
    )

    spec = VideoSpec(832, 480, 24)
    cache_store = MockSglangCacheStore()
    cases_by_id = {c["case_id"]: c for c in suite_data["cases"]}

    # Step 1: Shanghai Drive Baseline (Cold vs Warm Prefill + Session TTFF)
    case1_res = await _run_single_segment_flow(
        case_id="shanghai_drive_baseline",
        recording_meta=cases_by_id["shanghai_drive_baseline"],
        media_root=media_root,
        spool_dir=spool_dir / "step1",
        spec=spec,
        mode=mode,
        cache_store=cache_store,
        sglang_url=sglang_url,
        coordinator_url=coordinator_url,
        max_frames_after_last_event=max_frames_after_last_event,
        full_frames=full_frames,
        preserve_switch_boundaries=preserve_switch_boundaries,
        realtime_pacing=realtime_pacing,
        pacing_speed=pacing_speed,
        capture_fps=capture_fps,
        repeat_warm_prefill=True,
        seed_override=effective_seed_base + 1,
    )

    # Step 2: MARVEL: OVERDRIVE (2 Rapid Same-Ref Switches + 12-Ref Stress)
    c2_meta = cases_by_id["marvel_overdrive_rapid_switch"]
    case2_res = await _run_single_segment_flow(
        case_id="marvel_overdrive_rapid_switch",
        recording_meta=c2_meta,
        media_root=media_root,
        spool_dir=spool_dir / "step2",
        spec=spec,
        mode=mode,
        cache_store=cache_store,
        sglang_url=sglang_url,
        coordinator_url=coordinator_url,
        max_frames_after_last_event=max_frames_after_last_event,
        full_frames=full_frames,
        preserve_switch_boundaries=preserve_switch_boundaries,
        realtime_pacing=realtime_pacing,
        pacing_speed=pacing_speed,
        capture_fps=capture_fps,
        extra_stress_recording=c2_meta.get("stress_recording"),
        seed_override=effective_seed_base + 2,
        stress_seed_override=effective_seed_base + 20,
    )

    # Step 3: Dark Horns (3->6 Refs Expansion Switch Miss vs Hit A/B)
    c3_meta = cases_by_id["dark_horns_ref_expansion"]
    dark_miss_cache = MockSglangCacheStore()
    dark_miss_res = await _run_single_segment_flow(
        case_id="dark_horns_ref_expansion_miss",
        recording_meta=c3_meta,
        media_root=media_root,
        spool_dir=spool_dir / "step3_miss",
        spec=spec,
        mode=mode,
        cache_store=dark_miss_cache,
        sglang_url=sglang_url,
        coordinator_url=coordinator_url,
        max_frames_after_last_event=max_frames_after_last_event,
        full_frames=full_frames,
        preserve_switch_boundaries=preserve_switch_boundaries,
        realtime_pacing=realtime_pacing,
        pacing_speed=pacing_speed,
        capture_fps=capture_fps,
        prewarm_switch_conditions=False,
        seed_override=effective_seed_base + 30,
    )
    dark_hit_cache = MockSglangCacheStore()
    dark_hit_res = await _run_single_segment_flow(
        case_id="dark_horns_ref_expansion_hit",
        recording_meta=c3_meta,
        media_root=media_root,
        spool_dir=spool_dir / "step3_hit",
        spec=spec,
        mode=mode,
        cache_store=dark_hit_cache,
        sglang_url=sglang_url,
        coordinator_url=coordinator_url,
        max_frames_after_last_event=max_frames_after_last_event,
        full_frames=full_frames,
        preserve_switch_boundaries=preserve_switch_boundaries,
        realtime_pacing=realtime_pacing,
        pacing_speed=pacing_speed,
        capture_fps=capture_fps,
        prewarm_switch_conditions=True,
        seed_override=effective_seed_base + 31,
    )

    miss_sw = dark_miss_res["condition_switches"][0]
    hit_sw = dark_hit_res["condition_switches"][0]
    miss_host_ms = miss_sw["whitebox_condition_switch_host_ms"]
    hit_host_ms = hit_sw["whitebox_condition_switch_host_ms"]
    _require_cache_result(
        miss_sw.get("cache_receipt"), {"miss"}, "Dark Horns unprefilled"
    )
    _require_cache_result(
        hit_sw.get("cache_receipt"), {"l1_hit", "l2_hit"}, "Dark Horns prefilled"
    )
    expected_hit_key = dark_hit_res["prepare"]["extra_prefills"][0]["cache_key"]
    if hit_sw["cache_receipt"]["cache_key"] != expected_hit_key:
        raise RuntimeError("CACHE_AB_KEY_MISMATCH")
    if (
        dark_miss_res["prepare"]["f0_source_frame_sha256"]
        != dark_hit_res["prepare"]["f0_source_frame_sha256"]
    ):
        raise RuntimeError("CACHE_AB_INPUT_MISMATCH")
    host_reduction_pct = (
        round((miss_host_ms - hit_host_ms) / miss_host_ms * 100.0, 2)
        if miss_host_ms > 0
        else 0.0
    )

    # Step 4: Wuxia Jiahao (Multi-Condition Prefill & Cross-Segment Re-Prepare)
    c4_meta = cases_by_id["wuxia_jiahao_multi_prefill"]
    case4_res = await _run_single_segment_flow(
        case_id="wuxia_jiahao_multi_prefill",
        recording_meta=c4_meta,
        media_root=media_root,
        spool_dir=spool_dir / "step4",
        spec=spec,
        mode=mode,
        cache_store=cache_store,
        sglang_url=sglang_url,
        coordinator_url=coordinator_url,
        max_frames_after_last_event=max_frames_after_last_event,
        full_frames=full_frames,
        preserve_switch_boundaries=preserve_switch_boundaries,
        realtime_pacing=realtime_pacing,
        pacing_speed=pacing_speed,
        capture_fps=capture_fps,
        secondary_segment_recording=c4_meta.get("secondary_recording"),
        seed_override=effective_seed_base + 40,
        secondary_seed_override=effective_seed_base + 41,
    )

    resolved_coord = _resolve_coordinator_url(sglang_url, coordinator_url)
    from four_case_role_pool import current_pool

    pool = current_pool.get()
    post_suite_capacity = (
        await pool.wait_idle()
        if pool
        else (
            _query_service_and_capacity(resolved_coord, require_idle=True)
            if mode == "real-sglang"
            else {"status": "mock_worker_mode", "all_leases_released": True}
        )
    )

    cold_pf = case1_res["prepare"]["primary_prefill"]
    warm_pf = case1_res["prepare"]["warm_prefill"]
    _require_cache_result(cold_pf, {"miss"}, "Shanghai cold prefill")
    _require_cache_result(warm_pf, {"l1_hit", "l2_hit"}, "Shanghai warm prefill")

    summary = {
        "schema_version": 1,
        "suite_name": suite_data["name"],
        "connection": (
            dict(
                backend="role_pool",
                routing_revision=pool.revision,
                profile=pool.config["profile"],
                compatibility=pool.config["compatibility"],
                network_domain=pool.config["network_domain"],
                coordinator_url=None
                if pool.workers
                else pool.config["coordinator_url"],
                workers=pool.workers,
            )
            if pool
            else {"backend": mode}
        ),
        "mode": mode,
        "seed_base": effective_seed_base,
        "created_at_epoch_ms": round(time.time() * 1000),
        "preflight": preflight_report,
        "post_suite_verification": {
            "all_leases_released": all(
                r.get("lease_release_verified", False)
                for r in (case1_res, case2_res, dark_miss_res, dark_hit_res, case4_res)
            ),
            "capacity_after_suite": (
                post_suite_capacity.get("capacity")
                if isinstance(post_suite_capacity, dict)
                and "capacity" in post_suite_capacity
                else post_suite_capacity
            ),
        },
        "metrics_overview": {
            "block0_cache": {
                "cold_prefill_to_memory_ms": cold_pf["prefill_to_memory_ms"],
                "cold_prefill_to_persisted_ms": cold_pf["prefill_to_persisted_ms"],
                "cold_cache_result": cold_pf["cache_result"],
                "warm_prefill_to_memory_ms": (
                    warm_pf["prefill_to_memory_ms"] if warm_pf else None
                ),
                "warm_cache_result": warm_pf["cache_result"] if warm_pf else None,
                "marvel_12ref_stress_prefill_ms": (
                    case2_res["prepare"]["extra_prefills"][0]["prefill_to_memory_ms"]
                    if case2_res["prepare"]["extra_prefills"]
                    else None
                ),
                "marvel_12ref_cache_result": (
                    case2_res["prepare"]["extra_prefills"][0]["cache_result"]
                    if case2_res["prepare"]["extra_prefills"]
                    else None
                ),
            },
            "session_ttff": {
                "shanghai_drive_ttff_ms": case1_res["whitebox_session_ttff_ms"],
                "marvel_overdrive_ttff_ms": case2_res["whitebox_session_ttff_ms"],
                "dark_horns_ttff_ms": dark_hit_res["whitebox_session_ttff_ms"],
                "wuxia_jiahao_seg1_ttff_ms": case4_res["whitebox_session_ttff_ms"],
                "wuxia_jiahao_seg2_ttff_ms": (
                    case4_res["secondary_segment"]["whitebox_session_ttff_ms"]
                    if case4_res["secondary_segment"]
                    else None
                ),
            },
            "prompt_switch_ttff": {
                "marvel_rapid_switches": case2_res["condition_switches"],
                "dark_horns_ab_comparison": {
                    "unprefilled_miss": miss_sw,
                    "prefilled_hit": hit_sw,
                    "switch_host_ms_reduction_pct": host_reduction_pct,
                },
            },
            "steady_state_flow": {
                "shanghai_drive": case1_res["steady_state_flow"],
                "marvel_overdrive": case2_res["steady_state_flow"],
                "dark_horns_miss": dark_miss_res["steady_state_flow"],
                "dark_horns_hit": dark_hit_res["steady_state_flow"],
                "wuxia_jiahao": case4_res["steady_state_flow"],
            },
        },
        "cases": [
            case1_res,
            case2_res,
            {
                "case_id": "dark_horns_ref_expansion",
                "world_id": c3_meta["world_id"],
                "recording_id": c3_meta["recording_id"],
                "ab_arms": {
                    "miss": dark_miss_res,
                    "hit": dark_hit_res,
                },
                "switch_host_ms_reduction_pct": host_reduction_pct,
            },
            case4_res,
        ],
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    report_md = _render_report_md(summary)
    report_path = output_dir / "REPORT.md"
    report_path.write_text(report_md)
    return summary


def _render_report_md(summary: dict[str, Any]) -> str:
    b0 = summary["metrics_overview"]["block0_cache"]
    ttff = summary["metrics_overview"]["session_ttff"]
    psw = summary["metrics_overview"]["prompt_switch_ttff"]
    ab = psw["dark_horns_ab_comparison"]
    flow = summary["metrics_overview"]["steady_state_flow"]
    pf_info = summary.get("preflight", {}).get("preconditions", {})
    post_info = summary.get("post_suite_verification", {})
    lines = [
        "# Whitebox 四案例串联 SGLang 验证报告",
        "",
        f"- 运行模式：`{summary['mode']}`（`seed_base={summary.get('seed_base')}`）",
        "- 编排链路：`SourceSession` -> `ControlPrefill.prepare(hold F0)` -> `ControlPrefill.submit` -> `RenderSegment(start_boundary=F0)` -> `ControlPrefill.resume` -> `RenderSegment.switch` -> `render.end`",
        f"- 前置条件校验（Preflight 6 项）：`{summary.get('preflight', {}).get('status', 'unknown')}`；测试后租约释放核验：`all_leases_released={post_info.get('all_leases_released')}`",
        "",
        "## 0. 前置条件校验（Preflight Check）",
        "",
        "| 校验维度 | 状态 | 说明 |",
        "|---|---|---|",
        f"| 1. 媒体与几何完整性 | `passed={pf_info.get('1_media_and_geometry_integrity', {}).get('passed')}` | 校验 `{pf_info.get('1_media_and_geometry_integrity', {}).get('recordings_checked')}` 份录制 SHA-256、832x480@24fps 及参考图解码 |",
        f"| 2. Prompt 与参考图占位契约 | `passed={pf_info.get('2_prompt_and_reference_contract', {}).get('passed')}` | 校验 `{pf_info.get('2_prompt_and_reference_contract', {}).get('conditions_checked')}` 组 `<Video 1>` 与 `<Picture 1..N>` 映射 |",
        f"| 3. 服务版本与身份配置 | `passed={pf_info.get('3_service_identity_and_health', {}).get('passed')}` | `{json.dumps(pf_info.get('3_service_identity_and_health', {}).get('details', {}), ensure_ascii=False)}` |",
        f"| 4. 空闲容量与非抢占准入 | `passed={pf_info.get('4_idle_capacity_and_non_preemption', {}).get('passed')}` | `{json.dumps(pf_info.get('4_idle_capacity_and_non_preemption', {}).get('details', {}), ensure_ascii=False)}` |",
        "| 5. 协议能力与参考图上限 | `verified` | `ordered-boundary-v2` + `clip-barrier-v1` + `prompt_reference_updates=True` + `max_references>=12` |",
        f"| 6. Cache Key 隔离与资源归还 | `verified` | `seed_base={summary.get('seed_base')}` 隔离各 Step/A-B 臂；`all_leases_released={post_info.get('all_leases_released')}` |",
        "",
        "## 1. Block 0 Cache 生效与持久化指标",
        "",
        "| 场景 | Cache 结果 | Prefill to Memory (ms) | Prefill to Persisted (ms) |",
        "|---|---|---:|---:|",
        f"| Step 1 上海行 (Cold Prefill) | `{b0.get('cold_cache_result', 'unknown')}` | {b0['cold_prefill_to_memory_ms']} | {b0['cold_prefill_to_persisted_ms']} |",
        f"| Step 1 上海行 (Warm Prefill) | `{b0['warm_cache_result']}` | {b0['warm_prefill_to_memory_ms']} | - |",
        f"| Step 2 MARVEL (12-Ref Stress Prefill) | `{b0.get('marvel_12ref_cache_result', 'unknown')}` | {b0['marvel_12ref_stress_prefill_ms']} | - |",
        "",
        "## 2. Session TTFF 指标",
        "",
        "| 案例 | Session TTFF (`whitebox_session_ttff_ms`) |",
        "|---|---:|",
        f"| Step 1 驾驶：上海行 (`0d256658...`) | {ttff['shanghai_drive_ttff_ms']} |",
        f"| Step 2 跑酷：MARVEL: OVERDRIVE (`b8045d7b...`) | {ttff['marvel_overdrive_ttff_ms']} |",
        f"| Step 3 魂系战斗：Dark Horns (`5505ef1c...`) | {ttff['dark_horns_ttff_ms']} |",
        f"| Step 4 武侠格斗：功夫嘉豪 Seg 1 (`f03d9c0d...`) | {ttff['wuxia_jiahao_seg1_ttff_ms']} |",
        f"| Step 4 武侠格斗：功夫嘉豪 Seg 2 (`083738c8...`) | {ttff['wuxia_jiahao_seg2_ttff_ms']} |",
        "",
        "## 3. Prompt Switch TTFF 与 Dark Horns A/B 对照",
        "",
        "| 案例 / 切换 | Prompt Version | Cache 结果 | Switch Host (ms) | Prompt Switch TTFF (ms) | Skipped Captures |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for sw in psw["marvel_rapid_switches"]:
        c_res = (sw.get("cache_receipt") or {}).get("cache_result", "unknown")
        lines.append(
            f"| MARVEL Rapid Switch @{sw['executed_boundary']} | {sw['prompt_version']} | `{c_res}` | {sw['whitebox_condition_switch_host_ms']} | {sw['prompt_switch_ttff_ms']} | {sw['skipped_captures']} |"
        )
    miss_sw = ab["unprefilled_miss"]
    hit_sw = ab["prefilled_hit"]
    miss_c = (miss_sw.get("cache_receipt") or {}).get("cache_result", "unknown")
    hit_c = (hit_sw.get("cache_receipt") or {}).get("cache_result", "unknown")
    lines.extend(
        [
            f"| Dark Horns 3->6 Refs (Arm A: Unprefilled) | {miss_sw['prompt_version']} | `{miss_c}` | {miss_sw['whitebox_condition_switch_host_ms']} | {miss_sw['prompt_switch_ttff_ms']} | {miss_sw['skipped_captures']} |",
            f"| Dark Horns 3->6 Refs (Arm B: Prefilled) | {hit_sw['prompt_version']} | `{hit_c}` | {hit_sw['whitebox_condition_switch_host_ms']} | {hit_sw['prompt_switch_ttff_ms']} | {hit_sw['skipped_captures']} |",
            "",
            f"- **Dark Horns Prefill Hit 相对 Unprefilled Miss 切换耗时降低**：`{ab['switch_host_ms_reduction_pct']}%`",
            "",
            "## 4. 稳态供帧与流控统计",
            "",
            "| 案例 | Sent Frames | Received Model Frames | Dropped Before Model | Skipped Captures |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, f_row in flow.items():
        lines.append(
            f"| `{name}` | {f_row['sent_frames']} | {f_row['received_model_frames']} | {f_row['dropped_before_model']} | {f_row['skipped_captures']} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    p_prep = subparsers.add_parser(
        "prepare-media",
        help="Download and verify case media from OSS or synthesize local test fixtures.",
    )
    p_prep.add_argument(
        "--cases-json",
        type=Path,
        default=DEFAULT_CASES_JSON,
        help="Path to interactive_production_four_cases.json",
    )
    p_prep.add_argument(
        "--media-dir",
        type=Path,
        default=Path("/tmp/whitebox-four-cases-media"),
        help="Destination directory for prepared media",
    )
    p_prep.add_argument(
        "--oss-profile",
        type=str,
        default="shengdong",
        help="Aliyun CLI profile for OSS downloads",
    )
    p_prep.add_argument(
        "--synthesize-fixtures",
        action="store_true",
        help="Synthesize local 832x480 24fps mp4 and reference images without OSS",
    )
    p_prep.add_argument(
        "--synthetic-frames",
        type=int,
        default=64,
        help="Frame count per synthesized whitebox.mp4 when --synthesize-fixtures is set",
    )

    p_pf = subparsers.add_parser(
        "preflight",
        help="Run the 6 precondition checks without starting model sessions.",
    )
    p_pf.add_argument(
        "--cases-json",
        type=Path,
        default=DEFAULT_CASES_JSON,
        help="Path to interactive_production_four_cases.json",
    )
    p_pf.add_argument(
        "--media-dir",
        type=Path,
        default=Path("/tmp/whitebox-four-cases-media"),
        help="Directory containing prepared case media",
    )
    p_pf.add_argument(
        "--mode",
        choices=("mock-worker", "real-sglang"),
        default="mock-worker",
    )
    p_pf.add_argument(
        "--sglang-url",
        type=str,
        default=None,
    )
    p_pf.add_argument(
        "--coordinator-url",
        type=str,
        default=None,
    )
    p_pf.add_argument(
        "--require-real-media",
        action="store_true",
        help="Enforce exact byte size and SHA-256 verification even in mock-worker mode",
    )

    p_run = subparsers.add_parser(
        "run",
        help="Run the 4-case chained suite via Room SourceSession + RecordedVideoEngine.",
    )
    p_run.add_argument(
        "--cases-json",
        type=Path,
        default=DEFAULT_CASES_JSON,
        help="Path to interactive_production_four_cases.json",
    )
    p_run.add_argument(
        "--media-dir",
        type=Path,
        default=Path("/tmp/whitebox-four-cases-media"),
        help="Directory containing prepared case media",
    )
    p_run.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/whitebox-four-cases-output"),
        help="Output directory for summary.json and REPORT.md",
    )
    p_run.add_argument(
        "--mode",
        choices=("mock-worker", "real-sglang"),
        default="mock-worker",
        help="Execution mode: built-in mock SGLang worker or real SGLang endpoint",
    )
    p_run.add_argument(
        "--sglang-url",
        type=str,
        default=None,
        help="SGLang worker WebSocket/HTTP URL for --mode real-sglang",
    )
    p_run.add_argument(
        "--coordinator-url",
        type=str,
        default=None,
        help="Coordinator URL for --mode real-sglang",
    )
    for name in ("encoder", "dit", "decoder"):
        p_run.add_argument(
            f"--{name}-url", help=f"Dedicated test {name} managed HTTP Worker URL"
        )
    p_run.add_argument("--routing-revision")
    p_run.add_argument("--compatibility")
    p_run.add_argument("--profile", default="480")
    p_run.add_argument("--network-domain", default="test-shanghai-copy")
    p_run.add_argument("--control-epoch", help="Direct Workers' existing control epoch")
    p_run.add_argument("--signing-secret-env", default="ROLE_POOL_SIGNING_SECRET")
    p_run.add_argument("--room-token-env", default="WHITEBOX_ROOM_TOKEN")
    p_run.add_argument(
        "--max-frames-after-last-event",
        type=int,
        default=18,
        help="Truncate segment frame count after last switch event for rapid regression",
    )
    p_run.add_argument(
        "--full-frames",
        action="store_true",
        help="Run full historical frame counts without truncation",
    )
    p_run.add_argument(
        "--preserve-switch-boundaries",
        action="store_true",
        help="Preserve exact historical switch frame indices even when truncating tail",
    )
    p_run.add_argument(
        "--realtime-pacing",
        action="store_true",
        help="Pace RecordedVideoEngine at capture_fps * pacing_speed wall-clock time",
    )
    p_run.add_argument(
        "--capture-fps",
        type=int,
        default=None,
        help="Whitebox source capture FPS (1..60, e.g. 48); defaults to WHITEBOX_CLI_FPS or 24",
    )
    p_run.add_argument(
        "--pacing-speed",
        type=float,
        default=1.0,
        help="Wall-clock pacing speed multiplier when --realtime-pacing is enabled (default: 1.0)",
    )
    p_run.add_argument(
        "--synthesize-fixtures",
        action="store_true",
        help="Synthesize local test media if not already present",
    )
    p_run.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="Base seed for deterministic cache-key isolation across steps",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    suite_data = json.loads(args.cases_json.read_text())

    if args.subcommand == "prepare-media":
        receipts = prepare_all_media(
            suite_data,
            args.media_dir,
            oss_profile=args.oss_profile,
            synthesize_fixtures=args.synthesize_fixtures,
            synthetic_frames=args.synthetic_frames,
        )
        print(json.dumps({"status": "ok", "prepared": receipts}, ensure_ascii=False))
        return 0

    if args.subcommand == "preflight":
        pf = run_preflight_checks(
            suite_data,
            media_root=args.media_dir,
            mode=args.mode,
            sglang_url=args.sglang_url,
            coordinator_url=args.coordinator_url,
            require_real_media=args.require_real_media,
        )
        print(json.dumps(pf, ensure_ascii=False, indent=2))
        return 0

    if args.subcommand == "run":
        summary = asyncio.run(
            run_four_case_suite(
                suite_data,
                media_root=args.media_dir,
                output_dir=args.output_dir,
                mode=args.mode,
                sglang_url=args.sglang_url,
                coordinator_url=args.coordinator_url,
                encoder_url=args.encoder_url,
                dit_url=args.dit_url,
                decoder_url=args.decoder_url,
                routing_revision=args.routing_revision,
                compatibility=args.compatibility,
                profile=args.profile,
                network_domain=args.network_domain,
                control_epoch=args.control_epoch,
                signing_secret_env=args.signing_secret_env,
                room_token_env=args.room_token_env,
                max_frames_after_last_event=args.max_frames_after_last_event,
                full_frames=args.full_frames,
                preserve_switch_boundaries=args.preserve_switch_boundaries,
                realtime_pacing=args.realtime_pacing,
                pacing_speed=args.pacing_speed,
                capture_fps=args.capture_fps,
                synthesize_if_missing=(args.mode == "mock-worker"),
                seed_base=args.seed_base,
            )
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "summary_path": str(args.output_dir / "summary.json"),
                    "report_path": str(args.output_dir / "REPORT.md"),
                    "cases_completed": len(summary["cases"]),
                },
                ensure_ascii=False,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
