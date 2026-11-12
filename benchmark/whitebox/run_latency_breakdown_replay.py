#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Room-SGLang 白膜+按键自适应供帧与三角色（Encoder/DiT/Decoder）逐帧耗时分解回放测试。

支持两类工作流：
1. 从真实运行过的 WTP 归档案例中截取 ≤20s 的白膜视频（保留原始 30fps 供帧节奏）与精确对齐的玩家按键流；
2. 无需真实 Whitebox CLI，由进程内 Room `SourceSession` + `RecordedVideoEngine` 读取白膜视频并并发注入按键，
   自适应 SGLang 推理速度选帧供帧，采集每个入模产出帧在 Encoder / DiT / Decoder 的：
   - 等待攒帧耗时 (`wait_ms`)
   - 处理计算耗时 (`process_ms`)
   - 返回下游/调用方结果耗时 (`return_ms`)
   并调用 `latency_breakdown_visualizer.py` 产出 JSON 数据集、交互式 HTML 与高清概览 PNG。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
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

os.environ.setdefault("SGLANG_LIGHTWEIGHT_RUNTIME", "1")

# PyAV 14+ compatibility on local macOS dev environments where Flags.global_header was uppercased
try:
    import av.codec.context as _av_codec_ctx  # type: ignore[import-not-found]

    if not hasattr(_av_codec_ctx.Flags, "global_header") and hasattr(
        _av_codec_ctx.Flags, "GLOBAL_HEADER"
    ):
        _av_codec_ctx.Flags.global_header = _av_codec_ctx.Flags.GLOBAL_HEADER  # type: ignore[attr-defined]
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from latency_breakdown_visualizer import (
    build_timeline_dataset,
    render_html_report,
    render_png_overview,
)

from sglang.multimodal_gen.runtime.realtime.player_lifecycle import validate_sdk_cleanup
from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
    AsyncArchiveManager,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import (
    VideoSpec,
    capture_pts,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.recorded_engine import (
    RecordedVideoEngine,
    create_recorded_video_engine,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
    RenderJournal,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import SourceSession
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import MemoryRoomStore
from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig


class _BrowserUIStub:
    metadata = {"mode": "browser", "files": {}}

    async def close(self) -> None:
        pass


def _ref_format_from_path(path: Path | str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    if ext == "jpg":
        return "jpeg"
    if ext in {"png", "jpeg", "webp"}:
        return ext
    return "png"


def extract_wtp_clip(
    wtp_root: Path,
    clip_dir: Path,
    *,
    clip_start_s: float = 3.0,
    clip_duration_s: float = 16.0,
    capture_fps: int = 30,
    width: int = 832,
    height: int = 480,
) -> dict[str, Any]:
    """从 WTP 实际运行归档目录截取 ≤20s 的白膜视频、对齐按键序列与参考图。"""
    if clip_duration_s <= 0 or clip_duration_s > 20.0:
        raise ValueError("clip_duration_s must be within (0, 20.0] seconds")

    plan_path = wtp_root / "replay-input" / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"WTP replay plan.json not found: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    archive_root = wtp_root / "archive"
    game_dirs = sorted(p for p in archive_root.iterdir() if p.is_dir())
    if not game_dirs:
        raise FileNotFoundError(f"No archive game dir found under {archive_root}")
    game_dir = game_dirs[0]
    stream_dirs = sorted(p for p in (game_dir / "streams").iterdir() if p.is_dir())
    if not stream_dirs:
        raise FileNotFoundError(f"No stream dir found under {game_dir / 'streams'}")
    stream_dir = stream_dirs[0]

    src_mp4 = stream_dir / "whitebox.mp4"
    if not src_mp4.exists():
        src_mp4 = (
            wtp_root
            / "replay-input"
            / str(plan.get("source_video", "source-24fps.mp4"))
        )
    if not src_mp4.exists():
        raise FileNotFoundError(f"Whitebox source video not found: {src_mp4}")

    clip_dir.mkdir(parents=True, exist_ok=True)
    refs_out_dir = clip_dir / "references"
    refs_out_dir.mkdir(parents=True, exist_ok=True)

    # Copy references
    references_meta: list[dict[str, Any]] = []
    for idx, ref in enumerate(plan.get("references", [])):
        src_ref = wtp_root / "replay-input" / ref["path"]
        if not src_ref.exists():
            raise FileNotFoundError(f"Reference image missing: {src_ref}")
        fmt = _ref_format_from_path(src_ref)
        rel_dst = f"references/{idx:02d}.{fmt}"
        dst_ref = clip_dir / rel_dst
        data = src_ref.read_bytes()
        dst_ref.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        references_meta.append(
            {
                "id": ref.get("id", f"<Picture {idx + 1}>"),
                "role": ref.get("role", "scene"),
                "path": rel_dst,
                "format": fmt,
                "bytes": len(data),
                "sha256": sha,
            }
        )

    # Transcode / trim [clip_start_s, clip_start_s + clip_duration_s] at capture_fps
    out_mp4 = clip_dir / "whitebox.mp4"
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg binary is required to trim WTP whitebox clip")
    num_frames = int(round(clip_duration_s * capture_fps))
    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-y",
        "-ss",
        f"{clip_start_s:.3f}",
        "-t",
        f"{clip_duration_s:.3f}",
        "-i",
        str(src_mp4),
        "-vf",
        f"fps={capture_fps},scale={width}:{height}:flags=lanczos",
        "-frames:v",
        str(num_frames),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, timeout=120)

    # Extract aligned player input events from stream inputs.jsonl
    stream_inputs_path = stream_dir / "inputs.jsonl"
    start_us = int(round(clip_start_s * 1_000_000))
    end_us = int(round((clip_start_s + clip_duration_s) * 1_000_000))
    initial_held_keys: list[str] = []
    initial_held_buttons: list[int] = []
    extracted_events: list[dict[str, Any]] = []

    if stream_inputs_path.exists():
        for raw_line in stream_inputs_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            t0_us = int(row.get("start_time_us", 0))
            observed = (row.get("input") or {}).get("observed") or {}
            latest = observed.get("latest") or {}
            if t0_us < start_us:
                initial_held_keys = [str(k) for k in (latest.get("held_keys") or [])]
                initial_held_buttons = [
                    int(b) for b in (latest.get("held_buttons") or [])
                ]
                continue
            if t0_us > end_us:
                break
            rel_s = round(max(0.0, (t0_us - start_us) / 1_000_000.0), 6)
            for ev in observed.get("events") or []:
                payload = ev.get("payload")
                if not isinstance(payload, dict):
                    continue
                extracted_events.append(
                    {
                        "sequence": len(extracted_events),
                        "orig_sequence": ev.get("sequence"),
                        "rel_time_s": rel_s,
                        "payload": {
                            "held_keys": list(payload.get("held_keys") or []),
                            "held_buttons": list(payload.get("held_buttons") or []),
                            "events": list(payload.get("events") or []),
                        },
                    }
                )

    inputs_out_path = clip_dir / "inputs.jsonl"
    inputs_out_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in extracted_events)
        + ("\n" if extracted_events else ""),
        encoding="utf-8",
    )

    mp4_bytes = out_mp4.read_bytes()
    case_meta = {
        "schema_version": 1,
        "case_id": f"wtp-{game_dir.name[:8]}-{int(clip_duration_s)}s",
        "world_id": plan.get("archive_world_id", "interactive-production/wtp-case"),
        "archive_recording_id": game_dir.name,
        "archive_stream_id": stream_dir.name,
        "archive_segment_id": plan.get("archive_segment_id", ""),
        "model_revision": plan.get("model_revision", ""),
        "clip_start_s": clip_start_s,
        "duration_s": clip_duration_s,
        "capture_fps": capture_fps,
        "model_fps": 24,
        "width": width,
        "height": height,
        "frames": num_frames,
        "seed": int(plan.get("seed", 42)),
        "prompt": str(plan["prompt"]),
        "whitebox_mp4": {
            "path": "whitebox.mp4",
            "bytes": len(mp4_bytes),
            "sha256": hashlib.sha256(mp4_bytes).hexdigest(),
        },
        "references": references_meta,
        "inputs_path": "inputs.jsonl",
        "input_packets": len(extracted_events),
        "initial_held_keys": initial_held_keys,
        "initial_held_buttons": initial_held_buttons,
    }
    (clip_dir / "case_meta.json").write_text(
        json.dumps(case_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return case_meta


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


class MockStagePipelineWebSocket:
    """进程内模拟 Encoder [1,8,8] pixels、DiT [3,2] latents 和 Lynn 逐 latent 输出。"""

    def __init__(
        self,
        spec: VideoSpec,
        *,
        worker_epoch: str,
        player_expires_at: float,
        dit_chunk_ms: float = 215.0,
    ) -> None:
        self.spec = spec
        self.worker_epoch = worker_epoch
        self.player_expires_at = player_expires_at
        self.dit_chunk_ms = dit_chunk_ms
        self.outgoing: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.closed = False
        self.active_op_id: str | None = None
        self.active_kind: str | None = None
        self.active_session_id: str | None = None
        self.active_generation_id: str | None = None
        self.prompt_id: int | None = None

        self._enc_buffer: list[tuple[int, float, bytes]] = []
        self._dit_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._pipeline_task: asyncio.Task[None] | None = None
        self._chunk_counter = 0

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
            raise EOFError("MockStagePipelineWebSocket closed")
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
            self._enc_buffer.clear()
            self._chunk_counter = 0
            self._enqueue({"type": "bound.opened", "operation_id": op_id})
            if self.active_kind == "generate":
                self._pipeline_task = asyncio.create_task(
                    self._run_dit_decoder_loop(op_id)
                )
            return

        if mtype == "bound.cancel":
            op_id = msg["operation_id"]
            self._enqueue(
                {"type": "bound.completed", "operation_id": op_id, "cleaned": True}
            )
            return

        if mtype == "bound.data":
            op_id = msg["operation_id"]
            inner = msgspec.msgpack.decode(msg["payload"])
            if self.active_kind == "prefill":
                await asyncio.sleep(0.015)
                req_id = inner["request_id"]
                cache_key = hashlib.sha256(req_id.encode()).hexdigest()
                self._enqueue_child_data(
                    op_id,
                    {
                        "type": "prefill.memory_ready",
                        "request_id": req_id,
                        "session_id": self.active_session_id,
                        "cache_key": cache_key,
                        "cache_result": "populated",
                        "hit": False,
                        "prefill_to_memory_ms": 15.0,
                        "timings": {"elapsed_host_ms": 15.0},
                    },
                )
                self._enqueue(
                    {"type": "bound.completed", "operation_id": op_id, "cleaned": True}
                )
                self._enqueue(
                    {
                        "type": "prefill.persisted",
                        "request_id": req_id,
                        "session_id": self.active_session_id,
                        "cache_key": cache_key,
                        "cache_result": "populated",
                        "hit": False,
                        "memory_ready": True,
                        "prefill_to_memory_ms": 15.0,
                        "prefill_to_persisted_ms": 18.0,
                    },
                )
            else:
                await self._handle_generate_data(op_id, inner)

    def _target_enc_group_size(self, first_frame_idx: int) -> int:
        # Each codec clip has 17 pixels -> 5 latents, encoded in 1/8/8 groups.
        return 1 if first_frame_idx % 17 == 0 else 8

    async def _flush_encoder_group(self) -> None:
        if not self._enc_buffer:
            return
        batch = list(self._enc_buffer)
        self._enc_buffer.clear()
        enc_start = time.perf_counter()
        proc_s = 0.008 if len(batch) == 1 else 0.012
        ret_s = 0.0015
        await asyncio.sleep(proc_s + ret_s)
        dit_arrived = time.perf_counter()
        enc_frames: list[dict[str, Any]] = []
        for idx, arrived_s, rgb in batch:
            enc_frames.append(
                {
                    "index": idx,
                    "rgb": rgb,
                    "encoder": {
                        "wait_ms": round(max(0.0, (enc_start - arrived_s) * 1000.0), 3),
                        "process_ms": round(proc_s * 1000.0, 3),
                        "return_ms": round(ret_s * 1000.0, 3),
                    },
                    "dit_arrived_s": dit_arrived,
                }
            )
        # A VAE group of 8 pixels emits two latents. Keep latent boundaries,
        # including the one-pixel head of every 17-pixel clip and the EOS tail.
        offset = 0
        while offset < len(enc_frames):
            count = 1 if enc_frames[offset]["index"] % 17 == 0 else 4
            await self._dit_queue.put({"frames": enc_frames[offset : offset + count]})
            offset += count

    async def _run_dit_decoder_loop(self, op_id: str) -> None:
        pending_latents: list[dict[str, Any]] = []
        while not self.closed:
            item = await self._dit_queue.get()
            if item is None:
                if pending_latents:
                    await self._execute_dit_and_decoder(op_id, pending_latents)
                    pending_latents.clear()
                self._enqueue_child_data(op_id, {"type": "session.completed"})
                self._enqueue(
                    {"type": "bound.completed", "operation_id": op_id, "cleaned": True}
                )
                return

            pending_latents.append(item)
            # Match the current [3,2] latent schedule, split_first_latent=False.
            needed_latents = (3, 2)[self._chunk_counter % 2]
            if len(pending_latents) >= needed_latents:
                to_run = list(pending_latents[:needed_latents])
                del pending_latents[:needed_latents]
                await self._execute_dit_and_decoder(op_id, to_run)

    async def _execute_dit_and_decoder(
        self, op_id: str, latent_groups: list[dict[str, Any]]
    ) -> None:
        all_frames = [f for g in latent_groups for f in g["frames"]]
        start_idx = all_frames[0]["index"]
        count = len(all_frames)
        chunk_idx = self._chunk_counter
        self._chunk_counter += 1

        dit_start = time.perf_counter()
        # Mild periodic compute variation to exercise adaptive source frame skipping
        wave = 1.0 + (
            0.28 if (chunk_idx % 5 == 2) else (0.10 if chunk_idx % 3 == 0 else -0.05)
        )
        dit_proc_s = 0.095 if count == 1 else (self.dit_chunk_ms / 1000.0) * wave
        dit_ret_s = 0.0024
        await asyncio.sleep(dit_proc_s + dit_ret_s)

        breakdown: list[dict[str, Any]] = []
        # Lynn consumes one latent at a time, publishing each valid RGB slice
        # before starting the next one. Do not coalesce the whole AR chunk.
        for latent_group in latent_groups:
            frames = latent_group["frames"]
            slice_start = frames[0]["index"]
            slice_count = len(frames)
            dec_wait_s = 0.0006
            dec_proc_s = 0.016 if slice_count == 1 else 0.017
            await asyncio.sleep(dec_wait_s + dec_proc_s)
            t_before_send = time.perf_counter()
            self._enqueue_child_data(
                op_id,
                {
                    "type": "frame_batch",
                    "output_frame_start": slice_start,
                    "num_frames": slice_count,
                    "pts_us": self.spec.pts(slice_start),
                    "width": self.spec.width,
                    "height": self.spec.height,
                    "format": "rgb24",
                    "encoding": "raw",
                    "payload": b"".join(f["rgb"] for f in frames),
                    "event_id": self.prompt_id,
                    "sink_chunks": 2,
                    "sink_chunk_ids": [0, 1],
                    "source_alignment": [
                        {
                            "source_frame_start": slice_start,
                            "output_frame_start": slice_start,
                            "num_frames": slice_count,
                        }
                    ],
                },
            )
            dec_stage = {
                "wait_ms": round(dec_wait_s * 1000.0, 3),
                "process_ms": round(dec_proc_s * 1000.0, 3),
                "return_ms": round(
                    max(1.2, (time.perf_counter() - t_before_send) * 1000.0 + 1.8), 3
                ),
            }
            for f in frames:
                breakdown.append(
                    {
                        "output_frame": f["index"],
                        "encoder": dict(f["encoder"]),
                        "dit": {
                            "wait_ms": round(
                                max(0.0, (dit_start - f["dit_arrived_s"]) * 1000.0), 3
                            ),
                            "process_ms": round(dit_proc_s * 1000.0, 3),
                            "return_ms": round(dit_ret_s * 1000.0, 3),
                        },
                        "decoder": dict(dec_stage),
                    }
                )

        summary: dict[str, float] = {}
        for role in ("encoder", "dit", "decoder"):
            for metric in ("wait_ms", "process_ms", "return_ms"):
                vals = [row[role][metric] for row in breakdown]
                summary[f"{role}_{metric}"] = round(sum(vals) / len(vals), 3)

        self._enqueue_child_data(
            op_id,
            {
                "type": "chunk_telemetry",
                "chunk_index": chunk_idx,
                "output_frame_start": start_idx,
                "num_frames": count,
                "frame_stage_breakdown": breakdown,
                **summary,
            },
        )
        end_idx = start_idx + count
        self._enqueue_child_data(
            op_id,
            {
                "type": "source.consumed",
                "source_frame_end": end_idx,
                "source_bytes_end": end_idx * self.spec.frame_bytes,
            },
        )

    async def _handle_generate_data(self, op_id: str, inner: dict[str, Any]) -> None:
        if inner.get("type") == "init":
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
            return

        kind = inner.get("kind")
        if kind == "source.append":
            payload = inner["payload"]
            idx = int(payload["source_frame_start"])
            rgb = payload.get("data") or (bytes([100]) * self.spec.frame_bytes)
            self._enqueue_child_data(
                op_id,
                {
                    "type": "source.accepted",
                    "source_frame_start": idx,
                    "source_frame_end": idx + 1,
                },
            )
            self._enc_buffer.append((idx, time.perf_counter(), rgb))
            first_in_group = self._enc_buffer[0][0]
            if len(self._enc_buffer) >= self._target_enc_group_size(first_in_group):
                await self._flush_encoder_group()
            return

        if kind == "source.end":
            if self._enc_buffer:
                await self._flush_encoder_group()
            await self._dit_queue.put(None)

    async def close(self) -> None:
        self.closed = True
        self.outgoing.put_nowait(None)
        if self._pipeline_task and not self._pipeline_task.done():
            self._pipeline_task.cancel()
            await asyncio.gather(self._pipeline_task, return_exceptions=True)


class MockStageBoundLease:
    def __init__(self, spec: VideoSpec, *, dit_chunk_ms: float = 215.0) -> None:
        self.spec = spec
        self.dit_chunk_ms = dit_chunk_ms
        self.operation_path = "/v1/realtime_video/bound"
        self.cleanup_timeout_s = 30
        self.sdk_cleanup_provider: Any = None
        self.strict_release = True
        self.resource_state = "none"

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
        ws = MockStagePipelineWebSocket(
            self.spec,
            worker_epoch=worker_epoch,
            player_expires_at=assignment.player_expires_at,
            dit_chunk_ms=self.dit_chunk_ms,
        )
        try:
            yield ws, assignment
        finally:
            await ws.close()
            if self.sdk_cleanup_provider is not None:
                receipt = await self.sdk_cleanup_provider(assignment)
                validate_sdk_cleanup(assignment, receipt)
            self.resource_state = "released"


def _create_real_lease(
    *,
    coordinator_url: str | None,
    spec: VideoSpec,
    trace_id: str,
    segment: dict[str, Any],
):
    from source_replay_role_pool import current_pool

    pool = current_pool.get()
    if pool is not None:
        return pool.lease(spec, trace_id, segment)

    import websockets

    try:
        from websockets.asyncio.client import connect as ws_connect
    except ImportError:
        ws_connect = websockets.connect

    from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import (
        create_model_leases,
    )

    revision = os.environ.get("WHITEBOX_ROUTING_REVISION", "real-sglang")
    profile_hint = segment.get("model_size") or f"{spec.width}x{spec.height}"
    model_session_id = segment.get("model_session_id") or uuid4().hex
    generation_id = segment.get("generation_id") or uuid4().hex

    if os.environ.get("WHITEBOX_MODEL_BACKEND") == "role_pool":
        return create_model_leases(
            None,
            ws_connect,
            user_id="latency-breakdown-runner",
            profile_hint=profile_hint,
            trace_id=segment.get("segment_id") or trace_id,
            model_revision=revision,
            vae_fingerprint="local",
            max_bytes=64 * 1024 * 1024,
            strict_release=True,
            session_id=model_session_id,
            generation_id=generation_id,
        )

    if coordinator_url:
        from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
            HTTPCoordinatorClient,
        )

        coord_client = HTTPCoordinatorClient(coordinator_url)
        return create_model_leases(
            coord_client,
            ws_connect,
            user_id="latency-breakdown-runner",
            profile_hint=profile_hint,
            trace_id=segment.get("segment_id") or trace_id,
            model_revision=revision,
            vae_fingerprint="local",
            max_bytes=64 * 1024 * 1024,
            strict_release=True,
            session_id=model_session_id,
            generation_id=generation_id,
        )

    raise ValueError(
        "Real-sglang mode requires --encoder-url/--dit-url/--decoder-url, "
        "--coordinator-url, or WHITEBOX_MODEL_BACKEND=role_pool"
    )


async def run_replay_session(
    *,
    clip_dir: Path,
    output_dir: Path,
    mode: str,
    sampling: str = "adaptive",
    pacing_speed: float = 1.0,
    dit_chunk_ms: float = 215.0,
    num_inference_steps: int | None = None,
    coordinator_url: str | None = None,
    encoder_url: str | None = None,
    dit_url: str | None = None,
    decoder_url: str | None = None,
    profile: str | None = None,
    compatibility: str | None = None,
    network_domain: str | None = None,
    control_epoch: str | None = None,
    resolution: int | None = None,
    server_revision: str | None = None,
    return_stage_timing: bool = True,
) -> dict[str, Any]:
    """执行一次完整的白膜+按键自适应供帧回放并生成逐帧耗时分解数据与图表。"""
    case_meta = json.loads((clip_dir / "case_meta.json").read_text(encoding="utf-8"))
    mp4_path = clip_dir / case_meta["whitebox_mp4"]["path"]
    inputs_path = clip_dir / case_meta["inputs_path"]
    input_events = [
        json.loads(line)
        for line in inputs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    width = int(case_meta.get("width", 832))
    height = int(case_meta.get("height", 480))
    capture_fps = int(case_meta.get("capture_fps", 30))
    total_source_frames = int(
        case_meta.get("frames", round(case_meta["duration_s"] * capture_fps))
    )
    if resolution is not None and height != resolution:
        raise ValueError(
            f"Clip height {height} does not match requested resolution {resolution}"
        )
    if height not in (480, 736):
        raise ValueError("Latency replay requires an explicit 480 or 736 pixel height")
    if (
        mode == "real-sglang"
        and coordinator_url
        and (encoder_url or dit_url or decoder_url)
    ):
        raise ValueError("Choose either a Coordinator or direct Workers, not both")
    spec = VideoSpec(width=width, height=height, fps=24)

    os.environ["WHITEBOX_BOUND_COMPUTE"] = "1"
    output_dir.mkdir(parents=True, exist_ok=True)
    spool_dir = output_dir / "spool"
    if spool_dir.exists():
        shutil.rmtree(spool_dir)
    spool_dir.mkdir(parents=True, exist_ok=True)

    pool_cm = None
    if mode == "real-sglang" and (
        coordinator_url or encoder_url or dit_url or decoder_url
    ):
        if not coordinator_url and not (encoder_url and dit_url and decoder_url):
            raise ValueError(
                "Direct worker mode requires all of --encoder-url, --dit-url, --decoder-url"
            )
        from source_replay_role_pool import SuiteRolePool

        pool_cm = SuiteRolePool(
            coordinator_url=coordinator_url,
            revision=os.environ.get("WHITEBOX_ROUTING_REVISION"),
            video_size=(width, height),
            workers={
                "encoder": encoder_url,
                "dit": dit_url,
                "decoder": decoder_url,
            }
            if not coordinator_url
            else None,
            profile=profile or (str(height) if coordinator_url else None),
            compatibility=compatibility,
            network_domain=network_domain,
            control_epoch=control_epoch,
        )

    async def _execute_inside_pool() -> dict[str, Any]:
        created_engines: list[RecordedVideoEngine] = []
        engine_factory = create_recorded_video_engine(
            mp4_path,
            max_frames=total_source_frames,
            capture_fps=capture_fps,
            realtime_pacing=True,
            pacing_speed=pacing_speed,
            loop=False,
            on_created=created_engines.append,
        )

        archives = AsyncArchiveManager(
            recorder_slots=4,
            queue_depth=512,
            queue_bytes=256 * 1024 * 1024,
            max_age=90.0,
            reserve_bytes=0,
        )
        case_id = case_meta["case_id"]
        journal = RenderJournal(
            MemoryRoomStore(),
            user_id="latency-user",
            source_session_id=f"src-{case_id}-{uuid4().hex[:6]}",
            source_epoch="epoch-1",
            owner_epoch="room-1",
        )
        await journal.create()

        def _make_leases(_user_id: str, seg: dict[str, Any]):
            if mode == "real-sglang":
                return _create_real_lease(
                    coordinator_url=coordinator_url,
                    spec=spec,
                    trace_id=case_id,
                    segment=seg,
                )
            return MockStageBoundLease(spec, dit_chunk_ms=dit_chunk_ms)

        host = SimpleNamespace(
            runtime=SimpleNamespace(players={}),
            make_segment_leases=_make_leases,
            sessions={},
        )
        in_memory_selected: list[dict[str, Any]] = []

        class _TelemetrySourceSession(SourceSession):
            async def _open_engine(self, world_id, world, init):
                init["return_stage_timing"] = return_stage_timing
                return await super()._open_engine(world_id, world, init)

            async def segment_status(self, seg, st, **fields):
                if fields.get("phase") == "protocol_ready" and not getattr(
                    seg, "_latency_hook_installed", False
                ):
                    seg._latency_hook_installed = True
                    orig_media = seg._media

                    async def _hooked_media(
                        stream, index, frame, encoded, captured, recording_index=None
                    ):
                        if stream == "model":
                            prov = frame.provenance or {}
                            mt = prov.get("model_timing") or {}
                            in_memory_selected.append(
                                {
                                    "received_mono_s": time.perf_counter(),
                                    "source_frame": int(frame.index),
                                    "model_frame": int(
                                        prov.get(
                                            "model_source_frame",
                                            recording_index
                                            if recording_index is not None
                                            else index,
                                        )
                                    ),
                                    "pts_us": int(capture_pts(frame)),
                                    "prefill_hold_us": prov.get("prefill_hold_us"),
                                    "prompt_version": int(
                                        (frame.rendering or {}).get("prompt_version", 1)
                                    ),
                                    "model_timing": dict(mt),
                                }
                            )
                        await orig_media(
                            stream, index, frame, encoded, captured, recording_index
                        )

                    seg._media = _hooked_media
                await super().segment_status(seg, st, **fields)

        eff_steps = (
            num_inference_steps
            if num_inference_steps is not None
            else int(
                os.environ.get(
                    "WHITEBOX_INFERENCE_STEPS",
                    "2" if mode == "real-sglang" else "4",
                )
            )
        )
        source = _TelemetrySourceSession(
            host,
            journal,
            source_epoch="epoch-1",
            user_id="latency-user",
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
                model_source_sampling=sampling,
                num_inference_steps=eff_steps,
                model_input_frames=int(
                    os.environ.get("WHITEBOX_MODEL_INPUT_FRAMES", "34")
                ),
                startup_seconds=240 if mode == "real-sglang" else 30,
                play_seconds=600,
                drain_seconds=60 if mode == "real-sglang" else 15,
            ),
            archives=archives,
        )

        messages: list[dict[str, Any]] = []
        error_messages: list[dict[str, Any]] = []
        prefill_receipts: list[dict[str, Any]] = []
        model_timing_messages: list[dict[str, Any]] = []
        frame_batches_count = 0
        req_counter = 0
        default_timeout = 240.0 if mode == "real-sglang" else 45.0

        async def _consume() -> None:
            nonlocal frame_batches_count
            while True:
                raw = await source.recv()
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
                    frame_batches_count += int(msg.get("num_frames", 1))
                elif mtype in {
                    "prefill.memory_ready",
                    "prefill.persisted",
                    "prefill.source_resumed",
                } or (mtype == "prefill.failed" and msg.get("memory_ready") is True):
                    prefill_receipts.append(msg)
                elif mtype == "whitebox.model_timing":
                    model_timing_messages.append(msg)

        async def _wait_for(predicate, timeout_s: float | None = None):
            eff_timeout = default_timeout if timeout_s is None else timeout_s
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
            await source.send(
                msgspec.msgpack.encode(
                    {
                        "type": "command",
                        "op": op,
                        "request_id": req_id,
                        "source_session_id": source.session_id,
                        "source_epoch": source.source_epoch,
                        **fields,
                    }
                )
            )
            return req_id

        async def _inject_player_inputs(engine: RecordedVideoEngine) -> int:
            sent_packets = 0
            origin = engine._clock_origin or time.monotonic()
            for item in input_events:
                if engine.closed:
                    break
                rel_s = float(item.get("rel_time_s", 0.0))
                target_mono = origin + (rel_s / pacing_speed)
                delay = target_mono - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                if engine.closed:
                    break
                await source.send(
                    msgspec.msgpack.encode(
                        {
                            "type": "event",
                            "kind": "input",
                            "sequence": sent_packets,
                            "payload": item["payload"],
                        }
                    )
                )
                sent_packets += 1
            return sent_packets

        consumer_task = asyncio.create_task(_consume())
        input_task: asyncio.Task[int] | None = None
        try:
            world_short = str(case_meta["world_id"]).split("/")[-1]
            await source.send(
                msgspec.msgpack.encode(
                    {
                        "type": "init",
                        "whitebox": {"world_id": world_short},
                        "prompt": case_meta["prompt"],
                    }
                )
            )
            await _wait_for(lambda: source.ready is not None and source.ready.is_set())
            assert source.model_binding is not None
            await source.model_binding.wait_ready()

            # 1. Hold F0 via render.prepare
            state = await journal.read()
            prep_req_id = await _send_cmd(
                "render.prepare",
                expected_render_revision=state["render_revision"],
            )
            prep_done = await _wait_for(
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

            # 2. Upload reference images
            initial_refs: list[dict[str, Any]] = []
            for ref in case_meta.get("references", []):
                req_counter += 1
                up_req_id = f"{case_id}-upload-{req_counter}"
                ref_bytes = (clip_dir / ref["path"]).read_bytes()
                await source.send(
                    msgspec.msgpack.encode(
                        {
                            "type": "asset.upload",
                            "version": 2,
                            "request_id": up_req_id,
                            "source_session_id": source.session_id,
                            "source_epoch": source.source_epoch,
                            "format": ref["format"],
                            "payload": ref_bytes,
                        }
                    )
                )
                up_done = await _wait_for(
                    lambda: next(
                        (
                            m
                            for m in messages
                            if m.get("type") == "operation.result"
                            and m.get("request_id") == up_req_id
                            and m.get("state") == "complete"
                        ),
                        None,
                    )
                )
                initial_refs.append(
                    {
                        "id": ref["id"],
                        "asset_id": up_done["asset"]["asset_id"],
                        "role": ref.get("role", "scene"),
                    }
                )

            # 3. Prefill conditioning on F0
            req_counter += 1
            pf_req_id = f"{case_id}-pf-{req_counter}"
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
                            "context_id": prep_done["context_id"],
                            "prompt": case_meta["prompt"],
                            "seed": int(case_meta["seed"]),
                            "references": initial_refs,
                        },
                    }
                )
            )
            await _wait_for(
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

            # 4. Start streaming segment & resume whitebox supply + key injection
            state = await journal.read()
            start_wall = time.perf_counter()
            await _send_cmd(
                "render.start",
                expected_render_revision=state["render_revision"],
                context_id=prep_done["context_id"],
                start_boundary={"source_frame_index": prep_done["source_frame_index"]},
                prompt=case_meta["prompt"],
                seed=int(case_meta["seed"]),
                references=initial_refs,
            )

            await _wait_for(
                lambda: next(
                    (
                        m
                        for m in prefill_receipts
                        if m.get("type") == "prefill.source_resumed"
                        and m.get("context_id") == prep_done["context_id"]
                    ),
                    None,
                )
            )
            assert created_engines, "RecordedVideoEngine should be created"
            engine = created_engines[-1]
            input_task = asyncio.create_task(_inject_player_inputs(engine))

            await _wait_for(
                lambda: (
                    source.current_segment is not None
                    and source.current_segment.first_decodable
                    and frame_batches_count > 0
                )
            )
            segment = source.current_segment
            assert segment is not None
            ttff_ms = round((time.perf_counter() - start_wall) * 1000.0, 3)

            # Wait until all whitebox clip frames have been supplied by RecordedVideoEngine
            supply_timeout = (case_meta["duration_s"] / max(0.1, pacing_speed)) + 45.0
            await _wait_for(
                lambda: source.submitted >= total_source_frames,
                timeout_s=supply_timeout,
            )
            injected_packets = await input_task
            # Brief settle so the final supplied frame can be considered before drain
            await asyncio.sleep(0.08)

            # 5. Drain in-flight frames cleanly via render.end(mode="drain")
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

            seg_dir = segment.recording.directory
            src_dir = source.recording.directory
            seg_manifest = dict(segment.recording.manifest)
            paused_offset_s = engine._paused_pts_offset_us / 1_000_000.0
            await source.unbind("LATENCY_REPLAY_COMPLETED")

        finally:
            if input_task and not input_task.done():
                input_task.cancel()
                await asyncio.gather(input_task, return_exceptions=True)
            await source.close()
            consumer_task.cancel()
            await asyncio.gather(consumer_task, return_exceptions=True)
            await archives.close()

        # Parse all supplied whitebox source frames (in-memory whitebox.states first, disk fallback)
        source_frames_raw: list[dict[str, Any]] = []
        state_msgs = [m for m in messages if m.get("type") == "whitebox.states"]
        if state_msgs:
            for row in state_msgs:
                idx = int(row["source_frame_index"])
                raw_pts_s = float(row["pts_us"]) / 1_000_000.0
                aligned_s = 0.0 if idx == 0 else max(0.0, raw_pts_s - paused_offset_s)
                obs_latest = ((row.get("observed_input") or {}).get("latest")) or {}
                source_frames_raw.append(
                    {
                        "source_frame_index": idx,
                        "capture_time_s": aligned_s,
                        "pts_s": round(idx / capture_fps, 6),
                        "sdk_frame": idx,
                        "held_keys": list(obs_latest.get("held_keys") or []),
                    }
                )
        else:
            src_states_path = src_dir / "source-states.jsonl"
            if src_states_path.exists():
                for line in src_states_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    idx = int(row["source_frame_index"])
                    raw_pts_s = float(row["pts_us"]) / 1_000_000.0
                    aligned_s = (
                        0.0 if idx == 0 else max(0.0, raw_pts_s - paused_offset_s)
                    )
                    obs_latest = ((row.get("observed_input") or {}).get("latest")) or {}
                    source_frames_raw.append(
                        {
                            "source_frame_index": idx,
                            "capture_time_s": aligned_s,
                            "pts_s": round(idx / capture_fps, 6),
                            "sdk_frame": idx,
                            "held_keys": list(obs_latest.get("held_keys") or []),
                        }
                    )

        # Parse selected & produced frames (in-memory hook first, disk states.jsonl fallback)
        selected_frames_raw: list[dict[str, Any]] = []
        if in_memory_selected:
            for item in in_memory_selected:
                mt = item["model_timing"]
                s_idx = item["source_frame"]
                m_idx = item["model_frame"]
                raw_pts_s = float(item["pts_us"]) / 1_000_000.0
                aligned_s = 0.0 if s_idx == 0 else max(0.0, raw_pts_s - paused_offset_s)
                src_to_sel_ms = float(mt.get("source_to_selection_ms", 0.0))
                if s_idx == 0 and item.get("prefill_hold_us"):
                    src_to_sel_ms = max(
                        0.0, src_to_sel_ms - float(item["prefill_hold_us"]) / 1000.0
                    )
                selected_frames_raw.append(
                    {
                        "source_frame": s_idx,
                        "model_frame": m_idx,
                        "output_batch_start": mt.get("output_batch_start"),
                        "output_batch_frames": mt.get("output_batch_frames"),
                        "prompt_version": item["prompt_version"],
                        "capture_time_s": aligned_s,
                        "source_to_selection_ms": src_to_sel_ms,
                        "selection_to_dispatch_ms": float(
                            mt.get("selection_to_dispatch_ms", 0.0)
                        ),
                        "dispatch_to_raw_output_ms": float(
                            mt.get("dispatch_to_raw_output_ms", 0.0)
                        ),
                        "source_to_raw_output_ms": (
                            src_to_sel_ms
                            + float(mt.get("selection_to_dispatch_ms", 0.0))
                            + float(mt.get("dispatch_to_raw_output_ms", 0.0))
                        ),
                        "input_admission_wait_ms": float(
                            mt.get("input_admission_wait_ms", 0.0)
                        ),
                        "input_capacity_blocked_ms": dict(
                            mt.get("input_capacity_blocked_ms") or {}
                        ),
                    }
                )
        else:
            seg_states_path = seg_dir / "states.jsonl"
            if seg_states_path.exists():
                for line in seg_states_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    prov = row.get("source") or {}
                    mt = prov.get("model_timing") or {}
                    s_idx = int(prov.get("source_frame_index", row["frame"]))
                    m_idx = int(prov.get("model_source_frame", row["frame"]))
                    raw_pts_s = float(row["pts_us"]) / 1_000_000.0
                    aligned_s = (
                        0.0 if s_idx == 0 else max(0.0, raw_pts_s - paused_offset_s)
                    )
                    src_to_sel_ms = float(mt.get("source_to_selection_ms", 0.0))
                    if s_idx == 0 and prov.get("prefill_hold_us"):
                        src_to_sel_ms = max(
                            0.0, src_to_sel_ms - float(prov["prefill_hold_us"]) / 1000.0
                        )
                    selected_frames_raw.append(
                        {
                            "source_frame": s_idx,
                            "model_frame": m_idx,
                            "output_batch_start": mt.get("output_batch_start"),
                            "output_batch_frames": mt.get("output_batch_frames"),
                            "prompt_version": int(
                                (row.get("rendering") or {}).get("prompt_version", 1)
                            ),
                            "capture_time_s": aligned_s,
                            "source_to_selection_ms": src_to_sel_ms,
                            "selection_to_dispatch_ms": float(
                                mt.get("selection_to_dispatch_ms", 0.0)
                            ),
                            "dispatch_to_raw_output_ms": float(
                                mt.get("dispatch_to_raw_output_ms", 0.0)
                            ),
                            "source_to_raw_output_ms": (
                                src_to_sel_ms
                                + float(mt.get("selection_to_dispatch_ms", 0.0))
                                + float(mt.get("dispatch_to_raw_output_ms", 0.0))
                            ),
                            "input_admission_wait_ms": float(
                                mt.get("input_admission_wait_ms", 0.0)
                            ),
                            "input_capacity_blocked_ms": dict(
                                mt.get("input_capacity_blocked_ms") or {}
                            ),
                        }
                    )

        # Collect frame_stage_breakdown from segment manifest & live whitebox.model_timing messages
        breakdown_by_frame: dict[int, dict[str, Any]] = {}
        for msg in model_timing_messages:
            cidx = msg.get("chunk_index")
            for item in msg.get("frame_stage_breakdown") or []:
                if isinstance(item, dict) and type(item.get("output_frame")) is int:
                    breakdown_by_frame[item["output_frame"]] = dict(
                        item, chunk_index=cidx
                    )
        for item in seg_manifest.get("frame_stage_breakdown") or []:
            if isinstance(item, dict) and type(item.get("output_frame")) is int:
                breakdown_by_frame.setdefault(item["output_frame"], dict(item))

        for sel in selected_frames_raw:
            b = breakdown_by_frame.get(sel["model_frame"])
            if b and "chunk_index" in b:
                sel["chunk_index"] = b["chunk_index"]

        dataset = build_timeline_dataset(
            case_meta=case_meta,
            source_frames=source_frames_raw,
            selected_frames=selected_frames_raw,
            frame_stage_breakdown=list(breakdown_by_frame.values()),
            input_events=input_events,
            mode=mode,
        )
        dataset["summary"]["request_width"] = width
        dataset["summary"]["request_height"] = height
        dataset["summary"]["resolution"] = str(height)
        dataset["summary"]["capture_fps"] = capture_fps
        dataset["summary"]["model_fps"] = spec.fps
        dataset["summary"]["server_revision_supplied"] = server_revision
        dataset["summary"]["case_meta_sha256"] = hashlib.sha256(
            (clip_dir / "case_meta.json").read_bytes()
        ).hexdigest()
        dataset["summary"]["inputs_sha256"] = hashlib.sha256(
            inputs_path.read_bytes()
        ).hexdigest()
        dataset["summary"]["tool_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
        dataset["summary"]["tool_worktree_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
            ).strip()
        )
        dataset["summary"]["session_ttff_ms"] = ttff_ms
        received = [r["received_mono_s"] for r in in_memory_selected]
        if received:
            output_elapsed_s = max(received) - start_wall
            dataset["summary"]["room_output_elapsed_s"] = output_elapsed_s
            dataset["summary"]["room_output_fps"] = len(received) / output_elapsed_s
            dataset["summary"]["room_output_frames"] = len(received)
            dataset["summary"]["room_output_timing_boundary"] = (
                "render.start to last Room model media callback; excludes prefill and HTML/PNG rendering"
            )
        dataset["summary"]["injected_input_packets"] = injected_packets
        dataset["summary"]["engine_received_inputs"] = len(engine.inputs)
        dataset["summary"]["model_timing_chunks"] = len(model_timing_messages)
        dataset["summary"]["sampling_policy"] = sampling
        dataset["summary"]["return_stage_timing"] = return_stage_timing

        data_json = output_dir / "latency-breakdown-data.json"
        html_path = output_dir / "index.html"
        png_path = output_dir / "latency-breakdown-overview.png"
        data_json.write_text(
            json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        render_html_report(dataset, html_path)
        render_png_overview(dataset, png_path)

        # Clean up heavy spool directory after extracting telemetry
        shutil.rmtree(spool_dir, ignore_errors=True)
        return {
            "data_json": str(data_json),
            "html": str(html_path),
            "png": str(png_path),
            "summary": dataset["summary"],
        }

    if pool_cm is not None:
        async with pool_cm as pool:
            res = await _execute_inside_pool()
            await pool.wait_idle(timeout_s=20)
            return res
    return await _execute_inside_pool()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Room-SGLang 白膜+按键自适应供帧与 Encoder/DiT/Decoder 逐帧耗时分解回放器"
    )
    p.add_argument(
        "--extract-from-wtp",
        type=Path,
        default=None,
        help="从本地 WTP 归档目录截取 ≤20s 的白膜视频、按键流与参考图到 --clip-dir",
    )
    p.add_argument(
        "--clip-start-s",
        type=float,
        default=3.0,
        help="截取起始时间（秒，默认 3.0s）",
    )
    p.add_argument(
        "--clip-duration-s",
        type=float,
        default=16.0,
        help="截取时长（秒，必须 ≤20.0s，默认 16.0s）",
    )
    p.add_argument(
        "--capture-fps",
        type=int,
        default=30,
        help="白膜供给帧率（默认 30fps，复现真实 CLI 高于 24fps 模型的自适应供帧）",
    )
    p.add_argument(
        "--clip-dir",
        type=Path,
        required=True,
        help="截取后或已准备好的轻量测试素材目录（含 whitebox.mp4, inputs.jsonl, case_meta.json）",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（保存 latency-breakdown-data.json, index.html, latency-breakdown-overview.png）",
    )
    p.add_argument(
        "--extract-only",
        action="store_true",
        help="仅执行 WTP 片段截取，不启动回放",
    )
    p.add_argument(
        "--mode",
        choices=("mock", "real-sglang"),
        default="mock",
        help="运行模式：mock（本地三阶段流水线模拟）或 real-sglang（真实 GPU SGLang 集群）",
    )
    p.add_argument(
        "--sampling",
        choices=("adaptive", "latest"),
        default="adaptive",
        help="Room 源帧采样策略（默认 adaptive）",
    )
    p.add_argument(
        "--pacing-speed",
        type=float,
        default=1.0,
        help="回放倍速（默认 1.0x 实时供帧）",
    )
    p.add_argument(
        "--mock-dit-chunk-ms",
        type=float,
        default=215.0,
        help="mock 模式下 8 帧 DiT chunk 的基准耗时（毫秒）",
    )
    p.add_argument(
        "--num-inference-steps",
        type=int,
        choices=(1, 2, 4),
        default=None,
        help="推理去噪步数（real-sglang 默认 2 对应 1000,500；也可通过 WHITEBOX_INFERENCE_STEPS 设置）",
    )
    p.add_argument(
        "--resolution",
        type=int,
        choices=(480, 736),
        default=None,
        help="Assert clip height; never silently resize an existing case",
    )
    p.add_argument("--width", type=int, default=832, help="WTP extraction width")
    p.add_argument(
        "--height",
        type=int,
        choices=(480, 736),
        default=480,
        help="WTP extraction height",
    )
    p.add_argument(
        "--server-revision",
        default=None,
        help="Deployer-supplied server SHA (not auto-verified)",
    )
    p.add_argument(
        "--stage-timing",
        choices=("on", "off"),
        default="on",
        help="Request-scoped detailed timing; use off for the same-code baseline",
    )
    p.add_argument("--coordinator-url", type=str, default=None)
    p.add_argument("--encoder-url", type=str, default=None)
    p.add_argument("--dit-url", type=str, default=None)
    p.add_argument("--decoder-url", type=str, default=None)
    p.add_argument("--profile", type=str, default=None)
    p.add_argument("--compatibility", type=str, default=None)
    p.add_argument("--network-domain", type=str, default=None)
    p.add_argument("--control-epoch", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.extract_from_wtp is not None:
        if args.resolution is not None and args.height != args.resolution:
            raise ValueError("Extraction height must match --resolution")
        meta = extract_wtp_clip(
            args.extract_from_wtp,
            args.clip_dir,
            clip_start_s=args.clip_start_s,
            clip_duration_s=args.clip_duration_s,
            capture_fps=args.capture_fps,
            width=args.width,
            height=args.height,
        )
        print(
            f"[extract] 已截取 {meta['duration_s']}s 白膜片段: "
            f"{meta['frames']} 帧 @ {meta['capture_fps']}fps, "
            f"{meta['input_packets']} 个按键事件包 -> {args.clip_dir}"
        )
        if args.extract_only:
            return 0

    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --extract-only is set")

    result = asyncio.run(
        run_replay_session(
            clip_dir=args.clip_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            sampling=args.sampling,
            pacing_speed=args.pacing_speed,
            dit_chunk_ms=args.mock_dit_chunk_ms,
            num_inference_steps=args.num_inference_steps,
            coordinator_url=args.coordinator_url,
            encoder_url=args.encoder_url,
            dit_url=args.dit_url,
            decoder_url=args.decoder_url,
            profile=args.profile,
            compatibility=args.compatibility,
            network_domain=args.network_domain,
            control_epoch=args.control_epoch,
            resolution=args.resolution,
            server_revision=args.server_revision,
            return_stage_timing=args.stage_timing == "on",
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
