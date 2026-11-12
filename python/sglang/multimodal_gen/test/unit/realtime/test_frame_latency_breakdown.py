# SPDX-License-Identifier: Apache-2.0
"""Unit tests for per-frame Encoder/DiT/Decoder latency breakdown & visualization."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import Mock, patch

import msgspec.msgpack
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.queued_dit import (
    QueuedDiTSession,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.remote_decoder import (
    RemoteLynnDecoder,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_bridge import (
    SourceChunkBridge,
)
from sglang.multimodal_gen.runtime.realtime.decoder_output import DecoderOutputBroker
from sglang.multimodal_gen.runtime.realtime.whitebox.telemetry import (
    CHUNK_FIELDS,
    numeric_durations,
    sanitize_frame_stage_breakdown,
)
from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
    PixelRange,
    Provenance,
    SourceSpan,
)

# Allow importing benchmark/whitebox visualizer in unit tests
_REPO_ROOT = Path(__file__).resolve().parents[6]
_WHITEBOX_BENCH = _REPO_ROOT / "benchmark" / "whitebox"
if str(_WHITEBOX_BENCH) not in sys.path:
    sys.path.insert(0, str(_WHITEBOX_BENCH))

from latency_breakdown_visualizer import (  # noqa: E402
    build_key_hold_spans,
    build_timeline_dataset,
    render_html_report,
    render_png_overview,
)


def test_sanitize_frame_stage_breakdown_whitelists_and_bounds():
    raw = [
        {
            "output_frame": 8,
            "encoder": {
                "wait_ms": 41.2,
                "process_ms": 9.8,
                "return_ms": 1.3,
                "extra": 99,
            },
            "dit": {"wait_ms": 82.5, "process_ms": 185.0, "return_ms": 2.4},
            "decoder": {"wait_ms": 0.4, "process_ms": 31.1, "return_ms": 3.7},
            "unexpected": "ignored",
        },
        {
            "output_frame": 9,
            "encoder": {"wait_ms": -5.0, "process_ms": float("nan"), "return_ms": 1.1},
            "dit": {"wait_ms": True, "process_ms": 180.0, "return_ms": float("inf")},
            "decoder": {"wait_ms": 0.0, "process_ms": 30.0, "return_ms": 2.5},
        },
    ]
    cleaned = sanitize_frame_stage_breakdown(raw, start_frame=8, num_frames=2)
    assert len(cleaned) == 2
    assert cleaned[0] == {
        "output_frame": 8,
        "encoder": {"wait_ms": 41.2, "process_ms": 9.8, "return_ms": 1.3},
        "dit": {"wait_ms": 82.5, "process_ms": 185.0, "return_ms": 2.4},
        "decoder": {"wait_ms": 0.4, "process_ms": 31.1, "return_ms": 3.7},
    }
    assert cleaned[1] == {
        "output_frame": 9,
        "encoder": {"return_ms": 1.1},
        "dit": {"process_ms": 180.0},
        "decoder": {"wait_ms": 0.0, "process_ms": 30.0, "return_ms": 2.5},
    }

    # All-or-nothing rejection on length mismatch or out-of-order output_frame
    assert sanitize_frame_stage_breakdown(raw, start_frame=8, num_frames=3) == []
    assert sanitize_frame_stage_breakdown(raw, start_frame=0, num_frames=2) == []

    summary_msg = {
        "encoder_wait_ms": 20.5,
        "encoder_process_ms": 9.8,
        "encoder_return_ms": 1.2,
        "dit_wait_ms": 45.0,
        "dit_process_ms": 182.5,
        "dit_return_ms": 2.4,
        "decoder_wait_ms": 0.4,
        "decoder_process_ms": 30.5,
        "decoder_return_ms": 3.1,
    }
    parsed = numeric_durations(summary_msg, CHUNK_FIELDS)
    assert parsed == summary_msg


def test_source_chunk_bridge_propagates_encoder_and_dit_arrival_timings():
    plan = ChunkPlan((3, 2))

    def _prov(pixels: PixelRange, dep: int) -> Provenance:
        return Provenance(
            pixels,
            (
                SourceSpan(
                    1,
                    pixels.start,
                    pixels.start,
                    pixels.end - pixels.start,
                ),
            ),
            dep,
        )

    bridge = SourceChunkBridge(
        plan,
        clip_frames=17,
        clip_latents=5,
        temporal_ratio=4,
        capacity_latents=15,
        capacity_bytes=64 * 1024 * 1024,
        seed=42,
        provenance_for=_prov,
    )
    enc_timings = [
        {"frame": 0, "wait_ms": 0.2, "process_ms": 7.5, "return_ms": 1.1},
        *[
            {
                "frame": p,
                "wait_ms": float((4 - p) * 33),
                "process_ms": 11.0,
                "return_ms": 1.2,
            }
            for p in range(1, 5)
        ],
        *[
            {
                "frame": p,
                "wait_ms": float((8 - p) * 33),
                "process_ms": 11.0,
                "return_ms": 1.2,
            }
            for p in range(5, 9)
        ],
    ]
    sample_events = (object(), object())
    tensor_3 = torch.zeros((1, 16, 3, 4, 4), dtype=torch.float32)
    bridge.append(
        tensor_3,
        latent_start=0,
        pixel_end=9,
        source_dependency_end=9,
        slice_timing={
            "arrived_ns": 1_000_000_000,
            "sample_ms": 1.5,
            "sample_events": sample_events,
            "encoder_frame_timings": enc_timings,
        },
    )
    chunk0 = bridge.pop()
    assert chunk0 is not None
    assert chunk0.provenance.pixels == PixelRange(0, 9)
    assert chunk0.input_timings is not None
    assert len(chunk0.input_timings) == 3
    assert chunk0.input_timings[0]["pixel_start"] == 0
    assert chunk0.input_timings[0]["pixel_end"] == 1
    assert chunk0.input_timings[0]["arrived_ns"] == 1_000_000_000
    assert chunk0.input_timings[0]["sample_ms"] == 1.5
    assert all(lt["sample_events"] is sample_events for lt in chunk0.input_timings)
    assert chunk0.input_timings[1]["encoder_frames"][1]["wait_ms"] == 99.0


def test_remote_decoder_builds_chunk_frame_timings():
    decoder = RemoteLynnDecoder.__new__(RemoteLynnDecoder)
    timing_ctx = {
        "dit_start_ns": 2_100_000_000,  # 2100 ms
        "ev_dit_start": None,
        "ev_dit_end": None,
        "input_timings": [
            {
                "pixel_start": 1,
                "pixel_end": 5,
                "arrived_ns": 2_000_000_000,  # arrived 100 ms before dit_start
                "sample_ms": 2.0,
                "encoder_frames": {
                    1: {"wait_ms": 95.0, "process_ms": 10.5, "return_ms": 1.4},
                    2: {"wait_ms": 62.0, "process_ms": 10.5, "return_ms": 1.4},
                },
            }
        ],
    }
    rows = decoder._build_chunk_frame_timings(
        timing_ctx,
        start_frame=1,
        end_frame=3,
        send_t0_ns=2_280_000_000,
        send_done_ns=2_283_000_000,
    )
    assert len(rows) == 2
    assert rows[0] == {
        "output_frame": 1,
        "encoder": {"wait_ms": 95.0, "process_ms": 10.5, "return_ms": 1.4},
        "dit": {
            "wait_ms": 98.0,
            "process_ms": 182.0,
            "return_ms": 3.0,
        },
    }
    assert rows[1]["output_frame"] == 2
    assert rows[1]["encoder"]["wait_ms"] == 62.0


@pytest.mark.parametrize("timing_enabled", [False, True])
def test_cuda_sampling_events_cover_sampler_without_ingress_sync(
    monkeypatch, timing_enabled
):
    class CudaTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    calls = []
    events = [Mock(), Mock()]
    for index, event in enumerate(events):
        event.record.side_effect = lambda stream, i=index: calls.append(i)
        event.synchronize.side_effect = AssertionError("Unexpected sampling sync")
    monkeypatch.setattr(torch.cuda, "Event", Mock(side_effect=events))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "stream")
    bridge = Mock(
        accepted_latents=0,
        capacity_latents=5,
        capacity_bytes=4096,
        buffered_latents=0,
        buffered_bytes=0,
    )

    def sample(value, *, latent_start):
        calls.append("sample")
        return value

    session = QueuedDiTSession(
        session_id="s",
        generation_id="g",
        create_session=lambda: None,
        **({"return_stage_timing": True} if timing_enabled else {}),
        bridge=bridge,
        sample_moments=sample,
        on_output=lambda value: None,
    )
    # Mock signatures are generic; opt into the real bridge timing contract.
    session._bridge_accepts_timing = True
    session.initial_received = True
    asyncio.run(
        session.put(
            dict(
                type="source.moments",
                session_id="s",
                generation_id="g",
                tensor=torch.zeros((1, 2, 1, 1, 1)).as_subclass(CudaTensor),
                latent_start=0,
                latent_end=1,
                pixel_end=1,
                source_dependency_end=1,
            )
        )
    )
    assert calls == ([0, "sample", 1] if timing_enabled else ["sample"])
    assert session.return_stage_timing is timing_enabled
    if timing_enabled:
        timing = bridge.append.call_args.kwargs["slice_timing"]
        assert timing["sample_events"] == tuple(events)
        assert timing["sample_ms"] >= 0
    else:
        assert "slice_timing" not in bridge.append.call_args.kwargs
        torch.cuda.Event.assert_not_called()
    for event in events:
        event.synchronize.assert_not_called()
        event.elapsed_time.assert_not_called()


def test_remote_decoder_counts_cuda_sampling_instead_of_host_launch_time():
    # Host sampling launches in 0.1 ms, but its CUDA work takes 40 ms.
    sample_start, sample_end = Mock(), Mock()
    dit_start, dit_end = Mock(), Mock()
    dit_start.elapsed_time.return_value = 180.0
    sample_start.elapsed_time.side_effect = lambda end: (
        dit_end.synchronize.assert_called_once() or 40.0
    )
    rows = RemoteLynnDecoder.__new__(RemoteLynnDecoder)._build_chunk_frame_timings(
        dict(
            dit_start_ns=2_100_000_000,
            ev_dit_start=dit_start,
            ev_dit_end=dit_end,
            input_timings=[
                dict(
                    pixel_start=0,
                    pixel_end=1,
                    arrived_ns=2_000_000_000,
                    sample_ms=0.1,
                    sample_events=(sample_start, sample_end),
                )
            ],
        ),
        start_frame=0,
        end_frame=1,
        send_t0_ns=2_280_000_000,
        send_done_ns=2_283_000_000,
    )
    assert rows[0]["dit"] == dict(wait_ms=99.9, process_ms=220.0, return_ms=3.0)
    sample_start.elapsed_time.assert_called_once_with(sample_end)
    sample_end.synchronize.assert_not_called()


@pytest.mark.parametrize("timing_enabled", [False, True])
def test_decoder_output_broker_emits_chunk_telemetry_only_when_timing_present(
    timing_enabled,
):
    cancelled = []
    broker = DecoderOutputBroker(cancelled.append)
    session = broker.open(
        dict(
            session_id="s1",
            generation_id="g1",
            width=4,
            height=4,
            fps=24,
            **({"return_stage_timing": True} if timing_enabled else {}),
        )
    )
    app = FastAPI()
    broker.install(app)

    async def fake_adapter_send(self, ws, sess, result, batch):
        start = batch.extra["zing_source_output_blocks"][0]["output_frame_start"]
        await ws.send_bytes(
            msgspec.msgpack.encode(
                {
                    "type": "frame_batch",
                    "output_frame_start": start,
                    "num_frames": 2,
                }
            )
        )
        return {
            "raw_payload_build_ms": 0.4,
            "msgpack_encode_ms": 0.2,
            "ws_send_ms": 1.1,
        }

    with patch(
        "sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter.RawRGBRealtimeOutputAdapter.send",
        new=fake_adapter_send,
    ):
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime_video/output") as ws:
                ws.send_bytes(
                    msgspec.msgpack.encode(
                        {
                            "type": "output.attach",
                            "session_id": "s1",
                            "generation_id": "g1",
                            "token": session.token,
                        }
                    )
                )
                assert (
                    msgspec.msgpack.decode(ws.receive_bytes())["type"] == "output.ready"
                )

                # Batch 0: accompanied by decoder.timing -> emits chunk_telemetry
                session.put(
                    {
                        "type": "decoder.frames",
                        "reference": {"request_id": "r1", "chunk_index": 0},
                        "metadata": {"format": "rgb24", "width": 4, "height": 4},
                        "output_metadata": {
                            "output_frame_start": 0,
                            "num_frames": 2,
                            "source_dependency_end": 2,
                            "source_alignment": [
                                {
                                    "source_id": 1,
                                    "source_frame_start": 0,
                                    "output_frame_start": 0,
                                    "num_frames": 2,
                                }
                            ],
                        },
                        "timing_state": {"wait_ms": 0.8, "process_ms": 28.5},
                    },
                    timeout=5,
                )
                session.put(
                    {
                        "type": "decoder.timing",
                        "request_id": "r1",
                        "chunk_index": 0,
                        "output_frame_start": 0,
                        "num_frames": 2,
                        "frame_stage_breakdown": [
                            {
                                "output_frame": 0,
                                "encoder": {
                                    "wait_ms": 30.0,
                                    "process_ms": 10.0,
                                    "return_ms": 1.0,
                                },
                                "dit": {
                                    "wait_ms": 60.0,
                                    "process_ms": 190.0,
                                    "return_ms": 2.0,
                                },
                            },
                            {
                                "output_frame": 1,
                                "encoder": {
                                    "wait_ms": 0.0,
                                    "process_ms": 10.0,
                                    "return_ms": 1.0,
                                },
                                "dit": {
                                    "wait_ms": 27.0,
                                    "process_ms": 190.0,
                                    "return_ms": 2.0,
                                },
                            },
                        ],
                    },
                    timeout=5,
                )

                # Batch 1: without decoder.timing -> NO extra chunk_telemetry emitted
                session.put(
                    {
                        "type": "decoder.frames",
                        "reference": {"request_id": "r1", "chunk_index": 1},
                        "metadata": {"format": "rgb24", "width": 4, "height": 4},
                        "output_metadata": {
                            "output_frame_start": 2,
                            "num_frames": 2,
                            "source_dependency_end": 4,
                            "source_alignment": [
                                {
                                    "source_id": 1,
                                    "source_frame_start": 2,
                                    "output_frame_start": 2,
                                    "num_frames": 2,
                                }
                            ],
                        },
                    },
                    timeout=5,
                )
                session.put(
                    {"type": "session.completed", "output_frames": 4}, timeout=5
                )

                m1 = msgspec.msgpack.decode(ws.receive_bytes())
                m2 = msgspec.msgpack.decode(ws.receive_bytes())
                m3 = msgspec.msgpack.decode(ws.receive_bytes())
                if not timing_enabled:
                    assert m1["type"] == "frame_batch" and m1["output_frame_start"] == 0
                    assert m2["type"] == "frame_batch" and m2["output_frame_start"] == 2
                    assert m3["type"] == "session.completed"
                    return
                m4 = msgspec.msgpack.decode(ws.receive_bytes())
                assert m1["type"] == "frame_batch" and m1["output_frame_start"] == 0
                assert m2["type"] == "chunk_telemetry"
                assert m2["num_frames"] == 2
                assert m2["encoder_wait_ms"] == 15.0
                assert m2["dit_process_ms"] == 190.0
                assert m2["decoder_wait_ms"] == 0.8
                assert m2["decoder_process_ms"] == 28.5
                assert m2["decoder_return_ms"] == pytest.approx(1.7, abs=0.05)
                assert len(m2["frame_stage_breakdown"]) == 2
                assert m2["frame_stage_breakdown"][0]["decoder"]["process_ms"] == 28.5
                assert m3["type"] == "frame_batch" and m3["output_frame_start"] == 2
                assert m4["type"] == "session.completed"


def test_visualizer_key_spans_and_report_generation(tmp_path: Path):
    input_events = [
        {
            "sequence": 0,
            "rel_time_s": 0.5,
            "payload": {
                "held_keys": ["KeyW"],
                "held_buttons": [],
                "events": [{"type": "keydown", "code": "KeyW"}],
            },
        },
        {
            "sequence": 1,
            "rel_time_s": 1.2,
            "payload": {
                "held_keys": ["KeyW", "KeyE"],
                "held_buttons": [],
                "events": [{"type": "keydown", "code": "KeyE"}],
            },
        },
        {
            "sequence": 2,
            "rel_time_s": 1.6,
            "payload": {
                "held_keys": ["KeyW"],
                "held_buttons": [],
                "events": [{"type": "keyup", "code": "KeyE"}],
            },
        },
        {
            "sequence": 3,
            "rel_time_s": 2.5,
            "payload": {
                "held_keys": [],
                "held_buttons": [],
                "events": [{"type": "keyup", "code": "KeyW"}],
            },
        },
    ]
    spans, markers = build_key_hold_spans(input_events, duration_seconds=4.0)
    assert len(spans) == 2
    assert len(markers) == 4
    assert {s["track"] for s in spans} == {0, 1}

    source_frames = [
        {
            "source_frame_index": i,
            "capture_time_s": i * (1.0 / 30.0),
            "pts_s": i * (1.0 / 30.0),
        }
        for i in range(90)
    ]
    selected_frames = [
        {
            "source_frame": 0,
            "model_frame": 0,
            "capture_time_s": 0.0,
            "source_to_selection_ms": 2.0,
            "selection_to_dispatch_ms": 1.0,
            "dispatch_to_raw_output_ms": 135.0,
            "source_to_raw_output_ms": 138.0,
        },
        {
            "source_frame": 4,
            "model_frame": 1,
            "capture_time_s": 4 / 30.0,
            "source_to_selection_ms": 5.0,
            "selection_to_dispatch_ms": 1.2,
            "dispatch_to_raw_output_ms": 280.0,
            "source_to_raw_output_ms": 286.2,
        },
    ]
    breakdown = [
        {
            "output_frame": 0,
            "encoder": {"wait_ms": 0.2, "process_ms": 8.0, "return_ms": 1.2},
            "dit": {"wait_ms": 0.5, "process_ms": 95.0, "return_ms": 2.1},
            "decoder": {"wait_ms": 0.4, "process_ms": 22.0, "return_ms": 3.5},
        },
        {
            "output_frame": 1,
            "encoder": {"wait_ms": 33.0, "process_ms": 11.0, "return_ms": 1.4},
            "dit": {"wait_ms": 65.0, "process_ms": 195.0, "return_ms": 2.4},
            "decoder": {"wait_ms": 0.6, "process_ms": 32.0, "return_ms": 4.0},
        },
    ]
    dataset = build_timeline_dataset(
        case_meta={"case_id": "unit-case", "duration_s": 3.0, "clip_start_s": 0.0},
        source_frames=source_frames,
        selected_frames=selected_frames,
        frame_stage_breakdown=breakdown,
        input_events=input_events,
        mode="mock",
    )
    assert dataset["summary"]["source_frames"] == 90
    assert dataset["summary"]["selected_model_frames"] == 2
    assert len(dataset["skip_intervals"]) == 1
    assert dataset["skip_intervals"][0]["skipped"] == 3

    html_out = render_html_report(dataset, tmp_path / "index.html")
    png_out = render_png_overview(dataset, tmp_path / "overview.png")
    assert html_out.exists() and "等待攒帧" in html_out.read_text(encoding="utf-8")
    assert "模拟模式（mock；非真实 GPU 实测）" in html_out.read_text(encoding="utf-8")
    assert png_out.exists() and png_out.stat().st_size > 20_000
    assert json.loads(json.dumps(dataset))["schema_version"] == 2


@pytest.mark.parametrize("complete", [True, False])
def test_report_keeps_measured_output_and_missing_samples(tmp_path, complete):
    rows = (
        [
            {
                "output_frame": 0,
                **{
                    role: {"wait_ms": 0.0, "process_ms": 50.0, "return_ms": 0.0}
                    for role in ("encoder", "dit", "decoder")
                },
            },
        ]
        if complete
        else []
    )
    dataset = build_timeline_dataset(
        case_meta={"duration_s": 2},
        source_frames=[
            {"source_frame_index": i, "capture_time_s": float(i)} for i in range(2)
        ],
        selected_frames=[
            {"source_frame": i, "model_frame": i, "dispatch_to_raw_output_ms": 100.0}
            for i in range(2)
        ],
        frame_stage_breakdown=rows,
        input_events=[],
        mode="real-sglang",
    )
    assert dataset["selected_frames"][0]["output_s"] == 0.1
    assert dataset["selected_frames"][1]["output_s"] == 1.1
    stats = dataset["summary"]["stages"]["encoder"]["process_ms"]
    assert stats["count"] == int(complete)
    assert stats["missing_count"] == 2 - int(complete)
    assert stats["coverage"] == (0.5 if complete else 0.0)
    assert stats["mean"] == (50.0 if complete else None)
    assert dataset["selected_frames"][1]["stages"]["encoder"]["total_ms"] is None
    assert all(v is None for v in dataset["selected_frames"][1]["projected"].values())
    assert "wire_overhead_ms" not in dataset["selected_frames"][0]
    html = render_html_report(dataset, tmp_path / "index.html").read_text()
    assert "真实服务模式（real-sglang" in html
    assert "本模拟演示" not in html
    assert "不表示真实起止或重叠" in html and "未测" in html
    assert render_png_overview(dataset, tmp_path / "missing.png").exists()


def test_partial_stage_does_not_invent_a_total():
    dataset = build_timeline_dataset(
        case_meta={},
        source_frames=[],
        selected_frames=[{"source_frame": 0, "model_frame": 0, "capture_time_s": 0.0}],
        frame_stage_breakdown=[
            {
                "output_frame": 0,
                "encoder": {
                    "wait_ms": True,
                    "process_ms": 50.0,
                    "return_ms": float("nan"),
                },
            }
        ],
        input_events=[],
        mode="real-sglang",
    )
    stage = dataset["selected_frames"][0]["stages"]["encoder"]
    assert stage == {
        "wait_ms": None,
        "process_ms": 50.0,
        "return_ms": None,
        "total_ms": None,
    }


@pytest.mark.parametrize("height", [480, 736])
@pytest.mark.parametrize("declared_height", [480, 736])
def test_direct_pool_checks_requested_dimensions_before_admission(
    monkeypatch, height, declared_height
):
    import asyncio

    import httpx
    from source_replay_role_pool import SuiteRolePool

    monkeypatch.setenv(
        "ROLE_POOL_SIGNING_SECRET", "test-only-signing-secret-at-least-32-characters"
    )
    width = 832 if height == 480 else 1280
    declared_width = 832 if declared_height == 480 else 1280

    def respond(req):
        if req.url.path == "/managed/discovery":
            return httpx.Response(200, json={"control_epoch": "test-epoch"})
        return httpx.Response(
            200,
            json={
                "control_epoch": "test-epoch",
                "service": {"role": req.url.host},
                "deployment": {
                    "compatibility": "test-model",
                    "network_domain": "test-network",
                },
                "profiles": {str(declared_height): {}},
                "profile_limits": {
                    str(declared_height): {
                        "source_sizes": [[declared_width, declared_height]],
                        "target_sizes": [[declared_width, declared_height]],
                        "max_references": 3,
                        "max_fps": 24,
                    }
                },
            },
        )

    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            pool = SuiteRolePool(
                workers={
                    role: f"http://{role}" for role in ("encoder", "dit", "decoder")
                },
                profile=str(height),
                video_size=(width, height),
                client=client,
            )
            pool.client = client
            if height == declared_height:
                await pool.discover_workers()
                assert pool.config["profile"] == str(height)
            else:
                with pytest.raises(ValueError, match="Expected one common"):
                    await pool.discover_workers()

    asyncio.run(check())


@pytest.mark.parametrize("generation", ["current", "stale"])
def test_decoder_timing_sideband_keeps_generation_fence(generation):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_worker import (
        _queue_chunk_timing,
    )

    items = []
    session = SimpleNamespace(
        config={"generation_id": "current"},
        put=lambda item, timeout, **kwargs: items.append((item, timeout, kwargs)),
    )
    command = dict(
        generation_id=generation,
        request_id="request",
        chunk_index=2,
        output_frame_start=9,
        num_frames=8,
        frame_stage_breakdown=[],
    )
    if generation == "stale":
        with pytest.raises(ValueError, match="identity is invalid"):
            _queue_chunk_timing({"output_session": session}, command, 5)
        assert not items
    else:
        _queue_chunk_timing({"output_session": session}, command, 5)
        item, timeout, flags = items[0]
        assert item["type"] == "decoder.timing"
        assert item["output_frame_start"] == 9 and item["num_frames"] == 8
        assert timeout == 5 and flags == {"discard_if_stopped": True}


def test_grouped_protocol_and_observed_output_packets():
    breakdown = []
    selected = []
    for frame in range(34):
        offset = frame % 17
        batch_start = frame - (0 if offset == 0 else (offset - 1) % 4)
        selected.append(
            {
                "source_frame": frame,
                "model_frame": frame,
                "capture_time_s": frame / 30,
                "chunk_index": (frame // 17) * 2 + int(offset >= 9),
                "output_batch_start": batch_start,
                "output_batch_frames": 1 if offset == 0 else 4,
                "dispatch_to_raw_output_ms": 100,
            }
        )
        breakdown.append(
            {
                "output_frame": frame,
                **{
                    role: {"wait_ms": 1.0, "process_ms": 2.0, "return_ms": 3.0}
                    for role in ("encoder", "dit", "decoder")
                },
            }
        )
    d = build_timeline_dataset(
        case_meta={},
        source_frames=[],
        selected_frames=selected,
        frame_stage_breakdown=breakdown,
        input_events=[],
        mode="mock",
    )
    assert [g["pixel_frames"] for g in d["stage_groups"]["encoder"]] == [1, 8, 8] * 2
    assert [g["latent_frames"] for g in d["stage_groups"]["encoder"]] == [1, 2, 2] * 2
    assert [g["latent_frames"] for g in d["stage_groups"]["dit"]] == [3, 2] * 2
    assert [g["pixel_frames"] for g in d["stage_groups"]["decoder"]] == [
        1,
        4,
        4,
        4,
        4,
    ] * 2
    assert [b["num_frames"] for b in d["output_batches"]] == [1, 4, 4, 4, 4] * 2
    assert sum(b["observed_frames"] for b in d["output_batches"]) == 34
    # Never infer packet counts from the protocol when they were not captured.
    for row in selected:
        row.pop("output_batch_start")
        row.pop("output_batch_frames")
    old = build_timeline_dataset(
        case_meta={},
        source_frames=[],
        selected_frames=selected,
        frame_stage_breakdown=breakdown,
        input_events=[],
        mode="mock",
    )
    assert old["output_batches"] == []


def test_mock_codec_groups_and_progressive_output_match_runtime():
    import asyncio

    from run_latency_breakdown_replay import MockStagePipelineWebSocket

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec

    async def run():
        spec = VideoSpec(width=32, height=32, fps=24)
        ws = MockStagePipelineWebSocket(
            spec, worker_epoch="test", player_expires_at=1e20, dit_chunk_ms=0.01
        )
        assert [ws._target_enc_group_size(i) for i in (0, 1, 9, 17, 18, 26)] == [
            1,
            8,
            8,
        ] * 2
        task = asyncio.create_task(ws._run_dit_decoder_loop("op"))
        for i in range(34):
            await ws._handle_generate_data(
                "op",
                {
                    "kind": "source.append",
                    "payload": {
                        "source_frame_start": i,
                        "data": bytes(spec.frame_bytes),
                    },
                },
            )
        await ws._handle_generate_data("op", {"kind": "source.end"})
        await asyncio.wait_for(task, 3)
        messages = []
        while not ws.outgoing.empty():
            outer = msgspec.msgpack.decode(ws.outgoing.get_nowait())
            if outer.get("type") == "bound.data":
                messages.append(msgspec.msgpack.decode(outer["payload"]))
        output = [m for m in messages if m["type"] == "frame_batch"]
        chunks = [m for m in messages if m["type"] == "chunk_telemetry"]
        assert [m["num_frames"] for m in output] == [1, 4, 4, 4, 4] * 2
        assert [m["num_frames"] for m in chunks] == [9, 8] * 2
        assert [m["output_frame_start"] for m in output] == [
            0,
            1,
            5,
            9,
            13,
            17,
            18,
            22,
            26,
            30,
        ]
        await ws.close()

    asyncio.run(run())


@pytest.mark.parametrize("selected", [True, False])
def test_report_untrusted_data_cannot_create_html(tmp_path, selected):
    payload = '</script><script>globalThis.pwned=1</script><img src=x onerror="globalThis.pwned=2">&\u2028\u2029'
    dataset = build_timeline_dataset(
        case_meta={"case_id": payload, "duration_s": 2},
        source_frames=[
            {
                "source_frame_index": i,
                "capture_time_s": float(i),
                "held_keys": [payload],
            }
            for i in range(2)
        ],
        selected_frames=[{"source_frame": 0, "model_frame": 0}],
        frame_stage_breakdown=[],
        input_events=[],
        mode="mock",
    )
    dataset["skip_intervals"] = [
        dict(start_s=0, end_s=1, gap_seconds=1, skipped=1, reason=payload)
    ]
    dataset["summary"]["source_frames"] = payload
    dataset["summary"]["stages"]["encoder"]["wait_ms"]["count"] = payload
    dataset["selected_frames"][0]["chunk_index"] = payload
    dataset["selected_frames"][0]["output_batch_frames"] = payload
    html = render_html_report(dataset, tmp_path / "index.html").read_text()

    class ReportParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tags = []

        def handle_starttag(self, tag, attrs):
            self.tags.append(tag)

    parser = ReportParser()
    parser.feed(html)
    assert parser.tags.count("script") == 1
    assert "img" not in parser.tags
    script = html.split("<script>\n", 1)[1].split("</script>", 1)[0]
    encoded = script.split("const D = ", 1)[1].split(";\nconst S", 1)[0]
    assert all(char not in encoded for char in ("<", ">", "&", "\u2028", "\u2029"))
    assert json.loads(encoded) == dataset

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the generated report JavaScript")
    # Execute the complete generated report with DOM/canvas stubs, capturing
    # every HTML sink. Exercise both selected and skipped frame inspection.
    harness = r"""
const fs = require('fs'), vm = require('vm');
const assert = require('assert');
const nodes = new Map();
const noop = () => {};
const canvas = new Proxy({}, {get: () => noop});
const document = {
  getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, {
      clientWidth: 1200, getContext: () => canvas, addEventListener: noop,
      classList: {toggle: noop}, style: {}, innerHTML: '', textContent: ''
    });
    return nodes.get(id);
  },
  querySelectorAll: () => [], addEventListener: noop, body: {style: {}}
};
const context = vm.createContext({document, window: {addEventListener: noop}});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
vm.runInContext('updateInspection(' + process.argv[2] + ')', context);
assert.strictEqual(context.pwned, undefined);
for (const node of nodes.values()) {
  assert(!/<(?:script|img)\b/i.test(node.innerHTML), node.innerHTML);
}
for (const id of ['stats', 'stageTable', 'skips', 'inspection']) {
  assert(nodes.get(id).innerHTML.includes('&lt;'), id);
}
assert(nodes.get('subtitle').textContent.includes('</script>'));
"""
    js_path = tmp_path / "report.js"
    js_path.write_text(script)
    subprocess.run(
        [node, "-e", harness, str(js_path), "0" if selected else "1"],
        check=True,
        capture_output=True,
        text=True,
    )
