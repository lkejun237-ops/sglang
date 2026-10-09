# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from types import SimpleNamespace

import msgspec.msgpack
import pytest

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters import (
    zing_h3_source_adapter as source,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RawRGBRealtimeOutputAdapter,
)
from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
    PixelRange,
    Provenance,
    SourceSpan,
)


@pytest.fixture
def configured(tmp_path, monkeypatch):
    raw = json.dumps({"causal": True, "h3_media_config": {"fps": 24}}).encode()
    (tmp_path / "zing_config.json").write_bytes(raw)
    (tmp_path / "zing_realtime_profile.json").write_text(
        json.dumps(
            dict(
                model_config_sha256=hashlib.sha256(raw).hexdigest(),
                solver="few_step",
                denoising_step_list=[1000, 750, 500, 250],
                guidance_scale=1,
                noise_version="indexed-cpu-v1",
            )
        )
    )
    args = SimpleNamespace(
        model_path=str(tmp_path),
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(
                    vae_clip_length=17, spatial_compression_ratio=16
                )
            ),
            dit_config=SimpleNamespace(
                arch_config=SimpleNamespace(patch_size=(1, 2, 2))
            ),
        ),
    )
    monkeypatch.setattr(source, "get_global_server_args", lambda: args)
    return args


def initialize(**overrides):
    request = RealtimeVideoGenerationsRequest(
        **{
            "type": "init",
            "prompt": "Render snow",
            "zing_realtime_api_version": 1,
            "source_stream": {"format": "rgb24", "width": 32, "height": 32, "fps": 24},
            "size": "32x32",
            **overrides,
        }
    )
    adapter, session = source.ZingH3SourceStreamAdapter(), GenerateSession()
    session.set_adapter(adapter)
    asyncio.run(adapter.on_init(session, request))
    session.set_request(request)
    return adapter, session


@pytest.mark.parametrize(
    "overrides",
    [
        {"zing_realtime_api_version": 2},
        {"zing_request": {}},
        {"num_inference_steps": 50},
        {"zing_solver": "euler"},
        {"max_chunks": 3},
        {"zing_window_chunks": 100},
        {"zing_rope_position_mode": "absolute"},
        {"guidance_scale": 2},
        {"realtime_preview_max_width": 16},
        {"condition_inputs": {}},
    ],
)
def test_reject_mixed_protocol_or_profile(configured, overrides):
    with pytest.raises(ValueError):
        initialize(**overrides)


def test_source_over_capacity_rejects_with_transport_error(configured):
    from sglang.multimodal_gen.runtime.realtime.source_transport import (
        SourceTransportError,
    )

    with pytest.raises(SourceTransportError) as error:
        initialize(
            size="4096x4096",
            source_stream={"format": "rgb24", "width": 4096, "height": 4096, "fps": 24},
        )
    assert error.value.code == "source_message_too_large"


def test_default_encoder_dispatches_first_frame(configured):
    adapter, session = initialize()
    assert adapter.source_encoder["mode"] == "progressive"
    assert adapter.source_encoder["factory"].endswith(
        "progressive_encoder:create_session"
    )
    adapter.pending_frames = 1
    asyncio.run(asyncio.wait_for(adapter.wait_for_next_chunk(session), 1))


def test_source_presentation_dispatches_one_frame_even_with_clip17_encoder(configured):
    from pathlib import Path

    path = Path(configured.model_path) / "zing_realtime_profile.json"
    profile = json.loads(path.read_text())
    profile["source_encoder"] = {"mode": "clip17"}
    path.write_text(json.dumps(profile))
    adapter, session = initialize()
    assert adapter.source_encoder == {"mode": "clip17"}

    async def run():
        adapter.pending_frames = 1
        waiter = asyncio.create_task(adapter.wait_for_next_chunk(session))
        await asyncio.sleep(0)
        assert waiter.done()
        await asyncio.wait_for(waiter, 1)

    asyncio.run(run())


@pytest.mark.parametrize(
    "mode,frames", [("clip17", 17)] + [("progressive", n) for n in [1, 2, 8, 17]]
)
def test_ready_preserves_unknown_total_and_accepted_precedes_dispatch(
    configured, mode, frames
):
    adapter, session = initialize()
    adapter.source_encoder = {"mode": mode}
    assert session.request.num_frames is None and session.request.max_chunks is None
    assert adapter.session_ready_message(session)["realtime_output_format"] == "raw"

    async def run():
        sent = []

        class Socket:
            async def send_bytes(self, payload):
                assert not adapter.pending
                sent.append(msgspec.msgpack.decode(payload))
                await asyncio.sleep(0)

        waiter = asyncio.create_task(adapter.wait_for_next_chunk(session))
        await asyncio.sleep(0)
        assert not waiter.done()
        payload = dict(
            client_seq=0,
            source_frame_start=0,
            num_frames=frames,
            pts_us=0,
            data=bytes(frames * 32 * 32 * 3),
        )
        await adapter.ingest_wire_event(
            Socket(),
            session,
            dict(type="event", kind="source.append", event_id=1, payload=payload),
            len(payload["data"]) + 200,
        )
        await asyncio.wait_for(waiter, 1)
        assert len(adapter.pending) == 1 and sent[0]["type"] == "source.accepted"
        assert adapter.ledger.consumed_frames == 0

    asyncio.run(run())
    adapter.dispose(session)
    assert not adapter.pending and adapter.ledger.closed


@pytest.mark.parametrize("output_format", ["raw", "jpeg", "webp"])
def test_source_provenance_survives_transport_splitting(output_format):
    p = Provenance(
        PixelRange(0, 17), (SourceSpan(1, 0, 0, 10), SourceSpan(2, 10, 10, 7)), 34
    )
    sent = []

    class Socket:
        async def send_bytes(self, payload):
            sent.append(payload)

    async def run():
        return await RawRGBRealtimeOutputAdapter()._send_frame_batches(
            Socket(),
            [[bytes(32 * 32 * 3)] * 17],
            content_type="application/x-raw-rgb",
            chunk_index_start=0,
            request_id="dispatch",
            output_format=output_format,
            preview_max_width=32,
            frame_metadata={
                "format": "rgb24",
                "width": 32,
                "height": 32,
                "channels": 3,
                "bytes_per_frame": 32 * 32 * 3,
            },
            source_blocks=[
                dict(block_idx=7, request_id="origin", event_id=91, provenance=p)
            ],
            source_fps=24,
        )

    stats = asyncio.run(run())
    cursor, wires = 0, iter(sent)
    for wire in wires:
        header = msgspec.msgpack.decode(wire)
        payload = (
            next(wires) if header["type"] == "frame_batch_header" else header["payload"]
        )
        assert len(payload) == header["total_size"]
        assert header["output_frame_start"] == cursor
        assert header["pts_us"] == cursor * 1_000_000 // 24
        assert header["event_id"] == 91 and header["chunk_index"] == 7
        assert (
            header["request_id"] == "origin" and header["source_dependency_end"] == 34
        )
        end = cursor + header["num_frames"]
        assert (
            header["source_alignment"]
            == p.slice(PixelRange(cursor, end)).metadata()["source_alignment"]
        )
        cursor = end
    assert cursor == stats["num_frames"] == 17


def test_explicit_prompt_video_preserves_reference_ids(configured, monkeypatch):
    import io

    from PIL import Image

    data = io.BytesIO()
    Image.new("RGB", (32, 32)).save(data, format="JPEG")
    refs = [
        dict(id=f"<Picture {i + 1}>", format="jpeg", payload=data.getvalue())
        for i in range(9)
    ]
    frame = io.BytesIO()
    Image.new("RGB", (32, 32)).save(frame, format="PNG")
    adapter, session = initialize(
        user_media=[
            dict(
                type="video",
                id="<Video 1>",
                format="png",
                payload=frame.getvalue(),
                fps=24,
                clip=dict(start=0, end=0),
            ),
            *[dict(ref, type="image", reference=True) for ref in refs],
        ],
    )
    assert adapter.ready["max_references"] == 9
    assert adapter.ready["max_reference_bytes"] == 8 << 20
    monkeypatch.setattr(
        source, "build_realtime_sampling_params", lambda *a, **kw: SimpleNamespace()
    )
    params = adapter.build_sampling_params(
        session,
        None,
        SimpleNamespace(request_id="test"),
        SimpleNamespace(prompt="original prompt"),
        17,
    )
    user, target = params.zing_request["messages"]
    assert user["content"][0]["type"] == "video"
    assert user["content"][0]["clip"] == dict(start=0, end=0)
    assert [c["id"] for c in user["content"][1:-1]] == [r["id"] for r in refs]
    assert user["content"][-1]["text"] == "original prompt"
    assert [c["content_id"] for c in target["conditions"]] == [r["id"] for r in refs]
    assert "source" not in target
    # PyAV must decode the PNG as exactly one video frame for the prompt loader.
    import av

    with av.open(user["content"][0]["uri"]) as container:
        assert len(list(container.decode(video=0))) == 1
    adapter.clear_state(session)
    assert not adapter.reference_items and adapter.source_preview_sha256 is None
    assert not adapter.auto_source_preview


def test_reference_bytes_remain_bounded(configured):
    with pytest.raises(ValueError, match="too large"):
        initialize(
            references=[dict(id="large", format="jpeg", payload=b"x" * ((8 << 20) + 1))]
        )


def test_typed_media_cannot_duplicate_reference_id(configured):
    import io

    from PIL import Image

    data = io.BytesIO()
    Image.new("RGB", (32, 32)).save(data, format="JPEG")
    with pytest.raises(ValueError, match="uniquely named"):
        initialize(
            user_media=[
                dict(type="image", id="same", format="jpeg", payload=data.getvalue()),
                dict(
                    type="video",
                    id="same",
                    format="png",
                    payload=b"",
                    fps=24,
                    clip=dict(start=0, end=0),
                ),
            ],
        )


def test_nine_references_are_preserved_and_ten_rejected(configured):
    import io
    from pathlib import Path

    from PIL import Image

    refs = []
    for i in range(10):
        data = io.BytesIO()
        fmt = "PNG" if i == 0 else "JPEG"
        Image.new("RGB", (32, 32), color=(i * 20, 0, 0)).save(data, format=fmt)
        refs.append(
            dict(id=f"picture-{9 - i}", format=fmt.lower(), payload=data.getvalue())
        )
    adapter, session = initialize(references=refs[:9])
    directory = Path(session.input_temp_dir)
    assert adapter.session_ready_message(session)["max_references"] == 9
    assert len(adapter.reference_paths) == 9
    assert adapter.reference_ids == [ref["id"] for ref in refs[:9]]
    assert [Path(p).read_bytes() for p in adapter.reference_paths] == [
        ref["payload"] for ref in refs[:9]
    ]
    adapter.dispose(session)
    # Temp files are owned by the shared session cleanup.
    import shutil

    shutil.rmtree(directory)
    with pytest.raises(ValueError, match="at most nine"):
        initialize(references=refs)


@pytest.mark.parametrize("parts", [(1, 3), (3, 1), (2, 2)])
def test_source_preview_uses_s0_once_without_consuming_credit(configured, parts):
    adapter, session = initialize()
    assert "source_preview" not in adapter.ready
    assert adapter.auto_source_preview
    assert adapter.source_preview_sha256 is None
    assert "prompt_video_frame_format" not in adapter.ready

    class Socket:
        async def send_bytes(self, payload):
            pass

    async def run():
        start = 0
        preview = None
        for seq, count in enumerate(parts):
            payload = dict(
                client_seq=seq,
                source_frame_start=start,
                num_frames=count,
                pts_us=start * 1_000_000 // 24,
                data=b"".join(
                    bytes([i + 10]) * (32 * 32 * 3) for i in range(start, start + count)
                ),
            )
            await adapter.ingest_wire_event(
                Socket(),
                session,
                dict(
                    type="event",
                    kind="source.append",
                    event_id=seq + 1,
                    payload=payload,
                ),
                len(payload["data"]) + 200,
            )
            await asyncio.wait_for(adapter.wait_for_next_chunk(session), 1)
            if preview is None:
                preview = adapter.source_preview_sha256
            assert adapter.source_preview_sha256 == preview
            assert adapter.pending[-1]["payload"] is payload
            start += count
            await asyncio.sleep(0)  # paused producer resumes with the same S0
        assert adapter.ledger.accepted_frames == sum(parts)
        assert adapter.ledger.consumed_frames == 0
        assert (
            adapter.source_preview_sha256
            == hashlib.sha256(bytes([10]) * 3072).hexdigest()
        )
        assert session.input_temp_dir is None  # S0 stays in the original append.
        assert adapter.reference_items == []

    asyncio.run(run())
    adapter.dispose(session)
    assert adapter.source_preview_sha256 is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_preview": "ar-first-frame"},
        {"source_preview": "guess"},
        {"prompt_video_frame": {}},
        {"user_media": [{"type": "video", "id": "v", "uri": "/etc/passwd"}]},
    ],
)
def test_removed_preview_options_and_unsafe_media_are_rejected(configured, kwargs):
    with pytest.raises(ValueError):
        initialize(**kwargs)


def test_source_preview_reserves_video_id_for_automatic_media(configured):
    with pytest.raises(ValueError, match="different name"):
        initialize(references=[dict(id="<Video 1>", format="png", payload=b"")])


def test_invalid_first_packet_and_empty_eos_do_not_create_preview(configured):
    adapter, session = initialize()

    class Socket:
        async def send_bytes(self, payload):
            raise AssertionError("Invalid input must not be acknowledged")

    async def run():
        for kind, payload in [
            ("source.end", {"source_frame_end": 0}),
            (
                "source.append",
                dict(
                    client_seq=1,
                    source_frame_start=0,
                    num_frames=1,
                    pts_us=0,
                    data=bytes(3072),
                ),
            ),
        ]:
            with pytest.raises(ValueError):
                await adapter.ingest_wire_event(
                    Socket(),
                    session,
                    dict(type="event", kind=kind, event_id=1, payload=payload),
                    3300,
                )
            assert adapter.source_preview_sha256 is None
            assert adapter.ledger.accepted_frames == 0

    asyncio.run(run())
    adapter.dispose(session)


def test_dispose_before_s0_wakes_waiter(configured):
    adapter, session = initialize()

    async def run():
        waiter = asyncio.create_task(adapter.wait_for_next_chunk(session))
        await asyncio.sleep(0)
        adapter.dispose(session)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiter, 1)

    asyncio.run(run())
    replacement, new_session = initialize()
    assert replacement.source_preview_sha256 is None
    replacement.dispose(new_session)


@pytest.mark.parametrize("width,height", [(832, 480), (896, 512), (1280, 736)])
def test_high_resolution_ready(configured, width, height):
    adapter, session = initialize(
        size=f"{width}x{height}",
        source_stream=dict(format="rgb24", width=width, height=height, fps=24),
        source_message_budget_bytes=64 << 20,
    )
    ready = adapter.session_ready_message(session)
    assert adapter.clip == 17
    assert ready["max_source_batch_frames"] == 17
    assert adapter.ledger.limits.pending_frames == 4 * 17
    assert ready["max_message_bytes"] == 64 << 20
    assert ready["max_source_message_bytes"] == 17 * width * height * 3 + 4096
    assert ready["max_pending_source_bytes"] == 4 * 17 * width * height * 3


@pytest.mark.parametrize(
    "width,height,budget,code",
    [
        (1280, 720, 64 << 20, "source_dimensions_unaligned"),
        (1920, 1088, 64 << 20, "source_message_too_large"),
        (1280, 736, 32 << 20, "source_transport_mismatch"),
    ],
)
def test_transport_rejected_before_media_allocation(
    configured, monkeypatch, width, height, budget, code
):
    from sglang.multimodal_gen.runtime.realtime.source_transport import (
        SourceTransportError,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Media allocation happened before transport validation")

    monkeypatch.setattr(source.tempfile, "mkdtemp", forbidden)
    monkeypatch.setattr(source.Image, "open", forbidden)
    with pytest.raises(SourceTransportError) as error:
        initialize(
            size=f"{width}x{height}",
            source_stream=dict(format="rgb24", width=width, height=height, fps=24),
            source_message_budget_bytes=budget,
            references=[dict(id="image", format="png", payload=b"image")],
        )
    assert error.value.code == code


def test_dimension_error_has_stable_wire_code(configured, monkeypatch):
    from fastapi import WebSocketDisconnect

    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
        realtime_video_api as api,
    )

    adapter = source.ZingH3SourceStreamAdapter()
    monkeypatch.setattr(api, "get_global_server_args", lambda: configured)
    monkeypatch.setattr(
        api,
        "get_realtime_model_adapter",
        lambda args: SimpleNamespace(for_request=lambda req: adapter),
    )

    class Socket:
        def __init__(self):
            self.received = False
            self.sent = []

        async def receive_bytes(self):
            if self.received:
                raise WebSocketDisconnect()
            self.received = True
            return msgspec.msgpack.encode(
                dict(
                    type="init",
                    prompt="Render snow",
                    size="1280x720",
                    zing_realtime_api_version=1,
                    source_stream=dict(format="rgb24", width=1280, height=720, fps=24),
                )
            )

        async def send_bytes(self, data):
            self.sent.append(msgspec.msgpack.decode(data))

    ws = Socket()
    session = GenerateSession()
    with pytest.raises(WebSocketDisconnect):
        asyncio.run(api._listen_generate_request(ws, session))
    assert len(ws.sent) == 1
    assert ws.sent[0]["code"] == "source_dimensions_unaligned"
    assert "divisible by 32" in ws.sent[0]["content"]
    assert session.input_temp_dir is None
    assert session.request is None


def test_startup_rejects_insufficient_concurrent_transport_memory():
    from sglang.multimodal_gen.runtime.realtime.source_transport import (
        SourceTransportBudget,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    args = SimpleNamespace(
        batching_mode="dynamic",
        batching_max_size=1,
        batching_delay_ms=0,
        realtime_max_sessions=8,
        realtime_max_sessions_per_worker=8,
        realtime_max_message_bytes=64 << 20,
        realtime_transport_host_bytes=8 * SourceTransportBudget().session_bytes - 1,
    )
    with pytest.raises(ValueError, match="host budget"):
        ServerArgs._validate_batching(args)
