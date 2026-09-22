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
    raw = json.dumps({"h3_media_config": {"fps": 24}}).encode()
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
            vae_config=SimpleNamespace(arch_config=SimpleNamespace(vae_clip_length=17))
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
        {
            "size": "1920x1088",
            "source_stream": {
                "format": "rgb24",
                "width": 1920,
                "height": 1088,
                "fps": 24,
            },
        },
    ],
)
def test_reject_mixed_protocol_or_profile(configured, overrides):
    with pytest.raises(ValueError):
        initialize(**overrides)


def test_default_encoder_dispatches_first_frame(configured):
    adapter, session = initialize()
    assert adapter.source_encoder["mode"] == "progressive"
    assert adapter.source_encoder["factory"].endswith("progressive_encoder:create_session")
    adapter.pending_frames = 1
    asyncio.run(asyncio.wait_for(adapter.wait_for_next_chunk(session), 1))


def test_explicit_clip17_waits_for_full_clip(configured):
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
        assert not waiter.done()
        adapter.pending_frames = 17
        adapter.available.set()
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


def test_explicit_prompt_video_preserves_reference_ids(
    configured, monkeypatch
):
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
        references=refs,
        prompt_video_frame=dict(id="<Video 1>", format="png", payload=frame.getvalue()),
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
    assert not adapter.reference_items and adapter.prompt_video is None


def test_reference_bytes_remain_bounded(configured):
    with pytest.raises(ValueError, match="too large"):
        initialize(
            references=[dict(id="large", format="jpeg", payload=b"x" * ((8 << 20) + 1))]
        )


def test_prompt_video_frame_cannot_duplicate_reference_id(configured):
    import io
    from PIL import Image

    data = io.BytesIO()
    Image.new("RGB", (32, 32)).save(data, format="JPEG")
    with pytest.raises(ValueError, match="uniquely named PNG"):
        initialize(
            references=[dict(id="same", format="jpeg", payload=data.getvalue())],
            prompt_video_frame=dict(id="same", format="png", payload=b""),
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
