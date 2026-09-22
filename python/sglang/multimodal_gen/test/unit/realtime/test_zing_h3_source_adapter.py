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


def test_ready_preserves_unknown_total_and_accepted_precedes_dispatch(configured):
    adapter, session = initialize()
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
            num_frames=17,
            pts_us=0,
            data=bytes(17 * 32 * 32 * 3),
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


def test_five_references_are_preserved_and_six_rejected(configured):
    import io
    from pathlib import Path
    from PIL import Image

    data = io.BytesIO()
    Image.new("RGB", (32, 32)).save(data, format="JPEG")
    refs = [
        dict(id=f"picture-{i}", format="jpeg", payload=data.getvalue())
        for i in range(6)
    ]
    adapter, session = initialize(references=refs[:5])
    directory = Path(session.input_temp_dir)
    assert adapter.session_ready_message(session)["max_references"] == 5
    assert len(adapter.reference_paths) == 5
    assert adapter.reference_ids == [ref["id"] for ref in refs[:5]]
    assert all(Path(p).read_bytes() == data.getvalue() for p in adapter.reference_paths)
    adapter.dispose(session)
    # Temp files are owned by the shared session cleanup.
    import shutil

    shutil.rmtree(directory)
    with pytest.raises(ValueError, match="at most five"):
        initialize(references=refs)
