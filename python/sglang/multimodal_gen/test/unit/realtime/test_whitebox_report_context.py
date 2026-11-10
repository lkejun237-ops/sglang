# SPDX-License-Identifier: Apache-2.0
"""Local HTTP evidence reads: no model, cloud, database, or session mutation."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.realtime.whitebox.reports import Reports

PARENT, CHILD, FOREIGN = "a" * 32, "b" * 32, "c" * 32
TOKEN = "test-recording-capability"
HEADERS = {"Authorization": "Bearer " + TOKEN}
BASE = "/v1/whitebox/recordings/" + PARENT


def fixture(tmp_path, *, diskless=False):
    raw = (
        json.dumps(
            dict(
                frame=7,
                pts_us=12300000,
                presentation_pts_us=12490000,
                archive_pts_us={"whitebox": 12490000},
                source={"source_frame_index": 14},
                states={"secret_opaque_state": "not returned"},
            )
        )
        + "\n"
    ).encode()
    objects = {
        "states.jsonl": dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
        "whitebox.mp4": dict(bytes=100, sha256="f" * 64),
    }
    source = dict(
        recording_id=PARENT,
        recording_kind="source",
        world_id="world",
        trace_id="trace-source",
        source_session_id="session-original",
        source_epoch="epoch-original",
        object_prefix="whitebox/" + PARENT + "/",
        upload_status="complete",
        recording_status="complete",
        objects=objects,
        videos={
            "whitebox": dict(
                path="whitebox.mp4",
                frames=8,
                first_pts_us=12000000,
                playback_timeline="mp4_preserved_pts_v1",
                playback_time_origin_us=0,
            )
        },
        states={"path": "states.jsonl"},
        timeline={"mode": "source_capture_vfr"},
        render_segments=[dict(recording_id=CHILD, segment_epoch="segment-original")]
        + [dict(recording_id=f"{i:032x}") for i in range(65)],
        inference={"token": "must-not-return"},
    )
    child = dict(
        source,
        recording_id=CHILD,
        recording_kind="render_segment",
        segment_id=CHILD,
        segment_epoch="segment-original",
        trace_id="trace-model",
        render_segments=[],
        object_prefix=source["object_prefix"] + "segments/" + CHILD + "/",
    )
    if diskless:
        source["archive_storage"] = child["archive_storage"] = "oss_streaming_v1"
    manifests = {PARENT: source, CHILD: child}
    calls = []

    async def read_index(recording_id):
        if recording_id != PARENT:
            raise FileNotFoundError()
        return dict(token_hash=hashlib.sha256(TOKEN.encode()).hexdigest())

    async def read_manifest(recording_id):
        calls.append(recording_id)
        if recording_id not in manifests:
            raise FileNotFoundError()
        return manifests[recording_id]

    async def read_object(key, limit):
        calls.append(key)
        assert key in {
            source["object_prefix"] + "states.jsonl",
            child["object_prefix"] + "states.jsonl",
        }
        assert limit == 8 << 20
        return raw

    store = SimpleNamespace(
        bucket="fixture-bucket",
        worker_config={"region": "fixture-region"},
        playback_url=lambda key: "https://signed.invalid/" + key + "?signature=test",
    )
    archives = SimpleNamespace(
        store=store,
        diskless=diskless,
        read_index=read_index,
        read_manifest=read_manifest,
        read_object=read_object,
    )
    if not diskless:
        access = tmp_path / ".reports"
        access.mkdir()
        (access / (PARENT + ".json")).write_text(
            json.dumps(dict(token_hash=hashlib.sha256(TOKEN.encode()).hexdigest()))
        )
        for key, manifest in manifests.items():
            directory = tmp_path / key
            directory.mkdir()
            (directory / "manifest.json").write_text(json.dumps(manifest))
            (directory / "states.jsonl").write_bytes(raw)
    reports = Reports(tmp_path, archives)
    app = FastAPI()
    reports.install(app)
    return app, manifests, calls


@pytest.mark.parametrize("diskless", [False, True])
def test_retained_context_exact_segment_outside_last64_and_stable_locations(
    tmp_path, diskless
):
    app, _, calls = fixture(tmp_path, diskless=diskless)
    with TestClient(app) as client:
        result = client.get(
            BASE + "/context", params={"segment_id": CHILD}, headers=HEADERS
        )
        assert result.status_code == 200
        assert result.headers["cache-control"] == "private, no-store"
        value = result.json()
        assert value["identity"]["source_epoch"] == "epoch-original"
        assert value["segments"][0]["identity"]["segment_id"] == CHILD
        assert value["segment_selection"] == "exact"
        artifact = next(
            a for a in value["segments"][0]["artifacts"] if a["artifact_id"] == "states"
        )
        assert artifact["locator"] == dict(
            provider="oss",
            bucket="fixture-bucket",
            region="fixture-region",
            object_key="whitebox/" + PARENT + "/segments/" + CHILD + "/states.jsonl",
        )
        assert (
            "signature" not in result.text
            and TOKEN not in result.text
            and "must-not-return" not in result.text
        )
        assert value["mapping"]["status"] == "unavailable"
        # Without a selector no implicit latest segment is read or returned.
        assert client.get(BASE + "/context", headers=HEADERS).json()["segments"] == []
        access = client.get(
            BASE + "/context/artifact-access",
            params={"segment_id": CHILD, "artifact_id": "states"},
            headers=HEADERS,
        )
        assert (
            access.status_code == 200 and "/segments/" + CHILD in access.json()["url"]
        )


def test_wrong_capability_foreign_segment_and_arbitrary_paths_are_rejected(tmp_path):
    app, _, calls = fixture(tmp_path, diskless=True)
    with TestClient(app) as client:
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            assert client.get(BASE + "/context", headers=headers).status_code == 404
        assert calls == []
        assert (
            client.get(
                BASE + "/context", params={"segment_id": FOREIGN}, headers=HEADERS
            ).status_code
            == 404
        )
        assert FOREIGN not in calls
        for artifact_id in (
            "../../private",
            "https://internal.invalid/metadata",
            "video:foreign",
        ):
            assert (
                client.get(
                    BASE + "/context/artifact-access",
                    params={"artifact_id": artifact_id},
                    headers=HEADERS,
                ).status_code
                == 404
            )


@pytest.mark.parametrize("diskless", [False, True])
def test_vfr_mapping_uses_verified_presentation_pts_and_exact_archive_frame(
    tmp_path, diskless
):
    app, manifests, _ = fixture(tmp_path, diskless=diskless)
    with TestClient(app) as client:
        value = client.get(
            BASE + "/context", params={"frame_index": 7}, headers=HEADERS
        ).json()
        assert value["mapping"]["seek_seconds"] == 12.49
        assert value["mapping"]["pts_us"] == 12490000
        assert value["mapping"]["method"] == "states_archive_pts"
        assert value["mapping"]["playback_time_origin_us"] == 0
        raw = client.get(
            BASE + "/context",
            params={"source_frame_index": 14, "segment_epoch": "segment-original"},
            headers=HEADERS,
        ).json()
        assert raw["mapping"]["frame_index"] == 7
        assert raw["segments"][0]["identity"]["segment_id"] == CHILD
        assert (
            client.get(
                BASE + "/context", params={"segment_epoch": "wrong"}, headers=HEADERS
            ).status_code
            == 404
        )
        if diskless:
            video = manifests[PARENT]["videos"]["whitebox"]
            video.pop("playback_timeline")
            assert (
                client.get(
                    BASE + "/context", params={"frame_index": 7}, headers=HEADERS
                ).json()["mapping"]["reason"]
                == "archive_playback_timeline_unknown"
            )
            manifests[PARENT]["videos"]["whitebox"].pop("first_pts_us")
            assert (
                client.get(
                    BASE + "/context", params={"frame_index": 7}, headers=HEADERS
                ).json()["mapping"]["reason"]
                == "archive_video_pts_origin_missing"
            )


def test_pending_archive_and_missing_hash_remain_explicit(tmp_path):
    app, manifests, _ = fixture(tmp_path, diskless=True)
    manifests[PARENT]["upload_status"] = "pending"
    with TestClient(app) as client:
        value = client.get(
            BASE + "/context", params={"frame_index": 7}, headers=HEADERS
        ).json()
        assert value["mapping"]["status"] == "pending"
        assert all(a["availability"] == "pending" for a in value["source"]["artifacts"])
        assert (
            client.get(
                BASE + "/context/artifact-access",
                params={"artifact_id": "manifest"},
                headers=HEADERS,
            ).status_code
            == 409
        )
        manifests[PARENT]["upload_status"] = "complete"
        manifests[PARENT]["objects"]["states.jsonl"]["sha256"] = "0" * 64
        assert (
            client.get(
                BASE + "/context", params={"frame_index": 7}, headers=HEADERS
            ).json()["mapping"]["reason"]
            == "states_read_failed"
        )


def test_gateway_forwards_exact_segment_and_frame_queries_to_fixed_owner():
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_proxy import RoomProxy

    async def run():
        requests = []

        def upstream(request):
            requests.append(request)
            return httpx.Response(
                200, json={"ok": True}, headers={"Cache-Control": "private, no-store"}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as owner:
            app = FastAPI()
            RoomProxy(
                None, {"one": "http://room-owner"}, "internal", owner, None
            ).install(app)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://gateway"
            ) as client:
                response = await client.get(
                    "/v1/whitebox/owners/one/recordings/" + PARENT + "/context",
                    params={"segment_id": CHILD, "frame_index": "7"},
                    headers=HEADERS,
                )
                assert response.status_code == 200
                assert requests[0].url.host == "room-owner"
                assert dict(requests[0].url.params) == {
                    "segment_id": CHILD,
                    "frame_index": "7",
                }
                assert requests[0].headers["authorization"] == HEADERS["Authorization"]
                assert (
                    await client.get(
                        "/v1/whitebox/owners/unknown/recordings/" + PARENT + "/context",
                        headers=HEADERS,
                    )
                ).status_code == 404

    asyncio.run(run())


def test_actual_source_mux_records_capture_pts_even_when_presentation_differs(tmp_path):
    import av

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import (
        EngineFrame,
        VideoSpec,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.playback import (
        H264PlaybackEncoder,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.recording import Recording

    async def run():
        spec = VideoSpec(32, 32)
        recording = Recording(
            directory=tmp_path / PARENT,
            spec=spec,
            world_id="world",
            trace_id="trace",
            inference={},
            ui={"mode": "browser"},
            references=[],
            object_prefix="fixture/",
            synchronous=True,
            source_only=True,
        )
        encoder = H264PlaybackEncoder(spec)
        try:
            for index, pts in enumerate((12000000, 12490000)):
                encoded = encoder.encode(index, bytes(spec.frame_bytes))
                frame = EngineFrame(
                    index,
                    spec.pts(index),
                    b"",
                    {},
                    effective_input={},
                    provenance={
                        "capture_pts_us": pts,
                        "presentation_pts_us": 99000000 + index,
                        "source_frame_index": index * 7,
                    },
                )
                recording.encoders["whitebox"].append(index, encoded, pts)
                recording.aligned_metadata(frame, "", archive_pts_us={"whitebox": pts})
            await recording.finish("completed")
        finally:
            encoder.close()
        manifest = recording.manifest
        assert manifest["videos"]["whitebox"]["first_pts_us"] == 12000000
        assert manifest["videos"]["whitebox"]["playback_time_origin_us"] == 0
        rows = [
            json.loads(line)
            for line in (tmp_path / PARENT / "states.jsonl").read_text().splitlines()
        ]
        assert rows[1]["archive_pts_us"]["whitebox"] == 12490000
        with av.open(str(tmp_path / PARENT / "whitebox.mp4")) as movie:
            decoded = list(movie.decode(video=0))
            assert [
                int(frame.pts * frame.time_base * 1000000) for frame in decoded
            ] == [12000000, 12490000]
        return manifest

    manifest = asyncio.run(run())
    manifest["upload_status"] = "complete"
    (tmp_path / PARENT / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / ".reports").mkdir()
    (tmp_path / ".reports" / (PARENT + ".json")).write_text(
        json.dumps(dict(token_hash=hashlib.sha256(TOKEN.encode()).hexdigest()))
    )
    app = FastAPI()
    Reports(tmp_path, SimpleNamespace(store=SimpleNamespace())).install(app)
    with TestClient(app) as client:
        mapping = client.get(
            BASE + "/context", params={"frame_index": 1}, headers=HEADERS
        ).json()["mapping"]
        # MP4's preserved edit list has initialTime=12s in HTMLVideo. Subtracting
        # first_pts (0.49s) clamps to its first frame; 12.49s selects frame 1.
        assert mapping["seek_seconds"] == 12.49
        assert mapping["playback_timeline"] == "mp4_preserved_pts_v1"


def test_expected_client_ui_receipt_has_explicit_pending_artifact(tmp_path):
    app, manifests, _ = fixture(tmp_path, diskless=True)
    manifests[PARENT]["client_ui"] = {"status": "awaiting_upload"}
    with TestClient(app) as client:
        value = client.get(BASE + "/context", headers=HEADERS).json()
        artifact = next(
            a for a in value["source"]["artifacts"] if a["artifact_id"] == "client_ui"
        )
        assert artifact["availability"] == "pending" and artifact["locator"] is None
        assert (
            client.get(
                BASE + "/context/artifact-access",
                params={"artifact_id": "client_ui"},
                headers=HEADERS,
            ).status_code
            == 409
        )


def test_small_mapping_sidecar_resolves_frame_without_reading_large_opaque_states(
    tmp_path,
):
    app, manifests, _ = fixture(tmp_path)
    source = manifests[PARENT]
    states = tmp_path / PARENT / "states.jsonl"
    states.write_bytes(b"x" * (9 << 20))
    source["objects"]["states.jsonl"] = dict(
        bytes=states.stat().st_size, sha256="0" * 64
    )
    sidecar = (
        json.dumps(
            dict(
                schema_version=1,
                frame=7,
                source_frame_index=14,
                archive_pts_us={"whitebox": 12490000},
            )
        )
        + "\n"
    ).encode()
    (tmp_path / PARENT / "report-frames.jsonl").write_bytes(sidecar)
    source["report_frames"] = dict(
        path="report-frames.jsonl", format="report_frame_index_v1"
    )
    source["objects"]["report-frames.jsonl"] = dict(
        bytes=len(sidecar), sha256=hashlib.sha256(sidecar).hexdigest()
    )
    manifest_path = tmp_path / PARENT / "manifest.json"
    manifest_path.write_text(json.dumps(source))
    with TestClient(app) as client:
        value = client.get(
            BASE + "/context", params={"source_frame_index": 14}, headers=HEADERS
        ).json()
        assert value["mapping"]["seek_seconds"] == 12.49
        assert value["mapping"]["method"] == "report_frame_index_v1"
        assert value["mapping"]["index_path"] == "report-frames.jsonl"
        source.pop("report_frames")
        manifest_path.write_text(json.dumps(source))
        unavailable = client.get(
            BASE + "/context", params={"source_frame_index": 14}, headers=HEADERS
        ).json()
        assert (
            unavailable["mapping"]["reason"] == "states_read_budget_or_digest_missing"
        )


def test_model_wire_parent_source_identity_matches_real_archive_sidecar(tmp_path):
    import time
    from types import MethodType

    import msgspec
    from test_whitebox_stream_archive import MemoryS3, prototype

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import EngineFrame
    from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import (
        InteractiveSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.playback import (
        H264PlaybackEncoder,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.stream_recording import (
        StreamingData,
    )

    async def run():
        kwargs, manifest = await prototype(tmp_path)
        client = MemoryS3()
        data = StreamingData(
            client,
            {"bucket": "b"},
            kwargs,
            manifest,
            dict(
                upload_bytes=32 << 20,
                archive_bytes=2 << 30,
                part_bytes=65536,
                finish_seconds=10,
                index_key="fixture-index",
            ),
        )
        spec = kwargs["spec"]
        archive = SimpleNamespace(
            accepting=True,
            pair=lambda frame, source, encoded, prompt: data.operation(
                "pair", (frame, source, encoded, prompt)
            ),
        )
        session = SimpleNamespace(
            received=0,
            model_preview_count=0,
            recording=archive,
            config=SimpleNamespace(room_owned=True, spec=spec),
            model_geometry=SimpleNamespace(output=spec),
            pairs={},
            pair_condition=asyncio.Condition(),
            pairing_age_budget=lambda: 3,
            video_transport=None,
            outgoing=asyncio.Queue(),
            trace_id="fixture",
        )
        session._media = MethodType(InteractiveSession._media, session)
        encoder = H264PlaybackEncoder(spec)
        wires = []
        try:
            for local, parent_index in enumerate((100, 107)):
                encoded = encoder.encode(local, bytes(spec.frame_bytes))
                frame = EngineFrame(
                    parent_index,
                    spec.pts(parent_index),
                    b"",
                    {},
                    effective_input={},
                    provenance={
                        "source_frame_index": parent_index + 1000,
                        "capture_pts_us": 12000000 + local * 490000,
                    },
                )
                session.pairs[local] = (frame, time.perf_counter(), encoded)
                await InteractiveSession._deliver_model_frame(
                    session, local, frame, encoded, time.perf_counter(), "prompt"
                )
                wires.append(msgspec.msgpack.decode(await session.outgoing.get()))
            result = data.finish("completed", manifest)
            rows = [
                json.loads(line)
                for line in client.objects[
                    ("b", manifest["object_prefix"] + "report-frames.jsonl")
                ][0].splitlines()
            ]
            assert [wire["recording_frame_index"] for wire in wires] == [0, 1]
            assert [wire["source_frame_index"] for wire in wires] == [100, 107]
            assert [row["frame"] for row in rows] == [
                wire["recording_frame_index"] for wire in wires
            ]
            assert [row["source_frame_index"] for row in rows] == [
                wire["source_frame_index"] for wire in wires
            ]
            assert [row["archive_pts_us"]["model"] for row in rows] == [
                wire["pts_us"] for wire in wires
            ]
            assert result["videos"]["model"]["first_pts_us"] == wires[0]["pts_us"]
        finally:
            encoder.close()
            data.close()

    asyncio.run(run())
