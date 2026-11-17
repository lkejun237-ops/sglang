# SPDX-License-Identifier: Apache-2.0
"""Exact paired archives and applied prompt context; no GPU or external storage."""

import asyncio
import hashlib
import json
import time
from types import MethodType, SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_whitebox_report_context import BASE, CHILD, HEADERS, PARENT, TOKEN
from test_whitebox_stream_archive import MemoryS3, prototype

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import EngineFrame
from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import (
    InteractiveSession,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.playback import H264PlaybackEncoder
from sglang.multimodal_gen.runtime.realtime.whitebox.render_segment import RenderSegment
from sglang.multimodal_gen.runtime.realtime.whitebox.reports import Reports
from sglang.multimodal_gen.runtime.realtime.whitebox.stream_recording import (
    StreamingData,
)


def recorded(tmp_path):
    async def run():
        kwargs, manifest = await prototype(tmp_path)
        prefix = "whitebox/" + PARENT + "/"
        manifest.update(
            recording_id=CHILD,
            recording_kind="render_segment",
            source_session_id="source-original",
            source_epoch="source-epoch",
            segment_id=CHILD,
            segment_epoch="segment-epoch",
            object_prefix=prefix + "segments/" + CHILD + "/",
        )
        kwargs.update(
            directory=tmp_path / CHILD, object_prefix=manifest["object_prefix"]
        )
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

        async def nothing(*args):
            return None

        old, new = "original city", "city after rain"
        old_reference, new_reference = b"original reference", b"rain reference"
        old_sha, new_sha = (
            hashlib.sha256(value).hexdigest()
            for value in (old_reference, new_reference)
        )
        data.blob("references/original.bin", old_reference)
        data.blob("references/rain.bin", new_reference)
        conditions = SimpleNamespace(
            room_timings={},
            first_output_at=None,
            sink=SimpleNamespace(output=nothing),
            prompt_by_event={
                None: dict(
                    version=1,
                    text=old,
                    sha256=hashlib.sha256(old.encode()).hexdigest(),
                    reference_descriptors=[
                        dict(id="scene", sha256=old_sha, path="references/original.bin")
                    ],
                ),
                7: dict(
                    version=2,
                    text=new,
                    sha256=hashlib.sha256(new.encode()).hexdigest(),
                    reference_descriptors=[
                        dict(id="scene", sha256=new_sha, path="references/rain.bin")
                    ],
                ),
            },
            active_reference_descriptors=[],
            owner=SimpleNamespace(segment_prompt_output=nothing),
            periodic_refresh=None,
            segment_id=CHILD,
            segment_epoch="segment-epoch",
        )
        encoder = H264PlaybackEncoder(spec)
        try:
            # Event 7 is already accepted before frame0. Actual outputs still
            # identify the old event for two frames, then apply 7 at frame2.
            for local, (parent_index, pts, event) in enumerate(
                ((100, 12000000, None), (107, 12490000, None), (114, 15000000, 7))
            ):
                frame = EngineFrame(
                    parent_index,
                    spec.pts(parent_index),
                    b"",
                    {},
                    effective_input={"held_keys": [str(local)]},
                    provenance={"capture_pts_us": pts},
                )
                text, rendering = await RenderSegment.model_output_condition(
                    conditions,
                    dict(event_id=event, output_frame_start=local),
                    {local: (frame, 0, None)},
                    None,
                    {},
                )
                from dataclasses import replace

                frame = replace(
                    frame,
                    rendering=rendering,
                    references=conditions.active_reference_descriptors,
                )
                encoded = encoder.encode(local, bytes(spec.frame_bytes))
                session.pairs[local] = (frame, time.perf_counter(), encoded)
                await InteractiveSession._deliver_model_frame(
                    session, local, frame, encoded, time.perf_counter(), text
                )
            result = data.finish("completed", manifest)
            parent = dict(
                recording_id=PARENT,
                recording_kind="source",
                source_session_id="source-original",
                source_epoch="source-epoch",
                object_prefix=prefix,
                world_id="world",
                upload_status="complete",
                render_segments=[
                    dict(recording_id=CHILD, segment_epoch="segment-epoch")
                ],
            )
            return client, {PARENT: parent, CHILD: result}
        finally:
            encoder.close()
            data.close()

    return asyncio.run(run())


def application(tmp_path):
    client, manifests = recorded(tmp_path)

    async def read_index(recording_id):
        if recording_id != PARENT:
            raise FileNotFoundError()
        return dict(token_hash=hashlib.sha256(TOKEN.encode()).hexdigest())

    async def read_manifest(recording_id):
        if recording_id not in manifests:
            raise FileNotFoundError()
        return manifests[recording_id]

    async def read_object(key, max_bytes):
        raw = client.objects[("b", key)][0]
        if len(raw) > max_bytes:
            raise ValueError("budget")
        return raw

    app = FastAPI()
    Reports(
        tmp_path,
        SimpleNamespace(
            diskless=True,
            store=SimpleNamespace(bucket="b"),
            read_index=read_index,
            read_manifest=read_manifest,
            read_object=read_object,
        ),
    ).install(app)
    return app, client, manifests


def change_object(client, manifest, name, values):
    raw = b"".join((json.dumps(value) + "\n").encode() for value in values)
    client.objects[("b", manifest["object_prefix"] + name)] = (raw, {})
    manifest["objects"][name] = dict(
        bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
    )


def test_pair_comparison_has_shared_actual_pts_with_nonzero_origin_and_gaps(tmp_path):
    app, _, _ = application(tmp_path)
    with TestClient(app) as client:
        result = client.get(
            BASE + "/context",
            params={"segment_epoch": "segment-epoch"},
            headers=HEADERS,
        )
        pair = result.json()["segments"][0]["comparison"]
        assert pair["status"] == "aligned"
        assert pair["recording_id"] == CHILD and pair["source_epoch"] == "source-epoch"
        assert (
            pair["frames"] == 3
            and pair["first_playback_seconds"] == 12
            and pair["last_playback_seconds"] == 15
        )
        assert (
            pair["source_artifact_id"] == "video:whitebox"
            and pair["model_artifact_id"] == "video:model"
        )


@pytest.mark.parametrize(
    "problem,reason",
    [
        ("pts", "paired_frame_pts_mismatch"),
        ("missing", "paired_video_missing"),
        ("partial", "paired_recording_incomplete"),
        ("origin", "archive_playback_timeline_unknown"),
    ],
)
def test_pair_alignment_is_not_inferred_from_counts_or_browser_time(
    tmp_path, problem, reason
):
    app, storage, manifests = application(tmp_path)
    manifest = manifests[CHILD]
    if problem == "pts":
        name = manifest["report_frames"]["path"]
        index = [
            json.loads(line)
            for line in storage.objects[("b", manifest["object_prefix"] + name)][
                0
            ].splitlines()
        ]
        index[1]["archive_pts_us"]["model"] += 100000
        change_object(storage, manifest, name, index)
    elif problem == "missing":
        manifest["videos"].pop("whitebox")
    elif problem == "partial":
        manifest["recording_status"] = "partial"
    else:
        manifest["videos"]["model"].pop("playback_timeline")
    with TestClient(app) as client:
        pair = client.get(
            BASE + "/context", params={"segment_id": CHILD}, headers=HEADERS
        ).json()["segments"][0]["comparison"]
        assert pair["status"] == "unavailable" and pair["reason"] == reason


def test_timeline_records_output_application_instead_of_accepted_prompt_and_retains_prior(
    tmp_path,
):
    app, _, _ = application(tmp_path)
    with TestClient(app) as client:
        result = client.get(
            BASE + "/context/timeline",
            params={"segment_id": CHILD, "from_seconds": 13, "to_seconds": 17},
            headers=HEADERS,
        )
        value = result.json()
        assert value["recording_id"] == CHILD and value["source_recording_id"] == PARENT
        assert (
            value["semantic"] == "model_output_applied"
            and value["includes_prior_context"]
        )
        assert value["coverage"] == {"from_seconds": 12, "to_seconds": 15}
        assert [
            (
                event["playback_seconds"],
                event["prompt"]["text"],
                event["prompt"]["version"],
            )
            for event in value["events"]
        ] == [(12, "original city", 1), (15, "city after rain", 2)]
        assert [event["source_frame_index"] for event in value["events"]] == [100, 114]
        final = client.get(
            BASE + "/context/timeline",
            params={"segment_id": CHILD, "from_seconds": 12, "to_seconds": 15},
            headers=HEADERS,
        ).json()
        assert final["events"][-1]["playback_seconds"] == 15
        assert final["events"][-1]["prompt"]["text"] == "city after rain"
        assert [event["references"][0]["sha256"] for event in value["events"]] == [
            hashlib.sha256(b"original reference").hexdigest(),
            hashlib.sha256(b"rain reference").hexdigest(),
        ]
        assert all(
            event["references"][0]["artifact_id"].startswith("reference:")
            for event in value["events"]
        )
        assert result.headers["cache-control"] == "private, no-store"
        early = client.get(
            BASE + "/context/timeline",
            params={"segment_id": CHILD, "from_seconds": 0, "to_seconds": 11},
            headers=HEADERS,
        ).json()
        assert early["events"] == [] and not early["includes_prior_context"]
        late = client.get(
            BASE + "/context/timeline",
            params={"segment_id": CHILD, "from_seconds": 16, "to_seconds": 20},
            headers=HEADERS,
        ).json()
        assert late["events"] == [] and not late["includes_prior_context"]
        limited = client.get(
            BASE + "/context/timeline",
            params={
                "segment_id": CHILD,
                "from_seconds": 13,
                "to_seconds": 17,
                "limit": 1,
            },
            headers=HEADERS,
        ).json()
        assert limited["truncated"] and limited["next_from_seconds"] == 15
        assert (
            client.get(
                BASE + "/context/timeline",
                params={"segment_id": "c" * 32},
                headers=HEADERS,
            ).status_code
            == 404
        )
        assert (
            client.get(
                BASE + "/context/timeline",
                params={"segment_id": CHILD},
                headers={"Authorization": "Bearer wrong"},
            ).status_code
            == 404
        )


def test_legacy_bounded_inputs_join_actual_archive_pts_but_unknown_index_is_not_fabricated(
    tmp_path,
):
    app, _, manifests = application(tmp_path)
    manifests[CHILD].pop("output_contexts")
    with TestClient(app) as client:
        value = client.get(
            BASE + "/context/timeline",
            params={"segment_id": CHILD, "from_seconds": 13, "to_seconds": 17},
            headers=HEADERS,
        ).json()
        assert (
            value["status"] == "available"
            and value["provenance"] == "model_output_inputs_v1"
        )
        assert [event["playback_seconds"] for event in value["events"]] == [12, 15]
        manifests[CHILD].pop("report_frames")
        unknown = client.get(
            BASE + "/context/timeline", params={"segment_id": CHILD}, headers=HEADERS
        ).json()
        assert unknown["status"] == "unavailable" and unknown["events"] == []


def test_timeline_bounds_and_pending_are_explicit(tmp_path):
    app, _, manifests = application(tmp_path)
    with TestClient(app) as client:
        assert (
            client.get(
                BASE + "/context/timeline",
                params={"segment_id": CHILD, "to_seconds": 301},
                headers=HEADERS,
            ).status_code
            == 400
        )
        assert (
            client.get(
                BASE + "/context/timeline",
                params={"segment_id": CHILD, "limit": 129},
                headers=HEADERS,
            ).status_code
            == 422
        )
        manifests[CHILD]["upload_status"] = "pending"
        pending = client.get(
            BASE + "/context/timeline", params={"segment_id": CHILD}, headers=HEADERS
        ).json()
        assert pending["status"] == "pending" and pending["events"] == []


def test_reference_preview_mime_is_restricted_to_owned_image_receipts(tmp_path):
    app, _, manifests = application(tmp_path)
    manifest = manifests[CHILD]
    manifest["objects"]["references/preview.png"] = dict(bytes=10, sha256="a" * 64)
    manifest["objects"]["references/unsafe.svg"] = dict(bytes=10, sha256="b" * 64)
    with TestClient(app) as client:
        artifacts = client.get(
            BASE + "/context", params={"segment_id": CHILD}, headers=HEADERS
        ).json()["segments"][0]["artifacts"]
        png = next(
            item
            for item in artifacts
            if (item.get("locator") or {})
            .get("object_key", "")
            .endswith("/preview.png")
        )
        svg = next(
            item
            for item in artifacts
            if (item.get("locator") or {}).get("object_key", "").endswith("/unsafe.svg")
        )
        assert (
            png["content_type"] == "image/png"
            and png["content_type_provenance"] == "archive_reference_extension"
        )
        assert "content_type" not in svg


@pytest.mark.parametrize(
    "problem", ["incomplete", "out_of_range", "pts", "source_frame", "legacy_gap"]
)
def test_unknown_or_invalid_intermediate_output_context_never_carries_forward(
    tmp_path, problem
):
    app, storage, manifests = application(tmp_path)
    manifest = manifests[CHILD]
    if problem == "incomplete":
        manifest["output_contexts"]["frames"] -= 1
    elif problem == "legacy_gap":
        manifest.pop("output_contexts")
        name = manifest["inputs"]["path"]
        values = [
            json.loads(line)
            for line in storage.objects[("b", manifest["object_prefix"] + name)][
                0
            ].splitlines()
        ]
        values.pop(1)
        change_object(storage, manifest, name, values)
    else:
        name = manifest["output_contexts"]["path"]
        values = [
            json.loads(line)
            for line in storage.objects[("b", manifest["object_prefix"] + name)][
                0
            ].splitlines()
        ]
        if problem == "out_of_range":
            values[-1]["frame"] = 99
        elif problem == "pts":
            values[-1]["archive_pts_us"]["model"] = 99000000
        else:
            values[-1]["source_frame_index"] = -1
        change_object(storage, manifest, name, values)
    with TestClient(app) as client:
        value = client.get(
            BASE + "/context/timeline",
            params={"segment_id": CHILD, "from_seconds": 13, "to_seconds": 17},
            headers=HEADERS,
        ).json()
        assert value["status"] == "unavailable" and value["events"] == []
