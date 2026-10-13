# SPDX-License-Identifier: Apache-2.0
"""Small persistent contracts; protocol fault injection stays out of the suite."""

import copy
import json

import pytest
from bench_client import Delivery, calculate_metrics, prepare_case, summarize


def header(block=0, part=0, parts=1, frames=2):
    return {
        "format": "rgb24",
        "channels": 3,
        "width": 2,
        "height": 1,
        "num_frames": frames,
        "total_size": frames * 6,
        "chunk_index": block,
        "frame_batch_index": part,
        "num_frame_batches": parts,
        "is_final_frame_batch": part == parts - 1,
        "request_id": f"block-{block}",
    }


def test_known_unequal_chunks_use_frames_and_exclude_completion():
    delivery = Delivery(10, 2, 1)
    for block, (frames, at) in enumerate([(2, 2), (3, 5), (5, 9)]):
        delivery.accept(header(block=block, frames=frames), bytes(frames * 6), at)
    delivery.complete()
    metrics = calculate_metrics(delivery.chunks)
    assert metrics["ttff_seconds"] == 2
    assert metrics["e2e_seconds"] == 9
    assert metrics["fps"] == pytest.approx(10 / 9)
    assert metrics["tpoc_seconds"] == 3.5
    assert [row["icl_seconds"] for row in delivery.chunks] == [None, 3, 4]
    assert metrics["icl_seconds"]["median"] == 3.5
    assert metrics["icl_seconds"]["p95"] == pytest.approx(3.95)


def test_empty_and_single_chunk_do_not_invent_intervals():
    assert calculate_metrics([])["ttff_seconds"] is None
    assert calculate_metrics([])["icl_seconds"]["count"] == 0
    metrics = calculate_metrics([{"arrival_seconds": 2, "num_frames": 10}])
    assert metrics["fps"] == 5
    assert metrics["tpoc_seconds"] is None
    assert metrics["icl_seconds"]["median"] is None


def test_split_batches_are_distinct_deliveries_of_one_server_chunk():
    delivery = Delivery(4, 2, 1)
    delivery.accept(header(part=0, parts=2), bytes(12), 2)
    delivery.accept(header(part=1, parts=2), bytes(12), 3)
    delivery.complete()
    assert delivery.frames == 4
    assert len(delivery.chunks) == 2
    assert delivery.chunks[1]["output_frame_start"] == 2
    assert calculate_metrics(delivery.chunks)["tpoc_seconds"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"chunk_index": 1},
        {"frame_batch_index": 1},
        {"total_size": 1},
        {"num_frames": 0},
        {"format": "jpeg"},
        {"output_frame_start": 3},
    ],
)
def test_invalid_delivery_is_not_counted(change):
    delivery = Delivery(4, 2, 1)
    bad = {**header(), **change}
    with pytest.raises(ValueError):
        delivery.accept(bad, bytes(12), 2)
    assert not delivery.chunks
    assert delivery.frames == 0


def test_duplicate_or_incomplete_delivery_cannot_pass():
    delivery = Delivery(4, 2, 1)
    delivery.accept(header(), bytes(12), 2)
    with pytest.raises(ValueError):
        delivery.accept(header(), bytes(12), 3)
    with pytest.raises(ValueError):
        delivery.complete()


def test_summary_excludes_warmup_and_failed_partial_metrics():
    rows = []
    for phase, success, e2e in [
        ("warmup", True, 100),
        ("measured", True, 9),
        ("measured", False, 1),
    ]:
        rows.append(
            {
                "label": "5s",
                "phase": phase,
                "success": success,
                "metrics": {
                    "ttff_seconds": 2,
                    "e2e_seconds": e2e,
                    "fps": 10 / e2e,
                    "tpoc_seconds": 3.5,
                },
                "icl_values_seconds": [3, 4],
            }
        )
    summary = summarize(rows, [{"label": "5s", "case_id": "fixed"}], 3)["5s"]
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert summary["not_run_measured"] == 1
    assert summary["metrics"]["e2e_seconds"]["median"] == 9
    assert summary["icl_seconds"]["count"] == 2


def test_missing_30s_is_not_replaced_with_another_case(tmp_path):
    with pytest.raises(ValueError, match="unavailable"):
        prepare_case("30s", tmp_path / "absent.jsonl", tmp_path, {})


def test_old_five_reference_variant_cannot_be_relabelled_as_fixed_case(tmp_path):
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "whitebox_video_smoke_cases.jsonl"
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    variant = copy.deepcopy(cases[1])
    variant["sample_id"] = cases[0]["sample_id"]
    manifest = tmp_path / "wrong.jsonl"
    manifest.write_text(json.dumps(variant))
    with pytest.raises(ValueError, match="six-reference"):
        prepare_case("5s", manifest, tmp_path, {})


@pytest.mark.parametrize(
    "label,references", [("30s-wasteland", 7), ("30s-town", 7), ("30s-indoor", 3)]
)
def test_registered_long_cases_preserve_conditions_and_full_source(
    monkeypatch, tmp_path, label, references
):
    import bench_client as client

    monkeypatch.setattr(client, "expected_assets", lambda case: {})
    manifest = client.default_manifest(label)
    prepared = client.prepare_case(label, manifest, tmp_path, {"seed": 123})
    original = client.load_cases(manifest)[client.CASES[label][0]]
    assert prepared["init"]["zing_request"] == original
    assert prepared["frames"] == 720
    assert prepared["reference_count"] == references
    assert (
        len(prepared["init"]["zing_request"]["messages"][1]["conditions"]) == references
    )
    source = prepared["init"]["zing_request"]["messages"][1]["source"]
    assert source.get("clip") is None


def test_default_matrix_is_four_distinct_business_cases():
    import bench_client as client

    assert client.DEFAULT_CASES == ("5s", "30s-wasteland", "30s-town", "30s-indoor")
    assert len({client.CASES[label][0] for label in client.DEFAULT_CASES}) == 4
    assert "30s" not in client.DEFAULT_CASES


def test_named_collection_preserves_order_requests_and_provenance():
    import bench_client as client
    from case_sets import load_case_set

    collection = load_case_set()
    assert collection["definition"]["name"] == "whitebox-basic"
    assert [m["case_id"] for m in collection["members"]] == [
        client.CASES[label][0] for label in client.DEFAULT_CASES
    ]
    assert [
        m["request"]["messages"][1]["content"][0]["output"]["frames"]
        for m in collection["members"]
    ] == [120, 720, 720, 720]
    assert [
        len(m["request"]["messages"][1]["conditions"]) for m in collection["members"]
    ] == [6, 7, 7, 3]
    for member in collection["members"]:
        assert (
            member["request"]
            == client.load_cases(member["manifest"])[member["case_id"]]
        )


@pytest.mark.parametrize("problem", ["missing-case", "duplicate", "missing-manifest"])
def test_collection_errors_are_not_silently_skipped(tmp_path, problem):
    import bench_client as client
    from case_sets import load_case_set

    member = {
        "manifest": str(client.default_manifest("5s")),
        "case_id": client.CASES["5s"][0],
    }
    entries = [member]
    if problem == "missing-case":
        member["case_id"] = "absent"
    elif problem == "duplicate":
        entries.append(member)
    else:
        member["manifest"] = str(tmp_path / "absent.jsonl")
    path = tmp_path / "collection.json"
    path.write_text(json.dumps({"name": "custom", "cases": entries}))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_case_set(path)


def test_custom_collection_preflight_uses_existing_nondefault_case(
    tmp_path, monkeypatch
):
    import bench_client as client

    manifest = client.default_manifest("5s")
    originals = client.load_cases(manifest)
    case_id = next(key for key in originals if key != client.CASES["5s"][0])
    path = tmp_path / "custom.json"
    definition = {
        "name": "custom",
        "cases": [{"manifest": str(manifest), "case_id": case_id}],
    }
    path.write_text(json.dumps(definition))
    sampling, deployment = tmp_path / "sampling.json", tmp_path / "deployment.json"
    sampling.write_text('{"seed":123,"num_inference_steps":4}')
    deployment.write_text('{"resident":true}')
    monkeypatch.setattr(client, "expected_assets", lambda case: {})
    out = tmp_path / "client"
    client.main(
        [
            "--case-set",
            str(path),
            "--media-root",
            str(tmp_path),
            "--sampling-params",
            str(sampling),
            "--deployment",
            str(deployment),
            "--output-dir",
            str(out),
            "--preflight-only",
        ]
    )
    config = json.loads((out / "config.json").read_text())
    assert config["case_set"]["definition"] == definition
    assert len(config["cases"]) == 1
    assert config["cases"][0]["case_id"] == case_id
    assert config["cases"][0]["init"]["zing_request"] == originals[case_id]


@pytest.mark.parametrize("returned_session", ["session", "other-session"])
def test_profile_delivery_does_not_change_media_metrics(returned_session):
    import asyncio

    import bench_client as client
    import msgpack
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    async def run():
        async def handler(ws):
            await ws.send(
                msgpack.packb(
                    {
                        "type": "frame_batch",
                        **header(),
                        "request_id": "session_dispatch",
                        "payload": bytes(12),
                    }
                )
            )
            await ws.send(
                msgpack.packb(
                    {
                        "type": "profile_result",
                        "session_id": returned_session,
                        "status": "complete",
                        "stages": [
                            {
                                "request_id": "session",
                                "dispatch_id": "session_dispatch",
                                "rank": rank,
                            }
                            for rank in (0, 1)
                        ],
                        "health": [],
                    }
                )
            )
            await ws.close(code=1000, reason="generation complete")

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as ws:
                delivery = client.Delivery(2, 2, 1)
                times = iter([2, 10, 12])
                if returned_session != "session":
                    with pytest.raises(ValueError, match="completed session"):
                        await client.receive(ws, delivery, 0, clock=lambda: next(times))
                    return
                ended = await client.receive(ws, delivery, 0, clock=lambda: next(times))
                assert ended == 12
                assert delivery.profile_wait_seconds == 8
                assert {r["rank"] for r in delivery.profile["stages"]} == {0, 1}
                metrics = client.calculate_metrics(delivery.chunks)
                assert metrics["ttff_seconds"] == metrics["e2e_seconds"] == 2
                assert metrics["fps"] == 1

    asyncio.run(run())


def test_request_saves_received_profile_beside_client_results(tmp_path):
    import asyncio
    from types import SimpleNamespace

    import bench_client as client
    import msgpack
    from websockets.asyncio.server import serve

    async def run():
        async def handler(ws):
            assert (
                msgpack.unpackb(await ws.recv(), raw=False)["return_stage_timing"]
                is True
            )
            await ws.send(
                msgpack.packb(
                    {
                        "type": "frame_batch",
                        **header(),
                        "request_id": "s_dispatch",
                        "payload": bytes(12),
                    }
                )
            )
            await ws.send(
                msgpack.packb(
                    {
                        "type": "profile_result",
                        "session_id": "s",
                        "status": "complete",
                        "stages": [
                            {
                                "request_id": "s",
                                "dispatch_id": "s_dispatch",
                                "rank": rank,
                            }
                            for rank in (0, 1)
                        ],
                        "health": [],
                    }
                )
            )
            await ws.close(code=1000, reason="generation complete")

        async with serve(handler, "127.0.0.1", 0) as server:
            args = SimpleNamespace(
                url=f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                output_dir=tmp_path,
                save_media=False,
                receive_stage_timing=True,
                timeout=5,
            )
            result = await client.run_request(
                {
                    "label": "case",
                    "case_id": "fixed",
                    "init": {"type": "init"},
                    "frames": 2,
                    "width": 2,
                    "height": 1,
                },
                args,
                "measured",
                0,
            )
            assert result["success"] and result["profile_status"] == "complete"
            assert result["profile_wait_seconds"] >= 0
            profile = json.loads(
                (tmp_path / "profiles/case-measured-000.json").read_text()
            )
            assert {row["rank"] for row in profile["stages"]} == {0, 1}
            assert (
                json.loads((tmp_path / "requests.jsonl").read_text())["profile_status"]
                == "complete"
            )

    asyncio.run(run())
