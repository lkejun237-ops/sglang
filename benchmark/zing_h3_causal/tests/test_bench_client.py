# SPDX-License-Identifier: Apache-2.0
"""Metric contracts and incremental source client lifecycle checks."""

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
            if returned_session == "session":
                assert msgpack.unpackb(await ws.recv(), raw=False) == {
                    "type": "profile_ack",
                    "session_id": "session",
                }
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
            assert msgpack.unpackb(await ws.recv(), raw=False) == {
                "type": "profile_ack",
                "session_id": "s",
            }
            assert (tmp_path / "profiles/case-measured-000.json").exists()
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


async def streaming_peer(ws, observations, *, disconnect=False):
    """Protocol peer: echo bytes, with bounded credit; it is not a model."""
    import asyncio

    import msgpack

    async def send(message):
        await ws.send(msgpack.packb(message, use_bin_type=True))

    init = msgpack.unpackb(await ws.recv(), raw=False)
    observations.append(init)
    width, height = (init["source_stream"][key] for key in ("width", "height"))
    stride = width * height * 3
    await send(
        dict(
            type="session.ready",
            session_id="test-session",
            zing_realtime_api_version=1,
            num_inference_steps=4,
            realtime_output_format="raw",
            source_stream=init["source_stream"],
            max_message_bytes=init["source_message_budget_bytes"],
            max_source_message_bytes=8192,
            max_source_batch_frames=2,
            max_pending_source_frames=2,
            max_pending_source_bytes=2 * stride,
        )
    )
    frames = 0
    async for wire in ws:
        message = msgpack.unpackb(wire, raw=False)
        observations.append(message)
        if message["kind"] == "source.end":
            assert message["payload"]["source_frame_end"] == frames
            await send(dict(type="session.completed", output_frames=frames))
            if init.get("return_stage_timing"):
                await asyncio.sleep(0.03)
                await send(
                    dict(
                        type="profile_result",
                        session_id="test-session",
                        status="complete",
                        stages=[],
                        health=[],
                    )
                )
                async for ack_wire in ws:
                    ack = msgpack.unpackb(ack_wire, raw=False)
                    if ack.get("type") == "profile_ack":
                        assert ack["session_id"] == "test-session"
                        observations.append(ack)
                        break
            return
        if message["kind"] != "source.append":
            continue
        packet = message["payload"]
        assert packet["source_frame_start"] == frames
        count = packet["num_frames"]
        assert count <= 2
        start, frames = frames, frames + count
        source_id = packet["client_seq"] + 1
        await send(
            dict(
                type="source.accepted",
                source_id=source_id,
                client_seq=packet["client_seq"],
                event_id=message["event_id"],
                source_frame_start=start,
                source_frame_end=frames,
            )
        )
        if disconnect:
            await ws.close()
            return
        await asyncio.sleep(0.015)  # Force the producer to wait for credit.
        await send(
            dict(
                type="source.consumed",
                source_frame_end=frames,
                source_bytes_end=frames * stride,
            )
        )
        await send(
            dict(
                type="frame_batch_header",
                output_frame_start=start,
                num_frames=count,
                total_size=count * stride,
                format="rgb24",
                encoding="raw",
                channels=3,
                width=width,
                height=height,
                pts_us=start * 1_000_000 // 24,
                source_dependency_end=frames,
                source_alignment=[
                    dict(
                        source_id=source_id,
                        source_frame_start=start,
                        output_frame_start=start,
                        num_frames=count,
                    )
                ],
            )
        )
        await ws.send(packet["data"])


@pytest.mark.parametrize("paced", [False, True])
@pytest.mark.parametrize("source_format", ["rgb", "video"])
def test_streaming_client_delivers_before_eos_and_obeys_credit(
    tmp_path, paced, source_format
):
    import asyncio

    import websockets
    from continuous_source_client import parser, run

    raw = bytes(range(72))  # Six 2x2 RGB frames.
    source = tmp_path / "source.rgb"
    source.write_bytes(raw)
    observations = []
    if source_format == "video":
        import shutil
        import subprocess

        import imageio_ffmpeg

        ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
        video = tmp_path / "source.mkv"
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "2x2",
                "-r",
                "24",
                "-i",
                str(source),
                "-c:v",
                "ffv1",
                str(video),
            ],
            check=True,
        )
        source_args = ["--source-video", str(video), "--ffmpeg", ffmpeg]
    else:
        source_args = ["--source-rgb", str(source)]

    async def exercise():
        async with websockets.serve(
            lambda ws: streaming_peer(ws, observations), "127.0.0.1", 0
        ) as server:
            port = server.sockets[0].getsockname()[1]
            args = parser().parse_args(
                [
                    "--url",
                    f"ws://127.0.0.1:{port}",
                    *source_args,
                    "--width",
                    "2",
                    "--height",
                    "2",
                    "--prompt",
                    "test",
                    "--output",
                    str(tmp_path / "result"),
                    "--batch-pattern",
                    "17",
                    "--expected-frames",
                    "6",
                    *(["--paced"] if paced else []),
                ]
            )
            return await run(args)

    result = asyncio.run(exercise())
    assert result["received"] == result["submitted"] == result["consumed"] == 6
    assert result["max_pending"] <= 2
    assert result["first_output_before_source_end"]
    assert result["credit_wait_seconds"] > 0
    assert result["tail_drain_seconds"] >= 0
    assert result["source_to_first_output_seconds"] < result["ttff_seconds"]
    assert result["input_sha256"] == result["output_sha256"]
    assert result["ifl_seconds"] is None
    assert result["icl_seconds"]["count"] == 2
    assert result["input_mode"] == ("paced" if paced else "unpaced")
    assert (tmp_path / "result/output.rgb").read_bytes() == raw
    assert "num_frames" not in observations[0]["source_stream"]
    assert observations[-1]["kind"] == "source.end"


def test_streaming_pipe_reassembles_short_reads_and_preserves_eof():
    import asyncio

    from continuous_source_input import read_packet

    async def exercise():
        reader = asyncio.StreamReader()
        task = asyncio.create_task(read_packet(reader, 6))
        reader.feed_data(b"ab")
        await asyncio.sleep(0)
        assert not task.done()
        reader.feed_data(b"cdefg")
        assert await task == b"abcdef"
        reader.feed_eof()
        assert await read_packet(reader, 6) == b"g"
        assert await read_packet(reader, 6) == b""

    asyncio.run(exercise())


def test_live_pipe_partial_reads_and_cleanup(tmp_path, monkeypatch):
    import asyncio
    import os
    import sys
    from types import SimpleNamespace

    from continuous_source_input import open_source

    read_fd, write_fd = os.pipe()
    with (
        os.fdopen(read_fd, "rb") as source,
        os.fdopen(write_fd, "wb", buffering=0) as sink,
    ):
        monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=source))

        async def exercise():
            async with open_source(
                SimpleNamespace(source_rgb="-"), 12, tmp_path / "unused.log"
            ) as read:
                pending = asyncio.create_task(read(12))
                sink.write(b"abcde")
                await asyncio.sleep(0)
                assert not pending.done()
                sink.write(b"fghijkl")
                assert await asyncio.wait_for(pending, 1) == b"abcdefghijkl"
            # Cleanup must not need EOF from a still-running live producer.
            await asyncio.sleep(0)
            assert source.closed

        asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["disconnect", "partial_frame", "wrong_count"])
def test_streaming_failure_never_writes_success_summary(tmp_path, failure):
    import asyncio
    from builtins import ExceptionGroup

    import websockets
    from continuous_source_client import parser, run

    source = tmp_path / "source.rgb"
    source.write_bytes(bytes(13 if failure == "partial_frame" else 72))

    async def exercise():
        async with websockets.serve(
            lambda ws: streaming_peer(ws, [], disconnect=failure == "disconnect"),
            "127.0.0.1",
            0,
        ) as server:
            args = parser().parse_args(
                [
                    "--url",
                    f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                    "--source-rgb",
                    str(source),
                    "--width",
                    "2",
                    "--height",
                    "2",
                    "--prompt",
                    "test",
                    "--output",
                    str(tmp_path / "result"),
                    "--expected-frames",
                    "7" if failure == "wrong_count" else "6",
                    "--timeout",
                    "2",
                ]
            )
            with pytest.raises(ExceptionGroup):
                await run(args)

    asyncio.run(exercise())
    assert not (tmp_path / "result/summary.json").exists()


def test_video_decode_is_incremental_and_cancel_closes_process(tmp_path):
    import asyncio
    import shutil
    import subprocess
    from types import SimpleNamespace

    import imageio_ffmpeg
    from continuous_source_input import open_source

    ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    video = tmp_path / "input.mkv"
    # Enough frames to fill the decoder pipe; the client deliberately stops early.
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=32x32:rate=24",
            "-frames:v",
            "120",
            "-c:v",
            "ffv1",
            str(video),
        ],
        check=True,
    )
    args = SimpleNamespace(
        source_video=str(video), source_rgb=None, width=32, height=32, ffmpeg=ffmpeg
    )

    async def exercise():
        async with open_source(args, 3072, tmp_path / "decode.log") as read:
            assert len(await read(3072)) == 3072
        # Exiting without draining 120 frames must reap FFmpeg rather than hang.

    async def bounded():
        await asyncio.wait_for(exercise(), 5)

    asyncio.run(bounded())


def test_streaming_benchmark_keeps_warmup_out_of_summary_and_exports(
    tmp_path, monkeypatch
):
    import asyncio
    from pathlib import Path
    from types import SimpleNamespace

    import bench_streaming_source as streaming

    seen = []

    async def observe(args):
        output = Path(args.output)
        seen.append(output.name)
        output.mkdir(parents=True)
        events = [
            dict(
                kind="frames",
                received_seconds=at,
                header=dict(
                    num_frames=2,
                    output_frame_start=i * 2,
                    source_dependency_end=(i + 1) * 2,
                ),
            )
            for i, at in enumerate((2, 3, 4))
        ]
        (output / "events.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in events)
        )
        return dict(
            session_id=f"session-{len(seen)}",
            input_sha256="input-hash",
            output_sha256="output-hash",
            received=6,
            first_source_send=0.5,
            last_source_send=3.5,
            ttff_seconds=2,
            e2e_seconds=4,
            fps=1.5,
            tpoc_seconds=1,
            icl_seconds={},
            stall_seconds=0.5,
            ifl_seconds=None,
            source_to_first_output_seconds=1,
            tail_drain_seconds=1,
            credit_wait_seconds=0,
            first_output_before_source_end=True,
            dependency_to_output_seconds={},
            source_to_output_seconds={},
        )

    monkeypatch.setattr(streaming, "run", observe)
    case = dict(
        label="5s",
        case_id="fixture",
        frames=6,
        width=2,
        height=2,
        source_video="video.mkv",
        prompt="p",
        references=[],
        reference_ids=[],
    )
    args = SimpleNamespace(
        output_dir=tmp_path,
        paced=False,
        warmup_runs=1,
        measured_runs=3,
        seed=123,
        url="ws://unused",
        timeout=10,
        ffmpeg="ffmpeg",
        batch_pattern=[1],
    )
    summary = asyncio.run(streaming.benchmark([case], args))
    assert seen == ["warmup-0", "measured-0", "measured-1", "measured-2"]
    assert summary["5s"]["succeeded"] == 3
    assert summary["5s"]["metrics"]["ttff_seconds"]["count"] == 3
    exported = json.loads((tmp_path / "results.json").read_text())
    assert len(exported["requests"]) == 4
    assert exported["config"]["transport"] == "continuous_source"
    assert exported["requests"][1]["server_session_id"] == "session-2"
    assert exported["requests"][1]["metrics"]["stall_seconds"] == 0.5
    assert exported["requests"][1]["metrics"]["ifl_seconds"] is None
    assert exported["requests"][1]["streaming"]["source_send_span_seconds"] == 3
    assert (tmp_path / "results.html").is_file()


def test_streaming_client_waits_for_server_cleanup_before_return(tmp_path):
    import asyncio

    import websockets
    from continuous_source_client import parser, run

    source = tmp_path / "source.rgb"
    source.write_bytes(bytes(72))

    async def exercise():
        completed, release = asyncio.Event(), asyncio.Event()

        async def peer(ws):
            await streaming_peer(ws, [])
            completed.set()
            await release.wait()

        async with websockets.serve(peer, "127.0.0.1", 0) as server:
            args = parser().parse_args(
                [
                    "--url",
                    f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                    "--source-rgb",
                    str(source),
                    "--width",
                    "2",
                    "--height",
                    "2",
                    "--prompt",
                    "test",
                    "--output",
                    str(tmp_path / "result"),
                ]
            )
            task = asyncio.create_task(run(args))
            try:
                await asyncio.wait_for(completed.wait(), 2)
                assert not task.done()
            finally:
                release.set()
            assert (await asyncio.wait_for(task, 2))["completed"]

    asyncio.run(exercise())


def test_streaming_benchmark_stops_after_failed_warmup_and_keeps_receipt(
    tmp_path, monkeypatch
):
    import asyncio
    from builtins import ExceptionGroup
    from types import SimpleNamespace

    import bench_streaming_source as streaming

    calls = []

    async def fail(args):
        calls.append(args.output)
        raise ExceptionGroup(
            "source tasks failed", [RuntimeError("source disconnected")]
        )

    monkeypatch.setattr(streaming, "run", fail)
    case = dict(
        label="5s",
        case_id="fixture",
        frames=6,
        width=2,
        height=2,
        source_video="video.mkv",
        prompt="p",
        references=[],
        reference_ids=[],
    )
    args = SimpleNamespace(
        output_dir=tmp_path,
        paced=False,
        warmup_runs=1,
        measured_runs=3,
        seed=123,
        url="ws://unused",
        timeout=10,
        ffmpeg="ffmpeg",
        batch_pattern=[1],
    )
    with pytest.raises(RuntimeError, match="source tasks failed"):
        asyncio.run(streaming.benchmark([case], args))
    assert len(calls) == 1
    result = json.loads((tmp_path / "results.json").read_text())
    assert "source disconnected" in result["requests"][0]["error_traceback"]
    assert result["summary"]["5s"]["warmup_failed"] == 1
    assert result["summary"]["5s"]["not_run_measured"] == 3
    assert result["summary"]["5s"]["metrics"]["ttff_seconds"]["median"] is None


@pytest.mark.parametrize(
    "label,references", [("5s", 6), ("30s-wasteland", 7), ("30s-town", 7)]
)
def test_streaming_case_preflight_preserves_reference_order_and_checks_assets(
    tmp_path, monkeypatch, label, references
):
    import hashlib

    import bench_streaming_source as streaming
    from prepare_whitebox_video_smoke import local_source_path

    manifest, case_id = streaming.CASES[label]
    case = streaming.load_cases(streaming.HERE / manifest)[case_id]
    for asset in case["provenance"]["assets"]:
        path = local_source_path(asset["uri"], tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asset["uri"].encode()
        path.write_bytes(data)
        asset.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(streaming, "load_cases", lambda path: {case_id: case})
    prepared = streaming.prepare_case(label, tmp_path, tmp_path / "prepared")
    assert len(prepared["references"]) == references
    assert prepared["reference_ids"] == [
        item["content_id"] for item in case["messages"][1]["conditions"]
    ]
    assert prepared["frames"] == (120 if label == "5s" else 720)
    assert (
        prepared["conditioning"]
        == "text_and_references_plus_first_accepted_source_frame"
    )
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from frozen manifest"):
        streaming.prepare_case(label, tmp_path, tmp_path / "bad")


def test_streaming_profile_is_saved_and_acknowledged_outside_delivery_time(tmp_path):
    import asyncio

    import websockets
    from continuous_source_client import parser, run

    source = tmp_path / "source.rgb"
    source.write_bytes(bytes(72))
    observations = []

    async def exercise():
        async with websockets.serve(
            lambda ws: streaming_peer(ws, observations), "127.0.0.1", 0
        ) as server:
            args = parser().parse_args(
                [
                    "--url",
                    f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                    "--source-rgb",
                    str(source),
                    "--width",
                    "2",
                    "--height",
                    "2",
                    "--prompt",
                    "test",
                    "--output",
                    str(tmp_path / "result"),
                    "--receive-stage-timing",
                ]
            )
            result = await run(args)
            assert observations[0]["return_stage_timing"] is True
            assert observations[-1] == {
                "type": "profile_ack",
                "session_id": "test-session",
            }
            assert (
                json.loads((tmp_path / "result/profile.json").read_text())["session_id"]
                == "test-session"
            )
            assert result["elapsed_seconds"] - result["e2e_seconds"] >= 0.03

    asyncio.run(exercise())


def test_benchmark_defaults_to_host_request_and_can_opt_out():
    from bench_client import parser

    required = [
        "--media-root",
        "media",
        "--sampling-params",
        "sampling.json",
        "--deployment",
        "deployment.json",
        "--output-dir",
        "out",
    ]
    assert parser().parse_args(required).receive_stage_timing
    assert (
        not parser().parse_args([*required, "--no-stage-timing"]).receive_stage_timing
    )
