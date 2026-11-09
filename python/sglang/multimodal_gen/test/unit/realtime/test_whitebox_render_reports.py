# SPDX-License-Identifier: Apache-2.0
import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.realtime.whitebox.recording import ArchiveManager
from sglang.multimodal_gen.runtime.realtime.whitebox.reports import Reports


def test_source_capability_exposes_only_owned_segments_and_current_model_metrics(
    tmp_path,
):
    parent, child, foreign = "a" * 32, "b" * 32, "c" * 32
    base = dict(
        trace_id="trace",
        recording_status="complete",
        upload_status="complete",
        dataset_status="excluded_unvalidated",
        timeline={},
        videos={},
    )
    source = dict(
        base,
        recording_id=parent,
        recording_kind="source",
        source_session_id="source",
        source_epoch="epoch",
        object_prefix="source/",
        trace=[dict(stage="cli_start", count=1, mean_ms=25)],
        render_segments=[
            dict(recording_id=foreign),
            dict(recording_id=child, region_id="garden"),
        ],
    )
    model = dict(
        base,
        recording_id=child,
        recording_kind="render_segment",
        source_session_id="source",
        source_epoch="epoch",
        object_prefix="source/segments/" + child + "/",
        trace=[dict(stage="model_dit", count=1, mean_ms=400)],
        videos=dict(model=dict(path="model.mp4", codec="h264", frames=4)),
    )
    other = dict(
        model,
        recording_id=foreign,
        source_session_id="someone-else",
        object_prefix="foreign/",
    )
    for name, value in ((parent, source), (child, model), (foreign, other)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps(value))
        (directory / "inputs.jsonl").write_text(
            json.dumps({"prompt": "changed", "start_frame": 0, "end_frame": 4}) + "\n"
        )
    reports = tmp_path / ".reports"
    reports.mkdir()
    (reports / (parent + ".json")).write_text(
        json.dumps({"token_hash": hashlib.sha256(b"parent-token").hexdigest()})
    )
    archives = ArchiveManager()
    archives.store = SimpleNamespace(
        playback_url=lambda key: "https://archive.invalid/" + key, bucket="test"
    )
    app = FastAPI()
    Reports(tmp_path, archives).install(app)
    with TestClient(app) as client:
        path = "/v1/whitebox/recordings/" + parent
        headers = {"Authorization": "Bearer parent-token"}
        value = client.get(path, headers=headers).json()
        assert {item["stage"] for item in value["timings"]} == {
            "cli_start",
            "model_dit",
        }
        assert len(value["videos"]) == 1 and value["videos"][0]["url"].startswith(
            "https://archive.invalid/source/segments/" + child
        )
        assert value["render_segments"][0]["recording_status"] == "unavailable"
        url = path + "/" + value["videos"][0]["inputs_path"]
        assert (
            client.get(url, headers=headers).json()["intervals"][0]["prompt"]
            == "changed"
        )
        assert (
            client.get(url, headers={"Authorization": "Bearer wrong"}).status_code
            == 404
        )
        assert (
            client.get(
                path + "/segments/" + foreign + "/inputs", headers=headers
            ).status_code
            == 404
        )
        assert (
            client.get("/v1/whitebox/recordings/" + child, headers=headers).status_code
            == 404
        )
    asyncio.run(archives.close())


@pytest.mark.parametrize("live", [True, False])
def test_concurrent_required_ui_assets_wait_for_bounded_io_instead_of_failing(
    tmp_path, monkeypatch, live
):
    import threading
    from pathlib import Path

    import httpx

    from sglang.multimodal_gen.runtime.realtime.whitebox.reports import SessionReport

    async def run():
        recording_id = "d" * 32
        directory = tmp_path / recording_id / "ui_bundle"
        directory.mkdir(parents=True)
        files = {
            name: ("verified " + name).encode()
            for name in (
                "components.mjs",
                "catalog.json",
                "definition.json",
                "state-schema.json",
            )
        }
        for name, data in files.items():
            (directory / name).write_bytes(data)
        report = SessionReport("required-ui-assets")
        report.recording = SimpleNamespace(
            pair=True,
            manifest={
                "ui": {
                    "files": {
                        name: hashlib.sha256(data).hexdigest()
                        for name, data in files.items()
                    }
                }
            },
        )
        reports = Reports(tmp_path, SimpleNamespace())
        access = tmp_path / ".reports"
        if live:
            reports.live[recording_id] = report
        else:
            access.mkdir()
            (access / (recording_id + ".json")).write_text(
                json.dumps(
                    {"token_hash": hashlib.sha256(report.token.encode()).hexdigest()}
                )
            )
            (directory.parent / "manifest.json").write_text(
                json.dumps(report.recording.manifest)
            )
        app = FastAPI()
        reports.install(app)
        gate = threading.Event()
        original = Path.is_file

        def delayed_validation(path):
            if path.parent in (directory, access):
                gate.wait(2)
            return original(path)

        monkeypatch.setattr(Path, "is_file", delayed_validation)
        requests = []
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://ui-owner"
            ) as client:
                requests = [
                    asyncio.create_task(
                        client.get(
                            f"/v1/whitebox/recordings/{recording_id}/ui/{name}",
                            headers={"Authorization": "Bearer " + report.token},
                        )
                    )
                    for name in files
                ]
                async with asyncio.timeout(1):
                    while reports.io_active < 2:
                        await asyncio.sleep(0)
                await asyncio.sleep(0.02)
                assert reports.io_active == 2
                gate.set()
                responses = await asyncio.gather(*requests)
                assert [response.status_code for response in responses] == [200] * 4
                for response, data in zip(responses, files.values()):
                    assert response.content == data
                    assert (
                        response.headers["x-content-sha256"]
                        == hashlib.sha256(data).hexdigest()
                    )
                assert (
                    await client.get(
                        f"/v1/whitebox/recordings/{recording_id}/ui/catalog.json",
                        headers={"Authorization": "Bearer foreign"},
                    )
                ).status_code == 404
                assert (
                    await client.get(
                        f"/v1/whitebox/recordings/{recording_id}/ui/undeclared.json",
                        headers={"Authorization": "Bearer " + report.token},
                    )
                ).status_code == 404
                assert reports.io_active == 0
        finally:
            gate.set()
            await asyncio.gather(*requests, return_exceptions=True)
            report.close()

    asyncio.run(run())


def test_ui_io_waiters_are_bounded_cancelled_and_share_the_worker_deadline(tmp_path):
    import threading

    from fastapi import HTTPException

    async def run():
        reports = Reports(tmp_path, SimpleNamespace())
        reports.io_timeout = 0.03
        reports.ui_io_max_waiters = 1
        gate = threading.Event()
        calls = []

        def stalled():
            calls.append("worker")
            gate.wait(2)

        waiting = None
        workers = [asyncio.create_task(reports.run_io(stalled)) for _ in range(2)]
        try:
            results = await asyncio.gather(*workers, return_exceptions=True)
            assert all(
                isinstance(result, HTTPException) and result.status_code == 503
                for result in results
            )
            assert reports.io_active == 2
            waiting = asyncio.create_task(reports.run_io(stalled, wait_for_slot=True))
            async with asyncio.timeout(1):
                while reports.ui_io_waiters != 1:
                    await asyncio.sleep(0)
            with pytest.raises(HTTPException, match="Recording storage is busy"):
                await reports.run_io(stalled, wait_for_slot=True)
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            assert reports.ui_io_waiters == 0 and reports.io_active == 2
            with pytest.raises(HTTPException, match="Recording storage is slow"):
                await reports.run_io(stalled, wait_for_slot=True)
            # A timed-out request cannot free a still-running thread or create
            # more workers; waiting, cancellation and overload stay local.
            assert calls == ["worker", "worker"]
            assert reports.ui_io_waiters == 0 and reports.io_active == 2
        finally:
            gate.set()
            if waiting:
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
            async with asyncio.timeout(1):
                while reports.io_active:
                    await asyncio.sleep(0.001)
        assert (
            await reports.run_io(lambda: "released", wait_for_slot=True) == "released"
        )

    asyncio.run(run())
