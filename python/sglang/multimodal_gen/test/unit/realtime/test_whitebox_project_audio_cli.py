# SPDX-License-Identifier: Apache-2.0
"""Opt-in actual SDK author/compiler/CLI -> SG source-only audio; no model peer."""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import httpx
import msgspec
import pytest
from fastapi import FastAPI
from test_whitebox_cli_integration import server
from websockets.asyncio.client import connect

from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
    AsyncArchiveManager,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import cli_room
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import RenderHost
from sglang.multimodal_gen.runtime.realtime.whitebox.reports import Reports
from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
    MemoryRoomStore,
    RoomDirectory,
    new_id,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig


@pytest.mark.skipif(
    not os.environ.get("WHITEBOX_TEST_CLI_SDK"),
    reason="Requires candidate SDK checkout and Chromium",
)
@pytest.mark.parametrize("start_tick", [1, 120])
@pytest.mark.asyncio
async def test_real_cli_no_inference_project_audio(tmp_path, monkeypatch, start_tick):
    assert not os.environ.get("WHITEBOX_TEST_GPU_COORDINATOR_URL"), (
        "CPU audio lane refuses GPU coordinator"
    )
    monkeypatch.setenv("WHITEBOX_TEST_PROJECT_AUDIO", "1")
    monkeypatch.setenv("WHITEBOX_TEST_AUDIO_START_TICK", str(start_tick))
    monkeypatch.delenv("WHITEBOX_BOUND_COMPUTE", raising=False)
    sdk = Path(os.environ["WHITEBOX_TEST_CLI_SDK"]).resolve()
    log = (tmp_path / "fixture.log").open("wb")
    fixture = await asyncio.create_subprocess_exec(
        "node",
        "--import",
        "tsx",
        str(Path(__file__).with_name("whitebox_wtp_fixture.mjs")),
        cwd=sdk,
        stdout=asyncio.subprocess.PIPE,
        stderr=log,
    )
    host = runtime = None
    try:
        line = await asyncio.wait_for(fixture.stdout.readline(), 120)
        assert line, (tmp_path / "fixture.log").read_text()[-5000:]
        startup = json.loads(line)
        url = startup["url"]
        assert startup["candidate"]["runtimeCacheHit"] is False
        (tmp_path / "audio-runtime-identity.json").write_text(
            json.dumps(startup["candidate"], indent=2)
        )
        for key, value in dict(
            WHITEBOX_CLI_ENTRYPOINT=str(sdk / "packages/cli/bin/worldkit.mjs"),
            WHITEBOX_CLI_PLATFORM_URL=url,
            WORLD_TEST_API_TOKEN="local-fixture-token",
            WHITEBOX_CLI_RENDERING="software",
            WHITEBOX_CLI_SERVICE="off",
            WHITEBOX_CLI_RESOURCE_TRANSPORT="archive",
            WHITEBOX_CLI_STARTUP_SECONDS="90",
        ).items():
            monkeypatch.setenv(key, value)
        spec = VideoSpec(832, 480)
        archives = AsyncArchiveManager(queue_depth=256)
        directory = RoomDirectory(MemoryRoomStore(), {"audio-test": "http://loopback"})
        game = new_id()
        await directory.resolve(game, "s/w", create=True)
        runtime = RoomRuntime(
            directory=directory,
            owner_id="audio-test",
            driver_factory=cli_room,
            spool=tmp_path / "room",
            spec=spec,
            archives=archives,
        )
        reports = Reports(runtime.spool, archives)
        host = RenderHost(
            runtime,
            WhiteboxConfig(
                engine_factory=None,
                ui_factory=None,
                spool=runtime.spool,
                spec=spec,
                room_owned=True,
                startup_seconds=90,
                drain_seconds=3,
            ),
            archives,
            reports,
            None,
            None,
        )

        def forbidden(*_):
            pytest.fail("No-inference audio test attempted to create a model lease")

        host.make_segment_leases = forbidden
        app = FastAPI()
        host.install(app, "local-audio-token")
        reports.install(app)
        ready = evidence = None
        evidence_at = None
        first_audio_time = None
        audio_keys, presented_keys = set(), set()
        loop_starts = set()

        def source_key(header):
            source = header["source"]
            return (
                header["sessionId"],
                header["epoch"],
                source["presentationId"],
                source["sdkEpoch"],
                source["sourceFrameId"],
                source["sourceTimeUs"],
            )

        async with server(app) as address:
            async with connect(
                address.replace("http", "ws") + "/internal/render-v2",
                additional_headers={"Authorization": "Bearer local-audio-token"},
                max_size=16 << 20,
            ) as ws:
                await ws.send(
                    msgspec.msgpack.encode(
                        dict(
                            user_id="local-audio",
                            trace_id="local-audio",
                            message=dict(
                                type="init",
                                control_protocol="whitebox-render-v2",
                                source_operation="create.manual",
                                inference_control="manual",
                                prompt="local project audio",
                                whitebox=dict(game_id=game, world_id="s/w"),
                            ),
                        )
                    )
                )
                async with asyncio.timeout(100):
                    while evidence_at is None or time.monotonic() - evidence_at < 3:
                        message = msgspec.msgpack.decode(await ws.recv())
                        assert message.get("type") != "error", message
                        if message.get("type") == "session.ready":
                            ready = message
                            assert ready["inference"] == dict(
                                enabled=False, revision=0, state="disabled"
                            )
                            assert ready["project_audio"]["manifestSha256"]
                            for stream in ("whitebox", "model"):
                                await ws.send(
                                    msgspec.msgpack.encode(
                                        dict(
                                            type="event",
                                            kind="playback_ack",
                                            payload=dict(
                                                protocol="whitebox-dual-v2",
                                                stream=stream,
                                                media_stream_id=(
                                                    ready["source_epoch"]
                                                    if stream == "whitebox"
                                                    else None
                                                ),
                                                frame=-1,
                                                window=4,
                                            ),
                                        )
                                    )
                                )
                        if message.get("type") == "whitebox.states":
                            audio = message.get("states", {}).get("audio_snapshot")
                            if audio:
                                audio_keys.add(source_key(audio))
                                loop_starts.update(
                                    (voice["voiceId"], voice["startedAtSourceTimeUs"])
                                    for voice in audio["voices"]
                                )
                            if (
                                audio
                                and audio["voices"]
                                and any(
                                    row.get("positionMetersXYZ")
                                    for row in audio["events"]
                                )
                            ):
                                evidence = message
                                if evidence_at is None:
                                    evidence_at = time.monotonic()
                                    first_audio_time = audio["source"]["sourceTimeUs"]
                        if message.get("type") in {"whitebox.frame", "frame_batch"}:
                            if (
                                evidence_at is not None
                                and message.get("stream") == "whitebox"
                                and message["source"]["cli_frame_header"]["source"][
                                    "sourceTimeUs"
                                ]
                                >= first_audio_time
                            ):
                                presented_keys.add(
                                    source_key(message["source"]["cli_frame_header"])
                                )
                            await ws.send(
                                msgspec.msgpack.encode(
                                    dict(
                                        type="event",
                                        kind="playback_ack",
                                        payload=dict(
                                            protocol="whitebox-dual-v2",
                                            stream=message["stream"],
                                            media_stream_id=message.get(
                                                "media_stream_id"
                                            ),
                                            frame=message["output_frame_start"],
                                            window=4,
                                        ),
                                    )
                                )
                            )
                assert len(loop_starts) == 1
                assert len(presented_keys) >= 24
                assert presented_keys <= audio_keys, (
                    f"Missing exact audio proof for {len(presented_keys - audio_keys)} accepted source frames"
                )
                (tmp_path / "audio-bridge-evidence.json").write_text(
                    json.dumps(
                        dict(
                            startTick=start_tick,
                            acceptedPreviewFrames=len(presented_keys),
                            audioSnapshotFrames=len(audio_keys),
                            stableLoopSeconds=time.monotonic() - evidence_at,
                            missingExactAudioProof=len(presented_keys - audio_keys),
                            inferenceEnabled=False,
                            audioResourcePrefetch=ready["engine"][
                                "audio_resource_prefetch"
                            ],
                        ),
                        indent=2,
                    )
                )
                assert len(runtime.players) == len(runtime.rooms) == 1
                owner = next(iter(host.sessions.values()))
                assert owner.session.current_segment is None
                audio = evidence["states"]["audio_snapshot"]
                assert (
                    audio["sessionId"]
                    == evidence["source"]["cli_frame_header"]["sessionId"]
                )
                async with httpx.AsyncClient(trust_env=False) as client:
                    base = (
                        address
                        + f"/v1/whitebox/recordings/{ready['recording_id']}/audio/"
                    )
                    assert (
                        await client.get(base + "project-audio.json")
                    ).status_code == 404
                    headers = {"Authorization": "Bearer " + ready["report_token"]}
                    response = await client.get(
                        base + "project-audio.json", headers=headers
                    )
                    assert response.status_code == 200
                    assert (
                        hashlib.sha256(response.content).hexdigest()
                        == ready["project_audio"]["manifestSha256"]
                    )
                    for resource in response.json()["resources"]:
                        response = await client.get(
                            base + resource["path"], headers=headers
                        )
                        assert response.status_code == 200
                        assert (
                            hashlib.sha256(response.content).hexdigest()
                            == resource["sha256"]
                        )
                    assert (
                        await client.get(
                            base + "assets/audio/not-declared.wav", headers=headers
                        )
                    ).status_code == 404
                    from sglang.multimodal_gen.runtime.realtime.whitebox.room_proxy import (
                        RoomProxy,
                    )

                    proxy = RoomProxy(
                        None, {"audio-test": address}, "local-audio-token", client, None
                    )
                    gateway = FastAPI()
                    proxy.install(gateway)
                    async with server(gateway) as gateway_address:
                        owner_url = (
                            gateway_address + "/v1/" + ready["report_path"] + "/audio/"
                        )
                        denied = await client.get(owner_url + "project-audio.json")
                        assert denied.status_code == 403
                        owner_manifest = await client.get(
                            owner_url + "project-audio.json", headers=headers
                        )
                        assert owner_manifest.status_code == 200
                        assert (
                            hashlib.sha256(owner_manifest.content).hexdigest()
                            == ready["project_audio"]["manifestSha256"]
                        )
                        for resource in owner_manifest.json()["resources"]:
                            owner_bytes = await client.get(
                                owner_url + resource["path"], headers=headers
                            )
                            assert owner_bytes.status_code == 200
                            assert (
                                hashlib.sha256(owner_bytes.content).hexdigest()
                                == resource["sha256"]
                            )
    finally:
        if host:
            await host.close()
        if runtime:
            await runtime.close()
        if fixture.returncode is None:
            fixture.terminate()
            try:
                await asyncio.wait_for(fixture.wait(), 10)
            except TimeoutError:
                fixture.kill()
                await fixture.wait()
        log.close()
        assert fixture.returncode is not None
