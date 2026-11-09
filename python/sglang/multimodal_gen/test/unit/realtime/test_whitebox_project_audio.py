# SPDX-License-Identifier: Apache-2.0
"""CPU-only loopback tests for the sealed project-audio bridge."""

import asyncio
import copy
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from sglang.multimodal_gen.runtime.realtime.whitebox.project_audio import (
    AUDIO_PREFIX,
    audio_path_allowed,
    install_audio_package,
    read_audio_snapshot,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.reports import Reports


def snapshot():
    return dict(
        type="audio.snapshot",
        version=1,
        sessionId="session",
        epoch=1,
        source=dict(
            presentationId="presentation",
            sdkEpoch=1,
            sourceFrameId=4,
            sourceTimeUs=1000,
            simulationTick=4,
            worldRevision=0,
        ),
        completeThroughUs=1000,
        eventsCompleteFromUs=0,
        events=[
            dict(
                entityId="actor",
                sequence=1,
                soundId="step",
                resource="step",
                gain=1,
                command="play",
                sourceTimeUs=900,
            )
        ],
        voices=[],
    )


def test_snapshot_preserves_history_and_rejects_invalid_clock():
    original = snapshot()
    accepted = read_audio_snapshot(original)
    assert accepted == original
    accepted["events"].clear()
    assert len(original["events"]) == 1
    supported = copy.deepcopy(original)
    supported["events"][0].update(
        entityId="entity" * 100, soundId="sound" * 100, resource="alias" * 200, gain=17
    )
    assert read_audio_snapshot(supported) == supported
    for edit in (
        lambda s: s.update(version=True),
        lambda s: s.update(completeThroughUs=999),
        lambda s: s["events"][0].update(sourceTimeUs=1001),
        lambda s: s["events"][0].update(positionMetersXYZ=[float("nan"), 0, 0]),
        lambda s: s.update(voices=[{}]),
    ):
        invalid = copy.deepcopy(original)
        edit(invalid)
        with pytest.raises(ValueError):
            read_audio_snapshot(invalid)


@pytest.mark.parametrize(
    "path",
    [
        "../secret",
        "assets/audio/../secret",
        "assets/audio/%2fsecret",
        "assets/audio/a?x",
        "https://example.com/a",
        "assets/audio//a",
        "assets/audio/a\\b",
    ],
)
def test_audio_paths_deny_traversal_and_remote_urls(path):
    assert not audio_path_allowed(path)


@pytest.mark.asyncio
async def test_real_http_session_capability_hash_and_recording_route(tmp_path):
    raw = b"sealed-test-audio-bytes"
    resource = dict(
        id="step",
        path="assets/audio/step.wav",
        byteLength=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        decoded=dict(
            channels=1,
            sampleRate=8000,
            frames=8,
            decodedBytes=32,
            durationSeconds=0.001,
        ),
    )
    manifest = json.dumps(
        dict(
            contract="worldkit.project-audio-byte-seal",
            version=1,
            bindings={},
            resources=[resource],
        )
    ).encode()
    requests = []
    content = {"project-audio.json": manifest, resource["path"]: raw}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            if self.headers.get("Authorization") != "Bearer native-token":
                self.send_error(404)
                return
            name = self.path.removeprefix("/v1/sessions/session/audio/")
            if name not in content:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(content[name])

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        descriptor = dict(
            baseUrl=base,
            sessionId="session",
            accessToken="native-token",
            audio=dict(
                manifestUrl=base + "/v1/sessions/session/audio/project-audio.json",
                manifestSha256=hashlib.sha256(manifest).hexdigest(),
            ),
        )

        async def original_asset(_):
            raise FileNotFoundError()

        renderer = SimpleNamespace(
            asset=original_asset, metadata=dict(mode="browser", files={})
        )
        view = SimpleNamespace(descriptor=descriptor, renderer=renderer, metadata={})
        await install_audio_package(view)
        assert requests == [
            ("/v1/sessions/session/audio/project-audio.json", "Bearer native-token"),
            ("/v1/sessions/session/audio/assets/audio/step.wav", "Bearer native-token"),
        ]
        assert view.metadata["project_audio"] == {
            "manifestSha256": hashlib.sha256(manifest).hexdigest()
        }
        assert await renderer.asset(AUDIO_PREFIX + resource["path"]) == raw
        with pytest.raises(FileNotFoundError):
            await renderer.asset(AUDIO_PREFIX + "assets/audio/unlisted.wav")
        # Same authenticated recording route, including cold archive readback.
        recording_id = "a" * 32
        directory = tmp_path / recording_id
        for name in content:
            target = directory / "ui_bundle" / AUDIO_PREFIX / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content[name])
        (directory / "manifest.json").write_text(json.dumps(dict(ui=renderer.metadata)))
        reports = Reports(tmp_path, SimpleNamespace())
        token = "report-token"
        access = tmp_path / ".reports"
        access.mkdir()
        (access / (recording_id + ".json")).write_text(
            json.dumps(dict(token_hash=hashlib.sha256(token.encode()).hexdigest()))
        )
        app = FastAPI()
        reports.install(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://recording"
        ) as client:
            url = f"/v1/whitebox/recordings/{recording_id}/audio/"
            assert (await client.get(url + "project-audio.json")).status_code == 404
            for name in content:
                response = await client.get(
                    url + name, headers={"Authorization": "Bearer " + token}
                )
                assert response.status_code == 200
                assert response.content == content[name]
                assert (
                    response.headers["x-content-sha256"]
                    == hashlib.sha256(content[name]).hexdigest()
                )
            assert (
                await client.get(
                    url + "assets/audio/unlisted.wav",
                    headers={"Authorization": "Bearer " + token},
                )
            ).status_code == 404
            (directory / "ui_bundle" / AUDIO_PREFIX / resource["path"]).write_bytes(
                b"changed"
            )
            assert (
                await client.get(
                    url + resource["path"], headers={"Authorization": "Bearer " + token}
                )
            ).status_code == 404
        # Seals are verified before the package becomes visible to playback.
        content[resource["path"]] = b"changed"
        with pytest.raises(ValueError, match="identity changed"):
            await install_audio_package(
                SimpleNamespace(
                    descriptor=descriptor,
                    renderer=SimpleNamespace(asset=original_asset, metadata={}),
                    metadata={},
                )
            )
        descriptor["audio"]["manifestUrl"] = "https://example.com/arbitrary"
        count = len(requests)
        with pytest.raises(ValueError, match="capability"):
            await install_audio_package(view)
        assert len(requests) == count
    finally:
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.mark.asyncio
async def test_real_websocket_retains_source_history_and_listener():
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import CLIView

    message = snapshot()
    message["listener"] = dict(
        positionMetersXYZ=[1, 2, 3], forwardXYZ=[0, 0, -1], upXYZ=[0, 1, 0]
    )
    newer = copy.deepcopy(message)
    newer["source"]["sourceTimeUs"] = newer["completeThroughUs"] = 2000
    newer["source"]["sourceFrameId"] = newer["source"]["simulationTick"] = 5

    async def sender(socket):
        for value in (
            message,
            newer,
            dict(type="session.ended", epoch=1, reason="stopped"),
        ):
            await socket.send(json.dumps(value))

    room = SimpleNamespace(
        descriptor=dict(sessionId="session", epoch=1),
        spec=SimpleNamespace(fps=30),
        prepared_delivery=None,
    )
    view = CLIView(room)
    async with serve(sender, "127.0.0.1", 0) as server:
        async with connect(
            f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        ) as socket:
            view.control = socket
            await view._control()
    assert view.audio_snapshot == newer
    view.descriptor["audio"] = {}
    assert view._frame_audio(message) == {"audio_snapshot": message}
    assert view._frame_audio(newer) == {"audio_snapshot": newer}
    assert view.audio_snapshot["events"][0]["sourceTimeUs"] == 900
    assert view.audio_snapshot["listener"] == message["listener"]
    assert view.stopped


@pytest.mark.asyncio
async def test_real_websocket_rejects_foreign_audio_source():
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import CLIView

    message = snapshot()
    message["sessionId"] = "foreign"

    async def sender(socket):
        await socket.send(json.dumps(message))

    room = SimpleNamespace(
        descriptor=dict(sessionId="session", epoch=1),
        spec=SimpleNamespace(fps=30),
        prepared_delivery=None,
    )
    view = CLIView(room)
    async with serve(sender, "127.0.0.1", 0) as server:
        async with connect(
            f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        ) as socket:
            view.control = socket
            with pytest.raises(ValueError, match="source identity changed"):
                await view._control()
    assert view.audio_snapshot is None


def test_static_loop_keeps_every_exact_source_audio_snapshot_in_replay():
    from test_whitebox_ui_history import observation, owner, publish

    pipeline, live = owner()
    for index in range(1, 61):
        message = observation(index)
        header = message["source"]["cli_frame_header"]
        audio = snapshot()
        audio.update(
            sessionId=header["sessionId"],
            epoch=header["epoch"],
            source=copy.deepcopy(header["source"]),
            completeThroughUs=header["source"]["sourceTimeUs"],
            voices=[
                dict(
                    voiceId="loop",
                    entityId="game",
                    sequence=0,
                    soundId="loop",
                    resource="tone",
                    gain=1,
                    startedAtSourceTimeUs=0,
                )
            ],
        )
        audio["listener"] = dict(
            positionMetersXYZ=[1, 2, 3], forwardXYZ=[0, 0, -1], upXYZ=[0, 1, 0]
        )
        message["states"]["audio_snapshot"] = read_audio_snapshot(audio)
        publish(pipeline, message)
    assert len(live) == len(pipeline.history) == 60
    assert [
        row[2]["states"]["audio_snapshot"]["source"]["sourceFrameId"]
        for row in pipeline.history
    ] == list(range(1, 61))


@pytest.mark.asyncio
async def test_source_reset_retires_previous_audio_snapshot():
    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import NativeCLIView

    room = SimpleNamespace(
        descriptor=dict(sessionId="session", epoch=1),
        spec=SimpleNamespace(fps=30),
        prepared_delivery=None,
    )
    view = NativeCLIView(room)
    view.audio_snapshot = snapshot()

    async def ignore(*_):
        pass

    view.renderer = SimpleNamespace(consume_ui=ignore)
    view.control = SimpleNamespace(send=ignore)
    await view._source_reset(
        dict(
            type="source.reset", epoch=1, presentationId="new-presentation", sdkEpoch=2
        )
    )
    assert view.audio_snapshot is None


@pytest.mark.asyncio
async def test_no_audio_session_does_not_read_resources_or_mutate_metadata():
    view = SimpleNamespace(descriptor={}, metadata={"unchanged": True})
    await install_audio_package(view)
    assert view.metadata == {"unchanged": True}


def test_ready_ui_projection_excludes_audio_archive_bytes_without_mutation():
    from sglang.multimodal_gen.runtime.realtime.whitebox.project_audio import (
        public_ui_metadata,
    )

    metadata = {
        "mode": "browser",
        "files": {
            "module.js": "a" * 64,
            "__project_audio/assets/audio/bgm.wav": "b" * 64,
        },
    }
    public = public_ui_metadata(metadata)
    assert public["files"] == {"module.js": "a" * 64}
    assert len(metadata["files"]) == 2
    assert public_ui_metadata(public) is public
