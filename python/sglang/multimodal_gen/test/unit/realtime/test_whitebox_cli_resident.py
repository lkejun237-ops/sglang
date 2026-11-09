# SPDX-License-Identifier: Apache-2.0
"""Opt-in official CLI/CDN/cache/Room integration; real Chromium, no GPU/model."""

import asyncio
import json
import os
import ssl
import subprocess
from pathlib import Path

import httpx
import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import cli_room
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec


@pytest.mark.skipif(
    not os.environ.get("WHITEBOX_TEST_CLI_SDK"),
    reason="Requires a tested SDK checkout with Chromium",
)
@pytest.mark.parametrize("publication_ready", [True, False])
def test_official_cli_prefetch_shares_managed_cache_and_reuses_resident(
    tmp_path, monkeypatch, publication_ready
):
    async def run():
        sdk = Path(os.environ["WHITEBOX_TEST_CLI_SDK"]).resolve()
        cert, key = tmp_path / "ca.pem", tmp_path / "key.pem"
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-subj",
                "/CN=localhost",
                "-addext",
                "subjectAltName=IP:127.0.0.1",
                "-days",
                "1",
                "-keyout",
                str(key),
                "-out",
                str(cert),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for name, value in {
            "WHITEBOX_TEST_PLAYABLE_READY": str(publication_ready).lower(),
            "WHITEBOX_TEST_TLS_CERT": str(cert),
            "WHITEBOX_TEST_TLS_KEY": str(key),
            "NODE_EXTRA_CA_CERTS": str(cert),
            "WHITEBOX_CLI_ENTRYPOINT": str(sdk / "packages/cli/bin/worldkit.mjs"),
            "WHITEBOX_CLI_RENDERING": "software",
            "WHITEBOX_CLI_RESOURCE_TRANSPORT": "cdn",
            "WHITEBOX_CLI_GAME_CACHE_DIRECTORY": str(tmp_path / "games"),
            "WHITEBOX_CLI_SERVICE": "auto",
            "WHITEBOX_CLI_SERVICE_NAME": "whitebox-local",
            "WORLDKIT_SERVICE_DIR": str(tmp_path / "services"),
            "WORLDKIT_SESSION_DIR": str(tmp_path / "sessions"),
            "WORLD_TEST_API_TOKEN": "local-fixture-token",
        }.items():
            monkeypatch.setenv(name, value)

        async def cli(*args):
            process = await asyncio.create_subprocess_exec(
                "node",
                str(sdk / "packages/cli/bin/worldkit.mjs"),
                *args,
                cwd=sdk,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await asyncio.wait_for(process.communicate(), 40)
            assert process.returncode == 0, err.decode()[-4000:]
            return json.loads(out)

        with (tmp_path / "fixture.log").open("wb") as err:
            fixture = await asyncio.create_subprocess_exec(
                "node",
                "--import",
                "tsx",
                str(Path(__file__).with_name("whitebox_wtp_fixture.mjs")),
                cwd=sdk,
                stdout=asyncio.subprocess.PIPE,
                stderr=err,
            )
            room = None
            try:
                line = await asyncio.wait_for(fixture.stdout.readline(), 120)
                assert line, (tmp_path / "fixture.log").read_text()[-4000:]
                url = json.loads(line)["url"]
                monkeypatch.setenv("WHITEBOX_CLI_PLATFORM_URL", url)
                # The resource-only prefetch must populate the SAME full game
                # cache as streaming; it must not allocate a browser/model.
                if publication_ready:
                    prepare = await asyncio.create_subprocess_exec(
                        "node",
                        "--import",
                        "tsx",
                        str(
                            Path(__file__).parents[3]
                            / "runtime/realtime/whitebox/worldkit_cdn_prepare.mjs"
                        ),
                        cwd=sdk,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    out, errors = await asyncio.wait_for(
                        prepare.communicate(b'{"worldId":"s/w"}'), 120
                    )
                    assert prepare.returncode == 0, errors.decode() + out.decode()
                    assert json.loads(out.splitlines()[-1])["state"] == "ready"
                assert not (tmp_path / "services").exists()
                service_ids = []
                roots = []
                for index in range(2):
                    room = await cli_room(
                        world_id="s/w",
                        spec=VideoSpec(160, 90),
                        trace_id="local-resident",
                    )
                    view = await room.join(player_id="local", spec=room.spec)
                    frame = await asyncio.wait_for(anext(view.frames()), 20)
                    assert len(frame.rgb) == 160 * 90 * 3
                    if publication_ready or index > 0:
                        assert view.metadata["cache"]["downloadedBytes"] == 0
                    if publication_ready or index > 0:
                        assert view.metadata["cache"]["cacheHitBytes"] > 0
                    service_ids.append(view.metadata["service"]["instanceId"])
                    roots.append(Path(room.ui_package["root"]))
                    asset = next(iter(view.renderer.files))
                    assert await view.renderer.asset(asset)
                    await room.close()
                    room = None
                    assert list((tmp_path / "sessions").glob("*.json")) == []
                    assert not roots[-1].exists()
                assert service_ids[0] == service_ids[1]
                context = ssl.create_default_context(cafile=str(cert))
                async with httpx.AsyncClient(verify=context) as http:
                    result = await http.get(
                        url + "/__metrics",
                        headers={"Authorization": "Bearer local-fixture-token"},
                    )
                assert result.json()["archives"] == 0
                assert result.json()["reconciles"] == (0 if publication_ready else 1)
            finally:
                if room:
                    await room.close()
                try:
                    await cli(
                        "serve", "stop", "--service-name", "whitebox-local", "--json"
                    )
                finally:
                    if fixture.returncode is None:
                        fixture.terminate()
                    await asyncio.wait_for(fixture.wait(), 10)

    asyncio.run(run())


@pytest.mark.skipif(
    not os.environ.get("WHITEBOX_TEST_CLI_SDK"),
    reason="Requires the public SDK reference resolver",
)
@pytest.mark.parametrize(
    "mode",
    [
        "ready",
        "changed",
        "unknown-post",
        "deadline",
        "cancel",
        "unauthenticated",
        "foreign",
    ],
)
def test_publication_preparation_preserves_one_attempt_and_owner_deadline(
    tmp_path, monkeypatch, mode
):
    import signal
    import socket
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    sdk = Path(os.environ["WHITEBOX_TEST_CLI_SDK"]).resolve()
    observed = {"posts": 0, "gets": 0, "bad_auth": 0}
    posted = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def response(self, value):
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            assert self.path == "/api/worlds/s/w/playable-location?transport=cdn"
            observed["gets"] += 1
            observed["bad_auth"] += (
                self.headers.get("Authorization") != "Bearer local-only"
            )
            if mode == "ready":
                self.response({"ready": True, "location": {"publicationId": "p"}})
            elif mode == "changed" and observed["posts"]:
                self.response(
                    {"ready": True, "location": {"publicationId": "replacement"}}
                )
            else:
                self.response({"ready": False, "publicationId": "p"})

        def do_POST(self):
            assert self.path == "/api/cdn/playables/reconcile"
            assert json.loads(self.rfile.read(int(self.headers["Content-Length"]))) == {
                "sourceId": "s",
                "worldId": "w",
            }
            observed["posts"] += 1
            posted.set()
            if mode == "unknown-post":
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
            else:
                self.response({"accepted": True})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "WHITEBOX_CLI_ENTRYPOINT", str(sdk / "packages/cli/bin/worldkit.mjs")
    )
    monkeypatch.setenv(
        "WHITEBOX_CLI_PLATFORM_URL", f"http://127.0.0.1:{server.server_port}"
    )
    monkeypatch.setenv("WORLD_TEST_API_TOKEN", "local-only")
    if mode == "unauthenticated":
        monkeypatch.delenv("WORLD_TEST_API_TOKEN")

    async def run():
        process = await asyncio.create_subprocess_exec(
            "node",
            "--import",
            "tsx",
            str(
                Path(__file__).parents[3]
                / "runtime/realtime/whitebox/worldkit_publication_prepare.mjs"
            ),
            cwd=sdk,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        task = asyncio.create_task(
            process.communicate(
                json.dumps(
                    {
                        "worldId": "https://foreign.invalid/play/s/w"
                        if mode == "foreign"
                        else "s/w",
                        "timeoutSeconds": 0.3 if mode == "deadline" else 5,
                    }
                ).encode()
            )
        )
        try:
            if mode == "cancel":
                assert await asyncio.to_thread(posted.wait, 8)
                process.send_signal(signal.SIGTERM)
            out, err = await asyncio.wait_for(task, 10)
            if mode == "ready":
                assert process.returncode == 0, err.decode()
                assert json.loads(out)["ready"] is True
            else:
                assert process.returncode != 0
                assert out == b""
                expected = {
                    "changed": "WTP_PLAYABLE_PUBLICATION_CHANGED",
                    "foreign": "WTP_REFERENCE_ORIGIN_MISMATCH",
                    "unauthenticated": "WTP_PLAYABLE_AUTHENTICATION_REQUIRED",
                }.get(mode, "WTP_PLAYABLE_PREPARATION_FAILED")
                assert err.decode().strip() == expected
            assert observed["posts"] == (
                0 if mode in {"ready", "foreign", "unauthenticated"} else 1
            )
            assert observed["bad_auth"] == 0
            assert (
                (observed["gets"] == 0)
                if mode in {"foreign", "unauthenticated"}
                else (observed["gets"] >= 1)
            )
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()
