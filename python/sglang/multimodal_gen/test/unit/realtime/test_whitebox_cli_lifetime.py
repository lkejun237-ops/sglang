# SPDX-License-Identifier: Apache-2.0
"""SDK terminal messages over real local sockets; no model or cloud service."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from websockets.asyncio.server import serve

from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import CLIRoom
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec


@pytest.mark.parametrize(
    "stderr,expected",
    [
        (b"WTP_PLAYABLE_PREPARATION_FAILED\n", "WTP_PLAYABLE_PREPARATION_FAILED"),
        (b"WTP_PLAYABLE_PUBLICATION_CHANGED\n", "WTP_PLAYABLE_PUBLICATION_CHANGED"),
        (
            b"WTP_PLAYABLE_AUTHENTICATION_REQUIRED\n",
            "WTP_PLAYABLE_AUTHENTICATION_REQUIRED",
        ),
        (b"WTP_REFERENCE_ORIGIN_MISMATCH\n", "WTP_REFERENCE_ORIGIN_MISMATCH"),
        (b"private/token/path\n", "WTP_PLAYABLE_PREPARATION_FAILED"),
        (
            b"WTP_PLAYABLE_PUBLICATION_CHANGED\nprivate/token",
            "WTP_PLAYABLE_PREPARATION_FAILED",
        ),
    ],
)
def test_publication_preparation_preserves_only_known_public_errors(
    monkeypatch, stderr, expected
):
    from importlib import import_module
    from pathlib import Path

    owner = import_module("sglang.multimodal_gen.runtime.realtime.whitebox.cli_room")
    from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import (
        WhiteboxDiagnostic,
        public_error,
    )

    class Process:
        returncode = 1

        async def communicate(self, _):
            return b"", stderr

    async def create(*args, **kwargs):
        return Process()

    monkeypatch.setattr(owner.asyncio, "create_subprocess_exec", create)
    with pytest.raises(WhiteboxDiagnostic) as failure:
        asyncio.run(
            owner.prepare_publication(
                Path("/tmp/sdk/packages/cli/bin/worldkit.mjs"), "source/world", 5
            )
        )
    assert failure.value.code == expected
    result = public_error(failure.value, "owned-trace")
    assert result["stage"] == "startup"
    assert "private" not in result["content"]
    assert result["code"] == expected


@pytest.mark.parametrize("reason", ["timeout", "cancel"])
def test_publication_preparation_deadline_and_cancel_release_owned_process(
    monkeypatch, reason
):
    from importlib import import_module
    from pathlib import Path

    owner = import_module("sglang.multimodal_gen.runtime.realtime.whitebox.cli_room")
    from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import (
        WhiteboxDiagnostic,
    )

    signals = []

    class Process:
        returncode = None
        pid = 12345

        async def communicate(self, _):
            if reason == "cancel":
                raise asyncio.CancelledError()
            raise asyncio.TimeoutError()

        async def wait(self):
            self.returncode = -15

    async def create(*args, **kwargs):
        return Process()

    monkeypatch.setattr(owner.asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(
        owner.os, "killpg", lambda pid, signal: signals.append((pid, signal))
    )
    expected = asyncio.CancelledError if reason == "cancel" else WhiteboxDiagnostic
    with pytest.raises(expected) as failure:
        asyncio.run(
            owner.prepare_publication(
                Path("/tmp/sdk/packages/cli/bin/worldkit.mjs"), "source/world", 5
            )
        )
    if reason == "timeout":
        assert failure.value.code == "STREAM_STARTUP_TIMEOUT"
        assert failure.value.stage == "startup"
    assert signals == [(12345, owner.signal.SIGTERM)]


@pytest.mark.parametrize("media_closes_first", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("reason", ["stopped", "idle-timeout"])
@pytest.mark.parametrize("epoch", [None, 0])
def test_normal_cli_end_finishes_pending_frame_read(
    native, reason, epoch, media_closes_first
):
    async def run():
        end = asyncio.Event()

        async def connection(ws):
            if ws.request.path.startswith("/control"):
                await ws.send(
                    json.dumps(dict(type="session.ready", epoch=0, role="controller"))
                )
                config = (
                    dict(
                        kind="pixels",
                        pixelFormat="rgb24",
                        widthPixels=32,
                        heightPixels=32,
                        rowStrideBytes=96,
                        colorSpace="srgb",
                        fps=24,
                    )
                    if native
                    else dict(codec="vp8", codedWidth=32, codedHeight=32)
                )
                await ws.send(
                    json.dumps(
                        dict(
                            type="media.config",
                            epoch=0,
                            mediaGeneration=1,
                            config=config,
                        )
                    )
                )
                await end.wait()
                if media_closes_first:
                    await asyncio.sleep(0.02)
                message = dict(type="session.ended", reason=reason)
                if epoch is not None:
                    message["epoch"] = epoch
                await ws.send(json.dumps(message))
                await ws.close()
                await ws.wait_closed()
            else:
                if media_closes_first:
                    await end.wait()
                    await ws.close()
                await ws.wait_closed()

        class Renderer:
            async def consume_ui(self, _):
                pass

            async def close(self):
                pass

        async with serve(connection, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            channel = "pixels" if native else "video"
            descriptor = dict(
                sessionId="test",
                epoch=0,
                fps=24,
                accessToken="local",
                streams={
                    key: dict(url=f"ws://127.0.0.1:{port}/{value}")
                    for key, value in [("ui", "control"), (channel, "media")]
                },
            )
            if native:
                descriptor["media"] = dict(kind="pixels")
            room = CLIRoom(
                SimpleNamespace(returncode=0),
                descriptor,
                VideoSpec(32, 32),
                lambda **_: Renderer(),
                "local-lifetime",
            )
            view = await room.join(player_id="player", spec=room.spec)
            iterator = view.frames()
            pending = asyncio.create_task(anext(iterator, None))
            try:
                await asyncio.sleep(0.02)
                end.set()
                assert await asyncio.wait_for(pending, 1) is None
                assert view.end_reason == reason
                assert view.stopped
                await view.control.wait_closed()
                await view.finish()
            finally:
                await iterator.aclose()
                await room.close()
            assert view.control_task.done()
            assert view.renderer is not None

    asyncio.run(run())


@pytest.mark.parametrize(
    "message",
    [
        dict(type="session.ended", reason="producer-disconnected"),
        dict(type="session.ended", reason="stopped", epoch=1),
        dict(type="session.reset", epoch=1),
        dict(type="error", error={"code": "SOURCE_FAILED"}),
        dict(type="session.ready", role="controller"),
    ],
)
def test_cli_end_does_not_hide_reset_failure_or_missing_data_epoch(message):
    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import CLIView

    async def run():
        view = CLIView(
            SimpleNamespace(
                descriptor={"epoch": 0}, spec=VideoSpec(32, 32), prepared_delivery=None
            )
        )

        async def messages():
            yield json.dumps(message)

        view.control = messages()
        with pytest.raises(ValueError):
            await view._control()

    asyncio.run(run())


def test_latest_cli_launch_reuses_one_named_resident_cache_and_lifetime_policy():
    from pathlib import Path

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import cli_arguments

    args = cli_arguments(
        Path("/sdk/packages/cli/bin/worldkit.mjs"),
        "s/w",
        VideoSpec(32, 32),
        {
            "WHITEBOX_CLI_RESOURCE_TRANSPORT": "cdn",
            "WHITEBOX_CLI_CACHE_DIRECTORY": "/cache/legacy",
            "WHITEBOX_CLI_IDLE_TIMEOUT_SECONDS": "75",
            "WHITEBOX_CLI_CONNECTION_TIMEOUT_SECONDS": "40",
            "WHITEBOX_CLI_MAX_DURATION_SECONDS": "1200",
            "WORLD_TEST_API_TOKEN": "private-not-argv",
        },
    )
    for key, value in {
        "--service": "auto",
        "--service-name": "whitebox",
        "--cache-dir": "/cache/legacy/games",
        "--idle-timeout": "75",
        "--connection-timeout": "40",
        "--max-duration": "1200",
        "--timeout": "420",
        "--resource-transport": "cdn",
    }.items():
        assert key in args and args[args.index(key) + 1] == value
    assert "private-not-argv" not in args


@pytest.mark.parametrize(
    "key,value",
    [
        ("WHITEBOX_CLI_IDLE_TIMEOUT_SECONDS", "-1"),
        ("WHITEBOX_CLI_CONNECTION_TIMEOUT_SECONDS", "1.5"),
        ("WHITEBOX_CLI_MAX_DURATION_SECONDS", "2147484"),
        ("WHITEBOX_CLI_SERVICE", "other"),
        ("WHITEBOX_CLI_GAME_CACHE_DIRECTORY", "relative"),
    ],
)
def test_latest_cli_policy_rejects_invalid_settings_before_spawning(key, value):
    from pathlib import Path

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import cli_arguments

    with pytest.raises(ValueError):
        cli_arguments(
            Path("/sdk/packages/cli/bin/worldkit.mjs"),
            "s/w",
            VideoSpec(32, 32),
            {key: value},
        )


def test_local_ui_handoff_checks_exact_manifest_hash_and_rejects_symlinks(tmp_path):
    import hashlib

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import (
        load_local_ui_package,
    )

    raw = b'{"kind":"worldkit-ui-bundle","schemaVersion":1}'
    root = tmp_path / "world-ui"
    root.mkdir()
    (root / "manifest.json").write_bytes(raw)
    descriptor = dict(
        localResources=dict(uiRoot=str(root)),
        uiBundleHash=hashlib.sha256(raw).hexdigest(),
    )
    package = load_local_ui_package(descriptor)
    assert package["root"] == str(root)
    assert package["manifest"]["kind"] == "worldkit-ui-bundle"
    (root / "manifest.json").write_bytes(raw + b" ")
    with pytest.raises(ValueError):
        load_local_ui_package(descriptor)
    (root / "manifest.json").unlink()
    target = tmp_path / "manifest.json"
    target.write_bytes(raw)
    (root / "manifest.json").symlink_to(target)
    with pytest.raises(ValueError):
        load_local_ui_package(descriptor)


def test_official_cli_stage_diagnostics_retain_bounded_startup_metrics():
    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import cli_diagnostics

    async def run():
        async def lines():
            for value in [
                b'worldkit: Preparation: {"elapsedMs":125.5}\n',
                b'worldkit: Startup timing: {"phase":"browser-context","elapsedMs":12.25}\n',
                b'worldkit: Startup timing: {"phase":"unexpected-stage","elapsedMs":999}\n',
                b'worldkit: Startup timing: {"phase":"world-load","elapsedMs":-1}\n',
            ]:
                yield value

        metrics = {}
        assert (
            await cli_diagnostics(
                SimpleNamespace(stderr=lines()), "metrics-test", metrics
            )
            is None
        )
        assert metrics == {"preparationMs": 125.5, "browserContextMs": 12.25}

    asyncio.run(run())


def test_cancel_during_input_renewal_teardown_still_drains_original_cli():
    async def run():
        started, cancelling, gate = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []
        process = SimpleNamespace(returncode=None)
        room = CLIRoom(process, None, VideoSpec(32, 32), None, "test")

        async def renew():
            try:
                started.set()
                await asyncio.Future()
            finally:
                cancelling.set()
                await gate.wait()

        async def close_view():
            calls.append("view")

        async def stop_cli():
            calls.append("cli")
            process.returncode = 0

        room.view = SimpleNamespace(close=close_view)
        room._stop_cli = stop_cli
        room.input_renew_task = asyncio.create_task(renew())
        await started.wait()
        close = asyncio.create_task(room.close())
        try:
            await asyncio.wait_for(cancelling.wait(), 1)
            close.cancel()
            with pytest.raises(asyncio.CancelledError):
                await close
            assert calls == ["view", "cli"]
        finally:
            gate.set()
            close.cancel()
            await asyncio.gather(
                close,
                *([room.input_renew_task] if room.input_renew_task else []),
                return_exceptions=True,
            )

    asyncio.run(run())


@pytest.mark.parametrize("context_fails", [False, True])
def test_room_close_drains_view_cleanup_started_by_failed_renewal(
    monkeypatch, context_fails
):
    from sglang.multimodal_gen.runtime.realtime.whitebox import cli_room

    monkeypatch.setattr(cli_room, "INPUT_SESSION_RENEW_SECONDS", 0)

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        process = SimpleNamespace(returncode=None)
        room = CLIRoom(process, {}, VideoSpec(32, 32), None, "test")
        view = cli_room.CLIView(room)
        room.view, room.player_id = view, "player"

        class Context:
            async def close(self):
                calls.append("context")
                entered.set()
                await release.wait()
                if context_fails:
                    raise ValueError("fixture context cleanup failed")

        def resource(name):
            async def close():
                calls.append(name)

            return SimpleNamespace(close=close)

        async def fail_renewal(*_):
            raise ConnectionError("fixture renewal failed")

        async def stop_cli():
            calls.append("cli")
            process.returncode = 0

        view.model_context = Context()
        view.media, view.control, view.renderer = (
            resource("media"),
            resource("control"),
            resource("renderer"),
        )
        room.public_session, room._stop_cli = fail_renewal, stop_cli
        renewal = asyncio.create_task(room._renew_input_session("https://gateway.test"))
        room.input_renew_task = renewal
        closing = None
        try:
            await asyncio.wait_for(entered.wait(), 1)
            closing = asyncio.create_task(room.close())
            await asyncio.wait_for(asyncio.gather(renewal, return_exceptions=True), 1)
            await asyncio.sleep(0)
            # The room must still await the original resource cleanup, even
            # after cancelling the renewal task that first requested it.
            assert not closing.done()
            release.set()
            if context_fails:
                with pytest.raises(ValueError, match="context cleanup failed"):
                    await closing
            else:
                await closing
                await view.close()
            assert sorted(calls) == sorted(
                ["context", "media", "control", "renderer", "cli"]
            )
            assert process.returncode == 0
        finally:
            release.set()
            await asyncio.gather(
                renewal, *([closing] if closing else []), return_exceptions=True
            )

    asyncio.run(run())
