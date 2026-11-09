import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import (
    capture_fps,
    cli_arguments,
    cli_diagnostics,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.diagnostics import (
    WhiteboxDiagnostic,
    cli_fps_override,
    public_error,
)


def test_capture_override_is_per_source_and_preserves_model_clock():
    spec = VideoSpec()
    env = {"WHITEBOX_CLI_FPS": "35"}
    assert capture_fps(spec, env) == 35
    for fps in [1, 24, 35, 48, 60]:
        args = cli_arguments(
            Path("/sdk/packages/cli/dist/main.js"), "world", spec, env, cli_fps=fps
        )
        assert args[args.index("--fps") + 1] == str(fps)
        assert spec.fps == 24 and spec.pts(24) == 1_000_000
        assert env["WHITEBOX_CLI_FPS"] == "35"
    for bad in [True, False, 0, 61, 24.5, "35", [], {}]:
        with pytest.raises(WhiteboxDiagnostic) as e:
            cli_fps_override(bad)
        assert e.value.code == "INVALID_CLI_FPS"


def test_startup_error_evidence_does_not_forward_raw_exception_or_urls():
    private = ValueError(
        "Bearer PRIVATE https://host/path?token=secret /run/private.key"
    )
    msg = public_error(private, "trace-123")
    assert (
        msg["code"] == "ROOM_SESSION_LOST"
        and "PRIVATE" not in str(msg)
        and "https:" not in str(msg)
    )
    msg = public_error(
        WhiteboxDiagnostic("STREAM_DOWNLOAD_TIMEOUT", "download"), "trace-123"
    )
    assert (
        msg["code"] == "STREAM_DOWNLOAD_TIMEOUT"
        and msg["stage"] == "download"
        and msg["trace_id"] == "trace-123"
    )
    assert public_error(private, "https://secret?token=private")["trace_id"] is None


def test_cli_stderr_preserves_only_known_public_startup_diagnostic():
    async def run():
        async def stderr():
            yield b"raw stack: secret\n"
            yield b'{"kind":"worldkit-cli-error","stage":"download","error":{"code":"WTP_PLAYABLE_NOT_READY"}}\n'

        error = await cli_diagnostics(SimpleNamespace(stderr=stderr()), "trace")
        assert error.code == "WTP_PLAYABLE_NOT_READY" and error.stage == "download"

    asyncio.run(run())


def test_runtime_passes_override_only_to_cli_and_releases_failed_start(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
        AsyncArchiveManager,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
        MemoryRoomStore,
        RoomDirectory,
        new_id,
    )

    async def run():
        directory = RoomDirectory(MemoryRoomStore(), {"room": "http://room"})
        archives = AsyncArchiveManager()
        calls = []

        async def driver(**kwargs):
            calls.append(kwargs)
            raise WhiteboxDiagnostic("STREAM_DOWNLOAD_TIMEOUT", "download")

        runtime = RoomRuntime(
            directory=directory,
            owner_id="room",
            driver_factory=driver,
            spool=tmp_path,
            spec=VideoSpec(),
            archives=archives,
        )
        try:
            for override in [48, None]:
                game = new_id()
                await directory.resolve(game, "world", create=True)
                body = dict(
                    game_id=game,
                    player_id=new_id(),
                    stream_id=new_id(),
                    session_id=new_id(),
                    world_id="world",
                    user_id="test",
                    trace_id="test",
                    spec=dict(width=832, height=480, fps=24),
                )
                if override is not None:
                    body["cli_fps"] = override
                with pytest.raises(WhiteboxDiagnostic):
                    await runtime.join(body)
                assert (
                    calls[-1].get("cli_fps") == override and calls[-1]["spec"].fps == 24
                )
                assert not runtime.rooms and not runtime.players
        finally:
            await runtime.close()
            await archives.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "outcome,level",
    [("failed", "ERROR"), ("cancelled", "INFO"), ("not-completed", "WARNING")],
)
def test_cli_forwards_only_bounded_structured_blocking_fields(caplog, outcome, level):
    import json

    value = dict(
        schemaVersion=1,
        event="worldkit.production.blocked",
        stage="model-context.receipt",
        outcome=outcome,
        level="error",
        sessionId="source-session",
        conditionId="condition",
        promptVersion=7,
        prompt="PRIVATE_PROMPT",
        token="PRIVATE_TOKEN",
        error=dict(
            code="MODEL_CONTEXT_RECEIPT_FAILED",
            message="PRIVATE_MESSAGE",
            cause=dict(code="MODEL_EXECUTION_FAILED", private="PRIVATE_CAUSE"),
        ),
    )

    async def run():
        async def stderr():
            yield b"PRIVATE_RAW https://private/?token=PRIVATE_TOKEN\n"
            yield (json.dumps(value) + "\n").encode()

        return await cli_diagnostics(
            SimpleNamespace(stderr=stderr()), "operation-trace"
        )

    with caplog.at_level("INFO"):
        assert asyncio.run(run()) is None
    rows = [
        row
        for row in caplog.records
        if "worldkit.production.blocked" in row.getMessage()
    ]
    assert len(rows) == 1 and rows[0].levelname == level
    event = json.loads(rows[0].getMessage())
    assert (
        event["trace_id"] == "operation-trace"
        and event["sessionId"] == "source-session"
    )
    assert event["conditionId"] == "condition" and event["promptVersion"] == 7
    assert event["error"]["cause"]["code"] == "MODEL_EXECUTION_FAILED"
    assert "PRIVATE_" not in rows[0].getMessage()


def test_cli_collector_descriptor_is_inherited_and_closed_after_spawn(capfd):
    import os
    import subprocess
    import sys

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_logging import cli_log_sink

    original = dict(os.environ)
    with cli_log_sink(original) as (environment, descriptors):
        descriptor = descriptors[0]
        assert environment["WORLDKIT_PRODUCTION_LOG_FD"] == str(descriptor)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os;os.write(int(os.environ['WORLDKIT_PRODUCTION_LOG_FD']),b'collector-probe\\n');print('protocol-output')",
            ],
            env=environment,
            pass_fds=descriptors,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "protocol-output" and result.stderr == ""
    assert "collector-probe" in capfd.readouterr().err
    with pytest.raises(OSError):
        os.fstat(descriptor)
    assert original == dict(os.environ)
