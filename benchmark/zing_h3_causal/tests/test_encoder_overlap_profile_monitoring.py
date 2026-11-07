# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from continuous_source_client import websocket_options
from encoder_overlap_profile_server import configure_monitoring
from run_encoder_overlap_matrix import endpoint_phase_groups, server_command

from sglang.multimodal_gen.runtime.realtime.source_transport import (
    DEFAULT_MESSAGE_BYTES,
)


def test_profile_uses_separate_warmed_server_without_changing_five_samples():
    assert endpoint_phase_groups() == [
        (False, ["warmup", *[f"measured-{i}" for i in range(5)]]),
        (True, ["profile-warmup", "profile"]),
    ]
    assert endpoint_phase_groups(capture=True) == [(False, ["capture"])]
    assert endpoint_phase_groups(soak=True) == [(False, ["warmup", "measured-0"])]


def test_only_profile_server_changes_monitoring_and_keeps_model_arguments():
    regular = server_command("model", "config", 4)
    profile = server_command("model", "config", 4, profile_monitoring=True)
    assert regular[1:3] == ["-m", "sglang.multimodal_gen.runtime.launch_server"]
    assert "--realtime-session-max-lifetime-s" not in regular
    assert profile[1].endswith("encoder_overlap_profile_server.py")
    for flag in (
        "--realtime-session-max-lifetime-s",
        "--realtime-session-idle-timeout-s",
    ):
        assert profile[profile.index(flag) + 1] == "3600"
    assert (
        regular[regular.index("--model-path") :]
        == profile[profile.index("--model-path") :]
    )


def test_uvicorn_override_retains_protocol_and_does_not_mutate_package():
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return "started"

    package = SimpleNamespace(run=run)
    launch = SimpleNamespace(uvicorn=package)
    configure_monitoring(launch)
    assert (
        launch.uvicorn.run("app", ws="websockets-sansio", ws_max_queue=1) == "started"
    )
    assert calls == [
        (("app",), dict(ws="websockets-sansio", ws_max_queue=1, ws_ping_timeout=3600))
    ]
    assert package.run is run


def test_client_profile_ping_budget_is_finite_and_regular_default_unchanged():
    ordinary = websocket_options(profile=False, timeout=3600)
    profile = websocket_options(profile=True, timeout=3600)
    assert ordinary.pop("ping_timeout") == 20
    assert profile.pop("ping_timeout") == 3600
    assert ordinary == profile


def test_stage_timing_has_bounded_diagnostic_keepalive_budget():
    assert (
        websocket_options(profile=False, stage_timing=True, timeout=123)["ping_timeout"]
        == 123
    )
    assert (
        websocket_options(profile=False, stage_timing=False, timeout=123)[
            "ping_timeout"
        ]
        == 20
    )


def test_combined_completion_stops_heartbeats_while_split_waits_for_input_completion():
    import asyncio

    from continuous_source_client import keep_source_heartbeat

    async def scenario(split, completed):
        state = dict(ended=True, completed=completed, input_completed=False)
        messages = []

        async def send(message):
            messages.append(message)
            state["input_completed"] = True

        await keep_source_heartbeat(send, state, split_output=split, interval=0)
        assert len(messages) == int(split or not completed)

    asyncio.run(scenario(False, True))
    asyncio.run(scenario(False, False))
    asyncio.run(scenario(True, True))


def test_terminal_profile_drain_joins_children_on_failure_and_cancellation():
    import asyncio
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.stage_profile_delivery import (
        collect_profile_with_input_drain,
    )

    async def scenario(mode):
        baseline = asyncio.all_tasks()
        entered = asyncio.Event()
        collector_closed = asyncio.Event()
        receiver_closed = asyncio.Event()

        async def receive():
            try:
                if mode == "disconnect":
                    raise EOFError("Disconnected")
                await asyncio.Event().wait()
            finally:
                receiver_closed.set()

        async def collect():
            try:
                entered.set()
                if mode == "collector_error":
                    raise ValueError("Collection failed")
                if mode == "success":
                    return "profile"
                await asyncio.Event().wait()
            finally:
                collector_closed.set()

        task = asyncio.create_task(
            collect_profile_with_input_drain(
                SimpleNamespace(receive_bytes=receive), collect()
            )
        )
        await entered.wait()
        if mode == "cancel":
            task.cancel()
        try:
            result = await task
            assert mode == "success" and result == "profile"
        except (ValueError, EOFError, asyncio.CancelledError) as error:
            expected = {
                "collector_error": ValueError,
                "disconnect": EOFError,
                "cancel": asyncio.CancelledError,
            }
            assert mode in expected and isinstance(error, expected[mode])
        assert collector_closed.is_set() and receiver_closed.is_set()
        assert asyncio.all_tasks() == baseline

    for mode in ["success", "collector_error", "disconnect", "cancel"]:
        asyncio.run(scenario(mode))


def test_client_profile_options_preserve_negotiated_transport_budget():
    for profile in (False, True):
        options = websocket_options(
            profile=profile, timeout=3600, message_budget=48 << 20
        )
        assert options["max_size"] == 48 << 20
        assert options["max_queue"] == 1
        assert options["compression"] is None
    assert (
        websocket_options(profile=False, timeout=600)["max_size"]
        == DEFAULT_MESSAGE_BYTES
    )


def test_terminal_close_drains_late_input_after_application_close():
    import asyncio
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.stage_profile_delivery import (
        close_with_input_drain,
    )

    async def scenario():
        queue = asyncio.Queue(maxsize=1)
        queue.put_nowait(dict(type="websocket.receive", bytes=b"late heartbeat"))
        closing, handshake = asyncio.Event(), asyncio.Event()

        async def receive():
            await closing.wait()
            message = await queue.get()
            handshake.set()
            return message

        async def close(*, code, reason):
            assert (code, reason) == (1000, "generation complete")
            closing.set()
            # A bounded application queue must be consumed before the transport
            # can process the peer's CLOSE response.
            await handshake.wait()

        await asyncio.wait_for(
            close_with_input_drain(
                SimpleNamespace(receive=receive, close=close),
                code=1000,
                reason="generation complete",
            ),
            timeout=1,
        )
        assert queue.empty() and handshake.is_set()

    asyncio.run(scenario())


def test_terminal_close_joins_receiver_on_close_error_and_cancellation():
    import asyncio
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.stage_profile_delivery import (
        close_with_input_drain,
    )

    async def scenario(cancel):
        baseline = asyncio.all_tasks()
        entered, received = asyncio.Event(), asyncio.Event()
        receiver_closed = asyncio.Event()

        async def receive():
            received.set()
            try:
                await asyncio.Event().wait()
            finally:
                receiver_closed.set()

        async def close(**kwargs):
            entered.set()
            await received.wait()
            if not cancel:
                raise OSError("Close failed")
            await asyncio.Event().wait()

        task = asyncio.create_task(
            close_with_input_drain(
                SimpleNamespace(receive=receive, close=close), code=1000, reason="done"
            )
        )
        await entered.wait()
        await received.wait()
        if cancel:
            task.cancel()
        try:
            await task
            raise AssertionError("Expected close failure or cancellation")
        except (OSError, asyncio.CancelledError) as error:
            assert isinstance(error, asyncio.CancelledError if cancel else OSError)
        assert receiver_closed.is_set() and asyncio.all_tasks() == baseline

    asyncio.run(scenario(False))
    asyncio.run(scenario(True))
