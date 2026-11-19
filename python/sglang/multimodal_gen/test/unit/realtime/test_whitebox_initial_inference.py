# SPDX-License-Identifier: Apache-2.0
"""Frozen launch intent reaches the original journal before World subscription."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from test_whitebox_render_contract import command, setup

from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
    SourceInference,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import SourceSession
from sglang.multimodal_gen.runtime.realtime.whitebox.startup import (
    model_transport_ready,
)


async def owner(enabled=True, publication=True):
    journal = await setup()
    await journal.initialize_inference(manual=True)
    source = SourceSession.__new__(SourceSession)
    source.journal = journal
    source.session_id, source.source_epoch = "source", "epoch"
    source.initial_inference_enabled = enabled
    source.expected_publication = {"worldId": "world"} if publication else None
    source.first_presented_required = False
    source.config = NS(startup_seconds=1)
    source.closed = source.closing = source.binding_ended = False
    source.context_startup_ids = None
    source.context_channel = None
    source.inference = SourceInference(
        source,
        dict(
            source_operation="create.manual",
            inference_control="manual",
            whitebox={"world_id": "world"},
            prompt="snow",
            seed=42,
        ),
    )
    return source


@pytest.mark.parametrize("enabled,publication", [(False, True), (True, False)])
def test_no_implicit_enable_without_explicit_bound_intent(enabled, publication):
    async def run():
        source = await owner(enabled, publication)
        assert await source.inference.accept_initial() is None
        assert not source.journal.snapshot["inference"]["enabled"]

    asyncio.run(run())


def test_initial_intent_opens_world_subscription_without_pause_or_second_switch():
    async def run():
        source = await owner()
        inference = source.inference
        operation = await inference.accept_initial()
        assert operation["state"] == "accepted"
        assert source.journal.snapshot["inference"] == dict(
            enabled=True, revision=1, state="starting"
        )
        assert await inference.accept_initial() is None
        channel = NS(probe_subscription_pause=AsyncMock(), switch_segment=AsyncMock())
        channel.bound = asyncio.Event()
        channel.bound.set()
        channel.check = lambda: None
        channel.suspended = False
        calls = []

        async def bind(**kwargs):
            calls.append(kwargs)
            return channel

        source.engine = NS(bind_model_context=bind, model_context_capability={})
        inference.open_prewarm = AsyncMock()
        await source.bind_source_conditions()
        assert len(calls) == 1 and "start_suspended" not in calls[0]
        assert calls[0]["require_subscription_pause"] is True
        segment = source.journal.snapshot["model_context_segment_ids"]["segment_id"]
        ready = asyncio.Event()
        ready.set()
        source.current_segment = NS(
            segment_id=segment, model_ready=ready, closed_event=asyncio.Event()
        )
        source.model_binding = NS(
            closed=False, lease=NS(resource_state="owned"), wait_ready=AsyncMock()
        )
        await inference.enable(operation)
        channel.probe_subscription_pause.assert_not_awaited()
        channel.switch_segment.assert_not_awaited()
        inference.open_prewarm.assert_awaited_once_with(segment, 1)

    asyncio.run(run())


def test_explicit_disable_while_waiting_for_media_wins_even_at_revision_zero():
    async def run():
        source = await owner()
        source.first_presented_required = True
        source.first_presented_player_ready = lambda: True
        task = asyncio.create_task(source.inference.accept_initial())
        await asyncio.sleep(0)
        assert not task.done()
        await source.journal.configure_inference(
            command(
                "model.inference.configure",
                "user-off",
                enabled=False,
                expected_inference_revision=0,
            )
        )
        model_transport_ready(source)
        assert await task is None
        assert source.journal.snapshot["inference"]["enabled"] is False

    asyncio.run(run())


def test_cancel_waiting_launch_does_not_create_inference_intent():
    async def run():
        source = await owner()
        source.first_presented_required = True
        task = asyncio.create_task(source.inference.accept_initial())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        model_transport_ready(source)
        assert not source.journal.snapshot["inference"]["enabled"]
        assert source.inference.initial_request_id is None

    asyncio.run(run())


@pytest.mark.parametrize("stale", [False, True])
def test_detach_invalidates_only_the_current_media_readiness(stale):
    from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_routes import (
        RTCRenderPipeline,
    )

    async def run():
        source = await owner()
        model_transport_ready(source)
        sub = NS(stop=lambda _: None)
        pipeline = RTCRenderPipeline.__new__(RTCRenderPipeline)
        pipeline.session = source
        pipeline.subscription = object() if stale else sub
        pipeline.retention = None
        pipeline.closed = True
        pipeline.terminal = None
        pipeline.detach(sub)
        assert source.inference.media_ready.is_set() is stale
        assert source._model_transport_ready is stale

    asyncio.run(run())


def test_initial_dispatch_cannot_cancel_a_later_explicit_disable():
    async def run():
        source = await owner()
        initial = await source.inference.accept_initial()
        source.inference.dispatch = lambda _: pytest.fail(
            "obsolete intent was dispatched"
        )
        await source.journal.configure_inference(
            command(
                "model.inference.configure",
                "later-off",
                enabled=False,
                expected_inference_revision=1,
            )
        )
        await source.inference.dispatch_initial(initial)
        assert not source.journal.snapshot["inference"]["enabled"]

    asyncio.run(run())


def test_reset_during_initial_subscription_fences_the_original_enable():
    async def run():
        source = await owner()
        source.native_source_generation = 0
        initial = await source.inference.accept_initial()
        source.native_source_generation = 1
        source.operation_message = AsyncMock()
        source.inference.dispatch = lambda _: pytest.fail("stale source was dispatched")
        await source.inference.dispatch_initial(initial)
        assert source.journal.snapshot["inference"]["state"] == "failed"
        source.operation_message.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize("supported", [False, True])
def test_initial_active_subscription_still_requires_the_native_pause_capability(
    supported,
):
    import json

    from test_whitebox_source_conditions import condition

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderError,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
        CLIConditionChannel,
    )

    async def run():
        c = condition()
        sent = []

        async def send(wire):
            sent.append(json.loads(wire))

        capability = dict(
            protocol=c["protocol"],
            required=True,
            configHash=c["configHash"],
            gameBuildHash=c["source"]["gameBuildHash"],
            compilerVersion=1,
            visible=False,
        )
        view = NS(
            descriptor={"epoch": c["source"]["epoch"], "modelContext": capability},
            closed=False,
            stopped=False,
            client_id=c["audience"]["consumerId"],
            transport_generation=0,
            control=NS(send=send),
        )
        channel = CLIConditionChannel(view)
        try:
            await channel.bind(
                segment_id=c["segmentId"],
                create_adapter=lambda *_: pytest.fail(
                    "no condition should be admitted by the capability receipt"
                ),
                require_subscription_pause=True,
            )
            assert sent == [
                dict(
                    type="model.context.subscribe",
                    protocol=c["protocol"],
                    epoch=c["source"]["epoch"],
                    segmentId=c["segmentId"],
                    requireSubscriptionPause=True,
                )
            ]
            receipt = dict(
                type="model.context.subscribed",
                protocol=c["protocol"],
                epoch=c["source"]["epoch"],
                segmentId=c["segmentId"],
                consumer=c["audience"],
                sampleControlFloor=1,
                paused=False,
            )
            if supported:
                await channel.consume(dict(receipt, subscriptionPause=True))
                assert channel.bound.is_set() and not channel.suspended
            else:
                with pytest.raises(RenderError, match="pause"):
                    await channel.consume(receipt)
                assert channel.consumer is None
        finally:
            await channel.close()

    asyncio.run(run())


def test_internal_recovery_without_request_id_keeps_ordinary_probe_and_switch():
    async def run():
        source = await owner()
        operation = await source.journal.configure_inference(
            command(
                "model.inference.configure",
                "explicit-on",
                enabled=True,
                expected_inference_revision=0,
            )
        )
        ids = await source.journal.reserve_model_context_segment()
        channel = NS(probe_subscription_pause=AsyncMock(), switch_segment=AsyncMock())
        source.context_channel = channel
        source.model_binding = NS(
            closed=False, lease=NS(resource_state="owned"), wait_ready=AsyncMock()
        )
        ready = asyncio.Event()
        ready.set()
        source.current_segment = NS(
            segment_id=ids["segment_id"],
            model_ready=ready,
            closed_event=asyncio.Event(),
        )
        source.inference.open_prewarm = AsyncMock()
        await source.inference.enable({**operation, "request_id": None})
        channel.probe_subscription_pause.assert_awaited_once()
        channel.switch_segment.assert_awaited_once_with(ids["segment_id"])

    asyncio.run(run())


def test_initial_unsupported_presentation_fails_only_inference_not_source():
    async def run():
        source = await owner()
        source.first_presented_required = True
        source.first_presented_player_ready = lambda: False

        async def release_compute():
            await source.journal.update(
                lambda state: state["inference"].update(enabled=False)
            )

        source.inference.disable = AsyncMock(side_effect=release_compute)
        model_transport_ready(source)
        result = await source.inference.accept_initial()
        assert result["state"] == "failed"
        assert result["code"] == "MODEL_PRESENTATION_UNSUPPORTED"
        assert source.journal.snapshot["source_state"] == "ready"
        assert not source.journal.snapshot["inference"]["enabled"]
        source.inference.disable.assert_awaited_once()
        source.inference.dispatch = lambda _: pytest.fail(
            "failed initial intent dispatched"
        )
        await source.inference.dispatch_initial(result)

    asyncio.run(run())


def test_initial_enable_does_not_outlive_failed_subscription_handshake():
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderError,
    )

    async def run():
        source = await owner()
        operation = await source.inference.accept_initial()
        bound = asyncio.Event()
        bound.set()
        source.context_channel = NS(
            bound=bound,
            check=lambda: None,
            suspended=True,
            subscription_probe_error="MODEL_CONTEXT_SUBSCRIPTION_PAUSE_UNSUPPORTED",
        )
        source.model_binding = NS(
            closed=False, lease=NS(resource_state="owned"), wait_ready=AsyncMock()
        )
        with pytest.raises(RenderError, match="pause"):
            await source.inference.enable(operation)

    asyncio.run(run())
