import asyncio
import re
from copy import deepcopy
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.opening_presentation import (
    OpeningPresentationOffers,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import RenderError
from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_transport import (
    RTCSubscription,
)


def fixture():
    source = dict(sessionId="cli", epoch=2, presentationId="camera", sdkEpoch=3)
    startup = dict(
        op="render.start",
        request_id="initial",
        prompt_version=1,
        model_context=dict(source=source, segmentId="segment", conditionId="initial"),
        model_context_execution=dict(state="applied"),
        model_context_lifecycle=dict(lifecycleId="round", generation=1),
    )
    state = dict(
        source_state="ready",
        display_mode="model",
        display_revision=4,
        segment=dict(
            state="streaming",
            segment_id="segment",
            segment_epoch="model-stream",
            context_segment_id="segment",
            accepted_version=1,
            applied_version=1,
            request_id="initial",
        ),
        operations={"initial": startup},
    )
    session = SimpleNamespace(
        binding_ended=False,
        close_requested=False,
        closing=False,
        current_model_stream="model-stream",
        context_channel=SimpleNamespace(
            segment="segment", first_presented_supported=True, suspended=False
        ),
        journal=SimpleNamespace(snapshot=state),
    )
    frame = dict(
        stream="model",
        media_stream_id="model-stream",
        output_frame_start=2,
        pts_us=2000,
        rendering=dict(
            segment_id="segment", segment_epoch="model-stream", prompt_version=1
        ),
        source=dict(
            cli_frame_header=dict(
                protocolVersion=1,
                sessionId="cli",
                epoch=2,
                mediaGeneration=1,
                outputFrameId=8,
                outputPtsUs=4000,
                uiRevision=6,
                uiCompleteThroughUs=4000,
                widthPixels=640,
                heightPixels=360,
                source=dict(
                    presentationId="camera",
                    sdkEpoch=3,
                    sourceFrameId=8,
                    sourceTimeUs=4000,
                    simulationTick=7,
                    worldRevision=2,
                ),
            )
        ),
    )
    return session, frame


def test_offer_accepts_only_exact_synchronized_model_evidence():
    session, frame = fixture()
    owner = OpeningPresentationOffers()
    offer = owner.offer(session, frame)
    assert offer is not None
    assert owner.resolve(session, offer, 6)["request_id"] == "initial"
    with pytest.raises(RenderError, match="no longer current"):
        owner.resolve(session, offer, 7)
    assert owner.offer(session, dict(frame, stream="whitebox")) is None
    bad = deepcopy(frame)
    bad["source"]["cli_frame_header"]["uiCompleteThroughUs"] = 3
    assert owner.offer(session, bad) is None


def test_new_attachment_display_and_segment_reject_a_previous_offer():
    session, frame = fixture()
    owner = OpeningPresentationOffers()
    offer = owner.offer(session, frame)
    with pytest.raises(RenderError, match="no longer current"):
        OpeningPresentationOffers().resolve(session, offer, 6)
    session.journal.snapshot["display_revision"] += 1
    with pytest.raises(RenderError, match="no longer current"):
        owner.resolve(session, offer, 6)
    session.journal.snapshot["display_mode"] = "whitebox"
    assert owner.offer(session, frame) is None
    session.journal.snapshot["display_mode"] = "model"
    current = owner.offer(session, frame)
    session.journal.snapshot["segment"]["accepted_version"] = 2
    with pytest.raises(RenderError, match="no longer current"):
        owner.resolve(session, current, 6)


def test_recent_offers_are_bounded_and_confirmed_start_cannot_mint_more():
    session, frame = fixture()
    owner = OpeningPresentationOffers()
    old = owner.offer(session, frame)
    for index in range(1, 161):
        current = owner.offer(
            session, dict(frame, output_frame_start=index, pts_us=index * 1000)
        )
    with pytest.raises(RenderError, match="no longer current"):
        owner.resolve(session, old, 6)
    assert owner.resolve(session, current, 6)["request_id"] == "initial"
    session.journal.snapshot["operations"]["initial"]["first_presented"] = True
    assert owner.offer(session, frame) is None


def replace_condition(session, frame, version, *, generation=1):
    state = session.journal.snapshot
    request_id = f"condition-{version}"
    operation = deepcopy(state["operations"]["initial"])
    operation.pop("first_presented", None)
    operation.update(
        op="prompt.replace",
        request_id=request_id,
        prompt_version=version,
        model_context_lifecycle=dict(lifecycleId="round", generation=generation),
    )
    operation["model_context"]["conditionId"] = request_id
    state["operations"][request_id] = operation
    state["segment"].update(
        accepted_version=version,
        applied_version=version,
        model_context_applied_request_id=request_id,
    )
    frame["rendering"]["prompt_version"] = version
    return operation


def test_newest_applied_condition_replaces_opening_candidate():
    session, frame = fixture()
    owner = OpeningPresentationOffers()
    old = owner.offer(session, frame)
    replacement = replace_condition(session, frame, 2)
    with pytest.raises(RenderError, match="no longer current"):
        owner.resolve(session, old, 6)
    current = owner.offer(session, frame)
    assert current and owner.resolve(session, current, 6) == replacement


def test_host_rejected_old_condition_reopens_only_for_current_replacement():
    session, frame = fixture()
    owner = OpeningPresentationOffers()
    initial = session.journal.snapshot["operations"]["initial"]
    initial["first_presented"] = "rejected"
    session.journal.snapshot["model_context_first_presented_request_id"] = "initial"
    assert owner.offer(session, frame) is None
    replacement = replace_condition(session, frame, 2)
    assert owner.offer(session, frame)
    replacement["first_presented"] = "acknowledged"
    session.journal.snapshot["model_context_first_presented_request_id"] = replacement[
        "request_id"
    ]
    replace_condition(session, frame, 3)
    assert owner.offer(session, frame) is None
    replace_condition(session, frame, 4, generation=2)
    assert owner.offer(session, frame)


def test_offer_expires_without_any_new_rtp_frames():
    session, frame = fixture()
    clock = SimpleNamespace(now=10.0)
    owner = OpeningPresentationOffers(clock=lambda: clock.now)
    old = owner.offer(session, frame)
    clock.now += 3.1
    with pytest.raises(RenderError, match="no longer current") as failure:
        owner.resolve(session, old, 6)
    assert failure.value.retryable
    fresh = owner.offer(session, frame)
    assert fresh and owner.resolve(session, fresh, 6)["request_id"] == "initial"


def test_rtp_mapping_adds_offer_only_after_both_capabilities_are_ready():
    async def run():
        session, frame = fixture()
        session.source_epoch = "source"
        session.current_segment = None
        session.context_channel.first_presented_supported = False
        owner = SimpleNamespace(session=session, host=SimpleNamespace(config=None))

        async def close():
            pass

        sub = RTCSubscription(owner, 8, SimpleNamespace(close=close))
        owner.subscription = sub
        messages = []
        sub.offer_control = messages.append
        event = dict(type="frame.meta", track="model", rtp_timestamp=50, frame=frame)
        try:
            sub.native_event(event)
            assert "first_presentation_offer" not in messages[-1]["frame"]
            sub.enable_first_presentation()
            sub.native_event(event)
            assert "first_presentation_offer" not in messages[-1]["frame"]
            session.context_channel.first_presented_supported = True
            sub.native_event(event)
            assert messages[-1]["rtp_timestamp"] == 50
            offer = messages[-1]["frame"]["first_presentation_offer"]
            assert re.fullmatch(r"[0-9a-f]{32}", offer)
            assert "first_presentation_offer" not in frame
            command = dict(offer_id=offer, connection_generation=8, ui_revision=6)
            state = session.journal.snapshot
            assert (
                sub.resolve_first_presentation_offer(command, state)["request_id"]
                == "initial"
            )
            owner.subscription = None
            with pytest.raises(RenderError, match="no longer current"):
                sub.resolve_first_presentation_offer(command, state)
        finally:
            sub.stop("DISCONNECTED")
            await sub.wait_closed()

    asyncio.run(run())


def test_confirm_accepts_the_player_control_envelope_without_version():
    from sglang.multimodal_gen.runtime.realtime.whitebox.opening_presentation import (
        CONFIRM,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def run():
        seen = []

        async def confirm(command, resolve_offer):
            assert resolve_offer is source.resolve_first_presentation_offer
            seen.append(command)
            return dict(request_id=command["request_id"], state="accepted")

        async def reply(result):
            seen.append(result)

        source = object.__new__(SourceSession)
        source.journal = SimpleNamespace(confirm_opening_presentation=confirm)
        source.resolve_first_presentation_offer = lambda *_: None
        source.source_conditions = None
        source.operation_message = reply
        # WhiteboxClient.control supplies its v2 discriminator via negotiation.
        command = dict(
            type="command",
            op=CONFIRM,
            request_id="request",
            source_session_id="source",
            source_epoch="epoch",
            offer_id="a" * 32,
            connection_generation=1,
            ui_revision=6,
        )
        await source.confirm_opening_presentation(command)
        assert seen == [command, {"request_id": "request", "state": "accepted"}]
        with pytest.raises(RenderError) as failure:
            await source.confirm_opening_presentation(dict(command, version=1))
        assert failure.value.code == "INVALID_COMMAND"

    asyncio.run(run())


def test_opted_in_source_rejects_an_old_player_before_model_parent_admission():
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def run():
        source = object.__new__(SourceSession)
        source.closed = source.binding_ended = False
        source.first_presented_required = True
        source.context_channel = None
        source.first_presented_player_ready = lambda: False

        def reserve(_):
            pytest.fail("An unsupported Player cannot reserve model compute")

        source.start_model_binding = reserve
        inference = object.__new__(SourceInference)
        inference.source = source
        inference.manual = True
        inference.frozen = {}
        with pytest.raises(RenderError) as failure:
            await inference.enable({"inference": {"revision": 1}})
        assert failure.value.code == "MODEL_PRESENTATION_UNSUPPORTED"

    asyncio.run(run())


def test_descriptor_can_require_only_the_known_opening_protocol():
    from sglang.multimodal_gen.runtime.realtime.whitebox.opening_presentation import (
        PROTOCOL,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_conditions import (
        validate_capability,
    )

    old = dict(
        protocol="worldkit-model-context-v1",
        required=True,
        configHash="a" * 64,
        gameBuildHash="b" * 64,
        compilerVersion=1,
        visible=False,
    )
    validate_capability(old)
    validate_capability(dict(old, firstPresentedProtocol=PROTOCOL))
    with pytest.raises(ValueError):
        validate_capability(dict(old, firstPresentedProtocol="unknown"))
