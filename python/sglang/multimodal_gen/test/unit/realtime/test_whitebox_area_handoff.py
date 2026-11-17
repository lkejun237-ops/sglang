"""AREA ownership over real existing native/Room routes; cloud/model transport mocked."""

import copy
import json

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.area_contract import (
    read_area_frame,
    read_area_identity,
    read_area_receipt,
)


def identity():
    return dict(
        operationId="native-operation",
        areaRevision=1,
        runtimeSessionId="world-runtime",
        worldEpoch=3,
        areaId="indoor",
        selectedGroups=["npc"],
        selectionKey='{"areaId":"indoor","groupIds":["indoor","npc"],"selectedGroups":["npc"]}',
        publication=dict(
            sourceId="source",
            worldId="world",
            publicationId="publication",
            manifestSha256="a" * 64,
            loadingSha256="b" * 64,
            sourceManifestSha256="c" * 64,
            renditionInputSha256="d" * 64,
        ),
    )


def area_frame():
    source = dict(
        presentationId="presentation",
        epoch=4,
        sourceFrameId=9,
        simulationTick=18,
        worldRevision=2,
        capturedAtMilliseconds=123.456,
        widthPixels=64,
        heightPixels=36,
    )
    frame = dict(
        protocolVersion=1,
        sessionId="native-session",
        epoch=0,
        mediaGeneration=1,
        outputFrameId=9,
        outputPtsUs=375000,
        uiRevision=7,
        uiCompleteThroughUs=375000,
        widthPixels=64,
        heightPixels=36,
        source=dict(
            presentationId="presentation",
            sdkEpoch=4,
            sourceFrameId=9,
            sourceTimeUs=375000,
            simulationTick=18,
            worldRevision=2,
        ),
    )
    return dict(
        type="area.frame", epoch=0, identity=identity(), frame=frame, source=source
    )


def test_native_attestation_and_receipt_are_separate_nonce_free_contracts():
    attestation = area_frame()
    assert read_area_frame(attestation) == attestation
    receipt = dict(
        identity=identity(),
        frame=attestation["frame"],
        uiRevision=7,
        uiStatus="synchronized",
    )
    assert read_area_receipt(receipt) == receipt
    for extra in (
        dict(nonce="native-secret"),
        dict(challenge="room-only"),
        dict(model="mapping"),
    ):
        with pytest.raises(ValueError):
            read_area_receipt({**receipt, **extra})
    receipt["frame"]["source"]["worldRevision"] = 99
    assert read_area_frame(area_frame())["source"]["worldRevision"] == 2


@pytest.mark.parametrize(
    "field",
    ["worldEpoch", "areaRevision", "selectedGroups", "publication", "selectionKey"],
)
def test_area_identity_requires_the_exact_closed_native_source_selection(field):
    value = identity()
    value[field] = -1 if field in {"worldEpoch", "areaRevision"} else None
    with pytest.raises(ValueError):
        read_area_identity(value)


@pytest.mark.parametrize(
    "edit",
    [
        lambda v: v.update(nonce="private"),
        lambda v: v["source"].update(sourceFrameId=8),
        lambda v: v["frame"]["source"].update(worldRevision=1),
        lambda v: v["frame"].update(area="must-not-expand-frame"),
        lambda v: v["source"].update(capturedAtMilliseconds=float("inf")),
    ],
)
def test_area_frame_refuses_fabricated_or_mismatched_source_fields(edit):
    value = area_frame()
    edit(value)
    with pytest.raises(ValueError):
        read_area_frame(value)


def test_native_area_control_uses_the_existing_cli_socket_through_local_and_authenticated_remote_runtime():
    import asyncio
    from types import SimpleNamespace

    import httpx
    from websockets.asyncio.server import serve

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import CLIView
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_client import RemoteEngine
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        LocalRoomEngine,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import (
        RoomRuntime,
        create_app,
    )

    async def scenario():
        sent = []
        transition = dict(
            type="area.transition",
            epoch=0,
            revision=1,
            status=dict(
                phase="awaiting-presented",
                inputHeld=True,
                draining=False,
                recoverable=True,
                residentAreas=1,
                transition=identity(),
            ),
            runtime=dict(runtimeHash="e" * 64, runtimeManifestSha256="f" * 64),
            ack=dict(nonce="native-private"),
        )

        async def connection(ws):
            if ws.request.path.startswith("/control"):
                for message in (
                    dict(type="session.ready", epoch=0, role="controller"),
                    transition,
                    area_frame(),
                ):
                    await ws.send(json.dumps(message))
                async for wire in ws:
                    command = json.loads(wire)
                    if command["type"] == "area.presented":
                        sent.append(command)
                        await ws.send(
                            json.dumps(
                                dict(
                                    type="area.presentation-result",
                                    epoch=0,
                                    requestId=command["requestId"],
                                    operationId=command["identity"]["operationId"],
                                    status="confirmed",
                                )
                            )
                        )
            else:
                await ws.wait_closed()

        class Renderer:
            async def consume_ui(self, _):
                pass

            async def close(self):
                pass

        async with serve(connection, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            descriptor = dict(
                sessionId="native-session",
                epoch=0,
                accessToken="local",
                area=dict(protocol="worldkit-area-presented-v1"),
                streams={
                    key: dict(url=f"ws://127.0.0.1:{port}/{channel}")
                    for key, channel in (("ui", "control"), ("pixels", "media"))
                },
                media=dict(kind="pixels"),
            )
            view = CLIView(
                SimpleNamespace(
                    descriptor=descriptor,
                    spec=VideoSpec(64, 36),
                    prepared_delivery=None,
                    renderer_factory=lambda **_: Renderer(),
                )
            )
            await view.start()
            try:
                async with asyncio.timeout(2):
                    while await view.area_status() is None:
                        await asyncio.sleep(0.001)
                assert "native-private" not in json.dumps(await view.area_status())
                runtime = object.__new__(RoomRuntime)
                runtime.spec = VideoSpec(64, 36)
                runtime.players = {
                    "a" * 32: dict(
                        game_id="b" * 32, engine=view, identity={}, spec=runtime.spec
                    )
                }

                class Log:
                    async def event(self, *args):
                        return {}

                runtime.rooms = {"b" * 32: dict(input_lock=asyncio.Lock(), log=Log())}
                attachment = dict(
                    stream_id="a" * 32, trace_id="local", engine={}, ui={}
                )
                local = LocalRoomEngine(runtime, attachment)
                assert (await local.area_status())["status"]["transition"] == identity()
                app = create_app(runtime, "local-token")
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://room"
                ) as client:
                    remote = RemoteEngine(
                        client,
                        "http://room",
                        "local-token",
                        attachment,
                        owned_client=False,
                    )
                    receipt = dict(
                        identity=identity(),
                        frame=area_frame()["frame"],
                        uiRevision=7,
                        uiStatus="synchronized",
                    )
                    assert (
                        await remote.area_frame(area_frame()["frame"]) == area_frame()
                    )
                    # Repeated metadata lookup does not consume the actual source join.
                    assert (
                        await remote.area_frame(area_frame()["frame"]) == area_frame()
                    )
                    result = await remote.area_presented(
                        dict(request_id="room-request", receipt=receipt)
                    )
                    assert result["status"] == "confirmed"
                checked = []
                result = await local.area_presented(
                    dict(request_id="guarded-request", receipt=receipt),
                    before_send=lambda: checked.append(True),
                )
                assert result["status"] == "confirmed" and checked == [True]

                def detached():
                    raise ValueError("AREA_PRESENTATION_STALE")

                with pytest.raises(ValueError, match="AREA_PRESENTATION_STALE"):
                    await local.area_presented(
                        dict(request_id="detached-request", receipt=receipt),
                        before_send=detached,
                    )
                assert len(sent) == 2 and all(
                    row["nonce"] == "native-private" for row in sent
                )
                assert [row["requestId"] for row in sent] == [
                    "room-request",
                    "guarded-request",
                ]
                assert view.area_pending == {}
            finally:
                await view.close()

    asyncio.run(scenario())


def test_explicit_area_source_selection_round_trips_to_cli_without_complete_cache_or_archive_fallback():
    from pathlib import Path

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import cli_arguments
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec

    env = {"WHITEBOX_CLI_SERVICE": "off", "WHITEBOX_CLI_RESOURCE_TRANSPORT": "cdn"}
    normal = cli_arguments(
        Path("/sdk/packages/cli/bin/worldkit.mjs"), "s/w", VideoSpec(64, 36), env
    )
    assert "--loading" not in normal and "--area-group" not in normal
    selected = cli_arguments(
        Path("/sdk/packages/cli/bin/worldkit.mjs"),
        "s/w",
        VideoSpec(64, 36),
        env,
        source_options=dict(loading="areas", initial_groups=["npc"]),
    )
    assert selected[-4:] == ["--loading", "areas", "--area-group", "npc"]
    for conflicting in (
        {**env, "WHITEBOX_CLI_CACHE_UPDATE": "async"},
        {**env, "WHITEBOX_CLI_RESOURCE_TRANSPORT": "archive"},
    ):
        with pytest.raises(ValueError):
            cli_arguments(
                Path("/sdk/packages/cli/bin/worldkit.mjs"),
                "s/w",
                VideoSpec(64, 36),
                conflicting,
                source_options=dict(loading="areas"),
            )
    with pytest.raises(ValueError):
        cli_arguments(
            Path("/sdk/packages/cli/bin/worldkit.mjs"),
            "s/w",
            VideoSpec(64, 36),
            env,
            delivery="/archive.tar.gz",
            source_options=dict(loading="areas"),
        )


def test_area_binding_is_frozen_in_the_existing_segment_and_prompt_journal():
    import asyncio

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderJournal,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import MemoryRoomStore

    async def run():
        journal = RenderJournal(
            MemoryRoomStore(),
            user_id="u",
            source_session_id="source",
            source_epoch="epoch",
            owner_epoch="owner",
        )
        await journal.create()
        await journal.update(lambda s: s.update(source_state="ready"))
        scope = dict(source_session_id="source", source_epoch="epoch")
        binding = dict(protocol="worldkit-area-condition-v1", identity=identity())
        started = await journal.start(
            dict(
                **scope,
                op="render.start",
                request_id="start",
                expected_render_revision=0,
                prompt="indoor",
                area_binding=binding,
            )
        )
        state = await journal.read()
        segment = state["segment"]
        assert segment["area_binding"] == binding
        binding["identity"]["worldEpoch"] = 9
        assert (await journal.read())["segment"]["area_binding"]["identity"][
            "worldEpoch"
        ] == 3
        replacement = dict(
            protocol="worldkit-area-condition-v1",
            identity={**identity(), "operationId": "next-op"},
        )
        await journal.update(lambda s: s["segment"].update(state="streaming"))
        await journal.prompt(
            dict(
                **scope,
                op="prompt.replace",
                request_id="prompt",
                segment_id=started["segment_id"],
                segment_epoch=started["segment_epoch"],
                expected_prompt_version=1,
                prompt_version=2,
                prompt="new indoor",
                effective=dict(policy="at_or_after", source_frame_index=2),
                area_binding=replacement,
            )
        )
        assert (await journal.read())["segment"]["prompt_queue"][0][
            "area_binding"
        ] == replacement

    asyncio.run(run())


def test_model_mapping_uses_actual_condition_event_and_aligned_original_source():
    import asyncio
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_segment import (
        RenderSegment,
    )

    async def run():
        seen = []

        async def output(*args):
            seen.append(args)

        segment = object.__new__(RenderSegment)
        from sglang.multimodal_gen.runtime.realtime.whitebox.sink_control import (
            SinkControl,
        )

        segment.sink = SinkControl(segment)
        segment.periodic_refresh = None
        segment.first_output_at = None
        from sglang.multimodal_gen.runtime.realtime.whitebox.ttff_metrics import (
            TimingMetrics,
        )

        segment.detailed_timing = True
        segment.timing_metrics = TimingMetrics()
        segment.room_timings = {}
        segment.segment_id = "segment"
        segment.segment_epoch = "segment-epoch"
        segment.segment = {"seed": 42}
        segment.owner = SimpleNamespace(segment_prompt_output=output)
        refs = [dict(id="scene", sha256="a" * 64)]
        condition = dict(
            version=2,
            text="actual applied",
            sha256="b" * 64,
            area_binding=dict(
                protocol="worldkit-area-condition-v1", identity=identity()
            ),
            reference_descriptors=refs,
        )
        segment.prompt_by_event = {"actual-event": condition}
        segment.active_reference_descriptors = []
        original = SimpleNamespace(
            index=9, pts_us=375000, provenance=dict(area_frame=area_frame())
        )
        message = dict(
            event_id="actual-event",
            output_frame_start=4,
            source_alignment=[
                dict(source_frame_start=4, output_frame_start=4, num_frames=1)
            ],
        )
        prompt, rendering = await segment.model_output_condition(
            message, {4: (original, 1)}, None, dict(model_epoch="model-epoch")
        )
        assert prompt == "actual applied" and seen[0][3] is original
        assert rendering["area_condition"]["binding"] == condition["area_binding"]
        assert rendering["area_condition"]["event_id"] == "actual-event"
        assert rendering["area_condition"]["reference_sha256"]
        assert rendering["area_condition"]["seed"] == 42
        assert rendering["area_condition"]["effective_source"] == dict(
            source_frame_index=9, source_pts_us=375000
        )
        with pytest.raises(ValueError):
            await segment.model_output_condition(
                {**message, "event_id": "unknown"}, {4: (original, 1)}, None, {}
            )

    asyncio.run(run())


@pytest.mark.parametrize("transport", ["websocket", "webrtc-rtp"])
def test_room_offer_is_owned_by_attachment_display_and_existing_reliable_history(
    transport,
):
    from collections import deque
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import (
        RenderPipeline,
    )

    pipe = object.__new__(RenderPipeline)
    pipe.media_transport = transport
    pipe.generation = 1
    pipe.subscription = SimpleNamespace(
        generation=1, closed=False, area_challenge="attachment-challenge"
    )
    status = dict(
        type="area.transition",
        epoch=0,
        revision=1,
        runtime=dict(runtimeHash="e" * 64, runtimeManifestSha256="f" * 64),
        status=dict(phase="awaiting-presented", inputHeld=True, transition=identity()),
    )
    pipe.session = SimpleNamespace(
        area_state=status,
        source_epoch="room-source-epoch",
        session_id="room-source",
        current_model_stream="model-stream",
        binding_ended=False,
        journal=SimpleNamespace(
            snapshot=dict(
                display_mode="whitebox",
                display_revision=2,
                render_revision=1,
                segment=None,
            )
        ),
    )
    pipe.history = deque()
    frame = dict(
        type="whitebox.frame",
        stream="whitebox",
        media_stream_id="room-source-epoch",
        output_frame_start=3,
        pts_us=375000,
        source=dict(area_frame=area_frame()),
    )
    offer = pipe.area_offer(frame)
    assert offer["challenge"] == "attachment-challenge" and offer["display"] == dict(
        mode="whitebox", revision=2
    )
    assert offer["frame"] == area_frame()[
        "frame"
    ] and "native-private" not in json.dumps(offer)
    pipe.history.append((1, 100, offer))
    command = dict(
        challenge=offer["challenge"],
        receipt=dict(
            identity=identity(),
            frame=offer["frame"],
            uiRevision=7,
            uiStatus="synchronized",
        ),
        display=offer["display"],
        connection_generation=1,
    )
    assert pipe.resolve_area_offer(command) == offer
    pipe.session.journal.snapshot["display_revision"] = 3
    with pytest.raises(ValueError):
        pipe.resolve_area_offer(command)
    pipe.session.journal.snapshot["display_revision"] = 2
    pipe.generation = 2
    with pytest.raises(ValueError):
        pipe.resolve_area_offer(command)
    pipe.generation = 1
    pipe.session.area_state["status"]["transition"] = {
        **identity(),
        "operationId": "replacement",
    }
    assert pipe.area_offer(frame) is None
    with pytest.raises(ValueError):
        pipe.resolve_area_offer(command)


@pytest.mark.parametrize("wire_envelope", [False, True])
def test_area_command_journals_before_native_send_and_reconciles_original_unknown(
    wire_envelope,
):
    import asyncio
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderJournal,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import MemoryRoomStore

    async def run():
        journal = RenderJournal(
            MemoryRoomStore(),
            user_id="u",
            source_session_id="source",
            source_epoch="epoch",
            owner_epoch="owner",
        )
        await journal.create()
        await journal.update(lambda s: s.update(source_state="ready"))
        sent = []

        async def status():
            return dict(
                status=dict(
                    phase="awaiting-presented", inputHeld=True, transition=identity()
                )
            )

        async def presented(body, *, before_send):
            before_send()
            assert (await journal.read())["operations"][body["request_id"]][
                "state"
            ] == "dispatching"
            sent.append(body)
            raise ConnectionError("post-send unknown")

        source = object.__new__(SourceSession)
        source.journal = journal
        source.config = SimpleNamespace(play_seconds=420)
        source.engine = SimpleNamespace(
            stream_id="view", area_status=status, area_presented=presented
        )
        source.source_epoch = "epoch"
        source.session_id = "source"
        source.outgoing = asyncio.Queue()
        source.control_tasks = set()
        source.binding_ended = False
        source.control_tokens = 4
        source.control_token_time = __import__("time").monotonic()
        source.source_state_value = "ready"
        source.display_mode = "whitebox"
        source.play_elapsed = source.wait_elapsed = 0
        source.resolve_area_offer = lambda c: dict(identity=identity())
        receipt = dict(
            identity=identity(),
            frame=area_frame()["frame"],
            uiRevision=7,
            uiStatus="synchronized",
        )
        command = dict(
            op="area.presented",
            request_id="original",
            source_session_id="source",
            source_epoch="epoch",
            connection_generation=1,
            challenge="challenge",
            receipt=receipt,
            display=dict(mode="whitebox", revision=0),
        )
        if wire_envelope:
            command.update(type="command", version=2)
        await source.command(command)
        await asyncio.gather(*source.control_tasks)
        operation = (await journal.read())["operations"]["original"]
        assert (
            operation["state"] == "reconciling"
            and operation["code"] == "AREA_PRESENTATION_UNKNOWN"
        )
        assert len(sent) == 1 and set(sent[0]) == {"request_id", "receipt"}
        await source.command(command)
        assert len(sent) == 1
        await source.command(
            dict(
                op="operation.get",
                request_id="query",
                operation_request_id="original",
                source_session_id="source",
                source_epoch="epoch",
            )
        )
        import msgspec

        messages = []
        while not source.outgoing.empty():
            messages.append(msgspec.msgpack.decode(source.outgoing.get_nowait()))
        assert messages[-1]["operation"]["request_id"] == "original"
        assert messages[-1]["operation"]["state"] == "reconciling"
        source.control_tokens = 0
        source.control_token_time = __import__("time").monotonic()
        await source.command({**command, "request_id": "rate-limited"})
        await asyncio.gather(*source.control_tasks)
        assert "rate-limited" not in (await journal.read())["operations"]
        assert (
            msgspec.msgpack.decode(source.outgoing.get_nowait())["code"]
            == "LIMIT_EXCEEDED"
        )

    asyncio.run(run())


def test_pending_native_area_context_can_request_new_condition_without_claiming_model_pixels():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import (
        RenderPipeline,
    )

    pipe = object.__new__(RenderPipeline)
    pipe.generation = 2
    pipe.subscription = SimpleNamespace(
        generation=2, closed=False, area_challenge="new-attachment"
    )
    pending = {**identity(), "operationId": "new-area"}
    pipe.session = SimpleNamespace(
        session_id="source",
        source_epoch="epoch",
        current_model_stream="model",
        binding_ended=False,
        area_state=dict(
            runtime=dict(runtimeHash="e" * 64, runtimeManifestSha256="f" * 64),
            status=dict(phase="awaiting-presented", inputHeld=True, transition=pending),
        ),
        journal=SimpleNamespace(
            snapshot=dict(
                display_mode="model",
                display_revision=1,
                render_revision=2,
                segment=dict(
                    segment_id="segment",
                    segment_epoch="model",
                    state="streaming",
                    accepted_version=1,
                    area_binding=dict(
                        protocol="worldkit-area-condition-v1", identity=identity()
                    ),
                ),
            )
        ),
    )
    context = pipe.area_context()
    assert context["identity"] == pending and context["type"] == "area.context"
    assert all(key not in context for key in ("frame", "model", "nonce", "ack"))
    message = dict(
        stream="model",
        source=dict(area_frame=area_frame()),
        media_stream_id="model",
        output_frame_start=1,
        pts_us=100,
    )
    assert pipe.area_offer(message) is None
    pipe.session.area_state["status"].update(phase="interactive", inputHeld=False)
    current = pipe.area_context()
    assert current["status"] == "interactive" and current["identity"] == pending
    assert pipe.area_offer(message) is None


def test_area_model_condition_requires_the_current_native_operation_before_model_intent():
    import asyncio
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def run():
        async def status():
            return dict(
                status=dict(
                    phase="awaiting-presented", inputHeld=True, transition=identity()
                )
            )

        source = object.__new__(SourceSession)
        source.engine = SimpleNamespace(
            metadata={"area_protocol": "worldkit-area-presented-v1"}, area_status=status
        )
        await source.validate_area_binding(
            dict(
                area_binding=dict(
                    protocol="worldkit-area-condition-v1", identity=identity()
                )
            )
        )
        for bad in (
            {},
            dict(
                area_binding=dict(
                    protocol="worldkit-area-condition-v1",
                    identity={**identity(), "operationId": "stale"},
                )
            ),
        ):
            with pytest.raises(ValueError):
                await source.validate_area_binding(bad)

    asyncio.run(run())


def test_area_public_parser_limits_match_sdk_utf16_and_reason_codes():
    from sglang.multimodal_gen.runtime.realtime.whitebox.area_contract import (
        read_area_result,
        read_area_transition,
    )

    value = identity()
    value["operationId"] = "😀" * 2049
    with pytest.raises(ValueError):
        read_area_identity(value)
    with pytest.raises(ValueError):
        read_area_result(
            dict(
                type="area.presentation-result",
                epoch=0,
                requestId="r",
                operationId="o",
                status="rejected",
                code="private error detail",
            )
        )
    transition = dict(
        type="area.transition",
        epoch=0,
        revision=1,
        runtime=dict(runtimeHash="e" * 64, runtimeManifestSha256="f" * 64),
        status=dict(
            phase="failed",
            inputHeld=True,
            draining=False,
            recoverable=True,
            residentAreas=0,
            error="private detail",
        ),
    )
    with pytest.raises(ValueError):
        read_area_transition(transition)


def test_cli_area_attachment_pins_runtime_publication_and_revokes_replaced_pending_operation():
    import asyncio
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import CLIView
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec

    async def run():
        descriptor = dict(
            sessionId="native-session",
            epoch=0,
            worldBuildHash="a" * 64,
            area=dict(protocol="worldkit-area-presented-v1"),
            loading=dict(
                kind="areas",
                initialPreparation=dict(identity=identity()["publication"]),
            ),
        )
        view = CLIView(
            SimpleNamespace(
                descriptor=descriptor, spec=VideoSpec(64, 36), prepared_delivery=None
            )
        )
        status = dict(
            type="area.transition",
            epoch=0,
            revision=1,
            runtime=dict(runtimeHash="e" * 64, runtimeManifestSha256="f" * 64),
            status=dict(
                phase="awaiting-presented",
                inputHeld=True,
                draining=False,
                recoverable=True,
                residentAreas=1,
                transition=identity(),
            ),
            ack=dict(nonce="private"),
        )
        view._accept_area_transition(status)
        future = asyncio.get_running_loop().create_future()
        view.area_pending["request"] = ("native-operation", future)
        changed = copy.deepcopy(status)
        changed["revision"] = 2
        changed["status"]["transition"]["operationId"] = "replacement"
        view._accept_area_transition(changed)
        with pytest.raises(ValueError):
            await future
        for mutate in (
            lambda v: v["runtime"].update(runtimeHash="0" * 64),
            lambda v: v["status"]["transition"]["publication"].update(
                publicationId="other"
            ),
        ):
            bad = copy.deepcopy(changed)
            bad["revision"] = 3
            mutate(bad)
            with pytest.raises(ValueError):
                view._accept_area_transition(bad)
        assert view.area_transition == changed

    asyncio.run(run())


def test_area_ready_negotiates_only_actual_native_descriptor_and_supported_transport(
    monkeypatch,
):
    import copy
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import (
        RenderPipeline,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        RoomPipeline,
    )

    pipeline = object.__new__(RenderPipeline)
    pipeline.init = {"whitebox": {}}
    pipeline.session = SimpleNamespace(
        model_context_segment=SourceSession.model_context_segment,
        source_epoch="source",
        config=SimpleNamespace(play_seconds=420),
        play_elapsed=20,
        player_remaining_seconds=400,
        current_model_stream=None,
        journal=SimpleNamespace(snapshot={}),
    )
    base = dict(
        capabilities={"render_segments": True},
        engine={},
        source_video_stream={},
        video_stream={},
    )
    monkeypatch.setattr(RoomPipeline, "ready_message", lambda _: copy.deepcopy(base))
    for transport, protocol, supported in (
        ("websocket", None, False),
        ("websocket", "unknown", False),
        ("webrtc", "worldkit-area-presented-v1", False),
        ("websocket", "worldkit-area-presented-v1", True),
        ("webrtc-rtp", "worldkit-area-presented-v1", True),
    ):
        pipeline.media_transport = transport
        base["engine"] = {"area_protocol": protocol} if protocol else {}
        ready = pipeline.ready_message()
        assert ready["play_remaining_seconds"] == 400
        assert ready["model_context_segment"] is None
        capabilities = ready["capabilities"]
        assert capabilities["render_segments"] is True
        assert ("area_presented" in capabilities) is supported
        if supported:
            assert capabilities["area_presented"] == "worldkit-area-presented-v1"
            assert capabilities["area_mapping"] == dict(
                whitebox=True, model=True, transports=[transport]
            )
        else:
            assert "area_mapping" not in capabilities


@pytest.mark.parametrize("transport", ["websocket", "webrtc-rtp"])
def test_native_confirmation_completion_survives_its_own_interactive_status_refresh(
    transport,
):
    import asyncio
    import copy
    import time
    from collections import deque
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderJournal,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import (
        RenderPipeline,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import MemoryRoomStore

    async def run():
        journal = RenderJournal(
            MemoryRoomStore(),
            user_id="u",
            source_session_id="source",
            source_epoch="epoch",
            owner_epoch="owner",
        )
        await journal.create()
        await journal.update(lambda state: state.update(source_state="ready"))
        source = object.__new__(SourceSession)
        source.journal = journal
        source.source_epoch = "epoch"
        source.session_id = "source"
        source.outgoing = asyncio.Queue()
        source.control_tasks = set()
        source.binding_ended = False
        source.control_tokens = 4
        source.control_token_time = time.monotonic()
        source.source_state_value = "ready"
        source.display_mode = "whitebox"
        source.play_elapsed = source.wait_elapsed = 0
        source.current_model_stream = None
        native = dict(
            runtime=dict(runtimeHash="e" * 64, runtimeManifestSha256="f" * 64),
            status=dict(
                phase="awaiting-presented", inputHeld=True, transition=identity()
            ),
        )

        async def status():
            return copy.deepcopy(native)

        sent = []

        async def presented(body, *, before_send):
            before_send()
            sent.append(body)
            native["status"].update(phase="interactive", inputHeld=False)
            # A real capture can refresh the native status while this RPC's reply is in flight.
            await source.refresh_area_state()
            return dict(status="confirmed", operationId=identity()["operationId"])

        source.engine = SimpleNamespace(
            stream_id="view", area_status=status, area_presented=presented
        )
        await source.refresh_area_state()
        pipeline = object.__new__(RenderPipeline)
        pipeline.media_transport = transport
        pipeline.session = source
        pipeline.generation = 1
        pipeline.subscription = SimpleNamespace(
            generation=1, closed=False, area_challenge="attachment"
        )
        pipeline.history = deque()
        offer = pipeline.area_offer(
            dict(
                type="whitebox.frame",
                stream="whitebox",
                media_stream_id="epoch",
                output_frame_start=0,
                pts_us=0,
                source=dict(area_frame=area_frame()),
            )
        )
        pipeline.history.append((1, 100, offer))
        source.resolve_area_offer = pipeline.resolve_area_offer
        command = dict(
            type="command",
            version=2,
            op="area.presented",
            request_id="original",
            source_session_id="source",
            source_epoch="epoch",
            connection_generation=1,
            challenge="attachment",
            display=offer["display"],
            receipt=dict(
                identity=identity(),
                frame=offer["frame"],
                uiRevision=7,
                uiStatus="synchronized",
            ),
        )
        await source.command(command)
        await asyncio.gather(*source.control_tasks)
        operation = (await journal.read())["operations"]["original"]
        assert (
            operation["state"] == "completed"
            and operation["native_status"] == "confirmed"
        )
        assert len(sent) == 1
        # Interactive status never authorizes a fresh presentation dispatch.
        with pytest.raises(ValueError):
            pipeline.resolve_area_offer(command)

    asyncio.run(run())


@pytest.mark.parametrize("boundary", ["input_lock", "receipt_log", "native_status"])
@pytest.mark.parametrize(
    "revoke", ["detach", "generation", "display", "binding", "none"]
)
def test_area_confirmation_rechecks_attachment_at_native_dispatch(boundary, revoke):
    import asyncio
    from collections import deque
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import CLIView
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderJournal,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import (
        RenderPipeline,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        LocalRoomEngine,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import MemoryRoomStore

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        sent = []
        journal = RenderJournal(
            MemoryRoomStore(),
            user_id="u",
            source_session_id="source",
            source_epoch="epoch",
            owner_epoch="owner",
        )
        await journal.create()
        await journal.update(
            lambda state: state.update(
                source_state="ready",
                display_mode="whitebox",
                display_revision=2,
                render_revision=1,
                segment=None,
            )
        )
        view = object.__new__(CLIView)
        view.finish_requested = asyncio.Event()
        view.descriptor = dict(
            sessionId="native-session",
            epoch=0,
            area=dict(protocol="worldkit-area-presented-v1"),
        )
        view.closed = view.stopped = False
        view.area_transition = dict(
            type="area.transition",
            epoch=0,
            revision=1,
            runtime=dict(runtimeHash="e" * 64, runtimeManifestSha256="f" * 64),
            status=dict(
                phase="awaiting-presented", inputHeld=True, transition=identity()
            ),
            ack=dict(nonce="native-private"),
        )
        view.area_frames = {9: area_frame()}
        view.area_pending = {}
        view.control_task = asyncio.create_task(asyncio.Event().wait())

        class Control:
            async def send(self, wire):
                message = json.loads(wire)
                sent.append(message)
                view.area_pending[message["requestId"]][1].set_result(
                    dict(status="confirmed", operationId=identity()["operationId"])
                )

        view.control = Control()
        native_status = view.area_status
        status_calls = 0

        async def status():
            nonlocal status_calls
            status_calls += 1
            if boundary == "native_status" and status_calls == 2:
                entered.set()
                await release.wait()
            return await native_status()

        view.area_status = status

        class Log:
            async def event(self, *args):
                if boundary == "receipt_log":
                    entered.set()
                    await release.wait()
                return {}

        runtime = object.__new__(RoomRuntime)
        runtime.players = {"a" * 32: dict(game_id="b" * 32, engine=view, identity={})}
        lock = asyncio.Lock()
        runtime.rooms = {"b" * 32: dict(input_lock=lock, log=Log())}
        source = object.__new__(SourceSession)
        source.journal = journal
        source.engine = LocalRoomEngine(
            runtime, dict(stream_id="a" * 32, engine={}, ui={})
        )
        source.source_epoch, source.session_id = "epoch", "source"
        source.outgoing = asyncio.Queue()
        source.binding_ended = False
        source.area_state = await native_status()
        pipe = object.__new__(RenderPipeline)
        pipe.session, pipe.generation = source, 1
        pipe.subscription = SimpleNamespace(
            generation=1, closed=False, area_challenge="attachment"
        )
        pipe.history = deque()
        offer = pipe.area_offer(
            dict(
                type="whitebox.frame",
                stream="whitebox",
                media_stream_id="epoch",
                output_frame_start=3,
                pts_us=375000,
                source=dict(area_frame=area_frame()),
            )
        )
        pipe.history.append((1, 100, offer))
        source.resolve_area_offer = pipe.resolve_area_offer
        command = dict(
            op="area.presented",
            request_id="visible-request",
            source_session_id="source",
            source_epoch="epoch",
            connection_generation=1,
            challenge=offer["challenge"],
            display=offer["display"],
            receipt=dict(
                identity=identity(),
                frame=offer["frame"],
                uiRevision=7,
                uiStatus="synchronized",
            ),
        )
        await journal.command(
            command, lambda state, operation: operation.update(state="accepted")
        )
        if boundary == "input_lock":
            await lock.acquire()
        task = asyncio.create_task(source.confirm_area(command))
        try:
            async with asyncio.timeout(2):
                if boundary == "input_lock":
                    while (
                        journal.snapshot["operations"][command["request_id"]]["state"]
                        != "dispatching"
                    ):
                        if task.done():
                            await task
                            pytest.fail("Confirmation ended before reaching dispatch")
                        await asyncio.sleep(0)
                else:
                    await entered.wait()
                if revoke == "detach":
                    pipe.subscription = None
                elif revoke == "generation":
                    pipe.generation += 1
                elif revoke == "display":
                    await journal.update(lambda state: state.update(display_revision=3))
                elif revoke == "binding":
                    source.binding_ended = True
                if boundary == "input_lock":
                    lock.release()
                release.set()
                await task
            operation = journal.snapshot["operations"][command["request_id"]]
            if revoke == "none":
                assert len(sent) == 1
                assert operation["state"] == "completed"
            else:
                assert sent == []
                assert operation["state"] == "rejected"
                assert operation["code"] == "AREA_PRESENTATION_STALE"
            assert view.area_transition["status"]["inputHeld"] is True
            assert view.area_pending == {}
        finally:
            task.cancel()
            view.control_task.cancel()
            await asyncio.gather(task, view.control_task, return_exceptions=True)

    asyncio.run(run())


def test_rtc_attachment_installs_presentation_resolvers_before_receiving_commands(
    monkeypatch,
):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox import rtc_routes

    pipeline = object.__new__(rtc_routes.RTCRenderPipeline)
    pipeline.session = SimpleNamespace(resolve_area_offer=None)
    pipeline.generation = 1
    pipeline.rtc_peer = object()

    def resolve_opening_offer(*_args):
        raise AssertionError("Attachment does not resolve an offer")

    def verify_scene_model_frame(*_args):
        raise AssertionError("Attachment does not verify a frame")

    sub = SimpleNamespace(
        closed=False,
        generation=1,
        first_presentation_offers=None,
        resolve_first_presentation_offer=resolve_opening_offer,
        scene_transition_supported=False,
        scene_model_supported=False,
        verify_scene_model_frame=verify_scene_model_frame,
    )
    monkeypatch.setattr(rtc_routes, "RTCSubscription", lambda *args: sub)
    pipeline.subscription = pipeline.create_subscription()
    assert pipeline.session.resolve_area_offer == pipeline.resolve_area_offer
    assert pipeline.session.resolve_first_presentation_offer is resolve_opening_offer
    assert not pipeline.session.first_presented_player_ready()
    sub.first_presentation_offers = object()
    assert pipeline.session.first_presented_player_ready()
    assert pipeline.session.resolve_scene_model_frame is verify_scene_model_frame
    assert pipeline.session.scene_player_generation() is None
    assert not pipeline.session.scene_model_player_ready()
    sub.scene_transition_supported = True
    assert pipeline.session.scene_player_generation() == 1
    sub.scene_model_supported = True
    assert pipeline.session.scene_model_player_ready()
    sub.closed = True
    assert not pipeline.session.first_presented_player_ready()
    assert pipeline.session.scene_player_generation() is None
    assert not pipeline.session.scene_model_player_ready()


def test_rtc_media_carries_the_same_area_offer_as_reliable_history(monkeypatch):
    from collections import deque

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import (
        RenderPipeline,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        RoomPipeline,
    )

    pipe = object.__new__(RenderPipeline)
    pipe.media_transport = "webrtc-rtp"
    pipe.history = deque()
    pipe.subscription = None
    offer = dict(
        type="area.offer",
        challenge="current-attachment",
        output=dict(output_frame_start=3),
    )
    monkeypatch.setattr(RenderPipeline, "area_context", lambda _: None)
    monkeypatch.setattr(
        RenderPipeline, "area_offer", lambda _, message: copy.deepcopy(offer)
    )
    published = []
    monkeypatch.setattr(
        RoomPipeline,
        "publish",
        lambda _, message: published.append(copy.deepcopy(message)),
    )
    original = dict(type="whitebox.frame", output_frame_start=3)
    pipe.publish(original)
    assert published[0] == offer
    assert published[1]["area_offer"] == offer
    assert "area_offer" not in original
