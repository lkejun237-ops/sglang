import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import msgspec
import pytest
from test_whitebox_source_conditions import condition, handle_settled, setup

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.control_prefill import (
    ControlPrefill,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import LatestFrame
from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import RenderError
from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import SourceSession
from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
    CLIConditionChannel,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.source_conditions import (
    envelope,
    request_identity,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.startup_prewarm import (
    PROTOCOL,
    StartupPrewarm,
    parse_decision,
)


def decision(window, revision=1, value=None):
    value = condition() if value is None else value
    return dict(
        **envelope(value),
        type="model.context.prewarm",
        windowId=window.window_id,
        decisionRevision=revision,
        decision="candidate",
        intentId="intent",
        candidateRevision=1,
        eventId="future-choice",
        targetHash="0" * 64,
        contentHash="a" * 64,
        lifecycle=value["lifecycle"],
        prompt="Future condition",
        references=value["references"],
        effective=value["effective"],
    )


def test_prewarm_strict_candidate_uses_formal_resources():
    candidate = decision(SimpleNamespace(window_id="window"))
    assert parse_decision(candidate) == candidate
    candidate["references"][0]["resource"]["epoch"] += 1
    with pytest.raises(ValueError):
        parse_decision(candidate)


def test_prewarm_owns_one_compute_and_waits_for_verified_release():
    async def scenario():
        c = condition()
        latest = LatestFrame()
        captured = SimpleNamespace(
            index=50,
            rgb=b"one-native-frame",
            provenance={
                "cli_frame_header": {
                    "sessionId": c["source"]["sessionId"],
                    "epoch": c["source"]["epoch"],
                    "source": {
                        **c["source"],
                        **c["effective"],
                    },
                }
            },
        )
        await latest.put((captured, 1))
        emitted, calls = [], []
        computing, cleanup = asyncio.Event(), asyncio.Event()

        class Prefill:
            failure = None

            async def start_startup(self, value, refs, first, *, current, observe):
                assert current()
                calls.append((value, refs, first))
                observe(dict(type="prefill.accepted", request_id="request"))

                async def run():
                    try:
                        computing.set()
                        await asyncio.Future()
                    finally:
                        await cleanup.wait()

                return asyncio.create_task(run())

        async def resolve(value):
            return value["references"]

        async def emit(value):
            emitted.append(value)

        owner = SimpleNamespace(
            latest=latest,
            spawn_control=asyncio.create_task,
            control_prefill=Prefill(),
            context_seed=41,
        )
        loop = asyncio.get_running_loop()
        window = StartupPrewarm(owner, c["segmentId"], 7, loop.time() + 10)
        try:
            _, released = window.gate(c, emit=emit, resolve=resolve, check=lambda: None)
            window.receive(decision(window))
            await asyncio.wait_for(computing.wait(), 1)
            window.receive(decision(window, revision=2))
            await asyncio.sleep(0)
            assert not released.done()
            cleanup.set()
            await asyncio.wait_for(asyncio.shield(released), 1)
            assert len(calls) == 1 and calls[0][0]["seed"] == 41
            assert calls[0][2] == (1, (captured, 1))
            assert window.first_for(c) is None
        finally:
            cleanup.set()
            await window.close()

    asyncio.run(scenario())


def test_prewarm_uncertain_child_release_keeps_gate_closed():
    async def scenario():
        c = condition()
        latest = LatestFrame()
        await latest.put(
            (
                SimpleNamespace(
                    index=50,
                    rgb=b"first",
                    provenance={
                        "cli_frame_header": {
                            "sessionId": c["source"]["sessionId"],
                            "epoch": c["source"]["epoch"],
                            "source": {**c["source"], **c["effective"]},
                        }
                    },
                ),
                1,
            )
        )

        class Prefill:
            failure = None

            async def start_startup(self, value, refs, first, *, current, observe):
                async def run():
                    self.failure = RenderError("RELEASE_UNKNOWN")
                    observe(dict(type="prefill.failed", request_id="request"))

                return asyncio.create_task(run())

        async def resolve(value):
            return value["references"]

        async def emit(_):
            pass

        owner = SimpleNamespace(
            latest=latest,
            spawn_control=asyncio.create_task,
            control_prefill=Prefill(),
            context_seed=42,
        )
        loop = asyncio.get_running_loop()
        window = StartupPrewarm(owner, c["segmentId"], 7, loop.time() + 10)
        try:
            _, released = window.gate(c, emit=emit, resolve=resolve, check=lambda: None)
            window.receive(decision(window))
            with pytest.raises(RenderError) as error:
                await asyncio.wait_for(asyncio.shield(released), 1)
            assert error.value.code == "RELEASE_UNKNOWN"
        finally:
            with pytest.raises(RenderError) as error:
                await window.close()
            assert error.value.code == "RELEASE_UNKNOWN"

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["none", "reopen", "new-boundary"])
def test_prewarm_and_formal_start_share_pinned_capture_and_bound_slot(tmp_path, change):
    async def scenario():
        room = object.__new__(SourceSession)
        messages, borrowed, starts, observations = [], [], [], []
        room.prewarm_observer = observations.append
        announced, started = asyncio.Event(), asyncio.Event()

        async def emit(value):
            messages.append(value)
            if value["type"] == "model.context.prewarm.window":
                announced.set()

        async def transition(operation, refs, first):
            starts.append((operation, refs, first))
            started.set()

        def gate(value):
            return window.gate(
                value,
                emit=emit,
                resolve=lambda c: adapter.prepare_references(c, adapter.capabilities()),
                check=adapter.check,
            )

        async def operation_message(_):
            pass

        class Child:
            resource_state = "reserved"

            def __init__(self):
                self.completed = asyncio.Event()

            @asynccontextmanager
            async def open(self):
                yield self, None

            async def send(self, wire):
                self.request = msgspec.msgpack.decode(wire)
                borrowed.append(self.request)

            async def recv(self):
                await room.latest.put((later_frame, 2))
                self.resource_state = "released"
                self.completed.set()
                return msgspec.msgpack.encode(
                    dict(
                        type="prefill.memory_ready",
                        request_id=self.request["request_id"],
                        cache_key="a" * 64,
                    )
                )

        adapter, c, room.journal, receipts, _, resources = await setup(
            tmp_path, startup_release=gate
        )
        await room.journal.update(
            lambda s: s.update(inference=dict(enabled=True, revision=7))
        )
        room.assets = adapter.assets
        room.context_seed = 71
        room.binding_ended = False
        room.session_id, room.source_epoch = "source", "epoch"
        room.context_channel = SimpleNamespace(suspended=False)
        room.engine = SimpleNamespace(stream_id="stream")
        room.outgoing = asyncio.Queue()
        room.latest = LatestFrame()
        room.config = SimpleNamespace(startup_seconds=2, spec=VideoSpec(2, 2))
        room.upstream = SimpleNamespace(init=dict(size="2x2"))
        room.model_binding = SimpleNamespace(
            assignment=SimpleNamespace(
                denoiser=SimpleNamespace(
                    worker_epoch="worker-epoch", worker_id="worker"
                )
            ),
            borrow=lambda segment, *, kind: Child() if kind == "prefill" else None,
        )
        room.control_prefill = ControlPrefill(room)
        room.spawn_control = asyncio.create_task
        room._transition_segment = transition
        room.operation_message = operation_message
        adapter.submit = room.submit_source_condition

        def frame_at(index, advance):
            marker = dict(
                policy="at-or-after",
                **{
                    key: c["effective"][key] + advance
                    for key in ("sourceFrameId", "sourceTimeUs", "simulationTick")
                },
            )
            return marker, SimpleNamespace(
                index=index,
                rgb=bytes([index]) * 12,
                provenance={
                    "cli_frame_header": {
                        "sessionId": c["source"]["sessionId"],
                        "epoch": c["source"]["epoch"],
                        "source": {**c["source"], **marker},
                    }
                },
            )

        pinned_marker, frame = frame_at(72, 1)
        later_marker, later_frame = frame_at(73, 2)
        await room.latest.put((frame, 1))
        loop = asyncio.get_running_loop()
        window = StartupPrewarm(room, c["segmentId"], 7, loop.time() + 10)
        room.inference = SimpleNamespace(prewarm=window)
        adapter.startup_effective = window.effective_for
        resources.set()
        try:
            await adapter.handle(c)
            await asyncio.wait_for(announced.wait(), 1)
            window.receive(decision(window, value=c))
            await asyncio.wait_for(adapter.preparation, 1)
            assert receipts[-1]["phase"] == "prepared"
            assert room.latest.version == 2
            assert receipts[-1]["startupEffective"] == pinned_marker
            await adapter.reconcile([request_identity(c)])
            assert receipts[-1]["startupEffective"] == pinned_marker
            assert not (await room.journal.read())["operations"]
            if change == "reopen":
                await room.journal.update(lambda s: s["inference"].update(revision=8))
            await handle_settled(
                adapter,
                request_identity(c)
                | dict(
                    type="model.condition.commit",
                    controlSequence=c["controlSequence"] + 1,
                    effective=(
                        later_marker
                        if change == "new-boundary"
                        else receipts[-1]["startupEffective"]
                    ),
                ),
            )
            if change == "reopen":
                assert receipts[-1]["phase"] == "rejected"
                assert receipts[-1]["reason"] == "MODEL_UNAVAILABLE"
                assert not (await room.journal.read())["operations"]
                return
            await asyncio.wait_for(started.wait(), 1)
            assert len(borrowed) == 1
            assert borrowed[0]["source_frame"] == room.model_geometry.pad_rgb(frame.rgb)
            assert borrowed[0]["size"] == "32x32"
            assert borrowed[0]["prompt"] == "Future condition"
            assert borrowed[0]["seed"] == room.context_seed
            assert starts[0][0]["op"] == "render.start"
            segment = (await room.journal.read())["segment"]
            assert segment["prompt"] == c["prompt"]
            if change == "new-boundary":
                assert starts[0][2] is None
                assert segment["start_boundary"] == dict(mode="next_available")
            else:
                assert starts[0][2][1][0] is frame
                assert segment["start_boundary"] == dict(source_frame_index=frame.index)
            await window.sender
            assert [value["phase"] for value in messages[1:]] == [
                "accepted",
                "computing",
                "memory-ready",
            ]
            assert {value["requestId"] for value in messages[1:]} == {
                borrowed[0]["request_id"]
            }
            assert messages[-1]["cacheKey"] == "a" * 64
            assert [event["status"] for event in observations] == [
                "window",
                "accepted",
                "computing",
                "memory-ready",
            ]
            if change == "none":
                # Persistence can fail after memory-ready has released the startup
                # gate and its compute deadline has passed.
                window.deadline = loop.time() - 1
                await room.control_prefill.receipt(
                    dict(
                        type="prefill.failed",
                        request_id=borrowed[0]["request_id"],
                        memory_ready=True,
                        code="PREFILL_PERSISTENCE_FAILED",
                    )
                )
                assert observations[-1]["status"] == "persistence-failed"
                assert observations[-1]["request_id"] == borrowed[0]["request_id"]
                assert messages[-1]["phase"] == "memory-ready"
                assert window.computed is not None
                await room.journal.update(
                    lambda s: s["segment"].update(state="streaming")
                )
                window.receive(
                    dict(decision(window, revision=2, value=c), intentId="late-intent")
                )
                await window.sender
                assert messages[-1]["phase"] == "skipped"
                assert messages[-1]["reason"] == "window-closed"
                assert messages[-1]["intentId"] == "late-intent"
                assert len(borrowed) == 1
                # An unowned late decision cannot displace the result that
                # belongs to the child already computed for this window.
                assert observations[-1]["status"] == "persistence-failed"
                assert (
                    room.prewarm_observations.latest["prepare"]["status"]
                    == "memory-ready"
                )
                room.current_segment = SimpleNamespace(
                    segment_id=segment["segment_id"],
                    segment_epoch=segment["segment_epoch"],
                    prompt_available=asyncio.Event(),
                    recording=SimpleNamespace(manifest={}),
                    assignment=SimpleNamespace(
                        session_id="model", generation_id="generation"
                    ),
                )
                future = dict(
                    c,
                    conditionId="future",
                    payloadHash="b" * 64,
                    prompt="Future condition",
                    effective=later_marker,
                    onsetOccurrenceIds=["future-onset"],
                )
                operation = await room.submit_source_condition(
                    future, segment["references"]
                )
                assert operation["op"] == "prompt.replace"
                assert operation["prompt_version"] == 2
                # Simulated native telemetry still must identify the exact
                # committed prompt version and cache key to count as reuse.
                await room.segment_cache_receipt(
                    room.current_segment,
                    {
                        "prompt_version": 1,
                        "cache_result": "l1_hit",
                        "cache_key": "a" * 64,
                    },
                )
                assert observations[-1]["status"] == "fulfilled"
                await room.segment_cache_receipt(
                    room.current_segment,
                    {
                        "prompt_version": 2,
                        "cache_result": "l1_hit",
                        "cache_key": "a" * 64,
                    },
                )
                assert observations[-1]["status"] == "hit"
                assert observations[-1]["formal_request_id"] == operation["request_id"]
                assert observations[-1]["cache_key"] == "a" * 64
        finally:
            await adapter.close()
            await window.close()

    asyncio.run(scenario())


def test_no_candidate_deadline_releases_with_no_synthetic_request(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.realtime.whitebox.startup_prewarm.WAIT_SECONDS",
        0.02,
    )

    async def scenario():
        c = condition()
        messages = []

        async def emit(message):
            messages.append(message)

        async def resolve(_):
            pytest.fail("no candidate should resolve references")

        owner = SimpleNamespace(
            spawn_control=asyncio.create_task,
            control_prefill=SimpleNamespace(failure=None),
        )
        loop = asyncio.get_running_loop()
        window = StartupPrewarm(owner, c["segmentId"], 7, loop.time() + 10)
        try:
            _, released = window.gate(c, emit=emit, resolve=resolve, check=lambda: None)
            await asyncio.wait_for(released, 1)
            await window.sender
            result = messages[-1]
            assert result["type"] == "model.context.prewarm.result"
            assert (result["decisionRevision"], result["phase"], result["reason"]) == (
                0,
                "skipped",
                "budget-exhausted",
            )
            assert not {"intentId", "requestId", "cacheKey"} & result.keys()
        finally:
            await window.close()

    asyncio.run(scenario())


def test_only_active_negotiated_source_can_send_prewarm_decisions():
    async def scenario():
        c = condition()
        announcements, received = [], []

        async def send(_):
            pass

        window = SimpleNamespace(
            window_id="window",
            announce=lambda: announcements.append(True),
            receive=received.append,
        )
        view = SimpleNamespace(
            descriptor=dict(
                epoch=c["source"]["epoch"],
                modelContext=dict(
                    protocol=c["protocol"],
                    required=True,
                    configHash=c["configHash"],
                    gameBuildHash=c["source"]["gameBuildHash"],
                    compilerVersion=1,
                    visible=False,
                ),
            ),
            client_id=c["audience"]["consumerId"],
            control=SimpleNamespace(send=send),
            closed=False,
            stopped=False,
        )
        channel = CLIConditionChannel(view)
        channel.segment = c["segmentId"]
        channel.adapter = SimpleNamespace(
            candidate=None, envelope=envelope(c), startup_window=lambda: window
        )
        channel.require_subscription_pause = True
        channel.subscription_probe = channel.suspended = True
        ack = dict(
            type="model.context.subscribed",
            epoch=c["source"]["epoch"],
            protocol=c["protocol"],
            segmentId=c["segmentId"],
            consumer=c["audience"],
            sampleControlFloor=1,
            subscriptionPause=True,
        )
        candidate = decision(window, value=c)
        await channel.consume(dict(ack, paused=True, prewarmProtocol=PROTOCOL))
        await channel.consume(candidate)
        assert not announcements and not received

        channel.subscription_probe = channel.suspended = False
        await channel.consume(dict(ack, paused=False))
        await channel.consume(candidate)
        assert not announcements and not received
        await channel.consume(dict(ack, paused=False, prewarmProtocol=PROTOCOL))
        await channel.consume(candidate)
        assert announcements == [True] and received == [candidate]
        assert channel.sequence == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("reset_during_open", [False, True])
def test_only_the_reserved_successor_gets_a_new_prewarm_window(
    monkeypatch, reset_during_open
):
    monkeypatch.setenv("WHITEBOX_MODEL_CONTEXT_PREWARM", "1")

    async def scenario():
        room = object.__new__(SourceSession)
        opened, switched = [], []

        async def open_window(segment_id, revision):
            opened.append((segment_id, revision))

        async def switch(segment_id):
            switched.append(segment_id)

        room.closed = room.binding_ended = False

        def current(revision):
            if reset_during_open:
                raise RenderError("SOURCE_EPOCH_MISMATCH")
            return revision == 7

        room.inference = SimpleNamespace(
            manual=True, open_prewarm=open_window, current=current
        )
        room.context_channel = SimpleNamespace(switch_segment=switch)
        state = dict(
            model_context_segment_ids=dict(segment_id="next"),
            inference=dict(enabled=True, revision=7),
        )
        await room.handoff_context_segment(dict(segment_id="old"), state)
        assert opened == switched == []
        if reset_during_open:
            with pytest.raises(RenderError) as error:
                await room.handoff_context_segment(dict(segment_id="next"), state)
            assert error.value.code == "SOURCE_EPOCH_MISMATCH" and switched == []
        else:
            await room.handoff_context_segment(dict(segment_id="next"), state)
            assert switched == ["next"]
        assert opened == [("next", 7)]

    asyncio.run(scenario())
