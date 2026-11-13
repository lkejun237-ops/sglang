import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox import (
    startup_prewarm_plan as startup_prewarm,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import RenderError
from sglang.multimodal_gen.runtime.realtime.whitebox.source_conditions import envelope


def condition():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/model-context-v1.json").read_text()
    )
    return copy.deepcopy(
        next(m for m in fixture["messages"] if m["type"] == "model.condition")
    )


def proposal(window, c, *, budget=1000, wait="memory-ready"):
    return dict(
        **envelope(c),
        type="model.context.prewarm.plan",
        windowId=window.window_id,
        decisionRevision=1,
        planId="plan",
        planRevision=1,
        budgetMs=budget,
        waitFor=wait,
        entries=[
            dict(
                id=f"item-{i}",
                candidateRevision=1,
                eventId=f"event-{i}",
                targetHash="0" * 64,
                contentHash="a" * 64,
                **{k: c[k] for k in ("lifecycle", "references", "effective")},
                prompt=f"future-{i}",
            )
            for i in range(3)
        ],
    )


def test_plan_runs_three_candidates_on_one_frame_and_replays_without_dispatch():
    assert hasattr(startup_prewarm, "StartupPrewarmPlan"), (
        "Room must implement negotiated plans"
    )

    async def scenario():
        c = condition()
        messages = []
        calls = []
        observers = []
        frame = SimpleNamespace(
            provenance={
                "cli_frame_header": {
                    "sessionId": c["source"]["sessionId"],
                    "epoch": c["source"]["epoch"],
                    "source": {**c["source"], **c["effective"]},
                }
            }
        )

        class Latest:
            async def get(self, version):
                return (1, (frame, 1))

        class Prefill:
            failure = None

            async def start_startup(self, value, refs, first, *, current, observe):
                assert current()
                calls.append(first)
                observers.append(observe)
                request = f"req-{len(calls)}"

                async def run():
                    observe({"type": "prefill.accepted", "request_id": request})
                    observe(
                        {
                            "type": "prefill.memory_ready",
                            "request_id": request,
                            "cache_key": "a" * 64,
                        }
                    )

                return asyncio.create_task(run())

        async def emit(m):
            messages.append(m)

        async def resolve(v):
            return v["references"]

        owner = SimpleNamespace(
            latest=Latest(),
            spawn_control=asyncio.create_task,
            control_prefill=Prefill(),
            context_seed=17,
        )
        window = startup_prewarm.StartupPrewarmPlan(
            owner, c["segmentId"], 7, asyncio.get_running_loop().time() + 10
        )
        _, released = window.gate(c, emit=emit, resolve=resolve, check=lambda: None)
        value = proposal(window, c)
        window.receive(value)
        await asyncio.wait_for(asyncio.shield(released), 1)
        window.receive(value)
        observers[0](
            {"type": "prefill.persisted", "request_id": "req-1", "cache_key": "a" * 64}
        )
        await window.sender
        assert len(calls) == 3 and all(f is calls[0] for f in calls)
        assert window.first_for(c) is calls[0]
        assert messages[-1]["phase"] == "persisted" and messages[-1]["id"] == "item-0"
        assert sum(m.get("phase") == "finished" for m in messages) >= 1
        await window.close()

    asyncio.run(scenario())


async def harness(
    *,
    blocked=1,
    budget=1000,
    wait="memory-ready",
    cleanup_unknown=False,
    member=None,
    appendable=False,
    empty_initial=False,
):
    c = condition()
    if member:
        c["audience"] = {
            **c["audience"],
            "consumerId": member,
            "playerId": member,
            "viewId": member,
        }
        c["source"] = {**c["source"], "sessionId": member}
        for reference in c["references"]:
            reference["resource"]["sessionId"] = member
    messages = []
    calls = []
    observers = []
    entered = asyncio.Event()
    cleanup = asyncio.Event()
    first = (
        1,
        (
            SimpleNamespace(
                provenance={
                    "cli_frame_header": {
                        "sessionId": c["source"]["sessionId"],
                        "epoch": c["source"]["epoch"],
                        "source": {**c["source"], **c["effective"]},
                    }
                }
            ),
            1,
        ),
    )

    class Latest:
        async def get(self, version):
            return first

    class Prefill:
        failure = None

        async def start_startup(self, value, refs, frame, *, current, observe):
            calls.append((value, frame))
            observers.append(observe)
            index = len(calls) - 1

            async def run():
                observe({"type": "prefill.accepted", "request_id": f"req-{index}"})
                if index == blocked:
                    entered.set()
                    try:
                        await asyncio.Future()
                    finally:
                        await cleanup.wait()
                        if cleanup_unknown:
                            self.failure = RenderError("RELEASE_UNKNOWN")
                else:
                    observe(
                        {
                            "type": "prefill.memory_ready",
                            "request_id": f"req-{index}",
                            "cache_key": "a" * 64,
                        }
                    )

            return asyncio.create_task(run())

    async def emit(m):
        messages.append(m)

    async def resolve(v):
        return v["references"]

    owner = SimpleNamespace(
        latest=Latest(),
        spawn_control=asyncio.create_task,
        control_prefill=Prefill(),
        context_seed=17,
        model_binding=SimpleNamespace(
            assignment=SimpleNamespace(
                denoiser=SimpleNamespace(worker_id="worker", worker_epoch="epoch")
            )
        ),
    )
    window = startup_prewarm.StartupPrewarmPlan(
        owner, c["segmentId"], 7, asyncio.get_running_loop().time() + 10
    )
    window.gate(c, emit=emit, resolve=resolve, check=lambda: None)
    value = proposal(window, c, budget=budget, wait=wait)
    if appendable:
        window.append_supported = True
        value["mode"] = "appendable"
    if empty_initial:
        value["entries"] = []
    window.receive(value)
    return SimpleNamespace(**locals())


def control(h, kind, **extra):
    return dict(
        **envelope(h.c),
        type=startup_prewarm.PREFIX + "." + kind,
        windowId=h.window.window_id,
        decisionRevision=2,
        planId="plan",
        planRevision=1,
        reason="USER_CANCELLED",
        **extra,
    )


@pytest.mark.parametrize("action", ["finish", "timeout", "cancel"])
def test_mid_second_control_waits_for_cleanup_and_preserves_first_persistence(action):
    async def scenario():
        h = await harness(budget=50 if action == "timeout" else 1000)
        await asyncio.wait_for(h.entered.wait(), 1)
        if action != "timeout":
            h.window.receive(
                control(
                    h,
                    action,
                    **(
                        {"id": "item-1", "candidateRevision": 1}
                        if action == "cancel"
                        else {}
                    ),
                )
            )
        else:
            await asyncio.sleep(0.06)
        await asyncio.sleep(0.01)
        assert not h.window.released.done() and len(h.calls) == 2
        h.cleanup.set()
        await asyncio.wait_for(asyncio.shield(h.window.released), 1)
        assert len(h.calls) == (3 if action == "cancel" else 2)
        h.observers[0](
            {"type": "prefill.persisted", "request_id": "req-0", "cache_key": "a" * 64}
        )
        await h.window.sender
        assert (
            h.messages[-1]["phase"] == "persisted" and h.messages[-1]["id"] == "item-0"
        )
        await h.window.close()

    asyncio.run(scenario())


def test_unknown_release_fails_gate_and_stops_queue():
    async def scenario():
        h = await harness(cleanup_unknown=True)
        await h.entered.wait()
        h.window.receive(control(h, "finish"))
        h.cleanup.set()
        with pytest.raises(RenderError) as error:
            await h.window.released
        assert error.value.code == "RELEASE_UNKNOWN"
        assert len(h.calls) == 2
        with pytest.raises(RenderError):
            await h.window.close()

    asyncio.run(scenario())


def test_persist_wait_does_not_dispatch_next_item_until_disk_fact():
    async def scenario():
        h = await harness(blocked=-1, wait="persisted")
        for index in range(3):
            for _ in range(100):
                if (
                    len(h.calls) == index + 1
                    and h.window.entries[f"item-{index}"]["memory_ready"]
                ):
                    break
                await asyncio.sleep(0.001)
            assert len(h.calls) == index + 1
            h.observers[index](
                {"type": "prefill.persisted", "request_id": f"req-{index}"}
            )
        await h.window.released
        assert len(h.calls) == 3
        await h.window.close()

    asyncio.run(scenario())


def test_worker_change_rejects_replay_and_late_receipts():
    async def scenario():
        h = await harness(blocked=-1)
        await h.window.released
        await h.window.sender
        before = len(h.messages)
        h.owner.model_binding.assignment.denoiser.worker_epoch = "new-epoch"
        h.window.receive(h.value)
        h.observers[0]({"type": "prefill.persisted", "request_id": "req-0"})
        await asyncio.sleep(0)
        assert len(h.messages) == before and len(h.calls) == 3
        assert h.window.first_for(h.c) is None
        with pytest.raises(RenderError):
            await h.window.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["entries"].append(p["entries"][0]),
        lambda p: p.update(entries=p["entries"] * 11),
        lambda p: p.update(budgetMs=120001),
        lambda p: p.update(budgetMs=True),
        lambda p: p["entries"][0].update(candidateRevision=True),
        lambda p: p.update(entries=[], budgetMs=1),
        lambda p: p["entries"][0].update(untrusted=True),
    ],
)
def test_plan_rejects_entire_invalid_batch(mutation):
    value = proposal(SimpleNamespace(window_id="window"), condition())
    mutation(value)
    with pytest.raises((ValueError, RenderError)):
        startup_prewarm.parse_plan(value)


def test_zero_budget_dispatches_nothing_and_old_request_does_not_extend_deadline():
    async def scenario():
        h = await harness(budget=0)
        deadline = h.window.deadline
        h.window.receive(dict(h.value, decisionRevision=2, budgetMs=120000))
        await h.window.released
        await h.window.sender
        assert not h.calls and h.window.deadline == deadline
        assert len([m for m in h.messages if m.get("phase") == "skipped"]) == 3
        await h.window.close()

    asyncio.run(scenario())


def test_shared_typescript_plan_fixture():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/prewarm-plan-v1.json").read_text()
    )
    for message in fixture["messages"]:
        if message["type"] in {
            startup_prewarm.PREFIX,
            startup_prewarm.PREFIX + ".append",
            startup_prewarm.PREFIX + ".cancel",
            startup_prewarm.PREFIX + ".finish",
        }:
            assert startup_prewarm.parse_plan(message) == message
    message = copy.deepcopy(fixture["messages"][1])
    message["entries"][0]["references"][0]["resource"]["epoch"] += 1
    with pytest.raises(ValueError):
        startup_prewarm.parse_plan(message)


def test_plan_observer_keeps_cancelled_item_and_late_persisted_separate():
    from sglang.multimodal_gen.runtime.realtime.whitebox.prewarm_observation import (
        PrewarmPlanObservations,
    )

    async def scenario():
        h = await harness(blocked=-1)
        store = PrewarmPlanObservations(
            source_session_id="source", source_epoch="epoch", consumer_id="consumer"
        )
        store.record(h.window, {"type": startup_prewarm.PREFIX + ".window"})

        def record(window, message, entry=None):
            store.record(window, message, entry)

        h.owner.record_startup_prewarm = record
        await h.window.released
        h.window.receive(control(h, "cancel", id="item-0", candidateRevision=1))
        h.observers[0]({"type": "prefill.persisted", "request_id": "req-0"})
        replay = store.replay("connection")
        statuses = [m["status"] for m in replay if m.get("candidate_id") == "item-0"]
        assert (
            "memory-ready" in statuses
            and "invalidated" in statuses
            and "persisted" in statuses
        )
        assert replay[-1]["connection_generation"] == "connection"
        assert len(store.rows) == 3
        await h.window.close()

    asyncio.run(scenario())


def test_only_negotiated_plan_channel_forwards_new_messages():
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
        CLIConditionChannel,
    )

    async def scenario():
        c = condition()
        received = []
        announced = []

        async def send(_):
            pass

        window = SimpleNamespace(
            window_id="window",
            announce=lambda: announced.append(True),
            receive=received.append,
        )
        view = SimpleNamespace(
            descriptor={
                "epoch": c["source"]["epoch"],
                "modelContext": {
                    "protocol": c["protocol"],
                    "required": True,
                    "configHash": c["configHash"],
                    "gameBuildHash": c["source"]["gameBuildHash"],
                    "compilerVersion": 1,
                    "visible": False,
                },
            },
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
        ack = {
            "type": "model.context.subscribed",
            "epoch": c["source"]["epoch"],
            "protocol": c["protocol"],
            "segmentId": c["segmentId"],
            "consumer": c["audience"],
            "sampleControlFloor": 1,
        }
        await channel.consume(ack)
        await channel.consume(proposal(window, c))
        assert not received
        await channel.consume(dict(ack, prewarmPlanProtocol=startup_prewarm.PROTOCOL))
        await channel.consume(proposal(window, c))
        assert not received
        view.descriptor["modelContext"]["prewarmPlanProtocol"] = (
            startup_prewarm.PROTOCOL
        )
        await channel.consume(dict(ack, prewarmPlanProtocol=startup_prewarm.PROTOCOL))
        await channel.consume(proposal(window, c))
        assert len(received) == 1 and len(announced) == 1 and channel.sequence == 0
        append_ack = dict(
            ack,
            prewarmPlanProtocol=startup_prewarm.PROTOCOL,
            prewarmAppendProtocol=startup_prewarm.APPEND_PROTOCOL,
        )
        await channel.consume(append_ack)
        assert not channel.prewarm_append_supported
        view.descriptor["modelContext"]["prewarmAppendProtocol"] = (
            startup_prewarm.APPEND_PROTOCOL
        )
        await channel.consume(append_ack)
        assert channel.prewarm_append_supported and window.append_supported
        await channel.consume(dict(append_ack, paused=True))
        assert not channel.prewarm_append_supported
        await channel.pause()
        await channel.consume(proposal(window, c))
        assert len(received) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("replace_formal", [False, "prompt", "lifecycle"])
def test_native_control_prefill_plan_prepares_formal_start_on_original_f0(
    tmp_path, replace_formal
):
    from contextlib import asynccontextmanager

    import msgspec
    from test_whitebox_source_conditions import setup

    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.control_prefill import (
        ControlPrefill,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import LatestFrame
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_conditions import (
        request_identity,
    )

    async def scenario():
        room = object.__new__(SourceSession)
        emitted = []
        borrowed = []
        starts = []

        async def emit(message):
            emitted.append(message)

        async def gate_references(value):
            return await adapter.prepare_references(value, adapter.capabilities())

        def gate(value):
            return window.gate(
                value, emit=emit, resolve=gate_references, check=adapter.check
            )

        async def transition(operation, refs, first):
            starts.append(first)

        async def operation_message(_):
            pass

        adapter, c, room.journal, receipts, _, resources = await setup(
            tmp_path, startup_release=gate
        )
        await room.journal.update(
            lambda s: s.update(inference={"enabled": True, "revision": 7})
        )

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
                self.resource_state = "released"
                self.completed.set()
                return msgspec.msgpack.encode(
                    {
                        "type": "prefill.memory_ready",
                        "request_id": self.request["request_id"],
                        "cache_key": "a" * 64,
                    }
                )

        room.assets = adapter.assets
        room.context_seed = 71
        room.binding_ended = False
        room.session_id = "source"
        room.source_epoch = "epoch"
        room.engine = SimpleNamespace(stream_id="stream")
        room.context_channel = SimpleNamespace(suspended=False)
        room.outgoing = asyncio.Queue()
        room.latest = LatestFrame()
        room.config = SimpleNamespace(startup_seconds=2, spec=VideoSpec(2, 2))
        room.upstream = SimpleNamespace(init={"size": "2x2"})
        room.model_binding = SimpleNamespace(
            assignment=SimpleNamespace(
                denoiser=SimpleNamespace(worker_epoch="epoch", worker_id="worker")
            ),
            borrow=lambda segment, kind: Child(),
        )
        room.control_prefill = ControlPrefill(room)
        room.spawn_control = asyncio.create_task
        room._transition_segment = transition
        room.operation_message = operation_message
        adapter.submit = room.submit_source_condition
        frame = SimpleNamespace(
            index=50,
            rgb=bytes(12),
            provenance={
                "cli_frame_header": {
                    "sessionId": c["source"]["sessionId"],
                    "epoch": c["source"]["epoch"],
                    "source": {**c["source"], **c["effective"]},
                }
            },
        )
        await room.latest.put((frame, 1))
        window = startup_prewarm.StartupPrewarmPlan(
            room, c["segmentId"], 7, asyncio.get_running_loop().time() + 10
        )
        room.inference = SimpleNamespace(prewarm=window)
        adapter.startup_effective = window.effective_for
        resources.set()
        try:
            await adapter.handle(c)
            for _ in range(100):
                if window.task is not None:
                    break
                await asyncio.sleep(0.001)
            value = proposal(window, c)
            value["entries"][0]["prompt"] = c["prompt"]
            window.receive(value)
            await asyncio.wait_for(adapter.preparation, 2)
            assert receipts[-1]["phase"] == "prepared"
            assert (
                len(borrowed) == 3
                and len({request["source_frame"] for request in borrowed}) == 1
            )
            assert len({request["request_id"] for request in borrowed}) == 3
            if replace_formal:
                from sglang.multimodal_gen.runtime.realtime.whitebox.source_conditions import (
                    payload_hash,
                )

                original = copy.deepcopy(c)
                await adapter.handle(
                    request_identity(c)
                    | dict(
                        type="model.condition.withdraw",
                        controlSequence=c["controlSequence"] + 1,
                        reason="state-changed",
                    )
                )
                c = dict(
                    c,
                    conditionId="replacement",
                    prompt="new formal prompt",
                    controlSequence=c["controlSequence"] + 2,
                )
                if replace_formal == "lifecycle":
                    c["lifecycle"] = dict(
                        c["lifecycle"], generation=c["lifecycle"]["generation"] + 1
                    )
                c["payloadHash"] = payload_hash(c)
                await adapter.handle(c)
                await adapter.preparation
                assert receipts[-1]["phase"] == "prepared"
                assert not window.prepared_for(original)
                assert window.prepared_for(c)
                assert len(borrowed) == 3
            await adapter.handle(
                request_identity(c)
                | {
                    "type": "model.condition.commit",
                    "controlSequence": c["controlSequence"] + 1,
                    "effective": receipts[-1].get("startupEffective", c["effective"]),
                }
            )
            for _ in range(100):
                if starts:
                    break
                await asyncio.sleep(0.001)
            if replace_formal == "lifecycle":
                # None delegates a fresh capture to the original transition.
                assert starts == [None]
            else:
                assert starts[0][1][0] is frame
            assert len(getattr(room, "prewarm_plan_consumption", {})) == (
                0 if replace_formal else 1
            )
            assert len(room.prewarm_plan_observations.rows) == 3
        finally:
            await adapter.close()
            await window.close()

    asyncio.run(scenario())


def test_late_persistence_failure_preserves_ready_and_cancellation():
    async def scenario():
        h = await harness(blocked=-1)
        await h.window.released
        h.window.receive(control(h, "cancel", id="item-0", candidateRevision=1))
        h.observers[0](
            {
                "type": "prefill.failed",
                "request_id": "req-0",
                "memory_ready": True,
                "code": "PREFILL_PERSISTENCE_FAILED",
            }
        )
        await h.window.sender
        assert h.messages[-1]["phase"] == "persistence-failed"
        assert h.window.entries["item-0"]["memory_ready"]
        assert h.window.entries["item-0"]["cancelled"]
        assert h.window.history[("item-0", "prepare")]["phase"] == "memory-ready"
        assert h.window.history[("item-0", "intent")]["phase"] == "invalidated"
        await h.window.close()

    asyncio.run(scenario())


def test_dynamic_members_open_independent_plan_windows_only_with_descriptor(
    monkeypatch,
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )

    monkeypatch.setenv("WHITEBOX_MODEL_CONTEXT_PREWARM", "1")

    async def scenario():
        def member(capability):
            source = SimpleNamespace(
                engine=SimpleNamespace(
                    shared_world=True, model_context_capability=capability
                ),
                control_prefill=object(),
                model_binding=object(),
                config=SimpleNamespace(startup_seconds=10),
            )
            return SourceInference(
                source,
                dict(
                    source_operation="create.manual",
                    inference_control="manual",
                    prompt="A multiplayer world",
                ),
            )

        capability = {"prewarmPlanProtocol": startup_prewarm.PROTOCOL}
        a, b, legacy = member(capability), member(capability), member({})
        await a.open_prewarm("segment-a", 1)
        await b.open_prewarm("segment-b", 1)
        await legacy.open_prewarm("segment-legacy", 1)
        assert isinstance(a.prewarm, startup_prewarm.StartupPrewarmPlan)
        assert isinstance(b.prewarm, startup_prewarm.StartupPrewarmPlan)
        assert legacy.prewarm is None
        previous = a.prewarm
        await a.open_prewarm("segment-a", 1)
        assert a.prewarm is previous
        await a.open_prewarm("segment-new", 2)
        assert previous.retired and not b.prewarm.retired
        assert a.prewarm.window_id != b.prewarm.window_id
        await a.prewarm.close()
        await b.prewarm.close()

    asyncio.run(scenario())


def test_descriptor_plan_capability_is_optional_and_strict():
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_conditions import (
        validate_capability,
    )

    c = condition()
    base = {
        "protocol": c["protocol"],
        "required": True,
        "configHash": c["configHash"],
        "gameBuildHash": c["source"]["gameBuildHash"],
        "compilerVersion": 1,
        "visible": False,
    }
    validate_capability(base)
    validate_capability(dict(base, prewarmPlanProtocol=startup_prewarm.PROTOCOL))
    validate_capability(
        dict(
            base,
            prewarmPlanProtocol=startup_prewarm.PROTOCOL,
            prewarmAppendProtocol=startup_prewarm.APPEND_PROTOCOL,
        )
    )
    for append in (True, "wrong", startup_prewarm.APPEND_PROTOCOL):
        with pytest.raises(ValueError):
            validate_capability(dict(base, prewarmAppendProtocol=append))
    with pytest.raises(ValueError):
        validate_capability(dict(base, prewarmPlanProtocol=True))


def test_member_observation_uses_original_member_and_source_scope():
    from sglang.multimodal_gen.runtime.realtime.whitebox.prewarm_observation import (
        dynamic_member_plan_scope,
    )

    checks = []
    member = SimpleNamespace(check=lambda: checks.append(True))
    channel = SimpleNamespace(view=SimpleNamespace(client_id="member-a"))
    session = SimpleNamespace(
        engine=SimpleNamespace(
            shared_world=True,
            model_context_capability={"prewarmPlanProtocol": startup_prewarm.PROTOCOL},
        ),
        inference=SimpleNamespace(manual=True),
        member_prefill=None,
        context_channel=channel,
        source_identity=lambda: {"source_session_id": "source-a", "source_epoch": 1},
    )
    message = {"source_session_id": "source-a", "source_epoch": 1}
    assert dynamic_member_plan_scope(session, member, message) == dict(
        message, consumer_id="member-a"
    )
    assert checks == [True]
    assert (
        dynamic_member_plan_scope(session, member, dict(message, source_epoch=2))
        is None
    )
    session.member_prefill = object()
    assert dynamic_member_plan_scope(session, member, message) is None
    session.member_prefill = None

    def revoked():
        raise RenderError("SOURCE_EPOCH_MISMATCH")

    member.check = revoked
    assert dynamic_member_plan_scope(session, member, message) is None


def test_two_member_plans_keep_frames_and_cleanup_independent():
    async def scenario():
        a = await harness(member="member-a")
        b = await harness(member="member-b", blocked=-1)
        await asyncio.wait_for(a.entered.wait(), 1)
        await asyncio.wait_for(asyncio.shield(b.window.released), 1)
        assert len(a.calls) == 2 and len(b.calls) == 3
        assert a.calls[0][1] is a.calls[1][1]
        assert all(call[1] is b.calls[0][1] for call in b.calls)
        assert a.calls[0][1] is not b.calls[0][1]
        a.window.receive(control(a, "finish"))
        await asyncio.sleep(0)
        assert not a.window.released.done()
        assert not b.window.retired and b.window.first_for(b.c) is b.calls[0][1]
        a.cleanup.set()
        await asyncio.wait_for(asyncio.shield(a.window.released), 1)
        assert len(a.calls) == 2 and len(b.calls) == 3
        await a.window.close()
        assert not b.window.retired
        await b.window.close()

    asyncio.run(scenario())


def test_late_first_plan_receives_closed_results_and_replays_without_compute():
    async def scenario():
        c = condition()
        messages = []

        async def emit(message):
            messages.append(message)

        async def resolve(value):
            raise AssertionError("closed window must not resolve inputs")

        owner = SimpleNamespace(
            spawn_control=asyncio.create_task,
            control_prefill=SimpleNamespace(failure=None),
            context_seed=17,
        )
        window = startup_prewarm.StartupPrewarmPlan(
            owner, c["segmentId"], 1, asyncio.get_running_loop().time() + 1.02
        )
        window.gate(c, emit=emit, resolve=resolve, check=lambda: None)
        await asyncio.wait_for(asyncio.shield(window.released), 1)
        value = proposal(window, c)
        window.receive(value)
        await window.sender
        assert [m["id"] for m in messages if m.get("phase") == "skipped"] == [
            "item-0",
            "item-1",
            "item-2",
        ]
        assert messages[-1]["phase"] == "finished"
        assert messages[-1]["reason"] == "window-closed"
        before = len(messages)
        window.receive(value)
        await window.sender
        assert len(messages) == before + 4
        await window.close()

    asyncio.run(scenario())


def test_member_revocation_fences_compute_and_late_persistence():
    async def scenario():
        h = await harness(blocked=1)
        await h.entered.wait()

        def revoked():
            raise ValueError("member left")

        h.window.check = revoked
        h.observers[0]({"type": "prefill.persisted", "request_id": "req-1"})
        assert not h.window.entries["item-0"]["persisted"]
        assert not h.window.current(h.value["entries"][1])
        h.cleanup.set()
        with pytest.raises(RenderError):
            await asyncio.wait_for(asyncio.shield(h.window.released), 1)
        assert len(h.calls) == 2
        with pytest.raises(RenderError):
            await h.window.close()

    asyncio.run(scenario())


def append_batch(h, revision=2, ids=("new",)):
    message = control(h, "append")
    message.pop("reason")
    message["decisionRevision"] = revision
    message["entries"] = [dict(h.value["entries"][0], id=id) for id in ids]
    return message


@pytest.mark.parametrize("during_compute", [False, True])
def test_append_waits_for_idle_and_preserves_f0_deadline_and_revision(during_compute):
    async def scenario():
        h = await harness(blocked=1 if during_compute else -1, appendable=True)
        deadline = h.window.deadline
        if during_compute:
            await h.entered.wait()
        else:
            while len(h.calls) != 3 or not all(
                i["done"] for i in h.window.entries.values()
            ):
                await asyncio.sleep(0)
            assert not h.window.released.done()
        h.owner.context_seed = 999
        value = append_batch(h)
        h.window.receive(value)
        h.window.receive(value)
        if during_compute:
            h.window.receive(
                dict(
                    control(h, "cancel", id="item-1", candidateRevision=1),
                    decisionRevision=3,
                )
            )
            h.cleanup.set()
        while not h.window.entries["new"]["done"]:
            await asyncio.sleep(0)
        assert len(h.calls) == 4
        assert all(call[1] is h.first for call in h.calls)
        assert all(call[0]["seed"] == 17 for call in h.calls)
        assert h.window.deadline == deadline
        await h.window.sender
        assert any(
            m.get("id") == "new" and m["decisionRevision"] == 2 for m in h.messages
        )
        h.window.announce()
        h.window.receive(h.value)
        h.window.receive(value)
        h.window.receive(dict(control(h, "finish"), decisionRevision=4))
        await h.window.released
        assert len(h.calls) == 4 and h.window.deadline == deadline
        h.window.receive(append_batch(h, 5, ("late",)))
        await h.window.sender
        assert h.messages[-1]["id"] == "late" and h.messages[-1]["phase"] == "skipped"
        assert len(h.calls) == 4
        await h.window.close()

    asyncio.run(scenario())


def test_append_batches_are_atomic_and_replay_conflicts_preserve_queue():
    async def scenario():
        h = await harness(blocked=-1, appendable=True)
        for ids in (
            ("new", "item-0"),
            ("new", "new"),
            tuple(f"new-{i}" for i in range(30)),
        ):
            with pytest.raises(ValueError):
                h.window.receive(append_batch(h, ids=ids))
            assert len(h.window.entries) == 3 and h.window.control_revision == 1
        value = append_batch(h)
        h.window.receive(value)
        conflict = copy.deepcopy(value)
        conflict["entries"][0]["prompt"] = "conflict"
        with pytest.raises(ValueError):
            h.window.receive(conflict)
        assert (
            h.window.entries["new"]["entry"]["prompt"] == value["entries"][0]["prompt"]
        )
        h.window.receive(dict(control(h, "finish"), decisionRevision=3))
        await h.window.released
        await h.window.close()

    asyncio.run(scenario())


def test_appendable_idle_timeout_does_not_extend_budget():
    async def scenario():
        h = await harness(blocked=-1, appendable=True, budget=40)
        deadline = h.window.deadline
        await asyncio.sleep(0.02)
        h.window.receive(append_batch(h))
        await asyncio.wait_for(h.window.released, 0.1)
        assert h.window.deadline == deadline
        h.window.receive(append_batch(h, 3, ("late",)))
        await h.window.sender
        assert h.messages[-1]["id"] == "late" and h.messages[-1]["phase"] == "skipped"
        assert len(h.calls) == 4
        await h.window.close()

    asyncio.run(scenario())


def test_append_requires_negotiation_and_fixed_plan_rejects_append():
    async def scenario():
        h = await harness(blocked=-1)
        with pytest.raises(ValueError):
            h.window.receive(append_batch(h))
        h.window.append_supported = True
        with pytest.raises(ValueError):
            h.window.receive(append_batch(h))
        await h.window.released
        assert len(h.calls) == 3
        await h.window.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("revoke", ["member", "worker"])
def test_append_rechecks_permission_before_adding_entries(revoke):
    async def scenario():
        h = await harness(blocked=-1, appendable=True)
        while not all(item["done"] for item in h.window.entries.values()):
            await asyncio.sleep(0)
        if revoke == "member":

            def check():
                raise RenderError("MODEL_CONTEXT_STALE_SOURCE")

            h.window.check = check
        else:
            h.owner.model_binding.assignment.denoiser.worker_epoch = "changed"
        h.window.receive(append_batch(h))
        assert "new" not in h.window.entries
        with pytest.raises(RenderError):
            await h.window.released
        assert len(h.calls) == 3
        with pytest.raises(RenderError):
            await h.window.close()

    asyncio.run(scenario())


def test_cancelled_entries_still_count_toward_append_history_limit():
    async def scenario():
        h = await harness(blocked=-1, appendable=True)
        h.window.receive(append_batch(h, ids=tuple(f"new-{i}" for i in range(29))))
        h.window.receive(
            dict(
                control(h, "cancel", id="item-0", candidateRevision=1),
                decisionRevision=3,
            )
        )
        with pytest.raises(ValueError):
            h.window.receive(append_batch(h, 4, ("overflow",)))
        assert len(h.window.entries) == 32
        h.window.receive(dict(control(h, "finish"), decisionRevision=4))
        await h.window.released
        await h.window.close()

    asyncio.run(scenario())


def test_stale_lifecycle_rejects_entire_append_batch_without_changing_queue():
    async def scenario():
        h = await harness(blocked=-1, appendable=True)
        async with asyncio.timeout(1):
            while not all(item["done"] for item in h.window.entries.values()):
                await asyncio.sleep(0)
        value = copy.deepcopy(append_batch(h, ids=("valid-new", "stale-new")))
        value["entries"][1]["lifecycle"]["generation"] += 1
        before = copy.deepcopy(h.window.entries)
        with pytest.raises(ValueError, match="MODEL_CONTEXT_STALE_SOURCE"):
            h.window.receive(value)
        assert h.window.entries == before and h.window.control_revision == 1
        assert len(h.calls) == 3
        h.window.receive(append_batch(h, ids=("valid-new",)))
        async with asyncio.timeout(1):
            while not h.window.entries["valid-new"]["done"]:
                await asyncio.sleep(0)
        assert len(h.calls) == 4
        h.window.receive(dict(control(h, "finish"), decisionRevision=3))
        await h.window.released
        await h.window.close()

    asyncio.run(scenario())


def test_appendable_empty_initial_keeps_original_budget_for_later_entries():
    async def scenario():
        h = await harness(blocked=-1, appendable=True, empty_initial=True)
        deadline = h.window.deadline
        await asyncio.sleep(0.01)
        assert not h.window.released.done() and h.calls == []
        value = control(h, "append")
        value.pop("reason")
        value["entries"] = proposal(h.window, h.c)["entries"][:1]
        h.window.receive(value)
        async with asyncio.timeout(1):
            while not h.window.entries["item-0"]["done"]:
                await asyncio.sleep(0)
        assert len(h.calls) == 1 and h.window.deadline == deadline
        h.window.receive(dict(control(h, "finish"), decisionRevision=3))
        await h.window.released
        await h.window.close()

    asyncio.run(scenario())


def test_persist_timeout_replay_retains_memory_ready_and_late_disk_fact():
    from sglang.multimodal_gen.runtime.realtime.whitebox.prewarm_observation import (
        PrewarmPlanObservations,
    )

    async def scenario():
        h = await harness(blocked=-1, wait="persisted", budget=100)
        store = PrewarmPlanObservations(
            source_session_id="source", source_epoch="epoch", consumer_id="consumer"
        )
        store.record(h.window, {"type": startup_prewarm.PREFIX + ".window"})
        h.owner.record_startup_prewarm = lambda window, message, entry=None: (
            store.record(window, message, entry)
        )
        try:
            await h.window.released
            await h.window.sender
            assert len(h.calls) == 1
            h.messages.clear()
            h.window.announce()
            await h.window.sender
            host = [m for m in h.messages if m.get("id") == "item-0"]
            assert {m["phase"] for m in host} >= {"memory-ready", "skipped"}
            assert (
                next(m for m in host if m["phase"] == "memory-ready")["cacheKey"]
                == "a" * 64
            )
            replay = [
                m
                for m in store.replay("reconnected")
                if m.get("candidate_id") == "item-0"
            ]
            assert {m["status"] for m in replay} >= {"memory-ready", "skipped"}
            assert (
                next(m for m in replay if m["status"] == "memory-ready")["cache_key"]
                == "a" * 64
            )
            h.observers[0]({"type": "prefill.persisted", "request_id": "req-0"})
            assert {
                m["status"]
                for m in store.replay("again")
                if m.get("candidate_id") == "item-0"
            } >= {"memory-ready", "skipped", "persisted"}
        finally:
            await h.window.close()

    asyncio.run(scenario())


def test_replaced_formal_condition_keeps_plan_capture_and_owner_fences():
    async def scenario():
        h = await harness(blocked=0)
        await h.entered.wait()
        try:
            original = copy.deepcopy(h.window.plan_condition)
            replacement = dict(h.c, conditionId="next", payloadHash="d" * 64)
            h.window.gate(
                replacement, emit=h.emit, resolve=h.resolve, check=lambda: None
            )
            assert h.window.plan_condition == original
            assert not h.window.prepared_for(replacement)  # child not released
            foreign = copy.deepcopy(replacement)
            foreign["source"]["epoch"] += 1
            with pytest.raises(ValueError, match="MODEL_CONTEXT_STALE_SOURCE"):
                h.window.gate(
                    foreign, emit=h.emit, resolve=h.resolve, check=lambda: None
                )
            h.window.receive(control(h, "finish"))
            h.cleanup.set()
            await h.window.released
            assert h.window.prepared_for(replacement)
            assert not h.window.prepared_for(h.c)
            h.owner.model_binding.assignment.denoiser.worker_epoch = "changed"
            assert not h.window.prepared_for(replacement)
        finally:
            h.cleanup.set()
            await h.window.close()

    asyncio.run(scenario())


def test_expired_window_allows_formal_replacement_without_starting_a_plan():
    async def scenario():
        c = condition()
        owner = SimpleNamespace()
        window = startup_prewarm.StartupPrewarmPlan(
            owner, c["segmentId"], 7, asyncio.get_running_loop().time()
        )

        async def noop(value):
            pass

        assert window.gate(c, emit=noop, resolve=noop, check=lambda: None) is None
        replacement = dict(c, conditionId="new", payloadHash="d" * 64)
        assert (
            window.gate(replacement, emit=noop, resolve=noop, check=lambda: None)
            is None
        )
        assert window.prepared_for(replacement)
        assert not window.prepared_for(c)
        assert window.task is None

    asyncio.run(scenario())
