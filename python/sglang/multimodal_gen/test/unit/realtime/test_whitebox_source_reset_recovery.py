import asyncio
import copy
from types import SimpleNamespace

import msgspec
import pytest
from test_whitebox_render_contract import command
from test_whitebox_source_conditions import setup

from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import RenderError
from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import SourceSession


@pytest.mark.parametrize("resource_state", ["released", "none", "release_unknown"])
@pytest.mark.parametrize("execution", ["pending", "applied", "initial"])
def test_authenticated_reset_retires_execution_without_inventing_application(
    tmp_path, resource_state, execution
):
    async def run():
        adapter, condition, journal, _, _, _ = await setup(tmp_path)
        initial = await journal.model_condition(condition, [])
        sid = initial["segment_id"]
        await journal.update(
            lambda s: s["segment"].update(state="streaming", resource_state="reserved")
        )
        operation = initial
        if execution != "initial":
            await journal.prompt_applied(
                sid, 1, dict(first_output_frame=0, num_frames=1)
            )
            changed = copy.deepcopy(condition)
            changed["conditionId"] = "pending-at-reset"
            changed["onsetOccurrenceIds"] = []
            operation = await journal.model_condition(changed, [])
            if execution == "applied":
                await journal.prompt_applied(
                    sid, 2, dict(first_output_frame=17, num_frames=1)
                )
        owner = object.__new__(SourceSession)
        owner.control_tasks = set()
        owner.journal = journal
        owner.outgoing = asyncio.Queue()
        owner.source_identity = lambda: {}

        async def noop(*a, **k):
            pass

        owner._status = noop
        owner.source_conditions = adapter

        async def stale_receipt(_):
            raise ValueError("MODEL_CONTEXT_STALE_SOURCE")

        adapter.emit = stale_receipt
        entered, release = asyncio.Event(), asyncio.Event()
        segment = SimpleNamespace(
            segment_id=sid,
            segment=copy.deepcopy(journal.snapshot["segment"]),
            publishing=True,
            source_view=SimpleNamespace(stop=lambda: None),
            pending_prompt=copy.deepcopy(journal.snapshot["segment"]["prompt_queue"][0])
            if journal.snapshot["segment"]["prompt_queue"]
            else None,
            resource_state="reserved",
            archive_failure=False,
            model_init_dispatched=True,
            closed_event=asyncio.Event(),
        )
        owner.current_segment = segment

        async def stop(*, cancel):
            assert cancel
            entered.set()
            await release.wait()
            segment.resource_state = resource_state
            await owner.segment_finished(segment, "cancelled", None)
            segment.closed_event.set()

        segment.stop = stop
        reset = dict(
            type="source.reset",
            epoch=condition["source"]["epoch"],
            presentationId="reset-world",
            sdkEpoch=condition["source"]["sdkEpoch"] + 1,
        )
        successor = command("render.context.start", "next", expected_render_revision=1)
        task = asyncio.create_task(
            owner.context_recovery_failed(
                "MODEL_CONTEXT_STALE_SOURCE", source_reset=reset
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert not task.done() and not segment.publishing
            with pytest.raises(RenderError):
                await journal.start_model_context_segment(successor)
            release.set()
            if resource_state == "release_unknown":
                with pytest.raises(RenderError) as failure:
                    await task
                assert failure.value.code == "RELEASE_UNKNOWN"
                assert (
                    "model_context_retirement"
                    not in journal.snapshot["operations"][operation["request_id"]]
                )
            else:
                await task
                old = journal.snapshot["operations"][operation["request_id"]]
                if execution == "applied":
                    assert old["state"] == "applied"
                    assert "model_context_retirement" not in old
                else:
                    assert old["model_context_retirement"]["source_reset"] == reset
                    assert old.get("model_context_execution", {}).get(
                        "state", old["state"]
                    ) in {"unknown", "reconciling"}
                    assert not old.get("never_applied")
                historical = copy.deepcopy(old)
                next_segment = await journal.start_model_context_segment(successor)
                assert next_segment["segment_id"] != sid
                assert (
                    journal.snapshot["operations"][operation["request_id"]]
                    == historical
                )
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await adapter.close()

    asyncio.run(run())


def test_source_reset_close_waits_for_cleanup_not_preview_transport():
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
        CLIConditionChannel,
    )

    async def run():
        preview_entered, cleanup_entered = asyncio.Event(), asyncio.Event()
        preview_release, cleanup_release = asyncio.Event(), asyncio.Event()

        async def send(_):
            preview_entered.set()
            await preview_release.wait()

        async def cleanup(*args, **kwargs):
            cleanup_entered.set()
            await cleanup_release.wait()

        channel = object.__new__(CLIConditionChannel)
        channel.__dict__.update(
            view=SimpleNamespace(
                control=SimpleNamespace(send=send),
                descriptor={"epoch": 0},
                transport_generation=0,
            ),
            prewarm_window=lambda: None,
            fail_interventions=lambda: None,
            discard_controls=lambda: None,
            on_recovery_failed=cleanup,
            reset_task=None,
            adapter=None,
            consumer=None,
            source_generation=0,
            sequence=0,
            samples={},
            retired_samples=set(),
            snapshots={},
            changed=asyncio.Condition(),
            closed=False,
            bound=asyncio.Event(),
            subscription_probe_ready=asyncio.Event(),
            task=None,
        )
        channel.retire_source(
            dict(type="source.reset", epoch=0, presentationId="new", sdkEpoch=1)
        )
        await asyncio.wait_for(preview_entered.wait(), 1)
        await asyncio.wait_for(cleanup_entered.wait(), 1)
        closing = asyncio.create_task(channel.close())
        try:
            await asyncio.sleep(0)
            assert not closing.done(), "Original cleanup must remain awaited"
            cleanup_release.set()
            await asyncio.wait_for(asyncio.shield(closing), 1)
            assert channel.reset_task.done()
            assert not preview_release.is_set()
        finally:
            cleanup_release.set()
            preview_release.set()
            await asyncio.gather(closing, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("fault", ["epoch", "sdkEpoch", "segment"])
def test_retirement_cannot_bypass_wrong_source_or_original_guard(tmp_path, fault):
    async def run():
        adapter, condition, journal, _, _, _ = await setup(tmp_path)
        initial = await journal.model_condition(condition, [])
        sid = initial["segment_id"]
        await journal.update(
            lambda s: s["segment"].update(state="cancelled", resource_state="released")
        )
        reset = dict(
            type="source.reset",
            epoch=condition["source"]["epoch"],
            presentationId="new",
            sdkEpoch=condition["source"]["sdkEpoch"] + 1,
        )
        if fault == "epoch":
            reset["epoch"] += 1
        if fault == "sdkEpoch":
            reset["sdkEpoch"] = condition["source"]["sdkEpoch"]
        before = copy.deepcopy(await journal.read())
        with pytest.raises(RenderError):
            await journal.start_model_context_segment(
                command("render.context.start", "blocked", expected_render_revision=1)
            )
        with pytest.raises(RenderError):
            await journal.retire_model_context_source(
                "wrong-segment" if fault == "segment" else sid, reset
            )
        assert await journal.read() == before
        await adapter.close()

    asyncio.run(run())


@pytest.mark.parametrize("prior_release_unknown", [False, True])
def test_native_reset_retires_preparing_window_before_source_fence(
    prior_release_unknown,
):
    from test_whitebox_native import native_frame

    from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
        CLIConditionChannel,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.startup_prewarm_plan import (
        StartupPrewarmPlan,
    )

    async def run():
        _, descriptor, _ = native_frame()
        from test_whitebox_source_conditions import condition

        c = condition()
        descriptor["modelContext"] = dict(
            protocol=c["protocol"],
            required=True,
            configHash=c["configHash"],
            gameBuildHash=c["source"]["gameBuildHash"],
            compilerVersion=1,
            visible=False,
        )
        channel = CLIConditionChannel(
            SimpleNamespace(descriptor=descriptor, transport_generation=1)
        )
        window = StartupPrewarmPlan(
            SimpleNamespace(), "segment", 1, asyncio.get_running_loop().time() + 10
        )
        window.binding = window.binding_identity()
        if prior_release_unknown:
            window.failure = RenderError("RELEASE_UNKNOWN")
        emitted = []

        async def emit(message):
            emitted.append(message)
            if channel.source_generation:
                raise ValueError("MODEL_CONTEXT_STALE_SOURCE")

        def check():
            if channel.source_generation:
                raise ValueError("MODEL_CONTEXT_STALE_SOURCE")

        async def close_adapter():
            pass

        window.emit, window.check = emit, check
        window.outbound.append({"type": "old-prewarm-feedback"})
        channel.adapter = SimpleNamespace(
            startup_window=lambda: window, close=close_adapter
        )
        channel.prewarm_plan_supported = True
        previews = []

        async def send_preview(value):
            previews.append(value)

        channel.view.control = SimpleNamespace(send=send_preview)
        channel.retire_source(
            dict(type="source.reset", epoch=1, presentationId="new", sdkEpoch=2)
        )
        try:
            assert window.retired, (
                "reset must retire pending prewarm before invalidating its writer"
            )
            assert not window.trusted()
            await window.flush()
            assert emitted == []
            if prior_release_unknown:
                with pytest.raises(RenderError, match="cleaned up"):
                    await window.close()
                assert window.failure.code == "RELEASE_UNKNOWN"
            else:
                await window.close()
                assert window.failure is None
        finally:
            await channel.reset_task

    asyncio.run(run())


@pytest.mark.parametrize("release_unknown", [False, True])
def test_reset_during_preparation_stops_opening_before_retry(tmp_path, release_unknown):
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )

    async def run():
        adapter, _, journal, _, _, _ = await setup(tmp_path)
        await journal.update(
            lambda s: s.update(
                inference=dict(enabled=True, revision=1, state="starting")
            )
        )
        source = object.__new__(SourceSession)
        source.control_tasks = set()
        source.journal, source.current_segment, source.source_conditions = (
            journal,
            None,
            adapter,
        )
        source.outgoing = asyncio.Queue()
        source.source_identity = lambda: {}
        source.current_model_stream, source.display_mode = None, "model"
        source.context_channel = None

        async def noop(*a, **k):
            pass

        source._status = source.status_snapshot = noop
        inference = object.__new__(SourceInference)
        source.inference = inference
        inference.source, inference.manual = source, True
        inference.lock = asyncio.Lock()
        inference.prewarm = SimpleNamespace(invalidate=lambda: None)
        inference.task = asyncio.create_task(asyncio.Event().wait())
        cleaned = []

        async def disable(*, reset_generation=None, expected_revision=None):
            assert inference.task.done()
            cleaned.append(True)
            if release_unknown:
                raise RenderError("RELEASE_UNKNOWN")

        inference.disable = disable
        reset = dict(type="source.reset", epoch=1, presentationId="new", sdkEpoch=2)
        try:
            if release_unknown:
                with pytest.raises(RenderError) as error:
                    await source.context_recovery_failed(
                        "MODEL_CONTEXT_STALE_SOURCE", source_reset=reset
                    )
                assert error.value.code == "RELEASE_UNKNOWN"
            else:
                await source.context_recovery_failed(
                    "MODEL_CONTEXT_STALE_SOURCE", source_reset=reset
                )
                assert journal.snapshot["inference"]["state"] == "failed"
                assert (
                    journal.snapshot["inference"]["code"]
                    == "MODEL_CONTEXT_STALE_SOURCE"
                )
            assert cleaned
        finally:
            inference.task.cancel()
            await asyncio.gather(inference.task, return_exceptions=True)
            await adapter.close()

    asyncio.run(run())


@pytest.mark.parametrize("outgoing_full", [False, True])
@pytest.mark.parametrize("prewarm_enabled", [False, True])
@pytest.mark.parametrize("release_unknown", [False, True])
def test_reset_before_binding_ready_retires_opening(
    tmp_path, monkeypatch, prewarm_enabled, release_unknown, outgoing_full
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )

    monkeypatch.setenv(
        "WHITEBOX_MODEL_CONTEXT_PREWARM", "1" if prewarm_enabled else "0"
    )

    async def run():
        adapter, c, journal, *_ = await setup(tmp_path)
        await journal.update(
            lambda s: s.update(
                inference=dict(enabled=False, revision=0, state="disabled")
            )
        )
        operation = await journal.configure_inference(
            command(
                "model.inference.configure",
                "opening",
                enabled=True,
                expected_inference_revision=0,
            )
        )
        source = object.__new__(SourceSession)
        source.journal, source.current_segment, source.source_conditions = (
            journal,
            None,
            adapter,
        )
        source.closed = source.binding_ended = False
        source.outgoing, source.segment_lock = asyncio.Queue(maxsize=2), asyncio.Lock()
        source.control_tasks = set()
        if outgoing_full:
            source.outgoing.put_nowait(b"held-one")
            source.outgoing.put_nowait(b"held-two")
        source.source_identity = dict
        source.current_model_stream, source.display_mode = None, "model"
        source.control_prefill = source.startup = None
        entered, ready = asyncio.Event(), asyncio.Event()
        calls = []

        async def noop(*a, **k):
            pass

        async def wait_ready():
            entered.set()
            await ready.wait()

        async def close():
            source.model_binding.lease.resource_state = (
                "release_unknown" if release_unknown else "released"
            )

        async def switch(_):
            calls.append("switch")

        source._status = source.status_snapshot = noop
        source.config = SimpleNamespace(startup_seconds=1)
        source.model_binding = SimpleNamespace(
            wait_ready=wait_ready,
            close=close,
            lease=SimpleNamespace(resource_state="reserved"),
            closed=False,
        )
        source.context_channel = SimpleNamespace(
            source_generation=0,
            reset_task=None,
            probe_subscription_pause=noop,
            pause=noop,
            switch_segment=switch,
        )
        original_binding = source.model_binding
        inference = object.__new__(SourceInference)
        source.inference = inference
        inference.source, inference.manual = source, True
        inference.prewarm = inference.task = inference.assets = inference.startup = None
        inference.lock, inference.frozen = asyncio.Lock(), {}
        inference.dispatch(operation)
        original = inference.task
        await asyncio.wait_for(entered.wait(), 1)
        source.context_channel.source_generation += 1
        reset = dict(
            type="source.reset",
            epoch=c["source"]["epoch"],
            presentationId="new",
            sdkEpoch=c["source"]["sdkEpoch"] + 1,
        )
        try:
            if release_unknown:
                settling, allow_settlement = asyncio.Event(), asyncio.Event()
                settle = journal.settle_inference

                async def delayed_settlement(*args, **kwargs):
                    settling.set()
                    await allow_settlement.wait()
                    return await settle(*args, **kwargs)

                journal.settle_inference = delayed_settlement

                async def blocked_pause():
                    await asyncio.Event().wait()

                inference.pause_channel = blocked_pause
                retirement = asyncio.create_task(
                    source.context_recovery_failed(
                        "MODEL_CONTEXT_STALE_SOURCE", source_reset=reset
                    )
                )
                try:
                    await asyncio.wait_for(settling.wait(), 1)
                    # Source teardown waits for retirement before cancelling its
                    # control tasks. Durable settlement must stay inside that wait.
                    assert not retirement.done()
                    allow_settlement.set()
                    with pytest.raises(RenderError) as error:
                        await retirement
                    assert error.value.code == "RELEASE_UNKNOWN"
                    assert (
                        journal.snapshot["operations"]["opening"]["state"] == "failed"
                    )
                finally:
                    allow_settlement.set()
                    retirement.cancel()
                    await asyncio.gather(retirement, return_exceptions=True)
            else:
                await source.context_recovery_failed(
                    "MODEL_CONTEXT_STALE_SOURCE", source_reset=reset
                )
                assert journal.snapshot["inference"]["state"] == "failed"
            assert original.done()
            # Cleanup and durable control completion cannot depend on an outgoing
            # reader. The old model application/release evidence stays separate.
            expected_code = (
                "RELEASE_UNKNOWN" if release_unknown else "MODEL_CONTEXT_STALE_SOURCE"
            )
            async with asyncio.timeout(1):
                while journal.snapshot["operations"]["opening"]["state"] == "accepted":
                    await asyncio.sleep(0)
            retired = journal.snapshot["operations"]["opening"]
            assert retired["state"] == "failed" and retired["code"] == expected_code
            assert journal.snapshot["inference"]["state"] == "failed"
            assert journal.snapshot["inference"]["code"] == expected_code
            assert original_binding.lease.resource_state == (
                "release_unknown" if release_unknown else "released"
            )
            if outgoing_full:
                assert source.outgoing.full()
                assert source.outgoing.get_nowait() == b"held-one"
                assert source.outgoing.get_nowait() == b"held-two"
            async with asyncio.timeout(1):
                while True:
                    receipt = msgspec.msgpack.decode(await source.outgoing.get())
                    if receipt.get("request_id") == "opening":
                        break
            assert receipt["type"] == "operation.result"
            assert receipt["state"] == "failed" and receipt["code"] == expected_code
            assert receipt["inference"] == journal.snapshot["inference"]
            ready.set()
            assert calls == []
        finally:
            ready.set()
            for task in list(source.control_tasks):
                task.cancel()
            await asyncio.gather(*list(source.control_tasks), return_exceptions=True)
            original.cancel()
            await asyncio.gather(original, return_exceptions=True)
            await adapter.close()

    asyncio.run(run())


@pytest.mark.parametrize("condition_admitted", [False, True])
def test_reset_retires_admitted_opening_not_old_closed_segment(
    tmp_path, condition_admitted
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )

    async def run():
        adapter, c, journal, *_ = await setup(tmp_path)
        old = await journal.model_condition(c, [])
        await journal.update(
            lambda s: s["segment"].update(state="streaming", resource_state="reserved")
        )
        await journal.prompt_applied(
            old["segment_id"], 1, dict(first_output_frame=0, num_frames=1)
        )
        await journal.update(
            lambda s: s["segment"].update(state="cancelled", resource_state="released")
        )
        new = await journal.start_model_context_segment(
            command("render.context.start", "successor", expected_render_revision=1)
        )
        next_condition = copy.deepcopy(c)
        next_condition.update(segmentId=new["segment_id"], conditionId="new-opening")
        opening = (
            await journal.model_condition(next_condition, [])
            if condition_admitted
            else None
        )
        await journal.update(
            lambda s: s.update(
                inference=dict(enabled=True, revision=1, state="starting")
            )
        )
        source = object.__new__(SourceSession)
        source.control_tasks = set()
        source.journal, source.source_conditions = journal, adapter
        source.outgoing, source.segment_lock = asyncio.Queue(), asyncio.Lock()
        source.source_identity = dict
        source.current_model_stream, source.display_mode = None, "model"
        source.context_channel = source.model_binding = source.control_prefill = (
            source.startup
        ) = None

        async def noop(*a, **k):
            pass

        source._status = source.status_snapshot = noop
        closed = asyncio.Event()
        closed.set()
        source.current_segment = SimpleNamespace(
            segment_id=old["segment_id"],
            closed_event=closed,
            resource_state="released",
            publishing=False,
            source_view=SimpleNamespace(stop=lambda: None),
            stop=noop,
        )
        inference = object.__new__(SourceInference)
        source.inference = inference
        inference.source, inference.manual = source, True
        inference.prewarm = None
        inference.task = asyncio.create_task(asyncio.Event().wait())
        inference.lock = asyncio.Lock()
        reset = dict(
            type="source.reset",
            epoch=c["source"]["epoch"],
            presentationId="new",
            sdkEpoch=c["source"]["sdkEpoch"] + 1,
        )
        try:
            await source.context_recovery_failed(
                "MODEL_CONTEXT_STALE_SOURCE", source_reset=reset
            )
            assert inference.task.done()
            assert journal.snapshot["inference"]["state"] == "failed"
            current = journal.snapshot["segment"]
            if condition_admitted:
                assert current["segment_id"] == opening["segment_id"]
                assert (
                    current["state"] == "cancelled"
                    and current["resource_state"] == "none"
                )
                assert journal.snapshot["operations"][opening["request_id"]][
                    "never_applied"
                ]
            else:
                assert current["segment_id"] == old["segment_id"]
                source.closed = source.binding_ended = False
                source.config = SimpleNamespace(startup_seconds=1)
                source.model_binding = SimpleNamespace(
                    wait_ready=noop,
                    lease=SimpleNamespace(resource_state="released"),
                    closed=False,
                )
                switched = []

                async def switch(segment_id):
                    switched.append(segment_id)
                    ready = asyncio.Event()
                    ready.set()
                    source.current_segment = SimpleNamespace(
                        segment_id=segment_id,
                        closed_event=asyncio.Event(),
                        model_ready=ready,
                    )

                source.context_channel = SimpleNamespace(
                    probe_subscription_pause=noop, switch_segment=switch
                )
                retry = await journal.configure_inference(
                    command(
                        "model.inference.configure",
                        "explicit-retry",
                        enabled=True,
                        expected_inference_revision=1,
                    )
                )
                await journal.update(
                    lambda s: s["segment"].update(resource_state="release_unknown")
                )
                with pytest.raises(RenderError) as error:
                    await inference.enable(retry)
                assert error.value.code == "RELEASE_UNKNOWN" and switched == []
                await journal.update(
                    lambda s: s["segment"].update(resource_state="released")
                )
                await inference.enable(retry)
                assert switched == [new["segment_id"]]
            assert (
                journal.snapshot["operations"][old["request_id"]]["state"] == "applied"
            )
        finally:
            inference.task.cancel()
            await asyncio.gather(inference.task, return_exceptions=True)
            await adapter.close()

    asyncio.run(run())


def test_new_disable_waits_for_reset_cleanup_without_locking_it(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )

    async def run():
        adapter, c, journal, *_ = await setup(tmp_path)
        await journal.update(
            lambda s: s.update(
                inference=dict(enabled=False, revision=0, state="disabled")
            )
        )
        source = object.__new__(SourceSession)
        source.journal, source.source_conditions = journal, adapter
        source.current_segment = source.startup = source.control_prefill = None
        source.closed = source.binding_ended = False
        source.current_model_stream, source.display_mode = None, "model"
        source.outgoing, source.segment_lock = asyncio.Queue(), asyncio.Lock()
        source.source_identity, source.spawn_control = dict, asyncio.create_task
        entered, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def noop(*a, **k):
            pass

        async def wait_ready():
            entered.set()
            await asyncio.Event().wait()

        async def close():
            closing.set()
            await release.wait()
            source.model_binding.lease.resource_state = "released"

        source._status = source.status_snapshot = noop
        source.config = SimpleNamespace(startup_seconds=1)
        source.model_binding = SimpleNamespace(
            wait_ready=wait_ready,
            close=close,
            lease=SimpleNamespace(resource_state="reserved"),
            closed=False,
        )
        channel = source.context_channel = SimpleNamespace(
            source_generation=0,
            reset_task=None,
            probe_subscription_pause=noop,
            pause=noop,
        )
        inference = source.inference = object.__new__(SourceInference)
        inference.source, inference.manual = source, True
        inference.prewarm = inference.task = inference.assets = inference.startup = None
        inference.lock, inference.frozen = asyncio.Lock(), {}
        first = await journal.configure_inference(
            command(
                "model.inference.configure",
                "first-enable",
                enabled=True,
                expected_inference_revision=0,
            )
        )
        inference.dispatch(first)
        old_task = inference.task
        await asyncio.wait_for(entered.wait(), 1)
        channel.source_generation = 1
        reset = dict(
            type="source.reset",
            epoch=c["source"]["epoch"],
            presentationId="new",
            sdkEpoch=c["source"]["sdkEpoch"] + 1,
        )
        channel.reset_task = asyncio.create_task(
            source.context_recovery_failed(
                "MODEL_CONTEXT_STALE_SOURCE", source_reset=reset
            )
        )
        try:
            await asyncio.wait_for(closing.wait(), 1)
            disable = await journal.configure_inference(
                command(
                    "model.inference.configure",
                    "new-disable",
                    enabled=False,
                    expected_inference_revision=1,
                )
            )
            inference.dispatch(disable)
            successor = inference.task
            await asyncio.sleep(0)
            assert not successor.done()
            release.set()
            await asyncio.wait_for(asyncio.gather(channel.reset_task, successor), 1)
            assert old_task.cancelled() and not successor.cancelled()
            assert journal.snapshot["inference"]["revision"] == 2
            assert journal.snapshot["inference"]["state"] == "disabled"
            assert (
                journal.snapshot["operations"][disable["request_id"]]["state"]
                == "completed"
            )
        finally:
            release.set()
            inference.task.cancel()
            channel.reset_task.cancel()
            await asyncio.gather(
                inference.task, channel.reset_task, return_exceptions=True
            )
            await adapter.close()

    asyncio.run(run())


def test_original_disable_can_cleanup_after_failed_source_retirement(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )

    async def run():
        adapter, _, journal, *_ = await setup(tmp_path)
        await journal.update(
            lambda s: s.update(inference=dict(enabled=True, revision=1, state="failed"))
        )
        operation = await journal.configure_inference(
            command(
                "model.inference.configure",
                "stop-after-failed-reset",
                enabled=False,
                expected_inference_revision=1,
            )
        )
        failed = asyncio.get_running_loop().create_future()
        failed.set_exception(RenderError("RELEASE_UNKNOWN"))
        failed.exception()
        source = SimpleNamespace(
            journal=journal,
            context_channel=SimpleNamespace(reset_task=failed),
            spawn_control=asyncio.create_task,
        )
        calls = []

        async def disable(**kwargs):
            calls.append("original-cleanup")

        async def noop(*args):
            pass

        source.operation_message = source.status_snapshot = noop
        owner = object.__new__(SourceInference)
        owner.source, owner.task, owner.lock = source, None, asyncio.Lock()
        owner.disable, owner.fence = disable, lambda: None
        try:
            owner.dispatch(operation)
            await owner.task
            assert calls == ["original-cleanup"]
            assert journal.snapshot["inference"]["state"] == "disabled"
            assert (
                journal.snapshot["operations"][operation["request_id"]]["state"]
                == "completed"
            )
            assert (
                source.context_channel.reset_task.exception().code == "RELEASE_UNKNOWN"
            )
        finally:
            await adapter.close()

    asyncio.run(run())


@pytest.mark.parametrize("dispatch_before_cleanup", [False, True])
def test_reset_cleanup_preserves_newer_explicit_enable_display(
    tmp_path, dispatch_before_cleanup
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        InferenceTaskScope,
        SourceInference,
    )

    async def run():
        adapter, c, journal, *_ = await setup(tmp_path)
        await journal.update(
            lambda s: s.update(
                inference=dict(enabled=True, revision=1, state="failed"),
                display_mode="whitebox",
            )
        )
        source = object.__new__(SourceSession)
        source.journal, source.current_segment, source.source_conditions = (
            journal,
            None,
            adapter,
        )
        source.closed = source.binding_ended = False
        source.outgoing, source.segment_lock = asyncio.Queue(), asyncio.Lock()
        source.source_identity, source.spawn_control = dict, asyncio.create_task
        source.current_model_stream, source.display_mode = None, "whitebox"
        source.control_prefill = source.startup = None
        source.config = SimpleNamespace(startup_seconds=1)

        async def noop(*a, **k):
            pass

        source._status = source.status_snapshot = source.operation_message = noop

        def binding(*args):
            async def close():
                source.model_binding.lease.resource_state = "released"

            source.model_binding = SimpleNamespace(
                wait_ready=noop,
                close=close,
                lease=SimpleNamespace(resource_state="reserved"),
                closed=False,
            )

        source.start_model_binding = binding
        binding()

        async def switch(segment_id):
            ready = asyncio.Event()
            ready.set()
            source.current_segment = SimpleNamespace(
                segment_id=segment_id, closed_event=asyncio.Event(), model_ready=ready
            )

        channel = source.context_channel = SimpleNamespace(
            source_generation=1,
            reset_task=None,
            probe_subscription_pause=noop,
            pause=noop,
            switch_segment=switch,
        )
        inference = source.inference = object.__new__(SourceInference)
        inference.source, inference.manual = source, True
        inference.prewarm = inference.task = inference.assets = inference.startup = None
        inference.lock, inference.frozen = asyncio.Lock(), {}
        op = await journal.configure_inference(
            command(
                "model.inference.configure",
                "new-explicit-enable",
                enabled=True,
                expected_inference_revision=1,
            )
        )
        source.display_mode = journal.snapshot["display_mode"]
        reset = dict(
            type="source.reset",
            epoch=c["source"]["epoch"],
            presentationId="new",
            sdkEpoch=c["source"]["sdkEpoch"] + 1,
        )
        channel.reset_task = asyncio.create_task(
            source.context_recovery_failed(
                "MODEL_CONTEXT_STALE_SOURCE", source_reset=reset
            )
        )
        if dispatch_before_cleanup:
            inference.dispatch(op)
        else:
            inference.task_scope = InferenceTaskScope(None, 1, 0, "old-enable")
            await channel.reset_task
            inference.dispatch(op)
        try:
            await asyncio.wait_for(
                asyncio.gather(channel.reset_task, inference.task), 1
            )
            assert journal.snapshot["inference"]["state"] == "enabled"
            assert journal.snapshot["inference"]["revision"] == 2
            assert journal.snapshot["display_mode"] == source.display_mode == "model"
        finally:
            inference.task.cancel()
            channel.reset_task.cancel()
            await asyncio.gather(
                inference.task, channel.reset_task, return_exceptions=True
            )
            await adapter.close()

    asyncio.run(run())


def test_reset_between_configure_and_dispatch_rejects_old_enable(tmp_path):
    async def run():
        adapter, _, journal, *_ = await setup(tmp_path)
        await journal.update(
            lambda s: s.update(
                inference=dict(enabled=False, revision=0, state="disabled")
            )
        )
        source = object.__new__(SourceSession)
        source.journal, source.startup = journal, None
        source.context_channel = SimpleNamespace(source_generation=0)
        source.outgoing = asyncio.Queue()
        source.require_opening_presentation_player = lambda: None
        entered, release = asyncio.Event(), asyncio.Event()
        dispatched = []
        source.inference = SimpleNamespace(manual=True, dispatch=dispatched.append)

        async def noop(*args):
            pass

        async def status():
            if not entered.is_set():
                entered.set()
                await release.wait()

        source.operation_message, source.status_snapshot = noop, status
        request = command(
            "model.inference.configure",
            "accepted-before-reset",
            enabled=True,
            expected_inference_revision=0,
        )
        task = asyncio.create_task(source.command(request))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            source.context_channel.source_generation = 1
            release.set()
            await asyncio.wait_for(task, 1)
            assert dispatched == []
            op = journal.snapshot["operations"][request["request_id"]]
            assert (
                op["state"] == "failed" and op["code"] == "MODEL_CONTEXT_STALE_SOURCE"
            )
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await adapter.close()

    asyncio.run(run())


@pytest.mark.parametrize("release_unknown", [False, True])
def test_reset_fences_binding_while_preview_send_is_blocked(tmp_path, release_unknown):
    import json

    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
        CLIConditionChannel,
    )

    async def run():
        adapter, c, journal, *_ = await setup(tmp_path)
        await journal.update(
            lambda s: s.update(
                inference=dict(enabled=False, revision=0, state="disabled")
            )
        )
        source = object.__new__(SourceSession)
        source.journal, source.current_segment, source.source_conditions = (
            journal,
            None,
            adapter,
        )
        source.closed = source.binding_ended = False
        source.outgoing, source.segment_lock = asyncio.Queue(), asyncio.Lock()
        source.source_identity, source.spawn_control = dict, asyncio.create_task
        source.current_model_stream, source.display_mode = None, "model"
        source.control_prefill = source.startup = None
        source.config = SimpleNamespace(startup_seconds=5)
        entered, ready, preview_started, preview_resume = [
            asyncio.Event() for _ in range(4)
        ]
        cleanup_entered, cleanup_resume, lease_closed, subscribed = [
            asyncio.Event() for _ in range(4)
        ]

        async def noop(*a, **k):
            pass

        async def wait_ready():
            entered.set()
            await ready.wait()

        lease = SimpleNamespace(resource_state="reserved")
        binding = SimpleNamespace(wait_ready=wait_ready, lease=lease, closed=False)

        async def close():
            binding.closed = True
            lease.resource_state = "release_unknown" if release_unknown else "released"
            lease_closed.set()

        binding.close = close
        source.model_binding = binding
        source._status = source.status_snapshot = noop
        messages = []

        async def send(raw):
            message = json.loads(raw)
            messages.append(message["type"])
            if message["type"] == "model.context.preview":
                preview_started.set()
                await preview_resume.wait()
            elif message["type"] == "model.context.subscribe":
                subscribed.set()
                model_ready = asyncio.Event()
                model_ready.set()
                source.current_segment = SimpleNamespace(
                    segment_id=message["segmentId"],
                    closed_event=asyncio.Event(),
                    model_ready=model_ready,
                )

        descriptor = dict(
            epoch=c["source"]["epoch"],
            modelContext=dict(
                protocol=c["protocol"],
                required=True,
                configHash=c["configHash"],
                gameBuildHash=c["source"]["gameBuildHash"],
                compilerVersion=1,
                visible=False,
            ),
        )
        channel = source.context_channel = CLIConditionChannel(
            SimpleNamespace(
                descriptor=descriptor,
                transport_generation=1,
                control=SimpleNamespace(send=send),
                closed=False,
                stopped=False,
                recover_model_context_transport=noop,
            )
        )
        channel.adapter, channel.probe_subscription_pause = adapter, noop

        async def recover(code, **kwargs):
            cleanup_entered.set()
            await cleanup_resume.wait()
            await source.context_recovery_failed(code, **kwargs)

        channel.on_recovery_failed = recover
        inference = source.inference = object.__new__(SourceInference)
        inference.source, inference.manual = source, True
        inference.prewarm = inference.task = inference.assets = inference.startup = None
        inference.lock, inference.frozen = asyncio.Lock(), {}
        operation = await journal.configure_inference(
            command(
                "model.inference.configure",
                "old-opening",
                enabled=True,
                expected_inference_revision=0,
            )
        )
        inference.dispatch(operation)
        old = inference.task
        await asyncio.wait_for(entered.wait(), 1)
        channel.retire_source(
            dict(
                type="source.reset",
                epoch=c["source"]["epoch"],
                presentationId="new",
                sdkEpoch=c["source"]["sdkEpoch"] + 1,
            )
        )
        waits = []
        try:
            await asyncio.wait_for(preview_started.wait(), 1)
            ready.set()
            waits = [
                asyncio.create_task(lease_closed.wait()),
                asyncio.create_task(subscribed.wait()),
            ]
            await asyncio.wait_for(
                asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED), 1
            )
            assert not subscribed.is_set(), (
                "old opening must not resubscribe after reset"
            )
            assert cleanup_entered.is_set(), "preview send must not gate owner cleanup"
            assert lease_closed.is_set() and not preview_resume.is_set()
            cleanup_resume.set()
            await asyncio.wait_for(
                asyncio.gather(old, channel.reset_task, return_exceptions=True), 1
            )
            assert not preview_resume.is_set(), "retirement must not wait for preview"
            original = journal.snapshot["operations"]["old-opening"]
            assert original["state"] == "failed"
            assert original["code"] == (
                "RELEASE_UNKNOWN" if release_unknown else "SOURCE_EPOCH_MISMATCH"
            )
            async with asyncio.timeout(1):
                while True:
                    receipt = msgspec.msgpack.decode(await source.outgoing.get())
                    if receipt.get("request_id") == "old-opening":
                        break
            assert receipt["type"] == "operation.result"
            assert receipt["state"] == "failed" and receipt["code"] == original["code"]
            assert "model.context.subscribe" not in messages and channel.suspended
            assert journal.snapshot["inference"]["state"] == "failed"
            await asyncio.wait_for(channel.close(), 1)
            assert not preview_resume.is_set(), "close must not wait for preview"
            if release_unknown:
                retry = await journal.configure_inference(
                    command(
                        "model.inference.configure",
                        "retry-unknown-release",
                        enabled=True,
                        expected_inference_revision=1,
                    )
                )
                with pytest.raises(RenderError) as error:
                    await inference.enable(retry)
                assert error.value.code == "RELEASE_UNKNOWN"
            else:
                assert lease.resource_state == "released"
        finally:
            cleanup_resume.set()
            preview_resume.set()
            ready.set()
            for task in [old, channel.reset_task, *waits]:
                task.cancel()
            await asyncio.gather(
                old, channel.reset_task, *waits, return_exceptions=True
            )
            await adapter.close()

    asyncio.run(run())


@pytest.mark.parametrize("release_unknown", [False, True])
def test_reset_then_close_releases_without_draining_outgoing(tmp_path, release_unknown):
    from collections import deque
    from unittest.mock import patch

    import msgspec

    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        SourceInference,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import (
        InteractiveSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        RoomPipeline,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
        CLIConditionChannel,
    )

    async def run():
        adapter, c, journal, *_ = await setup(tmp_path)
        await journal.update(
            lambda state: state.update(
                inference=dict(enabled=False, revision=0, state="disabled")
            )
        )
        operation = await journal.configure_inference(
            command(
                "model.inference.configure",
                "opening",
                enabled=True,
                expected_inference_revision=0,
            )
        )
        source = object.__new__(SourceSession)
        source.journal, source.current_segment, source.source_conditions = (
            journal,
            None,
            adapter,
        )
        source.closed = source.binding_ended = False
        source.outgoing = asyncio.Queue(maxsize=2)
        source.segment_lock = asyncio.Lock()
        source.source_identity = dict
        source.current_model_stream, source.display_mode = None, "model"
        source.control_prefill = source.startup = source.assets = None
        source.tasks, source.control_tasks = set(), set()
        source.changed = asyncio.Event()
        source.outcome, source.session_id = "cancelled", "source"
        source.retained = SimpleNamespace(clear=lambda: None)
        source._source_close_task = None
        source.config = SimpleNamespace(startup_seconds=120)
        entered = asyncio.Event()

        async def noop(*args, **kwargs):
            pass

        async def wait_ready():
            entered.set()
            await asyncio.Event().wait()

        lease = SimpleNamespace(resource_state="reserved", _active_assignment=None)

        async def release():
            lease.resource_state = "release_unknown" if release_unknown else "released"
            if release_unknown:
                raise RenderError("RELEASE_UNKNOWN")

        source.model_binding = SimpleNamespace(
            wait_ready=wait_ready,
            close=release,
            lease=lease,
            closed=False,
            assignment=None,
        )
        source.status_snapshot = source.operation_message = noop
        channel = source.context_channel = CLIConditionChannel(
            SimpleNamespace(
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
                transport_generation=1,
                control=SimpleNamespace(send=noop),
                closed=False,
                stopped=False,
                recover_model_context_transport=noop,
            )
        )
        channel.adapter, channel.probe_subscription_pause = adapter, noop
        channel.on_recovery_failed = source.context_recovery_failed
        inference = source.inference = object.__new__(SourceInference)
        inference.source, inference.manual = source, True
        inference.prewarm = inference.task = inference.assets = inference.startup = None
        inference.lock, inference.frozen = asyncio.Lock(), {}
        inference.dispatch(operation)
        await asyncio.wait_for(entered.wait(), 1)

        pipeline = object.__new__(RoomPipeline)
        pipeline.session = source
        pipeline.closing = pipeline.closed = False
        pipeline.commands = pipeline.retention = pipeline.subscription = None
        pipeline.terminal, pipeline.session_id = None, "source"
        pipeline.host = SimpleNamespace(sessions={"source": pipeline})
        pipeline.report = SimpleNamespace(
            gateway_done=asyncio.Event(), recording=None, close=lambda: None
        )
        pipeline.init, pipeline.history, pipeline.input_history = {}, deque(), {}
        pipeline.publish = lambda message: None
        pipeline.task = asyncio.create_task(pipeline._run())
        # Start the real outgoing consumer, then race a bounded producer burst
        # and native reset with the real close that cancels that consumer.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        wire = msgspec.msgpack.encode(dict(type="render.status"))
        source.outgoing.put_nowait(wire)
        source.outgoing.put_nowait(wire)
        producer = asyncio.create_task(source.outgoing.put(wire))
        channel.retire_source(
            dict(
                type="source.reset",
                epoch=c["source"]["epoch"],
                presentationId="new",
                sdkEpoch=c["source"]["sdkEpoch"] + 1,
            )
        )
        # Only renderer/encoder teardown is external; every retirement and
        # resource/registry-close method under test remains the production one.
        with patch.object(InteractiveSession, "close", noop):
            closing = asyncio.create_task(pipeline.close())
            try:
                if release_unknown:
                    with pytest.raises(RenderError) as error:
                        await asyncio.wait_for(asyncio.shield(closing), 1)
                    assert error.value.code == "RELEASE_UNKNOWN"
                    assert lease.resource_state == "release_unknown"
                    assert journal.snapshot["cleanup_state"] == "quarantined"
                    assert pipeline.host.retiring_sessions["source"] is pipeline
                else:
                    await asyncio.wait_for(asyncio.shield(closing), 1)
                    assert lease.resource_state == "released"
                    assert journal.snapshot["cleanup_state"] == "released"
                assert pipeline.closed and not pipeline.host.sessions
                assert pipeline.task.done() and channel.reset_task.done()
                assert not source.control_tasks
                assert source.outgoing.full(), (
                    "close must not depend on draining pending statuses"
                )
            finally:
                # Clean up the deliberately failing RED run without weakening
                # the positive assertions above or leaving background tasks.
                if not closing.done():
                    source.outgoing.get_nowait()
                    await asyncio.wait_for(
                        asyncio.gather(closing, return_exceptions=True), 1
                    )
                await asyncio.wait_for(producer, 1)
                await adapter.close()

    asyncio.run(run())


@pytest.mark.parametrize("prior_result", ["accepted", "completed", "superseded"])
def test_retired_opening_preserves_original_outcome_and_successor(
    tmp_path, prior_result
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.inference_control import (
        InferenceTaskScope,
        SourceInference,
    )

    async def run():
        adapter, _, journal, *_ = await setup(tmp_path)
        await journal.update(
            lambda s: s.update(
                inference=dict(enabled=False, revision=0, state="disabled")
            )
        )
        await journal.configure_inference(
            command(
                "model.inference.configure",
                "original",
                enabled=True,
                expected_inference_revision=0,
            )
        )
        if prior_result == "completed":
            await journal.settle_inference("original", 1, state="enabled")
        original = copy.deepcopy(journal.snapshot["operations"]["original"])
        if prior_result == "superseded":
            await journal.configure_inference(
                command(
                    "model.inference.configure",
                    "successor",
                    enabled=False,
                    expected_inference_revision=1,
                )
            )
        successor = copy.deepcopy(journal.snapshot["inference"])
        source = object.__new__(SourceSession)
        source.journal, source.current_segment = journal, None
        source.closed = source.binding_ended = False
        source.control_tasks, source.outgoing = set(), asyncio.Queue()
        source.source_identity = dict

        async def noop(*args, **kwargs):
            pass

        source.status_snapshot = noop
        source.context_channel = SimpleNamespace(source_generation=1, pause=noop)
        inference = object.__new__(SourceInference)
        inference.source, inference.manual, inference.prewarm = source, True, None
        inference.lock = asyncio.Lock()
        inference.task_scope = InferenceTaskScope(None, 1, 0, "original")
        inference.disable = noop
        try:
            await inference.retire_preparation("MODEL_CONTEXT_STALE_SOURCE")
            async with asyncio.timeout(1):
                receipt = msgspec.msgpack.decode(await source.outgoing.get())
            assert receipt["request_id"] == "original"
            assert receipt["state"] != "accepted"
            retired = journal.snapshot["operations"]["original"]
            if prior_result == "completed":
                assert retired == original
                assert receipt["state"] == "completed"
            elif prior_result == "superseded":
                assert retired["state"] == "completed"
                assert retired["superseded_by_inference_revision"] == 2
                assert journal.snapshot["inference"] == successor
                assert (
                    journal.snapshot["operations"]["successor"]["state"] == "accepted"
                )
            else:
                assert retired["state"] == "failed"
                assert retired["code"] == "MODEL_CONTEXT_STALE_SOURCE"
        finally:
            pending = list(source.control_tasks)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await adapter.close()

    asyncio.run(run())


@pytest.mark.parametrize("queued_updates", [0, 16])
def test_retirement_near_journal_limit_preserves_settled_history(
    tmp_path, monkeypatch, queued_updates
):
    from sglang.multimodal_gen.runtime.realtime.whitebox import render_contract
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_conditions import (
        parse_condition,
        payload_hash,
    )

    async def run():
        adapter, condition, journal, *_ = await setup(tmp_path)
        await journal.initialize_inference(manual=True)
        await journal.configure_inference(
            command(
                "model.inference.configure",
                "enable",
                enabled=True,
                expected_inference_revision=0,
            )
        )
        await journal.settle_inference("enable", 1, state="enabled")
        condition["references"] = []
        condition["onsetOccurrenceIds"] = []
        initial = None
        for version in range(1, 101 + queued_updates):
            condition["conditionId"] = f"history-{version}"
            condition["payloadHash"] = payload_hash(condition)
            parse_condition(condition)
            operation = await journal.model_condition(condition, [])
            if initial is None:
                initial = operation
                await journal.update(
                    lambda s: s["segment"].update(
                        state="streaming",
                        resource_state="reserved",
                    )
                )
            if version <= 100:
                await journal.prompt_applied(
                    initial["segment_id"],
                    version,
                    dict(first_output_frame=version, num_frames=1),
                )

        owner = object.__new__(SourceSession)
        owner.journal, owner.outgoing = journal, asyncio.Queue()
        owner.source_identity = dict

        async def noop(*args, **kwargs):
            pass

        owner._status = noop
        segment = SimpleNamespace(
            segment_id=initial["segment_id"],
            segment=copy.deepcopy(journal.snapshot["segment"]),
            resource_state="released",
            archive_failure=False,
            model_init_dispatched=True,
            pending_prompt=copy.deepcopy(journal.snapshot["segment"]["prompt_queue"][0])
            if queued_updates
            else None,
        )
        owner.current_segment = segment
        try:
            # Exercise ordinary cleanup: one sent update stays unknown, queued
            # unsent updates become cancelled, and init retains applied evidence.
            await owner.segment_finished(segment, "cancelled", None)
            before = copy.deepcopy(await journal.read())
            originals = before["operations"]
            pending = [
                key for key, op in originals.items() if op["state"] == "reconciling"
            ]
            assert len(pending) == (1 if queued_updates else 0)
            assert originals[initial["request_id"]]["state"] == "cancelled"
            assert (
                originals[initial["request_id"]]["model_context_execution"]["state"]
                == "applied"
            )
            before_bytes = len(render_contract.canonical(before))
            # Keep the real shutdown reserve; a smaller hard limit reproduces
            # the same headroom boundary without an O(history^2) test setup.
            monkeypatch.setattr(
                render_contract,
                "MAX_JOURNAL_BYTES",
                before_bytes + render_contract.JOURNAL_SHUTDOWN_RESERVE_BYTES + 1024,
            )
            successor = command(
                "render.context.start",
                "successor",
                expected_render_revision=before["render_revision"],
            )
            if pending:
                with pytest.raises(RenderError) as blocked:
                    await journal.start_model_context_segment(successor)
                assert blocked.value.code == "RENDER_TRANSITION_IN_PROGRESS"
            reset = dict(
                type="source.reset",
                epoch=condition["source"]["epoch"],
                presentationId="new",
                sdkEpoch=condition["source"]["sdkEpoch"] + 1,
            )
            await journal.retire_model_context_source(segment.segment_id, reset)
            after = await journal.read()
            assert (
                len(render_contract.canonical(after)) - before_bytes
                < render_contract.JOURNAL_SHUTDOWN_RESERVE_BYTES
            )
            for key, original in originals.items():
                result = copy.deepcopy(after["operations"][key])
                if key in pending:
                    assert (
                        result.pop("model_context_retirement")["source_reset"] == reset
                    )
                assert result == original
            await journal.retire_model_context_source(segment.segment_id, reset)
            assert journal.snapshot["operations"] == after["operations"]
            reserved = await journal.start_model_context_segment(successor)
            assert reserved["segment_id"] != segment.segment_id
            if pending:
                assert (
                    journal.snapshot["operations"][pending[0]]["state"] == "reconciling"
                )
        finally:
            await adapter.close()

    asyncio.run(run())
