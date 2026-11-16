import asyncio
import copy
import gc
import hashlib
import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from sglang.multimodal_gen.runtime.realtime.whitebox.render_assets import SourceAssets
from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
    RenderError,
    RenderJournal,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import MemoryRoomStore
from sglang.multimodal_gen.runtime.realtime.whitebox.source_conditions import (
    SourceConditions,
    encode_payload,
    envelope,
    parse_condition,
    payload_hash,
    request_identity,
    source_reached,
)

FIXTURE = Path(__file__).with_name("fixtures") / "model-context-v1.json"


def fixture():
    return json.loads(FIXTURE.read_text())


def condition():
    return copy.deepcopy(
        next(m for m in fixture()["messages"] if m["type"] == "model.condition")
    )


def test_shared_utf8_netstrings():
    value = fixture()
    assert encode_payload(value["payload"]).decode() == value["encoding"]
    assert payload_hash(value["payload"]) == value["sha256"]
    assert parse_condition(condition()) == condition()


@pytest.mark.parametrize("prompt", ["\ud800", "\udfff", "valid text \ud800 suffix"])
@pytest.mark.parametrize("validate", [encode_payload, parse_condition])
def test_lone_surrogate_prompt_rejects_with_protocol_code(prompt, validate):
    value = condition()
    value["prompt"] = prompt
    with pytest.raises(ValueError, match="^MODEL_CONTEXT_PROTOCOL_INVALID$"):
        validate(value)


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(prompt="changed"),
        lambda c: c.update(compilerVersion=True),
        lambda c: c["references"][0]["resource"].update(epoch=2),
        lambda c: c.update(extra="unsupported"),
        lambda c: c["effective"].update(sourceFrameId=True),
    ],
)
def test_strict_condition(change):
    value = condition()
    change(value)
    with pytest.raises((ValueError, RenderError)):
        parse_condition(value)


def test_source_clock_is_not_model_index():
    c = condition()
    c["effective"] = dict(
        sourceFrameId=100, sourceTimeUs=1000, simulationTick=30, policy="at-or-after"
    )
    frame = SimpleNamespace(
        index=3,
        provenance={
            "cli_frame_header": {
                "sessionId": "session",
                "epoch": 1,
                "source": dict(
                    worldId="world",
                    sourceEpoch="world-epoch",
                    presentationId="presentation",
                    sdkEpoch=1,
                    sourceFrameId=100,
                    sourceTimeUs=1000,
                    simulationTick=30,
                ),
            }
        },
    )
    assert source_reached(frame, c)
    frame.provenance["cli_frame_header"]["source"]["simulationTick"] = 29
    assert not source_reached(frame, c)


async def setup(tmp_path, *, reference_size=(2, 2), startup_release=None):
    journal = RenderJournal(
        MemoryRoomStore(),
        user_id="u",
        source_session_id="source",
        source_epoch="epoch",
        owner_epoch="owner",
    )
    await journal.create()
    await journal.update(lambda s: s.update(source_state="ready"))
    assets = SourceAssets(tmp_path, memory=True)
    c = condition()
    c["segmentId"] = (await journal.reserve_model_context_segment())["segment_id"]
    buf = io.BytesIO()
    Image.new("RGB", reference_size).save(buf, format="PNG")
    data = buf.getvalue()
    for ref in c["references"]:
        ref["sha256"] = ref["resource"]["sha256"] = hashlib.sha256(data).hexdigest()
    c["payloadHash"] = payload_hash(c)
    receipts, submits = [], []
    gate = asyncio.Event()

    async def read(handle):
        await gate.wait()
        return data

    async def emit(value):
        receipts.append(value)

    async def submit(value, references):
        op = await journal.model_condition(value, references)
        submits.append(op)
        return op

    adapter = SourceConditions(
        envelope=c,
        assets=assets,
        journal=journal,
        read_resource=read,
        emit=emit,
        submit=submit,
        capabilities=lambda: dict(
            max_references=9, max_reference_bytes=8 << 20, reference_updates=True
        ),
        validate_reference=lambda _asset: None,
        check=lambda: None,
        startup_release=startup_release,
    )
    return adapter, c, journal, receipts, submits, gate


async def handle_settled(adapter, value):
    """Tests asserting completed receipts explicitly await owned submission."""
    submission = await adapter.handle(value)
    if submission is not None:
        await submission


@pytest.mark.parametrize("close_pending", [False, True])
@pytest.mark.parametrize("conflicting_boundary", [False, True])
def test_commit_does_not_block_source_control_reader(
    tmp_path, close_pending, conflicting_boundary
):
    async def scenario():
        a, c, journal, receipts, submits, resources = await setup(tmp_path)
        entered, release = asyncio.Event(), asyncio.Event()
        original_submit = a.submit

        async def submit(value, references):
            entered.set()
            await release.wait()  # Source frames need later control samples.
            return await original_submit(value, references)

        a.submit = submit
        resources.set()
        try:
            await a.handle(c)
            await a.preparation
            commit = request_identity(c) | dict(
                type="model.condition.commit",
                controlSequence=c["controlSequence"] + 1,
                effective=c["effective"],
            )
            await asyncio.wait_for(a.handle(commit), 0.2)
            await asyncio.wait_for(entered.wait(), 1)
            assert not submits
            await asyncio.wait_for(a.handle(commit), 0.2)
            if conflicting_boundary:
                changed = copy.deepcopy(commit)
                changed["controlSequence"] += 1
                changed["effective"]["sourceFrameId"] += 20
                with pytest.raises(ValueError, match="MODEL_CONTEXT_COMMIT_CONFLICT"):
                    await a.handle(changed)
            if close_pending:
                await a.close()
                release.set()
                assert not (await journal.read())["operations"]
            else:
                release.set()
                async with asyncio.timeout(1):
                    while not submits:
                        await asyncio.sleep(0)
                assert len(submits) == 1
                assert receipts[-1]["phase"] == "accepted"
        finally:
            await a.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "outcome", ["stale-prepare", "stale-receipt", "receipt-failure", "cancel"]
)
def test_prepare_completion_observes_errors_without_admitting_stale_source(
    tmp_path, caplog, outcome
):
    async def scenario():
        a, c, journal, receipts, submits, gate = await setup(tmp_path)
        loop = asyncio.get_running_loop()
        unhandled = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))

        def retired():
            raise ValueError("MODEL_CONTEXT_STALE_SOURCE")

        async def failed_receipt(_value):
            if outcome == "stale-receipt":
                retired()
            raise RuntimeError("Condition receipt delivery failed")

        try:
            await handle_settled(a, c)
            # Let preparation block on the real resource reader before retirement.
            await asyncio.sleep(0)
            if outcome == "cancel":
                await a.close()
            else:
                if outcome == "stale-prepare":
                    a.check = retired
                else:
                    a.emit = failed_receipt
                gate.set()
                async with asyncio.timeout(1):
                    while a.preparations:
                        await asyncio.sleep(0)
                assert a.candidate["state"] == "rejected"
                assert a.candidate["reason"] == (
                    "MODEL_CONTEXT_PREPARE_FAILED"
                    if outcome == "receipt-failure"
                    else "MODEL_CONTEXT_STALE_SOURCE"
                )
                if outcome == "stale-prepare":
                    with pytest.raises(
                        ValueError, match="^MODEL_CONTEXT_STALE_SOURCE$"
                    ):
                        await handle_settled(a, c)
                    with pytest.raises(
                        ValueError, match="^MODEL_CONTEXT_STALE_SOURCE$"
                    ):
                        await a.receipt(c, "prepared")
                await a.close()
            assert [r["phase"] for r in receipts] == ["queued"]
            assert not submits and not (await journal.read())["operations"]
            # Do not await/result/exception the completed task: its owner must
            # observe failures even when close no longer finds it in the set.
            del a
            gc.collect()
            await asyncio.sleep(0)
            assert not unhandled
            errors = [r for r in caplog.records if r.levelname == "ERROR"]
            if outcome == "receipt-failure":
                assert len(errors) == 1
                assert isinstance(errors[0].exc_info[1], RuntimeError)
                assert str(errors[0].exc_info[1]) == "Condition receipt delivery failed"
            else:
                assert not errors
        finally:
            loop.set_exception_handler(previous_handler)

    asyncio.run(scenario())


def test_opt_in_debug_observes_original_prompt_and_references_at_receipt(tmp_path):
    async def scenario():
        adapter, condition, _, receipts, submits, gate = await setup(tmp_path)
        assert adapter.debug is None
        await handle_settled(adapter, condition)
        assert receipts[-1]["phase"] == "queued"
        assert "prompt" not in receipts[-1] and "references" not in receipts[-1]
        observed = []
        adapter.debug = lambda value, receipt: observed.append((value, receipt))
        gate.set()
        await adapter.preparation
        commit = {key: copy.deepcopy(condition[key]) for key in adapter.envelope}
        commit.update(
            type="model.condition.commit",
            conditionId=condition["conditionId"],
            payloadHash=condition["payloadHash"],
            controlSequence=condition["controlSequence"] + 1,
            effective=condition["effective"],
        )
        await handle_settled(adapter, commit)
        assert [receipt["phase"] for _, receipt in observed] == ["prepared", "accepted"]
        assert observed[-1][0]["prompt"] == condition["prompt"]
        assert observed[-1][0]["references"] == condition["references"]
        assert observed[-1][1]["requestId"] == submits[0]["request_id"]
        assert "prompt" not in receipts[-1] and "references" not in receipts[-1]
        await adapter.close()

    asyncio.run(scenario())


def test_prepare_commit_dedup_and_original_journal(tmp_path):
    async def scenario():
        a, c, journal, receipts, submits, gate = await setup(tmp_path)
        await handle_settled(a, c)
        assert not submits and receipts[-1]["phase"] == "queued"
        assert not (await journal.read())["operations"]
        gate.set()
        await a.preparation
        assert (
            receipts[-1]["phase"] == "prepared" and "promptVersion" not in receipts[-1]
        )
        commit = {k: copy.deepcopy(c[k]) for k in a.envelope}
        commit.update(
            type="model.condition.commit",
            conditionId=c["conditionId"],
            payloadHash=c["payloadHash"],
            controlSequence=c["controlSequence"] + 1,
            effective=c["effective"],
        )
        await handle_settled(a, commit)
        await handle_settled(a, commit)
        assert len(submits) == 1
        assert (
            receipts[-1]["phase"] == "accepted" and receipts[-1]["promptVersion"] == 1
        )
        query = {k: c[k] for k in a.envelope} | dict(
            conditionId=c["conditionId"], payloadHash=c["payloadHash"]
        )
        await a.reconcile([query])
        assert receipts[-1]["requestId"] == submits[0]["request_id"]
        withdraw = {k: commit[k] for k in commit if k != "effective"} | dict(
            type="model.condition.withdraw",
            reason="obsolete",
            controlSequence=commit["controlSequence"] + 1,
        )
        await handle_settled(a, withdraw)
        assert len(submits) == 1 and receipts[-1]["phase"] == "accepted"
        await a.close()

    asyncio.run(scenario())


def test_startup_release_keeps_query_queued_and_withdraws_only_old_condition(tmp_path):
    async def scenario():
        loop = asyncio.get_running_loop()
        released = loop.create_future()
        entered = asyncio.Queue()
        deadline = loop.time() + 2

        def startup_release(value):
            entered.put_nowait(value["conditionId"])
            return deadline, released

        a, c, journal, receipts, submits, resources = await setup(
            tmp_path, startup_release=startup_release
        )
        resources.set()
        try:
            await handle_settled(a, c)
            assert await asyncio.wait_for(entered.get(), 1) == c["conditionId"]
            await a.reconcile([request_identity(c)])
            assert [r["phase"] for r in receipts] == ["queued", "queued"]
            with pytest.raises(ValueError, match="MODEL_CONTEXT_NOT_PREPARED"):
                await handle_settled(
                    a,
                    request_identity(c)
                    | dict(
                        type="model.condition.commit",
                        controlSequence=c["controlSequence"] + 1,
                        effective=c["effective"],
                    ),
                )
            assert not (await journal.read())["operations"]

            first = a.preparation
            await asyncio.wait_for(
                a.handle(
                    request_identity(c)
                    | dict(
                        type="model.condition.withdraw",
                        controlSequence=c["controlSequence"] + 1,
                        reason="state-changed",
                    )
                ),
                1,
            )
            with pytest.raises(asyncio.CancelledError):
                await first
            assert not released.done()

            successor = copy.deepcopy(c)
            successor.update(
                conditionId="new-startup", controlSequence=c["controlSequence"] + 2
            )
            await handle_settled(a, successor)
            assert await asyncio.wait_for(entered.get(), 1) == successor["conditionId"]
            assert receipts[-1]["phase"] == "queued"
            released.set_result(None)
            await a.preparation
            assert [
                (r["conditionId"], r["phase"])
                for r in receipts
                if r["phase"] == "prepared"
            ] == [(successor["conditionId"], "prepared")]

            await handle_settled(
                a,
                request_identity(successor)
                | dict(
                    type="model.condition.commit",
                    controlSequence=successor["controlSequence"] + 1,
                    effective=successor["effective"],
                ),
            )
            assert len(submits) == 1 and submits[0]["op"] == "render.start"

            await journal.prompt_applied(
                c["segmentId"], 1, dict(first_output_frame=0, num_frames=1)
            )
            await a.reconcile([request_identity(successor)])
            later = dict(successor, conditionId="same-segment-update")
            await handle_settled(a, later)
            await a.preparation
            assert receipts[-1]["phase"] == "prepared" and entered.empty()
        finally:
            await a.close()

    asyncio.run(scenario())


def test_startup_release_timeout_does_not_release_shared_window(tmp_path):
    async def scenario():
        loop = asyncio.get_running_loop()
        released = loop.create_future()
        a, c, journal, receipts, _, resources = await setup(
            tmp_path, startup_release=lambda _: (loop.time() + 0.01, released)
        )
        resources.set()
        try:
            await handle_settled(a, c)
            await asyncio.wait_for(a.preparation, 1)
            assert [r["phase"] for r in receipts] == ["queued", "rejected"]
            assert receipts[-1]["reason"] == "RELEASE_UNKNOWN"
            assert not released.done()
            assert not (await journal.read())["operations"]
        finally:
            await a.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["reservation", "inference-revision"])
def test_startup_release_cannot_prepare_after_opening_changes(tmp_path, change):
    async def scenario():
        loop = asyncio.get_running_loop()
        released, entered = loop.create_future(), asyncio.Event()

        def startup_release(_):
            entered.set()
            return loop.time() + 2, released

        a, c, journal, receipts, _, resources = await setup(
            tmp_path, startup_release=startup_release
        )
        resources.set()
        try:
            await handle_settled(a, c)
            await asyncio.wait_for(entered.wait(), 1)
            if change == "reservation":
                await journal.update(
                    lambda s: s["model_context_segment_ids"].update(segment_id="next")
                )
                expected = "SEGMENT_EPOCH_MISMATCH"
            else:
                await journal.update(
                    lambda s: s.update(inference=dict(enabled=True, revision=1))
                )
                expected = "MODEL_UNAVAILABLE"
            released.set_result(None)
            await a.preparation
            assert [r["phase"] for r in receipts] == ["queued", "rejected"]
            assert receipts[-1]["reason"] == expected
            assert not (await journal.read())["operations"]
        finally:
            await a.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancellations", [1, 2])
def test_withdraw_during_asset_write_keeps_successor_preparation_valid(
    tmp_path, cancellations
):
    async def scenario():
        a, c, journal, receipts, submits, gate = await setup(tmp_path)
        a.assets = SourceAssets(tmp_path, memory=False)
        entered, resume = threading.Event(), threading.Event()
        original_open = Path.open

        class PausedWriter:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self.stream, name)

            def write(self, data):
                entered.set()
                assert resume.wait(5)
                return self.stream.write(data)

        def open_file(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return PausedWriter(stream) if args and args[0] == "xb" else stream

        old = None
        gate.set()
        with patch.object(Path, "open", open_file):
            try:
                await handle_settled(a, c)
                old = a.preparation
                assert await asyncio.to_thread(entered.wait, 3)
                await handle_settled(
                    a,
                    request_identity(c)
                    | dict(
                        type="model.condition.withdraw",
                        controlSequence=c["controlSequence"] + 1,
                        reason="state-changed",
                    ),
                )
                await asyncio.sleep(0)
                if cancellations == 2:
                    old.cancel()
                    await asyncio.sleep(0)
                # Withdrawal must not release a resource commit that is still running.
                assert a.assets.lock.locked()
                newer = copy.deepcopy(c)
                newer.update(conditionId="fresh", prompt="Fresh complete condition")
                newer["payloadHash"] = payload_hash(newer)
                await handle_settled(a, newer)
                assert not (await journal.read())["operations"]
                resume.set()
                await asyncio.gather(old, return_exceptions=True)
                await a.preparation
                assert old.cancelled()
                assert receipts[-1]["conditionId"] == "fresh"
                assert receipts[-1]["phase"] == "prepared"
                assert not any(r["phase"] == "rejected" for r in receipts)
                assert not any(
                    r["conditionId"] == c["conditionId"] and r["phase"] == "prepared"
                    for r in receipts
                )
                metadata = json.loads((a.assets.directory / "assets.json").read_text())
                assert set(metadata) == set(a.assets.assets)
                assert a.assets.bytes == sum(
                    asset["bytes"] for asset in metadata.values()
                )
                assert not list(a.assets.directory.glob("*.tmp"))
                assert len(await a.assets.resolve(a.candidate["references"])) == len(
                    c["references"]
                )
                assert not submits
            finally:
                resume.set()
                if old:
                    await asyncio.gather(old, return_exceptions=True)
                await a.close()

    asyncio.run(scenario())


def test_failed_asset_write_can_retry_without_a_partial_cache_entry(tmp_path):
    async def scenario():
        assets = SourceAssets(tmp_path, memory=False)
        data = io.BytesIO()
        Image.new("RGB", (2, 2)).save(data, format="PNG")
        payload = data.getvalue()
        original_open = Path.open

        class FailedWriter:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def write(self, value):
                self.stream.write(value[:4])
                raise OSError("test write interrupted")

        def open_file(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return FailedWriter(stream) if args and args[0] == "xb" else stream

        with patch.object(Path, "open", open_file):
            with pytest.raises(OSError, match="test write interrupted"):
                await assets.upload("png", payload)
        assert not list(assets.directory.iterdir())
        descriptor = await assets.upload("png", payload)
        assert (assets.directory / descriptor["path"]).read_bytes() == payload
        assert len(assets.assets) == 1 and assets.bytes == len(payload)

    asyncio.run(scenario())


def test_withdraw_cancels_only_preparation(tmp_path):
    async def scenario():
        a, c, journal, receipts, submits, gate = await setup(tmp_path)
        await handle_settled(a, c)
        await handle_settled(
            a,
            {k: c[k] for k in a.envelope}
            | dict(
                type="model.condition.withdraw",
                conditionId=c["conditionId"],
                payloadHash=c["payloadHash"],
                controlSequence=c["controlSequence"] + 1,
                reason="obsolete",
            ),
        )
        gate.set()
        await asyncio.sleep(0)
        assert not submits and not any(r["phase"] == "prepared" for r in receipts)
        await a.close()

    asyncio.run(scenario())


def test_journal_allocates_updates_and_conflicts_without_new_request(tmp_path):
    async def scenario():
        a, c, journal, receipts, submits, gate = await setup(tmp_path)
        first = await journal.model_condition(c, [])
        await journal.update(
            lambda s: s["segment"].update(state="streaming", applied_version=1)
        )
        update = copy.deepcopy(c)
        update.update(conditionId="second", prompt="rain now", onsetOccurrenceIds=[])
        update["payloadHash"] = payload_hash(update)
        second = await journal.model_condition(update, [])
        again = await journal.model_condition(update, [])
        assert first["prompt_version"] == 1 and second["prompt_version"] == 2
        assert again["request_id"] == second["request_id"] and again["duplicate"]
        altered = copy.deepcopy(update)
        altered["prompt"] = "conflict"
        altered["payloadHash"] = payload_hash(altered)
        with pytest.raises(RenderError) as conflict:
            await journal.model_condition(altered, [])
        assert conflict.value.code == "REQUEST_ID_CONFLICT"
        assert len((await journal.read())["operations"]) == 2
        item = journal.snapshot["segment"]["prompt_queue"][0]
        assert item["source_context"]["effective"] == update["effective"]
        assert "references" not in item
        await a.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("replace_references", [False, True])
def test_complete_condition_recovers_after_definite_prompt_failure(
    tmp_path, replace_references
):
    async def scenario():
        a, c, journal, receipts, submits, gate = await setup(tmp_path)
        await journal.model_condition(c, [])
        await journal.update(lambda s: s["segment"].update(state="streaming"))
        await journal.prompt_applied(
            c["segmentId"], 1, dict(first_output_frame=0, num_frames=1)
        )
        failed = copy.deepcopy(c)
        failed.update(
            conditionId="failed",
            prompt="failed event",
            onsetOccurrenceIds=["old-onset"],
        )
        failed["payloadHash"] = payload_hash(failed)
        failed_refs = (
            [
                dict(
                    id="<Picture %s>" % ref["picture"],
                    asset_id="sha256:" + ref["sha256"],
                    role="scene",
                )
                for ref in c["references"]
            ]
            if replace_references
            else []
        )
        original = await journal.model_condition(failed, failed_refs)
        await journal.prompt_failed(
            c["segmentId"], 2, never_applied=True, code="CONDITION_INVALID"
        )
        recovered = copy.deepcopy(c)
        recovered.update(
            conditionId="recovered",
            prompt="fresh full condition",
            onsetOccurrenceIds=["new-onset"],
        )
        recovered["payloadHash"] = payload_hash(recovered)
        gate.set()
        await handle_settled(a, recovered)
        await a.preparation
        await handle_settled(
            a,
            request_identity(recovered)
            | dict(
                type="model.condition.commit",
                controlSequence=recovered["controlSequence"] + 1,
                effective=recovered["effective"],
            ),
        )
        assert receipts[-1]["phase"] == "accepted"
        assert receipts[-1]["promptVersion"] == 3
        assert len(submits) == 1
        state = await journal.read()
        segment = state["segment"]
        assert segment["accepted_version"] == 3 and segment["applied_version"] == 1
        assert not segment["rebase_required"]
        assert segment["prompt_queue"][0]["one_shot_event_ids"] == ["new-onset"]
        assert segment["prompt_queue"][0].get("references") == a.candidate["references"]
        assert segment["prompt_events"]["old-onset"]["state"] == "never_applied"
        assert state["operations"][original["request_id"]]["state"] == "failed"
        await journal.prompt_applied(
            c["segmentId"], 3, dict(first_output_frame=1, num_frames=1)
        )
        await a.reconcile([request_identity(recovered)])
        assert receipts[-1]["phase"] == "applied"
        assert receipts[-1]["promptVersion"] == 3
        await a.close()

    asyncio.run(scenario())


def test_definite_commit_rejection_is_terminal_and_frees_next_condition(tmp_path):
    async def scenario():
        a, c, journal, receipts, submits, gate = await setup(tmp_path)
        await journal.model_condition(c, [])
        await journal.update(
            lambda s: s["segment"].update(state="streaming", applied_version=1)
        )
        rejected = copy.deepcopy(c)
        rejected.update(
            conditionId="rejected", prompt="too late", onsetOccurrenceIds=[]
        )
        rejected["payloadHash"] = payload_hash(rejected)
        gate.set()
        await handle_settled(a, rejected)
        await a.preparation
        assert receipts[-1]["phase"] == "prepared"
        await journal.update(lambda s: s["segment"].update(state="ended"))
        commit = request_identity(rejected) | dict(
            type="model.condition.commit",
            controlSequence=rejected["controlSequence"] + 1,
            effective=rejected["effective"],
        )
        await handle_settled(a, commit)
        for replay in (
            lambda: a.handle(commit),
            lambda: a.handle(rejected),
            lambda: a.reconcile([request_identity(rejected)]),
        ):
            assert receipts[-1]["phase"] == "rejected"
            assert receipts[-1]["reason"] == "MODEL_UNAVAILABLE"
            assert (
                "requestId" not in receipts[-1] and "promptVersion" not in receipts[-1]
            )
            await replay()
        assert receipts[-1]["phase"] == "rejected"
        assert receipts[-1]["reason"] == "MODEL_UNAVAILABLE"
        assert await journal.model_condition_query(request_identity(rejected)) is None
        assert len((await journal.read())["operations"]) == 1
        assert not submits
        # Re-enable admission only to isolate candidate recovery from segment lifetime.
        await journal.update(lambda s: s["segment"].update(state="streaming"))
        next_c = copy.deepcopy(rejected)
        next_c.update(conditionId="next", prompt="new complete condition")
        next_c["payloadHash"] = payload_hash(next_c)
        await handle_settled(a, next_c)
        await a.preparation
        await handle_settled(
            a,
            request_identity(next_c)
            | dict(
                type="model.condition.commit",
                controlSequence=next_c["controlSequence"] + 1,
                effective=next_c["effective"],
            ),
        )
        assert receipts[-1]["phase"] == "accepted"
        assert receipts[-1]["promptVersion"] == 2
        assert len(submits) == 1
        await a.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("retryable", [False, True])
def test_uncertain_commit_stays_unknown_without_resubmitting(tmp_path, retryable):
    async def scenario():
        a, c, journal, receipts, submits, gate = await setup(tmp_path)
        attempts = []

        async def unavailable(value, references):
            attempts.append(value)
            if retryable:
                raise RenderError("CONTROL_UNAVAILABLE", retryable=True)
            raise ConnectionError("reply unavailable")

        a.submit = unavailable
        gate.set()
        await handle_settled(a, c)
        await a.preparation
        commit = request_identity(c) | dict(
            type="model.condition.commit",
            controlSequence=c["controlSequence"] + 1,
            effective=c["effective"],
        )
        await handle_settled(a, commit)
        assert receipts[-1]["phase"] == "unknown"
        await a.reconcile([request_identity(c)])
        assert receipts[-1]["phase"] == "unknown"
        await handle_settled(a, c)
        assert receipts[-1]["phase"] == "unknown"
        await handle_settled(a, commit)
        assert receipts[-1]["phase"] == "unknown"
        assert len(attempts) == 1 and not submits
        assert not (await journal.read())["operations"]
        next_c = copy.deepcopy(c)
        next_c["conditionId"] = "next"
        with pytest.raises(ValueError, match="MODEL_CONTEXT_BACKPRESSURE"):
            await handle_settled(a, next_c)
        await a.close()

    asyncio.run(scenario())


def test_commit_error_after_cas_reconciles_original_operation(tmp_path):
    async def scenario():
        a, c, journal, receipts, submits, gate = await setup(tmp_path)
        original_submit = a.submit

        async def lost_reply(value, references):
            await original_submit(value, references)
            raise RenderError("MODEL_UNAVAILABLE")

        a.submit = lost_reply
        gate.set()
        await handle_settled(a, c)
        await a.preparation
        commit = request_identity(c) | dict(
            type="model.condition.commit",
            controlSequence=c["controlSequence"] + 1,
            effective=c["effective"],
        )
        await handle_settled(a, commit)
        assert receipts[-1]["phase"] == "accepted"
        original_receipt = receipts[-1]
        await handle_settled(a, commit)
        await a.reconcile([request_identity(c)])
        assert receipts[-1] == original_receipt
        assert len(submits) == 1
        assert len((await journal.read())["operations"]) == 1
        await a.close()

    asyncio.run(scenario())


def test_reference_clear_unsupported_and_hash_fail_before_prepared(tmp_path):
    async def scenario():
        a, c, journal, receipts, submits, gate = await setup(tmp_path)
        await journal.model_condition(
            c,
            [{"id": "<Picture 1>", "asset_id": "sha256:" + "1" * 64, "role": "scene"}],
        )
        await journal.update(
            lambda s: s["segment"].update(state="streaming", applied_version=1)
        )
        c["references"] = []
        c["conditionId"] = "clear"
        c["onsetOccurrenceIds"] = []
        c["payloadHash"] = payload_hash(c)
        a.capabilities = lambda: dict(
            max_references=9, max_reference_bytes=8 << 20, reference_updates=False
        )
        await handle_settled(a, c)
        await a.preparation
        assert receipts[-1]["reason"] == "REFERENCE_UPDATE_UNSUPPORTED"
        assert not submits
        await a.close()
        a, c, journal, receipts, submits, gate = await setup(tmp_path)
        c["references"][0]["sha256"] = c["references"][0]["resource"]["sha256"] = (
            "0" * 64
        )
        c["payloadHash"] = payload_hash(c)
        gate.set()
        await handle_settled(a, c)
        await a.preparation
        assert receipts[-1]["reason"] == "MODEL_CONTEXT_RESOURCE_HASH"
        await handle_settled(a, c)
        await a.reconcile([request_identity(c)])
        assert receipts[-1]["phase"] == "rejected"
        assert receipts[-1]["reason"] == "MODEL_CONTEXT_RESOURCE_HASH"
        assert not any(r["phase"] == "prepared" for r in receipts)
        assert not (await journal.read())["operations"]
        await a.close()

    asyncio.run(scenario())


def test_complete_reference_replacement_including_empty(tmp_path):
    async def scenario():
        a, c, journal, receipts, submits, gate = await setup(tmp_path)
        await journal.model_condition(
            c,
            [{"id": "<Picture 1>", "asset_id": "sha256:" + "1" * 64, "role": "scene"}],
        )
        await journal.update(
            lambda s: s["segment"].update(state="streaming", applied_version=1)
        )
        c["conditionId"] = "clear"
        c["onsetOccurrenceIds"] = []
        c["references"] = []
        c["payloadHash"] = payload_hash(c)
        await handle_settled(a, c)
        await a.preparation
        commit = {k: c[k] for k in a.envelope} | dict(
            type="model.condition.commit",
            conditionId=c["conditionId"],
            payloadHash=c["payloadHash"],
            controlSequence=c["controlSequence"] + 1,
            effective=c["effective"],
        )
        await handle_settled(a, commit)
        assert journal.snapshot["segment"]["prompt_queue"][0]["references"] == []
        changed = copy.deepcopy(commit)
        changed["effective"]["sourceFrameId"] += 1
        with pytest.raises(ValueError, match="COMMIT_CONFLICT"):
            await handle_settled(a, changed)
        await a.close()

    asyncio.run(scenario())


def header(c):
    return dict(
        protocolVersion=1,
        sessionId=c["source"]["sessionId"],
        epoch=c["source"]["epoch"],
        mediaGeneration=1,
        outputFrameId=9,
        outputPtsUs=10000,
        source=dict(
            presentationId=c["source"]["presentationId"],
            sdkEpoch=c["source"]["sdkEpoch"],
            sourceFrameId=c["effective"]["sourceFrameId"],
            sourceTimeUs=c["effective"]["sourceTimeUs"],
            simulationTick=c["effective"]["simulationTick"],
            worldRevision=1,
        ),
        uiRevision=1,
        uiCompleteThroughUs=c["effective"]["sourceTimeUs"],
        widthPixels=2,
        heightPixels=2,
    )


def test_actual_cli_room_source_binding_and_nonblocking_prepare(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import CLIView
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        LocalRoomEngine,
    )

    async def scenario():
        _, c, journal, receipts, submits, gate = await setup(
            tmp_path, reference_size=(128, 128)
        )
        sent = []

        class Socket:
            async def send(self, wire):
                sent.append(json.loads(wire))

        cap = dict(
            protocol=c["protocol"],
            required=True,
            configHash=c["configHash"],
            gameBuildHash=c["source"]["gameBuildHash"],
            compilerVersion=1,
            visible=False,
        )
        descriptor = dict(
            sessionId="session",
            epoch=1,
            modelContext=cap,
            media=dict(kind="pixels"),
            baseUrl="http://127.0.0.1:1234",
            accessToken="private-token",
        )
        view = CLIView(
            SimpleNamespace(
                descriptor=descriptor,
                spec=SimpleNamespace(fps=24),
                prepared_delivery=None,
            )
        )
        view.control = Socket()
        view.client_id = "consumer"
        owner = object.__new__(SourceSession)
        owner.source_epoch = "epoch"
        owner.session_id = "source"
        owner.closed = False
        owner.binding_ended = False
        owner.current_segment = None
        owner.assets = SourceAssets(tmp_path, memory=True)
        owner.journal = journal
        owner.source_conditions = owner.context_channel = None
        runtime = SimpleNamespace(player=lambda _: dict(engine=view))
        engine = LocalRoomEngine(runtime, dict(stream_id="stream", engine={}, ui={}))
        owner.engine = engine
        calls = []

        async def submit(value, refs):
            calls.append((value, refs))
            return await journal.model_condition(value, refs)

        owner.submit_source_condition = submit

        async def resource(_):
            await gate.wait()
            buf = io.BytesIO()
            Image.new("RGB", (128, 128)).save(buf, format="PNG")
            return buf.getvalue()

        view.read_model_context_resource = resource
        await owner.bind_source_conditions()
        channel = view.model_context
        c["segmentId"] = sent[0]["segmentId"]
        c["payloadHash"] = payload_hash(c)
        subscribed = dict(
            type="model.context.subscribed",
            epoch=1,
            protocol=c["protocol"],
            consumer=c["audience"],
            segmentId=c["segmentId"],
            subscriptionOccurrenceSequence=0,
            sampleControlFloor=1,
        )
        snapshot = copy.deepcopy(
            next(
                m
                for m in fixture()["messages"]
                if m["type"] == "model.context.snapshot"
            )
        )
        snapshot.update(
            segmentId=c["segmentId"],
            desired={
                k: c[k]
                for k in (
                    "protocol",
                    "source",
                    "audience",
                    "segmentId",
                    "configHash",
                    "compilerVersion",
                    "lifecycle",
                    "prompt",
                    "references",
                    "onsetOccurrenceIds",
                    "ongoingOccurrenceIds",
                )
            },
        )
        await channel.consume(subscribed)
        await channel.consume(snapshot)
        await channel.consume(c)
        assert owner.source_conditions is not None and not calls
        assert sent[-1]["type"] == "model.context.ack"
        # A second control is consumed while the actual image read is blocked.
        sample = dict(
            type="model.context.sample",
            epoch=1,
            consumerId="consumer",
            consumerEpoch=c["audience"]["consumerEpoch"],
            segmentId=c["segmentId"],
            header=header(c),
            metadata=dict(
                contextStateId=snapshot["contextStateId"],
                contextRevision=snapshot["contextRevision"],
                viewRevision=snapshot["viewRevision"],
                throughControlSequence=c["controlSequence"],
                compiledConditionId=snapshot["compiledConditionId"],
            ),
        )
        await channel.consume(sample)
        assert await channel.frame(header(c)) == sample["metadata"]
        gate.set()
        await owner.source_conditions.preparation
        assert sent[-1]["message"]["phase"] == "prepared"
        assert not calls
        assert "<Picture 1>" == owner.source_conditions.candidate["references"][0]["id"]
        await channel.close()
        assert sent[-1]["consumerEpoch"] == c["audience"]["consumerEpoch"]

    asyncio.run(scenario())


def test_actual_source_selection_waits_for_native_boundary_and_initial_output(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def scenario():
        a, c, journal, receipts, submits, gate = await setup(tmp_path)
        await journal.model_condition(c, [])
        await journal.update(lambda s: s["segment"].update(state="streaming"))
        next_c = copy.deepcopy(c)
        next_c["conditionId"] = "next"
        next_c["onsetOccurrenceIds"] = []
        next_c["prompt"] = "updated"
        next_c["payloadHash"] = payload_hash(next_c)
        await journal.model_condition(next_c, [])
        source = object.__new__(SourceSession)
        source.journal = journal
        sent = []

        async def event(ws, kind, fields, before_send):
            before_send("event")
            sent.append(fields)

        segment = SimpleNamespace(
            clip_switch=False,
            pending_prompt=None,
            sink=SimpleNamespace(busy=False),
            segment_id=journal.snapshot["segment"]["segment_id"],
            prompt_by_event={},
            send_model_event=event,
        )
        h = header(c)
        frame = SimpleNamespace(index=500, provenance={"cli_frame_header": h})
        await source.before_segment_source(segment, None, frame, 7)
        assert not sent  # Initial output is not yet proven.
        await journal.update(lambda s: s["segment"].update(applied_version=1))
        h["source"]["simulationTick"] -= 1
        await source.before_segment_source(segment, None, frame, 7)
        assert not sent
        h["source"]["simulationTick"] += 1
        await source.before_segment_source(segment, None, frame, 7)
        assert sent[0]["min_source_frame"] == 7
        assert sent[0]["min_source_frame"] != h["source"]["sourceFrameId"]
        await a.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("member", [False, True])
def test_declared_source_defers_legacy_init_until_first_context_commit(
    tmp_path, member
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.player_media import (
        MemberEngine,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_pipeline import (
        LocalRoomEngine,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.startup import PlayStartup

    async def scenario():
        _, c, journal, _, _, _ = await setup(tmp_path)
        cap = dict(
            protocol=c["protocol"],
            required=True,
            configHash=c["configHash"],
            gameBuildHash=c["source"]["gameBuildHash"],
            compilerVersion=1,
            visible=False,
        )
        engine = SimpleNamespace(descriptor=dict(modelContext=cap))
        runtime = SimpleNamespace(player=lambda *a, **k: dict(engine=engine))
        # Exercise the same inherited member capability accessor without joining
        # a second Room or granting its browser any new command permission.
        source_engine = object.__new__(MemberEngine if member else LocalRoomEngine)
        source_engine.runtime = runtime
        source_engine.stream_id = "stream"
        source_engine.closed = False
        owner = object.__new__(SourceSession)
        owner.engine = source_engine
        owner.member_prefill = None
        owner.config = SimpleNamespace(spec=VideoSpec(2, 2, 24))
        owner.journal = journal
        owner.initial_message = {"references": [{"legacy": True}]}
        owner.session_id = "source"
        owner.source_epoch = "epoch"
        from sglang.multimodal_gen.runtime.realtime.source_prepare import V2_PROTOCOL

        legacy = PlayStartup(
            owner,
            dict(
                protocol=V2_PROTOCOL,
                request_id="startup",
                intent="model",
                initial_render=dict(prompt="legacy frozen prompt", references=[]),
            ),
        )
        owner.startup = legacy
        await owner.source_model_ready({"prompt": "legacy"})
        assert legacy.closed and legacy.task is None and legacy.lease is None
        assert owner.startup is None and owner.world_driven
        assert owner.initial_message["references"] == []
        assert (await journal.read())["segment"] is None
        op = await journal.model_condition(c, [])
        assert op["prompt_version"] == 1 and op["segment_id"] == c["segmentId"]
        assert journal.snapshot["segment"]["prompt"] == c["prompt"]

    asyncio.run(scenario())


def test_noncontext_keeps_original_startup_once():
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def scenario():
        events = []

        class Startup:
            admitted = object()

            def launch(self, init):
                events.append(("launch", init))

            async def wait(self, event):
                events.append(("wait", event))

        owner = object.__new__(SourceSession)
        owner.engine = SimpleNamespace(descriptor={})
        owner.startup = Startup()
        original = owner.startup
        await owner.source_model_ready({"prompt": "original"})
        assert owner.startup is original and events == [
            ("launch", {"prompt": "original"}),
            ("wait", original.admitted),
        ]

    asyncio.run(scenario())


def test_world_driven_browser_prompt_and_asset_upload_remain_rejected(tmp_path):
    import msgspec

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def scenario():
        _, c, journal, _, _, _ = await setup(tmp_path)
        source = object.__new__(SourceSession)
        source.world_driven = True
        source.outgoing = asyncio.Queue()
        source.journal = journal
        await source.upload_asset(
            dict(request_id="browser-image", format="png", payload=b"untrusted")
        )
        await source.command(dict(op="prompt.replace", request_id="browser-prompt"))
        errors = [msgspec.msgpack.decode(await source.outgoing.get()) for _ in range(2)]
        assert all(m["code"] == "INVALID_COMMAND" for m in errors)
        assert not (await journal.read())["operations"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "waiting_at", ["adapter-close", "samples", "condition", "checkpoint"]
)
def test_native_reset_fences_inflight_handoff_and_condition_cursor(
    tmp_path, waiting_at
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
        CLIConditionChannel,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_conditions import (
        PAYLOAD,
    )

    async def scenario():
        old, c, journal, _, submits, _ = await setup(tmp_path)
        sent, created = [], []
        entered, release = asyncio.Event(), asyncio.Event()
        blocked_phase = {"condition": "queued", "checkpoint": "unknown"}.get(waiting_at)

        async def send(wire):
            value = json.loads(wire)
            sent.append(value)
            message = value.get("message", {})
            if blocked_phase and message.get("phase") == blocked_phase:
                entered.set()
                await release.wait()

        view = SimpleNamespace(
            descriptor=dict(
                sessionId=c["source"]["sessionId"],
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
            transport_generation=1,
        )
        channel = CLIConditionChannel(view)
        channel.consumer, channel.segment = copy.deepcopy(c["audience"]), c["segmentId"]
        channel.sample_floor = 1

        def create(e, emit, check):
            adapter = SourceConditions(
                envelope=e,
                assets=old.assets,
                journal=journal,
                read_resource=old.read_resource,
                emit=emit,
                submit=old.submit,
                capabilities=old.capabilities,
                validate_reference=old.validate_reference,
                check=check,
            )
            created.append(adapter)
            return adapter

        channel.create_adapter = create
        handoff = waiting_at in {"adapter-close", "samples"}
        message_condition = copy.deepcopy(c)
        if handoff:
            channel.adapter, channel.requested_segment = old, "successor"
            channel.sequence = 1
            message_condition["segmentId"] = "successor"
        snapshot = copy.deepcopy(fixture()["messages"][0])
        snapshot.update(
            **envelope(message_condition),
            controlSequence=2 if handoff else 1,
            baseControlSequence=1 if handoff else 0,
            desired={k: message_condition[k] for k in PAYLOAD},
            candidate=None,
        )
        if waiting_at == "condition":
            message = dict(c, controlSequence=1)
        elif waiting_at == "checkpoint":
            message = dict(
                type="model.context.checkpoint",
                afterControlSequence=0,
                snapshot=snapshot,
                condition=None,
                queries=[request_identity(c)],
            )
        else:
            message = snapshot
        task, lock_held = None, False
        try:
            if waiting_at == "adapter-close":
                # Real close awaits its actual cancelled preparation, not a fake close.
                await handle_settled(old, c)
                await asyncio.sleep(0)
                assert old.preparations
            if waiting_at == "samples":
                await channel.changed.acquire()
                lock_held = True
            task = asyncio.create_task(channel.consume(message))
            if handoff:
                await asyncio.sleep(0)
                assert not task.done()
                assert old.closed
            else:
                await asyncio.wait_for(entered.wait(), 1)
            reset = dict(
                type="source.reset",
                epoch=c["source"]["epoch"],
                presentationId="after-reset",
                sdkEpoch=c["source"]["sdkEpoch"] + 1,
            )
            channel.retire_source(reset)
            if lock_held:
                channel.changed.release()
                lock_held = False
            await asyncio.wait_for(channel.reset_task, 1)
            release.set()
            with pytest.raises(ValueError, match="^MODEL_CONTEXT_STALE_SOURCE$"):
                await asyncio.wait_for(task, 1)
            assert channel.adapter is None and channel.consumer is None
            assert channel.sequence == 0 and not channel.snapshots
            assert channel.sample_floor == 1
            assert not any(v["type"] == "model.context.ack" for v in sent)
            assert not submits and not (await journal.read())["operations"]
            # Explicit delivery for the new source must not resume the old envelope.
            channel.suspended = False
            fresh_consumer = dict(c["audience"], consumerEpoch="after-reset")
            await channel.consume(
                dict(
                    type="model.context.subscribed",
                    epoch=c["source"]["epoch"],
                    protocol=c["protocol"],
                    consumer=fresh_consumer,
                    segmentId=channel.segment,
                    sampleControlFloor=1,
                )
            )
            assert not any(
                v.get("message", {}).get("type") == "model.context.resume" for v in sent
            )
            fresh = copy.deepcopy(snapshot)
            fresh.update(
                segmentId=channel.segment,
                audience=fresh_consumer,
                controlSequence=1,
                baseControlSequence=0,
            )
            fresh["source"].update(
                presentationId=reset["presentationId"], sdkEpoch=reset["sdkEpoch"]
            )
            fresh["desired"].update(envelope(fresh))
            await channel.consume(fresh)
            channel.adapter.check()
            assert channel.adapter.envelope["source"]["sdkEpoch"] == reset["sdkEpoch"]
            assert sent[-1]["type"] == "model.context.ack"
            assert sent[-1]["consumerEpoch"] == "after-reset" and channel.sequence == 1
        finally:
            release.set()
            if lock_held:
                channel.changed.release()
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await channel.close()
            for adapter in created:
                await adapter.close()
            await old.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "outcome",
    ["retired-query", "retired-failure", "current-identity", "current-release"],
)
def test_worker_keeps_old_query_outcome_out_of_fresh_probe(tmp_path, outcome, caplog):
    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import NativeCLIView
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
        CLIConditionChannel,
    )

    async def scenario():
        template, c, journal, _, submits, _ = await setup(tmp_path)
        view = object.__new__(NativeCLIView)
        view.descriptor = dict(
            sessionId=c["source"]["sessionId"],
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
        view.client_id = c["audience"]["consumerId"]
        view.closed = view.stopped = False
        view.transport_generation = 1
        view.transport_lock = asyncio.Lock()
        view.ready, view.configured = asyncio.Event(), asyncio.Event()
        probe_sent, query_entered, query_release = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )
        closed_sockets = []
        channel = CLIConditionChannel(view)
        view.model_context = channel
        same_generation = outcome == "current-release"
        fresh_consumer = dict(c["audience"])
        if not same_generation:
            fresh_consumer["consumerEpoch"] = "fresh-consumer"

        class Socket:
            async def close(self):
                closed_sockets.append(self)

            async def send(self, wire):
                value = json.loads(wire)
                if value.get("type") == "model.context.subscribe" and value.get(
                    "paused"
                ):
                    consumer = dict(fresh_consumer)
                    if outcome == "current-identity":
                        consumer["consumerId"] = "different-consumer"
                    channel.offer(
                        dict(
                            type="model.context.subscribed",
                            epoch=c["source"]["epoch"],
                            protocol=c["protocol"],
                            segmentId=channel.segment,
                            consumer=consumer,
                            sampleControlFloor=1,
                            subscriptionPause=True,
                            paused=True,
                        )
                    )
                    probe_sent.set()

        view.control, view.media = Socket(), Socket()

        async def open_transport():
            view.control, view.media = Socket(), Socket()
            view.transport_generation += 1

        view._open_transport = open_transport
        channel.segment, channel.consumer = c["segmentId"], dict(c["audience"])

        def create(e, emit, check):
            return SourceConditions(
                envelope=e,
                assets=template.assets,
                journal=journal,
                read_resource=template.read_resource,
                emit=emit,
                submit=template.submit,
                capabilities=template.capabilities,
                validate_reference=template.validate_reference,
                check=check,
            )

        channel.create_adapter = create
        await channel.admit_envelope(c)
        old = channel.adapter
        original_get = journal.store.get

        async def delayed_get(key):
            if asyncio.current_task() is channel.task:
                query_entered.set()
                await query_release.wait()
                if outcome == "retired-failure":
                    raise RuntimeError("Previous storage query failed")
                if same_generation:
                    raise RenderError("RELEASE_UNKNOWN")
            return await original_get(key)

        journal.store.get = delayed_get
        channel.task = asyncio.create_task(channel.run())
        probe = None
        try:
            if outcome != "current-identity":
                channel.offer(
                    dict(
                        type="model.context.reconcile",
                        epoch=c["source"]["epoch"],
                        queries=[request_identity(c)],
                    )
                )
                await asyncio.wait_for(query_entered.wait(), 1)
            if same_generation:
                await channel.pause()
            else:
                channel.retire_source(
                    dict(
                        type="source.reset",
                        epoch=c["source"]["epoch"],
                        presentationId="fresh",
                        sdkEpoch=c["source"]["sdkEpoch"] + 1,
                    )
                )
                await asyncio.wait_for(channel.reset_task, 1)
                assert old.closed and channel.adapter is None
            probe = asyncio.create_task(channel.probe_subscription_pause(timeout=1))
            await asyncio.wait_for(probe_sent.wait(), 1)
            # Actual recovery closes both old sockets; the old storage query is
            # independent and remains owned by the original worker item.
            assert len(closed_sockets) == (0 if same_generation else 2)
            assert view.transport_generation == (1 if same_generation else 2)
            query_release.set()
            if outcome.startswith("retired-"):
                await asyncio.wait_for(probe, 1)
                assert channel.subscription_probe_error is None
                assert channel.consumer == fresh_consumer
                assert not [r for r in caplog.records if r.levelname == "ERROR"]
            else:
                with pytest.raises(RenderError) as error:
                    await asyncio.wait_for(probe, 1)
                assert error.value.code == (
                    "RELEASE_UNKNOWN" if same_generation else "RENDER_FAILED"
                )
                assert channel.subscription_probe_error == (
                    "RELEASE_UNKNOWN"
                    if same_generation
                    else "MODEL_CONTEXT_STALE_SOURCE"
                )
                assert channel.suspended
                if not same_generation:
                    assert channel.consumer is None
            assert not submits and not (await journal.read())["operations"]
        finally:
            query_release.set()
            if probe is not None and not probe.done():
                probe.cancel()
                await asyncio.gather(probe, return_exceptions=True)
            await channel.close()
            await template.close()

    asyncio.run(scenario())


def test_checkpoint_recovers_original_preparing_candidate_and_committed_query(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
        CLIConditionChannel,
    )

    async def scenario():
        a, c, journal, receipts, submits, gate = await setup(tmp_path)
        sent = []

        class Socket:
            async def send(self, wire):
                sent.append(json.loads(wire))

        view = SimpleNamespace(
            descriptor=dict(
                sessionId="session",
                epoch=1,
                modelContext=dict(
                    protocol=c["protocol"],
                    required=True,
                    configHash=c["configHash"],
                    gameBuildHash=c["source"]["gameBuildHash"],
                    compilerVersion=1,
                    visible=False,
                ),
            ),
            client_id="consumer",
            control=Socket(),
            closed=False,
            stopped=False,
        )
        channel = CLIConditionChannel(view)
        channel.segment = c["segmentId"]
        channel.consumer = c["audience"]
        channel.create_adapter = lambda *_: a
        snapshot = copy.deepcopy(fixture()["messages"][0])
        snapshot.update(
            segmentId=c["segmentId"],
            controlSequence=5,
            baseControlSequence=3,
            desired={
                k: c[k]
                for k in (
                    "protocol",
                    "source",
                    "audience",
                    "segmentId",
                    "configHash",
                    "compilerVersion",
                    "lifecycle",
                    "prompt",
                    "references",
                    "onsetOccurrenceIds",
                    "ongoingOccurrenceIds",
                )
            },
            candidate=dict(
                conditionId=c["conditionId"],
                payloadHash=c["payloadHash"],
                status="preparing",
            ),
        )
        checkpoint = dict(
            type="model.context.checkpoint",
            afterControlSequence=0,
            snapshot=snapshot,
            condition=c,
            queries=[],
        )
        await channel.consume(checkpoint)
        gate.set()
        await a.preparation
        assert receipts[-1]["phase"] == "prepared" and not submits
        commit = {k: c[k] for k in a.envelope} | dict(
            type="model.condition.commit",
            conditionId=c["conditionId"],
            payloadHash=c["payloadHash"],
            controlSequence=6,
            effective=c["effective"],
        )
        await channel.consume(commit)
        await asyncio.gather(*a.preparations)
        original = receipts[-1]
        snapshot.update(
            controlSequence=8,
            baseControlSequence=6,
            desired=None,
            compiledConditionId=None,
            candidate=dict(
                conditionId=c["conditionId"],
                payloadHash=c["payloadHash"],
                status="committed",
            ),
        )
        checkpoint.update(
            afterControlSequence=6,
            queries=[
                {k: c[k] for k in a.envelope}
                | dict(
                    conditionId=c["conditionId"],
                    payloadHash=c["payloadHash"],
                    requestId=original["requestId"],
                    promptVersion=original["promptVersion"],
                )
            ],
        )
        await channel.consume(checkpoint)
        assert receipts[-1] == original and len(submits) == 1
        assert sent[-1]["consumerEpoch"] == c["audience"]["consumerEpoch"]
        assert channel.sequence == 8 and not channel.samples
        await channel.close()

    asyncio.run(scenario())


def test_old_epoch_and_changed_original_member_reject_before_resource(tmp_path):
    async def scenario():
        a, c, journal, receipts, submits, gate = await setup(tmp_path)
        stale = copy.deepcopy(c)
        stale["audience"]["consumerEpoch"] = "old"
        stale["payloadHash"] = payload_hash(stale)
        with pytest.raises(ValueError, match="STALE_SOURCE"):
            await handle_settled(a, stale)
        assert not a.preparation and not submits

        def retired():
            raise ValueError("MODEL_CONTEXT_STALE_SOURCE")

        a.check = retired
        with pytest.raises(ValueError, match="STALE_SOURCE"):
            await handle_settled(a, c)
        assert not a.preparation and not (await journal.read())["operations"]
        await a.close()

    asyncio.run(scenario())


def test_resource_reader_uses_final_auth_and_refuses_redirect(tmp_path, monkeypatch):
    import httpx

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import CLIView

    async def scenario():
        _, c, _, _, _, _ = await setup(tmp_path)
        handle = c["references"][0]["resource"]
        view = object.__new__(CLIView)
        view.descriptor = dict(
            sessionId=handle["sessionId"],
            epoch=handle["epoch"],
            baseUrl="http://127.0.0.1:4321",
            accessToken="final-relay-token",
            modelContext=dict(
                configHash=handle["configHash"], gameBuildHash=handle["gameBuildHash"]
            ),
        )
        seen = []

        def route(request):
            seen.append(request)
            assert request.headers["Authorization"] == "Bearer final-relay-token"
            assert request.url.host == "127.0.0.1" and request.url.port == 4321
            assert (
                request.url.path == "/v1/sessions/session/model-context/resources/angel"
            )
            assert dict(request.url.params) == {
                k: str(handle[k])
                for k in ("epoch", "gameBuildHash", "configHash", "sha256")
            }
            return httpx.Response(
                302, headers={"location": "https://untrusted.invalid/picture"}
            )

        original = httpx.AsyncClient
        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **kwargs: original(transport=httpx.MockTransport(route), **kwargs),
        )
        with pytest.raises(ValueError, match="RESOURCE_FAILED"):
            await view.read_model_context_resource(handle)
        assert len(seen) == 1
        bad = dict(handle, sessionId="other")
        with pytest.raises(ValueError, match="RESOURCE_IDENTITY"):
            await view.read_model_context_resource(bad)
        assert len(seen) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("with_startup", [False, True])
@pytest.mark.parametrize("manual_reset", [False, True, "preparing"])
def test_full_source_owner_initial_output_then_update_uses_single_mock_model(
    tmp_path, monkeypatch, with_startup, manual_reset
):
    import msgspec
    from test_whitebox_interactive import BrowserUI, Engine
    from test_whitebox_render_lifecycle import StrictLeases
    from test_whitebox_startup import request

    from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
        AsyncArchiveManager,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import (
        EngineFrame,
        VideoSpec,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig

    monkeypatch.setenv("WHITEBOX_MODEL_STARTUP_PREPARE", "1")

    async def scenario():
        spec = VideoSpec(32, 32)
        c = condition()
        c.update(
            references=[],
            onsetOccurrenceIds=[],
            ongoingOccurrenceIds=[],
            effective=dict(
                sourceFrameId=0, sourceTimeUs=0, simulationTick=0, policy="at-or-after"
            ),
        )
        receipts = []
        cap = dict(
            protocol=c["protocol"],
            required=True,
            configHash=c["configHash"],
            gameBuildHash=c["source"]["gameBuildHash"],
            compilerVersion=1,
            visible=False,
        )

        class Source(Engine):
            stream_id = "view"
            metadata = {}
            descriptor = {"modelContext": cap}

            async def bind_model_context(
                self, segment_id, create_adapter, start_suspended=False
            ):
                c["segmentId"] = segment_id
                c["payloadHash"] = payload_hash(c)

                async def emit(value):
                    receipts.append(value)

                self.adapter = create_adapter(c, emit, lambda: None)

                from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
                    CLIConditionChannel,
                )

                channel = CLIConditionChannel(
                    SimpleNamespace(
                        descriptor={"epoch": c["source"]["epoch"], "modelContext": cap},
                        transport_generation=0,
                        control=SimpleNamespace(send=AsyncMock()),
                        closed=False,
                        stopped=False,
                    )
                )
                channel.segment, channel.adapter = segment_id, self.adapter
                channel.probe_subscription_pause = AsyncMock()
                channel.suspended = start_suspended

                async def switch_segment(segment_id):
                    if c["segmentId"] == segment_id and not self.adapter.closed:
                        channel.suspended = False
                        return
                    await self.adapter.close()
                    c.update(
                        segmentId=segment_id,
                        conditionId="after-eos",
                        prompt="Current context after EOS",
                        controlSequence=8,
                    )
                    c["payloadHash"] = payload_hash(c)
                    self.adapter = create_adapter(c, emit, lambda: None)
                    channel.segment, channel.adapter = segment_id, self.adapter
                    channel.suspended = False

                channel.switch_segment = switch_segment
                return channel

            async def read_model_context_resource(self, handle):
                raise AssertionError("empty references")

            async def push(self):
                i = self.next
                self.next += 1
                h = header(c)
                h["source"].update(
                    sourceFrameId=1000 + i, sourceTimeUs=i * 50000, simulationTick=i
                )
                h.update(
                    outputFrameId=i,
                    outputPtsUs=i * 50000,
                    widthPixels=32,
                    heightPixels=32,
                    uiCompleteThroughUs=i * 50000,
                )
                await self.queue.put(
                    EngineFrame(
                        i,
                        spec.pts(i),
                        bytes([30]) * spec.frame_bytes,
                        {},
                        provenance=dict(capture_pts_us=i * 50000, cli_frame_header=h),
                    )
                )

        engine = Source(spec)
        leases = StrictLeases(spec)
        archives = AsyncArchiveManager(recorder_slots=3)
        journal = RenderJournal(
            MemoryRoomStore(),
            user_id="user",
            source_session_id="source",
            source_epoch="epoch",
            owner_epoch="room",
        )
        await journal.create()

        async def factory(**_):
            assert not leases.opens
            return engine

        owner = SourceSession(
            SimpleNamespace(
                runtime=SimpleNamespace(players={}),
                make_segment_leases=lambda *_: leases,
            ),
            journal,
            source_epoch="epoch",
            user_id="user",
            session_id="source",
            trace_id="context",
            config=WhiteboxConfig(
                engine_factory=factory,
                ui_factory=lambda **_: BrowserUI(),
                spool=tmp_path,
                spec=spec,
                room_owned=True,
                startup_seconds=5,
                drain_seconds=2,
            ),
            archives=archives,
        )
        messages = []

        async def consume():
            while True:
                m = msgspec.msgpack.decode(await owner.recv())
                messages.append(m)
                if m.get("type") in ("whitebox.frame", "frame_batch"):
                    owner.media_delivered()

        async def produce():
            while not engine.closed:
                await engine.push()
                await asyncio.sleep(0.025)

        async def wait(predicate):
            async with asyncio.timeout(8):
                while not predicate():
                    await asyncio.sleep(0.01)

        tasks = []
        startup = request(False)
        startup["initial_render"]["seed"] = 7
        try:
            await owner.send(
                msgspec.msgpack.encode(
                    dict(
                        type="init",
                        whitebox=dict(world_id="world"),
                        prompt="legacy preview",
                        seed=19,
                        **(
                            {
                                "source_operation": "create.manual",
                                "inference_control": "manual",
                            }
                            if manual_reset
                            else {}
                        ),
                        **({"startup": startup} if with_startup else {}),
                    )
                )
            )
            tasks = [asyncio.create_task(consume()), asyncio.create_task(produce())]
            await wait(
                lambda: (
                    owner.ready.is_set()
                    and any(m.get("type") == "session.ready" for m in messages)
                )
            )
            assert (
                owner.startup is None
                and not leases.opens
                and journal.snapshot["segment"] is None
            )
            ready = next(m for m in messages if m.get("type") == "session.ready")
            assert ready.get("display_mode", "whitebox") == (
                "model" if with_startup and not manual_reset else "whitebox"
            )
            assert ready["capabilities"]["model_context"] == "worldkit-model-context-v1"
            assert ready["capabilities"]["model_context_instruction"] is True
            assert ready["model_context_segment"] == {
                k: journal.snapshot["model_context_segment_ids"][k]
                for k in ("segment_id", "segment_epoch")
            }
            if with_startup and not manual_reset:
                assert ready["startup_request_id"] == startup["request_id"]
                assert "startup_prepare_protocol" not in ready["capabilities"]
            if manual_reset:
                from test_whitebox_render_contract import command

                await owner.command(
                    command(
                        "model.inference.configure",
                        "original-enable",
                        enabled=True,
                        expected_inference_revision=0,
                    )
                )
                if manual_reset == "preparing":
                    original_ids = copy.deepcopy(
                        journal.snapshot["model_context_segment_ids"]
                    )
                    assert journal.snapshot["segment"] is None and not leases.opens
                    c["source"].update(
                        presentationId="reset-before-init",
                        sdkEpoch=c["source"]["sdkEpoch"] + 1,
                    )
                    owner.context_channel.retire_source(
                        dict(
                            type="source.reset",
                            epoch=c["source"]["epoch"],
                            presentationId=c["source"]["presentationId"],
                            sdkEpoch=c["source"]["sdkEpoch"],
                        )
                    )
                    await asyncio.wait_for(
                        asyncio.shield(owner.context_channel.reset_task), 5
                    )
                    await wait(lambda: c["conditionId"] == "after-eos")
                    assert all(
                        journal.snapshot["model_context_segment_ids"][k] != v
                        for k, v in original_ids.items()
                    )
            await engine.adapter.handle(c)
            await engine.adapter.preparation
            assert receipts[-1]["phase"] == "prepared" and not leases.opens

            def commit(value, sequence):
                return {k: value[k] for k in engine.adapter.envelope} | dict(
                    type="model.condition.commit",
                    conditionId=value["conditionId"],
                    payloadHash=value["payloadHash"],
                    controlSequence=sequence,
                    effective=value["effective"],
                )

            await engine.adapter.handle(commit(c, 3))
            await wait(
                lambda: any(
                    r["phase"] == "applied" and r["conditionId"] == c["conditionId"]
                    for r in receipts
                )
            )
            assert (
                len(leases.opens) == 1 and leases.opens[0].init["prompt"] == c["prompt"]
            )
            if manual_reset:
                await wait(lambda: journal.snapshot["inference"]["state"] == "enabled")
                if manual_reset == "preparing":
                    original = journal.snapshot["operations"]["original-enable"]
                    assert (
                        original["state"] == "failed"
                        and original["code"] == "MODEL_CONTEXT_STALE_SOURCE"
                    )
                    assert journal.snapshot["inference"] == dict(
                        enabled=True, revision=1, state="enabled"
                    )
                    assert len(leases.opens) == leases.active == 1
                    assert (
                        len(
                            [
                                o
                                for o in journal.snapshot["operations"].values()
                                if o["op"] == "model.inference.configure"
                            ]
                        )
                        == 1
                    )
                    return
                original_enable = copy.deepcopy(
                    journal.snapshot["operations"]["original-enable"]
                )
            assert leases.opens[0].init["references"] == []
            assert leases.opens[0].init["seed"] == (7 if with_startup else 19)
            update = copy.deepcopy(c)
            update.update(
                conditionId="later", prompt="New persistent context", controlSequence=4
            )
            update["payloadHash"] = payload_hash(update)
            await engine.adapter.handle(update)
            await engine.adapter.preparation
            await engine.adapter.handle(commit(update, 5))
            await wait(
                lambda: any(
                    r["phase"] == "applied" and r["conditionId"] == "later"
                    for r in receipts
                )
            )
            assert (
                len(leases.opens) == 1
                and journal.snapshot["segment"]["accepted_version"] == 2
            )
            assert [
                r["promptVersion"] for r in receipts if r["phase"] == "applied"
            ] == [1, 2]
            first = owner.current_segment
            if manual_reset:
                c["source"].update(
                    presentationId="native-reset", sdkEpoch=c["source"]["sdkEpoch"] + 1
                )
                owner.context_channel.retire_source(
                    dict(
                        type="source.reset",
                        epoch=c["source"]["epoch"],
                        presentationId=c["source"]["presentationId"],
                        sdkEpoch=c["source"]["sdkEpoch"],
                    )
                )
                await asyncio.wait_for(
                    asyncio.shield(owner.context_channel.reset_task), 5
                )
                await wait(lambda: c["conditionId"] == "after-eos")
                assert first.closed_event.is_set() and leases.active == 0
            else:
                await first.stop()
                assert first.closed_event.is_set() and leases.active == 0
                assert not engine.closed and not owner.finished.is_set()
                state = await journal.read()
                await owner.command(
                    dict(
                        type="command",
                        op="render.context.start",
                        request_id="context-after-eos",
                        source_session_id="source",
                        source_epoch="epoch",
                        expected_render_revision=state["render_revision"],
                    )
                )
            await engine.adapter.handle(c)
            await engine.adapter.preparation
            await engine.adapter.handle(commit(c, 9))
            await wait(
                lambda: any(
                    r["phase"] == "applied" and r["conditionId"] == "after-eos"
                    for r in receipts
                )
            )
            assert len(leases.opens) == 2 and leases.active == 1
            assert leases.opens[1].init["prompt"] == "Current context after EOS"
            assert leases.opens[1].init["seed"] == (7 if with_startup else 19)
            assert owner.current_segment.segment_id != first.segment_id
            assert journal.snapshot["segment"]["accepted_version"] == 1
            if manual_reset:
                await wait(lambda: journal.snapshot["inference"]["state"] == "enabled")
                assert journal.snapshot["inference"] == dict(
                    enabled=True, revision=1, state="enabled"
                )
                assert (
                    journal.snapshot["operations"]["original-enable"] == original_enable
                )
                restarts = [
                    o
                    for o in journal.snapshot["operations"].values()
                    if o["op"] == "render.context.start"
                ]
                assert len(restarts) == 1 and restarts[0]["request_id"].startswith(
                    "inference:source-reset:"
                )
            before_late = len(receipts)
            await owner.operation_message(
                journal.snapshot["operations"][first.segment["request_id"]]
            )
            assert len(receipts) == before_late
            assert not any(m.get("type") == "error" for m in messages)
        finally:
            await owner.close()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await archives.close()
        assert (
            leases.active == 0 and engine.closed and all(w.closed for w in leases.opens)
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["completed", "cancelled", "failed"])
def test_applied_initial_condition_survives_segment_termination(tmp_path, outcome):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def scenario():
        a, c, journal, receipts, _, _ = await setup(tmp_path)
        operation = await journal.model_condition(c, [])
        evidence = dict(first_output_frame=0, num_frames=1)
        await journal.prompt_applied(c["segmentId"], 1, evidence)
        # Lose the initial applied receipt, then terminate the real segment owner.
        owner = object.__new__(SourceSession)
        owner.journal = journal

        async def status(*args, **kwargs):
            pass

        async def publish(operation):
            await a.original(operation["model_context"])

        owner.segment_status = status
        owner.operation_message = publish
        segment = SimpleNamespace(
            segment_id=c["segmentId"],
            segment=journal.snapshot["segment"],
            resource_state="released",
            archive_failure=False,
            pending_prompt=None,
        )
        await owner.segment_finished(
            segment, outcome, "LATE_FAILURE" if outcome == "failed" else None
        )
        query = {
            **operation["model_context"],
            "requestId": operation["request_id"],
            "promptVersion": 1,
        }
        await a.reconcile([query])
        assert receipts[-1]["phase"] == "applied"
        assert receipts[-1]["requestId"] == operation["request_id"]
        assert receipts[-1]["promptVersion"] == 1
        result = await journal.model_condition_query(query)
        assert (
            result["state"]
            == {"completed": "ended", "cancelled": "cancelled", "failed": "failed"}[
                outcome
            ]
        )
        assert result["first_output_frame"] == 0 and result["num_frames"] == 1
        assert len(journal.snapshot["operations"]) == 1
        await a.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "wrong", [{"requestId": "wrong-request"}, {"promptVersion": 999}]
)
def test_adapter_reconciliation_rejects_wrong_original_identity(tmp_path, wrong):
    async def scenario():
        a, c, journal, receipts, _, _ = await setup(tmp_path)
        operation = await journal.model_condition(c, [])
        query = {
            **operation["model_context"],
            "requestId": operation["request_id"],
            "promptVersion": 1,
            **wrong,
        }
        with pytest.raises(RenderError) as failure:
            await a.reconcile([query])
        assert failure.value.code == "REQUEST_ID_CONFLICT"
        assert receipts == []
        assert len(journal.snapshot["operations"]) == 1
        await a.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("barrier", ["checkpoint", "execution-only"])
@pytest.mark.parametrize("ordering", ["buffered", "blocked-known", "blocked-unknown"])
def test_context_recovery_retires_buffered_pixels_and_recredits(
    tmp_path, barrier, ordering
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import NativeCLIView
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_conditions import (
        PAYLOAD,
    )

    async def scenario():
        a, c, _, _, _, _ = await setup(tmp_path)
        sent, reads = [], []
        queue = asyncio.Queue()

        class Socket:
            async def send(self, wire):
                sent.append(json.loads(wire))

            async def recv(self):
                wire = await queue.get()
                reads.append(wire)
                return wire

        descriptor = dict(
            sessionId="session",
            epoch=1,
            worldBuildHash="a" * 64,
            uiBundleHash="b" * 64,
            modelContext=dict(
                protocol=c["protocol"],
                required=True,
                configHash=c["configHash"],
                gameBuildHash=c["source"]["gameBuildHash"],
                compilerVersion=1,
                visible=False,
            ),
            media=dict(kind="pixels"),
        )
        view = NativeCLIView(
            SimpleNamespace(
                descriptor=descriptor, spec=VideoSpec(2, 2, 24), prepared_delivery=None
            )
        )
        view.control = view.media = Socket()
        view.config = dict(mediaGeneration=1)
        view.configured.set()
        view.control_task = asyncio.create_task(asyncio.Event().wait())
        channel = view.model_context
        channel.consumer, channel.segment = c["audience"], c["segmentId"]
        channel.sample_floor = 1
        channel.create_adapter = lambda *_: a
        channel.bound.set()
        snapshot = copy.deepcopy(fixture()["messages"][0])
        snapshot.update(
            segmentId=c["segmentId"], desired={k: c[k] for k in PAYLOAD}, candidate=None
        )
        await channel.consume(snapshot)
        old = header(c)
        new = copy.deepcopy(old)
        new["outputFrameId"] += 1
        new["outputPtsUs"] += 1000
        new["source"]["sourceFrameId"] += 1
        new["source"]["sourceTimeUs"] += 1000
        new["uiCompleteThroughUs"] += 1000

        def sample(h, sequence):
            return dict(
                type="model.context.sample",
                epoch=1,
                consumerId="consumer",
                consumerEpoch=c["audience"]["consumerEpoch"],
                segmentId=c["segmentId"],
                header=h,
                metadata={
                    k: snapshot[k]
                    for k in (
                        "contextStateId",
                        "contextRevision",
                        "viewRevision",
                        "compiledConditionId",
                    )
                }
                | dict(throughControlSequence=sequence),
            )

        def packet(h):
            envelope = dict(
                kind="worldkit-pixel-frame",
                schemaVersion=2,
                pixelFormat="rgb24",
                colorSpace="srgb",
                widthPixels=2,
                heightPixels=2,
                rowStrideBytes=6,
                payloadBytes=12,
                worldBuildHash="a" * 64,
                uiBundleHash="b" * 64,
                input=h,
            )
            encoded = json.dumps(envelope).encode()
            return len(encoded).to_bytes(4, "big") + encoded + bytes(12)

        async def until(predicate):
            async with asyncio.timeout(0.5):
                while not predicate():
                    await asyncio.sleep(0)

        stream = view.frames()
        pending = None
        try:
            if ordering != "blocked-unknown":
                # Known-but-not-contiguous metadata also must be retired and wake its reader.
                await channel.consume(
                    sample(old, 2 if ordering == "blocked-known" else 1)
                )
            if ordering != "buffered":
                await queue.put(packet(old))
                pending = asyncio.create_task(anext(stream))
                await until(lambda: len(reads) == 1)
                await asyncio.sleep(0)
                assert not pending.done()
            if barrier == "checkpoint":
                recovered = copy.deepcopy(snapshot)
                recovered.update(controlSequence=5, baseControlSequence=3)
                await channel.consume(
                    dict(
                        type="model.context.checkpoint",
                        afterControlSequence=1,
                        snapshot=recovered,
                        condition=None,
                        queries=[],
                    )
                )
                floor = 6
            else:
                recovered = copy.deepcopy(snapshot)
                recovered.update(
                    controlSequence=2, desired=None, compiledConditionId=None
                )
                await channel.consume(recovered)
                floor = 3
            if ordering == "blocked-unknown":
                # A barrier cannot guess the generation of an unmatched pixel.
                await asyncio.sleep(0)
                assert not pending.done()
                await channel.consume(sample(old, 1))
            if ordering == "buffered":
                await queue.put(packet(old))
                pending = asyncio.create_task(anext(stream))
            await queue.put(packet(new))
            await until(lambda: len(reads) == 2)
            await asyncio.sleep(0)
            # Old pixel is discarded/recredited; fresh unknown metadata is never admitted.
            assert not pending.done()
            assert [m["frames"] for m in sent if m["type"] == "media.credit"] == [2, 1]
            fresh = copy.deepcopy(snapshot)
            fresh.update(controlSequence=floor)
            await channel.consume(fresh)
            await asyncio.sleep(0)
            assert not pending.done()
            await channel.consume(sample(new, floor))
            frame = await asyncio.wait_for(pending, 0.5)
            assert frame.index == 0
            assert frame.provenance["cli_frame_header"]["outputFrameId"] == 10
            assert (
                frame.provenance["model_context_sample"]["throughControlSequence"]
                == floor
            )
        finally:
            if pending and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await stream.aclose()
            await channel.close()
            view.control_task.cancel()
            await asyncio.gather(view.control_task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("pending_count", [1, 128])
def test_slow_pixels_retain_their_snapshot_through_control_churn(pending_count):
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_conditions import (
        PAYLOAD,
    )

    """Control progress cannot invalidate an already admitted, unconsumed pixel."""
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
        CLIConditionChannel,
    )

    async def scenario():
        c = condition()

        async def send(_):
            pass

        channel = CLIConditionChannel(
            SimpleNamespace(
                descriptor=dict(
                    sessionId=c["source"]["sessionId"],
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
        )
        channel.consumer, channel.segment = c["audience"], c["segmentId"]
        channel.sample_floor = 1
        channel.create_adapter = lambda e, *_: SimpleNamespace(envelope=e)
        snapshot = copy.deepcopy(fixture()["messages"][0])
        snapshot.update(
            segmentId=c["segmentId"], desired={k: c[k] for k in PAYLOAD}, candidate=None
        )
        await channel.consume(snapshot)
        metadata = {
            k: snapshot[k]
            for k in (
                "contextStateId",
                "contextRevision",
                "viewRevision",
                "compiledConditionId",
            )
        } | dict(throughControlSequence=1)

        def sample(h, m):
            return dict(
                type="model.context.sample",
                epoch=c["source"]["epoch"],
                consumerId=c["audience"]["consumerId"],
                consumerEpoch=c["audience"]["consumerEpoch"],
                segmentId=c["segmentId"],
                header=h,
                metadata=m,
            )

        await channel.consume(sample(header(c), metadata))
        if pending_count == 128:
            for sequence in range(2, 129):
                current = copy.deepcopy(snapshot)
                current.update(
                    controlSequence=sequence,
                    contextStateId=f"context:{sequence}",
                    contextRevision=sequence,
                )
                await channel.consume(current)
                h = header(c)
                h["outputFrameId"] += sequence
                m = {k: current[k] for k in metadata if k != "throughControlSequence"}
                await channel.consume(
                    sample(h, m | dict(throughControlSequence=sequence))
                )
            current = copy.deepcopy(snapshot)
            current.update(
                controlSequence=129, contextStateId="context:129", contextRevision=129
            )
            with pytest.raises(ValueError, match="MODEL_CONTEXT_BACKPRESSURE"):
                await channel.consume(current)
            assert len(channel.snapshots) == 128
            assert await channel.frame(header(c)) == metadata
            await channel.consume(current)
            assert len(channel.snapshots) == 128
            return
        for sequence in range(2, 260):
            current = copy.deepcopy(snapshot)
            current.update(
                controlSequence=sequence,
                contextStateId=f"context:{sequence}",
                contextRevision=sequence,
            )
            await channel.consume(current)
        assert await channel.frame(header(c)) == metadata
        assert len(channel.snapshots) <= 128
        # Once consumed, that mapping is eligible for bounded retirement.
        snapshot.update(
            controlSequence=260, contextStateId="context:260", contextRevision=260
        )
        await channel.consume(snapshot)
        assert channel.snapshots[0][0] > 1

    asyncio.run(scenario())


def test_explicit_context_segment_restart_uses_current_full_condition(tmp_path):
    """The lifecycle owner may restart after EOS; events cannot restart a model."""

    async def scenario():
        a, c, journal, receipts, submits, gate = await setup(tmp_path)
        gate.set()
        await handle_settled(a, c)
        await a.preparation
        commit = dict(
            request_identity(c),
            type="model.condition.commit",
            controlSequence=c["controlSequence"] + 1,
            effective=c["effective"],
        )
        await handle_settled(a, commit)
        old = await journal.read()
        sid = old["segment"]["segment_id"]
        await journal.prompt_applied(sid, 1, dict(first_output_frame=0, num_frames=1))
        # This is the segment state reached after EOS and confirmed release.
        await journal.update(
            lambda s: s["segment"].update(state="ended", resource_state="released")
        )
        command = dict(
            op="render.context.start",
            request_id="explicit-next",
            source_session_id="source",
            source_epoch="epoch",
            expected_render_revision=old["render_revision"],
        )
        next_op = await journal.start_model_context_segment(command)
        assert next_op["segment_id"] != sid
        assert (await journal.start_model_context_segment(command))["duplicate"]
        ids = await journal.reserve_model_context_segment()
        assert ids["segment_id"] == next_op["segment_id"]
        current = dict(
            c,
            segmentId=ids["segment_id"],
            conditionId="current-after-eos",
            prompt="current state <Picture 1>",
            onsetOccurrenceIds=[],
            controlSequence=10,
        )
        current["payloadHash"] = payload_hash(current)
        next_adapter = SourceConditions(
            envelope=current,
            assets=a.assets,
            journal=journal,
            read_resource=a.read_resource,
            validate_reference=a.validate_reference,
            emit=a.emit,
            submit=a.submit,
            capabilities=lambda: dict(
                max_references=9, max_reference_bytes=8 << 20, reference_updates=False
            ),
            check=lambda: None,
        )
        await handle_settled(next_adapter, current)
        await next_adapter.preparation
        assert receipts[-1]["phase"] == "prepared"
        await handle_settled(
            next_adapter,
            dict(
                request_identity(current),
                type="model.condition.commit",
                controlSequence=11,
                effective=current["effective"],
            ),
        )
        state = await journal.read()
        assert state["source_state"] == "ready"
        assert state["segment"]["segment_id"] == ids["segment_id"]
        assert state["segment"]["prompt"] == current["prompt"]
        assert state["segment"]["prompt_version"] == 1
        assert not state["segment"]["prompt_events"]
        assert (
            state["segment"]["model_session_id"] != old["segment"]["model_session_id"]
        )
        assert (await journal.model_condition_query(request_identity(c)))[
            "state"
        ] == "applied"
        with pytest.raises(RenderError) as blocked:
            await journal.start_model_context_segment(
                dict(
                    command,
                    request_id="too-soon",
                    expected_render_revision=state["render_revision"],
                )
            )
        assert blocked.value.code == "RENDER_TRANSITION_IN_PROGRESS"
        await a.close()
        await next_adapter.close()

    asyncio.run(scenario())


def test_context_channel_segment_handoff_preserves_consumer_and_control_cursor(
    tmp_path,
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
        CLIConditionChannel,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_conditions import (
        PAYLOAD,
    )

    async def scenario():
        a, c, _, _, _, _ = await setup(tmp_path)
        sent = []

        class Socket:
            async def send(self, wire):
                sent.append(json.loads(wire))

        view = SimpleNamespace(
            descriptor=dict(
                sessionId="session",
                epoch=1,
                modelContext=dict(
                    protocol=c["protocol"],
                    required=True,
                    configHash=c["configHash"],
                    gameBuildHash=c["source"]["gameBuildHash"],
                    compilerVersion=1,
                    visible=False,
                ),
            ),
            client_id="consumer",
            control=Socket(),
            closed=False,
            stopped=False,
        )
        channel = CLIConditionChannel(view)
        channel.segment, channel.consumer = c["segmentId"], c["audience"]
        channel.adapter = a
        channel.sequence = 5
        created = []

        def create(e, emit, check):
            result = SourceConditions(
                envelope=e,
                assets=a.assets,
                journal=a.journal,
                read_resource=a.read_resource,
                emit=emit,
                submit=a.submit,
                capabilities=a.capabilities,
                validate_reference=a.validate_reference,
                check=check,
            )
            created.append(result)
            return result

        channel.create_adapter = create
        await channel.switch_segment("next-segment")
        subscribed = dict(
            type="model.context.subscribed",
            epoch=1,
            protocol=c["protocol"],
            consumer=c["audience"],
            segmentId=c["segmentId"],
            pendingSegmentId="next-segment",
            sampleControlFloor=6,
        )
        await channel.consume(subscribed)
        assert channel.adapter is a and channel.sequence == 5
        # Reconnect may discover that Host already switched, before the new
        # snapshot reached this adapter. Resume against Host's current envelope.
        subscribed.update(segmentId="next-segment")
        subscribed.pop("pendingSegmentId")
        await channel.consume(subscribed)
        assert sent[-1]["message"]["segmentId"] == "next-segment"
        current = dict(c, segmentId="next-segment", onsetOccurrenceIds=[])
        snapshot = copy.deepcopy(fixture()["messages"][0])
        snapshot.update(
            segmentId="next-segment",
            controlSequence=6,
            baseControlSequence=5,
            desired={k: current[k] for k in PAYLOAD},
            candidate=None,
        )
        await channel.consume(dict(snapshot, controlSequence=7))
        assert not a.closed and channel.adapter is a and channel.sequence == 5
        assert sent[-1]["message"]["type"] == "model.context.resume"
        assert sent[-1]["message"]["segmentId"] == "next-segment"
        await channel.consume(snapshot)
        assert a.closed and len(created) == 1
        assert channel.segment == "next-segment" and channel.sequence == 6
        assert channel.consumer == c["audience"]
        assert sent[-1]["type"] == "model.context.ack"
        # Host replay keeps one cursor across segment handoff. Its consumed
        # old prefix cannot roll back the adapter or query a new segment.
        for kind in (
            "model.context.snapshot",
            "model.condition",
            "model.condition.commit",
            "model.condition.withdraw",
        ):
            await channel.consume(dict(c, type=kind, controlSequence=5))
        assert channel.sequence == 6 and len(created) == 1
        assert channel.adapter is created[0] and not created[0].candidate
        forged = copy.deepcopy(c)
        forged["source"]["sourceEpoch"] = "foreign-source"
        with pytest.raises(ValueError, match="STALE_SOURCE"):
            await channel.consume(dict(forged, controlSequence=5))
        with pytest.raises(ValueError, match="STALE_SOURCE"):
            await channel.consume(dict(c, controlSequence=7))
        with pytest.raises(ValueError, match="STALE_SOURCE"):
            await channel.admit_envelope(c)
        await channel.close()

    asyncio.run(scenario())


def test_native_cli_reconnects_real_sockets_and_queries_original_commit(tmp_path):
    """Drop accepted receipt transport; same consumer/cursor and one journal init."""
    from urllib.parse import parse_qs, urlsplit

    from websockets.asyncio.server import serve

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import NativeCLIView
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_conditions import (
        PAYLOAD,
    )

    async def scenario():
        a, c, journal, _, submits, gate = await setup(tmp_path)
        gate.set()
        subscriptions, originals, resumes, clients, errors = [], [], [], [], []
        media = asyncio.Queue()
        reconnect = asyncio.Event()
        config = dict(
            type="media.config",
            epoch=1,
            mediaGeneration=1,
            config=dict(
                kind="pixels",
                pixelFormat="rgb24",
                widthPixels=2,
                heightPixels=2,
                rowStrideBytes=6,
                colorSpace="srgb",
                fps=24,
            ),
        )
        connections = 0

        class Renderer:
            async def consume_ui(self, value):
                pass

            async def close(self):
                pass

        async def handler(socket):
            nonlocal connections
            try:
                path = urlsplit(socket.request.path)
                clients.append(parse_qs(path.query)["clientId"][0])
                if path.path == "/pixels":
                    packet = asyncio.create_task(media.get())
                    closed = asyncio.create_task(socket.wait_closed())
                    done, _ = await asyncio.wait(
                        (packet, closed), return_when=asyncio.FIRST_COMPLETED
                    )
                    if closed in done:
                        packet.cancel()
                        await asyncio.gather(packet, return_exceptions=True)
                        return
                    closed.cancel()
                    await asyncio.gather(closed, return_exceptions=True)
                    await socket.send(packet.result())
                    await socket.wait_closed()
                    return
                connections += 1
                attempt = connections
                c["audience"]["consumerId"] = clients[-1]
                c["payloadHash"] = payload_hash(c)
                snapshot = copy.deepcopy(fixture()["messages"][0])
                snapshot.update(
                    segmentId=c["segmentId"],
                    audience=copy.deepcopy(c["audience"]),
                    desired={k: c[k] for k in PAYLOAD},
                    candidate=None,
                )

                async def send(value):
                    await socket.send(json.dumps(value))

                await send(dict(type="session.ready", epoch=1, role="spectator"))
                await send(config)
                async for wire in socket:
                    value = json.loads(wire)
                    kind = value["type"]
                    if kind == "model.context.subscribe":
                        subscriptions.append(value)
                        await send(
                            dict(
                                type="model.context.subscribed",
                                epoch=1,
                                protocol=c["protocol"],
                                consumer=c["audience"],
                                segmentId=c["segmentId"],
                                sampleControlFloor=1 if attempt == 1 else 4,
                            )
                        )
                        if attempt == 1:
                            await send(snapshot)
                            await send(c)
                    elif kind == "model.context.control":
                        message = value["message"]
                        if message["type"] == "model.context.resume":
                            resumes.append(message)
                            assert attempt == 2 and originals
                            assert message["afterControlSequence"] == 3
                            snapshot.update(controlSequence=4, baseControlSequence=3)
                            await send(snapshot)
                            frame = header(c)
                            await send(
                                dict(
                                    type="model.context.sample",
                                    epoch=1,
                                    consumerId=c["audience"]["consumerId"],
                                    consumerEpoch=c["audience"]["consumerEpoch"],
                                    segmentId=c["segmentId"],
                                    header=frame,
                                    metadata={
                                        k: snapshot[k]
                                        for k in (
                                            "contextStateId",
                                            "contextRevision",
                                            "viewRevision",
                                            "compiledConditionId",
                                        )
                                    }
                                    | dict(throughControlSequence=4),
                                )
                            )
                            envelope = dict(
                                kind="worldkit-pixel-frame",
                                schemaVersion=2,
                                pixelFormat="rgb24",
                                colorSpace="srgb",
                                widthPixels=2,
                                heightPixels=2,
                                rowStrideBytes=6,
                                payloadBytes=12,
                                worldBuildHash="a" * 64,
                                uiBundleHash="b" * 64,
                                input=frame,
                            )
                            encoded = json.dumps(envelope).encode()
                            await media.put(
                                len(encoded).to_bytes(4, "big") + encoded + bytes(12)
                            )
                            reconnect.set()
                        elif message.get("phase") == "prepared" and attempt == 1:
                            await send(
                                dict(
                                    request_identity(c),
                                    type="model.condition.commit",
                                    controlSequence=3,
                                    effective=c["effective"],
                                )
                            )
                        elif message.get("phase") == "accepted":
                            originals.append(message)
                            if attempt == 1:
                                # Commit is durable; deliberately lose its Host acknowledgement.
                                await socket.close(code=1011)
                                return
                    elif kind == "model.context.disconnect":
                        return
            except Exception as error:
                errors.append(error)
                raise

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            descriptor = dict(
                sessionId="session",
                epoch=1,
                worldBuildHash="a" * 64,
                uiBundleHash="b" * 64,
                media=dict(kind="pixels"),
                accessMode="capability",
                accessToken="a" * 48,
                streams={
                    k: dict(url=f"ws://127.0.0.1:{port}/{k}") for k in ("ui", "pixels")
                },
                modelContext=dict(
                    protocol=c["protocol"],
                    required=True,
                    configHash=c["configHash"],
                    gameBuildHash=c["source"]["gameBuildHash"],
                    compilerVersion=1,
                    visible=False,
                ),
            )
            view = NativeCLIView(
                SimpleNamespace(
                    descriptor=descriptor,
                    spec=VideoSpec(2, 2, 24),
                    prepared_delivery=None,
                    external_input=True,
                    renderer_factory=lambda **_: Renderer(),
                )
            )
            try:
                await view.start()

                def create(e, emit, check):
                    a.envelope = {k: e[k] for k in a.envelope}
                    a.emit, a.check = emit, check
                    return a

                await view.bind_model_context(
                    segment_id=c["segmentId"], create_adapter=create
                )
                async with asyncio.timeout(5):
                    frame = await anext(view.frames())
                assert not errors
                assert reconnect.is_set() and len(subscriptions) == 2
                assert subscriptions[0] == subscriptions[1]
                assert len(set(clients)) == 1
                assert view.transport_generation == 2
                assert (
                    len(submits) == 1 and len((await journal.read())["operations"]) == 1
                )
                assert len(originals) >= 2 and originals[0] == originals[1]
                assert view.model_context.consumer == c["audience"]
                assert (
                    frame.provenance["model_context_sample"]["throughControlSequence"]
                    == 4
                )
                assert view.model_context.sequence == 4 and not view.model_context.error
            finally:
                await view.close()
                # Release any pre-recovery socket handler still waiting for pixels.
                await media.put(b"")

    asyncio.run(scenario())


def test_source_owner_context_restart_is_lifecycle_only_and_idempotent(tmp_path):
    import time

    import msgspec

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def scenario():
        a, c, journal, _, _, _ = await setup(tmp_path)
        await journal.model_condition(c, [])
        await journal.prompt_applied(
            c["segmentId"], 1, dict(first_output_frame=0, num_frames=1)
        )
        await journal.update(
            lambda s: s["segment"].update(state="ended", resource_state="released")
        )
        switches = []

        class Channel:
            async def switch_segment(self, segment):
                switches.append(segment)

        source = object.__new__(SourceSession)
        source.journal, source.world_driven = journal, True
        source.context_channel = Channel()
        source.binding_ended = source.closed = False
        source.outgoing = asyncio.Queue()
        source.source_conditions = None
        source.engine = SimpleNamespace(stream_id="view")
        source.session_id, source.source_epoch = "source", "epoch"
        source.control_tokens, source.control_token_time = 4.0, time.monotonic()
        state = await journal.read()
        command = dict(
            type="command",
            op="render.context.start",
            request_id="restart",
            source_session_id="source",
            source_epoch="epoch",
            expected_render_revision=state["render_revision"],
        )
        from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import (
            RenderPipeline,
        )

        pipeline = object.__new__(RenderPipeline)
        pipeline.session = source
        pipeline.generation = 1
        pipeline.subscription = SimpleNamespace(closed=False, generation=1)
        pipeline.member_source = SimpleNamespace(check=lambda: None)
        pipeline.command_queue = asyncio.Queue(maxsize=32)
        pipeline.command_bytes, pipeline.command_byte_limit = 0, 1 << 20

        async def member_command(value):
            pipeline.accept(pipeline.subscription, msgspec.msgpack.encode(value))
            worker = asyncio.create_task(pipeline._commands())
            try:
                async with asyncio.timeout(1):
                    while source.outgoing.empty():
                        await asyncio.sleep(0)
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

        for forbidden in ("render.start", "render.replace", "prompt.replace"):
            with pytest.raises(ValueError, match="SDK player owns"):
                pipeline.accept(
                    pipeline.subscription,
                    msgspec.msgpack.encode(dict(command, op=forbidden)),
                )
        await member_command(
            dict(command, request_id="manual-prompt", prompt="untrusted")
        )
        assert (
            msgspec.msgpack.decode(await source.outgoing.get())["code"]
            == "INVALID_COMMAND"
        )
        await member_command(command)
        accepted = msgspec.msgpack.decode(await source.outgoing.get())
        assert (
            accepted["type"] == "operation.result" and accepted["state"] == "accepted"
        )
        await member_command(command)
        duplicate = msgspec.msgpack.decode(await source.outgoing.get())
        assert duplicate["duplicate"] and switches == [accepted["segment_id"]] * 2
        assert (await journal.read())["segment"]["state"] == "ended"
        assert len((await journal.read())["operations"]) == 2
        # Released segment identity may not be reused while release is unknown.
        await journal.update(
            lambda s: s["segment"].update(resource_state="reconciling")
        )
        with pytest.raises(RenderError) as blocked:
            await journal.start_model_context_segment(
                dict(
                    command,
                    request_id="blocked",
                    expected_render_revision=(await journal.read())["render_revision"],
                )
            )
        assert blocked.value.code == "RENDER_TRANSITION_IN_PROGRESS"
        await a.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure", ["control-error", "control-timeout", "source-ended", "cancelled"]
)
def test_model_subscription_wait_preserves_source_errors_and_cancellation(failure):
    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import NativeCLIView

    async def scenario():
        entered = asyncio.Event()
        suspended = []

        class SubscriptionEvent(asyncio.Event):
            async def wait(self):
                entered.set()
                return await super().wait()

        async def suspend(code):
            suspended.append(code)

        view = object.__new__(NativeCLIView)
        view.finish_requested = asyncio.Event()
        view.configured = asyncio.Event()
        view.configured.set()
        view.end_reason = None
        view.model_context = SimpleNamespace(
            suspended=False,
            bound=SubscriptionEvent(),
            check=lambda: None,
            suspend=suspend,
        )

        async def control():
            await entered.wait()
            if failure == "control-error":
                raise ValueError("Independent control failure")
            if failure == "control-timeout":
                raise TimeoutError("Independent control timeout")
            if failure == "source-ended":
                view.end_reason = "stopped"
                return
            await asyncio.Event().wait()

        view.control_task = asyncio.create_task(control())
        iterator = view.frames()
        pending = asyncio.create_task(anext(iterator))
        try:
            async with asyncio.timeout(1):
                await entered.wait()
                if failure == "cancelled":
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                else:
                    expected = {
                        "control-error": ValueError,
                        "control-timeout": TimeoutError,
                        "source-ended": StopAsyncIteration,
                    }[failure]
                    with pytest.raises(expected):
                        await pending
            assert not suspended
        finally:
            pending.cancel()
            view.control_task.cancel()
            await asyncio.gather(pending, view.control_task, return_exceptions=True)
            await iterator.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure",
    [
        "subscription-timeout",
        "resubscription-timeout",
        "sample-timeout",
        "host-error",
        "protocol-error",
        "backpressure",
    ],
)
def test_mapping_recovery_exhaustion_keeps_preview_and_stops_only_model(
    tmp_path, monkeypatch, failure
):
    from sglang.multimodal_gen.runtime.realtime.whitebox import source_condition_channel
    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import NativeCLIView
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def scenario():
        _, c, journal, _, _, _ = await setup(tmp_path)
        descriptor = dict(
            sessionId="session",
            epoch=1,
            worldBuildHash="a" * 64,
            uiBundleHash="b" * 64,
            media=dict(kind="pixels"),
            modelContext=dict(
                protocol=c["protocol"],
                required=True,
                configHash=c["configHash"],
                gameBuildHash=c["source"]["gameBuildHash"],
                compilerVersion=1,
                visible=False,
            ),
        )
        view = NativeCLIView(
            SimpleNamespace(
                descriptor=descriptor, spec=VideoSpec(2, 2, 24), prepared_delivery=None
            )
        )
        sent, stopped, status, recoveries = [], [], [], []
        packets = asyncio.Queue()

        class Socket:
            async def send(self, wire):
                sent.append(json.loads(wire))

            async def recv(self):
                if failure == "resubscription-timeout" and not recoveries:
                    raise OSError("The old media transport disconnected")
                return await packets.get()

            async def close(self):
                pass

        view.control = view.media = Socket()
        view.control_task = asyncio.create_task(asyncio.Event().wait())
        view.config, view.client_id = dict(mediaGeneration=1), "consumer"
        view.configured.set()
        if failure != "subscription-timeout":
            view.model_context.bound.set()
        owner = object.__new__(SourceSession)
        stopped_event = asyncio.Event()

        async def stop(cancel):
            stopped.append(cancel)

        segment = SimpleNamespace(
            publishing=True,
            source_view=SimpleNamespace(stop=stopped_event.set),
            stop=stop,
        )
        owner.journal = journal
        owner.current_segment = segment
        owner.control_tasks = set()

        async def record_status(state, **fields):
            status.append((state, fields))

        owner._status = record_status
        view.model_context.on_recovery_failed = owner.context_recovery_failed

        async def recover(generation):
            recoveries.append(generation)
            view.transport_generation += 1
            if failure == "resubscription-timeout":
                # Pixel/control transport is ready, but the model subscription
                # on that replacement connection never receives confirmation.
                view.model_context.bound.clear()

        view.recover_model_context_transport = recover

        def packet(index):
            h = header(c)
            h["outputFrameId"] = index
            h["source"]["sourceFrameId"] = index
            h["source"]["sourceTimeUs"] += view.spec.pts(index)
            h["source"]["simulationTick"] += index
            h["outputPtsUs"] += view.spec.pts(index)
            h["uiCompleteThroughUs"] = h["source"]["sourceTimeUs"]
            envelope = dict(
                kind="worldkit-pixel-frame",
                schemaVersion=2,
                pixelFormat="rgb24",
                colorSpace="srgb",
                widthPixels=2,
                heightPixels=2,
                rowStrideBytes=6,
                payloadBytes=12,
                worldBuildHash="a" * 64,
                uiBundleHash="b" * 64,
                input=h,
            )
            wire = json.dumps(envelope).encode()
            return len(wire).to_bytes(4, "big") + wire + bytes(12)

        for index in range(3):
            await packets.put(packet(index))
        iterator = view.frames()
        pending = None
        try:
            if failure in {"subscription-timeout", "resubscription-timeout"}:
                async with asyncio.timeout(1):
                    frame = await anext(iterator)
                code = "MODEL_CONTEXT_SUBSCRIPTION_TIMEOUT"
                assert len(recoveries) == (failure == "resubscription-timeout")
            elif failure == "sample-timeout":
                async with asyncio.timeout(1):
                    frame = await anext(iterator)
                code = "MODEL_CONTEXT_SAMPLE_RECOVERY_EXPIRED"
                assert len(recoveries) == 1
            else:
                channel = view.model_context
                channel.consumer = c["audience"]
                channel.task = asyncio.create_task(channel.run())
                entered = asyncio.Event()
                original_frame = channel.frame

                async def waiting_frame(h):
                    entered.set()
                    return await original_frame(h)

                channel.frame = waiting_frame
                pending = asyncio.create_task(anext(iterator))
                async with asyncio.timeout(1):
                    await entered.wait()
                code = {
                    "host-error": "MODEL_CONTEXT_HISTORY_UNAVAILABLE",
                    "backpressure": "MODEL_CONTEXT_BACKPRESSURE",
                    "protocol-error": "MODEL_CONTEXT_PROTOCOL_INVALID",
                }[failure]
                if failure == "backpressure":
                    for _ in range(129):
                        channel.offer({"type": "model.context.receipt-result"})
                else:
                    channel.offer(
                        dict(type="model.context.error", epoch=1, code=code)
                        if failure == "host-error"
                        else {"type": "model.condition"}
                    )
                async with asyncio.timeout(1):
                    frame = await pending
                assert not recoveries
            await asyncio.gather(*owner.control_tasks)
            assert stopped == [True] and stopped_event.is_set()
            assert not segment.publishing and view.model_context.suspended
            assert status == [
                (
                    "context_recovery_required",
                    dict(code=code),
                )
            ]
            assert "model_context_sample" not in frame.provenance
            assert not view.stopped and not view.closed
            assert any(m["type"] == "model.context.preview" for m in sent)
            assert not view.model_context.error
            if failure in {"subscription-timeout", "resubscription-timeout"}:
                await view.model_context.consume(
                    dict(
                        type="model.context.subscribed",
                        epoch=1,
                        protocol=c["protocol"],
                        segmentId=c["segmentId"],
                        consumer=c["audience"],
                        sampleControlFloor=1,
                    )
                )
                assert view.model_context.suspended
                async with asyncio.timeout(1):
                    following = await anext(iterator)
                assert following.index == frame.index + 1
                assert "model_context_sample" not in following.provenance
                assert sum(m["type"] == "model.context.preview" for m in sent) == 1
        finally:
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await iterator.aclose()
            await view.close()

    from sglang.multimodal_gen.runtime.realtime.whitebox import cli_room

    # Only deadline scenarios need accelerated clocks. Explicit Host/protocol
    # failures must not race an unrelated 10 ms sample timer on a loaded runner;
    # their outer one-second bound still proves prompt failure handling.
    if failure == "sample-timeout":
        monkeypatch.setattr(source_condition_channel, "SAMPLE_WAIT_SECONDS", 0.01)
    if failure in {"subscription-timeout", "resubscription-timeout"}:
        monkeypatch.setattr(cli_room, "MODEL_CONTEXT_SUBSCRIPTION_SECONDS", 0.01)
    asyncio.run(scenario())


def test_recovery_failure_between_commit_and_lease_cannot_start_from_preview(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def scenario():
        a, c, journal, _, _, _ = await setup(tmp_path)
        operation = await journal.model_condition(c, [])
        source = object.__new__(SourceSession)
        source.segment_lock = asyncio.Lock()
        source.current_segment = None
        source.binding_ended = False
        source.journal = journal
        source.context_channel = SimpleNamespace(suspended=True)
        source.assets = a.assets
        replies = []

        async def publish(value):
            replies.append(value)

        source.operation_message = publish
        await source._transition_segment(operation, [], None)
        assert replies[-1]["state"] == "cancelled" and replies[-1]["never_applied"]
        assert (await journal.read())["segment"]["state"] == "cancelled"
        assert (await journal.read())["source_state"] == "ready"
        assert source.current_segment is None
        with pytest.raises(RenderError) as rejected:
            await source.submit_source_condition(c, [])
        assert rejected.value.code == "MODEL_UNAVAILABLE"
        await a.close()

    asyncio.run(scenario())


def test_close_waits_for_withdrawn_image_commit_before_releasing_assets(
    tmp_path, monkeypatch
):
    async def scenario():
        a, c, _, receipts, submits, gate = await setup(tmp_path)
        gate.set()
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed_thread(fn, *args):
            entered.set()
            await release.wait()
            return fn(*args)

        monkeypatch.setattr(asyncio, "to_thread", delayed_thread)
        closing = old = successor = None
        try:
            await handle_settled(a, c)
            old = a.preparation
            await entered.wait()
            await handle_settled(
                a,
                {
                    **request_identity(c),
                    "type": "model.condition.withdraw",
                    "controlSequence": 3,
                    "reason": "obsolete",
                },
            )
            next_c = dict(c, conditionId="successor", controlSequence=4)
            next_c["payloadHash"] = payload_hash(next_c)
            await handle_settled(a, next_c)
            successor = a.preparation
            closing = asyncio.create_task(a.close())
            await asyncio.sleep(0.01)
            assert not closing.done(), (
                "close returned while a withdrawn asset commit was alive"
            )
            release.set()
            await asyncio.wait_for(closing, 1)
            assert old.done() and successor.done()
            a.assets.release_memory()
            await asyncio.sleep(0)
            assert a.assets.bytes == 0 and a.assets.assets == {}
            assert not submits and all(r["phase"] == "queued" for r in receipts)
        finally:
            release.set()
            if closing is not None:
                await closing
            await asyncio.gather(
                *(t for t in (old, successor) if t), return_exceptions=True
            )
            await a.close()

    asyncio.run(scenario())


def test_declared_model_startup_wait_does_not_consume_play_budget(tmp_path):
    import time

    from sglang.multimodal_gen.runtime.realtime.source_prepare import V2_PROTOCOL
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.startup import PlayStartup

    async def scenario():
        _, c, journal, _, _, _ = await setup(tmp_path)
        source = object.__new__(SourceSession)
        source.member_prefill = None
        source.engine = SimpleNamespace(
            descriptor={
                "modelContext": dict(
                    protocol=c["protocol"],
                    required=True,
                    configHash=c["configHash"],
                    gameBuildHash=c["source"]["gameBuildHash"],
                    compilerVersion=1,
                    visible=False,
                )
            }
        )
        source.host = SimpleNamespace(runtime=SimpleNamespace(players={}))
        source.engine.stream_id = "view"
        source.config = SimpleNamespace(spec=VideoSpec(160, 96, 24), play_seconds=1)
        source.journal = journal
        source.initial_message = {"references": []}
        source.session_id, source.source_epoch = "source", "epoch"
        source.startup = PlayStartup(
            source,
            dict(
                protocol=V2_PROTOCOL,
                request_id="startup",
                intent="model",
                initial_render=dict(prompt="legacy", seed=7, references=[]),
            ),
        )
        source.display_mode = "whitebox"
        await source.source_model_ready({"seed": 19})
        source.bound_at = source.last_tick = time.monotonic()
        source.binding_ended = False
        source.finished = asyncio.Event()
        source.source_state_value = "ready"
        source.play_elapsed = source.wait_elapsed = 0.0
        source.current_model_stream = "model"
        source.current_segment = SimpleNamespace(
            segment_epoch="model",
            first_decodable=False,
            closed_event=asyncio.Event(),
        )
        unbound = []

        async def unbind(code):
            unbound.append(code)

        source.unbind = unbind
        timer = asyncio.create_task(source._deadline())
        try:
            await asyncio.sleep(1.15)
            assert source.play_elapsed == 0 and source.wait_elapsed >= 1
            assert not unbound
            source.current_segment.first_decodable = True
            await asyncio.sleep(0.15)
            assert source.play_elapsed > 0
        finally:
            timer.cancel()
            await asyncio.gather(timer, return_exceptions=True)

    asyncio.run(scenario())


def test_close_during_queued_receipt_cannot_start_preparation(tmp_path):
    async def scenario():
        adapter, c, _, receipts, _, _ = await setup(tmp_path)
        entered, release = asyncio.Event(), asyncio.Event()
        reads = []

        async def emit(receipt):
            receipts.append(receipt)
            entered.set()
            await release.wait()

        async def read(handle):
            reads.append(handle)
            raise AssertionError("closed adapter started image preparation")

        adapter.emit, adapter.read_resource = emit, read
        handler = asyncio.create_task(adapter.handle(c))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await adapter.close()
            release.set()
            await handler
            await asyncio.sleep(0)
            assert not reads
            assert adapter.preparation is None
        finally:
            release.set()
            await asyncio.gather(handler, return_exceptions=True)
            await adapter.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("terminal", ["reset", "source-ended", "finish"])
def test_native_context_wait_observes_terminal_control(tmp_path, terminal):
    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import NativeCLIView
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_conditions import (
        PAYLOAD,
    )

    async def scenario():
        adapter, c, _, _, _, _ = await setup(tmp_path)
        control, media = asyncio.Queue(), asyncio.Queue()
        sent = []

        class Socket:
            async def send(self, wire):
                sent.append(json.loads(wire))

            async def recv(self):
                return await media.get()

            def __aiter__(self):
                return self

            async def __anext__(self):
                return await control.get()

        descriptor = dict(
            sessionId="session",
            epoch=1,
            worldBuildHash="a" * 64,
            uiBundleHash="b" * 64,
            media=dict(kind="pixels"),
            modelContext=dict(
                protocol=c["protocol"],
                required=True,
                configHash=c["configHash"],
                gameBuildHash=c["source"]["gameBuildHash"],
                compilerVersion=1,
                visible=False,
            ),
        )
        view = NativeCLIView(
            SimpleNamespace(
                descriptor=descriptor,
                spec=VideoSpec(2, 2, 24),
                prepared_delivery=None,
            )
        )
        view.control = view.media = Socket()
        view.config = dict(mediaGeneration=1)
        view.configured.set()
        channel = view.model_context
        channel.consumer, channel.segment = c["audience"], c["segmentId"]
        channel.sample_floor = 1
        channel.create_adapter = lambda *_: adapter
        channel.bound.set()
        snapshot = copy.deepcopy(fixture()["messages"][0])
        snapshot.update(
            segmentId=c["segmentId"],
            desired={k: c[k] for k in PAYLOAD},
            candidate=None,
        )
        await channel.consume(snapshot)
        h = header(c)
        envelope = dict(
            kind="worldkit-pixel-frame",
            schemaVersion=2,
            pixelFormat="rgb24",
            colorSpace="srgb",
            widthPixels=2,
            heightPixels=2,
            rowStrideBytes=6,
            payloadBytes=12,
            worldBuildHash="a" * 64,
            uiBundleHash="b" * 64,
            input=h,
        )
        encoded = json.dumps(envelope).encode()
        await media.put(len(encoded).to_bytes(4, "big") + encoded + bytes(12))
        entered = asyncio.Event()
        original_frame = channel.frame

        async def waiting_frame(value):
            entered.set()
            return await original_frame(value)

        channel.frame = waiting_frame
        channel.task = asyncio.create_task(channel.run())
        view.control_task = asyncio.create_task(view._control())
        iterator = view.frames()
        pending = asyncio.create_task(anext(iterator))
        try:
            async with asyncio.timeout(1):
                await entered.wait()
                if terminal == "finish":
                    await view.finish()
                    with pytest.raises(StopAsyncIteration):
                        await pending
                    assert channel.closed
                else:
                    # A valid old sample and a terminal control event can arrive
                    # together on control while pixels await their context mapping.
                    await control.put(
                        json.dumps(
                            dict(
                                type="model.context.sample",
                                epoch=1,
                                consumerId=c["audience"]["consumerId"],
                                consumerEpoch=c["audience"]["consumerEpoch"],
                                segmentId=c["segmentId"],
                                header=h,
                                metadata={
                                    k: snapshot[k]
                                    for k in (
                                        "contextStateId",
                                        "contextRevision",
                                        "viewRevision",
                                        "compiledConditionId",
                                    )
                                }
                                | dict(throughControlSequence=1),
                            )
                        )
                    )
                    await control.put(
                        json.dumps(
                            dict(type="session.ready", epoch=2, role="controller")
                            if terminal == "reset"
                            else dict(type="session.ended", epoch=1, reason="stopped")
                        )
                    )
                    with pytest.raises(
                        ValueError if terminal == "reset" else StopAsyncIteration,
                        match="The CLI session was reset"
                        if terminal == "reset"
                        else None,
                    ):
                        await pending
                assert sum(message["type"] == "media.credit" for message in sent) == 1
        finally:
            pending.cancel()
            view.control_task.cancel()
            await asyncio.gather(pending, view.control_task, return_exceptions=True)
            await iterator.aclose()
            await channel.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("prior_applied", [False, True])
@pytest.mark.parametrize(
    "size, expected",
    [
        ((2, 2), "rejected"),
        ((128, 16), "rejected"),
        ((32, 256), "rejected"),
        ((33, 63), "prepared"),
        ((128, 32), "prepared"),
    ],
)
def test_source_owner_checks_h3_reference_geometry_before_commit(
    tmp_path, prior_applied, size, expected
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def scenario():
        base, c, journal, receipts, submits, _ = await setup(tmp_path)
        data = b""

        def image_bytes(dimensions):
            output = io.BytesIO()
            Image.new("RGB", dimensions, "red").save(output, format="PNG")
            return output.getvalue()

        def reference_condition(identifier):
            value = copy.deepcopy(c)
            value.update(conditionId=identifier, onsetOccurrenceIds=[])
            for ref in value["references"]:
                ref["sha256"] = ref["resource"]["sha256"] = hashlib.sha256(
                    data
                ).hexdigest()
            value["payloadHash"] = payload_hash(value)
            return value

        async def resource(_handle):
            return data

        owner = object.__new__(SourceSession)
        owner.closed = owner.binding_ended = False
        owner.current_segment = None
        owner.assets, owner.journal = base.assets, journal
        owner.submit_source_condition = base.submit
        owner.context_channel = owner.source_conditions = None

        async def bind(*, segment_id, create_adapter):
            assert segment_id == c["segmentId"]
            create_adapter(c, base.emit, lambda: None)
            return SimpleNamespace()

        owner.engine = SimpleNamespace(
            model_context_capability={"protocol": c["protocol"]},
            bind_model_context=bind,
            read_model_context_resource=resource,
        )
        await owner.bind_source_conditions()
        adapter = owner.source_conditions
        try:
            if prior_applied:
                data = image_bytes((128, 128))
                initial = reference_condition("valid-initial")
                await handle_settled(adapter, initial)
                await adapter.preparation
                assert receipts[-1]["phase"] == "prepared"
                await handle_settled(
                    adapter,
                    request_identity(initial)
                    | dict(
                        type="model.condition.commit",
                        controlSequence=2,
                        effective=initial["effective"],
                    ),
                )
                await journal.update(lambda s: s["segment"].update(state="streaming"))
                await journal.prompt_applied(
                    c["segmentId"], 1, dict(first_output_frame=0, num_frames=1)
                )
                await adapter.original(initial)
                owner.current_segment = SimpleNamespace(
                    model_reference_limit=9,
                    model_reference_bytes=8 << 20,
                    clip_switch=True,
                    prompt_reference_updates=True,
                )
            before = await journal.read()
            submissions = len(submits)
            data = image_bytes(size)
            candidate = reference_condition("geometry-candidate")
            await handle_settled(adapter, candidate)
            await adapter.preparation
            assert receipts[-1]["phase"] == expected
            if expected == "rejected":
                assert receipts[-1]["reason"] == "CONDITION_INVALID"
            assert len(submits) == submissions
            after = await journal.read()
            assert after["operations"] == before["operations"]
            assert after["segment"] == before["segment"]
        finally:
            await adapter.close()
            await base.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("unconfirmed", ["initial", "update"])
def test_context_restart_rejects_unknown_without_changing_original_identity(
    tmp_path, unconfirmed
):
    async def scenario():
        adapter, c, journal, receipts, _, _ = await setup(tmp_path)
        first = await journal.model_condition(c, [])
        operation = first
        if unconfirmed == "update":
            await journal.prompt_applied(
                c["segmentId"], 1, dict(first_output_frame=0, num_frames=1)
            )
            await journal.update(lambda s: s["segment"].update(state="streaming"))
            later = copy.deepcopy(c)
            later.update(
                conditionId="last-before-end",
                prompt="last before end",
                onsetOccurrenceIds=[],
            )
            later["payloadHash"] = payload_hash(later)
            operation = await journal.model_condition(later, [])
            await journal.result(
                operation["request_id"],
                state="reconciling",
                code="SEGMENT_ENDED_UNCONFIRMED",
            )
        else:
            await journal.result(first["request_id"], state="ended")
        await journal.update(
            lambda s: s["segment"].update(state="ended", resource_state="released")
        )
        before = await journal.read()
        try:
            with pytest.raises(RenderError) as blocked:
                await journal.start_model_context_segment(
                    dict(
                        op="render.context.start",
                        request_id="restart-unknown",
                        source_session_id="source",
                        source_epoch="epoch",
                        expected_render_revision=before["render_revision"],
                    )
                )
            assert blocked.value.code == "RENDER_TRANSITION_IN_PROGRESS"
            assert await journal.read() == before
            query = {
                **operation["model_context"],
                "requestId": operation["request_id"],
                "promptVersion": operation["prompt_version"],
            }
            await adapter.reconcile([query])
            assert receipts[-1]["phase"] == "unknown"
            assert receipts[-1]["requestId"] == operation["request_id"]
            # Only original output evidence, including a late receipt, can open
            # the successor; an outer initial-operation EOS is not sufficient.
            await journal.prompt_applied(
                c["segmentId"],
                operation["prompt_version"],
                dict(first_output_frame=8, num_frames=1),
            )
            accepted = await journal.start_model_context_segment(
                dict(
                    op="render.context.start",
                    request_id="restart-unknown",
                    source_session_id="source",
                    source_epoch="epoch",
                    expected_render_revision=before["render_revision"],
                )
            )
            assert accepted["segment_id"] != c["segmentId"]
            assert (
                journal.snapshot["operations"][operation["request_id"]]["state"]
                == "applied"
            )
        finally:
            await adapter.close()

    asyncio.run(scenario())


async def source_waiting_on_context(tmp_path, outcome):
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
        CLIConditionChannel,
    )

    _, c, journal, _, _, _ = await setup(tmp_path)
    source = object.__new__(SourceSession)
    source.closed = source.closing = False
    source.outcome, source.trace_id = outcome, "normal-source-close"
    source.tasks, source.control_tasks = set(), set()
    source.changed = asyncio.Event()
    source.retained, source.pairs = {}, {}
    source.assets = source.startup = source.renderer = source.record_source = None
    source.playback = source.source_playback = source.recording = None
    source.journal = journal
    source.config = SimpleNamespace(drain_seconds=1)

    async def close_owner():
        pass

    source.engine = source.upstream = SimpleNamespace(close=close_owner)
    channel = source.context_channel = CLIConditionChannel(
        SimpleNamespace(
            closed=False,
            stopped=False,
            control=None,
            descriptor={
                "modelContext": dict(
                    protocol=c["protocol"],
                    required=True,
                    configHash=c["configHash"],
                    gameBuildHash=c["source"]["gameBuildHash"],
                    compilerVersion=1,
                    visible=False,
                )
            },
        )
    )
    closed_event = asyncio.Event()

    async def stop_segment(*, cancel):
        assert cancel
        # Real segment drain awaits its original generation/encoders. Give
        # the waiting capture and its completion callback the same window.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        closed_event.set()

    source.current_segment = SimpleNamespace(
        closed_event=closed_event, stop=stop_segment
    )
    source._spawn(channel.frame(header(c)), "engine-source")
    await asyncio.sleep(0)
    return source


@pytest.mark.parametrize("outcome", ["aborted", "cancelled", "completed"])
def test_source_close_drains_capture_before_closing_its_context_channel(
    tmp_path, outcome
):
    async def scenario():
        source = await source_waiting_on_context(tmp_path, outcome)
        await source.close()
        assert source.outcome == outcome
        assert source.closed and source.context_channel.closed
        assert source.current_segment.closed_event.is_set()
        assert source.journal.snapshot["source_outcome"] == outcome
        assert all(task.cancelled() for task in source.tasks)

    asyncio.run(scenario())


def test_active_capture_stale_source_still_fails_before_teardown(tmp_path):
    async def scenario():
        source = await source_waiting_on_context(tmp_path, "completed")
        await source.context_channel.close()
        results = await asyncio.gather(*source.tasks, return_exceptions=True)
        assert any(
            isinstance(error, ValueError) and str(error) == "MODEL_CONTEXT_STALE_SOURCE"
            for error in results
        )
        assert source.outcome == "failed" and not source.closing
        await source.close()
        assert source.journal.snapshot["source_outcome"] == "failed"

    asyncio.run(scenario())


def test_source_close_still_propagates_engine_cleanup_failure(tmp_path):
    async def scenario():
        source = await source_waiting_on_context(tmp_path, "cancelled")
        cleaned = []

        async def engine_close():
            raise RuntimeError("engine cleanup failed")

        async def upstream_close():
            cleaned.append("upstream")

        source.engine = SimpleNamespace(close=engine_close)
        source.upstream = SimpleNamespace(close=upstream_close)
        with pytest.raises(RenderError) as failure:
            await source.close()
        assert failure.value.code == "RELEASE_UNKNOWN"
        assert isinstance(failure.value.__cause__, RuntimeError)
        assert str(failure.value.__cause__) == "engine cleanup failed"
        assert cleaned == ["upstream"]
        assert source.closed and source.context_channel.closed
        assert all(task.cancelled() for task in source.tasks)

    asyncio.run(scenario())


def context_capacity_host(monkeypatch):
    from dataclasses import replace

    from sglang.multimodal_gen.runtime.realtime.coordinator import (
        InMemoryCoordinatorStore,
        RealtimeCoordinator,
        WorkerHeartbeat,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_host import RenderHost

    monkeypatch.setenv("WHITEBOX_ROUTING_REVISION", "model")
    monkeypatch.delenv("WHITEBOX_LARGE_REFERENCE_MODEL_REVISION", raising=False)
    store = InMemoryCoordinatorStore(ttl_s=30, worker_ttl_s=30)
    heartbeat = WorkerHeartbeat(
        worker_id="worker",
        worker_epoch="e" * 32,
        role="denoiser",
        endpoint="ws://worker/generate",
        reservation_endpoint="http://worker",
        az="shanghai",
        capacity=1,
        model_revision="model",
        vae_fingerprint="local",
    )
    reservations, peers = [], []

    class Reservations:
        async def reserve(self, slot, **value):
            reservations.append(value)
            await store.heartbeat(replace(heartbeat, active_sessions=1))

        async def release_verified(self, slot, *, token):
            # Worker cleanup is confirmed, but its next heartbeat has not arrived.
            pass

    coordinator = RealtimeCoordinator(store, reservation_client=Reservations())

    class Client:
        async def admit(self, **value):
            value.pop("trace_id")
            return await coordinator.admit(**value)

        async def release_verified(self, assignment):
            await coordinator.release_verified(assignment)

        async def renew(self, assignment):
            return await coordinator.renew(assignment)

    class Peer:
        closed = False

        async def close(self):
            self.closed = True

    async def connect(*args, **kwargs):
        peer = Peer()
        peers.append(peer)
        return peer

    host = object.__new__(RenderHost)
    host.coordinator, host.connect = Client(), connect
    host.config = SimpleNamespace(max_message_bytes=4 << 20)
    return SimpleNamespace(
        store=store,
        heartbeat=heartbeat,
        host=host,
        reservations=reservations,
        peers=peers,
    )


@pytest.mark.parametrize("cancel_during_wait", [False, True])
def test_explicit_context_restart_waits_for_released_worker_heartbeat(
    tmp_path, monkeypatch, cancel_during_wait
):
    async def scenario():
        capacity = context_capacity_host(monkeypatch)
        await capacity.store.heartbeat(capacity.heartbeat)
        adapter, c, journal, _, _, _ = await setup(tmp_path)
        first_operation = await journal.model_condition(c, [])
        first_state = copy.deepcopy(journal.snapshot["segment"])
        first = capacity.host.make_segment_leases("user", first_state)
        async with first.open():
            await journal.prompt_applied(
                c["segmentId"], 1, dict(first_output_frame=0, num_frames=1)
            )
        assert first.resource_state == "released"
        assert not capacity.store._assignments_by_token
        # Verified release clears the assignment before active_sessions refreshes.
        snapshot = await capacity.store.capacity_snapshot()
        assert snapshot["roles"]["denoiser"]["free_slots"] == 0
        await journal.result(first_operation["request_id"], state="ended")
        await journal.update(
            lambda s: s["segment"].update(state="ended", resource_state="released")
        )
        reserved = await journal.start_model_context_segment(
            dict(
                op="render.context.start",
                request_id="restart-after-release",
                source_session_id="source",
                source_epoch="epoch",
                expected_render_revision=journal.snapshot["render_revision"],
            )
        )
        current = copy.deepcopy(c)
        current.update(
            segmentId=reserved["segment_id"],
            conditionId="after-release",
            onsetOccurrenceIds=[],
        )
        current["payloadHash"] = payload_hash(current)
        operation = await journal.model_condition(current, [])
        successor = capacity.host.make_segment_leases(
            "user", journal.snapshot["segment"]
        )
        entered, finish = asyncio.Event(), asyncio.Event()
        assignments = []

        async def open_successor():
            async with successor.open() as (_, assignment):
                assignments.append(assignment)
                entered.set()
                await finish.wait()

        task = asyncio.create_task(open_successor())
        try:
            await asyncio.sleep(0.01)
            assert not task.done(), (
                "restart failed before the released worker refreshed its load"
            )
            assert not entered.is_set() and capacity.store._waiting
            assert first_state["capacity_wait_ms"] == 0
            if cancel_during_wait:
                task.cancel()
            await capacity.store.heartbeat(capacity.heartbeat)
            if cancel_during_wait:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 1)
                # Admission cancellation owns and releases any acquired lease,
                # without opening a second model socket or sending an init.
                assert len(capacity.peers) == 1
            else:
                await asyncio.wait_for(entered.wait(), 1)
                assert assignments[0].session_id == reserved["model_session_id"]
                assert assignments[0].generation_id == reserved["generation_id"]
                assert assignments[0].session_id != first_state["model_session_id"]
                assert len(capacity.store._assignments_by_token) == 1
                finish.set()
                await asyncio.wait_for(task, 1)
                assert len(capacity.peers) == 2
            assert successor.resource_state == "released"
            assert len(capacity.reservations) == 2
            assert (
                not capacity.store._assignments_by_token and not capacity.store._waiting
            )
            assert all(peer.closed for peer in capacity.peers)
            assert (
                journal.snapshot["operations"][operation["request_id"]][
                    "prompt_version"
                ]
                == 1
            )
        finally:
            finish.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await adapter.close()

    asyncio.run(scenario())


def test_initial_context_capacity_admission_remains_fail_fast(tmp_path, monkeypatch):
    from dataclasses import replace

    from sglang.multimodal_gen.runtime.realtime.coordinator import CoordinatorRejected

    async def scenario():
        capacity = context_capacity_host(monkeypatch)
        await capacity.store.heartbeat(replace(capacity.heartbeat, active_sessions=1))
        adapter, c, journal, _, _, _ = await setup(tmp_path)
        try:
            await journal.model_condition(c, [])
            lease = capacity.host.make_segment_leases(
                "user", journal.snapshot["segment"]
            )
            async with asyncio.timeout(0.2):
                with pytest.raises(CoordinatorRejected, match="CAPACITY_EXHAUSTED"):
                    async with lease.open():
                        raise AssertionError("busy initial admission opened a model")
            assert lease.resource_state == "none"
            assert (
                not capacity.store._waiting and not capacity.store._assignments_by_token
            )
            assert not capacity.reservations and not capacity.peers
        finally:
            await adapter.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "resource_state", ["release_unknown", "releasing", "quarantined"]
)
def test_context_capacity_wait_cannot_bypass_unverified_release(
    tmp_path, resource_state
):
    async def scenario():
        adapter, c, journal, _, _, _ = await setup(tmp_path)
        try:
            await journal.model_condition(c, [])
            await journal.prompt_applied(
                c["segmentId"], 1, dict(first_output_frame=0, num_frames=1)
            )
            await journal.update(
                lambda s: s["segment"].update(
                    state="ended", resource_state=resource_state
                )
            )
            before = await journal.read()
            with pytest.raises(RenderError) as rejected:
                await journal.start_model_context_segment(
                    dict(
                        op="render.context.start",
                        request_id="unsafe-restart",
                        source_session_id="source",
                        source_epoch="epoch",
                        expected_render_revision=before["render_revision"],
                    )
                )
            assert rejected.value.code == "RENDER_TRANSITION_IN_PROGRESS"
            assert await journal.read() == before
        finally:
            await adapter.close()

    asyncio.run(scenario())


def test_ordinary_applied_prompt_debug_uses_exact_segment_and_asset():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    session = object.__new__(SourceSession)
    session.condition_debug = None
    session.emit_applied_prompt_debug()  # Off path does not even inspect the journal.
    asset = {"id": "<Picture 1>", "asset_id": "sha256:" + "a" * 64}
    session.assets = SimpleNamespace(
        assets={asset["asset_id"]: {"format": "png", "sha256": "a" * 64}}
    )
    session.journal = SimpleNamespace(
        snapshot={
            "segment": {
                "segment_id": "segment",
                "applied_version": 2,
                "prompt": "exact ordinary condition",
                "references": [asset],
            }
        }
    )
    events = []
    session.condition_debug = lambda condition, receipt: events.append(
        (condition, receipt)
    )
    session.emit_applied_prompt_debug("request")
    condition, receipt = events.pop()
    assert condition["conditionId"] == "segment:2"
    assert condition["prompt"] == "exact ordinary condition"
    assert condition["references"][0]["asset"] == asset
    assert receipt == {"phase": "applied", "promptVersion": 2, "requestId": "request"}
    session.world_driven = True
    session.emit_applied_prompt_debug()
    assert events == []


@pytest.mark.parametrize("native", [False, True])
def test_cli_finish_wakes_pending_media_without_a_terminal_packet(native):
    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import (
        CLIView,
        NativeCLIView,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec

    async def scenario():
        entered = asyncio.Event()
        sent = []

        class Socket:
            async def send(self, wire):
                sent.append(json.loads(wire))

            async def recv(self):
                entered.set()
                await asyncio.Event().wait()

        view = (NativeCLIView if native else CLIView)(
            SimpleNamespace(
                descriptor=dict(sessionId="finish", epoch=1, media=dict(kind="pixels")),
                spec=VideoSpec(2, 2, 24),
                prepared_delivery=None,
            )
        )
        view.control = view.media = Socket()
        view.config = dict(mediaGeneration=1)
        view.configured.set()
        view.control_task = asyncio.create_task(asyncio.Event().wait())
        iterator = view.frames()
        pending = asyncio.create_task(anext(iterator))
        try:
            async with asyncio.timeout(1):
                await entered.wait()
                await view.finish()
                with pytest.raises(StopAsyncIteration):
                    await pending
            assert any(message["type"] == "input.release" for message in sent)
            assert not view.control_task.done()
        finally:
            pending.cancel()
            view.control_task.cancel()
            await asyncio.gather(pending, view.control_task, return_exceptions=True)
            await iterator.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "failure", ["reset", "producer-error", "transport-error", "media-error"]
)
def test_cli_finish_preserves_concurrent_control_failure(native, failure):
    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import (
        CLIView,
        NativeCLIView,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec

    async def scenario():
        entered = asyncio.Event()
        controls, media_errors = asyncio.Queue(), asyncio.Queue()

        class Socket:
            def __aiter__(self):
                return self

            async def __anext__(self):
                value = await controls.get()
                if isinstance(value, Exception):
                    raise value
                return json.dumps(value)

            async def send(self, wire):
                pass

            async def recv(self):
                entered.set()
                raise await media_errors.get()

        view = (NativeCLIView if native else CLIView)(
            SimpleNamespace(
                descriptor=dict(
                    sessionId="finish-race", epoch=1, media=dict(kind="pixels")
                ),
                spec=VideoSpec(2, 2, 24),
                prepared_delivery=None,
            )
        )
        view.control = view.media = Socket()
        view.config = dict(mediaGeneration=1)
        view.configured.set()
        view.control_task = asyncio.create_task(view._control_transport())
        iterator = view.frames()
        pending = asyncio.create_task(anext(iterator))
        try:
            async with asyncio.timeout(1):
                await entered.wait()
                terminal = (
                    dict(type="session.ready", epoch=2, role="controller")
                    if failure == "reset"
                    else dict(type="producer.error", code="CLI_CAPTURE_FAILED")
                    if failure == "producer-error"
                    else OSError("Independent control transport failure")
                )
                if failure == "media-error":
                    await media_errors.put(
                        OSError("Independent media transport failure")
                    )
                else:
                    await controls.put(terminal)
                finishing = asyncio.create_task(view.finish())
                await finishing
                with pytest.raises(
                    OSError
                    if failure in {"transport-error", "media-error"}
                    else ValueError,
                    match="Independent media transport failure"
                    if failure == "media-error"
                    else "Independent control transport failure"
                    if failure == "transport-error"
                    else "The CLI session was reset"
                    if failure == "reset"
                    else "CLI source error: CLI_CAPTURE_FAILED",
                ):
                    await pending
        finally:
            pending.cancel()
            view.control_task.cancel()
            await asyncio.gather(pending, view.control_task, return_exceptions=True)
            await iterator.aclose()

    asyncio.run(scenario())


def test_opening_receipt_replays_exact_operation_and_rearms_new_round(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.opening_presentation import (
        CONFIRM,
        opening_operation,
    )

    async def scenario():
        a, c, journal, receipts, _, _ = await setup(tmp_path)
        a.first_presented_supported = lambda: True
        terminal = []
        a.first_presented_result = terminal.append
        initial = await journal.model_condition(c, [], first_presented=True)
        evidence = dict(first_output_frame=0, num_frames=1)
        await journal.prompt_applied(initial["segment_id"], 1, evidence)

        def show(state):
            state["display_mode"] = "model"
            state["segment"]["state"] = "streaming"

        await journal.update(show)

        def resolve(_, state):
            operation = opening_operation(state)
            if operation is None:
                raise RenderError("MODEL_PRESENTATION_STALE")
            return operation

        def command(request_id):
            return dict(
                op=CONFIRM,
                request_id=request_id,
                source_session_id="source",
                source_epoch="epoch",
            )

        async def apply_next(version, generation):
            c.update(
                conditionId=f"condition-{version}",
                lifecycle={"lifecycleId": "visit", "generation": generation},
                onsetOccurrenceIds=[],
            )
            c["payloadHash"] = payload_hash(c)
            result = await journal.model_condition(c, [], first_presented=True)
            await journal.prompt_applied(
                initial["segment_id"],
                version,
                dict(first_output_frame=version, num_frames=1),
            )
            return result

        assert (await journal.confirm_opening_presentation(command("one"), resolve))[
            "state"
        ] == "accepted"
        assert (await journal.confirm_opening_presentation(command("one"), resolve))[
            "duplicate"
        ]
        a.schedule_first_presented()
        task = a.first_presented_task
        a.schedule_first_presented()
        assert task is a.first_presented_task
        # Lose the first Host ACK, so a bounded replay carries the same request.
        async with asyncio.timeout(3):
            while sum(r["phase"] == "first-presented" for r in receipts) < 2:
                await asyncio.sleep(0.02)
        seen = [r for r in receipts if r["phase"] == "first-presented"]
        assert seen[0] == seen[1]
        assert seen[0]["requestId"] == initial["request_id"]
        assert seen[0]["presentedLifecycle"] == c["lifecycle"]
        assert not await a.acknowledge_first_presented(
            "wrong", c["conditionId"], "accepted"
        )

        replacement = await apply_next(2, c["lifecycle"]["generation"])
        assert opening_operation(journal.snapshot) is None
        assert await a.acknowledge_first_presented(
            initial["request_id"], initial["model_context"]["conditionId"], "stale"
        )
        assert terminal[-1]["request_id"] == "one"
        assert terminal[-1]["state"] == "rejected"
        assert (
            opening_operation(journal.snapshot)["request_id"]
            == replacement["request_id"]
        )
        await journal.confirm_opening_presentation(command("two"), resolve)
        assert not await a.acknowledge_first_presented(
            initial["request_id"], initial["model_context"]["conditionId"], "accepted"
        )
        assert await a.acknowledge_first_presented(
            replacement["request_id"],
            replacement["model_context"]["conditionId"],
            "duplicate",
        )
        assert terminal[-1]["request_id"] == "two"
        assert terminal[-1]["state"] == "completed"
        assert opening_operation(journal.snapshot) is None

        next_round = await apply_next(3, c["lifecycle"]["generation"] + 1)
        assert next_round["segment_id"] == initial["segment_id"]
        assert (
            opening_operation(journal.snapshot)["request_id"]
            == next_round["request_id"]
        )
        await journal.confirm_opening_presentation(command("three"), resolve)
        assert not await a.acknowledge_first_presented(
            replacement["request_id"],
            replacement["model_context"]["conditionId"],
            "accepted",
        )
        assert await a.acknowledge_first_presented(
            next_round["request_id"],
            next_round["model_context"]["conditionId"],
            "timeout",
        )
        assert terminal[-1]["request_id"] == "three"
        assert terminal[-1]["state"] == "failed"
        assert terminal[-1]["code"] == "MODEL_PRESENTATION_UNCONFIRMED"
        assert (
            opening_operation(journal.snapshot)["request_id"]
            == next_round["request_id"]
        )
        await journal.confirm_opening_presentation(command("retry"), resolve)
        assert await a.acknowledge_first_presented(
            next_round["request_id"],
            next_round["model_context"]["conditionId"],
            "duplicate",
        )
        assert terminal[-1]["request_id"] == "retry"
        assert terminal[-1]["state"] == "completed"
        await a.close()

    asyncio.run(scenario())


async def pending_opening_confirmation(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.opening_presentation import (
        CONFIRM,
        opening_operation,
    )

    adapter, c, journal, receipts, _, _ = await setup(tmp_path)
    adapter.first_presented_supported = lambda: True
    terminal = []
    adapter.first_presented_result = terminal.append
    await journal.initialize_inference(manual=False)
    initial = await journal.model_condition(c, [], first_presented=True)
    await journal.prompt_applied(
        initial["segment_id"], 1, dict(first_output_frame=0, num_frames=1)
    )
    await journal.update(
        lambda state: (
            state.update(display_mode="model"),
            state["segment"].update(state="streaming"),
        )
    )
    await journal.confirm_opening_presentation(
        dict(
            type="command",
            op=CONFIRM,
            request_id="opening-confirm",
            source_session_id="source",
            source_epoch="epoch",
        ),
        lambda _, state: opening_operation(state),
    )
    return adapter, c, journal, receipts, initial, terminal


async def invalidate_opening_confirmation(journal, c, invalidation):
    if invalidation == "disable":
        await journal.configure_inference(
            dict(
                type="command",
                op="model.inference.configure",
                request_id="disable",
                source_session_id="source",
                source_epoch="epoch",
                expected_inference_revision=0,
                enabled=False,
            )
        )
    else:
        c.update(
            conditionId="next-lifecycle",
            onsetOccurrenceIds=[],
            lifecycle=dict(
                lifecycleId="visit", generation=c["lifecycle"]["generation"] + 1
            ),
        )
        c["payloadHash"] = payload_hash(c)
        successor = await journal.model_condition(c, [], first_presented=True)
        await journal.prompt_applied(
            successor["segment_id"], 2, dict(first_output_frame=1, num_frames=1)
        )


@pytest.mark.parametrize("invalidation", ["disable", "new-lifecycle"])
@pytest.mark.parametrize("when", ["before-schedule", "awaiting-ack"])
def test_opening_confirmation_settles_when_delivery_becomes_stale(
    tmp_path, invalidation, when
):
    async def scenario():
        (
            adapter,
            c,
            journal,
            receipts,
            initial,
            terminal,
        ) = await pending_opening_confirmation(tmp_path)
        first_receipt = asyncio.Event()
        emit = adapter.emit

        async def observe(value):
            await emit(value)
            if value.get("phase") == "first-presented":
                first_receipt.set()

        adapter.emit = observe
        try:
            if when == "awaiting-ack":
                adapter.schedule_first_presented()
                await asyncio.wait_for(first_receipt.wait(), 1)
            await invalidate_opening_confirmation(journal, c, invalidation)
            assert not await adapter.acknowledge_first_presented(
                initial["request_id"],
                initial["model_context"]["conditionId"],
                "accepted",
            )
            assert (
                await journal.acknowledge_opening_presentation(
                    initial["request_id"],
                    initial["model_context"]["conditionId"],
                    "duplicate",
                )
                is None
            )
            if when == "before-schedule":
                adapter.schedule_first_presented()
            task = adapter.first_presented_task
            assert task is not None, (
                "A stale accepted confirmation still needs terminal settlement"
            )
            await asyncio.wait_for(task, 2)
            state = await journal.read()
            result = state["operations"]["opening-confirm"]
            assert result["state"] == "rejected"
            assert result["code"] == "MODEL_PRESENTATION_STALE"
            assert terminal[-1]["request_id"] == result["request_id"]
            assert terminal[-1]["state"] == "rejected"
            assert (
                state["operations"][initial["request_id"]]["model_context_execution"][
                    "state"
                ]
                == "applied"
            )
            assert sum(r["phase"] == "first-presented" for r in receipts) == int(
                when == "awaiting-ack"
            )
        finally:
            await adapter.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("invalidation", ["disable", "new-lifecycle"])
@pytest.mark.parametrize("status", ["stale", "timeout"])
def test_opening_journal_can_settle_original_confirmation_after_invalidation(
    tmp_path, invalidation, status
):
    async def scenario():
        adapter, c, journal, _, initial, _ = await pending_opening_confirmation(
            tmp_path
        )
        try:
            await invalidate_opening_confirmation(journal, c, invalidation)
            result = await journal.acknowledge_opening_presentation(
                initial["request_id"], initial["model_context"]["conditionId"], status
            )
            assert result is not None
            assert result["request_id"] == "opening-confirm"
            assert result["state"] == "rejected"
            assert result["code"] == "MODEL_PRESENTATION_STALE"
        finally:
            await adapter.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["restored", "timeout", "disable", "new-lifecycle"])
def test_opening_confirmation_survives_pending_resubscription(tmp_path, outcome):
    from sglang.multimodal_gen.runtime.realtime.whitebox.opening_presentation import (
        PROTOCOL as PRESENTED_PROTOCOL,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.opening_presentation import (
        opening_operation,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
        CLIConditionChannel,
    )

    async def scenario():
        (
            adapter,
            c,
            journal,
            receipts,
            initial,
            terminal,
        ) = await pending_opening_confirmation(tmp_path)
        wire = []

        class Socket:
            async def send(self, value):
                wire.append(json.loads(value))

        view = SimpleNamespace(
            descriptor=dict(
                sessionId=c["source"]["sessionId"],
                epoch=c["source"]["epoch"],
                modelContext=dict(
                    protocol=c["protocol"],
                    required=True,
                    configHash=c["configHash"],
                    gameBuildHash=c["source"]["gameBuildHash"],
                    compilerVersion=1,
                    visible=False,
                    firstPresentedProtocol=PRESENTED_PROTOCOL,
                ),
            ),
            client_id=c["audience"]["consumerId"],
            control=Socket(),
            closed=False,
            stopped=False,
        )
        channel = CLIConditionChannel(view)
        channel.segment, channel.consumer, channel.adapter = (
            c["segmentId"],
            c["audience"],
            adapter,
        )
        channel.first_presented_supported = True
        adapter.first_presented_supported = lambda: channel.first_presented_supported
        presented = asyncio.Queue()
        emit = adapter.emit

        async def observe(value):
            await emit(value)
            await channel.emit(value)
            if value.get("phase") == "first-presented":
                presented.put_nowait(value)

        adapter.emit = observe
        try:
            adapter.schedule_first_presented()
            first = await asyncio.wait_for(presented.get(), 1)
            task = adapter.first_presented_task
            await channel.reconnect()
            # Let the first replay deadline elapse while the same consumer's
            # subscription is unresolved, without changing its game lifecycle.
            await asyncio.sleep(1.1)
            assert (
                journal.snapshot["operations"]["opening-confirm"]["state"] == "accepted"
            )
            assert not terminal
            assert presented.empty()
            assert not await adapter.acknowledge_first_presented(
                initial["request_id"],
                initial["model_context"]["conditionId"],
                "accepted",
            )
            if outcome in {"disable", "new-lifecycle"}:
                await invalidate_opening_confirmation(journal, c, outcome)
                await asyncio.wait_for(task, 3)
                assert terminal[-1]["state"] == "rejected"
                assert terminal[-1]["code"] == "MODEL_PRESENTATION_STALE"
                return
            if outcome == "timeout":
                await asyncio.wait_for(task, 7)
                assert terminal[-1]["state"] == "failed"
                assert terminal[-1]["code"] == "MODEL_PRESENTATION_UNCONFIRMED"
            await channel.consume(
                dict(
                    type="model.context.subscribed",
                    epoch=view.descriptor["epoch"],
                    protocol=c["protocol"],
                    segmentId=c["segmentId"],
                    consumer=c["audience"],
                    sampleControlFloor=1,
                    firstPresentedProtocol=PRESENTED_PROTOCOL,
                )
            )
            if outcome == "timeout":
                assert (
                    opening_operation(journal.snapshot)["request_id"]
                    == initial["request_id"]
                )
                return
            replay = await asyncio.wait_for(presented.get(), 3)
            assert replay == first
            await channel.consume(
                dict(
                    type="model.context.receipt-result",
                    epoch=view.descriptor["epoch"],
                    phase="first-presented",
                    requestId=first["requestId"],
                    conditionId=first["conditionId"],
                    status="accepted",
                )
            )
            await asyncio.wait_for(task, 1)
            result = journal.snapshot["operations"]["opening-confirm"]
            assert result["state"] == "completed"
            assert terminal == [result]
            assert len(journal.snapshot["operations"]) == 2
            assert (
                sum(
                    v.get("message", {}).get("phase") == "first-presented" for v in wire
                )
                == 2
            )
        finally:
            await channel.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure",
    ["before-commit", "after-commit", "until-query", "superseded", "cas-conflict"],
)
def test_opening_timeout_settlement_recovers_without_a_new_condition(tmp_path, failure):
    from sglang.multimodal_gen.runtime.realtime.whitebox.opening_presentation import (
        CONFIRM,
        opening_operation,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    async def scenario():
        (
            adapter,
            _,
            journal,
            _,
            initial,
            terminal,
        ) = await pending_opening_confirmation(tmp_path)
        cas = journal.store.cas
        attempts, recovered = 0, False

        async def fail_settlement(key, old, new, retention, **kwargs):
            nonlocal attempts
            state = json.loads(new)
            result = state["operations"].get("opening-confirm", {})
            if result.get("code") == "MODEL_PRESENTATION_UNCONFIRMED" and (
                attempts == 0 or failure == "until-query" and not recovered
            ):
                attempts += 1
                if failure in {"after-commit", "superseded", "cas-conflict"}:
                    await cas(key, old, new, retention, **kwargs)
                    if failure in {"superseded", "cas-conflict"}:
                        await journal.confirm_opening_presentation(
                            dict(
                                type="command",
                                op=CONFIRM,
                                request_id="new-confirm",
                                source_session_id="source",
                                source_epoch="epoch",
                            ),
                            lambda _, value: opening_operation(value),
                        )
                        adapter.schedule_first_presented()
                    if failure == "cas-conflict":
                        return False
                raise OSError("Simulated transient journal write failure")
            return await cas(key, old, new, retention, **kwargs)

        journal.store.cas = fail_settlement
        try:
            adapter.schedule_first_presented()
            await asyncio.wait_for(adapter.first_presented_task, 12)
            if failure == "until-query":
                assert 1 < attempts <= 4
                state = await journal.read()
                assert state["operations"]["opening-confirm"]["state"] == "accepted"
                assert not terminal
                recovered = True
                source = SimpleNamespace(
                    journal=journal,
                    outgoing=asyncio.Queue(),
                    source_conditions=adapter,
                    source_identity=lambda: {},
                    source_state_value="ready",
                    display_mode="model",
                    player_remaining_seconds=60,
                    wait_elapsed=0,
                    binding_ended=False,
                    model_context_segment=SourceSession.model_context_segment,
                    replay_opening_presentation=lambda: (
                        adapter.schedule_first_presented()
                    ),
                )
                await SourceSession.status_snapshot(source, "query", "opening-confirm")
                task = adapter.first_presented_task
                assert task is not None, (
                    "A healthy status read must rearm unresolved confirmation work"
                )
                await asyncio.wait_for(task, 12)
            state = await journal.read()
            original = state["operations"]["opening-confirm"]
            assert original["state"] == "failed"
            assert original["code"] == "MODEL_PRESENTATION_UNCONFIRMED"
            assert terminal[-1]["request_id"] == "opening-confirm"
            assert terminal[-1]["state"] == "failed"
            if failure in {"superseded", "cas-conflict"}:
                assert state["operations"]["new-confirm"]["state"] == "accepted"
                assert adapter.first_presented_task is not None
                assert await adapter.acknowledge_first_presented(
                    initial["request_id"],
                    initial["model_context"]["conditionId"],
                    "accepted",
                )
                assert (
                    journal.snapshot["operations"]["new-confirm"]["state"]
                    == "completed"
                )
            else:
                assert opening_operation(state)["request_id"] == initial["request_id"]
            assert len(state["operations"]) == (
                3 if failure in {"superseded", "cas-conflict"} else 2
            )
            assert (
                state["operations"][initial["request_id"]]["model_context_execution"][
                    "state"
                ]
                == "applied"
            )
        finally:
            await adapter.close()

    asyncio.run(scenario())


def test_native_reset_retires_model_context_until_explicit_retry(tmp_path):
    from types import SimpleNamespace

    from test_whitebox_native import native_frame

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import NativeCLIView

    async def run():
        template, c, journal, _, _, _ = await setup(tmp_path)
        spec, descriptor, _ = native_frame()
        descriptor.update(
            sessionId=c["source"]["sessionId"],
            epoch=c["source"]["epoch"],
            media={"kind": "pixels"},
            modelContext=dict(
                protocol=c["protocol"],
                required=True,
                configHash=c["configHash"],
                gameBuildHash=c["source"]["gameBuildHash"],
                compilerVersion=1,
                visible=False,
            ),
        )
        view = NativeCLIView(
            SimpleNamespace(descriptor=descriptor, spec=spec, prepared_delivery=None)
        )
        view.client_id = c["audience"]["consumerId"]
        view.source_identity = (c["source"]["presentationId"], c["source"]["sdkEpoch"])
        channel = view.model_context
        channel.segment, channel.consumer = c["segmentId"], copy.deepcopy(c["audience"])
        sent, failed = [], []
        control = asyncio.Queue()
        close_started, close_allowed = asyncio.Event(), asyncio.Event()
        reconnect_started, reconnect_allowed = asyncio.Event(), asyncio.Event()
        block_reconnect = False
        successor = dict(c["audience"], consumerEpoch="2")

        class Socket:
            def __aiter__(self):
                return self.messages()

            async def messages(self):
                while True:
                    yield await control.get()

            async def send(self, raw):
                value = json.loads(raw)
                sent.append(value)
                if value["type"] == "model.context.subscribe":
                    control.put_nowait(
                        json.dumps(
                            dict(
                                type="model.context.subscribed",
                                epoch=descriptor["epoch"],
                                protocol=c["protocol"],
                                segmentId=value["segmentId"],
                                consumer=successor,
                                sampleControlFloor=1,
                                subscriptionPause=True,
                                paused=value.get("paused", False),
                            )
                        )
                    )

        async def consume_ui(_):
            pass

        async def recovery_failed(code, **kwargs):
            failed.append(code)

        def create(envelope, emit, check):
            return SourceConditions(
                envelope=envelope,
                assets=template.assets,
                journal=journal,
                read_resource=template.read_resource,
                emit=emit,
                submit=template.submit,
                capabilities=template.capabilities,
                validate_reference=template.validate_reference,
                check=check,
            )

        async def recover(generation):
            if generation != view.transport_generation:
                return
            if block_reconnect:
                reconnect_started.set()
                await reconnect_allowed.wait()
            view.transport_generation += 1
            # Socket replacement discards any unread old control packets.
            while not control.empty():
                control.get_nowait()
            await channel.reconnect()

        view.control = Socket()
        view.renderer = SimpleNamespace(consume_ui=consume_ui)
        view.recover_model_context_transport = recover
        channel.create_adapter = create
        channel.on_recovery_failed = recovery_failed
        await channel.admit_envelope(dict(c, controlSequence=1))
        retired = channel.adapter
        close = retired.close

        async def slow_close():
            await close()
            close_started.set()
            await close_allowed.wait()

        retired.close = slow_close
        channel.sequence = 7
        channel.snapshots = [(7, {"old": True})]
        channel.samples[b"old"] = {"old": True}
        channel.task = asyncio.create_task(channel.run())
        view.control_task = asyncio.create_task(view._control())
        try:
            reset = dict(
                type="source.reset",
                epoch=descriptor["epoch"],
                presentationId="after-reset",
                sdkEpoch=c["source"]["sdkEpoch"] + 1,
            )
            control.put_nowait(json.dumps(reset))
            control.put_nowait(
                json.dumps(
                    dict(
                        type="model.context.error",
                        epoch=descriptor["epoch"],
                        code="MODEL_CONTEXT_STALE_SOURCE",
                    )
                )
            )
            control.put_nowait(
                json.dumps(
                    dict(type="input.receipt", epoch=descriptor["epoch"], sequence=1)
                )
            )
            async with asyncio.timeout(1):
                while not view.receipts:
                    if view.control_task.done():
                        view.control_task.result()
                    await asyncio.sleep(0)
            # The shared reader reaches fresh input while model retirement waits.
            assert channel.suspended and not close_allowed.is_set()
            await asyncio.wait_for(close_started.wait(), 1)
            assert failed == ["MODEL_CONTEXT_STALE_SOURCE"]
            assert retired.closed
            assert channel.adapter is None and channel.consumer is None
            assert not any(value["type"] == "model.context.subscribe" for value in sent)
            # Explicit retry cannot probe until original adapter retirement ends.
            probing = asyncio.create_task(channel.probe_subscription_pause(timeout=1))
            await asyncio.sleep(0)
            assert not probing.done()
            assert not any(value["type"] == "model.context.subscribe" for value in sent)
            close_allowed.set()
            await probing
            assert channel.sequence == 0
            assert not channel.samples and not channel.snapshots
            assert channel.consumer == successor and channel.suspended
            assert channel.adapter is None
            channel.bound.clear()
            await channel.switch_segment("after-reset")
            await asyncio.wait_for(channel.bound.wait(), 1)
            assert view.transport_generation == 1
            assert any(
                value["type"] == "model.context.subscribe"
                and value.get("paused") is not True
                for value in sent
            )
            async with asyncio.timeout(1):
                while channel.requested_segment:
                    # Native SDK will publish the new segment's fresh envelope.
                    fresh = copy.deepcopy(c)
                    fresh.update(segmentId="after-reset", audience=successor)
                    fresh["source"].update(
                        presentationId="after-reset", sdkEpoch=reset["sdkEpoch"]
                    )
                    await channel.admit_envelope(dict(fresh, controlSequence=1))
            assert channel.adapter is not retired
            before = len(sent)
            with pytest.raises(ValueError, match="MODEL_CONTEXT_STALE_SOURCE"):
                await retired.receipt(c, "prepared")
            assert len(sent) == before and channel.consumer == successor
            with pytest.raises(ValueError, match="MODEL_CONTEXT_STALE_SOURCE"):
                await channel.consume(
                    dict(
                        type="model.context.subscribed",
                        epoch=descriptor["epoch"],
                        protocol=c["protocol"],
                        segmentId="after-reset",
                        consumer=c["audience"],
                        sampleControlFloor=1,
                        subscriptionPause=True,
                        paused=False,
                    )
                )
            assert channel.consumer == successor
            assert channel.adapter.envelope == envelope(fresh)
            # Duplicate reset cannot retire the newly accepted subscription.
            active = channel.adapter
            await view._source_reset(reset)
            assert channel.adapter is active and not active.closed
            await channel.pause()
            await channel.probe_subscription_pause(timeout=1)
            assert view.transport_generation == 1
            assert channel.adapter is active and channel.consumer == successor
            # A later authenticated reset supersedes an already-awaiting switch.
            block_reconnect = True
            switching = asyncio.create_task(channel.switch_segment("superseded"))
            await asyncio.wait_for(reconnect_started.wait(), 1)
            view.pending_reset = None  # Prior native media reconfiguration completed.
            await view._source_reset(
                dict(
                    reset, presentationId="reset-again", sdkEpoch=reset["sdkEpoch"] + 1
                )
            )
            marker = channel.reset_transport_generation
            reconnect_allowed.set()
            with pytest.raises(ValueError, match="MODEL_CONTEXT_STALE_SOURCE"):
                await switching
            assert channel.suspended and channel.requested_segment is None
            assert channel.reset_transport_generation == marker
            assert not any(
                value["type"] == "model.context.subscribe"
                and value["segmentId"] == "superseded"
                for value in sent
            )
        finally:
            close_allowed.set()
            view.control_task.cancel()
            await asyncio.gather(view.control_task, return_exceptions=True)
            await channel.close()
            await template.close()

    asyncio.run(run())


@pytest.mark.parametrize("sdk_epoch", [0, 1, True])
def test_invalid_native_reset_cannot_retire_model_consumer(tmp_path, sdk_epoch):
    from types import SimpleNamespace

    from test_whitebox_native import native_frame

    from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import NativeCLIView

    async def run():
        template, c, _, _, _, _ = await setup(tmp_path)
        spec, descriptor, _ = native_frame()
        view = NativeCLIView(
            SimpleNamespace(descriptor=descriptor, spec=spec, prepared_delivery=None)
        )
        view.source_identity = ("original", 1)
        view.model_context = SimpleNamespace(consumer=c["audience"])
        with pytest.raises(ValueError):
            await view._source_reset(
                dict(
                    type="source.reset",
                    epoch=descriptor["epoch"],
                    presentationId="changed",
                    sdkEpoch=sdk_epoch,
                )
            )
        assert view.model_context.consumer == c["audience"]
        await template.close()

    asyncio.run(run())


def test_reset_retirement_closes_current_adapter_after_prior_cleanup_failure(tmp_path):
    from sglang.multimodal_gen.runtime.realtime.whitebox.source_condition_channel import (
        CLIConditionChannel,
    )

    async def run():
        adapter, c, _, _, _, _ = await setup(tmp_path)
        view = SimpleNamespace(
            descriptor={
                "modelContext": dict(
                    protocol=c["protocol"],
                    required=True,
                    configHash=c["configHash"],
                    gameBuildHash=c["source"]["gameBuildHash"],
                    compilerVersion=1,
                    visible=False,
                )
            },
            transport_generation=0,
            closed=False,
            stopped=False,
            control=None,
        )
        channel = CLIConditionChannel(view)
        original = RuntimeError("Original model-context cleanup failed")
        previous = asyncio.get_running_loop().create_future()
        previous.set_exception(original)
        channel.reset_task, channel.adapter = previous, adapter
        channel.retire_source(
            dict(type="source.reset", epoch=0, presentationId="reset", sdkEpoch=1)
        )
        with pytest.raises(RuntimeError) as error:
            await channel.reset_task
        assert error.value is original and adapter.closed
        channel.reset_task = None
        await channel.close()

    asyncio.run(run())


@pytest.mark.parametrize("initial_limit", [None, "12"])
def test_source_initial_reference_limit_is_independent_of_routing(
    tmp_path, monkeypatch, initial_limit
):
    from unittest.mock import AsyncMock

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import (
        SourceSession,
    )

    monkeypatch.setenv("WHITEBOX_DEFAULT_REFERENCE_CAPACITY", "3")
    monkeypatch.delenv("WHITEBOX_INITIAL_REFERENCE_CAPACITY", raising=False)
    if initial_limit is not None:
        monkeypatch.setenv("WHITEBOX_INITIAL_REFERENCE_CAPACITY", initial_limit)

    async def run():
        source = object.__new__(SourceSession)
        source.journal = SimpleNamespace(
            snapshot={},
            reserve_model_context_segment=AsyncMock(return_value={"segment_id": "s"}),
        )
        source.assets = SourceAssets(tmp_path)
        source.current_segment = None
        source.closed = source.binding_ended = False
        source.submit_source_condition = AsyncMock()

        async def bind(**kwargs):
            c = condition()
            kwargs["create_adapter"](
                {
                    k: c[k]
                    for k in (
                        "protocol",
                        "source",
                        "audience",
                        "segmentId",
                        "configHash",
                        "compilerVersion",
                    )
                },
                AsyncMock(),
                lambda: None,
            )
            return SimpleNamespace()

        output = io.BytesIO()
        Image.new("RGB", (320, 180), "red").save(output, format="PNG")
        data = output.getvalue()
        source.engine = SimpleNamespace(
            model_context_capability={},
            bind_model_context=bind,
            read_model_context_resource=AsyncMock(return_value=data),
        )
        await source.bind_source_conditions()
        expected = 20 if initial_limit is None else 12
        assert source.source_conditions.capabilities()["max_references"] == expected
        adapter = source.source_conditions
        for count in (7, expected):
            references = [
                dict(
                    resource={},
                    sha256=hashlib.sha256(data).hexdigest(),
                    format="png",
                    picture=i + 1,
                )
                for i in range(count)
            ]
            prepared = await adapter.prepare_references(
                dict(prompt="Room", references=references), adapter.capabilities()
            )
            assert len(prepared) == count
        with pytest.raises(RenderError) as rejected:
            await adapter.prepare_references(
                dict(prompt="Room", references=[{}] * (expected + 1)),
                adapter.capabilities(),
            )
        assert rejected.value.code == "LIMIT_EXCEEDED"
        source.current_segment = SimpleNamespace(
            model_reference_limit=7, model_reference_bytes=1024
        )
        assert source.source_conditions.capabilities()["max_references"] == 7
        assert source.source_conditions.capabilities()["max_reference_bytes"] == 1024

    asyncio.run(run())


def test_instruction_identity_is_validated_without_changing_model_payload():
    value = condition()
    original_hash = payload_hash(value)
    value["instruction"] = {"requestId": "user-1", "revision": 1}
    assert parse_condition(value)["instruction"] == value["instruction"]
    assert payload_hash(value) == original_hash
    value["instruction"]["revision"] = True
    with pytest.raises(ValueError):
        parse_condition(value)


def test_instruction_business_receipt_is_independent_of_debug(tmp_path):
    async def run():
        adapter, value, journal, receipts, submits, gate = await setup(tmp_path)
        value["instruction"] = {"requestId": "user-1", "revision": 1}
        received = []

        async def observe(instruction, phase, reason):
            received.append((instruction, phase, reason))

        adapter.instruction_receipt = observe
        assert adapter.debug is None
        await adapter.receipt(value, "accepted")
        assert received == []
        await adapter.receipt(value, "applied")
        assert received == [({"requestId": "user-1", "revision": 1}, "applied", None)]
        await adapter.close()

    asyncio.run(run())


def test_instruction_receipt_recovers_from_operation_without_candidate(tmp_path):
    async def run():
        adapter, value, journal, receipts, submits, gate = await setup(tmp_path)
        value["instruction"] = {"requestId": "user-durable", "revision": 3}
        await handle_settled(adapter, value)
        gate.set()
        await adapter.preparation
        commit = {k: copy.deepcopy(value[k]) for k in adapter.envelope}
        commit.update(
            type="model.condition.commit",
            conditionId=value["conditionId"],
            payloadHash=value["payloadHash"],
            controlSequence=value["controlSequence"] + 1,
            effective=value["effective"],
        )
        await handle_settled(adapter, commit)
        operation = submits[0]
        assert operation["model_context_instruction"] == value["instruction"]
        await journal.prompt_applied(
            operation["segment_id"],
            operation["prompt_version"],
            dict(first_output_frame=0, num_frames=1),
        )
        adapter.candidate = None
        observed = []

        async def receive(instruction, phase, reason):
            observed.append((instruction, phase))

        adapter.instruction_receipt = receive
        await adapter.original(operation["model_context"])
        assert observed == [(value["instruction"], "applied")]
        await adapter.close()

    asyncio.run(run())
