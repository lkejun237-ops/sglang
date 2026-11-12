"""Native reset must not label fresh instruction receipts with initial IDs."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import RenderError
from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import SourceSession


def identity(label):
    return dict(
        segment_id=label,
        segment_epoch=label + "-epoch",
        model_session_id=label + "-model",
        generation_id=label + "-generation",
    )


async def source_with_factory():
    ids = identity("old")
    source = object.__new__(SourceSession)
    source.closed = source.binding_ended = False
    source.current_segment = source.context_channel = source.source_conditions = None
    source.assets = SimpleNamespace()
    source.journal = SimpleNamespace(
        snapshot={"model_context_segment_ids": copy.deepcopy(ids)},
        reserve_model_context_segment=AsyncMock(return_value=copy.deepcopy(ids)),
    )
    source.submit_source_condition = AsyncMock()
    source.emit_instruction_receipt = AsyncMock()
    factory = None

    async def bind(*, segment_id, create_adapter, **kwargs):
        nonlocal factory
        assert segment_id == ids["segment_id"]
        factory = create_adapter
        return SimpleNamespace()

    source.engine = SimpleNamespace(
        model_context_capability={},
        bind_model_context=bind,
        read_model_context_resource=AsyncMock(),
    )
    await source.bind_source_conditions()
    fixture = Path(__file__).with_name("fixtures") / "model-context-v1.json"
    payload = json.loads(fixture.read_text())["payload"]
    envelope = {
        k: payload[k]
        for k in (
            "protocol",
            "source",
            "audience",
            "segmentId",
            "configHash",
            "compilerVersion",
        )
    }
    return source, factory, envelope


def test_instruction_receipt_uses_each_binding_and_preserves_old_callback_identity(
    monkeypatch,
):
    monkeypatch.setenv("WHITEBOX_MODEL_CONTEXT_PREWARM", "0")

    async def run():
        source, create, envelope = await source_with_factory()
        envelope["segmentId"] = "old"
        old = create(envelope, AsyncMock(), lambda: None)
        fresh = None
        old_request = {"requestId": "old-request", "revision": 1}
        new_request = {"requestId": "new-request", "revision": 1}
        try:
            await old.instruction_receipt(old_request, "applied", None)
            assert source.emit_instruction_receipt.call_args.args[3] == identity("old")
            # The live journal may mutate; each callback must retain its own IDs.
            source.journal.snapshot["model_context_segment_ids"].update(identity("new"))
            next_envelope = copy.deepcopy(envelope)
            next_envelope["segmentId"] = "new"
            next_envelope["source"]["sdkEpoch"] += 1
            fresh = create(next_envelope, AsyncMock(), lambda: None)
            await fresh.instruction_receipt(new_request, "applied", None)
            assert source.emit_instruction_receipt.call_args.args[3] == identity("new")
            await old.instruction_receipt(old_request, "applied", None)
            assert source.emit_instruction_receipt.call_args.args[3] == identity("old")
        finally:
            await old.close()
            if fresh is not None:
                await fresh.close()

    asyncio.run(run())


def test_instruction_receipt_rejects_a_binding_without_matching_reserved_ids(
    monkeypatch,
):
    monkeypatch.setenv("WHITEBOX_MODEL_CONTEXT_PREWARM", "0")

    async def run():
        source, create, envelope = await source_with_factory()
        envelope["segmentId"] = "different-binding"
        adapter = create(envelope, AsyncMock(), lambda: None)
        try:
            with pytest.raises(RenderError) as failure:
                await adapter.instruction_receipt(
                    {"requestId": "wrong-scope", "revision": 1}, "applied", None
                )
            assert failure.value.code == "SOURCE_EPOCH_MISMATCH"
            source.emit_instruction_receipt.assert_not_awaited()
        finally:
            await adapter.close()

    asyncio.run(run())
