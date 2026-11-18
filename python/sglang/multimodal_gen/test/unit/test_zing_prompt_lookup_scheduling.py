# SPDX-License-Identifier: Apache-2.0
"""Independent prompt admission and buffered-source switch regression."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch

from sglang.multimodal_gen.runtime.realtime.prompt_control import PromptControl


def test_independent_ready_notice_passes_existing_link_validation():
    from collections import deque
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_link import (
        DiTInputService,
        DiTLinkClient,
    )
    from sglang.multimodal_gen.runtime.realtime.polling_task_queue import (
        PollingTaskQueue,
    )

    replies = deque()
    channel = SimpleNamespace(
        send=replies.append, poll=lambda: bool(replies), receive=replies.popleft
    )
    service = DiTInputService(channel, None)
    service.owner = ("s", "g")
    service.current_transfer = 2

    async def drain(queue):
        queue.popleft()

    service.worker = SimpleNamespace(
        closed=False,
        queue=PollingTaskQueue(drain),
        condition_ready=dict(
            event_id=11, prompt_version=2, source_frame_index=17, state="kv_ready"
        ),
        conditioning_notice=None,
        take_cache_report=lambda event: {"event_id": event},
    )
    service.prompt_control = SimpleNamespace(take_candidate=lambda: None)
    asyncio.run(service.poll_prompt_control())
    client = DiTLinkClient(channel)
    client.owner, client.async_inputs, client.sequence = ("s", "g"), True, 3
    results = client.collect()
    assert [r["type"] for r in results] == ["conditioning.ready", "model.cache"]
    assert results[0]["event_id"] == 11
    assert not client.failed


@pytest.mark.parametrize("replace_refs", [False, True])
def test_independent_cache_key_matches_encoder_key(replace_refs):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_preparation import (
        DiTPreparation,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        preparation_identity,
        preparation_key,
        updated_preparation_request,
    )

    codecs = SimpleNamespace()
    config = dict(width=32, height=32)
    request = dict(
        prompt="old",
        world_id="world",
        media=[dict(id="ref", type="image", format="png", payload=b"old")],
        conditions=[
            dict(content_id="ref", role="reference", latent_modalities=["visual"])
        ],
    )
    identity = dict(model="fixed", schedule=[1000, 500])
    context = dict(
        identity=preparation_identity(codecs, request, None, config, identity),
        reference_ids=["ref"],
    )
    command = dict(text="new", version=2)
    if replace_refs:
        command["references"] = [dict(id="ref2", format="png", payload=b"new")]
    encoder_command = dict(command)
    if replace_refs:
        encoder_command["references"] = [
            dict(type="image", **ref) for ref in command["references"]
        ]
    updated = updated_preparation_request(
        codecs,
        request,
        SimpleNamespace(conditions=request["conditions"]),
        "new",
        encoder_command,
    )
    query = DiTPreparation(SimpleNamespace(block0_cache=None), None).query_prompt(
        context, command
    )
    assert query["cache_key"] == preparation_key(
        codecs, updated, None, config, identity
    )
    assert query["hit"] is False
    if replace_refs:
        import hashlib
        from unittest.mock import Mock
        from sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server import (
            EncoderConditioning,
        )

        # DiT adopted the new references locally. The Encoder still owns the old inputs.
        controller = DiTPreparation(SimpleNamespace(block0_cache=None), None)
        digest = dict(
            id="ref2", format="png", sha256=hashlib.sha256(b"new").hexdigest()
        )
        adopted = controller.query_prompt(
            context, dict(text="new", reference_digests=[digest])
        )
        assert adopted["cache_key"] == query["cache_key"]
        decision = controller.query_prompt(
            adopted["cache_query_context"], {"text": "third"}
        )
        encoder = SimpleNamespace(
            cache_request=request,
            cache_preview=None,
            codecs=codecs,
            cache_query=Mock(return_value={"identity": identity}),
            inputs=SimpleNamespace(conditions=request["conditions"]),
            cache_candidates={},
        )
        batch = SimpleNamespace(
            extra={"zing_source_command": {"config": config}},
            sampling_params=SimpleNamespace(prompt="third"),
        )
        miss = {"cache_key": decision["cache_key"], "lookup": decision["lookup"]}
        assert (
            EncoderConditioning._lookup(
                encoder,
                batch,
                update=dict(
                    encoder_command,
                    prompt="third",
                    event_id=3,
                    block0_miss=miss,
                    prompt_preparation="ticket",
                ),
                reuse_active=True,
            )
            is None
        )
        assert encoder.cache_key == decision["cache_key"]
        encoder.cache_query.assert_called_once_with("identity")


def test_http_query_is_session_scoped_and_preserves_tensor_payload():
    import io
    from types import SimpleNamespace

    import msgspec
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from sglang.multimodal_gen.runtime.realtime.prompt_control_http import (
        PATH,
        install_prompt_routes,
    )

    control = PromptControl(lambda *_: {"hit": True, "context": torch.arange(6)})
    token = control.open(("s", "g"))
    control.publish_context({})
    app = FastAPI()
    app.state.worker_connections = SimpleNamespace(
        prompt_control=control, state="connected", restart_required=False
    )
    install_prompt_routes(app)
    headers = {
        "Authorization": "Bearer " + token,
        "X-Session-Id": "s",
        "X-Generation-Id": "g",
    }

    def post(client, operation, **fields):
        return client.post(
            PATH + "/" + operation,
            headers=headers,
            content=msgspec.msgpack.encode(
                dict(session_id="s", generation_id="g", **fields)
            ),
        )

    try:
        with TestClient(app) as client:
            assert client.post(PATH + "/begin", content=b"bad").status_code == 403
            begin = post(client, "begin", command={"version": 2})
            assert begin.status_code == 200
            receipt = msgspec.msgpack.decode(begin.content)
            response = post(client, "result", ticket=receipt["ticket"])
            assert response.status_code == 200
            data = torch.load(io.BytesIO(response.content), weights_only=True)
            assert torch.equal(data["context"], torch.arange(6))
            assert post(client, "cancel", ticket=receipt["ticket"]).status_code == 200
            assert post(client, "cancel", ticket=receipt["ticket"]).status_code == 200
            control.open(("s", "next"))
            assert post(client, "begin", command={"version": 3}).status_code == 403
    finally:
        control.close()


def test_lookup_completes_while_model_executors_are_blocked():
    release, encoder_busy, dit_busy = Event(), Event(), Event()

    def block(started):
        started.set()
        assert release.wait(10)

    control = PromptControl(
        lambda context, command: {"hit": True, "key": context["key"]}
    )
    token = control.open(("s", "g"))
    control.publish_context({"key": "cache"})
    try:
        with ThreadPoolExecutor(1) as encoder, ThreadPoolExecutor(1) as dit:
            a = encoder.submit(block, encoder_busy)
            b = dit.submit(block, dit_busy)
            try:
                assert encoder_busy.wait(3) and dit_busy.wait(3)
                assert control.start_block(9)
                receipt = control.begin(
                    ("s", "g"), token, {"text": "new", "version": 2}
                )
                assert receipt["boundary"] == 17
                assert control.result(("s", "g"), token, receipt["ticket"], timeout=3)[
                    "hit"
                ]
                assert not a.done() and not b.done()
                assert control.start_block(17)
                assert not control.start_block(26)
            finally:
                release.set()
    finally:
        control.close()


@pytest.mark.parametrize("end,expected", [(9, 17), (17, 17), (26, 34), (34, 34)])
def test_boundary_follows_submitted_dit_block(end, expected):
    control = PromptControl(lambda *_: {"hit": False})
    try:
        token = control.open(("s", "g"))
        control.publish_context({})
        control.start_block(end)
        receipt = control.begin(("s", "g"), token, {"version": 2})
        assert receipt["boundary"] == expected
        control.cancel(("s", "g"), token, receipt["ticket"])
        assert control.start_block(expected + 9)
    finally:
        control.close()


def test_closed_generation_cannot_publish_query_or_candidate():
    entered, release = Event(), Event()

    def query(*_):
        entered.set()
        assert release.wait(5)
        return {"hit": True}

    control = PromptControl(query)
    try:
        token = control.open(("s", "g"))
        control.publish_context({})
        receipt = control.begin(("s", "g"), token, {"version": 2})
        assert entered.wait(3)
        control.open(("s", "next"))
        release.set()
        with pytest.raises(ValueError, match="session"):
            control.result(("s", "g"), token, receipt["ticket"])
        assert control.pending is None and control.context is None
    finally:
        release.set()
        control.close()


@pytest.mark.parametrize("path", ["hit", "rejected_hit", "disabled"])
def test_model_owner_routes_switch_at_reserved_boundary(path):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_link import (
        DiTInputService,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_preparation import (
        DiTPreparation,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.queued_dit import (
        QueuedDiTSession,
    )

    owner = ("s", "g")
    hit = path != "disabled"
    new_context = {
        "identity": {"request": {"prompt": "new"}},
        "reference_ids": ["new-ref"],
    }
    query = (
        (
            lambda *_: dict(
                hit=True, cache_key="k", lookup={}, cache_query_context=new_context
            )
        )
        if hit
        else DiTPreparation(SimpleNamespace(block0_cache=None), None).query_prompt
    )
    control = PromptControl(query)
    token = control.open(owner)
    state = SimpleNamespace(
        committed_pixel=17,
        next_chunk_size=None,
        discard_pending_prompt=Mock(),
        update_prompt=Mock(),
        activate_prompt_at_source_boundary=Mock(
            return_value={
                "kv_deferred": True,
                "source_frame_index": 17,
                "prompt_version": 2,
                "event_id": 11,
            }
        ),
    )
    worker = QueuedDiTSession(
        session_id="s",
        generation_id="g",
        create_session=lambda *_: state,
        bridge=SimpleNamespace(ended=False, clip_frames=17),
        sample_moments=None,
        on_output=None,
        confirm_condition_ready=lambda _: None,
    )
    worker.take_cache_report = lambda event: {"event_id": event}
    worker.prompt_control = control
    service = DiTInputService(Mock(), SimpleNamespace())
    service.current_transfer = 1
    service.owner, service.worker, service.prompt_control = owner, worker, control
    tensors = dict(
        context=torch.ones((1, 2)), row_role=torch.zeros(1, dtype=torch.int64)
    )

    def expand(message, **_):
        if path == "rejected_hit":
            raise ConditionError("CONDITION_INVALID", "The candidate exceeds capacity")
        return dict(message, tensors=tensors)

    service.create_session.cached_condition = expand
    service.create_session.receive_candidate = lambda value: value

    async def run():
        with pytest.raises(ValueError, match="not ready"):
            control.begin(owner, token, {"version": 2, "text": "new"})
        await worker.put(
            dict(
                type="conditioning",
                session_id="s",
                generation_id="g",
                tensors=tensors,
                **({"cache_query_context": {"old": True}} if hit else {}),
            )
        )
        await worker.queue.run_once()
        control.start_block(9)
        receipt = control.route(owner, token, dict(version=2, text="new"))
        assert not control.start_block(26)
        ticket = receipt["ticket"]
        if hit:
            # Run the HTTP caller independently while the model owner services its mailbox.
            adoption = asyncio.create_task(
                asyncio.to_thread(control.adopt, owner, token, ticket, 11, True)
            )
            while not adoption.done():
                await service.poll_prompt_control()
                await asyncio.sleep(0.001)
            result = await adoption
            assert result["event_id"] == 11
            if path == "hit":
                assert (
                    result["state"] == "source_ready" and control.context == new_context
                )
                state.update_prompt.assert_called_once()
            else:
                assert result["code"] == "CONDITION_INVALID" and control.context == {
                    "old": True
                }
                state.update_prompt.assert_not_called()
            service.channel.send.assert_not_called()
        else:
            assert (
                receipt["action"] == "prepare" and receipt["miss"]["cache_key"] is None
            )
            control.submit_candidate(
                owner,
                token,
                ticket,
                dict(
                    type="conditioning",
                    session_id="s",
                    generation_id="g",
                    event_id=11,
                    prompt_version=2,
                    source_boundary=17,
                    prompt_preparation=ticket,
                    tensors=tensors,
                ),
            )
            await service.poll_prompt_control()
            state.update_prompt.assert_called_once()
        assert control.pending is None and control.start_block(26) and not worker.closed

    try:
        asyncio.run(run())
    finally:
        control.close()
