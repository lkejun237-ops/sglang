# SPDX-License-Identifier: Apache-2.0

import queue
from concurrent.futures import ThreadPoolExecutor

import pytest

from sglang.multimodal_gen.runtime.realtime.decoder_output import DecoderOutputBroker


def setup_session():
    cancelled = []
    broker = DecoderOutputBroker(cancelled.append)
    session = broker.open(dict(session_id="s", generation_id="g", width=2, height=2))
    attachment = dict(
        type="output.attach", session_id="s", generation_id="g", token=session.token
    )
    return broker, session, attachment, cancelled


@pytest.mark.parametrize(
    "change",
    [
        {"session_id": "other"},
        {"generation_id": "old"},
        {"token": "wrong"},
        {"type": "init"},
    ],
)
def test_bad_attachment_preserves_owner(change):
    broker, session, attachment, _ = setup_session()
    with pytest.raises(ValueError):
        broker.attach(dict(attachment, **change))
    assert broker.attach(attachment) is session
    assert not session.stopped.is_set()


def test_single_active_output_and_new_generation_after_close():
    broker, session, attachment, _ = setup_session()
    broker.attach(attachment)
    with pytest.raises(ValueError):
        broker.attach(attachment)
    with pytest.raises(ValueError):
        broker.open(dict(session_id="second", generation_id="g2"))
    broker.close(session)
    replacement = broker.open(dict(session_id="s", generation_id="g2"))
    with pytest.raises(ValueError):
        broker.attach(attachment)
    broker.close(session)  # Late old cleanup must not release the new owner.
    assert broker.session is replacement


def test_ten_outputs_then_backpressure_and_eos_fifo():
    _, session, _, _ = setup_session()
    for index in range(10):
        session.put(dict(type="frames", reference=index), 0)
    with pytest.raises(TimeoutError):
        session.put(dict(type="session.completed"), 0)
    assert session.take()["reference"] == 0
    session.sent()
    session.put(dict(type="session.completed"), 0)
    for index in range(1, 10):
        assert session.take()["reference"] == index
        session.sent()
    assert session.take()["type"] == "session.completed"
    with pytest.raises(queue.Empty):
        session.take()


def test_disconnect_reclaims_inflight_and_queued_and_unblocks_producer():
    broker, session, _, cancelled = setup_session()
    session.put(dict(reference="inflight"), 0)
    session.take()
    for index in range(10):
        session.put(dict(reference=index), 0)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(session.put, dict(reference="not-owned"), 30)
        broker.close(session)
        with pytest.raises(RuntimeError, match="closed"):
            pending.result(timeout=1)
    assert cancelled == ["inflight", *range(10)]
    assert session.done.is_set() and not session.completed
    session.abort()
    assert len(cancelled) == 11


def test_finish_success_does_not_cancel_previously_consumed_reference():
    broker, session, _, cancelled = setup_session()
    session.put(dict(reference="sent"), 0)
    session.take()
    session.sent()
    session.completed = True
    session.done.set()
    broker.close(session)
    assert session.completed and cancelled == []


@pytest.mark.parametrize("open_replacement", [False, True])
def test_owner_abort_does_not_fail_connection_or_next_output(open_replacement):
    import msgspec.msgpack
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    broker, session, attachment, _ = setup_session()
    failed = []
    broker.on_send_failure = lambda: failed.append(True)
    app = FastAPI()
    broker.install(app)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime_video/output") as ws:
            ws.send_bytes(msgspec.msgpack.encode(attachment))
            assert msgspec.msgpack.decode(ws.receive_bytes())["type"] == "output.ready"
            broker.close(session)
            if open_replacement:
                replacement = broker.open(dict(session_id="next", generation_id="g2"))
            message = ws.receive()
            assert message["type"] == "websocket.close"
            assert message["code"] == 1000
        assert failed == []
        if not open_replacement:
            replacement = broker.open(dict(session_id="next", generation_id="g2"))
        assert broker.session is replacement
        with client.websocket_connect("/v1/realtime_video/output") as ws:
            ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        type="output.attach",
                        session_id="next",
                        generation_id="g2",
                        token=replacement.token,
                    )
                )
            )
            assert msgspec.msgpack.decode(ws.receive_bytes())["type"] == "output.ready"
            replacement.put(dict(type="session.completed"), 0)
            assert (
                msgspec.msgpack.decode(ws.receive_bytes())["type"]
                == "session.completed"
            )
            assert ws.receive()["type"] == "websocket.close"
        assert failed == []


def test_output_disconnect_cancels_without_reporting_connection_failure():
    import msgspec.msgpack
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    broker, _, attachment, _ = setup_session()
    failed = []
    broker.on_send_failure = lambda: failed.append(True)
    app = FastAPI()
    broker.install(app)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime_video/output") as ws:
            ws.send_bytes(msgspec.msgpack.encode(attachment))
            assert msgspec.msgpack.decode(ws.receive_bytes())["type"] == "output.ready"
            ws.close()
            assert ws.receive()["type"] == "websocket.close"
    assert failed == []


def test_cancelled_output_discards_late_rgb_and_credit_without_poisoning_owner():
    broker, session, _, cancelled = setup_session()
    session.abort()
    session.put(dict(reference="late-rgb"), 0, discard_if_stopped=True)
    session.source_progress(8, [[1, 0, 8]], 0)
    assert cancelled == ["late-rgb"] and session.queue.empty()
    with pytest.raises(RuntimeError, match="closed"):
        session.put(dict(reference="caller-owned"), 0)
    broker.close(session)
    replacement = broker.open(dict(session_id="next", generation_id="g2"))
    replacement.put(dict(type="session.completed"), 0)
    assert replacement.take()["type"] == "session.completed"
    assert cancelled == ["late-rgb"]


def test_cancel_unblocks_pending_late_rgb_publish_without_failure():
    broker, session, _, cancelled = setup_session()
    for index in range(10):
        session.put(dict(reference=index), 0)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            session.put,
            dict(reference="late-rgb"),
            30,
            discard_if_stopped=True,
        )
        broker.close(session)
        pending.result(timeout=1)
    assert set(cancelled) == {*range(10), "late-rgb"}
    assert len(cancelled) == 11


def test_output_disconnect_drains_late_rgb_before_next_session():
    import msgspec.msgpack
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    broker, session, attachment, cancelled = setup_session()
    app = FastAPI()
    broker.install(app)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime_video/output") as ws:
            ws.send_bytes(msgspec.msgpack.encode(attachment))
            assert msgspec.msgpack.decode(ws.receive_bytes())["type"] == "output.ready"
            ws.close()
            assert ws.receive()["type"] == "websocket.close"
    assert session.stopped.is_set()
    session.put(dict(reference="already-sent"), 0, discard_if_stopped=True)
    assert cancelled == ["already-sent"]
    broker.close(session)
    replacement = broker.open(dict(session_id="next", generation_id="g2"))
    replacement.put(dict(type="session.completed"), 0)
    assert replacement.take()["type"] == "session.completed"


def test_consumed_follows_current_rgb_and_uses_complete_packets_not_output_frames():
    _, session, _, _ = setup_session()
    records = [[1, 0, 8], [2, 8, 16]]
    session.put(dict(type="decoder.frames", reference="rgb", num_frames=1), 0)
    session.source_progress(12, records, 0)
    assert session.take()["type"] == "decoder.frames"
    session.sent()
    assert session.take() == dict(
        type="source.consumed", source_frame_end=8, source_bytes_end=96
    )
    session.source_progress(12, records, 0)
    with pytest.raises(queue.Empty):
        session.take()
    session.source_progress(16, records, 0)
    assert session.take()["source_frame_end"] == 16


def test_progress_without_rgb_releases_credit_without_waiting_for_video():
    _, session, _, _ = setup_session()
    records = [[1, 0, 8], [2, 8, 16]]
    session.source_progress(1, records, 0)
    assert session.queue.empty()
    session.source_progress(8, records, 0)
    assert session.take()["type"] == "source.consumed"


@pytest.mark.parametrize("source_size", [(832, 480), (1920, 1080)])
def test_credit_counts_wire_source_bytes_independently_of_output_size(source_size):
    broker = DecoderOutputBroker(lambda _: None)
    width, height = source_size
    session = broker.open(
        dict(
            session_id="s",
            generation_id="g",
            width=1280,
            height=736,
            source_stream=dict(width=width, height=height, fps=24, format="rgb24"),
            source_encoding=dict(width=832, height=480),
        )
    )
    session.source_progress(12, [[1, 0, 8], [2, 8, 16]], 0)
    assert session.take() == dict(
        type="source.consumed",
        source_frame_end=8,
        source_bytes_end=8 * width * height * 3,
    )


@pytest.mark.parametrize("source", [None, {}, {"width": True, "height": 2}])
def test_invalid_source_dimensions_cannot_grant_credit(source):
    _, session, _, _ = setup_session()
    session.config["source_stream"] = source
    with pytest.raises(ValueError, match="source stream dimensions"):
        session.source_progress(8, [[1, 0, 8]], 0)
    assert session.consumed_frames == 0 and session.queue.empty()


def test_output_failure_discards_queued_credit():
    _, session, _, cancelled = setup_session()
    session.put(dict(type="decoder.frames", reference="failed-output"), 0)
    session.source_progress(8, [[1, 0, 8]], 0)
    session.take()
    session.abort()
    assert cancelled == ["failed-output"]
    with pytest.raises(RuntimeError, match="closed"):
        session.take()


@pytest.mark.parametrize(
    "owned,records",
    [
        (17, [[1, 0, 16]]),
        (8, [[2, 4, 8]]),
        (8, [[1, 0, 4], [3, 4, 8]]),
        (8, [[1, 0, 4], [2, 5, 8]]),
    ],
)
def test_invalid_progress_cannot_grant_credit(owned, records):
    _, session, _, _ = setup_session()
    with pytest.raises(ValueError):
        session.source_progress(owned, records, 0)
    assert session.consumed_frames == 0 and session.queue.empty()


@pytest.mark.parametrize("attached", [False, True])
def test_initial_capacity_rejection_aborts_output_without_quarantining(
    monkeypatch, attached
):
    import asyncio
    from types import SimpleNamespace

    import msgspec.msgpack
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        dit_decoder,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_link import (
        DiTInputService,
        DiTLinkClient,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.remote_decoder import (
        RemoteLynnDecoder,
    )
    from sglang.multimodal_gen.runtime.realtime.worker_connections import (
        ServiceConnections,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        ZingKVGeometry,
        ZingKVPoolAdmissionError,
        ZingKVPoolConfig,
        ZingKVPoolPlan,
        ZingKVRequest,
    )

    manager = ServiceConnections(
        {},
        ready=lambda: True,
        build=lambda *_: None,
        destroy=lambda: None,
        rollback=lambda: None,
    )
    manager.state, manager.group = "connected", "existing-group"
    broker = DecoderOutputBroker(lambda ref: None)
    broker.on_send_failure = manager.failed
    app = FastAPI()
    broker.install(app)
    config = ZingKVPoolConfig(
        max_target_area=4, max_target_long_edge=2, window_chunks=2, sink_chunks=0
    )
    plan = ZingKVPoolPlan.build(
        config, ZingKVGeometry(1, (1, 1, 1), 2, ChunkPlan([3, 2])), token_capacity=64
    )

    class Worker:
        initial_received = complete = closed = False
        state = conditioning_notice = None
        queue = ()

        async def put(self, message):
            try:
                plan.admit(
                    ZingKVRequest(
                        2,
                        2,
                        None,
                        message["tensors"]["prefix_rows"],
                        config.window_chunks,
                        config.sink_chunks,
                        config.rope_position_mode,
                        config.rope_max_frame_gap,
                    )
                )
            except ZingKVPoolAdmissionError as exc:
                raise ConditionError("CONDITION_INVALID", exc.detail) from exc
            self.initial_received = True

        def close(self):
            self.closed = True

    class Wire:
        def __init__(self):
            self.replies = []

        def send(self, message, **kwargs):
            service.queue.append(message)
            asyncio.run(service.queue.run_once())

        def poll(self):
            return bool(self.replies)

        def receive(self):
            return self.replies.pop(0)

    wire = Wire()
    aborts = []

    def create_output(init):
        output = broker.open(dict(init, width=2, height=2))
        # Keep the real DiTDecoderOutput.close -> RemoteLynnDecoder.dispose
        # chain. Only the remote wire request is delivered in-process to the
        # same broker.close used by the Decoder worker's abort handler.
        remote = object.__new__(RemoteLynnDecoder)
        remote._pending_output = None
        remote.leader, remote.oneway = True, True
        remote.context, remote.failure = object(), None
        remote.closed, remote.provenance_for = False, None
        remote.output_attachment = dict(
            session_id=init["session_id"],
            generation_id=init["generation_id"],
            token=output.token,
        )

        def request(operation):
            assert operation == "abort"
            aborts.append(init["session_id"])
            broker.close(output)

        remote._request, remote._close_channel = request, lambda: None
        monkeypatch.setattr(dit_decoder, "create_decoder", lambda *a, **kw: remote)
        return dit_decoder.DiTDecoderOutput(None, init, {})

    service = DiTInputService(
        SimpleNamespace(send=wire.replies.append),
        lambda *_: Worker(),
        create_decoder=create_output,
    )
    client = DiTLinkClient(wire)
    init = dict(
        session_id="rejected", generation_id="g", seed=1, width=2, height=2, fps=24
    )
    with TestClient(app) as http:
        client.open(init)
        output = broker.session
        if attached:
            with http.websocket_connect("/v1/realtime_video/output") as ws:
                ws.send_bytes(
                    msgspec.msgpack.encode(
                        dict(
                            type="output.attach",
                            session_id="rejected",
                            generation_id="g",
                            token=output.token,
                        )
                    )
                )
                assert (
                    msgspec.msgpack.decode(ws.receive_bytes())["type"] == "output.ready"
                )
                reply = client.send(
                    "conditioning", event_id=1, tensors={"prefix_rows": 65}
                )
                assert (
                    reply[0]["type"] == "conditioning.rejected" and reply[0]["initial"]
                )
                closed = ws.receive()
                assert closed["type"] == "websocket.close" and closed["code"] == 1000
        else:
            reply = client.send("conditioning", event_id=1, tensors={"prefix_rows": 65})
            assert reply[0]["type"] == "conditioning.rejected" and reply[0]["initial"]
        assert aborts == ["rejected"] and broker.session is None
        assert (
            not manager.restart_required and manager.query()[0]["state"] == "connected"
        )
        client.release("rejected")
        assert service.owner is None and not client.failed

        client.open(dict(init, session_id="next", generation_id="g2"))
        output = broker.session
        with http.websocket_connect("/v1/realtime_video/output") as ws:
            ws.send_bytes(
                msgspec.msgpack.encode(
                    dict(
                        type="output.attach",
                        session_id="next",
                        generation_id="g2",
                        token=output.token,
                    )
                )
            )
            assert msgspec.msgpack.decode(ws.receive_bytes())["type"] == "output.ready"
            assert (
                client.send("conditioning", event_id=1, tensors={"prefix_rows": 0})
                == []
            )
            assert service.worker.initial_received
            client.release("next")
            closed = ws.receive()
            assert closed["type"] == "websocket.close" and closed["code"] == 1000
        assert aborts == ["rejected", "next"] and broker.session is None
        assert service.owner is None and not client.failed
        assert manager.state == "connected" and not manager.restart_required


@pytest.mark.parametrize("fault", ["disconnect", "protocol_error"])
def test_decoder_output_disconnect_cancels_but_protocol_error_quarantines(fault):
    import msgspec.msgpack
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from sglang.multimodal_gen.runtime.realtime.worker_connections import (
        ServiceConnections,
    )

    broker, session, attachment, _ = setup_session()
    manager = ServiceConnections(
        {},
        ready=lambda: True,
        build=lambda *_: None,
        destroy=lambda: None,
        rollback=lambda: None,
    )
    manager.state, manager.group = "connected", "existing-group"
    broker.on_send_failure = manager.failed
    app = FastAPI()
    broker.install(app)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime_video/output") as ws:
            ws.send_bytes(msgspec.msgpack.encode(attachment))
            assert msgspec.msgpack.decode(ws.receive_bytes())["type"] == "output.ready"
            if fault == "disconnect":
                ws.close()
            else:
                ws.send_bytes(msgspec.msgpack.encode(dict(type="unexpected")))
                error = msgspec.msgpack.decode(ws.receive_bytes())
                assert error["code"] == "DECODER_OUTPUT_FAILED" and error["fatal"]
            assert ws.receive()["type"] == "websocket.close"
    if fault == "disconnect":
        assert session.stopped.is_set()
        assert not manager.restart_required
        assert manager.query()[0]["state"] == "connected"
    else:
        assert manager.restart_required and manager.query()[1] == 503


@pytest.mark.parametrize("source_stream", [False, True])
def test_decoder_credit_counts_source480_independently_of_output736(source_stream):
    broker = DecoderOutputBroker(lambda reference: None)
    config = dict(session_id="s", generation_id="g", width=1280, height=736)
    if source_stream:
        config["source_stream"] = dict(width=832, height=480)
    session = broker.open(config)
    session.source_progress(8, [[1, 0, 8]], 0)
    credit = session.take()
    width, height = (832, 480) if source_stream else (1280, 736)
    assert credit["source_frame_end"] == 8
    assert credit["source_bytes_end"] == 8 * width * height * 3


def test_cancelled_mailbox_discards_late_rgb_and_credit_before_next_owner():
    broker, session, _, cancelled = setup_session()
    session.abort()
    session.put(
        dict(type="decoder.frames", reference="late-rgb"), 0, discard_if_stopped=True
    )
    session.source_progress(8, [[1, 0, 8]], 0)
    assert cancelled == ["late-rgb"] and session.queue.empty()
    broker.close(session)
    next_session = broker.open(
        dict(session_id="next", generation_id="new", width=2, height=2)
    )
    next_session.put(dict(type="session.completed", output_frames=0), 0)
    assert next_session.take()["type"] == "session.completed"
