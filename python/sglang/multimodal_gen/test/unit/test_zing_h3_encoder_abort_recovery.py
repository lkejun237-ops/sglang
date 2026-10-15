# SPDX-License-Identifier: Apache-2.0
"""Actual private-control helper tests; no GPU/NCCL recovery claim."""

import importlib.util
import threading
import time
from pathlib import Path

import pytest
import zmq

_spec = importlib.util.spec_from_file_location(
    "zing_encoder_recovery_cpu",
    Path(__file__).resolve().parents[2]
    / "runtime/pipelines_core/stages/model_specific_stages/zing_h3/encoder_recovery.py",
)
recovery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recovery)


def test_fresh_socket_after_open_acknowledged_append_timeout():
    context = zmq.Context()
    server = context.socket(zmq.REP)
    server.setsockopt(zmq.LINGER, 0)
    port = server.bind_to_random_port("tcp://127.0.0.1")
    owner = ["old"]
    failures = []

    def worker():
        try:
            server.recv_json()
            server.send_json({"ok": True})
            server.recv_json()
            time.sleep(0.08)
            server.send_json({"ok": True})
            command = server.recv_json()
            reply, quarantine = recovery.abort_owner(
                command,
                owner[0],
                None,
                False,
                lambda: owner.__setitem__(0, None),
                lambda value: [value, value],
            )
            assert not quarantine
            server.send_json(reply)
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    client = context.socket(zmq.REQ)
    client.setsockopt(zmq.LINGER, 0)
    client.setsockopt(zmq.RCVTIMEO, 30)
    client.connect(f"tcp://127.0.0.1:{port}")
    try:
        client.send_json({"op": "open"})
        assert client.recv_json()["ok"]
        client.send_json({"op": "append"})
        with pytest.raises(zmq.Again):
            client.recv_json()
        client.close()
        reply = recovery.request_abort(port, "old", 1)
        assert reply["released"] and owner[0] is None
        thread.join(2)
        assert not thread.is_alive() and not failures
    finally:
        client.close()
        server.close()
        context.term()


@pytest.mark.parametrize("last_released", [None, "old"])
def test_late_old_token_does_not_clean_new_owner(last_released):
    def forbidden():
        pytest.fail("New owner was cleaned by stale abort")

    reply, quarantine = recovery.abort_owner(
        {"session": "old"}, "new", last_released, False, forbidden, lambda x: [x, x]
    )
    assert not quarantine and not reply["released"]
    assert reply["already_released"] == (last_released == "old")
    assert reply["current_owner_unchanged"]


def test_unknown_open_ack_can_abort_owned_token():
    calls = []
    reply, quarantine = recovery.abort_owner(
        {"session": "old"},
        "old",
        None,
        False,
        lambda: calls.append(True),
        lambda x: [x, x],
    )
    assert reply["released"] and not quarantine and calls == [True]


def test_bridge_failure_is_quarantined_without_unsafe_cleanup():
    def forbidden():
        pytest.fail("Do not dispose potentially active data bridge")

    reply, quarantine = recovery.abort_owner(
        {"session": "old", "require_restart": True},
        "old",
        None,
        False,
        forbidden,
        lambda x: [x, x],
    )
    assert (
        quarantine
        and reply["code"] == "ENCODER_RESTART_REQUIRED"
        and not reply["released"]
    )


def test_rank_disagreement_and_cleanup_failure_never_ack_release():
    def forbidden():
        pytest.fail("No ownership consensus")

    reply, quarantine = recovery.abort_owner(
        {"session": "old"},
        "old",
        None,
        False,
        forbidden,
        lambda x: [x, ("new", None, False)],
    )
    assert quarantine and not reply["released"]
    calls = []

    def consensus(x):
        return [x, x] if isinstance(x, tuple) else [True, False]

    reply, quarantine = recovery.abort_owner(
        {"session": "old"}, "old", None, False, lambda: calls.append(True), consensus
    )
    assert quarantine and not reply["released"] and calls


def test_blocked_worker_is_unconfirmed_not_released():
    context = zmq.Context()
    server = context.socket(zmq.REP)
    server.setsockopt(zmq.LINGER, 0)
    port = server.bind_to_random_port("tcp://127.0.0.1")
    try:
        reply = recovery.request_abort(port, "old", 0.02)
        assert not reply["released"] and reply["code"] == "ENCODER_RECOVERY_UNCONFIRMED"
    finally:
        server.close()
        context.term()


@pytest.mark.parametrize("failure_at", ["open_ack", "append_ack", "bridge_setup"])
def test_actual_producer_recovers_unknown_ack_and_marks_data_failure(
    monkeypatch, failure_at
):
    import contextlib
    import json
    import queue
    from datetime import timedelta
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        remote_encoder,
    )

    sent, recovered, closed = [], [], []

    def receive():
        if failure_at == "open_ack" or len(sent) > 1:
            raise zmq.Again()
        return {"ok": True}

    sock = SimpleNamespace(
        setsockopt=lambda *a: None,
        connect=lambda *a: None,
        send_multipart=lambda parts: sent.append(json.loads(parts[0])),
        recv_json=receive,
        close=lambda: closed.append("socket"),
    )
    context = SimpleNamespace(
        socket=lambda *a: sock, term=lambda: closed.append("context")
    )

    def bridge(*a, **kw):
        if failure_at == "bridge_setup":
            raise RuntimeError("Injected bridge failure")
        return object()

    def abort(*args, **kwargs):
        assert closed == ["socket", "context"]
        recovered.append((args, kwargs))
        return {"ok": True, "released": True}

    env = dict(
        zmq=SimpleNamespace(
            Context=lambda: context,
            REQ=zmq.REQ,
            LINGER=zmq.LINGER,
            RCVTIMEO=zmq.RCVTIMEO,
            SNDTIMEO=zmq.SNDTIMEO,
        ),
        json=json,
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                set_device=lambda x: None,
                Stream=lambda **kw: object(),
                stream=lambda x: contextlib.nullcontext(),
            )
        ),
        dist=SimpleNamespace(TCPStore=lambda *a, **kw: SimpleNamespace(port=1234)),
        timedelta=timedelta,
        bridge_group=bridge,
        request_abort=abort,
        queue=queue,
    )
    for name, value in env.items():
        monkeypatch.setattr(remote_encoder, name, value)
    commands = queue.Queue()
    commands.put(
        (
            "append",
            SimpleNamespace(
                shape=(3, 1, 2, 2),
                numpy=lambda: SimpleNamespace(tobytes=lambda: b"x" * 12),
            ),
            0,
        )
    )
    session = SimpleNamespace(
        timeout=1,
        port=31001,
        token="old",
        device=0,
        profile_dir=None,
        stage_timing_dir=None,
        remote_usage={},
        admission_endpoint=None,
        accepted_frames=1,
        capacity_frames=51,
        capacity_bytes=1024,
        failure=None,
        recovery=None,
        commands=commands,
        stop=threading.Event(),
    )
    remote_encoder.RemoteEncoderSession._produce(session)
    assert session.failure is not None and len(recovered) == 1
    assert recovered[0][0] == (31001, "old", 1)
    assert recovered[0][1]["require_restart"] == (failure_at == "bridge_setup")
    assert all(command["op"] != "close" for command in sent)


def test_abort_send_and_receive_share_one_deadline(monkeypatch):
    from types import SimpleNamespace

    now = [0.0]
    budgets = {}

    def send(message):
        now[0] = 0.75

    socket = SimpleNamespace(
        setsockopt=lambda key, value: budgets.__setitem__(key, value),
        connect=lambda url: None,
        send_json=send,
        recv_json=lambda: dict(ok=True, session="old", released=True),
        close=lambda: None,
    )
    monkeypatch.setattr(
        recovery.zmq,
        "Context",
        lambda: SimpleNamespace(socket=lambda typ: socket, term=lambda: None),
    )
    monkeypatch.setattr(recovery.time, "monotonic", lambda: now[0])
    assert recovery.request_abort(31001, "old", 1)["released"]
    assert budgets[zmq.SNDTIMEO] == 1000
    assert budgets[zmq.RCVTIMEO] == 250
