# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang.multimodal_gen.runtime.realtime.worker_services import service_ports


@pytest.mark.parametrize("count", [1, 2, 3])
def test_service_ports_expand_independently(count):
    assert service_ports(count, command=31000, output=32000) == {
        "command": list(range(31000, 31000 + count)),
        "output": list(range(32000, 32000 + count)),
    }


@pytest.mark.parametrize("count", [0, -1, True])
def test_service_ports_reject_invalid_count(count):
    with pytest.raises(ValueError):
        service_ports(count, command=31000)


@pytest.mark.parametrize("bases", [dict(a=65535), dict(a=1023), dict(a=31000, b=31001)])
def test_service_ports_reject_overflow_and_collision(bases):
    with pytest.raises(ValueError):
        service_ports(2, **bases)


@pytest.mark.parametrize("router", [False, True])
def test_command_sockets_use_listener_identity_and_reply_to_its_peer(router):
    import json
    import socket

    import zmq

    from sglang.multimodal_gen.runtime.realtime.worker_services import (
        ServiceCommandSockets,
    )

    ports = []
    reservations = []
    for _ in range(2):
        reservation = socket.socket()
        reservation.bind(("127.0.0.1", 0))
        ports.append(reservation.getsockname()[1])
        reservations.append(reservation)
    for reservation in reservations:
        reservation.close()
    context = zmq.Context()
    server = ServiceCommandSockets(context, ports, timeout_seconds=2, router=router)
    peers = []
    try:
        for port in ports:
            peer = context.socket(zmq.REQ)
            peer.setsockopt(zmq.LINGER, 0)
            peer.setsockopt(zmq.RCVTIMEO, 2000)
            peer.connect(f"tcp://127.0.0.1:{port}")
            peers.append(peer)
        # The caller cannot target another service through a forged field.
        for index, peer in enumerate(peers):
            peer.send_json(dict(op="ping", _service_index=1 - index))
        seen = set()
        for _ in peers:
            assert server.poll(2000)
            command = json.loads(server.recv_multipart()[0])
            index = command["_service_index"]
            assert index not in seen
            seen.add(index)
            server.send_json(dict(service=index))
            assert peers[index].recv_json() == dict(service=index)
        assert seen == {0, 1}
    finally:
        server.close()
        for peer in peers:
            peer.close()
        context.term()


def test_oneway_data_does_not_leave_replies_before_lifecycle_barrier():
    import json
    import socket

    import zmq

    from sglang.multimodal_gen.runtime.realtime.worker_services import (
        OneWayCommandClient,
        ServiceCommandSockets,
    )

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    context = zmq.Context()
    server = ServiceCommandSockets(context, [port], timeout_seconds=2, router=True)
    peer = OneWayCommandClient(context)
    peer.setsockopt(zmq.LINGER, 0)
    peer.setsockopt(zmq.RCVTIMEO, 2000)
    peer.connect(f"tcp://127.0.0.1:{port}")
    try:
        # Send consecutive descriptors without receiving a single ACK.
        for op in ("append", "source_progress", "append"):
            peer.send_json(dict(op=op, oneway=True))
        peer.send_json(dict(op="finish"))
        for op in ("append", "source_progress", "append", "finish"):
            assert server.poll(2000)
            command = json.loads(server.recv_multipart()[0])
            assert command["op"] == op and command["_service_index"] == 0
            server.send_json(dict(ok=True, op=op))
        assert peer.recv_json() == dict(ok=True, op="finish")
        assert not peer.socket.poll(20)
    finally:
        peer.close()
        server.close()
        context.term()
