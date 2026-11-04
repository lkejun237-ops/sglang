# SPDX-License-Identifier: Apache-2.0
"""Startup-owned ports and TCPStore for service connection handshakes."""

import socket
from datetime import timedelta

from .worker_connections import service_descriptor
from .worker_services import service_ports


class ConnectionPorts:
    def __init__(self, role, count, *, worker_id, host, bind_host, **bases):
        self.ports = service_ports(count, **bases)
        self.services = [
            service_descriptor(
                role,
                worker_id,
                i,
                host,
                **{name: values[i] for name, values in self.ports.items()},
            )
            for i in range(count)
        ]
        self.listeners = []
        self.stores = [None] * count
        try:
            # Reserve rendezvous ports before announcing them. TCPStore adopts
            # the exact listening FD rather than closing/rebinding with a race.
            for i, port in enumerate(self.ports.get("rendezvous", [])):
                listener = socket.socket(
                    socket.AF_INET6 if ":" in bind_host else socket.AF_INET
                )
                self.listeners.append(listener)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((bind_host, port))
                listener.listen(128)
        except BaseException:
            self.close()
            raise

    def initialize_stores(self, *, timeout=120):
        import torch.distributed as dist

        for i, listener in enumerate(self.listeners):
            # libuv does not support master_listen_fd. A store belongs to the
            # service lifetime; each connection uses a separately fenced prefix.
            self.stores[i] = dist.TCPStore(
                self.services[i]["endpoints"]["rendezvous"]
                .split("://", 1)[1]
                .rsplit(":", 1)[0]
                .strip("[]"),
                listener.getsockname()[1],
                None,
                True,
                timeout=timedelta(seconds=timeout),
                wait_for_workers=False,
                master_listen_fd=listener.fileno(),
                use_libuv=False,
            )
            # TCPStore now owns the descriptor.
            listener.detach()
        self.listeners.clear()

    def close(self):
        self.stores = []
        for listener in self.listeners:
            listener.close()
        self.listeners.clear()


def bridge_epoch(store, group, rank, *, timeout=120):
    """Fence a reused startup TCPStore without exposing another public field."""
    import hashlib
    import json
    import secrets
    import time

    key = "connection-handshake/" + hashlib.sha256(group.encode()).hexdigest()
    nonce = secrets.token_hex(16)
    deadline = time.monotonic() + timeout
    if rank == 1:
        store.set(key + "/receiver", nonce)
    while time.monotonic() < deadline:
        if rank == 0 and store.check([key + "/receiver"]):
            receiver = store.get(key + "/receiver").decode()
            store.set(key + "/sender", json.dumps([receiver, nonce]))
            ack = key + "/ack/" + nonce
            if store.check([ack]) and store.get(ack).decode() == receiver:
                store.delete_key(ack)
                return receiver + ":" + nonce
        elif rank == 1 and store.check([key + "/sender"]):
            receiver, sender = json.loads(store.get(key + "/sender"))
            if receiver == nonce:
                store.set(key + "/ack/" + sender, nonce)
                return receiver + ":" + sender
        time.sleep(0.005)
    raise TimeoutError("Worker bridge handshake timed out")


def confirm_bridge_connection(store, prefix, rank, device, *, timeout=120):
    """Confirm local CUDA completion and the peer's completion before connected.

    ``prefix`` must identify this connection attempt (the bridge epoch), never
    just a reusable worker group ID. Work.wait() alone only orders CUDA work.
    This runs on the thread/stream that established the bridge.
    """
    import time

    import torch

    deadline = time.monotonic() + timeout
    if torch.device(device).type == "cuda":
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        while not event.query():
            if time.monotonic() >= deadline:
                raise TimeoutError("Worker connection completion timed out")
            time.sleep(0.005)
    key = prefix + "/connection-complete/"
    store.set(key + str(rank), "complete")
    while not store.check([key + str(1 - rank)]):
        if time.monotonic() >= deadline:
            raise TimeoutError("Worker peer connection confirmation timed out")
        time.sleep(0.005)
