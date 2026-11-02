# SPDX-License-Identifier: Apache-2.0
"""Independent service ports sharing one role's model execution loop."""

import json


class OneWayCommandClient:
    """DEALER wire adapter; lifecycle calls still receive their own replies.

    The empty delimiter also lets the ROUTER listener serve existing REQ peers.
    Data commands explicitly opt out of replies; no background ACK reader or
    unbounded pending-reply queue is required.
    """

    def __init__(self, context):
        import zmq

        self.socket = context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.SNDHWM, 10)

    def setsockopt(self, *args):
        self.socket.setsockopt(*args)

    def connect(self, endpoint):
        self.socket.connect(endpoint)

    def send_json(self, command):
        self.socket.send_multipart([b"", json.dumps(command).encode()])

    def recv_json(self):
        parts = self.socket.recv_multipart()
        if len(parts) != 2 or parts[0] != b"":
            raise ValueError("Invalid worker reply envelope")
        return json.loads(parts[1])

    def close(self):
        self.socket.close()


def service_ports(count, **bases):
    """Expand base ports before loading models; reject overlapping listeners."""
    if type(count) is not int or count < 1:
        raise ValueError("service-count must be a positive integer")
    ports = {}
    used = set()
    for name, base in bases.items():
        if base is None:
            continue
        if type(base) is not int or base < 1024 or base + count - 1 > 65535:
            raise ValueError(f"Invalid {name} service port range")
        values = list(range(base, base + count))
        if used.intersection(values):
            raise ValueError("Worker service port ranges overlap")
        used.update(values)
        ports[name] = values
    return ports


class ServiceCommandSockets:
    """Poll independent REP sockets fairly on the existing command owner thread.

    The listener supplies the service identity, never the remote message. The
    existing role control broadcasts it to every rank before executing work.
    """

    def __init__(
        self, context, ports, *, timeout_seconds, router=False, host="127.0.0.1"
    ):
        import zmq

        self.sockets = []
        self.poller = zmq.Poller()
        self.selected = None
        self.cursor = 0
        self.router = router
        self.envelope = None
        self.oneway = False
        try:
            for port in ports:
                socket = context.socket(zmq.ROUTER if router else zmq.REP)
                self.sockets.append(socket)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.SNDTIMEO, timeout_seconds * 1000)
                socket.setsockopt(zmq.RCVHWM, 10)
                socket.bind(f"tcp://{host}:{port}")
                self.poller.register(socket, zmq.POLLIN)
        except BaseException:
            self.close()
            raise

    def poll(self, timeout):
        ready = dict(self.poller.poll(timeout))
        for offset in range(len(self.sockets)):
            index = (self.cursor + offset) % len(self.sockets)
            if self.sockets[index] in ready:
                self.selected = index
                self.cursor = (index + 1) % len(self.sockets)
                return True
        return False

    def recv_multipart(self):
        parts = self.sockets[self.selected].recv_multipart()
        if self.router:
            # REQ and DEALER use the same empty-delimited command envelope.
            # Preserve the routing identity locally, never in the command body.
            if len(parts) < 3 or parts[1] != b"":
                raise ValueError("Invalid worker command envelope")
            self.envelope, parts = parts[:2], parts[2:]
        command = json.loads(parts[0])
        if not isinstance(command, dict):
            raise ValueError("Worker command must be an object")
        command["_service_index"] = self.selected
        self.oneway = self.router and command.get("oneway", False)
        if type(self.oneway) is not bool or (
            self.oneway and command.get("op") not in ("append", "source_progress")
        ):
            raise ValueError("Invalid one-way worker operation")
        return [json.dumps(command).encode(), *parts[1:]]

    def send_json(self, reply):
        if self.oneway:
            return
        if self.router:
            self.sockets[self.selected].send_multipart(
                [*self.envelope, json.dumps(reply).encode()]
            )
        else:
            self.sockets[self.selected].send_json(reply)

    def close(self):
        for socket in self.sockets:
            socket.close()


class LocalServiceCommands:
    """HTTP submits commands; the existing role loop remains their sole owner."""

    def __init__(self, sockets):
        from queue import Queue

        self.sockets = sockets
        self.commands = Queue()
        self.local = None
        self.selected = 0
        self.pending = None
        self.guard = None

    def submit(self, index, command, timeout):
        from concurrent.futures import Future

        future = Future()
        self.commands.put((index, command, future))
        return future.result(timeout=timeout)

    def poll(self, timeout):
        from queue import Empty

        try:
            self.selected, self.pending, self.local = self.commands.get_nowait()
            return True
        except Empty:
            if self.sockets.poll(min(timeout, 5)):
                self.selected = self.sockets.selected
                return True
            return False

    def recv_multipart(self):
        if self.local is not None:
            return [
                json.dumps(dict(self.pending, _service_index=self.selected)).encode()
            ]
        parts = self.sockets.recv_multipart()
        command = json.loads(parts[0])
        if self.guard is not None and not self.guard(self.selected, command):
            return [
                json.dumps(
                    dict(op="group_rejected", _service_index=self.selected)
                ).encode()
            ]
        return parts

    def send_json(self, reply):
        if self.local is not None:
            future, self.local = self.local, None
            future.set_result(reply)
        else:
            self.sockets.send_json(reply)

    def close(self):
        from queue import Empty

        error = RuntimeError("Worker command owner stopped")
        if self.local is not None and not self.local.done():
            self.local.set_exception(error)
            self.local = None
        while True:
            try:
                _, _, future = self.commands.get_nowait()
            except Empty:
                break
            future.set_exception(error)
        self.sockets.close()
