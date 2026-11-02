# SPDX-License-Identifier: Apache-2.0
"""Rank-0 transport ownership for the existing queued DiT service.

All collective methods run on the model-owner thread, in identical order on
all ranks. HTTP may only pause the local receiver; it never enters a collective.
Tensor distribution uses the runtime's existing tensor-dictionary broadcast.
"""


class ParallelChannel:
    def __init__(self, group=None, local=None):
        self.group = group
        self.local = local
        self.leader = group is None or group.rank_in_group == 0

    def leader_call(self, call):
        if self.group is None:
            return call()
        result = error = None
        if self.leader:
            try:
                result = call()
            except Exception as exc:  # noqa: BLE001 - propagate failures to the owner
                error = str(exc)
        status = {"result": result, "error": error}
        if self.group is not None:
            status = self.group.broadcast_object(status if self.leader else None)
        if status["error"] is not None:
            raise RuntimeError(status["error"])
        return status["result"]

    def check_errors(self, error):
        if self.group is None:
            if error is not None:
                raise error
            return
        errors = [
            self.group.broadcast_object(
                str(error) if error is not None else None, src=rank
            )
            for rank in range(self.group.world_size)
        ]
        if any(item is not None for item in errors):
            raise RuntimeError(f"DiT rank operation failed: {errors}")

    def pause_receiving_local(self):
        if self.leader and self.local is not None:
            self.local.pause_receiving()

    def pause_receiving(self):
        self.leader_call(self.pause_receiving_local)

    def poll(self):
        return self.leader_call(lambda: self.local.poll())

    @property
    def error(self):
        # Only used on the model thread during coordinated failure cleanup.
        return self.leader_call(
            lambda: (
                str(self.local.error)
                if self.local and self.local.error is not None
                else None
            )
        )

    def receive(self):
        message = None

        def receive():
            nonlocal message
            message = self.local.receive()

        self.leader_call(receive)
        if self.group is not None:
            message = self.group.broadcast_tensor_dict(message)
        return message

    def send(self, message):
        self.leader_call(lambda: self.local.send(message))

    def close(self):
        self.leader_call(lambda: self.local.close() if self.local is not None else None)
        self.local = None
