# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU command rendezvous for the dedicated encoder worker."""

import json


class EncoderControlError(RuntimeError):
    """Fatal control-plane disagreement or communication failure."""


class EncoderCommandControl:
    def __init__(self, dist, *, rank, socket, idle_group, operation_group):
        self.dist = dist
        self.rank = rank
        self.socket = socket
        self.idle_group = idle_group
        self.operation_group = operation_group
        self.world_size = dist.get_world_size(operation_group)
        self.sequence = 0
        self.request_pending = False

    def gather(self, value, *, group=None):
        values = [None] * self.world_size
        try:
            self.dist.all_gather_object(
                values, value, group=self.operation_group if group is None else group
            )
        except Exception as exc:
            raise EncoderControlError("Encoder control communication failed") from exc
        return values

    def agree(self, stage, value=None, *, error=None, group=None):
        # Only the failure flag crosses ranks. Exception detail stays in logs.
        values = self.gather((stage, value, error is not None), group=group)
        if any(row[2] for row in values):
            raise EncoderControlError(
                "Encoder operation failed on a participating rank"
            ) from error
        if any(row[:2] != values[0][:2] for row in values):
            raise EncoderControlError("Encoder command or session state does not match")

    def next(self):
        payload = None
        envelope = None
        if self.rank == 0:
            envelope = ("IDLE", self.sequence, None)
            # Business idle has no cumulative deadline. Every individual poll
            # and CPU collective is bounded so a lost peer stops the group.
            if self.socket.poll(250):
                self.request_pending = True
                try:
                    parts = self.socket.recv_multipart()
                    command = json.loads(parts[0])
                    if not isinstance(command, dict) or not isinstance(
                        command.get("op"), str
                    ):
                        raise ValueError("Invalid encoder command")
                    payload = parts[1] if len(parts) == 2 else None
                    if len(parts) > 2:
                        raise ValueError("Invalid encoder command payload")
                    envelope = ("COMMAND", self.sequence + 1, command)
                except (ValueError, TypeError, UnicodeError):
                    envelope = ("ERROR", self.sequence + 1, None)
        rows = [envelope]
        self.dist.broadcast_object_list(rows, src=0, group=self.idle_group)
        kind, sequence, command = rows[0]
        if kind == "IDLE" and sequence == self.sequence:
            return None, None
        if kind != "COMMAND" or sequence != self.sequence + 1:
            raise EncoderControlError("Encoder command channel failed")
        self.sequence = sequence
        return command, payload

    def reply(self, value):
        if self.rank == 0:
            self.socket.send_json(value)
            self.request_pending = False
