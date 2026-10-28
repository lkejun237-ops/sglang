# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU command rendezvous for disaggregated role workers."""

from __future__ import annotations

import json


class RoleControlError(RuntimeError):
    """Fatal control-plane disagreement or communication failure."""


class RoleCommandControl:
    """Synchronize work commands and consensus across all participating ranks of a role."""

    def __init__(
        self,
        dist,
        *,
        rank: int,
        socket,
        idle_group,
        operation_group,
        role_name: str = "Worker",
    ):
        self.dist = dist
        self.rank = rank
        self.socket = socket
        self.idle_group = idle_group
        self.operation_group = operation_group
        self.role_name = role_name
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
            raise RoleControlError(
                f"{self.role_name} control communication failed"
            ) from exc
        return values

    def agree(self, stage, value=None, *, error=None, group=None):
        values = self.gather((stage, value, error is not None), group=group)
        if any(row[2] for row in values):
            raise RoleControlError(
                f"{self.role_name} operation failed on a participating rank"
            ) from error
        if any(row[:2] != values[0][:2] for row in values):
            raise RoleControlError(
                f"{self.role_name} command or session state does not match"
            )

    def next(self):
        payload = None
        envelope = None
        if self.rank == 0:
            envelope = ("IDLE", self.sequence, None)
            if self.socket.poll(250):
                self.request_pending = True
                try:
                    parts = self.socket.recv_multipart()
                    command = json.loads(parts[0])
                    if not isinstance(command, dict) or not isinstance(
                        command.get("op"), str
                    ):
                        raise ValueError(f"Invalid {self.role_name.lower()} command")
                    payload = parts[1] if len(parts) == 2 else None
                    if len(parts) > 2:
                        raise ValueError(
                            f"Invalid {self.role_name.lower()} command payload"
                        )
                    envelope = ("COMMAND", self.sequence + 1, command)
                except (ValueError, TypeError, UnicodeError):
                    envelope = ("ERROR", self.sequence + 1, None)
        rows = [envelope]
        self.dist.broadcast_object_list(rows, src=0, group=self.idle_group)
        kind, sequence, command = rows[0]
        if kind == "IDLE" and sequence == self.sequence:
            # Broadcast completion alone need not acknowledge every receiver.
            # An idle CPU rendezvous detects a lost leaf within the watchdog,
            # without leaving a device-group collective pending during idle.
            self.agree("idle", self.sequence, group=self.idle_group)
            return None, None
        if kind != "COMMAND" or sequence != self.sequence + 1:
            raise RoleControlError(f"{self.role_name} command channel failed")
        self.sequence = sequence
        return command, payload

    def reply(self, value):
        if self.rank == 0:
            self.socket.send_json(value)
            self.request_pending = False
