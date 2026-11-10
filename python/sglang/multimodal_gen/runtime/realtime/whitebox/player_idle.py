# SPDX-License-Identifier: Apache-2.0
"""Player input inactivity, independent of media and owner heartbeats."""

import json
import logging
import math
from pathlib import Path


class PlayerIdlePolicy:
    def __init__(self, path):
        self.path = Path(path)
        self.seconds = 60.0
        self.next_read = 0.0

    def timeout(self, now):
        if now < self.next_read:
            return self.seconds
        self.next_read = now + 5
        try:
            with self.path.open() as source:
                value = json.loads(source.read(4097))["player_idle_timeout_seconds"]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 86400
            ):
                raise ValueError("Invalid player idle timeout")
            self.seconds = float(value)
        except FileNotFoundError:
            pass  # Default on first boot; retain last good value on replacement.
        except (OSError, ValueError, KeyError, TypeError):
            logging.getLogger(__name__).warning(
                "Cannot read player idle policy; keeping timeout_s=%s", self.seconds
            )
        return self.seconds


def meaningful_input(value):
    """The existing SDK input shape, after admission/validation by its owner."""
    return bool(
        value.get("heldKeys")
        or value.get("keyEdges")
        or any(
            type(value.get(key)) in (int, float)
            and math.isfinite(value[key])
            and value[key] != 0
            for key in ("yawDeltaRadians", "pitchDeltaRadians", "distanceDeltaMeters")
        )
    )


class PlayerInputActivity:
    def __init__(self):
        self.last_input = None
        self.expired = False
        self.pending = {}
        self.held = {}
        self.waiting = None
        self.semantic_generations = {}

    def start(self, now):
        if self.last_input is None:
            self.last_input = now

    def touch(self, now):
        # Input can arrive while the source is still preparing. Only a valid
        # source frame arms inactivity; early input must not start that clock.
        if not self.expired and self.last_input is not None:
            self.last_input = max(self.last_input, now)

    def is_expired(self, now, timeout):
        if (
            not self.expired
            and timeout > 0
            and self.last_input is not None
            and self.waiting is None
            and now >= self.last_input + timeout
        ):
            self.expired = True
        return self.expired

    def pause(self, operation):
        if not self.expired:
            self.waiting = operation

    def resume(self, operation, now):
        if self.waiting == operation:
            self.waiting = None
            self.touch(now)  # Full reaction time, once per completed foreground wait.

    def observe_semantic(self, client, direction, message, now):
        # The World is the semantic authority. Never interpret custom game fields
        # or count handshake/neutral traffic as user activity.
        kind = message.get("type")
        generation = message.get("generation")

        def integer(value, minimum=0):
            return type(value) is int and minimum <= value <= 9007199254740991

        if direction == "to-sdk":
            if kind in ("input.semantic.open", "input.semantic.cancel"):
                if kind == "input.semantic.open":
                    self.semantic_generations.pop(client, None)
                for key in list(self.pending):
                    if key[:2] == (client, "semantic"):
                        del self.pending[key]
                return
            if kind != "input.semantic.batch" or not integer(generation, 1):
                return
            batch = message.get("batch")
            if (
                not isinstance(batch, dict)
                or not integer(batch.get("sequence"), 1)
                or not integer(batch.get("stateRevision"))
            ):
                return
            if generation < self.semantic_generations.get(client, 0):
                return
            self.semantic_generations[client] = generation
            while len(self.semantic_generations) > 256:
                self.semantic_generations.pop(next(iter(self.semantic_generations)))
            key = (client, "semantic", generation, batch["sequence"])
            self.pending.setdefault(key, (now, batch["stateRevision"], None))
            while len(self.pending) > 256:
                self.pending.pop(next(iter(self.pending)))
            return
        if (
            kind != "input.semantic.receipt"
            or message.get("clientId") != client
            or not integer(generation, 1)
            or not integer(message.get("sequence"), 1)
            or not integer(message.get("stateRevision"))
            or not integer(message.get("simulationTick"))
            or type(message.get("meaningful")) is not bool
            or generation != self.semantic_generations.get(client)
        ):
            return
        key = (client, "semantic", generation, message["sequence"])
        pending = self.pending.get(key)
        if pending is not None and pending[1] == message["stateRevision"]:
            del self.pending[key]
            if message["meaningful"]:
                self.touch(
                    pending[0]
                )  # Correlated original send time, never late receipt time.

    def observe(self, client, direction, message, now):
        """Only a matching SDK receipt confirms browser input, not raw traffic."""
        kind = message.get("type")
        if isinstance(kind, str) and kind.startswith("input.semantic."):
            self.observe_semantic(client, direction, message, now)
            return
        if direction == "to-sdk":
            if kind == "input.state":
                value = message.get("input")
                if (
                    not isinstance(value, dict)
                    or type(value.get("sequence")) is not int
                ):
                    return
                key = (client, "input", value["sequence"])
                held = bool(value.get("heldKeys"))
                active = meaningful_input(value) or self.held.get(client, False)
            elif kind == "input.action" and isinstance(message.get("actionId"), str):
                key = (client, "action", message["actionId"])
                active, held = True, None
            else:
                return
            # Bound unacknowledged traffic. No heartbeat updates last_input.
            self.pending.setdefault(key, (now, active, held))
            while len(self.pending) > 256:
                self.pending.pop(next(iter(self.pending)))
            return
        if kind == "input.receipt" and message.get("clientId") == client:
            key = (client, "input", message.get("clientSequence"))
        elif kind == "action.receipt":
            key = (client, "action", message.get("actionId"))
        else:
            return
        # Untrusted malformed identifiers must not become dict keys.
        if type(key[2]) not in (str, int):
            return
        pending = self.pending.pop(key, None)
        if pending is None or (
            kind == "action.receipt" and message.get("status") != "completed"
        ):
            return
        received, active, held = pending
        if held is not None:
            self.held[client] = held
            while len(self.held) > 256:
                self.held.pop(next(iter(self.held)))
            for old in list(self.pending):
                if old[:2] == key[:2] and old[2] < key[2]:
                    del self.pending[old]
        if active:
            self.touch(received)  # A delayed/replayed receipt cannot buy more time.
