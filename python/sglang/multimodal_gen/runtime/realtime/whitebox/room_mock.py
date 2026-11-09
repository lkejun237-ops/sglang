# SPDX-License-Identifier: Apache-2.0
"""Deterministic shared-room driver for CPU tests; never eligible for training."""

import asyncio
import copy
import time

from .contracts import EngineFrame


class MockRoom:
    max_players = 16

    def __init__(self):
        self.players, self.inputs = {}, {}
        self.tick, self.started = 0, time.monotonic_ns()

    async def join(self, *, player_id, spec):
        view = MockView(self, player_id, spec)
        self.players[player_id] = view
        return view

    async def close(self):
        for view in list(self.players.values()):
            await view.close()


class MockView:
    metadata = {"mode": "shared_room_mock", "training_eligible": False}

    def __init__(self, room, player_id, spec):
        self.room, self.player_id, self.spec = room, player_id, spec
        self.renderer, self.stopped = MockUI(), False
        self.sequence, self.index = -1, 0
        self.origin = time.monotonic()

    async def send_input(self, sequence, payload):
        if sequence != self.sequence + 1:
            raise ValueError("Invalid player input sequence")
        self.sequence = sequence
        self.room.inputs[self.player_id] = {
            "sequence": sequence,
            "input": copy.deepcopy(payload),
        }
        self.room.tick += 1

    async def frames(self):
        while not self.stopped:
            await asyncio.sleep(
                max(0, self.origin + self.index / self.spec.fps - time.monotonic())
            )
            if self.stopped:
                break
            color = bytes(
                (self.room.tick % 255, int(self.player_id[:2], 16), self.index % 255)
            )
            state = {
                "players": sorted(self.room.players),
                "tick": self.room.tick,
                "viewer": self.player_id,
            }
            yield EngineFrame(
                self.index,
                self.spec.pts(self.index),
                color * (self.spec.width * self.spec.height),
                state,
                {"players": copy.deepcopy(self.room.inputs)},
                provenance={
                    "simulation_tick": self.room.tick,
                    "room_source_time_us": (time.monotonic_ns() - self.room.started)
                    // 1000,
                },
            )
            self.index += 1

    async def finish(self):
        self.stopped = True

    async def close(self):
        self.stopped = True
        self.room.players.pop(self.player_id, None)
        self.room.inputs.pop(self.player_id, None)


class MockUI:
    metadata = {"mode": "mock", "training_eligible": False}

    async def render(self, rgb, frame, spec):
        # A visible mock-only strip proves the third video is a separate output.
        return bytes([255, 255, 255]) * spec.width + rgb[spec.width * 3 :]

    async def close(self):
        pass


async def shared_mock_room(**kwargs):
    return MockRoom()
