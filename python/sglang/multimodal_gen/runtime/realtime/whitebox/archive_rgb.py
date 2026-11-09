# SPDX-License-Identifier: Apache-2.0
"""Bounded, parent-owned RGB leases for one archive worker generation."""

from __future__ import annotations

import mmap
import os
from collections import deque
from multiprocessing.shared_memory import SharedMemory
from uuid import uuid4


class RGBBudget:
    def __init__(self, limit=96 << 20):
        self.limit, self.used, self.high_water = limit, 0, 0

    def reserve(self, size):
        if size > self.limit - self.used:
            raise BufferError("Archive RGB capacity is unavailable")
        self.used += size
        self.high_water = max(self.high_water, self.used)

    def release(self, size):
        self.used -= size
        if self.used < 0:
            raise RuntimeError("Archive RGB budget released twice")


class RGBSlots:
    def __init__(self, frame_bytes, budget, *, frames=8, max_bytes=32 << 20):
        self.frames = min(frames, max_bytes // frame_bytes)
        if self.frames < 1:
            raise ValueError("Archive RGB frame exceeds its byte budget")
        self.frame_bytes, self.budget = frame_bytes, budget
        self.size = self.frames * frame_bytes
        if os.path.isdir("/dev/shm"):
            space = os.statvfs("/dev/shm")
            if space.f_bavail * space.f_frsize < self.size + (8 << 20):
                raise BufferError("Archive shared memory capacity is unavailable")
        self.budget.reserve(self.size)
        try:
            self.memory = SharedMemory(create=True, size=self.size)
        except BaseException:
            self.budget.release(self.size)
            raise
        self.generation = uuid4().hex
        self.free = deque(range(self.frames))
        self.leases = {}
        self.serial = 0
        self.closed = False
        self.high_water = 0

    def offer(self, rgb):
        if self.closed or len(rgb) != self.frame_bytes:
            raise ValueError("Invalid archive RGB frame")
        if not self.free:
            raise BufferError("Archive RGB slots are full")
        slot = self.free.popleft()
        self.serial += 1
        self.leases[slot] = self.serial
        start = slot * self.frame_bytes
        self.memory.buf[start : start + self.frame_bytes] = rgb
        self.high_water = max(self.high_water, len(self.leases))
        return dict(
            name=self.memory.name,
            generation=self.generation,
            slot=slot,
            serial=self.serial,
            frame_bytes=self.frame_bytes,
            frames=self.frames,
        )

    def release(self, descriptor):
        if (
            self.closed
            or descriptor["name"] != self.memory.name
            or descriptor["generation"] != self.generation
            or self.leases.get(descriptor["slot"]) != descriptor["serial"]
        ):
            raise ValueError("Stale archive RGB receipt")
        del self.leases[descriptor["slot"]]
        self.free.append(descriptor["slot"])

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.memory.close()
        self.memory.unlink()
        self.budget.release(self.size)
        self.leases.clear()
        self.free.clear()


class RGBReader:
    def __init__(self, frame_bytes):
        self.frame_bytes = frame_bytes
        self.buffers = {}
        self.serials = {}

    def read(self, descriptor):
        name, generation = descriptor["name"], descriptor["generation"]
        slot, serial, count = (
            descriptor["slot"],
            descriptor["serial"],
            descriptor["frames"],
        )
        if (
            descriptor["frame_bytes"] != self.frame_bytes
            or type(slot) is not int
            or type(serial) is not int
            or not 0 <= slot < count <= 8
            or serial <= self.serials.get(name, 0)
        ):
            raise ValueError("Invalid archive RGB lease")
        if name not in self.buffers:
            if len(self.buffers) >= 2:
                raise ValueError("Too many archive RGB buffers")
            memory = SharedMemory(name=name)
            # Darwin rounds the shared-memory object's allocation to a page.
            if (
                not count * self.frame_bytes
                <= memory.size
                < count * self.frame_bytes + mmap.PAGESIZE
            ):
                memory.close()
                raise ValueError("Archive RGB buffer size changed")
            self.buffers[name] = generation, memory
        expected, memory = self.buffers[name]
        if generation != expected:
            raise ValueError("Archive RGB worker generation changed")
        self.serials[name] = serial
        start = slot * self.frame_bytes
        # One bounded copy belongs to the codec's retained VFR input.
        return bytes(memory.buf[start : start + self.frame_bytes])

    def close(self):
        for _, memory in self.buffers.values():
            memory.close()
        self.buffers.clear()
