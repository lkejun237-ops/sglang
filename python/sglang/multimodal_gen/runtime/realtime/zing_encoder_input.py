# SPDX-License-Identifier: Apache-2.0
"""Arrange validated RGB appends into the progressive VAE's 1/8/8 groups.

Wire admission and credit remain with SourceLedger. Fragments retain their
original packet identity; a packet's memory reservation can be released only
after the group containing its last frame has finished execution.
"""

import time
from collections import deque


class EncoderInputArranger:
    def __init__(self, *, frame_bytes, fps):
        self.frame_bytes, self.fps = frame_bytes, fps
        self.clear()

    def clear(self):
        self.pending = deque()
        self.buffered_frames = self.offset = self.group_index = 0
        self.ended = False

    def __call__(self, item):
        if self.ended:
            raise ValueError("Encoder input has ended")
        if item.get("kind") == "source.end":
            self.ended = True
            tasks = []
            if self.buffered_frames:
                # Do not pad here. The original codec flushes its tail on EOS.
                tasks.append(self._take(self.buffered_frames))
            tasks.append(item)
            return tasks
        if "received_mono_ns" not in item:
            item = dict(item, received_mono_ns=time.perf_counter_ns())
        self.pending.append(item)
        self.buffered_frames += item["payload"]["num_frames"]
        tasks = []
        while self.buffered_frames >= (1, 8, 8)[self.group_index]:
            tasks.append(self._take((1, 8, 8)[self.group_index]))
            self.group_index = (self.group_index + 1) % 3
        return tasks

    def _take(self, count):
        appends, released = [], []
        remaining = count
        while remaining:
            item = self.pending[0]
            payload = item["payload"]
            take = min(remaining, payload["num_frames"] - self.offset)
            start = payload["source_frame_start"] + self.offset
            fragment = dict(
                payload,
                source_frame_start=start,
                num_frames=take,
                pts_us=start * 1_000_000 // self.fps,
                data=payload["data"][
                    self.offset * self.frame_bytes : (self.offset + take)
                    * self.frame_bytes
                ],
            )
            appends.append(dict(item, payload=fragment))
            self.offset += take
            remaining -= take
            if self.offset == payload["num_frames"]:
                released.append(payload["client_seq"])
                self.pending.popleft()
                self.offset = 0
        self.buffered_frames -= count
        return dict(
            kind="source.group",
            appends=appends,
            num_frames=count,
            release_client_seqs=released,
        )
