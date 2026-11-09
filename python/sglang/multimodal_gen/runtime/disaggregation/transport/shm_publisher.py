# SPDX-License-Identifier: Apache-2.0
"""Shared memory frame publisher for disaggregated streaming output frames."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    cancel_async_shared_memory_payload,
    discard_async_shared_memory_payload,
    publish_async_shared_memory_error,
    publish_async_shared_memory_payload,
    reserve_async_shared_memory_payload,
    wait_for_async_shared_memory_terminal,
)
from sglang.multimodal_gen.runtime.utils.realtime_video import stage_rgb24_segments

_MAX_OUTPUT_BYTES = 128 << 20


class SharedMemoryFramePublisher:
    """Two bounded dispatch slots, including all progressive parts in each slot."""

    def __init__(self, *, timeout_seconds=2.0, max_in_flight=2):
        self.timeout = timeout_seconds
        self._slots = threading.BoundedSemaphore(max_in_flight)
        self._executor = ThreadPoolExecutor(
            max_workers=max_in_flight, thread_name_prefix="disagg-shm-publish"
        )
        self._lock = threading.Lock()
        self._outstanding = {}

    @staticmethod
    def _parts(reference):
        return (
            reference["parts"]
            if reference.get("kind") == "raw_rgb24_sequence"
            else [reference]
        )

    def reserve(self, *, frames, width, height, identity, frame_slices=None):
        num_bytes = frames * width * height * 3
        if not 0 < num_bytes <= _MAX_OUTPUT_BYTES:
            raise ValueError("Shared memory output exceeds its payload capacity")
        counts = [frames] if frame_slices is None else list(frame_slices)
        if (
            not counts
            or sum(counts) != frames
            or (
                frame_slices is not None
                and (
                    len(counts) > 64
                    or any(type(n) is not int or not 1 <= n <= 4 for n in counts)
                )
            )
        ):
            raise ValueError("Invalid decoder output slices")
        if not self._slots.acquire(timeout=self.timeout):
            raise TimeoutError("Shared memory output slots are full")
        parts = []
        try:
            for count in counts:
                part = reserve_async_shared_memory_payload(count * width * height * 3)
                parts.append(part)
                part.update(
                    kind="raw_rgb24_frames",
                    version=1,
                    batch_num_frames=[count],
                    width=width,
                    height=height,
                    channels=3,
                    **identity,
                )
            reference = (
                parts[0]
                if frame_slices is None
                else dict(
                    kind="raw_rgb24_sequence",
                    version=1,
                    parts=parts,
                    batch_num_frames=[frames],
                    num_bytes=num_bytes,
                    width=width,
                    height=height,
                    channels=3,
                    **identity,
                )
            )
            with self._lock:
                self._outstanding[parts[0]["path"]] = [reference, False]
        except BaseException:
            for part in parts:
                discard_async_shared_memory_payload(part)
            self._slots.release()
            raise
        return reference

    @staticmethod
    def _publish(reference, frames):
        device_frames, counts = stage_rgb24_segments([frames])
        if (
            counts != reference["batch_num_frames"]
            or device_frames.numel() != reference["num_bytes"]
        ):
            raise ValueError("Decoded RGB does not match the reserved reference")
        publish_async_shared_memory_payload(reference, device_frames.cpu().numpy())

    def publish(self, reference, frames):
        self._publish(reference, frames)
        self._reclaim_later(reference)

    def publish_slice(self, reference, index, frames):
        parts = self._parts(reference)
        self._publish(parts[index], frames)
        if index + 1 == len(parts):
            self._reclaim_later(reference)

    def fail(self, reference, error=None):
        for part in self._parts(reference):
            try:
                # Previously published parts remain valid; fail only unreadied parts.
                if (
                    Path(part["owner_path"]).exists()
                    and not Path(part["ready_path"]).exists()
                ):
                    publish_async_shared_memory_error(part, error)
            except Exception:
                logging.exception("Shared memory output error publication failed")
        self._reclaim_later(reference)

    def _reclaim_later(self, reference):
        key = self._parts(reference)[0]["path"]
        with self._lock:
            state = self._outstanding.get(key)
            if state is None or state[1]:
                return
            state[1] = True
            self._executor.submit(self._reclaim, reference)

    def _reclaim(self, reference):
        try:
            for part in self._parts(reference):
                try:
                    wait_for_async_shared_memory_terminal(part, timeout_s=self.timeout)
                except Exception:
                    logging.exception(
                        "Shared memory output reference was not reclaimed"
                    )
        finally:
            with self._lock:
                self._outstanding.pop(self._parts(reference)[0]["path"], None)
            self._slots.release()

    def close(self):
        with self._lock:
            references = [value[0] for value in self._outstanding.values()]
        for reference in references:
            for part in self._parts(reference):
                cancel_async_shared_memory_payload(part)
            self._reclaim_later(reference)
        self._executor.shutdown(wait=True, cancel_futures=False)
