# SPDX-License-Identifier: Apache-2.0
"""Shared memory frame publisher for disaggregated streaming output frames."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    publish_async_shared_memory_error,
    publish_async_shared_memory_payload,
    reserve_async_shared_memory_payload,
    wait_for_async_shared_memory_terminal,
)
from sglang.multimodal_gen.runtime.utils.realtime_video import stage_rgb24_segments

_MAX_OUTPUT_BYTES = 128 << 20


class SharedMemoryFramePublisher:
    """Reserve a shared-memory reference per dispatch, publish after decode."""

    def __init__(self, *, timeout_seconds: float = 2.0, max_in_flight: int = 2):
        self.timeout = timeout_seconds
        self._slots = threading.BoundedSemaphore(max_in_flight)
        self._executor = ThreadPoolExecutor(
            max_workers=max_in_flight, thread_name_prefix="disagg-shm-publish"
        )

    def reserve(self, *, frames: int, width: int, height: int, identity: dict):
        num_bytes = frames * width * height * 3
        if not 0 < num_bytes <= _MAX_OUTPUT_BYTES:
            raise ValueError("Shared memory output exceeds its payload capacity")
        if not self._slots.acquire(timeout=self.timeout):
            raise TimeoutError("Shared memory output slots are full")
        try:
            reference = reserve_async_shared_memory_payload(num_bytes)
            reference.update(
                kind="raw_rgb24_frames",
                version=1,
                batch_num_frames=[frames],
                width=width,
                height=height,
                channels=3,
                **identity,
            )
        except BaseException:
            self._slots.release()
            raise
        return reference

    def publish(self, reference, frames):
        device_frames, counts = stage_rgb24_segments([frames])
        if counts != reference["batch_num_frames"] or (
            device_frames.numel() != reference["num_bytes"]
        ):
            raise ValueError("Decoded RGB does not match the reserved reference")
        publish_async_shared_memory_payload(reference, device_frames.cpu().numpy())
        self._executor.submit(self._reclaim, reference)

    def fail(self, reference, error=None):
        try:
            publish_async_shared_memory_error(reference, error)
        except Exception:
            logging.exception("Shared memory output error publication failed")
        self._executor.submit(self._reclaim, reference)

    def _reclaim(self, reference):
        try:
            wait_for_async_shared_memory_terminal(reference)
        except Exception:
            logging.exception("Shared memory output reference was not reclaimed")
        finally:
            self._slots.release()

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=False)
