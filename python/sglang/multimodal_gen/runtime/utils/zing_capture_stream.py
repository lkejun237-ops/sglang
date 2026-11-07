# SPDX-License-Identifier: Apache-2.0
"""Separate process-owned CUDA streams for Zing captures and Decoder jobs."""

import json
import os
import threading

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
_streams = {}
_lock = threading.Lock()


def _private_stream(role):
    """Keep one non-pooled stream per role/device for the process lifetime.

    torch.cuda.Stream() cycles through a process-wide pool. Holding its Python
    wrapper does not reserve the underlying stream against a later RPC thread.
    Such a thread must never synchronize a stream while it is being captured.

    Captures and Decoder jobs use different streams. Keep each alive through
    every graph and library workspace, including model teardown. Like PyTorch's
    own stream pools, the bounded device cache is reclaimed at process exit;
    Python finalization must not destroy streams after CUDA runtime shutdown.
    """
    device = torch.cuda.current_device()
    key = role, device
    with _lock:
        if key not in _streams:
            from cuda.bindings import driver

            error, handle = driver.cuStreamCreate(
                driver.CUstream_flags.CU_STREAM_NON_BLOCKING
            )
            if error != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"Cannot create private Zing {role} stream: {error}")
            try:
                stream = torch.cuda.ExternalStream(int(handle), device=device)
            except BaseException:
                driver.cuStreamDestroy(handle)
                raise
            _streams[key] = (handle, stream)
            logger.info(
                "ZING_%s_STREAM_READY_JSON %s",
                role.upper(),
                json.dumps(
                    dict(pid=os.getpid(), device=device, stream=stream.cuda_stream)
                ),
            )
        return _streams[key][1]


def capture_stream():
    return _private_stream("capture")


def decoder_stream():
    # The worker's existing step gate serializes all session/lane decode jobs.
    # Reusing this stream avoids populating allocator caches on each pool stream.
    return _private_stream("decoder")
