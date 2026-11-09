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
_ORIG_CUDA_STREAM = torch.cuda.Stream


def _resolve_device_index(device) -> int:
    if device is None:
        return int(torch.cuda.current_device())
    if isinstance(device, int):
        return device
    dev = torch.device(device)
    if dev.index is not None:
        return int(dev.index)
    return int(torch.cuda.current_device())


def _private_stream(role, device=None):
    """Keep one non-pooled stream per role/device for the process lifetime.

    torch.cuda.Stream() cycles through a process-wide pool. Holding its Python
    wrapper does not reserve the underlying stream against a later RPC thread.
    Such a thread must never synchronize a stream while it is being captured.

    Captures and Decoder jobs use different streams. Keep each alive through
    every graph and library workspace, including model teardown. Like PyTorch's
    own stream pools, the bounded device cache is reclaimed at process exit;
    Python finalization must not destroy streams after CUDA runtime shutdown.
    """
    if torch.cuda.Stream is not _ORIG_CUDA_STREAM or not torch.cuda.is_available():
        return (
            torch.cuda.Stream(device=device)
            if device is not None
            else torch.cuda.Stream()
        )
    device_index = _resolve_device_index(device)
    key = role, device_index
    with _lock:
        if key not in _streams:
            from cuda.bindings import driver

            with torch.cuda.device(device_index):
                error, handle = driver.cuStreamCreate(
                    driver.CUstream_flags.CU_STREAM_NON_BLOCKING
                )
            if error != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"Cannot create private Zing {role} stream: {error}")
            try:
                stream = torch.cuda.ExternalStream(int(handle), device=device_index)
            except BaseException:
                driver.cuStreamDestroy(handle)
                raise
            _streams[key] = (handle, stream)
            logger.info(
                "ZING_%s_STREAM_READY_JSON %s",
                role.upper(),
                json.dumps(
                    dict(
                        pid=os.getpid(),
                        device=device_index,
                        stream=stream.cuda_stream,
                    )
                ),
            )
        return _streams[key][1]


def capture_stream(device=None):
    return _private_stream("capture", device=device)


def decoder_stream(device=None):
    # The worker's existing step gate serializes all session/lane decode jobs.
    # Reusing this stream avoids populating allocator caches on each pool stream.
    return _private_stream("decoder", device=device)


def remote_encoder_stream(device=None):
    # Dedicated non-pooled stream for RemoteEncoderSession._produce so background
    # stream.synchronize() never collides with main-thread CUDA graph capture.
    return _private_stream("remote_encoder", device=device)


def dit_io_stream(device=None, key=None):
    # Dedicated non-pooled stream for DiTIOChannel background I/O threads.
    # Optional key isolates multi-peer channels from serializing on one CUstream.
    role = "dit_io" if key is None else f"dit_io_{key}"
    return _private_stream(role, device=device)


def conditioning_nccl_stream(device=None):
    # Dedicated non-pooled stream for NcclReceiver tensor exchanges.
    return _private_stream("conditioning_nccl", device=device)
