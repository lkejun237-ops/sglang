# SPDX-License-Identifier: Apache-2.0
"""CUDA capture must defer cyclic resource finalizers and restore caller state."""

import gc
import os
import pathlib
import subprocess
import sys
from contextlib import contextmanager

import pytest
import torch

import sglang
from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
    CudaGraphBackend,
)


def _fake_cuda_capture(monkeypatch, *, fail_at=None):
    from sglang.multimodal_gen.runtime.models.dits import zing_h3_denoise_graph

    class Stream:
        def wait_stream(self, other):
            pass

    original = Stream()
    current = [original]

    @contextmanager
    def use_stream(stream):
        previous = current[0]
        current[0] = stream
        try:
            yield
        finally:
            current[0] = previous

    @contextmanager
    def capture_context(graph, *, stream, **kwargs):
        current[0] = stream
        yield
        if fail_at == "capture_end":
            raise RuntimeError("capture end failed")

    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(zing_h3_denoise_graph, "capture_stream", Stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current[0])
    monkeypatch.setattr(
        torch.cuda, "set_stream", lambda stream: current.__setitem__(0, stream)
    )
    monkeypatch.setattr(torch.cuda, "stream", use_stream)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", object)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", object)
    monkeypatch.setattr(torch.cuda, "graph", capture_context)
    return original


@pytest.mark.parametrize("gc_enabled", [False, True])
@pytest.mark.parametrize("fail_at", [None, "warmup", "capture", "capture_end"])
def test_capture_restores_gc_policy_and_stream(monkeypatch, gc_enabled, fail_at):
    original = _fake_cuda_capture(monkeypatch, fail_at=fail_at)
    backend = CudaGraphBackend()
    previous_gc = gc.isenabled()
    (gc.enable if gc_enabled else gc.disable)()
    calls = []

    def forward():
        assert gc.isenabled() == (gc_enabled and not backend.capturing)
        calls.append(backend.capturing)
        if fail_at == ("capture" if backend.capturing else "warmup"):
            raise RuntimeError("forward failed")
        return ()

    try:
        if fail_at:
            with pytest.raises(RuntimeError):
                backend.capture(forward)
        else:
            backend.capture(forward)
        assert gc.isenabled() == gc_enabled
        assert torch.cuda.current_stream() is original
        assert not backend.capturing
        assert calls == ([False] if fail_at == "warmup" else [False, True])
    finally:
        (gc.enable if previous_gc else gc.disable)()


def test_capture_defers_cyclic_kernel_finalization(monkeypatch):
    _fake_cuda_capture(monkeypatch)
    backend = CudaGraphBackend()
    finalized = []
    previous_gc, thresholds = gc.isenabled(), gc.get_threshold()

    class Kernel:
        def __init__(self):
            self.cycle = self

        def __del__(self):
            finalized.append(backend.capturing)

    def forward():
        if backend.capturing:
            Kernel()
            # Trigger automatic cyclic collection without explicit gc.collect
            # in the captured forward, as compiled Python launchers can do.
            for _ in range(100):
                cycle = []
                cycle.append(cycle)
        return ()

    try:
        gc.enable()
        gc.set_threshold(1, 1, 1)
        backend.capture(forward)
        gc.collect()
        assert finalized == [False]
    finally:
        gc.set_threshold(*thresholds)
        (gc.enable if previous_gc else gc.disable)()


# A failed capture poisons CUDA state, so the real CUDA regression is isolated.
_CUDA_FINALIZER = """
import gc
import torch
from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import CudaGraphBackend

backend = CudaGraphBackend()
x = torch.ones(8, device='cuda')
finalized = []

class CudaResource:
    def __init__(self):
        self.cycle = self
    def __del__(self):
        finalized.append(torch.cuda.is_current_stream_capturing())
        torch.cuda.synchronize()

def forward():
    if backend.capturing:
        CudaResource()
        for _ in range(100):
            cycle = []
            cycle.append(cycle)
    return x + 1

gc.enable()
gc.set_threshold(1, 1, 1)
before = torch.cuda.current_stream()
graph, output = backend.capture(forward)
assert gc.isenabled() and torch.cuda.current_stream() == before
gc.collect()
assert finalized == [False], finalized
backend.replay(graph)
torch.cuda.synchronize()
assert torch.equal(output, x + 1)
print('CUDA_FINALIZER_CAPTURE_OK')
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_finalizer_runs_only_after_capture():
    env = dict(os.environ)
    root = str(pathlib.Path(sglang.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (root, env.get("PYTHONPATH")) if p)
    result = subprocess.run(
        [sys.executable, "-c", _CUDA_FINALIZER],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CUDA_FINALIZER_CAPTURE_OK" in result.stdout
