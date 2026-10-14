# SPDX-License-Identifier: Apache-2.0
"""CUDA graph replay of Zing denoise forwards whose history rotations are all reused.

The model decides eligibility and builds the key: everything a captured
forward reads by address or decides on the host. This module owns warm-up,
capture, replay and the bounded graph set. Static input tensors are allocated
outside the graph pool and refreshed before every replay.
"""

import collections

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
    nccl_during_capture,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Arbitrary; often enough to read a hit rate from a benchmark log.
_ROUTE_LOG_INTERVAL = 256


class GraphWarmupError(RuntimeError):
    """The eager warm-up before a capture failed; nothing was captured."""


# Route intentions a rank votes on before a sequence-parallel denoise forward.
ROUTE_INELIGIBLE = 0
ROUTE_WARMUP = 1
ROUTE_CAPTURE = 2
ROUTE_REPLAY = 3


class RouteCoordinator:
    """Agrees a route across a sequence-parallel group over its CPU group.

    Warm-up and capture issue different collectives than a replay, so ranks that
    pick different routes would deadlock; every denoise forward votes exactly once.
    """

    def __init__(self, group):
        self.group = group

    def agree(self, route):
        """True when every rank intends the same route."""
        votes = torch.tensor([route, -route], dtype=torch.int64)
        dist.all_reduce(votes, op=dist.ReduceOp.MAX, group=self.group)
        return votes[0].item() == -votes[1].item()

    def all_ok(self, ok):
        flag = torch.tensor([int(ok)], dtype=torch.int64)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.group)
        return bool(flag.item())


class CudaGraphBackend:
    """Captures torch.cuda.CUDAGraph objects into one shared memory pool.

    Graphs never run concurrently and every output is cloned right after its
    replay, so a later capture may reuse memory an earlier graph freed.
    """

    def __init__(self):
        self.capturing = False
        self._pool = None
        # One side stream for every capture: graphs bake in per-stream state
        # such as the cuBLAS workspace, and pooled streams would scatter it.
        self._stream = None

    def supports(self, device):
        return device.type == "cuda" and torch.version.hip is None

    def capture(self, fn, warmed=None):
        """Warm up ``fn`` eagerly, then capture it.

        ``warmed(ok)`` runs once the local warm-up ended and before anything is
        captured. It returns whether every peer rank warmed up too; a capture
        issues collectives, so no rank may start one while a peer failed.
        """
        current = torch.cuda.current_stream()
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        stream = self._stream
        stream.wait_stream(current)
        try:
            # Lazy initialization (cuBLAS workspaces, Triton and FA4 compilation)
            # must happen on the capture stream before capturing.
            warmup_error = None
            try:
                with torch.cuda.stream(stream):
                    fn()
            except Exception as error:
                warmup_error = error
            warmed_everywhere = warmed(warmup_error is None) if warmed else True
            if warmup_error is not None or not warmed_everywhere:
                raise GraphWarmupError(
                    "Zing denoise graph warm-up failed"
                    if warmup_error is not None
                    else "Zing denoise graph warm-up failed on a peer rank"
                ) from warmup_error
            current.wait_stream(stream)
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            self.capturing = True
            # Other threads (decoders, encoders) keep using CUDA meanwhile.
            with (
                nccl_during_capture(),
                torch.cuda.graph(
                    graph,
                    pool=self._pool,
                    stream=stream,
                    capture_error_mode="thread_local",
                ),
            ):
                outputs = fn()
        finally:
            self.capturing = False
            # torch.cuda.graph.__exit__ leaves the capture stream current when
            # capture_end raises on an invalidated capture.
            torch.cuda.set_stream(current)
            current.wait_stream(stream)
        return graph, outputs

    def replay(self, graph):
        graph.replay()


class _Entry:
    def __init__(self, graph, static, outputs):
        self.graph = graph
        self.static = static
        self.outputs = outputs
        # _foreach_copy_ takes its fused path only within one dtype.
        names = collections.defaultdict(list)
        for name, tensor in static.items():
            names[tensor.dtype].append(name)
        self.copies = tuple(
            (group, [static[name] for name in group]) for group in names.values()
        )

    def refresh(self, tensors):
        for names, static in self.copies:
            torch._foreach_copy_(static, [tensors[name] for name in names])


class ZingDenoiseGraphs:
    """Model-owned graphs keyed by everything a captured denoise forward reads.

    A key is captured on its second sighting, so the first forward of every
    new shape runs eager and warms its kernels. ``routes`` counts one outcome
    per denoise forward: an ineligibility reason, warmup, capture, replay,
    warmup_failed, capture_failed or graphs_disabled. A failed warm-up runs
    eager and keeps graphs enabled. A capture failure disables graphs for
    this model and propagates: a failed CUDA capture can leave RNG and allocator
    state unusable, so that forward cannot fall back to eager.

    With a ``coordinator`` (sequence parallel) every denoise forward votes once
    through ``vote_eager`` or ``run``; a disagreement runs eager on all ranks and
    forgets the key. A failed warm-up on any rank fails all of them before any
    capture collective is issued; a failed capture fails all of them too.
    """

    def __init__(self, *, backend=None, max_graphs=32):
        if max_graphs < 1:
            raise ValueError("Zing denoise graph budget must be positive")
        self.backend = CudaGraphBackend() if backend is None else backend
        self.max_graphs = max_graphs
        self.routes = collections.Counter()
        self.graphs = collections.OrderedDict()
        self._seen = collections.OrderedDict()
        self.disabled = False
        self.coordinator = None
        self._routed = 0

    def count(self, route):
        self.routes[route] += 1
        self._routed += 1
        if self._routed % _ROUTE_LOG_INTERVAL == 0:
            logger.info("Zing denoise graph routes: %s", dict(self.routes))

    def vote_eager(self):
        """Tell the other ranks this forward cannot use a graph."""
        if self.coordinator is not None:
            self.coordinator.agree(ROUTE_INELIGIBLE)

    def run(self, key, *, tensors, capture, replay_host):
        """Outputs of this forward, or None when it must run eager.

        ``capture(static)`` runs the forward on the static copies of
        ``tensors``. ``replay_host()`` applies the host effects that a replay
        skips; a capture applies them by running the forward's Python.
        """
        if self.disabled:
            self.count("graphs_disabled")
            return None
        entry = self.graphs.get(key)
        first_sighting = key not in self._seen
        if self.coordinator is not None:
            if entry is not None:
                route = ROUTE_REPLAY
            else:
                route = ROUTE_WARMUP if first_sighting else ROUTE_CAPTURE
            if not self.coordinator.agree(route):
                # Forgetting the key makes every rank warm it up together next time.
                self.graphs.pop(key, None)
                self._seen.pop(key, None)
                self.count("route_mismatch")
                return None
        self._remember(key)
        if entry is None:
            if first_sighting:
                self.count("warmup")
                return None
            entry = self._capture(key, tensors, capture)
            if entry is None:
                return None
            self.count("capture")
        else:
            self.graphs.move_to_end(key)
            entry.refresh(tensors)
            replay_host()
            self.count("replay")
        self.backend.replay(entry.graph)
        return tuple(output.clone() for output in entry.outputs)

    def _disable(self):
        self.disabled = True
        self.graphs.clear()
        self._seen.clear()
        self.count("capture_failed")
        logger.error(
            "Zing denoise graph capture failed; denoise graphs are disabled for "
            "this model. The process CUDA state may be unusable; a restart is "
            "recommended",
            exc_info=True,
        )

    def _fail_together(self):
        if self.coordinator is not None:
            self.coordinator.all_ok(False)

    def _remember(self, key):
        self._seen[key] = None
        self._seen.move_to_end(key)
        while len(self._seen) > 4 * self.max_graphs:
            self._seen.popitem(last=False)

    def _capture(self, key, tensors, capture):
        static = {name: tensor.clone() for name, tensor in tensors.items()}
        # Under sequence parallel the backend votes on the warm-up outcome before
        # any capture collective, so every rank leaves together on a failure.
        votes = {} if self.coordinator is None else {"warmed": self.coordinator.all_ok}
        try:
            graph, outputs = self.backend.capture(lambda: capture(static), **votes)
        except GraphWarmupError:
            if self.coordinator is not None:
                # Every rank raised from the same vote and none began a capture;
                # a warm-up that failed part-way may have skipped collectives, so
                # graphs stay off rather than retrying the key.
                self._disable()
                raise
            # No capture began, so CUDA state is intact: run this forward eager
            # and warm the key up again before the next attempt.
            self._seen.pop(key, None)
            self.count("warmup_failed")
            logger.warning("Zing denoise graph warm-up failed", exc_info=True)
            return None
        except Exception:
            self._fail_together()
            self._disable()
            raise
        if self.coordinator is not None and not self.coordinator.all_ok(True):
            self._disable()
            raise RuntimeError("Zing denoise graph capture failed on a peer rank")
        entry = _Entry(graph, static, outputs)
        self.graphs[key] = entry
        while len(self.graphs) > self.max_graphs:
            self.graphs.popitem(last=False)
        return entry
