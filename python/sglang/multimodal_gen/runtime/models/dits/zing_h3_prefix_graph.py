# SPDX-License-Identifier: Apache-2.0
"""Bounded prompt-prefix graphs over the model's fixed KV slot."""

import torch

from sglang.multimodal_gen.runtime.models.dits import zing_h3
from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
    CudaGraphBackend,
    RouteCoordinator,
    ZingChunkGraphs,
)
from sglang.multimodal_gen.runtime.models.dits.zing_h3_joint import PREFIX_PHASES


class PrefixGraphBackend(CudaGraphBackend):
    """Independent capture pools for variable prefix shapes and LRU eviction.

    Shared graph pools require replay/lifetime ordering that a prefix-keyed LRU
    does not promise. Keep the existing side-stream and SP capture protocol,
    but let each graph retain its own allocator pool.
    """

    def capture(self, fn, warmed=None):
        self._pool = None
        return super().capture(fn, warmed=warmed)


class PrefixJointGraphs(ZingChunkGraphs):
    """A bounded graph set over the caller's single fixed KV slot.

    Caller clears the logical cache before every prompt-prefix replacement.
    A full shape budget drains the old graph generation before building another.
    The first occurrence warms up; the second captures; later calls replay.
    """

    def __init__(self, *, backend=None, max_graphs=2, capture_enabled=True):
        super().__init__(
            backend=PrefixGraphBackend() if backend is None else backend,
            max_graphs=max_graphs,
            capture_enabled=capture_enabled,
        )
        self._shapes = set()

    def _check(self, model, packs, cache):
        if self.disabled:
            raise RuntimeError("Prefix graphs are disabled after a capture failure")
        if not cache.pooled or not cache.raw_store:
            raise ValueError("Prefix graphs require a fixed raw KV pool")
        if (
            cache.prefix is not None
            or cache.committed
            or cache.active is not None
            or cache.pending_source is not None
        ):
            raise ValueError("Prefix graphs require a cleared logical cache")
        if model.adaln_table is not None:
            raise ValueError("Prefix graphs require resident AdaLN weights")
        if self.capture_enabled and not self.backend.supports(packs[0]["x"].device):
            raise ValueError("Unsupported prefix graph device")
        size, _ = zing_h3.get_ulysses_ctx()
        if size > 1:
            reason = zing_h3._sequence_parallel_topology_rejection(
                enabled=model.denoise_graph_sp
            )
            if reason is not None:
                raise ValueError(f"Unsupported prefix graph SP topology: {reason}")

    def _prepare(self, sampler, packs, cache):
        shadow = cache.planning_copy()
        packed = self._joint_packed(packs, PREFIX_PHASES)
        plan = self._forward_plan(sampler, shadow, packed, "joint")
        return plan, plan.execute, shadow

    @torch.no_grad()
    def run(self, sampler, packs, cache):
        size, _ = zing_h3.get_ulysses_ctx()
        if size > 1 and self.coordinator is None:
            self.coordinator = RouteCoordinator(zing_h3.get_sp_group().cpu_group)
        self._prepare_together(lambda: self._check(sampler.generator, packs, cache))
        try:
            plan, execute, shadow = self._prepare_together(
                lambda: self._prepare(sampler, packs, cache)
            )
            reset = (
                plan.signature not in self._shapes
                and len(self._shapes) >= self.max_graphs
            )
            if self.coordinator is not None and not self.coordinator.agree(int(reset)):
                raise RuntimeError("Prefix graph cache generations differ across ranks")
            if reset:
                # Destroy the entire old generation BEFORE capturing a new graph.
                # Evicting one captured SP graph after a new capture reproducibly
                # invalidated later replays on the target NCCL/CUDA runtime.
                if packs[0]["x"].is_cuda:
                    torch.cuda.synchronize(packs[0]["x"].device)
                self.graphs.clear()
                self._seen.clear()
                self._shapes.clear()
                self.routes[("prefix_separate", "shape_reset")] += 1
            self._shapes.add(plan.signature)
            self._discard_rotated(cache, shadow)
            video, audio = self._run(
                "prefix_separate",
                [plan],
                plan.tensors,
                execute,
                plan.kv_plan.stores,
                plan.kv_plan.rotated,
            )
            cache.publish_plan(shadow)
            return video[-packs[-1]["x"].shape[0] :], audio
        except BaseException:
            cache.clear()
            raise
