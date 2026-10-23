# SPDX-License-Identifier: Apache-2.0
"""CUDA graph replay of Zing denoise forwards whose history rotations are all reused.

The model decides eligibility and builds the key: everything a captured
forward reads by address or decides on the host. This module owns warm-up,
capture, replay and the bounded graph set. Static input tensors are allocated
outside the graph pool and refreshed before every replay.
"""

import collections
import json

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


class ZingChunkGraphs:
    """Bounded, model-owned whole-chunk graphs over a fixed pooled KV slot."""

    def __init__(self, *, backend=None, max_graphs=16, capture_enabled=True):
        if type(max_graphs) is not int or max_graphs < 1:
            raise ValueError("Zing chunk graph budget must be positive")
        self.backend = CudaGraphBackend() if backend is None else backend
        self.max_graphs, self.capture_enabled = max_graphs, capture_enabled
        self.graphs = collections.OrderedDict()
        self._seen = collections.OrderedDict()
        self.routes = collections.Counter()
        self.coordinator = None
        self.disabled = False

    def eligible(self, sampler, cache, current):
        from sglang.multimodal_gen.runtime.models.dits import zing_h3

        world, _ = zing_h3.get_ulysses_ctx()
        if zing_h3._has_forward_hooks(sampler.generator):
            return False
        if (
            world > 1
            and zing_h3._sequence_parallel_topology_rejection(
                enabled=sampler.generator.denoise_graph_sp
            )
            is not None
        ):
            return False
        if world > 1 and self.coordinator is None:
            group = getattr(zing_h3.get_sp_group(), "cpu_group", None)
            if group is None:
                return False
            self.coordinator = RouteCoordinator(group)
        return (
            not self.disabled
            and world in (1, 2, 4)
            and cache.pooled
            and cache.raw_store
            and sampler.window_eviction_timing == "after_step0"
            and sampler.joint_forward == "cross_chunk"
            and sampler.solver == "few_step"
            and not sampler.do_cfg
            and current.audio is None
            and sampler.window_policy.window_chunks is not None
            and (
                not self.capture_enabled or self.backend.supports(current.visual.device)
            )
        )

    @staticmethod
    def _capture_store_views(plans, stores):
        """Only snapshot the K/V spans the capture warm-up can overwrite.

        Clean rows persist raw K; every current row writes V. History outside
        these spans is read-only, and rotated K is rebuilt before being read.
        Bounds come from host plans, without reading device index tensors.
        """
        views = []
        for column, count in enumerate(("clean_rows", "rows")):
            ranges = [
                (plan.kv_plan.write_start, plan.kv_plan.write_start + rows)
                for plan in plans
                if (rows := getattr(plan.kv_plan, count))
            ]
            if ranges:
                start = min(begin for begin, _ in ranges)
                stop = max(end for _, end in ranges)
                views.extend(pair[column][start:stop] for pair in stores)
        return tuple(views)

    def _run(self, kind, plans, tensors, execute, stores, rotated=()):
        signatures = tuple(plan.signature for plan in plans)
        # ``rotated`` is addressed but never snapshotted: it is rewritten by
        # every execution before it is read.
        addresses = tuple(
            value.data_ptr() for pair in stores for value in pair
        ) + tuple(value.data_ptr() for value in rotated)
        # Inputs contain all changing metadata by value, never absolute chunk ids.
        shapes = tuple(
            (name, tuple(t.shape), t.dtype, t.device) for name, t in tensors.items()
        )
        key = (kind, signatures, addresses, shapes)
        entry = self.graphs.get(key)
        first = key not in self._seen
        route = (
            ROUTE_REPLAY
            if entry is not None
            else ROUTE_WARMUP
            if first
            else ROUTE_CAPTURE
        )
        if self.coordinator is not None and not self.coordinator.agree(route):
            self.graphs.pop(key, None)
            self._seen.pop(key, None)
            self.routes[(kind, "route_mismatch")] += 1
            return execute(tensors)
        self._seen[key] = None
        self._seen.move_to_end(key)
        while len(self._seen) > 4 * self.max_graphs:
            self._seen.popitem(last=False)
        if not self.capture_enabled or first:
            self.routes[(kind, "eager" if not self.capture_enabled else "warmup")] += 1
            return execute(tensors)
        if entry is None:
            static = {name: tensor.clone() for name, tensor in tensors.items()}
            flat = self._capture_store_views(plans, stores)
            logger.info(
                "ZING_CHUNK_GRAPH_SNAPSHOT_JSON %s",
                json.dumps(
                    {
                        "kind": kind,
                        "snapshot_bytes": sum(
                            t.numel() * t.element_size() for t in flat
                        ),
                        "store_bytes": sum(
                            t.numel() * t.element_size()
                            for pair in stores
                            for t in pair
                        ),
                    }
                ),
            )
            saved = tuple(value.clone() for value in flat)

            def restore():
                if flat:
                    torch._foreach_copy_(list(flat), list(saved))

            def capture():
                # Production capture calls this eagerly once, then under capture.
                # Only the eager warmup restores state; restore must not be baked
                # into the graph, which would discard the next replay's history.
                if self.backend.capturing:
                    return execute(static)
                try:
                    return execute(static)
                finally:
                    restore()

            try:
                votes = (
                    {}
                    if self.coordinator is None
                    else {"warmed": self.coordinator.all_ok}
                )
                graph, outputs = self.backend.capture(capture, **votes)
            except GraphWarmupError:
                # warmed() already agreed this failure before capture. A second
                # vote is unnecessary and can desynchronize peer error paths.
                self.disabled = True
                self.graphs.clear()
                self._seen.clear()
                self.routes[(kind, "warmup_failed")] += 1
                raise
            except Exception:
                if self.coordinator is not None:
                    self.coordinator.all_ok(False)
                self.disabled = True
                self.graphs.clear()
                self._seen.clear()
                self.routes[(kind, "capture_failed")] += 1
                raise
            finally:
                restore()
            if self.coordinator is not None and not self.coordinator.all_ok(True):
                self.disabled = True
                self.graphs.clear()
                self._seen.clear()
                self.routes[(kind, "capture_failed")] += 1
                raise RuntimeError("Zing chunk graph capture failed on a peer rank")
            entry = _Entry(graph, static, outputs)
            self.graphs[key] = entry
            while len(self.graphs) > self.max_graphs:
                self.graphs.popitem(last=False)
            self.routes[(kind, "capture")] += 1
        else:
            entry.refresh(tensors)
            self.graphs.move_to_end(key)
            self.routes[(kind, "replay")] += 1
        self.backend.replay(entry.graph)
        return tuple(value.clone() for value in entry.outputs)

    def _prepare_together(self, prepare):
        candidate = error = None
        try:
            candidate = prepare()
        except BaseException as failure:
            error = failure
        all_ok = self.coordinator is None or self.coordinator.all_ok(error is None)
        if error is not None:
            raise error
        if not all_ok:
            raise ValueError("Zing chunk graph preparation failed on a peer rank")
        return candidate

    @staticmethod
    def _joint_packed(packs, phases):
        from sglang.multimodal_gen.runtime.models.dits.zing_h3 import ZingHostLayout
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
            tensor_identity,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3._sampling_helpers import (
            _GUARDED_ROWS,
            _HOST_GUARD,
            _HOST_LAYOUT,
            _HOST_T,
            _host_metadata,
        )

        names = (
            "x",
            "audio_x",
            "context",
            "t",
            "block_idx",
            "position_ids",
            "row_role",
        )
        packed = {name: torch.cat([p[name] for p in packs]) for name in names}
        packed["seq_lens"] = sum(p["seq_lens"] for p in packs)
        packed["clean_seq_lens"] = sum(p["clean_seq_lens"] for p in packs)
        metadata = [_host_metadata(p) for p in packs]
        if not all(m is not None for m in metadata):
            raise ValueError("Zing chunk graphs require guarded host row metadata")
        packed[_HOST_LAYOUT] = ZingHostLayout(
            **{
                name: sum(getattr(m["host_layout"], name) for m in metadata)
                if name in {"seq_lens", "clean_seq_lens"}
                else torch.cat([getattr(m["host_layout"], name) for m in metadata])
                for name in (
                    "seq_lens",
                    "clean_seq_lens",
                    "block_idx",
                    "position_ids",
                    "row_role",
                )
            }
        )
        packed[_HOST_T] = torch.cat([m["host_t"] for m in metadata])
        packed[_HOST_GUARD] = tuple(
            (name, packed[name], tensor_identity(packed[name]))
            for name in _GUARDED_ROWS
        )
        packed["_zing_joint_phases"] = phases
        return packed

    @staticmethod
    def _forward_plan(sampler, shadow, packed, phase, after=None):
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
            ZingFixedForwardKVPlan,
        )
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_joint import (
            joint_transaction,
        )

        host = packed["_zing_host_layout"]
        positions, blocks = host.position_ids, host.block_idx
        rows = positions.shape[0]
        transaction = None
        if phase == "joint":
            transaction = joint_transaction(
                shadow,
                packed["_zing_joint_phases"],
                positions,
                blocks,
                host.row_role,
                packed["_zing_host_t"],
                int(host.clean_seq_lens.sum()),
            )
            ends = transaction.ends
            clean = transaction.branches[-1].start
        else:
            ends = (rows,)
            clean = 0 if phase == "active" else rows
        kv = ZingFixedForwardKVPlan(
            shadow,
            positions,
            ends=ends,
            clean_rows=clean,
            prefix=phase == "prefix" or (transaction is not None and len(ends) == 6),
            after=after,
        )
        if transaction is not None and len(ends) == 6:
            kv.tensors["current_positions"] = transaction.mapped_positions.to(
                packed["x"].device
            )
        model = sampler.generator
        if (
            shadow._model_signature is None
            and model.time_embedder._frequency_cache is None
        ):
            # Register the lazy constant before binding the cold cache signature;
            # creating it in the first graph would look like a live model change.
            with torch.inference_mode(False), torch.no_grad():
                model.time_embedder(
                    torch.zeros(1, dtype=torch.float32, device=packed["x"].device)
                )
        model._bind_cache(shadow, kv.history_positions)
        # _bind_cache just validated the live model (including lazy constants).
        # Reuse that snapshot within this host preparation transaction instead
        # of rescanning every parameter for the same forward plan.
        plan = model.prepare_fixed_forward(
            packed=packed,
            kv_plan=kv,
            phase=phase,
            _model_state=model._zing_model_signatures,
        )
        layers = [(k[:rows], v[:rows]) for k, v in kv.stores]
        if transaction is not None:
            transaction.commit(
                shadow,
                packed["position_ids"],
                packed["block_idx"],
                layers,
                validation=(positions, blocks),
            )
        else:
            shadow.update(
                packed["position_ids"],
                packed["block_idx"],
                phase,
                0 if phase == "prefix" else int(positions[:, 0].unique().numel()),
                None if phase == "active" else layers,
                validation=(positions, blocks),
            )
        return plan

    @staticmethod
    def _discard_rotated(cache, shadow):
        """Unpublish rotated-K reuse before any execution may overwrite it.

        An execution rewrites the rotated-K scratch of every layer, so no
        earlier publication may describe it. The real cache covers a failed
        execution and the shadow covers publication of a successful one, which
        replaces the real cache's state wholesale.
        """
        cache._rotated_store_signatures.clear()
        shadow._rotated_store_signatures.clear()

    @staticmethod
    def _attach(plans, tensors):
        for index, plan in enumerate(plans):
            for name, value in plan.tensors.items():
                tensors[f"f{index}.{name}"] = value

    @staticmethod
    def _execute(plan, index, static, x):
        inputs = {name: static[f"f{index}.{name}"] for name in plan.tensors}
        inputs["x"] = x
        return plan.execute(inputs)

    def recovery(self, sampler, current):
        return self._initial_prediction(sampler, current, kind="recovery", prefix=False)

    def initialization(self, sampler, current):
        return self._initial_prediction(sampler, current, kind="init", prefix=True)

    def resume(self, sampler, current, packs):
        return self._initial_prediction(
            sampler, current, kind="resume", prefix=False, resume_packs=packs
        )

    def _initial_prediction(self, sampler, current, *, kind, prefix, resume_packs=None):
        source, target, _, _ = current.spans[0]
        # Finish every host collective before local model preparation. A peer
        # whose prefix plan fails must meet the others at the status vote, never
        # a different chunk-begin collective.
        if prefix:
            sampler._precheck_initial_prefix(
                current.cache, current.positive, source, target
            )
        else:
            blocks = sampler._layout_index(current.positive).host["block_idx"]
            sampler._agree_sp_boundary(
                current.cache,
                target,
                phase="chunk_begin",
                required_rows=int(((blocks == source) | (blocks == target)).sum()),
                layout_signature=sampler._collective_layout_signature(
                    current.positive, source, target
                ),
            )

        def prepare():
            shadow = current.cache.planning_copy()
            source, target, _, _ = current.spans[0]
            plans = []
            if prefix:
                if sampler._has_block(current.positive, 0):
                    packed = sampler._select_block(
                        current.positive, 0, phase="prefix", video_t=1, audio_t=1
                    )
                    plans.append(self._forward_plan(sampler, shadow, packed, "prefix"))
            if resume_packs is None:
                shadow.begin_chunk(target)
            vs, _ = self._schedules(sampler, current)
            if resume_packs is None:
                packs = [
                    sampler._select_block(
                        current.positive,
                        source,
                        phase="source",
                        video_t=1,
                        audio_t=None,
                    ),
                    sampler._select_block(
                        current.positive,
                        target,
                        phase="target",
                        video_t=sampler._host_list(vs.timesteps)[0],
                        audio_t=None,
                    ),
                ]
            else:
                packs = resume_packs
            phases = (
                ("source", "active")
                if resume_packs is None
                else ("final", "source", "active")
            )
            packed = self._joint_packed(packs, phases)
            plans.append(
                self._forward_plan(
                    sampler,
                    shadow,
                    packed,
                    "joint",
                    after=plans[-1].kv_plan if plans else None,
                )
            )
            tensors = {
                "source_x": packs[0]["x"]
                if resume_packs is None
                else torch.cat([p["x"] for p in packs[:-1]]),
                "latent": current.visual,
            }
            self._attach(plans, tensors)
            target_rows = packs[-1]["x"].shape[0]

            def execute(static):
                for index, plan in enumerate(plans[:-1]):
                    self._execute(plan, index, static, static[f"f{index}.x"])
                index = len(plans) - 1
                x = torch.cat(
                    (
                        static["source_x"],
                        sampler.processor.patchify_visual(static["latent"]).to(
                            static["source_x"]
                        ),
                    )
                )
                video, _ = self._execute(plans[-1], index, static, x)
                return (
                    sampler.processor.unpatchify_visual(
                        video[-target_rows:], 1, current.visual.shape[2:]
                    ),
                )

            return plans, tensors, execute, shadow

        plans, tensors, execute, shadow = self._prepare_together(prepare)
        self._discard_rotated(current.cache, shadow)
        kv = plans[0].kv_plan
        outputs = self._run(kind, plans, tensors, execute, kv.stores, kv.rotated)
        current.cache.publish_plan(shadow)
        sampler._forward_counts[
            "source_first" if resume_packs is None else "cross_chunk"
        ] += 1
        if len(plans) > 1:
            sampler._forward_counts["prefix"] += 1
        return outputs[0], None

    @staticmethod
    def _schedules(sampler, current):
        method = getattr(sampler, "schedules_for", None)
        return (
            method(current)
            if method is not None
            else (sampler.video_scheduler, sampler.audio_scheduler)
        )

    def chunk(self, sampler, current, first, next_state=None, *, defer_final=False):
        if defer_final and next_state is not None:
            raise ValueError(
                "Deferred final and ready lookahead are mutually exclusive"
            )

        def prepare():
            cache, pos = current.cache, current.positive
            shadow = cache.planning_copy()
            vs, _ = self._schedules(sampler, current)
            source, target, _, _ = current.spans[0]
            steps = len(vs.timesteps)
            # Draw only the real valid shape and preserve the scheduler's global
            # video/audio order. Graph warmup/capture reuse these inputs unchanged.
            tensors = {"latent": current.visual, "first": first[0]}
            old_chunks = tuple(shadow._stored_chunks())
            old_offsets = {id(c.positions): c.offset for c in old_chunks}
            shadow.complete_step0_and_evict(target)
            kept_chunks = tuple(shadow._stored_chunks())
            # Device rows are moved eagerly before the graph runs, so the graph
            # never carries a capacity-sized compaction.
            spans = tuple(
                (old_offsets[id(c.positions)], c.positions.shape[0])
                for c in kept_chunks
            )
            plans = []
            for step in range(1, steps):
                packed = sampler._select_block(
                    pos,
                    target,
                    phase="target",
                    video_t=sampler._host_list(vs.timesteps)[step],
                    audio_t=None,
                )
                plans.append(
                    self._forward_plan(
                        sampler,
                        shadow,
                        packed,
                        "active",
                        after=plans[-1].kv_plan if plans else None,
                    )
                )
            next_rows = None
            if next_state is not None:
                next_source, next_target, _, _ = next_state.spans[0]
                next_vs, _ = self._schedules(sampler, next_state)
                packs = [
                    sampler._select_block(
                        pos, target, phase="target", video_t=1, audio_t=None, final=True
                    ),
                    sampler._select_block(
                        next_state.positive,
                        next_source,
                        phase="source",
                        video_t=1,
                        audio_t=None,
                    ),
                    sampler._select_block(
                        next_state.positive,
                        next_target,
                        phase="target",
                        video_t=sampler._host_list(next_vs.timesteps)[0],
                        audio_t=None,
                    ),
                ]
                tensors["next_source_x"], tensors["next_latent"] = (
                    packs[1]["x"],
                    next_state.visual,
                )
                packed = self._joint_packed(packs, ("final", "source", "active"))
                next_rows = packs[-1]["x"].shape[0]
                plans.append(
                    self._forward_plan(
                        sampler,
                        shadow,
                        packed,
                        "joint",
                        after=plans[-1].kv_plan if plans else None,
                    )
                )
            elif not defer_final:
                packed = sampler._select_block(
                    pos, target, phase="target", video_t=1, audio_t=None, final=True
                )
                plans.append(
                    self._forward_plan(
                        sampler,
                        shadow,
                        packed,
                        "final",
                        after=plans[-1].kv_plan if plans else None,
                    )
                )
            self._attach(plans, tensors)
            for step in range(steps - 1):
                tensors[f"noise.{step}"] = torch.randn_like(current.visual)
            for step in range(steps):
                tensors[f"sigma.{step}"] = vs.sigmas[step]
            stores = plans[0].kv_plan.stores if plans else ()
            rotated = plans[0].kv_plan.rotated if plans else ()
            latent_shape = current.visual.shape[2:]

            def schedule(prediction, video, step, static):
                compute = torch.promote_types(
                    torch.float32, torch.promote_types(video.dtype, prediction.dtype)
                )
                x0 = (
                    video.to(compute)
                    + static[f"sigma.{step}"].to(compute) * prediction.to(compute)
                ).to(video.dtype)
                if step + 1 == steps:
                    return x0
                sigma = static[f"sigma.{step + 1}"].to(
                    torch.promote_types(torch.float32, video.dtype)
                )
                return ((1 - sigma) * x0 + sigma * static[f"noise.{step}"]).to(
                    video.dtype
                )

            def execute(static):
                video = schedule(static["first"], static["latent"], 0, static)
                for index, plan in enumerate(plans if defer_final else plans[:-1]):
                    x = sampler.processor.patchify_visual(video).to(
                        static[f"f{index}.x"]
                    )
                    rows, _ = self._execute(plan, index, static, x)
                    prediction = sampler.processor.unpatchify_visual(
                        rows, 1, latent_shape
                    )
                    video = schedule(prediction, video, index + 1, static)
                if defer_final:
                    return (video,)
                index = len(plans) - 1
                x = sampler.processor.patchify_visual(video).to(static[f"f{index}.x"])
                if next_state is not None:
                    x = torch.cat(
                        (
                            x,
                            static["next_source_x"],
                            sampler.processor.patchify_visual(static["next_latent"]).to(
                                x
                            ),
                        )
                    )
                rows, _ = self._execute(plans[-1], index, static, x)
                if next_state is None:
                    return (video,)
                prediction = sampler.processor.unpatchify_visual(
                    rows[-next_rows:], 1, next_state.visual.shape[2:]
                )
                return video, prediction

            return plans, tensors, execute, stores, rotated, shadow, steps, spans

        plans, tensors, execute, stores, rotated, shadow, steps, spans = (
            self._prepare_together(prepare)
        )
        cache, pos = current.cache, current.positive
        target = current.spans[0][1]
        kind = (
            "deferred"
            if defer_final
            else "steady"
            if next_state is not None
            else "finish"
        )
        self._discard_rotated(cache, shadow)
        # Step zero reads no K/V, so moving the surviving rows first is
        # equivalent to compacting between step zero and the first forward.
        cache.move_store_rows(spans)
        outputs = self._run(kind, plans, tensors, execute, stores, rotated)
        cache.publish_plan(shadow)
        current.visual.copy_(outputs[0])
        sampler._set_block_latents(pos, target, current.visual, None)
        sampler._forward_counts["denoise"] += steps - 1
        if not defer_final:
            sampler._forward_counts[
                "cross_chunk" if next_state is not None else "commit"
            ] += 1
        return outputs[0], None if next_state is None else (outputs[1], None)
