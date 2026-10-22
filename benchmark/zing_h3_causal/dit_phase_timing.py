# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only, synchronized DiT forward measurements and lossless summaries.

No hooks are installed in a serving worker. Forward timing deliberately isolates
calls; block baselines run separately without those synchronization points.
"""

import math
import statistics
import time
from collections import Counter, defaultdict
from types import MethodType


def distribution(values):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Timing samples must be nonempty, finite and nonnegative")
    return dict(
        count=len(values),
        median=statistics.median(values),
        min=min(values),
        max=max(values),
    )


def check_forward_rows(rows, spans, steps, counts):
    expected = {
        "clean_commit" if k == "commit" else k: v for k, v in counts.items() if v
    }
    if Counter(r["kind"] for r in rows) != expected:
        raise ValueError("Forward timing coverage differs from production counters")
    for chunk, span in enumerate(spans):
        current = [r for r in rows if r["chunk"] == chunk]
        if any(r["latent_frames"] != span["frames"] for r in current):
            raise ValueError("Timing phase differs from production chunk geometry")
        if sorted(r["step"] for r in current if r["kind"] == "denoise") != list(
            range(steps)
        ):
            raise ValueError("Missing or duplicated denoise step")
        if sum(r["kind"] == "clean_commit" for r in current) != 1:
            raise ValueError("Missing or duplicated clean commit")


def summarize(rows):
    """Never mix history, denoise step, startup, tail, or measurement modes."""
    groups = defaultdict(list)
    fields = (
        "mode",
        "kind",
        "chunk",
        "latent_frames",
        "step",
        "q_rows",
        "prefix_rows",
        "history_rows",
        "kv_rows",
    )
    for row in rows:
        if row["run_kind"] == "measured":
            groups[tuple(row[k] for k in fields)].append(row)
    result = []
    for key, samples in groups.items():
        result.append(
            dict(
                zip(fields, key),
                wall_ms=distribution([r["wall_ms"] for r in samples]),
                cuda_ms=distribution([r["cuda_ms"] for r in samples]),
            )
        )
    # Pair by run AND consecutive chunk identity; sum per run before reducing.
    # Chunk wall baselines include source + all steps + commit, without forward
    # timing synchronization. Startup 1,2,2 and any partial tail are excluded.
    blocks = {
        (r["run"], r["chunk"]): r
        for r in rows
        if r["run_kind"] == "measured" and r["mode"] == "block" and r["chunk"] >= 3
    }
    cycles = defaultdict(list)
    for (run, chunk), first in blocks.items():
        second = blocks.get((run, chunk + 1))
        if first["latent_frames"] == 3 and second and second["latent_frames"] == 2:
            cycles[chunk].append(
                dict(
                    run=run,
                    chunks=[chunk, chunk + 1],
                    wall_ms=first["wall_ms"] + second["wall_ms"],
                    cuda_ms=first["cuda_ms"] + second["cuda_ms"],
                )
            )
    return dict(
        groups=result,
        cycles=[
            dict(
                chunks=[chunk, chunk + 1],
                samples=samples,
                wall_ms=distribution([r["wall_ms"] for r in samples]),
                cuda_ms=distribution([r["cuda_ms"] for r in samples]),
            )
            for chunk, samples in sorted(cycles.items())
        ],
    )


class PhaseTimer:
    def __init__(self, sampler, *, mode, run, run_kind):
        if mode not in ("block", "forward"):
            raise ValueError("mode must be block or forward")
        self.sampler, self.mode = sampler, mode
        self.run, self.run_kind = run, run_kind
        self.rows = []
        self.chunk, self.frames, self.step = -1, 0, 0

    def measure(self, call, **fields):
        import torch

        # SP1 only. Synchronize at the measured call boundary, never per layer.
        torch.cuda.synchronize()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        started = time.perf_counter_ns()
        start.record()
        value = call()
        end.record()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - started) / 1e6
        self.rows.append(
            dict(
                run=self.run,
                run_kind=self.run_kind,
                mode=self.mode,
                chunk=self.chunk,
                latent_frames=self.frames,
                wall_ms=elapsed,
                cuda_ms=start.elapsed_time(end),
                **fields,
            )
        )
        return value

    def __enter__(self):
        sampler = self.sampler
        self.original_forward, self.original_step = sampler._forward, sampler.step

        def step(_sampler, state, **kwargs):
            self.chunk, self.step = state.index, 0
            self.frames = state.spans[state.index][3] - state.spans[state.index][2]
            call = lambda: self.original_step(state, **kwargs)
            if self.mode == "forward":
                return call()
            return self.measure(
                call,
                kind="block",
                step=None,
                q_rows=None,
                prefix_rows=None,
                history_rows=None,
                kv_rows=None,
            )

        def forward(_sampler, packed, cache, update):
            call = lambda: self.original_forward(packed, cache, update)
            if self.mode == "block":
                return call()
            kind = {"active": "denoise", "final": "clean_commit"}.get(update, update)
            chunks = [] if cache is None else cache._history()
            prefix = sum(c.positions.shape[0] for c in chunks if c.block == 0)
            history = sum(c.positions.shape[0] for c in chunks if c.block != 0)
            q_rows = packed["position_ids"].shape[0]
            index = self.step if kind == "denoise" else None
            value = self.measure(
                call,
                kind=kind,
                step=index,
                q_rows=q_rows,
                prefix_rows=prefix,
                history_rows=history,
                kv_rows=q_rows + (0 if update == "prefix" else prefix + history),
            )
            if kind == "denoise":
                self.step += 1
            return value

        sampler._forward = MethodType(forward, sampler)
        sampler.step = MethodType(step, sampler)
        return self

    def __exit__(self, *exc):
        self.sampler._forward, self.sampler.step = (
            self.original_forward,
            self.original_step,
        )
