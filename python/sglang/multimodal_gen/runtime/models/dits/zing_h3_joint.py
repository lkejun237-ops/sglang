# SPDX-License-Identifier: Apache-2.0
"""Validated mixed clean/active transactions for opt-in joint Zing forwards."""

import copy
from dataclasses import dataclass
from types import SimpleNamespace

import torch


class ZingJointUnsupported(ValueError):
    """A valid sequential request needs separate branch attention views."""


@dataclass(frozen=True)
class JointBranch:
    phase: str
    start: int
    stop: int
    frames: int


class ZingJointTransaction:
    def __init__(self, cache, phases, positions, blocks, roles, t, clean):
        if tuple(phases) not in (("source", "active"), ("final", "source", "active")):
            raise ValueError("Invalid Zing joint transaction phases")
        if not cache.raw_store or cache.local_frames != -1:
            raise ValueError(
                "Zing joint transactions require a raw, chunk-window cache"
            )
        unique, counts = torch.unique_consecutive(blocks, return_counts=True)
        if len(counts) != len(phases):
            raise ValueError(
                "Zing joint branches must contain distinct consecutive blocks"
            )
        shadow = copy.copy(cache)
        self.branches = []
        start = 0
        for phase, count in zip(phases, counts.tolist()):
            stop = start + count
            r, p, b = roles[start:stop], positions[start:stop], blocks[start:stop]
            role = 7 if phase == "source" else 10
            if not bool((r == role).all()):
                raise ValueError(
                    "Zing joint transactions currently require video-only branches"
                )
            frames = int(p[:, 0].unique().numel())
            block = shadow.validate(p, b, phase, frames)
            if phase != "active" and not bool((t[start:stop] == 1).all()):
                raise ValueError("Only clean t=1 states may enter the Zing KV cache")
            if phase == "source":
                shadow.pending_source = SimpleNamespace(block=block)
            elif phase == "final":
                next_id = block // 2
                ids = [c.temporal_chunk_id for c in cache.committed] + [next_id - 1]
                if cache.window_eviction_timing == "before_source" and not all(
                    cache.window_policy.retains(i, next_id) for i in ids
                ):
                    raise ValueError(
                        "Zing cross-chunk transaction requires window eviction"
                    )
                shadow.last_block = block
                shadow.active = shadow.pending_source = None
            self.branches.append(JointBranch(phase, start, stop, frames))
            start = stop
        if clean != self.branches[-1].start:
            raise ValueError("Zing joint clean rows must precede the active target")
        # A union map is valid only when every branch sees the same coordinates
        # as at its sequential execution point. Validate before any layer runs.
        history = (
            cache.host_positions()
            if all(c.host_positions is not None for c in cache._history())
            and positions.device.type == "cpu"
            else cache.positions()
        )
        current, mapped_history = cache._map_attention_positions(
            positions, False, history
        )
        for branch in self.branches:
            prior = positions[: branch.start]
            h = prior if history is None else torch.cat((history, prior))
            h = None if not h.numel() else h
            cp, hp = cache._map_attention_positions(
                positions[branch.start : branch.stop], False, h
            )
            expected_h = (
                current[: branch.start]
                if mapped_history is None
                else torch.cat((mapped_history, current[: branch.start]))
            )
            if not torch.equal(cp, current[branch.start : branch.stop]) or (
                hp is not None and not torch.equal(hp, expected_h)
            ):
                raise ZingJointUnsupported("different_rope_views")

    @property
    def ends(self):
        return tuple(b.stop for b in self.branches)

    def commit(self, cache, positions, blocks, layers, validation=None):
        try:
            for branch in self.branches:
                selection = slice(branch.start, branch.stop)
                if branch.phase == "source" and self.branches[0].phase == "final":
                    cache.begin_chunk(
                        int(
                            (blocks if validation is None else validation[1])[
                                selection
                            ][0]
                        )
                        + 1
                    )
                cache.update(
                    positions[selection],
                    blocks[selection],
                    branch.phase,
                    branch.frames,
                    None
                    if branch.phase == "active"
                    else [(k[selection], v[selection]) for k, v in layers],
                    validation=None
                    if validation is None
                    else tuple(v[selection] for v in validation),
                )
        except BaseException:
            # Existing request failure semantics: release all session state;
            # a partially advanced logical transaction is never reusable.
            cache.clear()
            raise
