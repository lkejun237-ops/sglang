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


PREFIX_PHASES = ("prefix", "history", "history", "history", "source", "active")


class PrefixJointTransaction:
    def __init__(self, cache, phases, positions, blocks, roles, t, clean):
        if (
            tuple(phases) != PREFIX_PHASES
            or not cache.raw_store
            or cache.local_frames != -1
        ):
            raise ValueError("Prompt joint requires the six-branch raw cache contract")
        if (
            cache.prefix is not None
            or cache.committed
            or cache.active is not None
            or cache.pending_source is not None
        ):
            raise ValueError("Prompt joint requires a cleared cache")
        ids, counts = torch.unique_consecutive(blocks, return_counts=True)
        if ids.tolist() != [0, 2, 4, 6, 7, 8]:
            raise ValueError("Expected prefix, history 1+2+2, source7 and target8")
        self.branches = []
        start = 0
        for phase, count in zip(phases, counts.tolist()):
            stop = start + count
            r = roles[start:stop]
            p = positions[start:stop]
            frames = 0 if phase == "prefix" else int(p[:, 0].unique().numel())
            if phase != "prefix" and not bool(
                (r == (7 if phase == "source" else 10)).all()
            ):
                raise ValueError("Only video branches are supported after the prefix")
            if phase not in ("prefix", "active") and not bool(
                (t[start:stop] == 1).all()
            ):
                raise ValueError("History and source must be clean")
            self.branches.append(JointBranch(phase, start, stop, frames))
            start = stop
        if [b.frames for b in self.branches[1:4]] != [
            1,
            2,
            2,
        ] or clean != self.branches[-1].start:
            raise ValueError("Incorrect clean history or active boundary")
        # Prefix coordinates stay absolute. All media use the same mapping as
        # after the sequential prefix commit. Verify every branch's view before
        # entering a transformer layer; never assume union RoPE is equivalent.
        first = self.branches[0].stop
        shadow = copy.copy(cache)
        shadow.prefix = SimpleNamespace(positions=positions[:first])
        mapped, _ = shadow._map_attention_positions(
            positions, False, history=positions[:0]
        )
        if not torch.equal(mapped[:first], positions[:first]):
            raise ValueError("Joint prefix coordinates changed")
        for b in self.branches[1:]:
            cp, hp = shadow._map_attention_positions(
                positions[b.start : b.stop], False, positions[: b.start]
            )
            if not torch.equal(cp, mapped[b.start : b.stop]) or not torch.equal(
                hp, mapped[: b.start]
            ):
                raise ValueError("Sequential and joint RoPE views differ")
        self.mapped_positions = mapped

    @property
    def ends(self):
        return tuple(b.stop for b in self.branches)

    def commit(self, cache, positions, blocks, layers, validation=None):
        try:
            for b in self.branches:
                s = slice(b.start, b.stop)
                check = None if validation is None else tuple(v[s] for v in validation)
                phase = b.phase
                if phase == "source":
                    cache.begin_chunk(8)
                if phase == "history":
                    cache.update(
                        positions[s], blocks[s], "active", b.frames, validation=check
                    )
                    phase = "final"
                cache.update(
                    positions[s],
                    blocks[s],
                    phase,
                    b.frames,
                    None if phase == "active" else [(k[s], v[s]) for k, v in layers],
                    validation=check,
                )
        except BaseException:
            cache.clear()
            raise


def joint_transaction(cache, phases, positions, blocks, roles, t, clean):
    transaction = (
        PrefixJointTransaction
        if tuple(phases) == PREFIX_PHASES
        else ZingJointTransaction
    )
    return transaction(cache, phases, positions, blocks, roles, t, clean)


def prefix_modulation(t, roles, ends, tags, num_modalities):
    """Host-only grouping that preserves each serial timestep GEMM shape."""
    groups, offsets, indices, local_times = [], {}, [], []
    offset, start = 0, 0
    for branch, stop in enumerate(ends):
        times, local = t[start:stop].unique(sorted=True, return_inverse=True)
        if branch and times.numel() != 1:
            raise ValueError("Prompt joint media branches require one timestep")
        key = tuple(times.tolist())
        if key not in offsets:
            offsets[key] = offset
            offset += times.numel()
            groups.append(times)
        r = roles[start:stop]
        if bool(((r == 5) | (r == 11)).any()):
            raise ValueError("Prompt joint supports video only")
        modality = torch.full_like(r, tags["video"])
        modality[r == 0] = tags["text"]
        modality[r == 1] = tags["condition_visual"]
        modality[r == 7] = tags["source_visual"]
        local = local + offsets[key]
        local_times.append(local)
        indices.append(local * num_modalities + modality)
        start = stop
    visual = (roles == 4) | (roles == 7) | (roles == 10)
    return (
        torch.cat(groups),
        tuple(len(g) for g in groups),
        torch.cat(indices),
        torch.cat(local_times),
        tuple(int(visual[:end].sum()) for end in ends),
    )
