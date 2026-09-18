# SPDX-License-Identifier: Apache-2.0
"""Request-owned, pre-RoPE KV transactions for Zing causal generation."""

from dataclasses import dataclass

import torch


@dataclass
class ZingKVChunk:
    positions: torch.Tensor
    block: int
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    frames: int = 0


class ZingH3KVCache:
    """Keep the prefix and complete source/target transactions.

    Active denoising forwards never persist K/V. Positions remain absolute and
    keys remain normalized but unrotated. A failed request must call clear().
    Window limits count target latent frames, never individual KV rows.
    """

    def __init__(self, num_layers: int, local_frames: int = -1, sink_frames: int = 0):
        if num_layers < 1 or local_frames < -1 or sink_frames < 0:
            raise ValueError("Invalid Zing KV layer count or frame limits")
        self.num_layers = num_layers
        self.local_frames = local_frames
        self.sink_frames = sink_frames
        self.clear()

    def clear(self):
        self.prefix = None
        self.pending_source = None
        self.committed = []
        self.active = None
        self.last_block = 0

    def _history(self):
        chunks = self.committed
        if self.local_frames >= 0:
            first, last, count = 0, len(chunks), 0
            while first < len(chunks) and count < self.sink_frames:
                count += chunks[first].frames
                first += 1
            count = 0
            while last > first and count < self.local_frames:
                last -= 1
                count += chunks[last].frames
            chunks = chunks[:first] + chunks[last:]
        return (
            ([self.prefix] if self.prefix is not None else [])
            + chunks
            + ([self.pending_source] if self.pending_source is not None else [])
        )

    def positions(self):
        chunks = self._history()
        return torch.cat([c.positions for c in chunks]) if chunks else None

    def history(self, layer):
        chunks = self._history()
        if not chunks:
            return None
        return tuple(
            torch.cat([c.layers[layer][i] for c in chunks], dim=0) for i in (0, 1)
        )

    def validate(self, positions, block_idx, phase, frames):
        if torch.is_grad_enabled():
            raise RuntimeError("Zing KV transactions require torch.no_grad()")
        if phase not in {"prefix", "source", "active", "final"}:
            raise ValueError("Invalid Zing KV transaction phase")
        blocks = torch.unique(block_idx)
        if blocks.numel() != 1 or positions.shape != (block_idx.numel(), 3):
            raise ValueError("A Zing KV transaction must contain one nonempty block")
        block = int(blocks.item())
        if phase == "prefix":
            if (
                block != 0
                or frames != 0
                or self.prefix is not None
                or self.last_block
                or self.active is not None
                or self.pending_source is not None
            ):
                raise ValueError(
                    "Zing prefix must be the first transaction, with block zero"
                )
        elif frames < 1:
            raise ValueError(
                "Zing source and target transactions require visual frames"
            )
        elif phase == "source":
            if (
                block != self.last_block + 1
                or block % 2 != 1
                or self.pending_source is not None
                or self.active is not None
            ):
                raise ValueError("Zing source must precede its target exactly once")
        else:
            if block != self.last_block + 2 or block % 2:
                raise ValueError("Zing target blocks must commit in consecutive order")
            if (
                self.pending_source is not None
                and self.pending_source.block + 1 != block
            ):
                raise ValueError("Zing target does not match its pending source")
            if self.active is not None and (
                self.active[0] != block or not torch.equal(self.active[1], positions)
            ):
                raise ValueError("Zing active target geometry changed before commit")
            if phase == "final" and self.active is None:
                raise ValueError("Zing final commit requires an active target")
        return block

    def update(self, positions, block_idx, phase, frames, layers=None):
        block = self.validate(positions, block_idx, phase, frames)
        if phase == "active":
            if layers is not None:
                raise ValueError("Active Zing K/V must not be persisted")
            self.active = (block, positions.detach().clone())
            return
        if layers is None or len(layers) != self.num_layers:
            raise ValueError("Zing KV commit requires every transformer layer")
        for k, v in layers:
            if (
                k.ndim != 3
                or k.shape != v.shape
                or k.shape[0] != positions.shape[0]
                or k.dtype != v.dtype
                or k.device != v.device
                or k.requires_grad
                or v.requires_grad
            ):
                raise ValueError(
                    "Zing K/V must be matching detached [rows, heads, dim] tensors"
                )
        chunk = ZingKVChunk(positions.detach().clone(), block, tuple(layers), frames)
        if phase == "prefix":
            self.prefix = chunk
        elif phase == "source":
            self.pending_source = chunk
        else:
            if self.pending_source is not None:
                source = self.pending_source
                chunk = ZingKVChunk(
                    torch.cat((source.positions, chunk.positions)),
                    block,
                    tuple(
                        tuple(torch.cat((a, b)) for a, b in zip(s, t))
                        for s, t in zip(source.layers, chunk.layers)
                    ),
                    frames,
                )
            self.committed.append(chunk)
            self.last_block = block
            self.pending_source = self.active = None
