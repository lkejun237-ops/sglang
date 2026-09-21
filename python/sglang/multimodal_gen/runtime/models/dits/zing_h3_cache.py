# SPDX-License-Identifier: Apache-2.0
"""Request-owned, pre-RoPE KV transactions for Zing causal generation."""

from dataclasses import dataclass

import torch

from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy


@dataclass
class ZingKVChunk:
    positions: torch.Tensor
    block: int
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    frames: int = 0

    @property
    def temporal_chunk_id(self):
        return self.block // 2 - 1


class ZingH3KVCache:
    """Keep the prefix and complete source/target transactions.

    Active denoising forwards never persist K/V. Positions remain absolute and
    keys remain normalized but unrotated. Under Ulysses, every rank stores the
    complete cached sequence for only its local attention-head shard. A failed
    request must call clear(). The opt-in temporal window counts whole source /
    target / audio transactions. Legacy local_frames is a selection-only API;
    it is not a storage bound and cannot be combined with the temporal policy.
    """

    def __init__(
        self,
        num_layers: int,
        local_frames: int = -1,
        sink_frames: int = 0,
        *,
        window_policy: ZingWindowPolicy | None = None,
        rope_position_mode: str = "absolute",
        rope_max_frame_gap: int = 12,
    ):
        if num_layers < 1 or local_frames < -1 or sink_frames < 0:
            raise ValueError("Invalid Zing KV layer count or frame limits")
        self.num_layers = num_layers
        self.local_frames = local_frames
        self.sink_frames = sink_frames
        self.window_policy = window_policy or ZingWindowPolicy()
        if self.window_policy.window_chunks is not None and (
            local_frames != -1 or sink_frames
        ):
            raise ValueError(
                "Zing temporal and legacy frame windows cannot be combined"
            )
        ZingWindowPolicy.validate_rope(rope_position_mode, rope_max_frame_gap)
        self.rope_position_mode = rope_position_mode
        self.rope_max_frame_gap = rope_max_frame_gap
        self.clear()

    def clear(self):
        self.prefix = None
        self.pending_source = None
        self.committed = []
        self.active = None
        self.last_block = 0
        self.layer_layouts = None

    def prepare_chunk(self, target_block):
        """Evict history before source execution; the caller closes on failure.

        One committed entry owns source + target/audio. Keep pre-RoPE K and
        absolute checkpoint positions: eviction never renumbers a retained key.
        Future post-RoPE caches must key validity by these exact positions and
        RoPE parameters, and discard entries for removed chunks/prompt prefixes.
        """
        if self.active is not None or self.pending_source is not None:
            raise ValueError("Zing window selection requires a completed boundary")
        if target_block != self.last_block + 2:
            raise ValueError("Zing window selection requires the next target block")
        current_id = target_block // 2 - 1
        self.committed = [
            chunk
            for chunk in self.committed
            if self.window_policy.retains(chunk.temporal_chunk_id, current_id)
        ]

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

    def attention_positions(self, current, *, prefix=False):
        """Rebuild RoPE coordinates only; cached positions/provenance stay absolute.

        Unlike MinWM token order, Zing repeats time across source, target and
        stereo audio. Compress sorted unique media timestamps together, so equal
        timestamps share a coordinate regardless of packed modality/role order.
        Prefix/reference coordinates are unchanged. Gap units are checkpoint
        temporal coordinates (not pixels, latent frames or temporal chunk IDs).
        """
        history = self.positions()
        if self.rope_position_mode == "absolute" or prefix:
            return current, history
        prefix_rows = 0 if self.prefix is None else self.prefix.positions.shape[0]
        joined = current if history is None else torch.cat((history, current))
        mapped = joined.clone()
        media = joined[prefix_rows:, 0]
        times, inverse = torch.unique(media, sorted=True, return_inverse=True)
        if times.numel():
            anchor = (
                joined[:prefix_rows, 0].max() + 1
                if prefix_rows
                else times.new_zeros(())
            )
            gaps = times[1:] - times[:-1]
            compressed = torch.cat(
                (times.new_zeros(1), gaps.clamp(max=self.rope_max_frame_gap))
            ).cumsum(0)
            mapped[prefix_rows:, 0] = anchor + compressed[inverse]
        n = 0 if history is None else history.shape[0]
        return mapped[n:], None if history is None else mapped[:n]

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
        layouts = []
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
            layouts.append((tuple(k.shape[1:]), k.dtype, k.device))
        if self.layer_layouts is None:
            self.layer_layouts = tuple(layouts)
        elif tuple(layouts) != self.layer_layouts:
            raise ValueError("Zing K/V head-shard layout changed within a request")
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
