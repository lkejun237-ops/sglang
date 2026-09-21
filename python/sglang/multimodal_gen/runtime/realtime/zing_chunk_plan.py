# SPDX-License-Identifier: Apache-2.0
"""Checkpoint chunk boundaries, independent of codec clips and wire batches."""

from dataclasses import dataclass

from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
    PackedRoleId,
    TemporalChunkId,
)


@dataclass(frozen=True)
class TemporalChunk:
    chunk_id: TemporalChunkId
    start: int
    end: int

    @property
    def source_block(self):
        return PackedRoleId(2 * self.chunk_id + 1)

    @property
    def target_block(self):
        return PackedRoleId(2 * self.chunk_id + 2)


@dataclass(frozen=True, init=False)
class ChunkPlan:
    """Repeat native num_frame_per_block from cycle entry zero.

    Ranges are absolute, half-open *latent* ranges internal to the worker. A
    source/target/audio pair is one temporal chunk, not two packed role blocks.
    split_first_latent splits only the original first chunk; it does not shift
    subsequent cycle boundaries. No attention-window or solver policy lives here.
    """

    lengths: tuple[int, ...]
    split_first_latent: bool

    def __init__(self, num_frame_per_block=5, split_first_latent=False):
        lengths = (
            (num_frame_per_block,)
            if type(num_frame_per_block) is int
            else tuple(num_frame_per_block)
            if isinstance(num_frame_per_block, (list, tuple))
            else ()
        )
        if not lengths or any(type(n) is not int or n < 1 for n in lengths):
            raise ValueError("Chunk lengths must be positive integers")
        if type(split_first_latent) is not bool:
            raise ValueError("The first-latent split flag must be boolean")
        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "split_first_latent", split_first_latent)

    def chunk(self, chunk_id, total_frames=None):
        if type(chunk_id) is not int or chunk_id < 0:
            raise ValueError("Chunk IDs must be nonnegative integers")
        if total_frames is not None and (
            type(total_frames) is not int or total_frames < 0
        ):
            raise ValueError("Latent frame count must be a nonnegative integer")
        split = self.split_first_latent and self.lengths[0] > 1
        if split and chunk_id < 2:
            start, end = (0, 1) if chunk_id == 0 else (1, self.lengths[0])
        else:
            index = chunk_id - int(split)
            cycles, offset = divmod(index, len(self.lengths))
            start = cycles * sum(self.lengths) + sum(self.lengths[:offset])
            end = start + self.lengths[offset]
        if total_frames is not None:
            if start >= total_frames:
                return None
            end = min(end, total_frames)
        return TemporalChunk(TemporalChunkId(chunk_id), start, end)

    def chunks(self, total_frames):
        if total_frames is None:
            raise ValueError("Enumerating chunks requires a finite latent frame count")
        index = 0
        while (chunk := self.chunk(index, total_frames)) is not None:
            yield chunk
            index += 1

    def count(self, total_frames):
        # This finite path shares exactly the EOS clipping rule used by packing.
        return sum(1 for _ in self.chunks(total_frames))
