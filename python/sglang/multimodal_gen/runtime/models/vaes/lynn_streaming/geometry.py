# SPDX-License-Identifier: Apache-2.0
"""Tile layout of the Lynn streaming decoder (RGB pixel units)."""

import math


def split_tiles(length, *, size, overlap, ratio):
    """Return (starts, sizes, overlaps) covering `length` pixels with latent-aligned overlaps."""
    if size >= length:
        return [0], [length], []
    count = math.ceil(length / size)
    while size * count - overlap * (count - 1) < length:
        count += 1
    overlaps = [overlap] * (count - 1)
    for i in range((size * count - sum(overlaps) - length) // ratio):
        overlaps[i % (count - 1)] += ratio
    starts = [0]
    for value in overlaps:
        starts.append(starts[-1] + size - value)
    return starts, [size] * count, overlaps


def tile_shard(count, rank, world_size):
    """Return the [start, end) tile range of `rank`: contiguous, remainder to leading ranks."""
    base, extra = divmod(count, world_size)
    start = rank * base + min(rank, extra)
    return start, start + base + (rank < extra)
