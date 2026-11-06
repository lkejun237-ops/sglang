# SPDX-License-Identifier: Apache-2.0
"""Temporal-chunk attention policy, separate from checkpoint latent chunk sizes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ZingWindowPolicy:
    # None preserves the checkpoint full-history baseline.
    window_chunks: int | None = None
    sink_chunks: int = 0
    sink_start: int = 0
    min_chunk_id: int = 0

    def __post_init__(self):
        if any(
            type(v) is not int or v < 0 for v in (self.sink_start, self.min_chunk_id)
        ):
            raise ValueError("Zing sink positions must be nonnegative integers")
        if type(self.sink_chunks) is not int or self.sink_chunks < 0:
            raise ValueError("Zing sink_chunks must be a nonnegative integer")
        if self.window_chunks is None:
            if self.sink_chunks:
                raise ValueError("Zing sink_chunks requires a finite window")
        elif (
            type(self.window_chunks) is not int
            or self.window_chunks < 1
            or self.sink_chunks >= self.window_chunks
        ):
            raise ValueError(
                "Zing window_chunks must include sinks and one current chunk"
            )

    @staticmethod
    def validate_rope(mode, max_frame_gap):
        if not isinstance(mode, str) or mode not in {"absolute", "block_relative"}:
            raise ValueError(
                "Zing rope_position_mode must be absolute or block_relative"
            )
        if type(max_frame_gap) is not int or max_frame_gap < 1:
            raise ValueError("Zing rope_max_frame_gap must be a positive integer")

    def retains(self, chunk_id: int, current_id: int) -> bool:
        """Keep fixed initial sinks and recent history; current has its own slot."""
        return self.window_chunks is None or (
            chunk_id >= self.min_chunk_id
            and (
                self.sink_start <= chunk_id < self.sink_start + self.sink_chunks
                or chunk_id >= current_id - (self.window_chunks - self.sink_chunks - 1)
            )
        )
