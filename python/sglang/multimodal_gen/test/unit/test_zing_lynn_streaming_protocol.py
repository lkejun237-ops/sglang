# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.geometry import (
    split_tiles,
)
from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.temporal import (
    TemporalProtocol,
)

H3 = {
    "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],
    "clip_length": 17,
    "token_drop": 3,
}


def test_frames_per_step_and_cumulative_counts_follow_the_five_phase_cycle():
    protocol = TemporalProtocol(H3)
    assert (protocol.ratio, protocol.phases, protocol.pre_padding) == (4, 5, 3)
    steps = [protocol.new_frames(i) for i in range(10)]
    assert steps == [1, 4, 4, 4, 4, 1, 4, 4, 4, 4]
    assert [protocol.frame_count(n) for n in range(1, 8)] == [1, 5, 9, 13, 17, 18, 22]
    assert protocol.frame_count(0) == 0


def test_business_chunking_matches_the_documented_cumulative_frames():
    protocol = TemporalProtocol(H3)
    chunks, total, cumulative = [1, 2, 2, 3, 2], 0, []
    for size in chunks:
        total += size
        cumulative.append(protocol.frame_count(total))
    assert cumulative == [1, 9, 17, 26, 34]


def test_teacher_window_is_seven_latents_for_h3():
    assert TemporalProtocol(H3).teacher_window == 7


@pytest.mark.parametrize(
    "length,starts,overlaps",
    [
        (200, [0], []),
        (256, [0], []),
        (480, [0, 112, 224], [144, 144]),
        (832, [0, 192, 384, 576], [64, 64, 64]),
    ],
)
def test_tile_layout_covers_the_frame_with_latent_aligned_overlap(
    length, starts, overlaps
):
    got_starts, sizes, got_overlaps = split_tiles(
        length, size=256, overlap=64, ratio=16
    )
    assert (got_starts, got_overlaps) == (starts, overlaps)
    assert got_starts[-1] + sizes[-1] >= length


def test_tile_layout_spreads_extra_coverage_in_latent_steps():
    starts, sizes, overlaps = split_tiles(496, size=256, overlap=64, ratio=16)
    assert (starts, overlaps) == ([0, 112, 240], [144, 128])
    assert starts[-1] + sizes[-1] == 496
