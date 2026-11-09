"""Source identity is independent of the model's optional RTP presentation clock."""

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.rtc_playout import (
    ModelPresentationClock,
)


def source_pts(index):
    # 17 consecutive 30 FPS captures selected every 850 ms: model throughput 20.
    return (index // 17) * 850_000 + round((index % 17) * 1_000_000 / 30)


def test_latest_source_holes_become_a_stable_model_cadence_without_queueing():
    clock = ModelPresentationClock()
    source = [source_pts(i) for i in range(170)]
    presentation = [clock.observe(pts) for pts in source]
    assert presentation[:17] == source[:17]
    assert all(b > a for a, b in zip(presentation, presentation[1:]))
    assert set(b - a for a, b in zip(presentation[17:], presentation[18:])) == {50_000}
    assert source[17] - source[16] > 300_000
    assert len(clock.intervals) <= 17


def test_ordinary_source_cadence_is_preserved_and_real_pause_resets_estimate():
    clock = ModelPresentationClock()
    pts = [round(i * 1_000_000 / 30) for i in range(30)]
    assert [clock.observe(p) for p in pts] == pts
    before = clock.observe(pts[-1] + 3_000_000)
    assert before - pts[-1] == 3_000_000
    assert not clock.smoothing
    assert clock.observe(pts[-1] + 3_033_333) == before + 33_333
    with pytest.raises(ValueError, match="increase"):
        clock.observe(pts[-1])


def test_new_model_epoch_does_not_inherit_an_old_clock():
    old = ModelPresentationClock()
    for i in range(100):
        old.observe(source_pts(i))
    new = ModelPresentationClock()
    assert new.observe(17) == 17
    assert not new.smoothing and new.count == 1


def test_partial_selection_groups_cannot_accumulate_minutes_of_clock_drift():
    import random

    rng = random.Random(7)
    clock = ModelPresentationClock()
    source = 0
    previous = -1
    for group in range(600):
        for _ in range(rng.choice([1, 7, 8, 9, 17, 25])):
            source += rng.randint(31_000, 35_000)
            pts = clock.observe(source)
            assert pts > previous
            assert abs(pts - source) <= 1_000_000
            previous = pts
        source += rng.randint(90_000, 1_300_000)
    assert len(clock.intervals) <= 17


def test_slow_continuous_source_keeps_its_clock_and_no_estimated_playout_delay():
    clock = ModelPresentationClock()
    pts = [i * 500_000 for i in range(100)]
    assert [clock.observe(p) for p in pts] == pts
    assert not clock.smoothing
