# SPDX-License-Identifier: Apache-2.0
"""Protocol accounting only; these tests do not establish GPU acceptance."""

import pytest

from sglang.multimodal_gen.runtime.realtime.zing_contracts import PixelRange
from sglang.multimodal_gen.runtime.realtime.zing_source_stream import (
    SourceLedger,
    SourceLimits,
)


def ledger(**kwargs):
    return SourceLedger(
        width=2,
        height=2,
        fps=24,
        limits=SourceLimits(17, 4096, 34, 34 * 12, 68),
        **kwargs,
    )


def append(stream, count, **changes):
    payload = dict(
        client_seq=stream.sequence,
        source_frame_start=stream.accepted_frames,
        num_frames=count,
        pts_us=stream.accepted_frames * 1_000_000 // 24,
        data=bytes(count * 12),
    )
    payload.update(changes)
    return stream.accept(payload, event_id=stream.event_id + 1, message_bytes=512)


def test_credit_alignment_and_partial_append_ownership():
    s = ledger()
    a, b = append(s, 17), append(s, 13)
    assert a.message()["source_id"] == 1 and b.message()["client_seq"] == 1
    assert s.consume(16) is None
    assert s.consume(20) == {
        "type": "source.consumed",
        "source_frame_end": 17,
        "source_bytes_end": 204,
    }
    p = s.provenance_for(PixelRange(15, 19), 30)
    assert [(v.source_id, v.num_frames) for v in p.source_alignment] == [(1, 2), (2, 2)]
    assert p.source_dependency_end == 30
    s.delivered(PixelRange(0, 19))
    assert len(s.records) == 1
    s.finish(30, event_id=3)
    with pytest.raises(ValueError):
        s.completed()
    s.consume(30)
    s.delivered(PixelRange(19, 30))
    assert not s.records
    assert s.completed() == {"type": "session.completed", "output_frames": 30}


@pytest.mark.parametrize(
    "changes",
    [
        {"client_seq": 1},
        {"client_seq": False},
        {"source_frame_start": 1},
        {"num_frames": True},
        {"pts_us": 1},
        {"data": b"bad"},
    ],
)
def test_invalid_append_does_not_advance(changes):
    s = ledger()
    with pytest.raises(ValueError):
        append(s, 1, **changes)
    assert s.accepted_frames == s.accepted_bytes == s.event_id == s.sequence == 0
    assert not s.records


def test_credit_limits_are_independent():
    s = ledger()
    append(s, 17)
    append(s, 17)
    with pytest.raises(ValueError, match="credit"):
        append(s, 1)
    s.consume(17)
    append(s, 1)
    s = ledger()
    s.limits = SourceLimits(17, 4096, 34, 12, 68)
    append(s, 1)
    with pytest.raises(ValueError, match="credit"):
        append(s, 1)


def test_finite_end_cancel_and_session_isolation():
    s, other = ledger(total_frames=3), ledger()
    append(s, 2)
    with pytest.raises(ValueError):
        s.finish(2, event_id=2)
    with pytest.raises(ValueError):
        append(s, 2)
    append(s, 1)
    s.finish(3, event_id=3)
    with pytest.raises(ValueError):
        append(s, 1)
    s.dispose()
    with pytest.raises(ValueError):
        s.completed()
    assert append(other, 1).source_id == 1


def test_long_stream_retires_metadata_after_both_owners_release():
    s = ledger()
    for i in range(5000):
        append(s, 1)
        s.delivered(PixelRange(i, i + 1))
        assert len(s.records) == 1
        s.consume(i + 1)
        assert not s.records
    assert s.sequence == 5000


def test_dependency_is_computation_watermark_not_latest_accepted():
    s = ledger()
    append(s, 17)
    append(s, 17)
    assert s.provenance_for(PixelRange(0, 1), 17).source_dependency_end == 17
    with pytest.raises(ValueError):
        s.provenance_for(PixelRange(0, 1), 35)
    with pytest.raises(ValueError):
        s.provenance_for(PixelRange(0, 2), 1)
