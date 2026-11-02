# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang.multimodal_gen.runtime.realtime.polling_task_queue import PollingTaskQueue
from sglang.multimodal_gen.runtime.realtime.zing_encoder_input import (
    EncoderInputArranger,
)


def packet(start, count, seq):
    return dict(
        event_id=seq + 1,
        message_bytes=count * 3 + 100,
        payload=dict(
            client_seq=seq,
            source_frame_start=start,
            num_frames=count,
            pts_us=start * 1_000_000 // 24,
            data=b"".join(bytes([i]) * 3 for i in range(start, start + count)),
        ),
    )


@pytest.mark.parametrize("pattern", [(1,), (17,), (8,), (3, 2, 11, 1), (17, 2, 8, 1)])
@pytest.mark.parametrize("frames", [1, 2, 8, 9, 10, 16, 17, 18, 25, 34, 35, 120])
def test_groups_preserve_every_frame_and_packet_lifetime(pattern, frames):
    arranger = EncoderInputArranger(frame_bytes=3, fps=24)
    queue = PollingTaskQueue(lambda q: None, arrange=arranger)
    start = seq = 0
    packet_ends = {}
    while start < frames:
        count = min(pattern[seq % len(pattern)], frames - start)
        queue.append(packet(start, count, seq))
        packet_ends[seq] = start + count
        start += count
        seq += 1
    # Before EOS only full groups are visible to the model.
    assert all(t["num_frames"] == (1, 8, 8)[i % 3] for i, t in enumerate(queue))
    assert arranger.buffered_frames < (1, 8, 8)[len(queue) % 3]
    eos = dict(kind="source.end", end=dict(frame_end=frames, event_id=seq + 1))
    queue.append(eos)
    assert queue[-1] is eos
    cursor, actual, released = 0, [], []
    for i, task in enumerate(list(queue)[:-1]):
        assert 0 < task["num_frames"] <= (1, 8, 8)[i % 3]
        assert (
            sum(x["payload"]["num_frames"] for x in task["appends"])
            == task["num_frames"]
        )
        before = cursor
        for fragment in task["appends"]:
            p = fragment["payload"]
            assert p["source_frame_start"] == cursor
            assert p["pts_us"] == cursor * 1_000_000 // 24
            assert fragment["event_id"] == p["client_seq"] + 1
            actual.append(p["data"])
            cursor += p["num_frames"]
        assert task["release_client_seqs"] == [
            seq for seq, end in packet_ends.items() if before < end <= cursor
        ]
        released.extend(task["release_client_seqs"])
    assert cursor == frames
    assert b"".join(actual) == packet(0, frames, 0)["payload"]["data"]
    assert released == list(packet_ends)
    assert not arranger.pending and arranger.buffered_frames == 0
    with pytest.raises(ValueError, match="ended"):
        queue.append(packet(frames, 1, seq))


def test_partial_input_is_not_runnable_and_eos_flushes_without_padding():
    arranger = EncoderInputArranger(frame_bytes=3, fps=24)
    queue = PollingTaskQueue(lambda q: None, arrange=arranger)
    queue.append(packet(0, 3, 0))
    first = queue.popleft()
    assert first["num_frames"] == 1
    assert first["release_client_seqs"] == []
    assert not queue and arranger.buffered_frames == 2
    queue.append(packet(3, 5, 1))
    assert not queue and arranger.buffered_frames == 7
    queue.append(dict(kind="source.end", end=dict(frame_end=8, event_id=3)))
    assert queue[0]["num_frames"] == 7
    assert queue[0]["release_client_seqs"] == [0, 1]
    assert queue[1]["kind"] == "source.end"


def test_clear_drops_staging_and_restarts_pattern():
    arranger = EncoderInputArranger(frame_bytes=3, fps=24)
    queue = PollingTaskQueue(lambda q: None, arrange=arranger)
    queue.append(packet(0, 8, 0))
    assert arranger.buffered_frames == 7
    queue.clear()
    arranger.clear()
    assert not arranger.pending and arranger.buffered_frames == 0
    queue.append(packet(0, 17, 0))
    assert [x["num_frames"] for x in queue] == [1, 8, 8]
