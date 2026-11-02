# SPDX-License-Identifier: Apache-2.0
"""RGB source accounting for the version-one Zing realtime contract.

The caller sends accepted before exposing a batch to the worker. It advances
consumed only after a bounded downstream owner takes over, and retires mapping
records only after both credit release and successful output delivery.
"""

from collections import deque
from dataclasses import dataclass

from sglang.multimodal_gen.runtime.realtime.source_transport import (
    PROTOCOL_OVERHEAD_BYTES,
    check,
)
from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
    PixelRange,
    Provenance,
    SourceSpan,
    integer,
    require,
)


def default_source_encoder():
    """Return a fresh trusted profile for the built-in progressive encoder."""
    return {
        "mode": "progressive",
        "contract_version": "h3-clip17-progressive-v1",
        "factory": (
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.zing_h3.progressive_encoder:create_session"
        ),
    }


@dataclass(frozen=True)
class SourceLimits:
    batch_frames: int
    message_bytes: int
    pending_frames: int
    pending_bytes: int
    metadata_batches: int

    def __post_init__(self):
        require(
            all(integer(v, 1) for v in vars(self).values()),
            "Source limits must be positive integers.",
        )


@dataclass(frozen=True)
class AcceptedSource:
    source_id: int
    event_id: int
    client_seq: int
    start: int
    end: int
    bytes_end: int

    def message(self):
        return {
            "type": "source.accepted",
            "event_id": self.event_id,
            "client_seq": self.client_seq,
            "source_id": self.source_id,
            "source_frame_start": self.start,
            "source_frame_end": self.end,
        }


def source_consumed_message(boundaries, consumed_frames, owned_frame_end):
    """Select whole input packets, shared by ingress accounting and Decoder WS."""
    boundary = None
    for end, bytes_end in boundaries:
        if consumed_frames < end <= owned_frame_end:
            boundary = end, bytes_end
    if boundary is None:
        return None
    return {
        "type": "source.consumed",
        "source_frame_end": boundary[0],
        "source_bytes_end": boundary[1],
    }


class SourceLedger:
    def __init__(self, *, width, height, fps, limits, total_frames=None):
        require(
            all(integer(v, 1) for v in (width, height, fps)),
            "Source dimensions and frame rate must be positive integers.",
        )
        require(
            total_frames is None or integer(total_frames, 1),
            "The requested frame count must be positive.",
        )
        self.width, self.height, self.fps = width, height, fps
        self.frame_bytes = width * height * 3
        self.limits, self.total_frames = limits, total_frames
        self.accepted_frames = self.accepted_bytes = self.sequence = 0
        self.consumed_frames = self.consumed_bytes = self.output_frames = 0
        self.event_id = 0
        self.ended = self.closed = False
        self.records = deque()

    def validate_event_id(self, event_id):
        require(
            integer(event_id, 1) and event_id > self.event_id,
            "Please send events in increasing order.",
        )

    def accept(self, payload, *, event_id, message_bytes):
        require(not self.closed and not self.ended, "The source stream has ended.")
        self.validate_event_id(event_id)
        require(
            isinstance(payload, dict)
            and set(payload)
            == {"client_seq", "source_frame_start", "num_frames", "pts_us", "data"},
            "Please provide a source frame batch with the supported fields.",
        )
        seq, start, count, pts = (
            payload.get(key)
            for key in ("client_seq", "source_frame_start", "num_frames", "pts_us")
        )
        data = payload.get("data")
        require(
            integer(seq)
            and seq == self.sequence
            and integer(start)
            and start == self.accepted_frames,
            "Please send consecutive source frame batches.",
        )
        require(
            integer(count, 1) and count <= self.limits.batch_frames,
            "The source frame batch is too large or empty.",
        )
        require(
            integer(pts) and pts == start * 1_000_000 // self.fps,
            "The source timestamp does not match its frame position.",
        )
        require(
            isinstance(data, bytes) and len(data) == count * self.frame_bytes,
            "The source frame data does not match its dimensions.",
        )
        check(
            integer(message_bytes, len(data))
            and message_bytes
            <= min(self.limits.message_bytes, len(data) + PROTOCOL_OVERHEAD_BYTES),
            "source_message_too_large",
            "The source message is too large.",
        )
        end, bytes_end = start + count, self.accepted_bytes + len(data)
        require(
            self.total_frames is None or end <= self.total_frames,
            "The source exceeds the requested frame count.",
        )
        check(
            end - self.consumed_frames <= self.limits.pending_frames
            and bytes_end - self.consumed_bytes <= self.limits.pending_bytes
            and len(self.records) < self.limits.metadata_batches,
            "source_credit_exceeded",
            "Please wait for source credit before sending more frames.",
        )
        record = AcceptedSource(self.sequence + 1, event_id, seq, start, end, bytes_end)
        self.records.append(record)
        self.sequence += 1
        self.accepted_frames, self.accepted_bytes, self.event_id = (
            end,
            bytes_end,
            event_id,
        )
        return record

    def consume(self, owned_frame_end):
        """Release complete appends covered by the downstream ownership watermark.

        Downstream has taken ownership; this does not claim GPU completion or
        that decoded frames have been delivered to the client.
        """
        require(
            not self.closed
            and integer(owned_frame_end)
            and self.consumed_frames <= owned_frame_end <= self.accepted_frames,
            "Invalid source ownership watermark.",
        )
        message = source_consumed_message(
            ((record.end, record.bytes_end) for record in self.records),
            self.consumed_frames,
            owned_frame_end,
        )
        if message is None:
            return None
        self.consumed_frames = message["source_frame_end"]
        self.consumed_bytes = message["source_bytes_end"]
        self._retire()
        return message

    def provenance_for(self, pixels, dependency):
        require(
            not self.closed
            and isinstance(pixels, PixelRange)
            and pixels.end <= self.accepted_frames
            and integer(dependency)
            and pixels.end <= dependency <= self.accepted_frames,
            "Invalid source output dependency.",
        )
        spans = []
        for record in self.records:
            start, end = max(pixels.start, record.start), min(pixels.end, record.end)
            if start < end:
                spans.append(SourceSpan(record.source_id, start, start, end - start))
        return Provenance(pixels, tuple(spans), dependency)

    def delivered(self, pixels):
        require(
            not self.closed
            and isinstance(pixels, PixelRange)
            and pixels.start == self.output_frames
            and pixels.end <= self.accepted_frames,
            "Output frames must be delivered in order.",
        )
        self.output_frames = pixels.end
        self._retire()

    def _retire(self):
        end = min(self.output_frames, self.consumed_frames)
        while self.records and self.records[0].end <= end:
            self.records.popleft()

    def finish(self, frame_end, *, event_id):
        require(not self.closed and not self.ended, "The source stream has ended.")
        self.validate_event_id(event_id)
        require(
            integer(frame_end, 1)
            and frame_end == self.accepted_frames
            and (self.total_frames is None or frame_end == self.total_frames),
            "The source ended with an unexpected frame count.",
        )
        self.ended, self.event_id = True, event_id

    def completed(self):
        require(
            not self.closed
            and self.ended
            and self.output_frames == self.accepted_frames
            and self.consumed_frames == self.accepted_frames,
            "The video has not finished processing.",
        )
        return {"type": "session.completed", "output_frames": self.output_frames}

    def dispose(self):
        self.records.clear()
        self.closed = True


def source_work_quantum(value=None):
    """Resolve trusted deployment policy, with explicit settings before legacy env."""
    import os

    if value is None:
        raw = os.environ.get("SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM", "0")
        if raw not in ("0", "1"):
            raise ValueError("Source work quantum must be 0 or 1")
        value = int(raw)
    if type(value) is not int or value not in (0, 1):
        raise ValueError("Source work quantum must be 0 or 1")
    return value
