# SPDX-License-Identifier: Apache-2.0
"""Shared Zing media counts, pixel provenance and identifiers.

No wire negotiation, credit ownership, session lifecycle or transport state.
The service integration owns those responsibilities and supplies actual source
alignment/dependencies rather than deriving them from the latest accepted batch.
"""

from dataclasses import asdict, dataclass
from typing import NewType

# Default deployment budgets; callers may configure a different image count.
# The presentation budget includes the source-preview video.
DEFAULT_MAX_REFERENCE_IMAGES = 21
DEFAULT_MAX_MEDIA_ITEMS = DEFAULT_MAX_REFERENCE_IMAGES + 1
MAX_REFERENCE_IMAGE_PIXELS = 16_777_216
# The old whole-upload cap also bounded any one encoded image. Preserve that
# single-image maximum when deriving a larger deployment's complete snapshot.
MAX_REFERENCE_IMAGE_BYTES = 64 << 20


@dataclass(frozen=True)
class ReferenceLimits:
    """Count-derived ingress budgets; no tensor allocation or token estimates."""

    max_images: int = DEFAULT_MAX_REFERENCE_IMAGES

    def __post_init__(self):
        if type(self.max_images) is not int or self.max_images < 0:
            raise ValueError("zing_reference_max_images must be a nonnegative integer")

    @property
    def max_media(self):
        return self.max_images + 1

    @property
    def reference_bytes(self):
        return self.max_images * MAX_REFERENCE_IMAGE_BYTES

    @property
    def decoded_reference_bytes(self):
        return self.max_images * MAX_REFERENCE_IMAGE_PIXELS * 3

    @property
    def media_bytes(self):
        # Explicit source-preview uploads also need a media slot and byte budget.
        return self.max_media * MAX_REFERENCE_IMAGE_BYTES

    @property
    def decoded_media_bytes(self):
        return self.max_media * MAX_REFERENCE_IMAGE_PIXELS * 3


TemporalChunkId = NewType("TemporalChunkId", int)
PackedRoleId = NewType("PackedRoleId", int)


class ContractError(ValueError):
    """Invalid shared metadata; transport error mapping belongs to the caller."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


@dataclass(frozen=True)
class PixelRange:
    start: int
    end: int

    def __post_init__(self):
        require(
            integer(self.start) and integer(self.end) and self.end > self.start,
            "Invalid frame range.",
        )

    @property
    def count(self):
        return self.end - self.start


@dataclass(frozen=True)
class SourceSpan:
    source_id: int
    source_frame_start: int
    output_frame_start: int
    num_frames: int

    def __post_init__(self):
        require(
            integer(self.source_id, 1)
            and integer(self.num_frames, 1)
            and integer(self.source_frame_start)
            and integer(self.output_frame_start)
            and self.source_frame_start == self.output_frame_start,
            "Invalid source alignment.",
        )


@dataclass(frozen=True)
class Provenance:
    """Absolute pixel mapping; dependency is supplied by actual computation.

    Temporal chunk IDs, packed role IDs and prompt versions are NOT source IDs.
    Slicing preserves dependency (including lookahead), never latest-accepted.
    """

    pixels: PixelRange
    source_alignment: tuple[SourceSpan, ...]
    source_dependency_end: int

    def __post_init__(self):
        require(isinstance(self.pixels, PixelRange), "Invalid frame range.")
        require(
            isinstance(self.source_alignment, (tuple, list))
            and all(isinstance(s, SourceSpan) for s in self.source_alignment),
            "Invalid source alignment.",
        )
        object.__setattr__(self, "source_alignment", tuple(self.source_alignment))
        cursor = self.pixels.start
        for span in self.source_alignment:
            require(span.output_frame_start == cursor, "Incomplete source alignment.")
            cursor += span.num_frames
        require(cursor == self.pixels.end, "Incomplete source alignment.")
        require(
            integer(self.source_dependency_end)
            and self.source_dependency_end >= self.pixels.end,
            "Invalid source dependency.",
        )

    def slice(self, pixels: PixelRange):
        require(
            self.pixels.start <= pixels.start < pixels.end <= self.pixels.end,
            "Invalid output frame range.",
        )
        spans = []
        for span in self.source_alignment:
            start = max(pixels.start, span.output_frame_start)
            end = min(pixels.end, span.output_frame_start + span.num_frames)
            if start < end:
                spans.append(SourceSpan(span.source_id, start, start, end - start))
        return Provenance(pixels, tuple(spans), self.source_dependency_end)

    @classmethod
    def concatenate(cls, parts):
        parts = tuple(parts)
        require(bool(parts), "Output frames are required.")
        for left, right in zip(parts, parts[1:]):
            require(
                left.pixels.end == right.pixels.start,
                "Output frames must be continuous.",
            )
        return cls(
            PixelRange(parts[0].pixels.start, parts[-1].pixels.end),
            tuple(s for p in parts for s in p.source_alignment),
            max(p.source_dependency_end for p in parts),
        )

    def metadata(self):
        return dict(
            output_frame_start=self.pixels.start,
            num_frames=self.pixels.count,
            source_alignment=[asdict(s) for s in self.source_alignment],
            source_dependency_end=self.source_dependency_end,
        )
