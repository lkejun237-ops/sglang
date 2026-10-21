# SPDX-License-Identifier: Apache-2.0
"""Caller -> API file bytes, retained in RAM until worker dispatch completes.

PNG/JPEG/WebP payloads arrive over WebSocket/MessagePack and pass unchanged to
the worker's memory decoder. No OSS upload, media file or transcoding occurs.
Pickle/IPC and SP broadcast still copy bytes; this is not a zero-copy path.
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
    DEFAULT_MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_IMAGE_BYTES,
    ReferenceLimits,
    require,
)

from ..conditioning import (
    release_media_reservations,
    reserve_media_memory,
    verify_presentation_upload,
)


@dataclass
class ReferenceUpload:
    """Own received encoded bytes and their RAM leases, not GPU tensors."""

    items: list[dict] = field(default_factory=list)
    reservations: list = field(default_factory=list)

    def release(self, session):
        self.items.clear()
        release_media_reservations(session, self.reservations)
        self.reservations.clear()


async def prepare_reference_upload(
    session,
    references,
    *,
    fixed_ids,
    fixed_bytes,
    max_references=DEFAULT_MAX_REFERENCE_IMAGES,
    max_decoded_bytes=None,
):
    limits = ReferenceLimits(max_references)
    max_media_items = limits.max_media
    if max_decoded_bytes is None:
        max_decoded_bytes = limits.decoded_reference_bytes
    require(
        isinstance(references, list) and len(references) <= max_references,
        f"Please provide a complete list of at most {max_references} reference images.",
    )
    require(
        len(references) + len(fixed_ids) <= max_media_items,
        f"Please provide at most {max_media_items} media items including source preview.",
    )
    seen, size = set(fixed_ids), fixed_bytes
    for item in references:
        require(
            isinstance(item, dict)
            and set(item) == {"id", "format", "payload"}
            and isinstance(item["id"], str)
            and bool(item["id"])
            and item["id"] not in seen
            and item["format"] in ("jpeg", "webp", "png")
            and isinstance(item["payload"], bytes),
            "Please provide uniquely named reference images with supported image bytes.",
        )
        seen.add(item["id"])
        require(
            len(item["payload"]) <= MAX_REFERENCE_IMAGE_BYTES,
            "The reference image is too large.",
        )
        size += len(item["payload"])
    require(size <= limits.media_bytes, "The reference images are too large.")
    upload = ReferenceUpload()
    try:
        if references:
            upload.reservations.append(
                reserve_media_memory(session, size - fixed_bytes)
            )
        decoded_bytes = 0
        for item in references:
            item_bytes = await verify_presentation_upload(
                session, item["payload"], item["format"]
            )
            if max_decoded_bytes is not None:
                decoded_bytes += item_bytes
                require(
                    decoded_bytes <= max_decoded_bytes,
                    "The reference images exceed the configured decoded size limit.",
                )
            upload.items.append(dict(item, type="image"))
        return upload
    except BaseException:
        upload.release(session)
        raise
