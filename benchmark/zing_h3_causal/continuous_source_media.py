# SPDX-License-Identifier: Apache-2.0
"""Reference payloads shared by the production source clients."""

import json
from io import BytesIO
from pathlib import Path

from PIL import Image


def reference_payload(path, media_id):
    data = Path(path).read_bytes()
    with Image.open(BytesIO(data)) as image:
        image_format = image.format.lower()
    if image_format not in {"jpeg", "png", "webp"}:
        raise ValueError("Reference images must be JPEG, PNG, or WebP.")
    return dict(id=media_id, format=image_format, payload=data)


def load_sink_updates(path):
    if path is None:
        return []
    updates = json.loads(Path(path).read_text())
    if not isinstance(updates, list):
        raise ValueError("Sink updates must be a JSON list")
    previous = -1
    for update in updates:
        if (
            not isinstance(update, dict)
            or set(update) != {"source_frame_start", "operation"}
            or type(update["source_frame_start"]) is not int
            or update["source_frame_start"] < max(0, previous)
            or not isinstance(update["operation"], str)
            or update["operation"] not in {"off", "refresh"}
        ):
            raise ValueError(
                "Sink updates require ordered nonnegative source frames and off/refresh operations"
            )
        previous = update["source_frame_start"]
    return updates


def load_prompt_updates(path):
    """Read ordered control metadata; load image bytes only when sending an event."""
    if path is None:
        return []
    path = Path(path)
    updates = json.loads(path.read_text())
    if not isinstance(updates, list):
        raise ValueError("Prompt updates must be a JSON list")
    previous = -1
    for update in updates:
        if (
            not isinstance(update, dict)
            or set(update) - {"source_frame_start", "prompt", "references"}
            or type(update.get("source_frame_start")) is not int
            or update["source_frame_start"] <= previous
            or not isinstance(update.get("prompt"), str)
            or not update["prompt"].strip()
        ):
            raise ValueError(
                "Prompt updates require increasing source frame positions and non-empty prompts"
            )
        previous = update["source_frame_start"]
        if "references" in update:
            references = update["references"]
            if not isinstance(references, list):
                raise ValueError("References must be a complete list of images")
            seen = set()
            for item in references:
                if (
                    not isinstance(item, dict)
                    or set(item) != {"id", "path"}
                    or not isinstance(item["id"], str)
                    or not item["id"]
                    or item["id"] in seen
                    or not isinstance(item["path"], str)
                ):
                    raise ValueError(
                        "References require unique IDs and local image paths"
                    )
                seen.add(item["id"])
                item["path"] = str(path.parent / item["path"])
    return updates
