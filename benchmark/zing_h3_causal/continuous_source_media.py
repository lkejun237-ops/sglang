# SPDX-License-Identifier: Apache-2.0
"""Reference payloads shared by the production source clients."""

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
