# SPDX-License-Identifier: Apache-2.0
"""Verify predecoded RGB outside request timing, preserving full case identity."""

import hashlib
import json
from pathlib import Path


def bind_frozen_sources(cases, manifest):
    manifest = Path(manifest)
    entries = json.loads(manifest.read_text())
    result = []
    for case in cases:
        row = entries[case["case_id"]]
        width, height = (
            case.get("source_width", case["width"]),
            case.get("source_height", case["height"]),
        )
        if (row["width"], row["height"], row["frames"]) != (
            width,
            height,
            case["frames"],
        ):
            raise ValueError("Frozen RGB geometry or frame count differs from case")
        path = Path(row["path"])
        if not path.is_absolute():
            path = manifest.parent / path
        expected = width * height * 3 * case["frames"]
        if row["bytes"] != expected or path.stat().st_size != expected:
            raise ValueError("Frozen RGB byte count differs from case")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != row["sha256"]:
            raise ValueError("Frozen RGB hash differs from manifest")
        result.append(
            {
                **case,
                "source_rgb": str(path.resolve()),
                "frozen_source": row,
                "source_rgb_manifest_sha256": hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest(),
            }
        )
    return result
