"""Prepare original sealed image bytes for managed protocol tests, without playing."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from urllib.parse import urljoin

from collect_cases import PLATFORM_ORIGIN, fetch, https_origin, platform_origin
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", type=Path)
    p.add_argument("case_directory", type=Path)
    a = p.parse_args()
    if a.stage != Path("/stage") or a.case_directory.parent != a.stage:
        raise ValueError("Use the exact private test bind")
    room = json.loads((a.stage / "before-r1/room-inspect.private.json").read_text())
    env = dict(item.split("=", 1) for item in room["Config"]["Env"])
    origin = platform_origin(env.get("WHITEBOX_CLI_PLATFORM_URL", PLATFORM_ORIGIN))
    for root in a.case_directory.iterdir():
        if not root.is_dir():
            continue
        detail = json.loads((root / "detail.private.json").read_text())["world"]
        artifacts = json.loads((root / "artifacts.private.json").read_text())
        playable = json.loads((root / "playable.private.json").read_text())
        raw_manifest = (root / "playable-manifest.private.json").read_bytes()
        location = playable["location"]
        assert playable["ready"] and detail["status"] == "ready"
        assert artifacts["attempt"] == detail["attempt"]
        assert hashlib.sha256(raw_manifest).hexdigest() == location["manifestSha256"]
        publication = {
            key: location[key]
            for key in [
                "sourceId",
                "worldId",
                "publicationId",
                "sourceManifestSha256",
                "renditionInputSha256",
                "manifestSha256",
            ]
        }
        publication["attempt"] = detail["attempt"]
        out = root / "input-files"
        out.mkdir(mode=0o700, exist_ok=False)
        proof = {}
        for name, key in [("reference", "reference"), ("opening", "openingFrame")]:
            item = artifacts["artifacts"]["images"][key]
            url = urljoin(origin, item["url"])
            token = (
                env["WORLD_TEST_API_TOKEN"]
                if https_origin(url) == https_origin(origin)
                else None
            )
            raw = fetch(url, token)
            if len(raw) != item["bytes"]:
                raise ValueError("Image delivery is not the declared original size")
            with Image.open(io.BytesIO(raw)) as image:
                if name == "opening":
                    if (
                        image.format != "PNG"
                        or image.mode not in ("RGB", "RGBA")
                        or len(raw) > 2 * 1024 * 1024
                        or raw[24] != 8
                    ):
                        raise ValueError("Unsupported initial PNG")
                    if getattr(image, "n_frames", 1) != 1 or any(
                        k in image.info for k in ("icc_profile", "exif")
                    ):
                        raise ValueError("Unsupported initial PNG metadata")
                    if image.mode == "RGBA" and image.getchannel("A").getextrema() != (
                        255,
                        255,
                    ):
                        raise ValueError("Initial PNG must be opaque")
                    metadata = {
                        "format": "png",
                        "width": image.width,
                        "height": image.height,
                        "rgb_sha256": hashlib.sha256(
                            image.convert("RGB").tobytes()
                        ).hexdigest(),
                    }
                else:
                    metadata = {
                        "format": {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}[
                            image.format
                        ]
                    }
            file = out / (name + "." + metadata["format"])
            file.write_bytes(raw)
            proof[name] = {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                **metadata,
                "file": str(file),
            }
        # Check the frozen publisher identity once more; CLI revalidates it again.
        current = json.loads(
            fetch(
                origin
                + "/api/worlds/"
                + publication["sourceId"]
                + "/"
                + publication["worldId"]
                + "/playable-location?transport=cdn",
                env["WORLD_TEST_API_TOKEN"],
            )
        )
        assert current["ready"] and all(
            current["location"][k] == publication[k]
            for k in publication
            if k != "attempt"
        )
        profile = {
            "source_id": publication["sourceId"],
            "world_id": publication["worldId"],
            "publication": publication,
            "frozen_publication": {
                "status": "verified",
                "scope": "world-frozen-publication",
                "world": publication,
            },
            "prompt": detail["prompt"],
            "seed": 42,
            "references": [
                {
                    "id": "reference-0",
                    "format": proof["reference"]["format"],
                    "file": proof["reference"]["file"],
                }
            ],
            "initial_frame": {
                k: v
                for k, v in proof["opening"].items()
                if k not in ("sha256", "bytes")
            },
        }
        (root / "input-profile.private.json").write_text(json.dumps(profile))
        (root / "input-proof.json").write_text(json.dumps(proof, indent=2))
        print(
            json.dumps(
                {
                    "world": root.name,
                    "publication": publication["publicationId"],
                    "prompt_sha256": hashlib.sha256(
                        detail["prompt"].encode()
                    ).hexdigest(),
                    "images": proof,
                }
            )
        )


if __name__ == "__main__":
    os.umask(0o077)
    main()
