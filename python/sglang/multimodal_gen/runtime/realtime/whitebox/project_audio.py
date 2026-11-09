# SPDX-License-Identifier: Apache-2.0
"""Source-clock audio and immutable recording resources; never model audio.

The private CLI session capability admits only its fixed audio route. Sealed
bytes join the existing owner-scoped recording asset archive, with an internal
namespace so UI and audio cannot overwrite one another.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import re
import time
from urllib.parse import quote, urlsplit

import httpx

AUDIO_PREFIX = "__project_audio/"
SHA = re.compile(r"[a-f0-9]{64}\Z")


def public_ui_metadata(metadata):
    """Expose UI bytes only; audio stays in the recording's private archive map."""
    files = metadata.get("files", {})
    if not any(path.startswith(AUDIO_PREFIX) for path in files):
        return metadata
    return {
        **metadata,
        "files": {
            path: digest
            for path, digest in files.items()
            if not path.startswith(AUDIO_PREFIX)
        },
    }


def audio_path_allowed(path):
    return isinstance(path, str) and (
        path == "project-audio.json"
        or (
            path.startswith("assets/audio/")
            and len(path) <= 512
            and not re.search(r"[\\:%?#]", path)
            and all(part not in {"", ".", ".."} for part in path.split("/"))
        )
    )


def read_audio_snapshot(value):
    def integer(n):
        return type(n) is int and 0 <= n <= 2**53 - 1

    def identity(s):
        return isinstance(s, str) and 0 < len(s) <= 128

    def finite(n):
        return type(n) in (int, float) and math.isfinite(n)

    def require(ok):
        if not ok:
            raise ValueError("Invalid source audio snapshot")

    require(
        isinstance(value, dict)
        and set(value) - {"listener"}
        == {
            "type",
            "version",
            "sessionId",
            "epoch",
            "source",
            "completeThroughUs",
            "eventsCompleteFromUs",
            "events",
            "voices",
        }
    )
    source = value["source"]
    require(
        value["type"] == "audio.snapshot"
        and type(value["version"]) is int
        and value["version"] == 1
        and identity(value["sessionId"])
        and integer(value["epoch"])
        and isinstance(source, dict)
        and set(source)
        == {
            "presentationId",
            "sdkEpoch",
            "sourceFrameId",
            "sourceTimeUs",
            "simulationTick",
            "worldRevision",
        }
    )
    require(
        identity(source["presentationId"])
        and all(integer(source[k]) for k in source if k != "presentationId")
    )
    require(
        integer(value["completeThroughUs"])
        and value["completeThroughUs"] == source["sourceTimeUs"]
        and integer(value["eventsCompleteFromUs"])
        and value["eventsCompleteFromUs"] <= source["sourceTimeUs"]
    )
    for key, limit in (("events", 1024), ("voices", 128)):
        rows = value[key]
        require(isinstance(rows, list) and len(rows) <= limit)
        for row in rows:
            fields = {"entityId", "sequence", "soundId", "resource", "gain"}
            fields |= (
                {"command", "sourceTimeUs"}
                if key == "events"
                else {"voiceId", "startedAtSourceTimeUs"}
            )
            require(
                isinstance(row, dict)
                and fields <= set(row)
                and not set(row) - fields - {"positionMetersXYZ"}
            )
            require(
                isinstance(row["entityId"], str)
                and bool(row["entityId"])
                and isinstance(row["soundId"], str)
                and bool(row["soundId"])
                and isinstance(row["resource"], str)
                and bool(row["resource"])
                and integer(row["sequence"])
                and finite(row["gain"])
                and 0 <= row["gain"]
            )
            time = (
                row["sourceTimeUs"] if key == "events" else row["startedAtSourceTimeUs"]
            )
            require(integer(time) and time <= source["sourceTimeUs"])
            require(
                row["command"] in ("play", "stop", "gain")
                if key == "events"
                else isinstance(row["voiceId"], str) and bool(row["voiceId"])
            )
            if "positionMetersXYZ" in row:
                require(
                    isinstance(row["positionMetersXYZ"], list)
                    and len(row["positionMetersXYZ"]) == 3
                    and all(finite(n) for n in row["positionMetersXYZ"])
                )
    if "listener" in value:
        listener = value["listener"]
        require(
            isinstance(listener, dict)
            and set(listener) == {"positionMetersXYZ", "forwardXYZ", "upXYZ"}
            and all(
                isinstance(vector, list)
                and len(vector) == 3
                and all(finite(n) for n in vector)
                for vector in listener.values()
            )
        )
    require(
        len(
            json.dumps(
                value, allow_nan=False, ensure_ascii=False, separators=(",", ":")
            ).encode()
        )
        <= 1 << 20
    )
    return copy.deepcopy(value)


async def install_audio_package(view):
    descriptor = view.descriptor
    audio = descriptor.get("audio")
    if audio is None:
        return
    base = descriptor["baseUrl"].rstrip("/")
    expected = (
        base
        + "/v1/sessions/"
        + quote(descriptor["sessionId"], safe="")
        + "/audio/project-audio.json"
    )
    parsed = urlsplit(base)
    if (
        not isinstance(audio, dict)
        or set(audio) != {"manifestUrl", "manifestSha256"}
        or audio["manifestUrl"] != expected
        or not isinstance(audio["manifestSha256"], str)
        or not SHA.fullmatch(audio["manifestSha256"])
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or parsed.scheme not in {"http", "https"}
        or (
            parsed.scheme == "http"
            and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        )
    ):
        raise ValueError("Invalid CLI audio capability")
    files = {}
    started = time.perf_counter()
    async with (
        asyncio.timeout(30),
        httpx.AsyncClient(
            timeout=30, follow_redirects=False, trust_env=False
        ) as client,
    ):

        async def read(path, digest, limit):
            if not audio_path_allowed(path):
                raise ValueError("Invalid CLI audio resource path")
            async with client.stream(
                "GET",
                expected.rsplit("/", 1)[0] + "/" + path,
                headers={"Authorization": "Bearer " + descriptor["accessToken"]},
            ) as response:
                response.raise_for_status()
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > limit:
                        raise ValueError("CLI audio resource exceeded its limit")
                    chunks.append(chunk)
                raw = b"".join(chunks)
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("CLI audio resource identity changed")
            files[AUDIO_PREFIX + path] = raw
            return raw

        manifest = json.loads(
            await read("project-audio.json", audio["manifestSha256"], 256 << 10)
        )
        if (
            not isinstance(manifest, dict)
            or manifest.get("contract") != "worldkit.project-audio-byte-seal"
            or manifest.get("version") != 1
            or not isinstance(manifest.get("resources"), list)
            or len(manifest["resources"]) > 64
        ):
            raise ValueError("Invalid CLI audio manifest")
        total, decoded = 0, 0
        paths = set()
        for resource in manifest["resources"]:
            if not isinstance(resource, dict):
                raise ValueError("Invalid CLI audio resource")
            path, digest, size, info = (
                resource.get(k) for k in ("path", "sha256", "byteLength", "decoded")
            )
            if (
                not audio_path_allowed(path)
                or path == "project-audio.json"
                or path in paths
                or not isinstance(digest, str)
                or not SHA.fullmatch(digest)
                or type(size) is not int
                or size <= 0
                or not isinstance(info, dict)
                or any(
                    type(info.get(k)) is not int or info[k] <= 0
                    for k in ("channels", "sampleRate", "frames", "decodedBytes")
                )
                or info["decodedBytes"] != info["channels"] * info["frames"] * 4
            ):
                raise ValueError("Invalid CLI audio resource seal")
            paths.add(path)
            total += size
            decoded += info["decodedBytes"]
            if total > 32 << 20 or decoded > 128 << 20:
                raise ValueError("CLI audio package exceeded its limit")
            if len(await read(path, digest, size)) != size:
                raise ValueError("CLI audio resource length changed")
    original_asset = view.renderer.asset

    async def asset(path):
        if path.startswith(AUDIO_PREFIX):
            if path not in files:
                raise FileNotFoundError("Audio resource is unavailable")
            return files[path]
        return await original_asset(path)

    if any(name in view.renderer.metadata.get("files", {}) for name in files):
        raise ValueError("CLI audio archive namespace is reserved")
    view.renderer.asset = asset
    view.renderer.metadata.setdefault("files", {}).update(
        {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    )
    view.metadata["project_audio"] = {"manifestSha256": audio["manifestSha256"]}
    view.metadata["audio_resource_prefetch"] = {
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "encoded_bytes": sum(len(data) for data in files.values()),
        "resource_count": len(manifest["resources"]),
    }
