"""Freeze existing, authorized playable metadata for Shanghai-copy validation."""

import argparse
import gzip
import hashlib
import io
import json
import os
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

PLATFORM_ORIGIN = "https://world-test-platform.loopit.com.cn"


def https_origin(url):
    if any(ord(char) <= 32 or ord(char) == 127 for char in url):
        raise ValueError("Unexpected whitespace in fetch URL")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Fetch URLs must use HTTPS without embedded credentials")
    return parsed.scheme, parsed.hostname, 443 if parsed.port is None else parsed.port


def platform_origin(url):
    parsed = urlsplit(url)
    if (
        https_origin(url) != https_origin(PLATFORM_ORIGIN)
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Expected the official HTTPS platform origin")
    return PLATFORM_ORIGIN


class HTTPSRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, authenticated_origin):
        self.authenticated_origin = authenticated_origin

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            target = https_origin(newurl)
            if (
                self.authenticated_origin is not None
                and target != self.authenticated_origin
            ):
                raise ValueError(
                    "Authenticated redirects must stay on the platform origin"
                )
        except ValueError:
            fp.close()
            raise
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url, token=None):
    origin = https_origin(url)
    authenticated_origin = None
    if token is not None:
        authenticated_origin = https_origin(PLATFORM_ORIGIN)
        if origin != authenticated_origin:
            raise ValueError("Platform credentials require the official HTTPS origin")
    request = urllib.request.Request(
        url, headers={"Authorization": "Bearer " + token} if token is not None else {}
    )
    # Check every hop before urllib copies Authorization to its next Request.
    opener = urllib.request.build_opener(HTTPSRedirects(authenticated_origin))
    with opener.open(request, timeout=30) as response:
        data = response.read(16 * 1024 * 1024 + 1)
        encoding = response.headers.get("Content-Encoding", "identity")
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("Metadata too large")
    if encoding == "gzip":
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
            data = stream.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024:
            raise ValueError("Decoded metadata too large")
    elif encoding != "identity":
        raise ValueError("Unsupported metadata content encoding")
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("snapshot", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("worlds", nargs="+")
    a = p.parse_args()
    if not str(a.snapshot).startswith(
        "/var/lib/whitebox-shanghai-copy/managed-parallel-startup-20261113/"
    ):
        raise ValueError("Expected private test snapshot")
    room = json.loads((a.snapshot / "room-inspect.private.json").read_text())
    gateway = json.loads((a.snapshot / "gateway-inspect.private.json").read_text())
    env = dict(v.split("=", 1) for v in room["Config"]["Env"])
    ge = dict(v.split("=", 1) for v in gateway["Config"]["Env"])
    origin = platform_origin(env.get("WHITEBOX_CLI_PLATFORM_URL", PLATFORM_ORIGIN))
    a.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    print(
        json.dumps(
            {
                "single_control_configured": bool(
                    ge.get("WHITEBOX_SINGLE_CONTROL_SECRET")
                ),
                "services": ge.get("WHITEBOX_SINGLE_CONTROL_SERVICES", ""),
                "catalog_origin": origin,
            }
        )
    )
    for world in a.worlds:
        if not world.startswith("world-") or not world[6:].isalnum():
            raise ValueError("Invalid world")
        root = a.output / world
        root.mkdir(mode=0o700)
        base = origin + "/api/worlds/interactive-production/" + world
        for kind, suffix in [
            ("detail", "?includeArtifacts=0"),
            ("artifacts", "/artifacts"),
            ("playable", "/playable-location?transport=cdn"),
        ]:
            try:
                data = fetch(base + suffix, env["WORLD_TEST_API_TOKEN"])
                value = json.loads(data)
                (root / (kind + ".private.json")).write_bytes(data)
                summary = {
                    "world": world,
                    "kind": kind,
                    "keys": list(value),
                    "bytes": len(data),
                }
                if kind == "playable" and value.get("ready"):
                    location = value["location"]
                    raw = fetch(value["manifestUrl"])
                    if hashlib.sha256(raw).hexdigest() != location["manifestSha256"]:
                        raise ValueError("Playable manifest identity changed")
                    (root / "playable-manifest.private.json").write_bytes(raw)
                    summary.update(
                        ready=True,
                        publication={
                            k: location[k]
                            for k in [
                                "sourceId",
                                "worldId",
                                "publicationId",
                                "sourceManifestSha256",
                                "renditionInputSha256",
                                "manifestSha256",
                            ]
                        },
                    )
                print(json.dumps(summary))
            except Exception as error:
                print(
                    json.dumps(
                        {
                            "world": world,
                            "kind": kind,
                            "error": type(error).__name__,
                            "http_status": getattr(error, "code", None),
                        }
                    )
                )


if __name__ == "__main__":
    os.umask(0o077)
    main()
