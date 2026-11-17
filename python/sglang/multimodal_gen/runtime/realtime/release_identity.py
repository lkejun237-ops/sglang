# SPDX-License-Identifier: Apache-2.0
"""Read-only release identity, independent of CUDA and request processing.

Image identity is baked by the manual builder. Regional config is supplied by
that release's launcher. No environment dump, credential, cache key or path is
returned. Container image digests must be read from the platform, not guessed
from a tag or copied into the image it identifies.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


def release_identity(role: str) -> dict:
    filename = os.environ.get("WHITEBOX_RELEASE_MANIFEST", "")
    value = {"schema_version": 1, "role": role, "identity_status": "unmanaged"}
    if not filename:
        return value
    manifest = json.loads(Path(filename).read_text())
    if manifest.get("source_sha") == manifest.get("sdk_sha") == "unmanaged":
        return value
    for field in ("source_sha", "sdk_sha"):
        if not re.fullmatch(r"[0-9a-f]{40}", manifest.get(field, "")):
            raise ValueError(f"Invalid release {field}")
    branch = manifest.get("source_branch", "")
    development = branch != "main"
    if development and not (
        os.environ.get("WHITEBOX_ALLOW_DEVELOPMENT_RELEASE") == "1"
        and re.fullmatch(r"codex/[A-Za-z0-9][A-Za-z0-9._/-]*", branch)
    ):
        raise ValueError("Release source must be main")
    value.update(
        {key: manifest[key] for key in ("source_sha", "sdk_sha", "source_branch")}
    )
    value["identity_status"] = "image_only"
    if development:
        value["release_scope"] = "isolated_development_test"
    config = os.environ.get("WHITEBOX_RELEASE_CONFIG_SHA256")
    region = os.environ.get("WHITEBOX_RELEASE_REGION")
    if config is not None or region is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", config or "") or region not in (
            "cn-shanghai",
            "ap-southeast-7",
        ):
            raise ValueError("Invalid release region or configuration digest")
        bundle_path = os.environ.get("WHITEBOX_RELEASE_BUNDLE")
        if not bundle_path:
            raise ValueError("Configured releases require their mounted bundle")
        bundle = json.loads(Path(bundle_path).read_text())
        actual = hashlib.sha256(
            json.dumps(
                {"policy": bundle["policy"], "bindings": bundle["bindings"]},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if (
            actual != config
            or bundle.get("config_sha256") != config
            or bundle.get("region") != region
        ):
            raise ValueError("Mounted release configuration digest does not match")
        expected = bundle["environments"][role]
        pin = bundle["bindings"].get("control_credential_refs")
        if pin is not None and role in ("room", "gateway", "coordinator"):
            if os.environ.get("WHITEBOX_CONTROL_CREDENTIAL_REFS") != pin:
                raise ValueError(
                    "Control credential references differ from the release bundle"
                )
        # Only release-owned values are compared. Authentication settings and
        # unrelated platform config remain in the existing secret/config store.
        mismatches = sorted(
            key for key, value in expected.items() if os.environ.get(key) != value
        )
        if mismatches:
            raise ValueError("Release environment differs: " + ", ".join(mismatches))
        value.update(config_sha256=config, region=region, identity_status="configured")
    for name in ("CHECKPOINT", "MODEL_CONFIG"):
        checksum = os.environ.get(f"WHITEBOX_{name}_SHA256")
        if checksum is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", checksum):
                raise ValueError("Invalid model identity digest")
            value[name.lower() + "_sha256"] = checksum
    return value


def install_release_route(app, role: str):
    # Freeze at process startup: changing a mounted manifest cannot make an old
    # process claim to run newly deployed code.
    identity = release_identity(role)
    if (
        role in ("room", "gateway", "coordinator")
        and "WHITEBOX_CONTROL_CREDENTIAL_REFS" in os.environ
    ):
        from .control_bindings import verify

        identity["control_binding_sha256"] = verify(role)
    app.state.whitebox_release = identity

    @app.get("/v1/whitebox/version", include_in_schema=False)
    async def version():
        return identity
