# SPDX-License-Identifier: Apache-2.0
"""Check deployment-owned OOS identities without fetching or installing secret values."""

import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

REFS_ENV = "WHITEBOX_CONTROL_CREDENTIAL_REFS"
REQUIRED = {"room_internal", "pool_room", "operator", "registration", "signing"}
OPTIONAL = {"rtc", "room_admin"}


def parse_manifest(serialized):
    if not isinstance(serialized, str) or len(serialized) > 16384:
        raise ValueError("Invalid control credential manifest size")
    m = json.loads(serialized)
    if (
        not isinstance(m, dict)
        or set(m) != {"environment", "region", "pool_id", "refs", "audiences"}
        or m["environment"] != "test-shanghai-copy"
        or m["region"] != "cn-shanghai"
        or not isinstance(m["pool_id"], str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", m["pool_id"])
        or not isinstance(m["refs"], dict)
        or not REQUIRED <= set(m["refs"]) <= REQUIRED | OPTIONAL
    ):
        raise ValueError("Invalid Shanghai test control credential manifest")
    paths = {
        "room_internal": "/internal/worlds/runtime",
        "pool_room": "/v1/role-pool/capacity",
    }
    if not isinstance(m["audiences"], dict) or set(m["audiences"]) != set(paths):
        raise ValueError("Expected Room status and pool capacity audiences")
    for purpose, path in paths.items():
        url = m["audiences"][purpose]
        if not isinstance(url, str) or any(ord(char) < 33 for char in url):
            raise ValueError("Invalid publisher audience for " + purpose)
        u = urlsplit(url)
        if (
            u.scheme not in ("http", "https")
            or not u.hostname
            or u.username is not None
            or u.password is not None
            or u.path != path
            or u.query
            or u.fragment
        ):
            raise ValueError("Invalid publisher audience for " + purpose)
    ids = set()
    for purpose, ref in m["refs"].items():
        if (
            not isinstance(ref, dict)
            or set(ref) != {"parameter_id", "parameter_name", "version"}
            or not isinstance(ref["parameter_id"], str)
            or not re.fullmatch(r"p-[A-Za-z0-9]+", ref["parameter_id"])
            or not isinstance(ref["parameter_name"], str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,180}", ref["parameter_name"])
            or type(ref["version"]) is not int
            or not 1 <= ref["version"] <= 100
            or ref["parameter_id"] in ids
        ):
            raise ValueError("Invalid or reused managed reference for " + purpose)
        ids.add(ref["parameter_id"])
    return m


def marker(ref):
    return "cn-shanghai/{parameter_name}/{parameter_id}/v{version}".format(**ref)


def manifest_digest(m):
    canonical = json.dumps(m, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def verify(role, pool=None, environ=None, preparing=False):
    """Compare issuer-delivered companion metadata at startup; native auth stays authoritative."""
    env = os.environ if environ is None else environ
    raw = env.get(REFS_ENV)
    if raw is None:
        return None
    m = parse_manifest(raw)
    needed = {}
    if role == "coordinator":
        path = env.get("SGLANG_ROLE_POOL_COORDINATOR_CONFIG")
        if path:
            pool = json.loads(Path(path).read_text())
            role = "pool_coordinator"
            if env.get("WHITEBOX_ROOM_OWNERS"):
                needed["room_internal"] = "WHITEBOX_INTERNAL_TOKEN"
        else:
            role = "directory_coordinator"
    if role in ("room", "gateway", "directory_coordinator", "route_publisher"):
        needed["room_internal"] = "WHITEBOX_INTERNAL_TOKEN"
    if role in ("room", "gateway"):
        path = env.get("WHITEBOX_ROLE_POOL_CONFIG")
        if path and preparing:
            # The container's config path may not exist on the host. Inspect only
            # the candidate's projected reference names; startup checks its actual alias and URL.
            aliases = [
                key[:-8]
                for key in env
                if key.endswith("_OOS_REF")
                and env[key] == marker(m["refs"]["pool_room"])
            ]
            if len(aliases) != 1:
                raise ValueError(
                    "Missing or ambiguous pool_room projection for " + role
                )
            needed["pool_room"] = aliases[0]
        elif path:
            pool = json.loads(Path(path).read_text())
            needed["pool_room"] = pool["room_token_env"]
            if (
                pool["coordinator_url"].rstrip("/") + "/v1/role-pool/capacity"
                != m["audiences"]["pool_room"]
            ):
                raise ValueError(
                    "Pool capacity audience does not match the configured route"
                )
        elif env.get("WHITEBOX_MODEL_BACKEND") == "role_pool":
            raise ValueError("Managed release is missing its Room role-pool config")
        if env.get("WHITEBOX_NATIVE_RTC_ENABLED") == "1" or "rtc" in m["refs"]:
            needed["rtc"] = "WHITEBOX_RTC_AUTH_SECRET"
    if (
        role in ("directory_coordinator", "pool_coordinator")
        and "room_internal" in needed
        and env.get("WHITEBOX_DYNAMIC_ROOMS") == "1"
    ):
        needed["room_admin"] = "WHITEBOX_ROOM_ADMIN_TOKEN"
    if role == "route_publisher":
        needed["pool_room"] = "ADAPTIVE_ROLE_POOL_ROOM_TOKEN"
    if role == "model_publisher":
        needed.update(
            operator="ROLE_POOL_OPERATOR_TOKEN", signing="ROLE_POOL_SIGNING_KEY"
        )
    if role == "pool_coordinator":
        needed.update(
            operator=pool["operator_token_env"], pool_room=pool["room_token_env"]
        )
    if role in ("pool_coordinator", "worker"):
        needed.update(
            registration=pool["role_token_env"], signing=pool["signing_secret_env"]
        )
    if (
        pool is not None
        and (role in ("pool_coordinator", "worker") or "pool_id" in pool)
        and pool.get("pool_id") != m["pool_id"]
    ):
        raise ValueError("Control credential manifest is for a different pool")
    if not needed or len(set(needed.values())) != len(needed):
        raise ValueError(
            "Invalid control credential consumer or aliased credential names"
        )
    problems = []
    for purpose, key in needed.items():
        ref = m["refs"].get(purpose)
        if (
            not ref
            or not env.get(key, "").strip()
            or env.get(key + "_OOS_REF") != marker(ref)
        ) or (
            role == "route_publisher"
            and env.get(key + "_AUDIENCE") != m["audiences"][purpose]
        ):
            problems.append(purpose)
    if problems:
        raise ValueError(
            "Unbound or mismatched managed references for "
            + role
            + ": "
            + ", ".join(sorted(problems))
        )
    return manifest_digest(m)
