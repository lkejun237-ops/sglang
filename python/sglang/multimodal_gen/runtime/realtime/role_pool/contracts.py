# SPDX-License-Identifier: Apache-2.0
"""CPU-only wire validation and bounded control credentials."""

import base64
import hashlib
import hmac
import json
import math
import time
from urllib.parse import urlsplit

ROLES = ("encoder", "dit", "decoder")
TERMINAL = {"released", "cancelled"}


class PoolError(ValueError):
    def __init__(self, code, status=409):
        self.code, self.status = code, status
        super().__init__(code)


def identifier(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 256:
        raise PoolError("INVALID_IDENTITY", 422)
    return value


def number(value, *, positive=True):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise PoolError("INVALID_CAPACITY", 422)
    if value < 0 or positive and value == 0:
        raise PoolError("INVALID_CAPACITY", 422)
    return float(value)


def vector(value):
    if not isinstance(value, dict) or not value or len(value) > 16:
        raise PoolError("INVALID_BUDGET", 422)
    return {identifier(k): number(v) for k, v in value.items()}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def descriptor(value, role=None):
    if not isinstance(value, dict) or set(value) != {
        "worker_id",
        "service_id",
        "role",
        "endpoints",
    }:
        raise PoolError("INVALID_SERVICE", 422)
    identifier(value["worker_id"])
    identifier(value["service_id"])
    if value["role"] not in ROLES or role is not None and value["role"] != role:
        raise PoolError("INVALID_ROLE", 422)
    names = {
        "encoder": {"http", "input_ws", "rendezvous"},
        "dit": {"http", "zmq", "rendezvous"},
        "decoder": {"http", "zmq", "output_ws"},
    }[value["role"]]
    if not isinstance(value["endpoints"], dict) or set(value["endpoints"]) != names:
        raise PoolError("INVALID_ENDPOINTS", 422)
    for name, endpoint in value["endpoints"].items():
        try:
            uri = urlsplit(endpoint)
            schemes = (
                {"http", "https"}
                if name == "http"
                else {"ws", "wss"}
                if name.endswith("ws")
                else {"tcp"}
            )
            if (
                uri.scheme not in schemes
                or not uri.hostname
                or not uri.port
                or uri.username
                or uri.password
                or uri.query
                or uri.fragment
            ):
                raise ValueError()
        except (TypeError, ValueError):
            raise PoolError("INVALID_ENDPOINT", 422) from None
    return value


def manifest(value):
    value = json.loads(canonical(value))
    required = {
        "process_id",
        "boot_epoch",
        "control_epoch",
        "pool_id",
        "role",
        "compatibility",
        "network_domain",
        "capacity",
        "profiles",
        "services",
        "hardware_ids",
        "profile_limits",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise PoolError("INVALID_PROCESS_MANIFEST", 422)
    for name in required - {
        "capacity",
        "profiles",
        "services",
        "hardware_ids",
        "profile_limits",
    }:
        identifier(value[name])
    if value["role"] not in ROLES:
        raise PoolError("INVALID_ROLE", 422)
    value["capacity"] = vector(value["capacity"])
    if (
        not isinstance(value["profiles"], dict)
        or not 1 <= len(value["profiles"]) <= 128
    ):
        raise PoolError("INVALID_PROFILES", 422)
    for name, costs in value["profiles"].items():
        identifier(name)
        costs = vector(costs)
        if set(costs) != set(value["capacity"]) or any(
            costs[k] > value["capacity"][k] for k in costs
        ):
            raise PoolError("PROFILE_EXCEEDS_CAPACITY", 422)
        value["profiles"][name] = costs
    if set(value["profile_limits"]) != set(value["profiles"]):
        raise PoolError("PROFILE_LIMITS_MISSING", 422)
    for limits in value["profile_limits"].values():
        profile_limits(limits)
    services = value["services"]
    if not isinstance(services, list) or not 1 <= len(services) <= 128:
        raise PoolError("INVALID_SERVICE_COUNT", 422)
    if value["role"] == "dit" and len(services) != 1:
        raise PoolError("DIT_REQUIRES_ONE_SERVICE", 422)
    ids = []
    for service in services:
        descriptor(service, value["role"])
        if service["worker_id"] != value["process_id"]:
            raise PoolError("PROCESS_SERVICE_MISMATCH", 422)
        ids.append(service["service_id"])
    if len(set(ids)) != len(ids):
        raise PoolError("DUPLICATE_SERVICE", 422)
    hardware = value["hardware_ids"]
    if (
        not isinstance(hardware, list)
        or not hardware
        or len(set(hardware)) != len(hardware)
    ):
        raise PoolError("INVALID_HARDWARE", 422)
    for identity in hardware:
        identifier(identity)
    return value


class Credentials:
    """Trusted Coordinator signs each operation; executors cannot extend leases."""

    def __init__(self, secret, control_epoch, *, clock=time.time):
        if not isinstance(secret, str) or len(secret) < 32:
            raise ValueError(
                "Role pool signing secret must contain at least 32 characters"
            )
        self.secret = secret.encode()
        self.epoch, self.clock = identifier(control_epoch), clock

    def sign(self, action, body, *, expires_at):
        now = self.clock()
        if not now < expires_at <= now + 120:
            raise PoolError("INVALID_CONTROL_DEADLINE")
        raw = canonical(
            dict(epoch=self.epoch, action=action, body=digest(body), exp=expires_at)
        ).encode()
        payload = base64.urlsafe_b64encode(raw).decode().rstrip("=")
        signature = hmac.new(self.secret, payload.encode(), hashlib.sha256).hexdigest()
        return payload + "." + signature

    def verify(self, token, action, body):
        try:
            payload, signature = token.split(".")
            expected = hmac.new(
                self.secret, payload.encode(), hashlib.sha256
            ).hexdigest()
            claims = json.loads(
                base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
            )
            if (
                not hmac.compare_digest(expected, signature)
                or claims["epoch"] != self.epoch
                or claims["action"] != action
                or claims["body"] != digest(body)
                or not self.clock() < claims["exp"] <= self.clock() + 125
            ):
                raise ValueError()
            return claims
        except (ValueError, TypeError, KeyError, AttributeError):
            raise PoolError("CONTROL_AUTHORIZATION_FAILED", 403) from None


def profile_limits(value):
    required = {
        "source_sizes",
        "target_sizes",
        "max_fps",
        "max_references",
        "max_reference_bytes",
        "max_reference_pixels",
        "max_prompt_chars",
        "max_preview_pixels",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise PoolError("INVALID_PROFILE_LIMITS", 422)
    for name in ("source_sizes", "target_sizes"):
        if (
            not isinstance(value[name], list)
            or not value[name]
            or len(value[name]) > 16
        ):
            raise PoolError("INVALID_PROFILE_GEOMETRY", 422)
        for size in value[name]:
            if (
                not isinstance(size, list)
                or len(size) != 2
                or any(type(n) is not int or not 1 <= n <= 8192 for n in size)
            ):
                raise PoolError("INVALID_PROFILE_GEOMETRY", 422)
    for name in required - {"source_sizes", "target_sizes"}:
        number(value[name], positive=name != "max_references")
    return value


def validate_business_message(value, limits):
    """Enforce the measured allocation envelope before consuming shared GPU work."""
    kind = value.get("type")
    update = value.get("kind") in {"prompt.update", "prompt"}
    if kind not in {"init", "prefill"} and not update:
        return
    if kind in {"init", "prefill"}:
        source = value.get("source_stream", {})
        if [source.get("width"), source.get("height")] not in limits["source_sizes"]:
            raise PoolError("PROFILE_SOURCE_SIZE_EXCEEDED", 422)
        size = value.get("size", "")
        try:
            target = [int(n) for n in size.lower().split("x")]
        except (AttributeError, ValueError):
            raise PoolError("INVALID_TARGET_SIZE", 422) from None
        if (
            target not in limits["target_sizes"]
            or source.get("fps", 0) > limits["max_fps"]
        ):
            raise PoolError("PROFILE_TARGET_EXCEEDED", 422)
        if kind == "prefill":
            preview = value.get("source_frame_size") or source
            if (
                preview.get("width", 0) * preview.get("height", 0)
                > limits["max_preview_pixels"]
            ):
                raise PoolError("PROFILE_PREVIEW_EXCEEDED", 422)
    payload = value.get("payload") if update else value
    prompt = (
        payload
        if isinstance(payload, str)
        else payload.get("text", payload.get("prompt", ""))
    )
    if not isinstance(prompt, str) or len(prompt) > limits["max_prompt_chars"]:
        raise PoolError("PROFILE_PROMPT_EXCEEDED", 422)
    refs = (
        payload.get("references", [])
        if isinstance(payload, dict)
        else value.get("references", [])
    )
    if not isinstance(refs, list) or len(refs) > limits["max_references"]:
        raise PoolError("PROFILE_REFERENCES_EXCEEDED", 422)
    pixels = total = 0
    for ref in refs:
        data = ref.get("payload", b"")
        if not isinstance(data, bytes):
            raise PoolError("INVALID_REFERENCE", 422)
        total += len(data)
        if total > limits["max_reference_bytes"]:
            raise PoolError("PROFILE_REFERENCE_BYTES_EXCEEDED", 422)
        import io

        from PIL import Image

        try:
            with Image.open(io.BytesIO(data)) as image:
                pixels += image.width * image.height
        except Exception:
            raise PoolError("INVALID_REFERENCE", 422) from None
        if pixels > limits["max_reference_pixels"]:
            raise PoolError("PROFILE_REFERENCE_PIXELS_EXCEEDED", 422)
