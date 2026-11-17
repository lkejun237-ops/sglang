# SPDX-License-Identifier: Apache-2.0
"""Append-only cache operation receipts beside a signed Room video archive.

This metadata writer is independent of model leases and video finalization, so a
late background write can be recorded after the player has released its Slot.
No endpoint or bucket is accepted from a client; those use regional server policy.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath
from uuid import uuid4


def secret():
    value = os.environ.get("WHITEBOX_CACHE_RECORDING_SECRET") or os.environ.get(
        "WHITEBOX_INTERNAL_TOKEN", ""
    )
    return value.encode() if len(value.encode()) >= 32 else None


def valid_prefix(prefix):
    return (
        isinstance(prefix, str)
        and len(prefix) <= 1024
        and prefix.startswith("white_box_game_record/")
        and ".." not in PurePosixPath(prefix).parts
        and "\\" not in prefix
        and all(ord(c) >= 32 for c in prefix)
    )


def issue_recording(world, prefix):
    key = secret()
    if (
        os.environ.get("WHITEBOX_CACHE_RECORDING_ENABLED") != "1"
        or key is None
        or not valid_prefix(prefix)
    ):
        return None
    payload = json.dumps(
        dict(
            v=1,
            world_id=world,
            prefix=prefix.rstrip("/") + "/cache-operations/",
            expires_at=int(time.time()) + 10800,
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    body = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    return body + "." + hmac.new(key, body.encode(), hashlib.sha256).hexdigest()


def verify_recording(token, world, *, now=None):
    key = secret()
    if key is None or not isinstance(token, str) or len(token) > 4096:
        return None
    try:
        body, signature = token.split(".")
        if not hmac.compare_digest(
            hmac.new(key, body.encode(), hashlib.sha256).hexdigest(), signature
        ):
            return None
        value = json.loads(
            base64.b64decode(
                body + "=" * (-len(body) % 4), altchars=b"-_", validate=True
            )
        )
        now = time.time() if now is None else now
        if (
            value.get("v") != 1
            or value.get("world_id") != world
            or not valid_prefix(value.get("prefix"))
            or not value["prefix"].endswith("/cache-operations/")
            or type(value.get("expires_at")) is not int
            or not now < value["expires_at"] <= now + 10860
        ):
            return None
        return value
    except (ValueError, TypeError, KeyError):
        return None


class CacheObserver:
    def __init__(self, policy, *, objects=None):
        self.policy, self.objects = policy, objects
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cache-video-receipt"
        )
        self.capacity = threading.BoundedSemaphore(32)
        self.closed = False
        self.dropped = self.failed = self.written = 0

    def offer(self, target, event):
        if target is None:
            return False
        if (
            self.closed
            or target["expires_at"] <= time.time()
            or not self.capacity.acquire(blocking=False)
        ):
            self.dropped += 1
            return False
        # Bound serialization and freeze the metadata before the player changes.
        try:
            raw = json.dumps(
                {
                    **event,
                    "schema_version": 1,
                    "observed_epoch_ms": time.time() * 1000,
                    "world_id": target["world_id"],
                },
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
            if len(raw) > 64 << 10:
                raise ValueError("Cache receipt exceeds its metadata budget")
            path = target["prefix"] + uuid4().hex + ".json"
            deadline = time.monotonic() + 120
        except (ValueError, TypeError):
            self.capacity.release()
            self.dropped += 1
            return False

        def write():
            try:
                if time.monotonic() >= deadline:
                    self.dropped += 1
                    return
                if self.objects is None:
                    from .cache_archive import S3CacheObjects
                    from .cache_tiers import CacheTiers

                    # Recording metadata remains enabled when payload L3 is off.
                    validated = CacheTiers.parse({**self.policy.__dict__, "l3": True})
                    self.objects = S3CacheObjects(validated)
                self.objects.put(path, raw)
                self.written += 1
            except Exception:
                self.failed += 1
                logging.getLogger(__name__).warning(
                    "Cache operation recording failed", exc_info=True
                )
            finally:
                self.capacity.release()

        try:
            self.executor.submit(write)
        except RuntimeError:
            self.capacity.release()
            self.dropped += 1
            return False
        return True

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=True)
