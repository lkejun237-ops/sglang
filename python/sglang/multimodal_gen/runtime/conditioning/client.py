# SPDX-License-Identifier: Apache-2.0
"""One bounded authenticated RPC, with no session or artifact ownership."""

import asyncio
import concurrent.futures
import ipaddress
import math
import os
import random
import struct
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx

from .contract import ConditionError, canonical_json, safe_json
from .inputs import validate_condition_media, validate_prompt


def _is_loopback_host(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_service_url(url):
    """Return the normalized URL; plaintext HTTP is accepted for loopback only."""
    parts = host = None
    if isinstance(url, str):
        try:
            parts = urlsplit(url)
            host = parts.hostname
            parts.port  # raises ValueError on a malformed port
        except ValueError:
            parts = None
    if (
        parts is None
        or not host
        or parts.username is not None
        or parts.password is not None
        or parts.scheme not in ("http", "https")
        or (parts.scheme == "http" and not _is_loopback_host(host))
    ):
        raise ConditionError(
            "CONDITION_INVALID",
            "A conditioning service URL must be https, or http on a loopback host",
        )
    return url.rstrip("/")


class _BoundedDNSExecutor(concurrent.futures.ThreadPoolExecutor):
    """Fixed shared resolver capacity; expired lookups cannot grow its queue."""

    def __init__(self):
        super().__init__(max_workers=2, thread_name_prefix="conditioning-dns")
        self.capacity = threading.BoundedSemaphore(4)

    def submit(self, function, /, *args, **kwargs):
        if not self.capacity.acquire(blocking=False):
            raise ConditionError("CONDITION_UNAVAILABLE")
        public = concurrent.futures.Future()

        def execute():
            # Cancelling asyncio's public future must not cancel the dispatch
            # Future: ThreadPoolExecutor leaves cancelled work in its queue.
            # Keep its permit until a worker dequeues it (or shutdown removes
            # it). Cancelled queued lookups then skip the actual OS resolver.
            if public.set_running_or_notify_cancel():
                return function(*args, **kwargs)

        try:
            dispatch = super().submit(execute)
        except BaseException:
            self.capacity.release()
            raise

        def complete(future):
            try:
                if future.cancelled():
                    public.cancel()
                elif not public.cancelled():
                    try:
                        result = future.result()
                    except BaseException as error:
                        public.set_exception(error)
                    else:
                        public.set_result(result)
            finally:
                self.capacity.release()

        dispatch.add_done_callback(complete)
        return public


class _NetworkLoop:
    """One process-wide I/O loop for synchronous workers, never a thread per RPC.

    A synchronous worker may already be inside an asyncio loop. A dedicated,
    fixed loop also lets cancelled DNS operations drain within a bounded shared
    resolver, without blocking a worker in asyncio.run's executor shutdown.
    """

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.loop.set_default_executor(_BoundedDNSExecutor())
        self.capacity = threading.BoundedSemaphore(32)
        self.thread = threading.Thread(
            target=self.loop.run_forever, name="conditioning-http", daemon=True
        )
        self.thread.start()

    def run(self, function, timeout):
        if not self.capacity.acquire(blocking=False):
            raise ConditionError("CONDITION_BUSY")
        try:
            future = asyncio.run_coroutine_threadsafe(function(), self.loop)
        except BaseException:
            self.capacity.release()
            raise
        # Keep admission until coroutine cleanup really ends, even if its sync
        # caller times out. Cancelling the concurrent future alone is not proof
        # that the socket/async task has reached its terminal state.
        try:
            return future.result(timeout=timeout)
        except (TimeoutError, concurrent.futures.TimeoutError):
            # The coroutine has the same absolute deadline. Let its timeout
            # cancel and close the transport, retaining capacity until cleanup;
            # cancelling before its first step could skip the owned finally.
            raise ConditionError("CONDITION_TIMEOUT") from None

    async def owned(self, function):
        try:
            return await function()
        finally:
            self.capacity.release()


_network_lock = threading.Lock()
_network = None
_network_pid = None


def _network_loop():
    global _network, _network_pid
    with _network_lock:
        if _network is None or _network_pid != os.getpid():
            _network, _network_pid = _NetworkLoop(), os.getpid()
        return _network


class ConditioningClient:
    def __init__(
        self,
        url,
        token,
        profile_id,
        *,
        timeout_s=10,
        sp_timeout_s=None,
        max_rows=16384,
        max_request_bytes=536870912,
        max_bytes=268435456,
        media_limits=None,
        retry_initial_s=0.05,
        retry_max_s=0.5,
        transport=None,
        clock=time.monotonic,
        sleep=time.sleep,
        jitter=random.random,
    ):
        url = validate_service_url(url)
        if type(token) is not str or not token or any(c in token for c in "\r\n"):
            raise ConditionError(
                "CONDITION_INVALID", "A conditioning bearer token is required"
            )
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ConditionError(
                "CONDITION_INVALID", "Conditioning timeout must be positive and finite"
            )
        if sp_timeout_s is not None and timeout_s >= sp_timeout_s:
            raise ConditionError(
                "CONDITION_INVALID",
                "Conditioning timeout must be below the SP CPU group timeout",
            )
        if (
            retry_initial_s <= 0
            or retry_max_s < retry_initial_s
            or not math.isfinite(retry_initial_s)
            or not math.isfinite(retry_max_s)
        ):
            raise ConditionError(
                "CONDITION_INVALID", "Invalid conditioning retry intervals"
            )
        if any(
            type(value) is not int or value <= 0
            for value in (max_rows, max_request_bytes, max_bytes)
        ):
            raise ConditionError(
                "CONDITION_INVALID", "Conditioning byte and row limits must be positive"
            )
        self.url, self.token, self.profile_id = url, token, profile_id
        self.timeout_s, self.max_rows, self.max_bytes = timeout_s, max_rows, max_bytes
        self.max_request_bytes = max_request_bytes
        self.media_limits = dict(media_limits or {})
        validate_condition_media([], **self.media_limits)
        self.retry_initial_s, self.retry_max_s = retry_initial_s, retry_max_s
        self.clock, self.sleep, self.jitter = clock, sleep, jitter
        self.transport = transport or self._transport

    def _transport(self, method, path, body, timeout):
        deadline = self.clock() + timeout
        network = _network_loop()
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise ConditionError("CONDITION_TIMEOUT")
        return network.run(
            lambda: network.owned(
                lambda: self._async_transport(method, path, body, deadline)
            ),
            remaining,
        )

    async def _async_transport(self, method, path, body, deadline):
        headers = {"Authorization": "Bearer " + self.token}
        if body is not None:
            headers.update(
                {
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(body)),
                }
            )
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise ConditionError("CONDITION_TIMEOUT")
        try:
            # Cancellation interrupts connection, upload, status/header parsing
            # and body reads under one deadline; socket inactivity timeouts are
            # only an additional bound. Environment proxies do not expand the
            # administrator-configured endpoint's trust boundary.
            # wait_for is supported on Python 3.10 and waits for cancellation
            # cleanup; the synchronous deadline returns independently while
            # network.owned retains admission through that cleanup.
            return await asyncio.wait_for(
                self._async_exchange(method, path, body, headers, deadline, remaining),
                timeout=remaining,
            )
        except (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException):
            raise ConditionError("CONDITION_TIMEOUT") from None

    async def _async_exchange(self, method, path, body, headers, deadline, remaining):
        async with httpx.AsyncClient(
            timeout=remaining, follow_redirects=False, trust_env=False
        ) as client:
            async with client.stream(
                method, self.url + path, content=body, headers=headers
            ) as response:
                if 300 <= response.status_code < 400:
                    raise ConditionError(
                        "CONDITION_INVALID", "Conditioning redirects are not allowed"
                    )
                length = response.headers.get("Content-Length")
                if length is not None:
                    try:
                        size = int(length)
                    except ValueError:
                        raise ConditionError(
                            "CONDITION_INVALID", "Invalid conditioning response length"
                        ) from None
                    if not 0 <= size <= self.max_bytes:
                        raise ConditionError(
                            "CONDITION_INVALID",
                            "Conditioning response exceeds the byte limit",
                        )
                parts, size = [], 0
                # RPC framing is raw safetensors, never HTTP content decoding.
                # Reading raw chunks also bounds bytes before an optional
                # Content-Encoding could expand an untrusted response.
                async for chunk in response.aiter_raw():
                    if self.clock() >= deadline:
                        raise ConditionError("CONDITION_TIMEOUT")
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise ConditionError(
                            "CONDITION_INVALID",
                            "Conditioning response exceeds the byte limit",
                        )
                    parts.append(chunk)
                if self.clock() >= deadline:
                    raise ConditionError("CONDITION_TIMEOUT")
                return SimpleNamespace(
                    status=response.status_code, body=b"".join(parts)
                )

    def _request(self, method, path, body, deadline):
        last, delay = None, self.retry_initial_s
        while True:
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise last or ConditionError("CONDITION_TIMEOUT")
            try:
                outgoing = body(remaining) if callable(body) else body
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise ConditionError("CONDITION_TIMEOUT")
                response = self.transport(method, path, outgoing, remaining)
                if self.clock() >= deadline:
                    raise ConditionError("CONDITION_TIMEOUT")
                if response.status == 200:
                    return response.body
                codes = {
                    400: "CONDITION_INVALID",
                    401: "CONDITION_UNAUTHORIZED",
                    409: "CONDITION_PROFILE_MISMATCH",
                    413: "CONDITION_INVALID",
                    429: "CONDITION_BUSY",
                    503: "CONDITION_UNAVAILABLE",
                    504: "CONDITION_TIMEOUT",
                }
                error = ConditionError(
                    codes.get(response.status, "CONDITION_UNAVAILABLE")
                )
                if response.status not in (429, 503):
                    raise error
                last = error
            except ConditionError:
                raise
            except (TimeoutError, httpx.RequestError, ConnectionError, OSError):
                last = ConditionError("CONDITION_UNAVAILABLE")
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise last
            self.sleep(min(delay * (0.5 + self.jitter()), remaining))
            delay = min(delay * 2, self.retry_max_s)

    def handshake(self):
        body = self._request(
            "GET", "/v1/conditioning/profile", None, self.clock() + self.timeout_s
        )
        document = safe_json(body)
        if type(document) is not dict or document.get("profile_id") != self.profile_id:
            raise ConditionError(
                "CONDITION_PROFILE_MISMATCH", "Conditioning deployment profile differs"
            )

    def validate_input(self, prompt, media):
        """Check metadata before hashing, cloning tensors or serializing RGB."""
        validate_prompt(prompt)
        items = validate_condition_media(media, **self.media_limits)
        # Even without either JSON header, raw RGB plus the two length prefixes
        # cannot fit a smaller body. The actual serialized body is checked too.
        if sum(item["frames"].numel() for item in items) + 12 > self.max_request_bytes:
            raise ConditionError(
                "CONDITION_INVALID", "Conditioning request exceeds the byte limit"
            )

    def encode(self, prompt, media, *, input_digest):
        from .wire import decode_response, encode_request

        deadline = self.clock() + self.timeout_s
        self.validate_input(prompt, media)
        body = encode_request(
            prompt,
            media,
            self.profile_id,
            int(self.timeout_s * 1000),
            input_digest=input_digest,
        )
        if len(body) > self.max_request_bytes:
            raise ConditionError(
                "CONDITION_INVALID", "Conditioning request exceeds the byte limit"
            )
        header_size = struct.unpack(">I", body[:4])[0]
        header = safe_json(body[4 : 4 + header_size])
        tensors = memoryview(body)[4 + header_size :]

        def remaining_request(remaining):
            # Serialization has already spent part of the total budget. Every
            # retry gives the service only the currently remaining time.
            current = dict(header, deadline_ms=max(1, int(remaining * 1000)))
            encoded = canonical_json(current)
            return b"".join((struct.pack(">I", len(encoded)), encoded, tensors))

        payload = self._request(
            "POST", "/v1/conditioning/encode", remaining_request, deadline
        )
        result = decode_response(
            payload,
            self.profile_id,
            input_digest,
            max_rows=self.max_rows,
            max_bytes=self.max_bytes,
        )
        if self.clock() >= deadline:
            raise ConditionError("CONDITION_TIMEOUT")
        return result
