# SPDX-License-Identifier: Apache-2.0
"""Session-scoped prompt intent, independent of model execution queues.

The model owner publishes a block before submitting it. HTTP admission takes
the same short lock to reserve its clip boundary; neither path holds the lock
while running a model or reading storage. Source buffers remain model-owned.
"""

import secrets
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy


class PromptControl:
    def __init__(self, query, *, clip_frames=17):
        self.query = query
        self.clip_frames = clip_frames
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prompt-query")
        self.owner = self.token = self.context = self.pending = None
        self.submitted_end = 0
        self.revision = 0

    def open(self, owner):
        with self.lock:
            self._clear()
            self.owner = tuple(owner)
            self.token = secrets.token_urlsafe(32)
            return self.token

    def _clear(self):
        if self.pending is not None:
            self.pending["future"].cancel()
            if completion := self.pending.get("completion"):
                completion.cancel()
        self.owner = self.token = self.context = self.pending = None
        self.submitted_end = self.revision = 0

    def release(self):
        with self.lock:
            self._clear()

    def close(self):
        self.release()
        self.pool.shutdown(wait=True, cancel_futures=True)

    def authenticate(self, owner, token):
        if (
            self.owner != tuple(owner)
            or not isinstance(token, str)
            or self.token is None
            or not secrets.compare_digest(self.token, token)
        ):
            raise ValueError("Prompt control session is unavailable")

    def publish_context(self, context):
        # Only the model owner publishes a successfully adopted condition.
        with self.lock:
            self.context = deepcopy(context)

    def begin(self, owner, token, command, *, routed=False):
        with self.lock:
            self.authenticate(owner, token)
            version = command.get("version")
            if type(version) is not int or version <= self.revision:
                raise ValueError("Prompt version must increase")
            if self.context is None or self.pending is not None:
                raise ValueError("Prompt control is not ready")
            boundary = (
                (max(self.submitted_end, self.clip_frames) + self.clip_frames - 1)
                // self.clip_frames
                * self.clip_frames
            )
            ticket = secrets.token_hex(16)
            context = deepcopy(self.context)
            command = deepcopy(command)
            self.pending = dict(
                ticket=ticket,
                boundary=boundary,
                command=command,
                future=self.pool.submit(self.query, context, command),
                candidate=None,
                delivered=False,
                routed=routed,
            )
            self.revision = version
            return dict(ticket=ticket, boundary=boundary, version=version)

    def _selection(self, ticket):
        if self.pending is None or self.pending["ticket"] != ticket:
            raise ValueError("Prompt preparation is unavailable")
        return self.pending

    def result(self, owner, token, ticket, *, timeout=120):
        with self.lock:
            self.authenticate(owner, token)
            selected = self._selection(ticket)
            future = selected["future"]
        value = future.result(timeout=timeout)
        with self.lock:
            self.authenticate(owner, token)
            if self._selection(ticket) is not selected:
                raise ValueError("Prompt preparation was replaced")
        return value

    def route(self, owner, token, command):
        receipt = self.begin(owner, token, command, routed=True)
        ticket = receipt["ticket"]
        try:
            result = self.result(owner, token, ticket)
            if result["hit"]:
                return dict(receipt, action="adopt")
            miss = {"cache_key": result["cache_key"], "lookup": result["lookup"]}
            return dict(receipt, action="prepare", miss=miss)
        except BaseException:
            self.cancel(owner, token, ticket)
            raise

    def adopt(self, owner, token, ticket, event_id, defer):
        """Only the model owner may resolve this acknowledgement, at the fence."""
        with self.lock:
            self.authenticate(owner, token)
            selected = self._selection(ticket)
            snapshot = selected["future"].result(timeout=0)
            if (
                not selected["routed"]
                or not snapshot["hit"]
                or selected.get("completion") is not None
                or selected["candidate"] is not None
                or selected["delivered"]
                or type(event_id) is not int
                or event_id < 1
                or type(defer) is not bool
            ):
                raise ValueError("Invalid local condition adoption")
            completion = selected["completion"] = Future()
            selected["candidate"] = dict(
                type="conditioning",
                session_id=owner[0],
                generation_id=owner[1],
                event_id=event_id,
                prompt_version=selected["command"]["version"],
                source_boundary=selected["boundary"],
                allow_deferred_kv=defer,
                prompt_preparation=ticket,
                direct_hit=True,
                cache_hit=True,
                cache_key=snapshot["cache_key"],
                cache_lookup=snapshot["lookup"],
                conditioning_identity=snapshot.get("conditioning_identity"),
                cache_query_context=snapshot["cache_query_context"],
            )
        return completion.result(timeout=120)

    def validate(self, ticket, boundary, version):
        with self.lock:
            selected = self._selection(ticket)
            if (boundary, version) != (
                selected["boundary"],
                selected["command"]["version"],
            ):
                raise ValueError("Prompt preparation identity does not match")

    def prepared(self, ticket):
        with self.lock:
            future = self._selection(ticket)["future"]
        return future.result(timeout=0)

    def submit_candidate(self, owner, token, ticket, candidate):
        with self.lock:
            self.authenticate(owner, token)
            selected = self._selection(ticket)
            if selected["candidate"] is not None or selected["delivered"]:
                raise ValueError("Prompt candidate was already submitted")
            if (
                candidate.get("prompt_version") != selected["command"]["version"]
                or candidate.get("source_boundary") != selected["boundary"]
                or candidate.get("prompt_preparation") != ticket
                or candidate.get("type") != "conditioning"
            ):
                raise ValueError("Prompt candidate identity does not match")
            if candidate.get("direct_hit"):
                raise ValueError("Only the cache owner can adopt a local hit")
            if selected["routed"]:
                decision = selected["future"].result(timeout=0)
                if (
                    decision["hit"]
                    or candidate.get("cache_hit")
                    or decision["cache_key"] != candidate.get("cache_key")
                ):
                    raise ValueError("Candidate does not match the cache miss")
            selected["candidate"] = candidate

    def take_candidate(self):
        with self.lock:
            if self.pending is None:
                return None
            candidate = self.pending["candidate"]
            if candidate is not None:
                self.pending["candidate"] = None
                self.pending["delivered"] = True
            return candidate

    def start_block(self, pixel_end):
        """Called before popping source or submitting any work for this block."""
        with self.lock:
            if self.pending is not None and pixel_end > self.pending["boundary"]:
                return False
            if pixel_end < self.submitted_end:
                raise ValueError("DiT execution position cannot move backwards")
            self.submitted_end = pixel_end
            return True

    def finish(self, ticket, *, context=None, outcome=None):
        with self.lock:
            selected = self._selection(ticket)
            if context is not None:
                self.context = deepcopy(context)
            if completion := selected.get("completion"):
                completion.set_result(outcome)
            selected["future"].cancel()
            self.pending = None

    def cancel(self, owner, token, ticket):
        with self.lock:
            self.authenticate(owner, token)
            if self.pending is None:
                return
            selected = self._selection(ticket)
            if selected["delivered"]:
                return  # The model owner is already responsible for finishing.
            if completion := selected.get("completion"):
                completion.cancel()
            selected["future"].cancel()
            self.pending = None
