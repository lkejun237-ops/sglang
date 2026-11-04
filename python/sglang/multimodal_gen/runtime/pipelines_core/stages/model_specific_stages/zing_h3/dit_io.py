# SPDX-License-Identifier: Apache-2.0
"""DiT-only communication owner, independent of the model consumption loop."""

import time
from concurrent.futures import Future, TimeoutError
from contextlib import nullcontext
from queue import Empty, Full, Queue
from threading import Event, Thread


class DiTIOChannel:
    """Own a transport on one thread and hand complete messages to the model.

    The factory, socket operations and transport teardown all run on the I/O
    thread. The model thread alone reads the inbox, submits replies and closes
    the owner. A full inbox pauses reception, without blocking outbound replies.
    The existing Encoder request/reply contract is unchanged.
    """

    def __init__(
        self,
        factory,
        *,
        max_pending=10,
        poll_interval_s=0.005,
        on_failure=None,
        cleanup_timeout=None,
    ):
        if max_pending < 1 or poll_interval_s <= 0:
            raise ValueError("DiT I/O queue and polling interval must be positive")
        self.factory = factory
        self.on_failure = on_failure
        self.cleanup_timeout = cleanup_timeout
        self.inbox = Queue(maxsize=max_pending)
        self.outbox = Queue(maxsize=max_pending)
        self.poll_interval_s = poll_interval_s
        self.stopping = Event()
        self.pause_requested = Event()
        self.pause_ack = Event()
        self.ready = Future()
        self.error = self.close_error = None
        self.transport = self.thread = None

    def start(self):
        if self.thread is not None:
            raise RuntimeError("DiT I/O loop has already started")
        self.thread = Thread(target=self._run, name="zing-dit-io", daemon=True)
        self.thread.start()
        self.ready.result()
        return self

    def _check(self):
        if self.error is not None:
            raise RuntimeError("DiT communication loop failed") from self.error
        if self.stopping.is_set() or self.thread is None:
            raise RuntimeError("DiT communication loop is not running")

    def poll(self):
        self._check()
        return not self.inbox.empty()

    def receive(self):
        self._check()
        deadline = time.monotonic() + 120
        while True:
            self._check()
            try:
                message = self.inbox.get(timeout=self.poll_interval_s)
                break
            except Empty:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Worker reply timed out")
        # CUDA receive completion and allocator ownership cross thread/stream
        # boundaries here. No socket operation runs on the model thread.
        adopt = getattr(self.transport, "adopt_received", None)
        if adopt is not None:
            adopt(message)
        return message

    def send(self, message, *, on_wait=None):
        self._check()
        prepare = getattr(self.transport, "prepare_send", None)
        if prepare is not None:
            prepare(message)
        done = Future()
        while True:
            self._check()
            try:
                self.outbox.put((message, done), timeout=self.poll_interval_s)
                break
            except Full:
                if on_wait is not None:
                    on_wait()
        # The I/O thread completes or fails every accepted reply. Reception
        # remains independent even while the model thread waits for backpressure.
        while True:
            try:
                return done.result(timeout=self.poll_interval_s)
            except TimeoutError:
                # A transport failure can race the final outbox put. Do not
                # wait forever if the I/O owner has already drained and exited.
                self._check()
                if on_wait is not None:
                    on_wait()

    def _run(self):
        pending = None
        try:
            self.transport = self.factory()
            context = getattr(self.transport, "io_context", nullcontext)
            with context():
                self.ready.set_result(None)
                while not self.stopping.is_set():
                    if self.pause_requested.is_set():
                        self.pause_ack.set()
                    try:
                        message, pending = self.outbox.get_nowait()
                    except Empty:
                        pass
                    else:
                        self.transport.send(message)
                        self._finish_transfer(message)
                        pending.set_result(None)
                        pending = None
                        continue
                    if (
                        not self.pause_requested.is_set()
                        and not self.inbox.full()
                        and self.transport.poll()
                    ):
                        message = self.transport.receive()
                        self._finish_transfer(message)
                        self.inbox.put_nowait(message)
                    else:
                        self.stopping.wait(self.poll_interval_s)
        except BaseException as exc:  # noqa: BLE001 - propagate thread failure to owner
            self.error = exc
        finally:
            self.stopping.set()
            error = self.error or RuntimeError("DiT communication loop stopped")
            if not self.ready.done():
                self.ready.set_exception(error)
            if pending is not None:
                pending.set_exception(error)
            while True:
                try:
                    _, done = self.outbox.get_nowait()
                except Empty:
                    break
                done.set_exception(error)
            from sglang.multimodal_gen.runtime.realtime.cleanup_watchdog import (
                CleanupWatchdog,
            )

            guard = (
                CleanupWatchdog(self.cleanup_timeout)
                if self.cleanup_timeout is not None
                else nullcontext()
            )
            with guard:
                # Notify before native shutdown: shutdown itself may hang.
                if self.error is not None and self.on_failure is not None:
                    self.on_failure()
                if self.transport is not None:
                    try:
                        if self.error is not None:
                            abort = getattr(self.transport, "abort", None)
                            if abort is not None:
                                abort()
                        self.transport.close()
                    except BaseException as exc:  # noqa: BLE001 - preserve owner failure
                        self.close_error = exc
                        self.error = self.error or exc
                        if self.on_failure is not None:
                            self.on_failure()
                        raise

    def _finish_transfer(self, message):
        finish = getattr(self.transport, "finish_transfer", None)
        if finish is not None:
            finish(message)

    def pause_receiving(self):
        self.pause_requested.set()
        while not self.pause_ack.wait(self.poll_interval_s):
            self._check()

    def abort(self):
        try:
            if self.transport is not None:
                abort = getattr(self.transport, "abort", None)
                if abort is not None:
                    abort()
        finally:
            # Waking a healthy I/O loop first would let its normal close path
            # claim graceful shutdown before the transport publishes abort.
            self.stopping.set()

    def close(self, *, fault=False):
        from sglang.multimodal_gen.runtime.realtime.cleanup_watchdog import (
            CleanupWatchdog,
        )

        guard = (
            CleanupWatchdog(self.cleanup_timeout)
            if self.cleanup_timeout is not None
            else nullcontext()
        )
        with guard:
            if fault:
                self.abort()
            self.stopping.set()
            if self.thread is not None:
                self.thread.join()
            if self.close_error is not None:
                raise RuntimeError(
                    "Worker transport cleanup failed"
                ) from self.close_error
            while True:
                try:
                    self.inbox.get_nowait()
                except Empty:
                    break
