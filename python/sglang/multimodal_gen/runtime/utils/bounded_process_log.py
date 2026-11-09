# SPDX-License-Identifier: Apache-2.0
"""Drain child diagnostics independently from bounded, best-effort disk logs."""

import errno
import logging
import logging.handlers
import queue
import threading


def diagnostic_print(*args, **kwargs):
    """A full diagnostics volume must not turn successful model work into failure."""
    try:
        print(*args, **kwargs)
        return True
    except OSError as exc:
        if exc.errno not in {errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EPIPE}:
            raise
        return False


class BoundedProcessLog:
    def __init__(self, path, *, max_bytes=64 << 20, backups=3):
        self.handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backups, delay=True, encoding="utf-8"
        )
        self.handler.setFormatter(logging.Formatter("%(message)s"))
        self.queue = queue.Queue(maxsize=128)
        self.stopped = threading.Event()
        self.reader = None
        self.dropped = self.write_errors = 0
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.writer.start()

    def attach(self, stream):
        def read():
            try:
                while True:
                    chunk = stream.readline(16384)
                    if not chunk:
                        break
                    try:
                        self.queue.put_nowait(
                            chunk.decode(errors="replace").rstrip("\r\n")
                        )
                    except queue.Full:
                        self.dropped += 1
            finally:
                stream.close()

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def _write(self):
        while not self.stopped.is_set() or not self.queue.empty():
            try:
                line = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                # Call the IO directly: logging's default error handler can
                # write another traceback to the same full disk/stderr.
                if self.handler.stream is None:
                    self.handler.stream = self.handler._open()
                encoded = line.encode("utf-8")
                if len(encoded) >= self.handler.maxBytes:
                    line = encoded[: self.handler.maxBytes - 1].decode(
                        "utf-8", errors="ignore"
                    )
                    self.dropped += 1
                if (
                    self.handler.stream.tell() + len(line.encode("utf-8")) + 1
                    > self.handler.maxBytes
                ):
                    self.handler.doRollover()
                    if self.handler.stream is None:
                        self.handler.stream = self.handler._open()
                self.handler.stream.write(line + "\n")
                self.handler.flush()
            except (OSError, ValueError):
                self.write_errors += 1
                stream, self.handler.stream = self.handler.stream, None
                if stream is not None:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass
            finally:
                self.queue.task_done()
        try:
            self.handler.close()
        except (OSError, ValueError):
            pass

    def close(self):
        if self.reader is not None:
            self.reader.join(timeout=2)
        self.stopped.set()
        self.writer.join(timeout=2)
