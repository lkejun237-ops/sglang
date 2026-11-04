# SPDX-License-Identifier: Apache-2.0
"""A process-external deadline for worker cleanup blocked in native code."""

import math
import os
import subprocess
import sys
import time

# The child never imports CUDA or joins a process group. It can still enforce
# the deadline when the worker's event loop, GIL or NCCL watchdog is blocked.
_WATCH = r"""
import os, select, signal, sys, time
owner, deadline = int(sys.argv[1]), float(sys.argv[2])
remaining = max(0.0, deadline - time.monotonic())
if select.select([0], [], [], remaining)[0]:
    command = os.read(0, 1)
    if command == b'C' or os.getppid() != owner:
        sys.exit(0)
    # Failed cleanup or a lost pipe while the owner is alive is not success.
    while time.monotonic() < deadline:
        if os.getppid() != owner:
            sys.exit(0)
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
if os.getppid() == owner:
    try:
        os.write(2, b'Worker cleanup exceeded its deadline; terminating worker.\n')
    finally:
        os.kill(owner, signal.SIGKILL)
"""


class CleanupWatchdog:
    """Cancel only after cleanup really returned; otherwise terminate owner.

    The owning process is the failure boundary. This does not kill peer
    services, promise in-process recovery, or report a timeout as clean release.
    """

    def __init__(self, timeout):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Worker cleanup timeout must be positive and finite")
        self.timeout = timeout
        self.process = None

    def __enter__(self):
        self.deadline = time.monotonic() + self.timeout
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-c", _WATCH, str(os.getpid()), str(self.deadline)],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
        except OSError:
            # Without a watcher there is no safe bound on native teardown.
            # Exit without running CUDA finalizers that may themselves hang.
            os._exit(70)
        return self

    def __exit__(self, kind, error, traceback):
        process = self.process
        completed = kind is None and time.monotonic() < self.deadline
        try:
            process.stdin.write(b"C" if completed else b"F")
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            process.stdin.close()
        if completed:
            # CUDA resources have already been released. Reap the lightweight
            # watcher without adding an unbounded wait to cleanup itself.
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        elif kind is None:
            raise TimeoutError("Worker cleanup exceeded its deadline")
        return False
