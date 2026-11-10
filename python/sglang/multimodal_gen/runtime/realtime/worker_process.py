# SPDX-License-Identifier: Apache-2.0
"""Keep native workers killable when their command is a container's PID 1."""

import os
import signal
import subprocess
import sys


def _supervise(command):
    # No model work or restart loop in this process. In a PID namespace, only
    # its init process survives here; the worker and its watchdog are children.
    child = None
    pending = []

    def forward(signum, _frame):
        if child is None:
            pending.append(signum)
            return
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    previous = {
        signum: signal.signal(signum, forward)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        child = subprocess.Popen(command, start_new_session=True)
        for signum in pending:
            forward(signum, None)
        while True:
            # Reap orphaned watchdogs as well as the main worker. Waiting only
            # for Popen.pid would accumulate zombies under a long-lived init.
            pid, status = os.waitpid(-1, 0)
            if pid == child.pid:
                child.returncode = os.waitstatus_to_exitcode(status)
                break
        return child.returncode if child.returncode >= 0 else 128 - child.returncode
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_worker(main):
    """Run normally under an init/launcher; otherwise supervise one child once.

    Linux ignores a descendant's SIGKILL against namespace PID 1. A separate
    child is required for CleanupWatchdog to enforce a native-code deadline.
    Exiting that child ends this supervisor too; it never restarts the worker
    or manufactures a model release receipt.
    """
    if sys.platform != "linux" or os.getpid() != 1:
        return main()
    # Preserve -m, script paths and interpreter flags. Re-exec, rather than
    # fork-and-run Python model code, so no imported runtime state is inherited.
    command = [sys.executable, *sys.orig_argv[1:]]
    raise SystemExit(_supervise(command))
