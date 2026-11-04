# SPDX-License-Identifier: Apache-2.0
"""Exercise forced cleanup only inside disposable subprocesses."""

import os
import signal
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ["success", "hang", "failure"])
def test_cleanup_deadline_is_enforced_outside_the_worker(mode):
    script = """
import sys, time
from sglang.multimodal_gen.runtime.realtime.cleanup_watchdog import CleanupWatchdog
mode = sys.argv[1]
try:
    with CleanupWatchdog(0.3):
        if mode == 'hang':
            time.sleep(10)
        if mode == 'failure':
            raise RuntimeError('injected cleanup failure')
except RuntimeError:
    pass
# A completed cleanup must disarm the watcher; an exception must not.
time.sleep(0.6)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, mode],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=5,
    )
    if mode == "success":
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode == -signal.SIGKILL, result.stderr
        assert "Worker cleanup exceeded its deadline" in result.stderr


def test_fault_abort_does_not_call_graceful_shutdown():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.disaggregation.transport.nccl_bridge import (
        abort_bridge_group,
    )

    calls = []
    group = SimpleNamespace(
        abort=lambda: calls.append("abort"),
        shutdown=lambda: pytest.fail("Fault cleanup must not drain communications"),
    )
    abort_bridge_group(group)
    assert calls == ["abort"]


def test_missing_abort_requires_restart_instead_of_graceful_fallback():
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.disaggregation.transport.nccl_bridge import (
        abort_bridge_group,
    )

    group = SimpleNamespace(shutdown=lambda: pytest.fail("No graceful fallback"))
    with pytest.raises(RuntimeError, match="restart is required"):
        abort_bridge_group(group)


def test_watchdog_launch_failure_exits_only_the_disposable_worker():
    script = """
from sglang.multimodal_gen.runtime.realtime import cleanup_watchdog

def unavailable(*args, **kwargs):
    raise OSError('injected process limit')

cleanup_watchdog.subprocess.Popen = unavailable
with cleanup_watchdog.CleanupWatchdog(1):
    raise AssertionError('Must not enter unguarded native cleanup')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 70
