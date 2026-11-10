# SPDX-License-Identifier: Apache-2.0
"""Exercise forced cleanup only inside disposable subprocesses."""

import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("cleanup_failed", [False, True])
def test_closed_watchdog_pipe_preserves_cleanup_error(cleanup_failed):
    from sglang.multimodal_gen.runtime.realtime.cleanup_watchdog import CleanupWatchdog

    class BrokenPipe:
        def write(self, _value):
            raise BrokenPipeError("watchdog already exited")

        def close(self):
            raise BrokenPipeError("buffered close retried the flush")

    watchdog = CleanupWatchdog(1)
    watchdog.deadline = time.monotonic() - 1
    watchdog.process = SimpleNamespace(stdin=BrokenPipe())
    failure = RuntimeError("original cleanup failed")
    if cleanup_failed:
        assert watchdog.__exit__(RuntimeError, failure, None) is False
    else:
        with pytest.raises(TimeoutError, match="cleanup exceeded"):
            watchdog.__exit__(None, None, None)


def test_unsupervised_init_cannot_enter_native_cleanup():
    script = """
from sglang.multimodal_gen.runtime.realtime import cleanup_watchdog
cleanup_watchdog.os.getpid = lambda: 1
with cleanup_watchdog.CleanupWatchdog(1):
    raise AssertionError('Must not enter unkillable native cleanup')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 70
    assert "supervised non-PID-1" in result.stderr


def test_worker_with_an_existing_init_runs_inline(monkeypatch):
    from sglang.multimodal_gen.runtime.realtime import worker_process

    monkeypatch.setattr(worker_process, "os", SimpleNamespace(getpid=lambda: 2))
    monkeypatch.setattr(
        worker_process,
        "_supervise",
        lambda _: pytest.fail("An existing init must not get another supervisor"),
    )
    assert worker_process.run_worker(lambda: 23) == 23


@pytest.mark.parametrize("mode", ["success", "exit", "hang"])
def test_supervisor_exits_with_its_only_worker_without_restart(mode):
    worker = """
import sys, time
from sglang.multimodal_gen.runtime.realtime.cleanup_watchdog import CleanupWatchdog
print('worker-started', flush=True)
if sys.argv[1] == 'exit':
    sys.exit(23)
with CleanupWatchdog(0.2):
    if sys.argv[1] == 'hang':
        time.sleep(10)
"""
    supervisor = """
import sys
from sglang.multimodal_gen.runtime.realtime.worker_process import _supervise
sys.exit(_supervise([sys.executable, '-c', sys.argv[1], sys.argv[2]]))
"""
    result = subprocess.run(
        [sys.executable, "-c", supervisor, worker, mode],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == {"success": 0, "exit": 23, "hang": 137}[mode]
    assert result.stdout.count("worker-started") == 1


def test_supervisor_forwards_termination_to_worker_group():
    worker = """
import signal, sys, time
signal.signal(signal.SIGTERM, lambda *_: sys.exit(24))
print('ready', flush=True)
while True:
    time.sleep(0.1)
"""
    supervisor = """
import sys
from sglang.multimodal_gen.runtime.realtime.worker_process import _supervise
sys.exit(_supervise([sys.executable, '-c', sys.argv[1]]))
"""
    process = subprocess.Popen(
        [sys.executable, "-c", supervisor, worker],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        import select

        assert select.select([process.stdout], [], [], 5)[0], "Worker did not start"
        assert process.stdout.readline().strip() == "ready"
        process.terminate()
        assert process.wait(timeout=5) == 24
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


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
