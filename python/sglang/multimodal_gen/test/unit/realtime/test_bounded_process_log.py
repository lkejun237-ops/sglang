# SPDX-License-Identifier: Apache-2.0
import errno
import subprocess
import sys

import pytest

from sglang.multimodal_gen.runtime.utils.bounded_process_log import (
    BoundedProcessLog,
    diagnostic_print,
)


def test_child_diagnostics_rotate_with_a_finite_retained_size(tmp_path):
    path = tmp_path / "encoder.log"
    log = BoundedProcessLog(path, max_bytes=1024, backups=2)
    child = subprocess.Popen(
        [sys.executable, "-c", "for i in range(300): print(str(i)+' \\u4e2d'*800)"],
        stdout=subprocess.PIPE,
    )
    log.attach(child.stdout)
    assert child.wait(timeout=5) == 0
    log.close()
    files = list(tmp_path.glob("encoder.log*"))
    assert 1 <= len(files) <= 3
    assert all(p.stat().st_size <= 1024 for p in files)


def test_full_log_volume_does_not_stop_child_or_create_unbounded_queue(
    tmp_path, monkeypatch
):
    log = BoundedProcessLog(tmp_path / "model.log")

    def full(*args):
        raise OSError(errno.ENOSPC, "No space left")

    monkeypatch.setattr(log.handler, "_open", full)
    child = subprocess.Popen(
        [sys.executable, "-c", "for i in range(500): print('x'*1000)"],
        stdout=subprocess.PIPE,
    )
    log.attach(child.stdout)
    assert child.wait(timeout=5) == 0
    log.close()
    assert log.write_errors > 0
    assert log.queue.qsize() <= 128


def test_diagnostic_print_isolates_only_io_errors(monkeypatch):
    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left")

    monkeypatch.setattr("builtins.print", full)
    assert diagnostic_print("completed", flush=True) is False

    def broken(*args, **kwargs):
        raise TypeError("Invalid diagnostic")

    monkeypatch.setattr("builtins.print", broken)
    with pytest.raises(TypeError):
        diagnostic_print("bad")
