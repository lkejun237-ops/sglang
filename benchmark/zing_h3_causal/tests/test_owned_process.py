# SPDX-License-Identifier: Apache-2.0
"""Real Linux ownership checks; no CUDA or external process discovery."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest
from owned_process import owned_popen, verify_owned_cleanup

linux = pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper fixture")


def read_ready(path):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            time.sleep(0.01)
    raise RuntimeError("Owned process fixture did not become ready")


def assert_gone(row):
    with contextlib.suppress(psutil.NoSuchProcess):
        process = psutil.Process(row["pid"])
        assert process.create_time() != row["created"] or not process.is_running()


@linux
def test_fast_main_exit_reaps_independent_session_without_touching_unrelated(tmp_path):
    ready = tmp_path / "ready.json"
    code = (
        "import json,pathlib,psutil,subprocess,sys;"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True);"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid':p.pid,'created':psutil.Process(p.pid).create_time()}))"
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    process = owned_popen([sys.executable, "-c", code, str(ready)])
    try:
        assert process.wait(timeout=15) == 0
        row = read_ready(ready)
        verify_owned_cleanup(process)
        assert process._zing_cleanup_receipt["complete"]
        assert_gone(row)
        assert unrelated.poll() is None
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        try:
            verify_owned_cleanup(process)
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=10)


@linux
def test_double_forked_grandchild_is_adopted_and_reaped(tmp_path):
    ready = tmp_path / "grandchild.json"
    intermediate = (
        "import json,pathlib,psutil,subprocess,sys;"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True);"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid':p.pid,'created':psutil.Process(p.pid).create_time()}))"
    )
    code = (
        "import pathlib,subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]],start_new_session=True);"
        "p.wait();"
    )
    process = owned_popen([sys.executable, "-c", code, intermediate, str(ready)])
    try:
        assert process.wait(timeout=15) == 0
        verify_owned_cleanup(process)
        assert_gone(read_ready(ready))
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        verify_owned_cleanup(process)


@linux
@pytest.mark.parametrize("via_stop", [False, True])
def test_signal_cleanup_owns_the_independent_session(tmp_path, via_stop):
    from run_isolated_stall_suite import stop

    ready = tmp_path / "ready.json"
    code = (
        "import json,pathlib,psutil,subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True);"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid':p.pid,'created':psutil.Process(p.pid).create_time()}));"
        "time.sleep(60)"
    )
    process = owned_popen([sys.executable, "-c", code, str(ready)])
    try:
        row = read_ready(ready)
        if via_stop:
            stop(process)
        else:
            os.kill(process.pid, signal.SIGTERM)
        assert process.wait(timeout=15) == 143
        stop(process)  # A reaped supervisor must still have an ownership receipt.
        assert_gone(row)
    finally:
        if process.poll() is None:
            stop(process)


@linux
def test_main_failure_is_preserved_after_verified_cleanup():
    process = owned_popen([sys.executable, "-c", "raise SystemExit(7)"])
    assert process.wait(timeout=15) == 7
    verify_owned_cleanup(process)


def test_missing_receipt_cannot_report_cleanup_success():
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    process = SimpleNamespace(
        poll=lambda: 0, _zing_cleanup_fd=read_fd, _zing_cleanup_receipt=None
    )
    with pytest.raises(RuntimeError, match="did not confirm descendant cleanup"):
        verify_owned_cleanup(process)
    assert process._zing_cleanup_fd is None


@pytest.mark.parametrize("main_code,expected", [(0, 125), (7, 7)])
def test_cleanup_failure_is_nonzero_and_preserves_main_failure(
    monkeypatch, main_code, expected
):
    import owned_process

    monkeypatch.setattr(owned_process.sys, "platform", "linux")
    monkeypatch.setattr(
        owned_process.ctypes,
        "CDLL",
        lambda *a, **k: SimpleNamespace(prctl=lambda *a: 0),
    )
    monkeypatch.setattr(owned_process.signal, "signal", lambda *a: None)
    monkeypatch.setattr(
        owned_process.subprocess,
        "Popen",
        lambda *a: SimpleNamespace(poll=lambda: main_code, returncode=main_code),
    )

    def failed(main):
        raise TimeoutError("owned fixture cleanup failed")

    monkeypatch.setattr(owned_process, "_cleanup", failed)
    read_fd, write_fd = os.pipe()
    try:
        assert owned_process._main(write_fd, ["fixture"]) == expected
        assert not json.loads(os.read(read_fd, 4096))["complete"]
    finally:
        os.close(read_fd)


def test_subreaper_failure_does_not_spawn_main(monkeypatch):
    import owned_process

    monkeypatch.setattr(owned_process.sys, "platform", "linux")
    monkeypatch.setattr(
        owned_process.ctypes,
        "CDLL",
        lambda *a, **k: SimpleNamespace(prctl=lambda *a: 1),
    )
    monkeypatch.setattr(owned_process, "_cleanup", lambda main: None)

    spawned = []

    def forbidden(*a):
        spawned.append(True)
        raise AssertionError("Subreaper admission failed; main must not spawn")

    monkeypatch.setattr(owned_process.subprocess, "Popen", forbidden)
    read_fd, write_fd = os.pipe()
    try:
        assert owned_process._main(write_fd, ["fixture"]) == 125
        assert json.loads(os.read(read_fd, 4096))["complete"]
        assert not spawned
    finally:
        os.close(read_fd)


def test_cpu_controller_cleanup_failure_stops_queue(monkeypatch, tmp_path):
    import hashlib

    import frozen_source
    import owned_process
    import run_isolated_stall_suite
    import stall_cpu_controller
    import torch

    root = tmp_path / "controller"
    (root / "queue").mkdir(parents=True)
    (root / "scripts").mkdir()
    script = root / "scripts/task.py"
    script.write_text("print('owned fixture')\n")
    request = dict(
        script="scripts/task.py",
        sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        timeout=5,
    )
    for name in ["001", "002"]:
        (root / "queue" / (name + ".json")).write_text(json.dumps(request))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "controller",
            "--root",
            str(root),
            "--harness-source",
            str(tmp_path),
            "--source-revision",
            "0" * 40,
            "--max-seconds",
            "300",
        ],
    )
    monkeypatch.setattr(frozen_source, "verify_source", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    launched = []

    def start(*a, **k):
        launched.append(True)
        return SimpleNamespace(wait=lambda **k: 0, poll=lambda: 0)

    def cleanup_failed(process):
        raise RuntimeError("owned fixture missing cleanup receipt")

    monkeypatch.setattr(owned_process, "owned_popen", start)
    monkeypatch.setattr(run_isolated_stall_suite, "stop", cleanup_failed)
    with pytest.raises(RuntimeError, match="controller stopped"):
        stall_cpu_controller.main()
    result = json.loads((root / "001.result.json").read_text())
    assert not result["passed"] and result["cleanup_error"]
    assert len(launched) == 1 and not (root / "002.result.json").exists()


@pytest.mark.parametrize(
    "name",
    [
        "stall_job",
        "stall_cpu_controller",
        "diagnostics.dit_replay.probe_compiled_norm_history",
    ],
)
def test_standalone_entry_defers_supervisor_until_harness_admission(
    monkeypatch, tmp_path, name
):
    import builtins
    import importlib
    import runpy

    module = importlib.import_module(name)
    original = builtins.__import__

    def admitted_import(module_name, *args, **kwargs):
        if module_name == "owned_process":
            raise AssertionError(
                "Standalone entry imported supervision before admission"
            )
        return original(module_name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", admitted_import)
    namespace = runpy.run_path(module.__file__, run_name="standalone_import_admission")
    assert callable(namespace["main"])
    monkeypatch.setattr(builtins, "__import__", original)
    copied = tmp_path / (name.rsplit(".", 1)[-1] + ".py")
    from pathlib import Path

    copied.write_bytes(Path(module.__file__).read_bytes())
    result = subprocess.run(
        [sys.executable, str(copied), "--help"],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH="", CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0 and "usage:" in result.stdout.lower()
