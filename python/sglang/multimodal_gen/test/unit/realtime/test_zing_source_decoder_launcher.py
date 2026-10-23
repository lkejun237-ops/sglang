# SPDX-License-Identifier: Apache-2.0
"""Launcher-owned Zing source decoder: device roles, readiness and cleanup.

Deployments name the physical role of each GPU explicitly, as indices into the
caller's CUDA_VISIBLE_DEVICES. The launcher only translates and checks them; it
never assigns devices itself.
"""

from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_h3_source_adapter import (
    _with_source_decoder_policy,
)
from sglang.multimodal_gen.runtime.managers import zing_source_decoder as launcher


def _args(**overrides):
    values = dict(
        num_gpus=2,
        zing_dit_gpus="0,1",
        zing_decoder_gpus="2",
        zing_source_decoder_port=None,
        model_path="/models/zing",
        port=30000,
        pipeline_config=SimpleNamespace(decoder_config={"type": "lynn_streaming"}),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _AliveProcess:
    pid = 4242

    def poll(self):
        return None


class _Prctl:
    def __init__(self, calls):
        self.calls = calls

    def prctl(self, option, signal_number):
        self.calls.append((option, signal_number))


def test_role_indices_translate_through_the_callers_device_list():
    # Indices refer to CUDA_VISIBLE_DEVICES order, not nvidia-smi IDs.
    args = _args(zing_dit_gpus="2,0", zing_decoder_gpus="1")
    assert launcher.resolve_role_devices(args, caller="4,5,6") == ("6,4", "5")


def test_unseen_role_index_is_rejected():
    with pytest.raises(ValueError, match="not visible"):
        launcher.resolve_role_devices(_args(zing_decoder_gpus="3"), caller="4,5,6")


def test_command_launches_the_decoder_worker():
    command = launcher.command(_args(), 31003)
    assert command[1:4] == ["-m", "torch.distributed.run", "--standalone"]
    assert "--nproc-per-node=1" in command
    assert command[command.index("--port") + 1] == "31003"
    assert command[command.index("--model") + 1] == "/models/zing"
    assert '"type": "lynn_streaming"' in command[command.index("--decoder-config") + 1]


def test_missing_decoder_config_is_rejected():
    with pytest.raises(ValueError, match="decoder_config"):
        launcher.command(_args(pipeline_config=SimpleNamespace()), 31003)


def test_start_publishes_the_port_on_the_named_decoder_devices(monkeypatch):
    started = {}

    def popen(command, env=None, **kwargs):
        started["command"], started["env"] = command, env
        started["kwargs"] = kwargs
        return _AliveProcess()

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher, "wait_ready", lambda process, port: None)
    monkeypatch.setattr(launcher.atexit, "register", lambda *args: None)
    monkeypatch.setattr(launcher, "free_port", lambda start=31003, avoid=(): 31007)
    monkeypatch.setattr(launcher, "_stop_on_signal", lambda process: None)
    args = _args()
    process = launcher.start(args, decoder_devices="5")

    assert isinstance(process, _AliveProcess)
    assert args.zing_source_decoder_port == 31007
    assert started["env"]["CUDA_VISIBLE_DEVICES"] == "5"
    assert started["command"][started["command"].index("--port") + 1] == "31007"
    assert "--nproc-per-node=1" in started["command"]
    assert started["kwargs"]["preexec_fn"] is launcher._die_with_parent


def test_server_death_kills_the_decoder(monkeypatch):
    """A SIGTERM or SIGKILL to the server never runs atexit; the kernel must
    reap the decoder so it does not keep holding its GPUs."""
    calls = []
    monkeypatch.setattr(launcher.ctypes, "CDLL", lambda *args, **kwargs: _Prctl(calls))
    launcher._die_with_parent()
    assert calls == [(1, launcher.signal.SIGKILL)]


def test_sigterm_stops_the_decoder_before_exiting(monkeypatch):
    installed, killed = {}, []
    process = _AliveProcess()
    monkeypatch.setattr(
        launcher.signal, "getsignal", lambda signum: launcher.signal.SIG_DFL
    )
    monkeypatch.setattr(
        launcher.signal,
        "signal",
        lambda signum, handler: installed.__setitem__(signum, handler),
    )
    monkeypatch.setattr(launcher, "stop", killed.append)
    monkeypatch.setattr(
        launcher.os, "kill", lambda pid, signum: killed.append((pid, signum))
    )
    monkeypatch.setattr(launcher.os, "getpid", lambda: 99)

    launcher._stop_on_signal(process)
    assert set(installed) == {launcher.signal.SIGTERM, launcher.signal.SIGINT}

    installed[launcher.signal.SIGTERM](launcher.signal.SIGTERM, None)
    assert killed == [process, (99, launcher.signal.SIGTERM)]


def test_a_chosen_shutdown_owner_is_not_overridden(monkeypatch):
    installed = []
    monkeypatch.setattr(launcher.signal, "getsignal", lambda signum: object())
    monkeypatch.setattr(
        launcher.signal, "signal", lambda signum, handler: installed.append(signum)
    )
    launcher._stop_on_signal(_AliveProcess())
    assert installed == []


def test_unready_decoder_is_stopped_before_the_dit_server(monkeypatch):
    stopped = []
    monkeypatch.setattr(
        launcher.subprocess, "Popen", lambda *args, **kwargs: _AliveProcess()
    )
    monkeypatch.setattr(launcher, "free_port", lambda start=31003, avoid=(): 31007)
    monkeypatch.setattr(launcher.atexit, "register", lambda *args: None)
    monkeypatch.setattr(launcher, "stop", stopped.append)
    monkeypatch.setattr(launcher, "_stop_on_signal", lambda process: None)

    def not_ready(process, port):
        raise RuntimeError("Zing source decoder exited before readiness")

    monkeypatch.setattr(launcher, "wait_ready", not_ready)
    args = _args()

    with pytest.raises(RuntimeError, match="readiness"):
        launcher.start(args, decoder_devices="2")
    assert len(stopped) == 1
    assert args.zing_source_decoder_port is None


def test_no_decoder_devices_starts_nothing(monkeypatch):
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("spawned a decoder process"),
    )
    assert launcher.start(_args(zing_decoder_gpus=None), decoder_devices=None) is None


def test_policy_overrides_only_when_the_launcher_assigns_a_port():
    profile = {"solver": "few_step"}
    assert _with_source_decoder_policy(profile, None) == {"solver": "few_step"}
    assert _with_source_decoder_policy(profile, 31007) == {
        "solver": "few_step",
        "source_decoder": {"remote": {"port": 31007}},
    }
    assert profile == {"solver": "few_step"}
