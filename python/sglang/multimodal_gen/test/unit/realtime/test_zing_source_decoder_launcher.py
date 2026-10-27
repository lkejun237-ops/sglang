# SPDX-License-Identifier: Apache-2.0
"""Unified Disaggregated Role Launcher tests: device mapping, worker lifecycle, and cleanup."""

import importlib
import importlib.util
import sys
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.server_args.disagg import DisaggServerArgsMixin


def _load_role_launcher():
    try:
        return importlib.import_module(
            "sglang.multimodal_gen.runtime.disaggregation.role_launcher"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"sglang.multimodal_gen.runtime.disaggregation.role_launcher not implemented: {exc}"
        )


class _AliveProcess:
    def __init__(self, pid=4242):
        self.pid = pid

    def poll(self):
        return None


class _Prctl:
    def __init__(self, calls):
        self.calls = calls

    def prctl(self, option, signal_number):
        self.calls.append((option, signal_number))


def _make_args(**overrides):
    values = dict(
        num_gpus=2,
        nnodes=1,
        node_rank=0,
        master_port=29500,
        encoder_gpus=None,
        denoiser_gpus="0,1",
        decoder_gpus="2",
        disagg_encoder_port=None,
        disagg_decoder_port=None,
        model_path="/models/zing",
        port=30000,
        host="127.0.0.1",
        log_level="info",
        pipeline_config=SimpleNamespace(decoder_config={"type": "lynn_streaming"}),
    )
    values.update(overrides)
    args = SimpleNamespace(**values)
    args._validate_disagg_gpu_roles = lambda: (
        DisaggServerArgsMixin._validate_disagg_gpu_roles(args)
    )
    return args


# ==============================================================================
# 1. Verification that legacy private manager is removed
# ==============================================================================


def test_legacy_zing_source_decoder_manager_is_removed():
    spec = importlib.util.find_spec(
        "sglang.multimodal_gen.runtime.managers.zing_source_decoder"
    )
    assert spec is None, (
        "sglang.multimodal_gen.runtime.managers.zing_source_decoder must be removed"
    )


# ==============================================================================
# 2. Device translation (resolve_role_devices)
# ==============================================================================


def test_resolve_role_devices_maps_devices_in_order():
    launcher = _load_role_launcher()
    args = _make_args(
        encoder_gpus="3",
        denoiser_gpus="2,0",
        decoder_gpus="1",
    )
    encoder_dev, denoiser_dev, decoder_dev = launcher.resolve_role_devices(
        args, caller="4,5,6,7"
    )
    assert encoder_dev == "7"
    assert denoiser_dev == "6,4"
    assert decoder_dev == "5"


def test_resolve_role_devices_handles_none_roles():
    launcher = _load_role_launcher()
    args = _make_args(
        encoder_gpus=None,
        denoiser_gpus="1,0",
        decoder_gpus="2",
    )
    encoder_dev, denoiser_dev, decoder_dev = launcher.resolve_role_devices(
        args, caller="4,5,6,7"
    )
    assert encoder_dev is None
    assert denoiser_dev == "5,4"
    assert decoder_dev == "6"


def test_resolve_role_devices_out_of_bounds_rejected():
    launcher = _load_role_launcher()
    args = _make_args(
        encoder_gpus="4",
        denoiser_gpus="0,1",
        decoder_gpus=None,
    )
    with pytest.raises(ValueError, match="not visible"):
        launcher.resolve_role_devices(args, caller="4,5,6,7")


def test_resolve_role_devices_rejects_negative_index():
    launcher = _load_role_launcher()
    with pytest.raises(ValueError, match="out of range"):
        launcher.resolve_role_devices("-1", caller="0,1")


# ==============================================================================
# 3. Role command generation (build_role_command)
# ==============================================================================


def test_build_role_command_encoder():
    launcher = _load_role_launcher()
    args = _make_args(
        encoder_gpus="0",
    )
    cmd = launcher.build_role_command(args, RoleType.ENCODER, 31001)
    assert cmd[1:4] == ["-m", "torch.distributed.run", "--standalone"]
    assert "--nproc-per-node=1" in cmd
    assert cmd[cmd.index("--port") + 1] == "31001"
    assert cmd[cmd.index("--model") + 1] == "/models/zing"
    assert any("encoder_worker" in part for part in cmd)


def test_build_role_command_decoder():
    launcher = _load_role_launcher()
    args = _make_args(
        decoder_gpus="2",
        pipeline_config=SimpleNamespace(decoder_config={"type": "lynn_streaming"}),
    )
    cmd = launcher.build_role_command(args, RoleType.DECODER, 31003)
    assert cmd[1:4] == ["-m", "torch.distributed.run", "--standalone"]
    assert "--nproc-per-node=1" in cmd
    assert cmd[cmd.index("--port") + 1] == "31003"
    assert cmd[cmd.index("--model") + 1] == "/models/zing"
    assert '"type": "lynn_streaming"' in cmd[cmd.index("--decoder-config") + 1]
    assert any("decoder_worker" in part for part in cmd)


def test_build_role_command_decoder_missing_config_rejected():
    launcher = _load_role_launcher()
    args = _make_args(
        decoder_gpus="2",
        pipeline_config=SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="decoder_config"):
        launcher.build_role_command(args, RoleType.DECODER, 31003)


# ==============================================================================
# 4. Worker lifecycle, port assignment, and atomic rollback
# ==============================================================================


def test_start_role_workers_spawns_and_sets_ports(monkeypatch):
    launcher = _load_role_launcher()
    monkeypatch.setattr(launcher.sys, "platform", "linux")
    started_processes = []

    def popen(command, env=None, **kwargs):
        proc = _AliveProcess(pid=len(started_processes) + 100)
        started_processes.append(
            {"command": command, "env": env, "kwargs": kwargs, "proc": proc}
        )
        return proc

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher, "wait_ready", lambda process, port: None)
    monkeypatch.setattr(launcher.atexit, "register", lambda *args: None)
    ports = [31001, 31003]
    monkeypatch.setattr(
        launcher, "free_port", lambda start=31000, avoid=(): ports.pop(0)
    )
    monkeypatch.setattr(launcher, "_stop_on_signal", lambda *procs: None)

    args = _make_args(
        encoder_gpus="0",
        denoiser_gpus="1,2",
        decoder_gpus="3",
    )
    workers = launcher.start_role_workers(
        args, encoder_devices="4", decoder_devices="5"
    )

    assert len(started_processes) == 2
    enc_info = next(
        p for p in started_processes if "encoder_worker" in " ".join(p["command"])
    )
    assert enc_info["env"]["CUDA_VISIBLE_DEVICES"] == "4"
    assert enc_info["kwargs"]["preexec_fn"] is launcher._die_with_parent

    dec_info = next(
        p for p in started_processes if "decoder_worker" in " ".join(p["command"])
    )
    assert dec_info["env"]["CUDA_VISIBLE_DEVICES"] == "5"
    assert dec_info["kwargs"]["preexec_fn"] is launcher._die_with_parent

    assert args.disagg_encoder_port == 31001
    assert args.disagg_decoder_port == 31003


def test_start_role_workers_atomic_rollback_on_failure(monkeypatch):
    launcher = _load_role_launcher()
    stopped = []

    def popen(command, env=None, **kwargs):
        proc = _AliveProcess(pid=len(stopped) + 200)
        return proc

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.atexit, "register", lambda *args: None)
    monkeypatch.setattr(launcher, "stop", lambda proc: stopped.append(proc))
    monkeypatch.setattr(launcher, "_stop_on_signal", lambda *procs: None)

    def wait_ready_mock(process, port):
        # Encoder (31001) passes, Decoder (31003) fails
        if port == 31003:
            raise RuntimeError("Decoder readiness failed")

    monkeypatch.setattr(launcher, "wait_ready", wait_ready_mock)
    ports = [31001, 31003]
    monkeypatch.setattr(
        launcher, "free_port", lambda start=31000, avoid=(): ports.pop(0)
    )

    args = _make_args(
        encoder_gpus="0",
        denoiser_gpus="1,2",
        decoder_gpus="3",
    )

    with pytest.raises(RuntimeError, match="readiness"):
        launcher.start_role_workers(args, encoder_devices="4", decoder_devices="5")

    # Both the successful encoder and the failed decoder must be cleaned up
    assert len(stopped) == 2
    assert args.disagg_encoder_port is None
    assert args.disagg_decoder_port is None


def test_no_role_devices_starts_nothing(monkeypatch):
    launcher = _load_role_launcher()
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("spawned an unexpected process"),
    )
    args = _make_args(encoder_gpus=None, decoder_gpus=None)
    result = launcher.start_role_workers(
        args, encoder_devices=None, decoder_devices=None
    )
    assert result is None or result == []
    assert args.disagg_encoder_port is None
    assert args.disagg_decoder_port is None


# ==============================================================================
# 5. Role policies overlay (_with_disagg_role_policies)
# ==============================================================================


def test_disagg_role_policies_overrides_ports_and_mode():
    launcher = _load_role_launcher()
    profile = {
        "solver": "few_step",
        "source_encoder": {"remote": {"cross_dispatch_prefetch": True}},
    }
    result = launcher._with_disagg_role_policies(
        profile, encoder_port=31005, decoder_port=31007
    )
    assert result["source_encoder"]["remote"]["port"] == 31005
    assert result["source_encoder"]["remote"]["mode"] == "overlap"
    assert result["source_encoder"]["remote"]["cross_dispatch_prefetch"] is True
    assert result["source_decoder"]["remote"]["port"] == 31007
    # Original profile must remain untouched
    assert "port" not in profile["source_encoder"]["remote"]
    assert "source_decoder" not in profile


def test_disagg_role_policies_with_none_ports_preserves_profile():
    launcher = _load_role_launcher()
    profile = {"solver": "few_step"}
    assert (
        launcher._with_disagg_role_policies(
            profile, encoder_port=None, decoder_port=None
        )
        == profile
    )


# ==============================================================================
# 6. Signal handling and cleanup
# ==============================================================================


def test_server_death_kills_role_workers(monkeypatch):
    launcher = _load_role_launcher()
    calls = []
    monkeypatch.setattr(launcher.ctypes, "CDLL", lambda *args, **kwargs: _Prctl(calls))
    launcher._die_with_parent()
    assert calls == [(1, launcher.signal.SIGKILL)]


def test_sigterm_stops_role_workers_before_exiting(monkeypatch):
    launcher = _load_role_launcher()
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

    launcher._stop_on_signal([process])
    assert set(installed) == {launcher.signal.SIGTERM, launcher.signal.SIGINT}

    installed[launcher.signal.SIGTERM](launcher.signal.SIGTERM, None)
    assert killed == [process, (99, launcher.signal.SIGTERM)]


def test_a_chosen_shutdown_owner_is_not_overridden(monkeypatch):
    launcher = _load_role_launcher()
    installed = []
    monkeypatch.setattr(launcher.signal, "getsignal", lambda signum: object())
    monkeypatch.setattr(
        launcher.signal, "signal", lambda signum, handler: installed.append(signum)
    )
    launcher._stop_on_signal([_AliveProcess()])
    assert installed == []


# ==============================================================================
# 7. Round 1 Review Fix verification tests
# ==============================================================================


def test_resolve_role_devices_when_caller_empty_or_whitespace():
    launcher = _load_role_launcher()
    args = _make_args(
        encoder_gpus="0",
        denoiser_gpus="1,2",
        decoder_gpus="3",
    )
    # When caller="" (CUDA_VISIBLE_DEVICES not set on host)
    enc, den, dec = launcher.resolve_role_devices(args, caller="")
    assert enc == "0"
    assert den == "1,2"
    assert dec == "3"

    # When caller has whitespace
    enc, den, dec = launcher.resolve_role_devices(args, caller=" 4 , 5 , 6 , 7 ")
    assert enc == "4"
    assert den == "5,6"
    assert dec == "7"


def test_start_role_workers_avoids_scheduler_and_master_ports(monkeypatch):
    launcher = _load_role_launcher()
    args = _make_args(
        port=30000,
        scheduler_port=31003,
        master_port=31004,
        encoder_gpus="0",
        decoder_gpus="2",
    )
    avoid_calls = []

    def fake_free_port(start=launcher.START_PORT, avoid=()):
        avoid_calls.append(tuple(avoid))
        return 32000 + len(avoid_calls)

    monkeypatch.setattr(launcher, "free_port", fake_free_port)
    monkeypatch.setattr(launcher, "build_role_command", lambda *a, **k: ["dummy"])
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: _AliveProcess())
    monkeypatch.setattr(launcher.atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "wait_ready", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_stop_on_signal", lambda *a, **k: None)

    launcher.start_role_workers(args, encoder_devices="0", decoder_devices="2")

    assert len(avoid_calls) == 2
    first_avoid = avoid_calls[0]
    assert 30000 in first_avoid
    assert 31003 in first_avoid
    assert 31004 in first_avoid

    second_avoid = avoid_calls[1]
    assert 30000 in second_avoid
    assert 31003 in second_avoid
    assert 31004 in second_avoid
    assert 32001 in second_avoid


def test_stop_role_workers_stops_all_and_resets_ports(monkeypatch):
    launcher = _load_role_launcher()
    stopped = []
    monkeypatch.setattr(launcher, "stop", stopped.append)

    proc1 = _AliveProcess(pid=101)
    proc2 = _AliveProcess(pid=102)
    args = _make_args(
        disagg_encoder_port=31001,
        disagg_decoder_port=31002,
    )

    launcher.stop_role_workers(args, [proc1, proc2])

    assert stopped == [proc1, proc2]
    assert args.disagg_encoder_port is None
    assert args.disagg_decoder_port is None


def test_launch_pool_disagg_server_delegates_zing_streaming_roles(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "setproctitle",
        SimpleNamespace(setproctitle=lambda *a, **k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.multimodal_gen.runtime.managers.gpu_worker",
        SimpleNamespace(run_scheduler_process=lambda *a, **k: None),
    )
    import sglang.multimodal_gen.runtime.launch_server as ls_mod
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
        ZingH3CausalPipelineConfig,
    )

    delegated = []

    def fake_launch_server(server_args, launch_http_server=True):
        delegated.append((server_args, launch_http_server))
        return ["fake_process"]

    monkeypatch.setattr(ls_mod, "launch_server", fake_launch_server)

    server_args = _make_args(
        model_path="/models/zing",
        num_gpus=1,
        pipeline_config=ZingH3CausalPipelineConfig(),
    )

    res = ls_mod.launch_pool_disagg_server(
        server_args,
        encoder_gpus=[[0]],
        denoiser_gpus=[[1, 2]],
        decoder_gpus=[[3]],
        launch_http_server=False,
    )

    assert res == ["fake_process"]
    assert len(delegated) == 1
    args, http = delegated[0]
    assert http is False
    assert args.encoder_gpus == "0"
    assert args.denoiser_gpus == "1,2"
    assert args.decoder_gpus == "3"
    assert args.num_gpus == 2


def test_launch_pool_disagg_server_reconstructs_server_args_with_updated_num_gpus(
    monkeypatch,
):
    import dataclasses

    monkeypatch.setitem(
        sys.modules,
        "setproctitle",
        SimpleNamespace(setproctitle=lambda *a, **k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.multimodal_gen.runtime.managers.gpu_worker",
        SimpleNamespace(run_scheduler_process=lambda *a, **k: None),
    )
    from collections import defaultdict

    import sglang.multimodal_gen.runtime.launch_server as ls_mod

    post_init_counts = defaultdict(int)

    @dataclasses.dataclass
    class _DummyDataclassArgs:
        model_path: str = "dummy"
        num_gpus: int = 1
        encoder_gpus: str | None = None
        denoiser_gpus: str | None = None
        decoder_gpus: str | None = None
        tp_size: int = 1
        pipeline_config: object = dataclasses.field(
            default_factory=lambda: SimpleNamespace(
                decoder_config={"type": "lynn_streaming"}
            )
        )
        post_init_called: int = dataclasses.field(default=0, init=False)
        port_normalized_count: int = dataclasses.field(default=0, init=False)

        def __post_init__(self):
            self.post_init_called += 1
            self.tp_size = self.num_gpus
            self.port_normalized_count += 1
            post_init_counts[id(self)] += 1

        def _validate_disagg_gpu_roles(self):
            pass

    orig_args = _DummyDataclassArgs(num_gpus=1)
    delegated = []

    def fake_launch_server(server_args, launch_http_server=True):
        delegated.append(server_args)
        return ["fake_process"]

    monkeypatch.setattr(ls_mod, "launch_server", fake_launch_server)

    ls_mod.launch_pool_disagg_server(
        orig_args,
        encoder_gpus=[[0]],
        denoiser_gpus=[[1, 2]],
        decoder_gpus=[[3]],
    )
    assert len(delegated) == 1
    new_args = delegated[0]
    # Must be a new instance with updated num_gpus and re-executed __post_init__ exactly once
    assert new_args is not orig_args
    assert new_args.num_gpus == 2
    assert new_args.tp_size == 2
    assert new_args.denoiser_gpus == "1,2"
    assert new_args.encoder_gpus == "0"
    assert new_args.decoder_gpus == "3"
    # dataclasses.replace() triggers __post_init__ exactly once; must not be called twice
    assert new_args.post_init_called == 1
    assert new_args.port_normalized_count == 1
    assert post_init_counts[id(orig_args)] == 1
    assert post_init_counts[id(new_args)] == 1


def test_assign_denoiser_visible_devices_rejects_initialized_cuda(monkeypatch):
    import torch

    launcher = _load_role_launcher()
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    with pytest.raises(
        RuntimeError, match="must be called before CUDA runtime initialization"
    ):
        launcher.assign_denoiser_visible_devices(
            SimpleNamespace(), denoiser_devices="1,2"
        )


def test_launch_server_returns_role_processes_when_offline_mode(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "setproctitle",
        SimpleNamespace(setproctitle=lambda *a, **k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.multimodal_gen.runtime.managers.gpu_worker",
        SimpleNamespace(run_scheduler_process=lambda *a, **k: None),
    )
    import sglang.multimodal_gen.runtime.launch_server as ls_mod

    role_p1 = _AliveProcess(pid=5001)
    role_p2 = _AliveProcess(pid=5002)

    monkeypatch.setattr(
        ls_mod, "resolve_role_devices", lambda args, caller="": ("0", "1", "2")
    )
    monkeypatch.setattr(
        ls_mod, "assign_denoiser_visible_devices", lambda args, devices: None
    )
    monkeypatch.setattr(
        ls_mod, "start_role_workers", lambda *a, **k: [role_p1, role_p2]
    )
    monkeypatch.setattr(ls_mod, "configure_logger", lambda *a, **k: None)
    monkeypatch.setattr(
        ls_mod,
        "SourceTransportBudget",
        SimpleNamespace(
            from_args=lambda a: SimpleNamespace(websocket_options=lambda: None)
        ),
    )

    fake_scheduler_p = _AliveProcess(pid=4001)

    class _FakeProcess:
        def __init__(self, *a, **k):
            self.pid = 4001

        def start(self):
            pass

    monkeypatch.setattr(ls_mod.mp, "Process", _FakeProcess)

    server_args = _make_args(
        encoder_gpus="0",
        denoiser_gpus="1",
        decoder_gpus="2",
        webui=False,
    )

    # In launch_server with launch_http_server=False, the returned processes list
    # must include both the scheduler worker process and the role worker processes.
    def mock_launch_server_only(args):
        pass

    monkeypatch.setattr(ls_mod, "launch_http_server_only", mock_launch_server_only)
    monkeypatch.setattr(ls_mod, "_run_http_server_process", lambda *a, **k: None)

    # Mock scheduler process pipes
    class _FakePipe:
        def recv(self):
            return {"status": "ready", "master_port": 1234}

        def close(self):
            pass

    monkeypatch.setattr(ls_mod.mp, "Pipe", lambda *a, **k: (_FakePipe(), _FakePipe()))

    result_processes = ls_mod.launch_server(server_args, launch_http_server=False)
    assert role_p1 in result_processes
    assert role_p2 in result_processes
