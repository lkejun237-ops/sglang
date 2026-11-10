# SPDX-License-Identifier: Apache-2.0
"""CPU-only deployment contract and subprocess lifecycle checks."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "tools" / "zing_b300_launcher.py"
spec = importlib.util.spec_from_file_location("zing_b300_launcher", MODULE_PATH)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def args(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    model, lynn = tmp_path / "export", tmp_path / "lynn"
    model.mkdir()
    lynn.mkdir()
    (model / "zing_config.json").write_text('{"split_first_latent": true}')
    (model / "zing_realtime_profile.json").write_text(
        json.dumps(
            {
                "source_attention_window": {"window_chunks": 21, "sink_chunks": 0},
                "source_encoder": {"mode": "progressive", "remote": {"port": 9999}},
                "source_decoder": {"remote": {"port": 9998}},
                "model_config_sha256": "unchanged",
            }
        )
    )
    (lynn / "decoder.pt").write_bytes(b"fixture")
    service, client, profile = [
        tmp_path / name for name in ("service.json", "client.json", "qwen-profile.json")
    ]
    service.write_text(
        json.dumps(
            {"native": {"gpu_id": 0}, "tokens": {"secret-value": {"tenant": "test"}}}
        )
    )
    client.write_text(json.dumps({"token": "secret-value"}))
    profile.write_text("{}")
    return launcher.parser().parse_args(
        [
            "--model-path",
            str(model),
            "--lynn-model-dir",
            str(lynn),
            "--conditioning-service-config",
            str(service),
            "--conditioning-config",
            str(client),
            "--conditioning-profile",
            str(profile),
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )


def test_plan_is_gpu_free_and_does_not_expose_tokens(args):
    before = set(sys.modules)
    plan = launcher.build_plan(args)
    assert "torch" not in set(sys.modules) - before
    assert not args.run_dir.exists()
    assert "secret-value" not in json.dumps(plan)
    assert plan["roles"]["server"]["devices"] == "0,1"
    assert (
        plan["roles"]["encoder"]["devices"]
        == plan["roles"]["conditioning"]["devices"]
        == "2"
    )
    assert plan["roles"]["decoder"]["devices"] == "3"
    command = plan["roles"]["server"]["command"]
    assert command[3:6] == ["_server", "30001", "30003"]
    for option, expected in (
        ("--sp-degree", "2"),
        ("--zing-source-work-quantum", "1"),
        ("--realtime-max-sessions", "1"),
        ("--zing-denoising-step-list", "1000,500"),
    ):
        assert command[command.index(option) + 1] == expected
    assert "--transformer-quantization" not in command
    assert plan["pipeline"]["zing_adaln_mode"] == "resident"
    assert "zing_adaln_table_path" not in plan["pipeline"]
    assert plan["environment"]["SGLANG_H3_QWEN_FP8"] == "1"
    assert plan["environment"]["SGLANG_ZING_DENOISE_CUDA_GRAPH"] == "1"


def test_materialized_request_and_pool_windows_agree_without_changing_export(args):
    original = (args.model_path / "zing_realtime_profile.json").read_bytes()
    plan = launcher.build_plan(args)
    launcher.materialize(plan)
    profile = json.loads(
        (args.run_dir / "model/zing_realtime_profile.json").read_text()
    )
    assert profile["source_attention_window"] == {"window_chunks": 6, "sink_chunks": 0}
    assert "remote" not in profile["source_encoder"]
    assert "source_decoder" not in profile
    assert profile["model_config_sha256"] == "unchanged"
    pipeline = json.loads((args.run_dir / "pipeline.json").read_text())
    assert (
        pipeline["zing_kv_pool_window_chunks"],
        pipeline["zing_kv_pool_sink_chunks"],
    ) == (6, 0)
    assert (args.model_path / "zing_realtime_profile.json").read_bytes() == original
    assert (
        args.run_dir / "model/zing_config.json"
    ).resolve() == args.model_path / "zing_config.json"
    with pytest.raises(ValueError, match="new directory"):
        launcher.build_plan(args)


@pytest.mark.parametrize("devices", ["0,1,2", "0,1,2,2", "0,1,2,"])
def test_invalid_layout_rejected(args, devices):
    args.gpus = devices
    with pytest.raises(ValueError, match="four distinct"):
        launcher.build_plan(args)


def test_inherited_allocation_is_respected(args, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")
    with pytest.raises(ValueError, match="inherited"):
        launcher.build_plan(args)
    args.gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    assert launcher.build_plan(args)["roles"]["encoder"]["devices"] == "GPU-c"


def test_qwen_gpu_and_auth_must_match(args):
    args.conditioning_service_config.write_text('{"native":{"gpu_id":4}}')
    with pytest.raises(ValueError, match="gpu_id must be 0"):
        launcher.build_plan(args)
    args.conditioning_service_config.write_text(
        '{"native":{"gpu_id":0},"tokens":{"other":{}}}'
    )
    with pytest.raises(ValueError, match="token must be registered"):
        launcher.build_plan(args)


@pytest.mark.parametrize("mib,cc", [(240 * 1024 - 1, [10, 3]), (280000, [10, 0])])
def test_hardware_gate_rejects_capacity_or_architecture(args, monkeypatch, mib, cc):
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *a, **kw: json.dumps([{"cc": cc, "mib": mib}] * 4),
    )
    with pytest.raises(ValueError, match="four CC10.3"):
        launcher.check_hardware(args.gpus, {})


def test_partial_startup_failure_cleans_up_owned_processes(args, monkeypatch):
    # Replace only the GPU/service boundaries; exercise the real supervisor finally path.
    import types

    module = "sglang.multimodal_gen.runtime.disaggregation.role_launcher"

    def fail(*a, **kw):
        raise RuntimeError("readiness failed")

    monkeypatch.setitem(sys.modules, module, types.SimpleNamespace(wait_ready=fail))
    monkeypatch.setattr(launcher, "check_hardware", lambda *a: [])
    monkeypatch.setattr(launcher, "check_ports", lambda *a: None)
    started = []
    real_popen = subprocess.Popen

    def start(*a, **kw):
        process = real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        started.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", start)
    with pytest.raises(RuntimeError, match="readiness failed"):
        launcher.serve(args, launcher.build_plan(args))
    assert len(started) == 1
    assert started[0].poll() is not None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_cleanup_kills_rank_descendants_after_leader_exit(tmp_path):
    pidfile = tmp_path / "rank.pid"
    code = "import subprocess,sys; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); Path(sys.argv[1]).write_text(str(p.pid))"
    leader = subprocess.Popen(
        [sys.executable, "-c", code, str(pidfile)], start_new_session=True
    )
    child = None
    try:
        leader.wait(timeout=10)
        child = int(pidfile.read_text())
        launcher.stop_processes([leader])
        # A dead orphan can briefly be a zombie; it must not remain running.
        for _ in range(50):
            result = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(child)],
                capture_output=True,
                text=True,
                check=False,
            )
            if not result.stdout.strip() or result.stdout.strip().startswith("Z"):
                break
            time.sleep(0.02)
        else:
            pytest.fail("Rank descendant remained alive after launcher cleanup")
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_documented_shell_entry_dry_run(args):
    repo = MODULE_PATH.parents[4]
    command = ["bash", str(repo / "scripts/zing/launch_b300.sh")]
    for field in (
        "model_path",
        "lynn_model_dir",
        "conditioning_service_config",
        "conditioning_config",
        "conditioning_profile",
        "run_dir",
    ):
        command += ["--" + field.replace("_", "-"), str(getattr(args, field))]
    result = subprocess.run(
        command + ["--dry-run"],
        env={
            **os.environ,
            "PYTHON": sys.executable,
            "PYTHONPATH": str(repo / "python"),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert "torch" not in result.stderr
    assert "secret-value" not in result.stdout
    assert not args.run_dir.exists()
    assert plan["pipeline"]["dit_precision"] == "bf16"
    assert plan["roles"]["server"]["devices"] == "0,1"


def test_port_preflight_rejects_existing_service():
    import socket

    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        with pytest.raises(OSError):
            launcher.check_ports(occupied.getsockname()[1])


def test_server_handoff_uses_only_dit_devices_and_injects_remote_ports(monkeypatch):
    import types

    args = types.SimpleNamespace()
    observed = {}

    def prepare(argv):
        observed["argv"] = argv
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3,6,7"
        return args

    def dispatch(value):
        observed["ports"] = (value.disagg_encoder_port, value.disagg_decoder_port)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,6,7")
    monkeypatch.setitem(
        sys.modules,
        "sglang.multimodal_gen.runtime.launch_server",
        types.SimpleNamespace(dispatch_launch=dispatch),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.multimodal_gen.runtime.server_args",
        types.SimpleNamespace(prepare_server_args=prepare),
    )
    launcher.server_main(["31001", "31003", "--num-gpus", "4"])
    assert observed == {"argv": ["--num-gpus", "4"], "ports": (31001, 31003)}


def test_run_directory_cannot_create_recursive_model_view(args):
    args.run_dir = args.model_path / "runs/current"
    with pytest.raises(ValueError, match="outside the source model"):
        launcher.build_plan(args)
    assert not args.run_dir.exists()


@pytest.mark.parametrize("old_timing", ["before_source", "after_step0"])
def test_materialize_aligns_eviction_without_mutating_export(args, old_timing):
    source = args.model_path / "zing_realtime_profile.json"
    profile = json.loads(source.read_text())
    profile["window_eviction_timing"] = old_timing
    source.write_text(json.dumps(profile))
    original = source.read_bytes()
    plan = launcher.build_plan(args)
    launcher.materialize(plan)
    generated = json.loads(
        (args.run_dir / "model/zing_realtime_profile.json").read_text()
    )
    assert (
        generated["window_eviction_timing"]
        == plan["pipeline"]["zing_kv_pool_window_eviction_timing"]
        == "after_step0"
    )
    assert source.read_bytes() == original


def test_pr647_four_card_policy_and_current_capacity(args):
    plan = launcher.build_plan(args)
    command = plan["roles"]["server"]["command"]
    expected = {
        "--num-gpus": "2",
        "--sp-degree": "2",
        "--ulysses-degree": "2",
        "--mem-fraction-static": "0.8",
        "--zing-source-prompt-joint": "true",
        "--zing-block0-cache-host-gb": "416",
        "--zing-block0-cache-host-total-gb": "416",
        "--zing-block0-cache-disk-gb": "700",
        "--zing-block0-cache-min-free-gb": "128",
        "--zing-block0-cache-inflight-gb": "128",
        "--zing-block0-cache-max-writers": "2",
    }
    for flag, value in expected.items():
        assert command[command.index(flag) + 1] == value
    assert "--zing-block0-cache-gc" in command
    assert "--static-connection" in plan["roles"]["decoder"]["command"]
    assert plan["pipeline"]["zing_kv_pool_sink_chunks"] == 0
    for name in ("size", "max_target_area", "max_target_long_edge", "max_prefix_rows"):
        assert "zing_kv_pool_" + name not in plan["pipeline"]


def test_explicit_fp8_changes_only_dit_precision_path(args):
    args.quantization = "fp8"
    plan = launcher.build_plan(args)
    command = plan["roles"]["server"]["command"]
    assert command[command.index("--quantization") + 1] == "fp8"
    assert command[command.index("--direct-gpu-weight-loading") + 1] == "false"
    assert plan["environment"]["SGLANG_ZING_QUANT_SCOPE"] == "ffn-attention"
    assert plan["environment"]["SGLANG_H3_QWEN_FP8"] == "1"
    assert plan["pipeline"]["decoder_config"]["precision"] == "bf16"


def test_explicit_bf16_opt_out_does_not_inherit_quant_scope(args):
    args.quantization = "none"
    plan = launcher.build_plan(args)
    command = plan["roles"]["server"]["command"]
    assert command[command.index("--quantization") + 1] == "none"
    assert "SGLANG_ZING_QUANT_SCOPE" not in plan["environment"]


def test_supervisor_and_generic_entry_share_pipeline_and_qwen_defaults(args):
    from sglang.multimodal_gen.tools.zing_h3_profile_launcher import (
        COMMON_ENV,
        server_policy,
    )

    plan = launcher.build_plan(args)
    command = plan["roles"]["server"]["command"]
    for option, value in server_policy(
        "b300", str(args.run_dir / "block0-cache")
    ).items():
        assert command.count(option) == 1
        if value is not None:
            assert command[command.index(option) + 1] == value
    for key, value in COMMON_ENV.items():
        assert plan["environment"][key] == value
    for key in (
        "FP8",
        "TEXT_FA4",
        "VISION_FA4",
        "HF_RMSNORM",
        "FUSED_MROPE",
        "FUSED_GATE_UP",
    ):
        assert plan["environment"]["SGLANG_H3_QWEN_" + key] == "1"
    for option in (
        "--zing-conditioning-prefetch",
        "--zing-source-encoder-prefetch",
        "--zing-source-async-output",
        "--zing-source-decoder-overlap",
        "--zing-source-defer-final",
        "--zing-source-prompt-joint",
    ):
        assert command[command.index(option) + 1] == "true"
    assert command[command.index("--zing-source-work-quantum") + 1] == "1"
