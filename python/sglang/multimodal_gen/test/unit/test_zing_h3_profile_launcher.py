# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for hardware selection and feature delivery to child processes."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Keep launcher checks independent of the GPU runtime and optional model imports.
_path = Path(__file__).resolve().parents[2] / "tools/zing_h3_profile_launcher.py"
_spec = importlib.util.spec_from_file_location("zing_launcher_under_test", _path)
launcher = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = launcher
_spec.loader.exec_module(launcher)


def gpu(profile="b300", memory=None, sms=None):
    return launcher.GPUInfo(
        name="L20D" if profile == "b300" else "L20X",
        capability=(10, 3) if profile == "b300" else (9, 0),
        total_memory_mib=memory or (274113 if profile == "b300" else 143771),
        multiprocessor_count=sms or (148 if profile == "b300" else 132),
    )


def contract(profile="b300", resolution="480", component="server", env=None, args=None):
    return launcher.build_launch_contract(
        requested_profile="auto",
        resolution=resolution,
        component=component,
        gpus=[gpu(profile)] * (4 if component == "server" else 1),
        arguments=args or [],
        environment=env or {},
    )


@pytest.mark.parametrize("profile", ["b300", "h200"])
@pytest.mark.parametrize("resolution", ["480", "736"])
def test_all_screenshot_accelerations_and_current_cache_cli(profile, resolution):
    result = contract(profile, resolution)
    command = result["command"]
    for option, value in launcher.SERVER_DEFAULTS.items():
        if value is None:
            assert option in command
        else:
            assert command[command.index(option) + 1] == value
    assert "SGLANG_ZING_BLOCK0_CACHE_ENTRIES" not in result["environment"]
    assert "SGLANG_ZING_BLOCK0_CACHE_MODE" not in result["environment"]
    assert command[command.index("--quantization") + 1] == (
        "fp8" if profile == "b300" else "none"
    )
    assert command[command.index("--zing-denoising-step-list") + 1] == "1000,500"
    assert "--zing-window-chunks" not in command
    for name in (
        "SP_DESTINATION_MAJOR",
        "FUSED_QKNORM_ROPE_KV",
        "HISTORY_ROPE_LIVE_ROWS",
        "HISTORY_ROPE_CROSS_GRAPH",
    ):
        assert result["environment"]["SGLANG_ZING_" + name] == "1"


@pytest.mark.parametrize(
    "profile,resolution,expected",
    [
        ("b300", "480", "1"),
        ("b300", "736", "0"),
        ("h200", "480", "0"),
        ("h200", "736", "0"),
    ],
)
def test_hardware_and_resolution_specific_fast_path(profile, resolution, expected):
    env = contract(profile, resolution)["environment"]
    assert env["SGLANG_ZING_FA4_FIXED"] == expected
    assert env["SGLANG_ZING_FUSED_CURRENT_ROPE_KV"] == expected


def test_explicit_environment_and_cli_overrides_survive():
    result = contract(
        env={"SGLANG_ZING_HISTORY_ROPE_CROSS_GRAPH": "0"},
        args=[
            "--zing-source-prompt-joint=false",
            "--zing-source-work-quantum=0",
            "--zing-block0-cache-mode",
            "off",
        ],
    )
    assert result["environment"]["SGLANG_ZING_HISTORY_ROPE_CROSS_GRAPH"] == "0"
    assert result["environment_overrides"] == ["SGLANG_ZING_HISTORY_ROPE_CROSS_GRAPH"]
    assert "--zing-source-prompt-joint" not in result["command"]
    assert "--zing-source-work-quantum" not in result["command"]
    assert "--zing-source-work-quantum=0" in result["command"]
    assert result["command"].count("--zing-block0-cache-mode") == 1


def test_qwen_service_gets_all_six_options_before_import():
    result = contract(
        component="conditioning", args=["serve", "--config", "secrets.json"]
    )
    assert result["command"][2] == "sglang.multimodal_gen.runtime.conditioning.cli"
    assert result["command"][3] == "serve"
    for option in (
        "HF_RMSNORM",
        "FUSED_MROPE",
        "FUSED_GATE_UP",
        "TEXT_FA4",
        "VISION_FA4",
        "FP8",
    ):
        assert result["environment"]["SGLANG_H3_QWEN_" + option] == "1"
    assert "--zing-source-prompt-joint" not in result["command"]


def test_rejects_h100_small_partitions_and_mismatched_profiles():
    for device in [gpu("h200", memory=81559), gpu(memory=200000)]:
        with pytest.raises(ValueError, match="No B300/H200 recipe"):
            launcher.resolve_profile("auto", [device])
    with pytest.raises(ValueError, match="does not match"):
        launcher.resolve_profile("h200", [gpu()])
    with pytest.raises(ValueError, match="one hardware family"):
        launcher.resolve_profile("auto", [gpu(), gpu("h200")])


def test_no_default_role_assignment_on_a_single_gpu():
    with pytest.raises(ValueError, match="four visible GPUs"):
        launcher.build_launch_contract(
            requested_profile="auto",
            resolution="480",
            component="server",
            gpus=[gpu()],
            arguments=[],
            environment={},
        )


def test_custom_role_layout_does_not_receive_default_gpu_mapping():
    command = contract(
        args=[
            "--encoder-gpus",
            "0,1",
            "--denoiser-gpus",
            "2,3",
            "--decoder-gpus",
            "4",
        ]
    )["command"]
    assert command.count("--encoder-gpus") == 1
    assert "--num-gpus" not in command


def test_fixed_480_fast_path_keeps_unmeasured_sm_count_off():
    result = launcher.build_launch_contract(
        requested_profile="b300",
        resolution="480",
        component="dit",
        gpus=[gpu(sms=140)] * 2,
        arguments=[],
        environment={},
    )
    assert result["environment"]["SGLANG_ZING_FUSED_CURRENT_ROPE_KV"] == "0"


def test_main_dry_run_does_not_exec(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "detect_gpus", lambda: [gpu("h200")] * 4)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launcher",
            "--resolution",
            "736",
            "--dry-run",
            "--",
            "--model-path",
            "/model",
        ],
    )
    monkeypatch.setattr(
        launcher.os, "execvpe", lambda *args: pytest.fail("dry-run executed")
    )
    launcher.main()
    result = launcher.json.loads(capsys.readouterr().out)
    assert result["profile"] == "h200"
    assert result["command"][-2:] == ["--model-path", "/model"]


def test_main_exec_preserves_environment_and_delivers_recipe(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "validate_launch_assets", lambda *args: None)
    monkeypatch.setattr(launcher, "detect_gpus", lambda: [gpu()] * 4)
    monkeypatch.setattr(sys, "argv", ["launcher", "--resolution", "480"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setenv("SGLANG_ZING_SP_DESTINATION_MAJOR", "0")
    calls = []
    monkeypatch.setattr(launcher.os, "execvpe", lambda *args: calls.append(args))
    launcher.main()
    executable, command, env = calls[0]
    assert executable == sys.executable
    assert command[:3] == [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.launch_server",
    ]
    assert env["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
    assert env["SGLANG_ZING_SP_DESTINATION_MAJOR"] == "0"
    assert env["SGLANG_ZING_HISTORY_ROPE_CROSS_GRAPH"] == "1"


@pytest.mark.parametrize("profile", ["h200", "b300"])
@pytest.mark.parametrize("resolution", ["480", "736"])
def test_dit_only_reuses_external_roles_without_local_worker_mapping(
    profile, resolution
):
    result = launcher.build_launch_contract(
        requested_profile=profile,
        resolution=resolution,
        component="dit",
        gpus=[gpu(profile)] * 2,
        arguments=["--model-path", "/prepared-model"],
        environment={},
    )
    command = result["command"]
    for option in ("--encoder-gpus", "--denoiser-gpus", "--decoder-gpus"):
        assert option not in command
    assert command[command.index("--num-gpus") + 1] == "2"
    assert command[command.index("--sp-degree") + 1] == "2"
    assert command[command.index("--ulysses-degree") + 1] == "2"
    assert command[command.index("--zing-source-decoder-overlap") + 1] == "true"
    assert result["environment"]["SGLANG_ZING_HISTORY_ROPE_CROSS_GRAPH"] == "1"


def test_dit_only_rejects_insufficient_devices_and_local_worker_roles():
    with pytest.raises(ValueError, match="two visible GPUs"):
        launcher.build_launch_contract(
            requested_profile="auto",
            resolution="480",
            component="dit",
            gpus=[gpu()],
            arguments=[],
            environment={},
        )
    with pytest.raises(ValueError, match="cannot start local role"):
        launcher.build_launch_contract(
            requested_profile="auto",
            resolution="480",
            component="dit",
            gpus=[gpu()] * 2,
            arguments=["--encoder-gpus=0"],
            environment={},
        )


@pytest.mark.parametrize("component", ["encoder", "decoder"])
@pytest.mark.parametrize("profile", ["h200", "b300"])
@pytest.mark.parametrize("resolution", ["480", "736"])
def test_standalone_workers_run_torchrun_with_only_component_defaults(
    component, profile, resolution
):
    arguments = ["--model", "/model", "--port", "31003"]
    if component == "decoder":
        arguments += ["--decoder-config", '{"type":"lynn_streaming"}']
    result = launcher.build_launch_contract(
        requested_profile=profile,
        resolution=resolution,
        component=component,
        gpus=[gpu(profile)],
        arguments=arguments,
        environment={},
    )
    command = result["command"]
    assert command[:8] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "1",
        "-m",
        (
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
            f"zing_h3.{component}_worker"
        ),
    ]
    assert command[8:] == arguments
    assert not any(
        key.startswith("SGLANG_ZING_DENOISE") for key in result["environment"]
    )
    assert "--zing-source-decoder-overlap" not in command
    if component == "encoder":
        assert result["environment"]["MINIMAX_H3_PROGRESSIVE_CHANNELS_LAST"] == "1"
        assert (
            result["environment"]["MINIMAX_H3_PROGRESSIVE_ENCODER_PERFORMANCE_AUTO"]
            == "1"
        )
        assert result["environment"]["MINIMAX_H3_PROGRESSIVE_ENCODER_TILING"] == "auto"
        assert "SGLANG_H3_QWEN_FP8" not in result["environment"]
    else:
        # Lynn fusion is automatic; do not invent an unused acceleration switch.
        assert result["environment"] == {}


@pytest.mark.parametrize("component", ["encoder", "decoder"])
def test_standalone_worker_count_and_environment_override(component):
    result = launcher.build_launch_contract(
        requested_profile="auto",
        resolution="736",
        component=component,
        gpus=[gpu()] * 2,
        arguments=[],
        worker_gpus=2,
        environment={"MINIMAX_H3_PROGRESSIVE_CHANNELS_LAST": "0"},
    )
    command = result["command"]
    assert command[command.index("--nproc-per-node") + 1] == "2"
    if component == "encoder":
        assert result["environment"]["MINIMAX_H3_PROGRESSIVE_CHANNELS_LAST"] == "0"
    for count in (0, 3):
        with pytest.raises(ValueError, match="Worker GPU count"):
            launcher.build_launch_contract(
                requested_profile="auto",
                resolution="480",
                component=component,
                gpus=[gpu()] * 2,
                arguments=[],
                worker_gpus=count,
                environment={},
            )


def test_worker_count_is_not_silently_ignored_by_server():
    with pytest.raises(ValueError, match="only to standalone"):
        launcher.build_launch_contract(
            requested_profile="auto",
            resolution="480",
            component="server",
            gpus=[gpu()] * 4,
            arguments=[],
            worker_gpus=2,
            environment={},
        )


@pytest.mark.parametrize("component", ["encoder", "decoder"])
def test_main_worker_mode_exec_preserves_visibility_and_passes_worker_args(
    component, monkeypatch, capsys
):
    monkeypatch.setattr(launcher, "detect_gpus", lambda: [gpu()] * 2)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launcher",
            "--resolution",
            "480",
            "--component",
            component,
            "--worker-gpus",
            "2",
            "--",
            "--model",
            "/model",
            "--port",
            "31003",
        ],
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    monkeypatch.setenv("MINIMAX_H3_PROGRESSIVE_CHANNELS_LAST", "0")
    calls = []
    monkeypatch.setattr(launcher.os, "execvpe", lambda *args: calls.append(args))
    launcher.main()
    _, command, environment = calls[0]
    assert command[2] == "torch.distributed.run"
    assert command[command.index("--nproc-per-node") + 1] == "2"
    assert command[-4:] == ["--model", "/model", "--port", "31003"]
    assert environment["CUDA_VISIBLE_DEVICES"] == "4,5"
    assert environment["MINIMAX_H3_PROGRESSIVE_CHANNELS_LAST"] == "0"
    assert launcher.json.loads(capsys.readouterr().out)["component"] == component


@pytest.mark.parametrize(
    "profile,quantum,overlap", [("h200", "1", "true"), ("b300", "1", "true")]
)
def test_shared_enabled_pipeline_defaults(profile, quantum, overlap):
    # Layout/budgets follow the frozen policy; pipeline/Qwen defaults follow the user-selected enabled configuration.
    command = contract(profile=profile)["command"]

    def value(flag):
        return command[command.index(flag) + 1]

    assert value("--zing-source-work-quantum") == quantum
    assert value("--zing-source-prompt-joint") == "true"
    for flag in (
        "--zing-source-encoder-prefetch",
        "--zing-source-async-output",
        "--zing-source-decoder-overlap",
        "--zing-source-defer-final",
    ):
        assert value(flag) == overlap
    assert value("--denoiser-gpus") == "0,1"
    assert value("--encoder-gpus") == "2"
    assert value("--decoder-gpus") == "3"
    assert value("--realtime-session-lease-ttl-s") == "1200"
    assert value("--realtime-session-idle-timeout-s") == "1200"
    assert value("--realtime-session-max-lifetime-s") == (
        "7800" if profile == "b300" else "7200"
    )
    assert value("--warmup-mode") == "off"
    assert value("--dist-timeout") == "600"
    assert value("--zing-source-encoder-mode") == "overlap"
    assert value("--zing-initial-block-count") == "3"
    assert value("--zing-initial-denoising-step-list") == "1000,750,500,250"
    assert value("--zing-reference-max-images") == "21"
    assert value("--encoder-parallel") == "replicate"
    assert value("--mem-fraction-static") == "0.8"
    assert value("--zing-block0-cache-host-gb") == "416"
    assert value("--zing-block0-cache-disk-gb") == "700"
    assert "--zing-block0-cache-gc" in command


def test_custom_cache_directory_owns_gc_budget_root():
    command = contract(args=["--zing-block0-cache-dir=/owned/cache/instance"])[
        "command"
    ]
    assert (
        command[command.index("--zing-block0-cache-budget-root") + 1] == "/owned/cache"
    )


@pytest.mark.parametrize("profile", ["h200", "b300"])
def test_prepared_assets_reject_policy_conflicts_without_mutation(tmp_path, profile):
    pipeline = {
        "zing_kv_pool_window_chunks": 6,
        "zing_kv_pool_sink_chunks": 0,
        "zing_adaln_mode": "resident" if profile == "b300" else "weights_free",
    }
    if profile == "h200":
        pipeline["zing_adaln_table_path"] = "/prepared/adaln-table"
    realtime = {"source_attention_window": {"window_chunks": 6, "sink_chunks": 0}}
    path, rp = tmp_path / "pipeline.json", tmp_path / "zing_realtime_profile.json"
    path.write_text(json.dumps(pipeline))
    rp.write_text(json.dumps(realtime))
    argv = ["--model-path", str(tmp_path), "--pipeline-config-path=" + str(path)]
    launcher.validate_launch_assets(argv, profile)
    original = path.read_bytes()
    for patch, message in [
        ({"zing_kv_pool_sink_chunks": 3}, "W6/S0"),
        ({"zing_kv_pool_window_eviction_timing": "before_source"}, "after_step0"),
        ({"zing_kv_pool_size": 1}, "retired"),
    ]:
        path.write_text(json.dumps({**pipeline, **patch}))
        with pytest.raises(ValueError, match=message):
            launcher.validate_launch_assets(argv, profile)
    path.write_bytes(original)
    realtime["window_eviction_timing"] = "before_source"
    rp.write_text(json.dumps(realtime))
    with pytest.raises(ValueError, match="after_step0"):
        launcher.validate_launch_assets(argv, profile)
    assert path.read_bytes() == original


def test_explicit_b300_bf16_and_quant_scope_override():
    result = contract(
        args=["--quantization=none"], env={"SGLANG_ZING_QUANT_SCOPE": "ffn"}
    )
    assert "--quantization" not in result["command"]
    assert "--quantization=none" in result["command"]
    assert result["environment"]["SGLANG_ZING_QUANT_SCOPE"] == "ffn"


@pytest.mark.parametrize(
    "profile,minimum", [("h200", 130 * 1024), ("b300", 240 * 1024)]
)
def test_pr647_hardware_memory_gate_uses_gib(profile, minimum):
    assert launcher.resolve_profile("auto", [gpu(profile, memory=minimum)]) == profile
    with pytest.raises(ValueError, match="No B300/H200 recipe"):
        launcher.resolve_profile("auto", [gpu(profile, memory=minimum - 1)])
