# SPDX-License-Identifier: Apache-2.0
"""CPU contracts against the real independent worker argument parsers."""

import argparse
import ast
import json
import sys
from pathlib import Path

import pytest

from sglang.multimodal_gen.tools import zing_h3_profile_launcher as launcher

ROOT = Path(__file__).parents[2]
STAGES = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3."
)
MODULES = {
    "encoder": "sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server",
    "dit": STAGES + "dit_worker",
    "decoder": STAGES + "decoder_worker",
}


def gpu(profile="h100"):
    return launcher.GPUInfo(
        profile, (9, 0), 81559 if profile == "h100" else 143771, 132
    )


def contract(role, *, arguments=(), gpus=None, sp=1, environment=None, profile="auto"):
    argv = ["--model", "/prepared/model", "--advertise-host", "10.0.0.12", *arguments]
    if role == "decoder":
        argv += ["--decoder-config", '{"type":"lynn_streaming","model_dir":"/lynn"}']
    return launcher.build_launch_contract(
        requested_profile=profile,
        resolution="480",
        component=role,
        gpus=gpus or [gpu("h200" if role == "dit" else "h100")],
        arguments=argv,
        environment=environment or {},
        mode="independent",
        sp_degree=sp,
    )


def worker_parser(role):
    # Execute only the actual parser definition, leaving GPU imports and main untouched.
    relative = (
        MODULES[role].split("sglang.multimodal_gen.", 1)[1].replace(".", "/") + ".py"
    )
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    namespace = {
        "argparse": argparse,
        "Path": Path,
        "__doc__": "worker parser contract",
    }
    if role == "dit":
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "argument_parser"
        )
        exec(  # noqa: S102 - execute trusted in-repo parser definitions only
            compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        return namespace["argument_parser"]()
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    statements = []
    for node in function.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "parse_args"
        ):
            break
        statements.append(node)
    exec(  # noqa: S102 - execute trusted in-repo parser statements only
        compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["parser"]


def native_args(role, result):
    command = result["command"]
    return worker_parser(role).parse_args(command[command.index(MODULES[role]) + 1 :])


@pytest.mark.parametrize(
    "role,profile", [("encoder", "h100"), ("decoder", "h100"), ("dit", "h200")]
)
def test_target_three_role_topology_uses_existing_worker_parsers(role, profile):
    before = set(sys.modules)
    result = contract(role)
    assert "torch" not in set(sys.modules) - before
    assert result["profile"] == profile
    assert result["mode"] == "independent"
    args = native_args(role, result)
    assert args.service_count == 1
    assert args.advertise_host == "10.0.0.12"
    assert "--static-connection" not in result["command"]
    assert "--encoder-gpus" not in result["command"]
    if role == "decoder":
        command = result["command"]
        assert command[2] == "torch.distributed.run"
        assert command[command.index("--nproc-per-node") + 1] == "1"
    else:
        assert result["command"][2] == MODULES[role]
    if role == "dit":
        assert args.quantization is None
        assert "--quantization" not in result["command"]
        assert args.sp_degree == 1
        assert args.mem_fraction_static == 0.8
        assert args.zing_block0_cache_host_gb == 416
        assert args.zing_block0_cache_disk_gb == 700
        assert args.zing_block0_cache_storage == "local"
        assert args.zing_source_defer_final is True


def test_encoder_owns_qwen_and_vae_options_without_dit_graphs():
    env = contract("encoder")["environment"]
    for key in launcher.CONDITIONING_ENV:
        assert env[key] == "1"
    assert env["MINIMAX_H3_PROGRESSIVE_ENCODER_TILING"] == "auto"
    assert not any(k.startswith("SGLANG_ZING_DENOISE") for k in env)
    assert contract("decoder")["environment"] == {}
    dit = contract("dit")["environment"]
    assert not set(launcher.CONDITIONING_ENV).intersection(dit)
    assert not set(launcher.ENCODER_ENV).intersection(dit)
    assert dit["SGLANG_ZING_FA4_FIXED"] == "0"


def test_native_dit_boolean_and_budget_overrides_are_preserved():
    result = contract(
        "dit",
        arguments=[
            "--no-zing-source-defer-final",
            "--mem-fraction-static=0.65",
            "--zing-block0-cache-mode",
            "off",
        ],
    )
    args = native_args("dit", result)
    assert args.zing_source_defer_final is False
    assert args.mem_fraction_static == 0.65
    assert args.zing_block0_cache_mode == "off"
    assert "--zing-source-defer-final" not in result["command"]
    with pytest.raises(ValueError, match="--no-zing-source-defer-final"):
        contract("dit", arguments=["--zing-source-defer-final=false"])


@pytest.mark.parametrize("role", ["encoder", "decoder", "dit"])
@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0", "::", "localhost"])
def test_cross_host_recipe_rejects_loopback_and_wildcards(role, host):
    with pytest.raises(ValueError, match="advertise-host"):
        launcher.build_launch_contract(
            requested_profile="auto",
            resolution="480",
            component=role,
            gpus=[gpu("h200" if role == "dit" else "h100")],
            arguments=["--advertise-host", host],
            environment={},
            mode="independent",
        )


@pytest.mark.parametrize(
    "option", ["--static-connection", "--model-path", "--encoder-gpus", "--num-gpus"]
)
def test_server_or_static_protocol_options_are_rejected(option):
    with pytest.raises(ValueError, match="server/static"):
        contract("decoder", arguments=[option, "1"])


def test_gpu_ranks_and_service_count_boundaries():
    with pytest.raises(ValueError, match="exactly 1"):
        contract("encoder", gpus=[gpu(), gpu()])
    with pytest.raises(ValueError, match="one service"):
        contract("dit", arguments=["--service-count", "2"])
    with pytest.raises(ValueError, match="positive integer"):
        contract("encoder", arguments=["--service-count", "0"])
    for role in ("encoder", "decoder"):
        multi = contract(role, arguments=["--service-count", "3"])
        assert native_args(role, multi).service_count == 3
    with pytest.raises(ValueError, match="one active"):
        contract("decoder", arguments=["--max-sessions", "2"])
    with pytest.raises(ValueError, match="No B300/H200"):
        contract("dit", gpus=[gpu()])
    with pytest.raises(ValueError, match="must match"):
        contract("dit", arguments=["--sp-degree", "2"])
    result = contract("dit", sp=2, gpus=[gpu("h200")] * 2)
    assert native_args("dit", result).sp_degree == 2
    assert result["command"][result["command"].index("--nproc-per-node") + 1] == "2"


@pytest.mark.parametrize("role", ["encoder", "decoder", "dit"])
def test_b300_three_role_topology_and_multi_service(role):
    device = launcher.GPUInfo("B300", (10, 3), 274113, 148)
    extra = ["--service-count", "2"] if role != "dit" else []
    result = contract(role, gpus=[device], arguments=extra, profile="b300")
    assert result["profile"] == "b300"
    args = native_args(role, result)
    assert args.service_count == (2 if role != "dit" else 1)
    if role == "encoder":
        assert args.session_max_lifetime_s == 7800
    elif role == "dit":
        assert args.quantization == "fp8"
        assert result["environment"]["SGLANG_ZING_QUANT_SCOPE"] == "ffn-attention"


def test_independent_main_exec_clears_inherited_ranks_and_preserves_gpu_selection(
    monkeypatch, capsys
):
    monkeypatch.setattr(launcher, "detect_gpus", lambda: [gpu()])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launcher",
            "--mode",
            "independent",
            "--component",
            "encoder",
            "--profile",
            "h100",
            "--resolution",
            "480",
            "--",
            "--model",
            "/model",
            "--advertise-host",
            "10.0.0.11",
        ],
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("SGLANG_H3_QWEN_FP8", "0")
    calls = []
    monkeypatch.setattr(launcher.os, "execvpe", lambda *args: calls.append(args))
    launcher.main()
    _, command, env = calls[0]
    assert command[2] == MODULES["encoder"]
    assert env["CUDA_VISIBLE_DEVICES"] == "4"
    assert env["SGLANG_H3_QWEN_FP8"] == "0"
    assert "RANK" not in env and "WORLD_SIZE" not in env
    assert json.loads(capsys.readouterr().out)["mode"] == "independent"


def test_independent_dry_run_never_starts_workers_or_groups(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "detect_gpus", lambda: [gpu("h200")])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launcher",
            "--mode",
            "independent",
            "--component",
            "dit",
            "--sp-degree",
            "1",
            "--resolution",
            "480",
            "--dry-run",
            "--",
            "--model",
            "/model",
            "--advertise-host",
            "10.0.0.12",
        ],
    )
    monkeypatch.setattr(
        launcher.os, "execvpe", lambda *args: pytest.fail("dry-run executed")
    )
    launcher.main()
    result = json.loads(capsys.readouterr().out)
    assert result["component"] == "dit"
    assert "External client" in result["group_management"]


def test_independent_dit_exec_validates_native_model_assets_before_handoff(
    tmp_path, monkeypatch, capsys
):
    pipeline = {
        "zing_kv_pool_window_chunks": 6,
        "zing_kv_pool_sink_chunks": 0,
        "zing_adaln_table_path": "/prepared/adaln-table",
    }
    realtime = {"source_attention_window": {"window_chunks": 6, "sink_chunks": 0}}
    config = tmp_path / "pipeline.json"
    config.write_text(json.dumps(pipeline))
    (tmp_path / "zing_realtime_profile.json").write_text(json.dumps(realtime))
    monkeypatch.setattr(launcher, "detect_gpus", lambda: [gpu("h200")])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launcher",
            "--mode",
            "independent",
            "--component",
            "dit",
            "--resolution",
            "480",
            "--",
            "--model=" + str(tmp_path),
            "--pipeline-config-path",
            str(config),
            "--advertise-host",
            "10.0.0.12",
        ],
    )
    calls = []
    monkeypatch.setattr(launcher.os, "execvpe", lambda *args: calls.append(args))
    launcher.main()
    assert len(calls) == 1
    assert calls[0][1][2] == MODULES["dit"]
    assert "--model-path" not in calls[0][1]
    assert json.loads(capsys.readouterr().out)["mode"] == "independent"
    pipeline["zing_kv_pool_sink_chunks"] = 3
    config.write_text(json.dumps(pipeline))
    with pytest.raises(ValueError, match="W6/S0"):
        launcher.main()
    assert len(calls) == 1


def test_b300_independent_dit_keeps_native_quantization_scope():
    device = launcher.GPUInfo("B300", (10, 3), 274113, 148)
    result = contract("dit", gpus=[device])
    assert native_args("dit", result).quantization == "fp8"
    assert result["environment"]["SGLANG_ZING_QUANT_SCOPE"] == "ffn-attention"
    h200_fp8 = contract("dit", arguments=["--quantization", "fp8"])
    assert native_args("dit", h200_fp8).quantization == "fp8"
    assert h200_fp8["environment"]["SGLANG_ZING_QUANT_SCOPE"] == "ffn-attention"
    assert native_args("dit", h200_fp8).zing_initial_block_count == 0
    assert native_args("dit", h200_fp8).zing_initial_denoising_step_list is None


@pytest.mark.parametrize(
    "option,first,last,message",
    [
        ("--service-count", "1", "2", "one service"),
        ("--advertise-host", "10.0.0.12", "127.0.0.1", "advertise-host"),
        ("--sp-degree", "1", "2", "must match"),
    ],
)
def test_final_native_argument_value_controls_role_validation(
    option, first, last, message
):
    with pytest.raises(ValueError, match=message):
        contract("dit", arguments=[option, first, option + "=" + last])


@pytest.mark.parametrize("option", ["--static", "--service-c", "--advertise-h"])
def test_abbreviated_protocol_or_admission_options_are_rejected(option):
    with pytest.raises(ValueError, match="complete worker option names"):
        contract("decoder", arguments=[option, "1"])


def test_topology_workaround_and_scoped_h200_memory_policy():
    from sglang.multimodal_gen.tools import zing_h3_role_launcher as role_launcher

    topo = role_launcher.resolve_topology_workaround("cross_host_sm90_h100_h200")
    assert topo["environment"] == {"NCCL_MIN_NCHANNELS": "2", "NCCL_MIN_CTAS": "2"}
    assert "2.27.3" in topo["nccl_version"]
    assert role_launcher.resolve_topology_workaround(None) == {}
    with pytest.raises(ValueError, match="Unknown topology"):
        role_launcher.resolve_topology_workaround("unknown_topology")

    # 0.875 applies only to exact model1027 + h200 + 480 + bf16 + sp1 + block0_cache=on.
    assert (
        role_launcher.resolve_scoped_dit_memory_fraction(
            profile="h200",
            resolution="480",
            sp_degree=1,
            quantization=None,
            cache_mode="on",
            model_id="model1027",
        )
        == "0.875"
    )
    base_params = dict(
        profile="h200",
        resolution="480",
        sp_degree=1,
        quantization=None,
        cache_mode="on",
        model_id="model1027",
    )
    for override in (
        {"model_id": None},
        {"model_id": "zing-h3-dmd-1102-step2000-generator"},
        {"profile": "b300"},
        {"resolution": "736"},
        {"sp_degree": 2},
        {"quantization": "fp8"},
        {"cache_mode": "off"},
    ):
        params = {**base_params, **override}
        assert role_launcher.resolve_scoped_dit_memory_fraction(**params) == "0.8"
