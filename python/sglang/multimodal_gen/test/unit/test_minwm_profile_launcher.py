import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.configs.minwm_resolution_buckets import (
    normalize_minwm_resolution_buckets,
)
from sglang.multimodal_gen.tools import minwm_profile_launcher as launcher
from sglang.multimodal_gen.tools.minwm_profile_launcher import (
    PROFILE_RESIDENT_SPEED,
    PROFILE_SM120_32G_SPEED,
    PROFILE_SM120_HIGHMEM_SPEED,
    GPUInfo,
    build_launch_contract,
    resolve_profile,
)


def _last_option_value(command: list[str], option: str) -> str | None:
    value = None
    for index, argument in enumerate(command):
        if argument == option:
            value = command[index + 1] if index + 1 < len(command) else None
        elif argument.startswith(f"{option}="):
            value = argument.split("=", 1)[1]
    return value


@pytest.mark.parametrize(
    ("gpu", "expected"),
    [
        (GPUInfo("GeForce RTX 5090", (12, 0), 32768), PROFILE_SM120_32G_SPEED),
        (
            GPUInfo("RTX PRO 6000 Blackwell Server Edition", (12, 0), 97887),
            PROFILE_SM120_HIGHMEM_SPEED,
        ),
        (GPUInfo("NVIDIA H200", (9, 0), 143771), PROFILE_RESIDENT_SPEED),
        (GPUInfo("NVIDIA B200", (10, 0), 183359), PROFILE_RESIDENT_SPEED),
        (GPUInfo("NVIDIA B300", (10, 3), 281000), PROFILE_RESIDENT_SPEED),
    ],
)
def test_auto_profile_selection(gpu, expected):
    assert resolve_profile("auto", gpu) == expected


def test_resolution_bucket_aliases_normalize_to_exact_dimensions():
    assert normalize_minwm_resolution_buckets("480p,720p,720p-wide") == (
        "832x480",
        "1248x704",
        "1280x704",
    )


def test_resolution_bucket_alias_duplicates_are_rejected():
    with pytest.raises(ValueError, match="duplicate canonical resolutions"):
        normalize_minwm_resolution_buckets("720p,1248x704")


def test_auto_profile_rejects_unvalidated_sm120_memory_size():
    with pytest.raises(ValueError, match="does not match a validated"):
        resolve_profile("auto", GPUInfo("Future SM120", (12, 0), 49152))


def test_auto_profile_rejects_undersized_b300_partition():
    with pytest.raises(ValueError, match="validated B300 profile"):
        resolve_profile("auto", GPUInfo("NVIDIA B300", (10, 3), 200000))


def test_explicit_sm120_profile_rejects_wrong_architecture():
    with pytest.raises(ValueError, match="requires SM120"):
        resolve_profile(
            PROFILE_SM120_32G_SPEED,
            GPUInfo("RTX 6000 Ada", (8, 9), 49140),
        )


def test_explicit_32g_profile_can_be_qualified_on_high_memory_sm120():
    assert (
        resolve_profile(
            PROFILE_SM120_32G_SPEED,
            GPUInfo("RTX PRO 6000 Blackwell Server Edition", (12, 0), 97887),
        )
        == PROFILE_SM120_32G_SPEED
    )


def test_auto_profile_rejects_ada():
    with pytest.raises(ValueError, match="unsupported MinWM inference GPU"):
        resolve_profile("auto", GPUInfo("RTX 6000 Ada", (8, 9), 49140))


def test_32g_contract_offloads_only_text_encoder():
    contract = build_launch_contract(
        requested_profile="auto",
        gpu=GPUInfo("GeForce RTX 5090", (12, 0), 32768),
        taehv_path=Path("/models/taehv/taew2_2.pth"),
        server_args=["--model-path", "/models/minwm", "--port", "30000"],
        validate_artifacts=False,
    )

    command = contract["command"]
    assert contract["resolved_profile"] == PROFILE_SM120_32G_SPEED
    assert command[:3] == [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.launch_server",
    ]
    assert command[command.index("--text-encoder-cpu-offload") + 1] == "true"
    assert command[command.index("--vae-cpu-offload") + 1] == "false"
    assert command[command.index("--dit-cpu-offload") + 1] == "false"
    assert command[command.index("--dit-layerwise-offload") + 1] == "false"
    assert contract["environment"]["MINWM_ATTENTION_IMPL"] == "packed"
    assert contract["environment"]["MINWM_PACKED_ATTENTION_DETERMINISTIC"] == "false"
    assert contract["environment"]["SGLANG_MINWM_REQUIRE_SM120_FA4"] == "1"
    assert contract["realtime_kv_cache_pool_buckets"] == ["832x480"]
    assert command[command.index("--realtime-causal-kv-cache-pool-size") + 1] == "1"
    assert command[command.index("--realtime-max-sessions") + 1] == "1"
    assert command[command.index("--realtime-max-sessions-per-worker") + 1] == "1"
    assert (
        command[command.index("--realtime-causal-kv-cache-pool-buckets") + 1]
        == "832x480"
    )
    assert "1248x704" not in command
    assert "1280x704" not in command


@pytest.mark.parametrize(
    "gpu",
    [
        GPUInfo("NVIDIA H100", (9, 0), 81559),
        GPUInfo("NVIDIA H200", (9, 0), 143771),
        GPUInfo("NVIDIA B200", (10, 0), 183359),
        GPUInfo("NVIDIA B300", (10, 3), 281000),
        GPUInfo("RTX PRO 6000 Blackwell Server Edition", (12, 0), 97887),
    ],
)
def test_high_memory_profiles_enable_both_exact_720p_resolutions(gpu):
    contract = build_launch_contract(
        requested_profile="auto",
        gpu=gpu,
        taehv_path=Path("/models/taehv/taew2_2.pth"),
        server_args=["--model-path", "/models/minwm"],
        validate_artifacts=False,
    )

    assert contract["realtime_kv_cache_pool_buckets"] == [
        "832x480",
        "1248x704",
        "1280x704",
    ]
    command = contract["command"]
    assert (
        command[command.index("--realtime-causal-kv-cache-pool-buckets") + 1]
        == "832x480,1248x704,1280x704"
    )


def test_profile_allows_parallelism_overrides_for_sp2():
    contract = build_launch_contract(
        requested_profile="auto",
        gpu=GPUInfo("GeForce RTX 5090", (12, 0), 32768, visible_device_count=2),
        taehv_path=Path("/models/taehv/taew2_2.pth"),
        server_args=[
            "--model-path",
            "/models/minwm",
            "--num-gpus",
            "2",
            "--sp-degree",
            "2",
            "--ulysses-degree",
            "2",
            "--ring-degree",
            "1",
            "--enable-cuda-graph",
            "true",
        ],
        validate_artifacts=False,
    )

    command = contract["command"]
    assert contract["gpu"]["visible_device_count"] == 2
    assert _last_option_value(command, "--num-gpus") == "2"
    assert _last_option_value(command, "--sp-degree") == "2"
    assert _last_option_value(command, "--ulysses-degree") == "2"
    assert _last_option_value(command, "--ring-degree") == "1"
    assert _last_option_value(command, "--enable-cuda-graph") == "true"


def test_profile_rejects_managed_override():
    with pytest.raises(ValueError, match="cannot be overridden"):
        build_launch_contract(
            requested_profile="auto",
            gpu=GPUInfo("GeForce RTX 5090", (12, 0), 32768),
            taehv_path=Path("/models/taehv/taew2_2.pth"),
            server_args=[
                "--model-path",
                "/models/minwm",
                "--dit-layerwise-offload",
                "true",
            ],
            validate_artifacts=False,
        )


@pytest.mark.parametrize(
    "option",
    [
        "--realtime-causal-kv-cache-pool-size",
        "--realtime-causal-kv-cache-pool-buckets",
    ],
)
def test_profile_rejects_kv_pool_override(option):
    with pytest.raises(ValueError, match="cannot be overridden"):
        build_launch_contract(
            requested_profile="auto",
            gpu=GPUInfo("GeForce RTX 5090", (12, 0), 32768),
            taehv_path=Path("/models/taehv/taew2_2.pth"),
            server_args=["--model-path", "/models/minwm", option, "480p"],
            validate_artifacts=False,
        )


def test_b300_multi_gpu_topology_can_override_parallelism():
    contract = build_launch_contract(
        requested_profile="auto",
        gpu=GPUInfo("NVIDIA B300", (10, 3), 281000, visible_device_count=8),
        taehv_path=Path("/models/taehv/taew2_2.pth"),
        server_args=[
            "--model-path",
            "/models/minwm",
            "--num-gpus",
            "8",
            "--sp-degree",
            "8",
            "--ulysses-degree",
            "8",
        ],
        validate_artifacts=False,
    )

    assert contract["resolved_profile"] == PROFILE_RESIDENT_SPEED
    assert contract["gpu"]["visible_device_count"] == 8
    assert contract["command"].count("--num-gpus") == 1
    assert contract["command"].count("--sp-degree") == 1
    assert contract["command"].count("--ulysses-degree") == 1
    assert contract["command"][-6:] == [
        "--num-gpus",
        "8",
        "--sp-degree",
        "8",
        "--ulysses-degree",
        "8",
    ]


def test_topology_rejects_more_gpus_than_are_visible():
    with pytest.raises(ValueError, match="visible CUDA device count"):
        build_launch_contract(
            requested_profile="auto",
            gpu=GPUInfo("GeForce RTX 5090", (12, 0), 32768),
            taehv_path=Path("/models/taehv/taew2_2.pth"),
            server_args=["--model-path", "/models/minwm", "--num-gpus", "2"],
            validate_artifacts=False,
        )


def test_detect_gpu_accepts_homogeneous_multi_gpu(monkeypatch):
    gib = 1024**3
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_capability=lambda index: (10, 3),
        get_device_name=lambda index: "NVIDIA B300",
        get_device_properties=lambda index: SimpleNamespace(total_memory=288 * gib),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))

    assert launcher._detect_gpu() == GPUInfo(
        "NVIDIA B300",
        (10, 3),
        288 * 1024,
        visible_device_count=2,
    )


def test_detect_gpu_rejects_mixed_gpu_families(monkeypatch):
    gib = 1024**3
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_capability=lambda index: ((10, 3), (12, 0))[index],
        get_device_name=lambda index: ("NVIDIA B300", "GeForce RTX 5090")[index],
        get_device_properties=lambda index: SimpleNamespace(total_memory=32 * gib),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))

    with pytest.raises(ValueError, match="homogeneous visible CUDA GPUs"):
        launcher._detect_gpu()


def test_contract_validates_taehv_and_tianpeng_model(tmp_path, monkeypatch):
    taehv_path = tmp_path / "taew2_2.pth"
    taehv_path.write_bytes(b"test-taehv")
    monkeypatch.setattr(
        launcher,
        "TAEHV_SHA256",
        hashlib.sha256(taehv_path.read_bytes()).hexdigest(),
    )
    model_path = tmp_path / "model"
    config_path = model_path / "transformer" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(launcher.EXPECTED_MODEL_CONFIG))

    contract = build_launch_contract(
        requested_profile="auto",
        gpu=GPUInfo("GeForce RTX 5090", (12, 0), 32768),
        taehv_path=taehv_path,
        server_args=["--model-path", str(model_path)],
    )

    assert contract["resolved_profile"] == PROFILE_SM120_32G_SPEED
