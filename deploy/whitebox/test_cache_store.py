import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_store import settings


def configuration():
    return json.loads(
        (Path(__file__).parent / "cache/test-shanghai-copy.example.json").read_text()
    )


def test_native_store_uses_single_host_lru_and_real_disk_budget():
    config = configuration()
    policy, command, env, store = settings(config, 1024 << 30)
    assert store["global_segment_size"] == 48 << 30
    assert store["protocol"] == "rdma"
    assert store["rdma_devices"] == "mlx5_0"
    assert store["enable_ssd_offload"] is True
    assert "--promotion_on_hit=true" in command
    assert env["MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY"] == "lru"
    assert int(env["MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES"]) == policy.disk_bytes
    assert policy.prefix.endswith("test-shanghai-copy/")
    config["master_server_address"] = "other-host:50051"
    with pytest.raises(ValueError, match="this host"):
        settings(config, 1024 << 30)


def test_disabled_l2_does_not_initialize_disk_or_offload():
    config = configuration()
    config["tiers"].update(l2=False, l1_to_l2=False, l2_to_l3=False, l1_to_l3=True)
    _, command, env, store = settings(config)
    assert "--enable_offload=false" in command
    assert store["enable_ssd_offload"] is False
    assert not any(key.startswith("MOONCAKE_OFFLOAD") for key in env)


def test_release_wires_shared_cache_policy_without_starting_store_per_slot():
    from config import ROOT, load, render
    from model_service import commands

    policy = load("shanghai")
    policy["model"]["cache_backend"] = "mooncake"
    policy["environment"]["room"]["WHITEBOX_CACHE_RECORDING_ENABLED"] = "1"
    bindings = json.loads((ROOT / "bindings.example.json").read_text())
    bindings["cache_config"] = "/etc/sglang/cache.json"
    bundle = render(policy, bindings)
    argv = commands(bundle)["dit"]["argv"]
    assert argv[argv.index("--zing-block0-cache-backend") + 1] == "mooncake"
    assert (
        argv[argv.index("--zing-block0-mooncake-config") + 1]
        == bindings["cache_config"]
    )
    assert "WHITEBOX_CACHE_RECORDING_SECRET" in bundle["required_secret_keys"]["model"]
    assert "WHITEBOX_CACHE_RECORDING_SECRET" in bundle["required_secret_keys"]["room"]
    assert all(
        "cache_store.py" not in " ".join(command["argv"])
        for command in commands(bundle).values()
    )


@pytest.mark.parametrize(
    "path",
    sorted((Path(__file__).parent / "cache").glob("*.example.json")),
    ids=lambda path: path.stem,
)
def test_region_requires_rdma(path):
    config = json.loads(path.read_text())
    assert config["protocol"] == "rdma"
    config["protocol"] = "tcp"
    with pytest.raises(ValueError, match="protocol"):
        settings(config, 1024 << 30)
