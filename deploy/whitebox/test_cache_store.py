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
    assert "--root_fs_dir=" in command
    assert store["enable_ssd_offload"] is False
    assert not any(key.startswith("MOONCAKE_OFFLOAD") for key in env)


def test_native_ssd_does_not_enable_a_second_filesystem_copy():
    """root_fs_dir adds DISK writes on Put alongside the Store's LOCAL_DISK copy."""
    policy, command, env, store = settings(configuration(), 1024 << 30)
    assert "--root_fs_dir=" in command
    assert not any(
        arg.startswith("--root_fs_dir=") and arg != "--root_fs_dir=" for arg in command
    )
    assert "--enable_offload=true" in command
    assert store["enable_ssd_offload"] is True
    assert store["ssd_offload_path"] == policy.disk_directory
    assert env["MOONCAKE_OFFLOAD_FILE_STORAGE_PATH"] == policy.disk_directory


def test_native_probe_diagnostics_are_not_store_configuration(tmp_path, monkeypatch):
    """Runtime provenance accidentally serialized into tiers must fail locally."""
    import importlib.util
    import types

    from sglang.multimodal_gen.runtime.realtime.cache_tiers import CacheTiers

    root = Path(__file__).resolve().parents[2]
    path = root / "benchmark/zing_h3_causal/check_native_cache_tiers.py"
    spec = importlib.util.spec_from_file_location("native_cache_probe", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    package = types.ModuleType("mooncake")
    store = types.ModuleType("mooncake.store")
    store.MooncakeDistributedStore = store.ReplicateConfig = object
    monkeypatch.setitem(sys.modules, "mooncake", package)
    monkeypatch.setitem(sys.modules, "mooncake.store", store)

    class ProbeStopped(Exception):
        pass

    def inspect_config(argv, **_):
        value = json.loads(Path(argv[-1]).read_text())
        CacheTiers.parse(value["tiers"])
        raise ProbeStopped()

    monkeypatch.setattr(probe.subprocess, "Popen", inspect_config)
    with pytest.raises(ProbeStopped):
        probe.main(tmp_path / "run", "mlx5_0", runtime_root=root, allow_overlay_l2=True)
    receipt = json.loads((tmp_path / "run/receipt.json").read_text())
    assert receipt["runtime_root"] == str(root)


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
