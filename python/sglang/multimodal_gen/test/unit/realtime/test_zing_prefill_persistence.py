import json
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.realtime.prefill_persistence import (
    _fingerprint,
    inspect_block0_persistence,
    inspect_persistence,
)


def entry(tmp_path, *, legacy=False):
    identity = {"model": {"checkpoint": "same"}, "layout": {"sp": 2}}
    descriptor = dict(
        key="a" * 64,
        model=_fingerprint(identity["model"]),
        layout=_fingerprint(identity["layout"]),
        sp_world_size=2,
        generation="losing-writer",
    )
    folder = (
        tmp_path
        / "v2"
        / descriptor["model"]
        / descriptor["layout"]
        / "aa"
        / descriptor["key"]
    )
    folder.mkdir(parents=True)
    ranks = {}
    for rank in range(2):
        d = folder / f"rank-{rank}"
        d.mkdir()
        (d / "kv.bin").write_bytes(b"kv00")
        (d / "conditions.pt").write_bytes(b"conditions")
        ranks[str(rank)] = dict(kv_bytes=4, generation="winner")
        if not legacy:
            ranks[str(rank)]["conditions_bytes"] = 10
    manifest = dict(
        schema="zing-block0-v2",
        early_key=descriptor["key"],
        identity=identity,
        sp_world_size=2,
        generation="winner",
        ranks=ranks,
    )
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return descriptor, folder


@pytest.mark.parametrize("legacy", [False, True])
def test_complete_requires_all_shards_and_accepts_compatible_winner(tmp_path, legacy):
    descriptor, folder = entry(tmp_path, legacy=legacy)
    assert inspect_persistence(tmp_path, descriptor)["state"] == "pending"
    (folder / ".complete").touch()
    result = inspect_persistence(tmp_path, descriptor)
    assert result["state"] == "persisted"
    assert result["generation"] == "winner"
    assert result["kv_bytes"] == 8 and result["conditions_bytes"] == 20
    (folder / "rank-1/kv.bin").unlink()
    assert inspect_persistence(tmp_path, descriptor)["state"] == "pending"


def test_corrupt_manifest_or_shard_cannot_report_persisted(tmp_path):
    descriptor, folder = entry(tmp_path)
    (folder / ".complete").touch()
    (folder / "rank-0/kv.bin").write_bytes(b"wrong size")
    assert inspect_persistence(tmp_path, descriptor)["state"] == "failed"
    with pytest.raises(ValueError):
        inspect_persistence(tmp_path, {**descriptor, "key": "../unsafe"})


def test_symlink_escape_rejected(tmp_path):
    descriptor, folder = entry(tmp_path)
    (folder / ".complete").touch()
    path = folder / "rank-0/kv.bin"
    path.unlink()
    path.symlink_to("/etc/hosts")
    assert inspect_persistence(tmp_path, descriptor)["state"] == "failed"


def test_managed_v3_receipt_waits_for_all_native_rank_markers(tmp_path):
    identity = {"zing_config": {"checkpoint_sha256": "test"}, "sp": {"size": 2}}
    descriptor = dict(
        schema="zing-block0-v3",
        disk_format=4,
        key="c" * 64,
        identity_sha256=_fingerprint(identity),
        sp_world_size=2,
    )
    folder = tmp_path / "entries" / descriptor["key"]
    folder.mkdir(parents=True)
    manifest = dict(
        schema="zing-block0-v3",
        disk_format=4,
        early_key=descriptor["key"],
        identity=identity,
        sp_world_size=2,
    )
    (folder / "manifest.json").write_text(json.dumps(manifest))
    assert inspect_persistence(tmp_path, descriptor)["state"] == "pending"
    for rank in range(2):
        base = folder / f"rank-{rank}"
        base.mkdir()
        (base / "kv.bin").write_bytes(b"four")
        (base / "conditions.pt").write_bytes(b"condition")
        (base / "ready.json").write_text(
            json.dumps(
                dict(
                    disk_format=4,
                    conditions_schema="zing-block0-conditions-v1",
                    kv_bytes=4,
                )
            )
        )
    result = inspect_persistence(tmp_path, descriptor)
    assert result["state"] == "persisted" and result["kv_bytes"] == 8
    assert "path" not in result
    # The receipt never needs to read the KV payload or deserialize tensors.
    (folder / "rank-0/kv.bin").write_bytes(b"bad")
    assert inspect_persistence(tmp_path, descriptor)["state"] == "pending"
    assert (
        inspect_persistence(tmp_path, {**descriptor, "identity_sha256": "d" * 64})[
            "state"
        ]
        == "failed"
    )


def test_native_disk_backend_directory_is_used_for_persistence(tmp_path):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache import (
        Block0DiskCache,
    )

    identity = {"checkpoint": "native", "sp": 2}
    disk = Block0DiskCache(tmp_path, identity=identity, world_size=2)
    assert not hasattr(disk, "root")
    descriptor = dict(
        schema="zing-block0-v3",
        disk_format=4,
        key="b" * 64,
        identity_sha256=_fingerprint(identity),
        sp_world_size=2,
    )
    cache = SimpleNamespace(disk_cache=disk)
    assert inspect_block0_persistence(cache, descriptor) == {"state": "pending"}
    assert (
        inspect_block0_persistence(None, descriptor)["code"]
        == "CACHE_PERSISTENCE_UNAVAILABLE"
    )
    assert (
        inspect_block0_persistence(SimpleNamespace(disk_cache=None), descriptor)["code"]
        == "CACHE_PERSISTENCE_UNAVAILABLE"
    )


@pytest.mark.parametrize(
    "cache",
    [None, SimpleNamespace(disk_cache=None), SimpleNamespace(backend="mooncake")],
)
def test_optional_disk_backend_has_no_descriptor_or_fake_persisted_receipt(cache):
    from sglang.multimodal_gen.runtime.realtime.prefill_persistence import (
        persistence_descriptor,
    )

    assert persistence_descriptor(cache, SimpleNamespace(key="a" * 64)) is None
    assert inspect_block0_persistence(cache, {}) == {
        "state": "failed",
        "code": "CACHE_PERSISTENCE_UNAVAILABLE",
    }
