# SPDX-License-Identifier: Apache-2.0
"""CPU-only startup, namespace, byte-budget and persisted-shard contracts."""

import json
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    block0_identity,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.block0_conditions import (
    CONDITIONS_SCHEMA,
    encode_conditions,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache import (
    Block0DiskCache,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
    Block0PreparationCache,
    preparation_key,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
    _source_key_policy,
)
from sglang.multimodal_gen.test.unit.realtime.test_zing_prompt_switch_cache import (  # noqa: F401
    cpu_dma,
    disk_commit,
)


def _write_pickle_marker(path):
    Path(path).write_text("Unrestricted pickle executed")


class _PicklePayload:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return _write_pickle_marker, (self.marker,)


def _assert_snapshot_equal(expected, actual):
    assert type(actual) is type(expected)
    if isinstance(expected, torch.Tensor):
        assert actual.device.type == "cpu"
        assert expected.dtype == actual.dtype and torch.equal(expected, actual)
    elif is_dataclass(expected):
        for field in fields(expected):
            _assert_snapshot_equal(
                getattr(expected, field.name), getattr(actual, field.name)
            )
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for name in expected:
            _assert_snapshot_equal(expected[name], actual[name])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for a, b in zip(expected, actual):
            _assert_snapshot_equal(a, b)
    else:
        assert actual == expected


def identity(sp=1, model="weights-a"):
    return {
        "model": {"dit": model, "qwen": "qwen-a", "vae": "vae-a"},
        "layout": {"sp": sp, "dtype": "bfloat16", "schema": block0_identity.SCHEMA},
    }


def disk(root, *, sp=1, rank=0, model="weights-a", budget=1 << 24):
    root.mkdir(parents=True, exist_ok=True)
    return Block0DiskCache(
        root, max_bytes=budget, identity=identity(sp, model), rank=rank, world_size=sp
    )


def test_startup_off_does_not_require_or_touch_storage():
    assert block0_identity.startup_config(NS(zing_block0_cache_mode="off")) is None


def test_no_gpu_pool_skips_cache_even_with_missing_parameters(monkeypatch, caplog):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
        ZingKVPoolConfig,
    )

    monkeypatch.setattr(ZingKVPoolConfig, "from_pipeline_config", lambda *a, **kw: None)
    stage = object.__new__(ZingH3GenerationStage)
    with caplog.at_level("INFO"):
        stage.initialize_realtime_kv_cache_pool(
            NS(pipeline_config=object(), realtime_max_sessions_per_worker=1)
        )
    assert stage.block0_cache is None and stage.kv_pool is None
    assert "no GPU KV pool" in caplog.text


@pytest.mark.parametrize("value", [None, 0, -1, True, float("nan"), float("inf")])
def test_invalid_or_missing_budget_fails_startup(value):
    with pytest.raises(ValueError, match="positive decimal GB"):
        block0_identity.budget_bytes(value, "budget")


@pytest.mark.parametrize("mode", ["late", "early", "overlap", "bad"])
def test_production_mode_has_only_on_and_off(mode):
    with pytest.raises(ValueError, match="on or off"):
        block0_identity.startup_config(NS(zing_block0_cache_mode=mode))


def test_budget_is_decimal_gb_and_missing_mount_is_not_created(tmp_path, monkeypatch):
    args = NS(
        zing_block0_cache_mode="on",
        zing_block0_cache_host_gb=800,
        zing_block0_cache_disk_gb=4000,
        zing_block0_cache_dir=str(tmp_path / "unmounted"),
    )
    with pytest.raises(ValueError, match="existing mounted directory"):
        block0_identity.startup_config(args)
    assert not (tmp_path / "unmounted").exists()
    args.zing_block0_cache_dir = str(tmp_path)
    monkeypatch.setattr(block0_identity, "storage_mounted", lambda root: False)
    with pytest.raises(ValueError, match="mounted before"):
        block0_identity.startup_config(args)
    monkeypatch.setattr(block0_identity, "storage_mounted", lambda root: True)
    host, budget, root = block0_identity.startup_config(args)
    assert host == 800_000_000_000 and budget == 4_000_000_000_000 and root == tmp_path


def test_pinned_slots_are_sized_by_bytes_and_floor_unused_remainder(cpu_dma):
    store = NS(k=torch.zeros(8, 2, 3), v=torch.zeros(8, 2, 3))
    pool = NS(
        _pool=NS(_slots=[NS(kv_cache=[store])]), plan=NS(config=NS(max_prefix_rows=8))
    )
    slot_bytes = 2 * 8 * 2 * 3 * 4
    owner = Block0PreparationCache(
        pool, max_bytes=3 * slot_bytes + 17, identity=identity(), disk_cache=None
    )
    assert owner.data.shape[0] == 3
    assert owner.data.numel() * owner.data.element_size() == 3 * slot_bytes
    with pytest.raises(ValueError, match="one maximum-prefix"):
        Block0PreparationCache(
            pool, max_bytes=slot_bytes - 1, identity=identity(), disk_cache=None
        )


def test_key_scopes_every_model_and_layout_but_not_rank():
    sampler = NS(
        window_policy=NS(window_chunks=6, sink_chunks=3),
        rope_position_mode="block_relative",
        rope_max_frame_gap=12,
    )
    stage = NS(
        codecs=NS(video_decoder_name="lynn"), block0_cache=NS(identity=identity())
    )

    def key():
        policy, _, _ = _source_key_policy(stage, sampler, 1, {})
        return preparation_key(
            NS(remote_read_fs=None),
            {"prompt": "scene"},
            None,
            {"width": 32, "height": 32, "fps": 24},
            policy,
        )

    original = key()
    for component in ("dit", "qwen", "vae"):
        stage.block0_cache.identity = identity()
        stage.block0_cache.identity["model"][component] += "-changed"
        assert key() != original
    stage.block0_cache.identity = identity(sp=2)
    assert key() != original
    stage.block0_cache.identity = identity()
    assert key() == original
    assert block0_identity.fingerprint({"a": 1, "b": 2}) == block0_identity.fingerprint(
        {"b": 2, "a": 1}
    )


def test_asset_content_not_mount_path_or_file_length_is_identity(tmp_path):
    for name in ("a", "b"):
        folder = tmp_path / name / "vae"
        folder.mkdir(parents=True)
        (folder / "weights.bin").write_bytes(b"original")
    before = block0_identity._assets(tmp_path / "a/vae")
    assert before == block0_identity._assets(tmp_path / "b/vae")
    (tmp_path / "a/vae/weights.bin").write_bytes(b"modified")
    assert before != block0_identity._assets(tmp_path / "a/vae")


def test_reuse_verified_adaln_identity_without_scanning_dit_again(tmp_path):
    for name in ("text_encoder", "tokenizer", "processor", "video_vae"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "asset.bin").write_bytes(name.encode())
    checkpoint = {
        "config_sha256": "config",
        "shards": {"weights.safetensors": "digest"},
    }
    table = NS(checkpoint_verified=True, manifest={"identity": checkpoint})
    stage = NS(
        model_path=str(tmp_path),
        modules={"transformer": NS(adaln_table=table)},
        conditioning_mode="local",
    )
    args = NS(model_paths={}, component_paths={})
    scanned = block0_identity._scan_models(stage, args)
    assert scanned["dit"] == checkpoint
    assert not (tmp_path / "transformer").exists()
    table.checkpoint_verified = False
    with pytest.raises(ValueError, match="verified AdaLN"):
        block0_identity._scan_models(stage, args)


def test_sp_identity_scan_broadcast_and_numeric_contract_without_cuda(monkeypatch):
    from sglang.multimodal_gen.runtime.conditioning import worker
    from sglang.multimodal_gen.runtime.distributed import parallel_state
    from sglang.multimodal_gen.runtime.models.dits import zing_h3_adaln_table

    @dataclass
    class Arch:
        hidden_size: int = 4

    group = NS(rank_in_group=0, world_size=2, ring_world_size=1)
    monkeypatch.setattr(parallel_state, "get_sp_group", lambda: group)
    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parallel_state, "get_ulysses_parallel_world_size", lambda: 2)
    monkeypatch.setattr(worker, "coordinate_validation", lambda value: [value, value])
    monkeypatch.setattr(worker, "broadcast_receipt", lambda value: value)
    monkeypatch.setattr(
        zing_h3_adaln_table,
        "arithmetic_contract",
        lambda device: {"device": str(device)},
    )
    scanned = []
    monkeypatch.setattr(
        block0_identity,
        "_scan_models",
        lambda *args: scanned.append(True) or identity()["model"],
    )
    stage = NS(
        modules={"transformer": torch.nn.Linear(2, 2)},
        conditioning_mode="local",
        config=NS(dit_config=NS(arch_config=Arch()), vae_config=NS(arch_config=Arch())),
        runtime={
            key: {}
            for key in (
                "vae_config",
                "rope_config",
                "condition_encoder_config",
                "h3_media_config",
                "num_frame_per_block",
                "split_first_latent",
            )
        },
    )
    args = NS(
        quantization=None,
        component_quantizations={},
        quantization_ignored_layers=None,
        component_quantization_ignored_layers={},
        component_precisions={},
        lora_scale=1.0,
        lora_alpha=None,
        lora_merge_mode="auto",
        lora_weight_name=None,
        enable_torch_compile=False,
        enable_cuda_graph=False,
        attention_backend=None,
        attention_backend_config=None,
        component_attention_backends={},
    )
    result = block0_identity.build_identity(stage, args, device="cpu")
    assert len(scanned) == 1 and result["layout"]["sp"] == 2
    group.rank_in_group = 1
    monkeypatch.setattr(
        worker,
        "broadcast_receipt",
        lambda value: {"ok": True, "models": result["model"]},
    )
    assert block0_identity.build_identity(stage, args, device="cpu") == result
    assert len(scanned) == 1
    monkeypatch.setattr(
        worker, "coordinate_validation", lambda value: [value, ("rank failed", None)]
    )
    with pytest.raises(ValueError, match="differs or is invalid"):
        block0_identity.build_identity(stage, args, device="cpu")


def test_sp2_only_publishes_after_both_shards_and_keeps_conditions_per_rank(tmp_path):
    a, b = disk(tmp_path, sp=2), disk(tmp_path, sp=2, rank=1)
    ka, _ = disk_commit(a, "case", value=1, rank=0)
    folder = a._entry_dir("case")
    assert not (folder / ".complete").exists()
    assert a.lookup("case") is None and b.lookup("case") is None
    kb, _ = disk_commit(b, "case", value=2, rank=1)
    assert (folder / ".complete").is_file()
    for owner, expected in ((a, ka), (b, kb)):
        hit = owner.lookup("case")
        assert hit is not None
        target = torch.empty_like(expected)
        owner.load_kv_into(hit, target)
        assert torch.equal(target, expected)
        assert torch.equal(
            hit.conditions["inputs"].positive["context"],
            torch.full_like(
                hit.conditions["inputs"].positive["context"],
                expected[0, 0, 0, 0, 0].item(),
            ),
        )
    (folder / "rank-1/kv.bin").write_bytes(b"truncated")
    assert a.lookup("case") is None and b.lookup("case") is None


def test_competing_fill_generations_never_mix_sp_shards(tmp_path):
    owner = disk(tmp_path, sp=2)
    disk_commit(owner, "case", rank=0)
    manifest_path = owner._entry_dir("case") / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["generation"] = "abandoned-other-deployment"
    manifest_path.write_text(json.dumps(manifest))
    disk_commit(owner, "case", rank=1)
    assert owner.lookup("case") is None
    assert not (owner._entry_dir("case") / ".complete").exists()
    disk_commit(owner, "case", rank=0)
    assert owner.lookup("case") is not None


def test_restart_identity_mismatch_legacy_and_tampering_are_misses(tmp_path):
    owner = disk(tmp_path)
    kv, _ = disk_commit(owner, "case")
    restarted = disk(tmp_path)
    hit = restarted.lookup("case")
    target = torch.empty_like(kv)
    restarted.load_kv_into(hit, target)
    assert torch.equal(kv, target)
    assert disk(tmp_path, model="other-weights").lookup("case") is None
    assert disk(tmp_path, sp=2).lookup("case") is None
    legacy = tmp_path / "ca/case"
    legacy.mkdir(parents=True)
    (legacy / ".complete").touch()
    path = owner._entry_dir("case") / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["identity"]["model"]["dit"] = "wrong"
    path.write_text(json.dumps(manifest))
    assert owner.lookup("case") is None
    assert owner.lookup("missing-legacy-key") is None


def test_same_byte_size_wrong_dtype_or_geometry_is_rejected(tmp_path):
    owner = disk(tmp_path)
    kv, _ = disk_commit(owner, "case")
    hit = owner.lookup("case")
    for target in (
        torch.empty_like(kv, dtype=torch.float16),
        torch.empty(kv.numel(), dtype=kv.dtype),
    ):
        with pytest.raises(ValueError, match="shape/dtype"):
            owner.load_kv_into(hit, target)


def test_more_than_fifty_cases_and_root_wide_budget_across_models(tmp_path):
    a = disk(tmp_path)
    for index in range(55):
        disk_commit(a, f"case-{index:02d}")
    index = json.loads(a.index_path.read_text())
    assert index["stats"]["total_entries"] == 55
    assert "max_entries" not in index["stats"]
    limit = index["stats"]["total_bytes"] // 2
    b = disk(tmp_path, model="model-b", budget=limit)
    disk_commit(b, "new")
    index = json.loads(a.index_path.read_text())
    assert index["stats"]["total_bytes"] <= limit
    assert not a._entry_dir("case-00").exists()
    assert b.lookup("new") is not None
    # A corrupt root index rebuilds both namespaces, not just this worker's.
    b.index_path.write_text("broken")
    assert b.lookup("new") is not None
    entries = json.loads(b.index_path.read_text())["entries"]
    assert any(e["dir"].startswith(f"v2/{a.model}/") for e in entries)
    assert any(e["dir"].startswith(f"v2/{b.model}/") for e in entries)


def test_read_failure_after_startup_is_a_miss(tmp_path):
    owner = disk(tmp_path)
    disk_commit(owner, "case")
    (owner._entry_dir("case") / "rank-0/conditions.pt").write_bytes(b"broken")
    assert owner.lookup("case") is None


def test_manifest_without_identity_is_never_accepted(tmp_path):
    owner = disk(tmp_path)
    disk_commit(owner, "case")
    path = owner._entry_dir("case") / "manifest.json"
    value = deepcopy(json.loads(path.read_text()))
    del value["identity"]
    path.write_text(json.dumps(value))
    assert owner.lookup("case") is None


def test_safe_conditions_roundtrip_rebuilds_inputs_and_pool_request(tmp_path):
    owner = disk(tmp_path)
    disk_commit(owner, "case")
    before = owner.lookup("case").conditions
    path = owner._entry_dir("case") / "rank-0/conditions.pt"
    prepared = before["inputs"]
    prepared.source_visual = torch.arange(8).reshape(1, 2, 1, 2, 2).bfloat16()
    prepared.audio = {"ref": torch.ones(2, 3, 4)}
    prepared.audio_shape = (2, 3, 4)
    prepared.sample_count = 100
    prepared.max_area_config = {"max_area": 1024}
    torch.save(encode_conditions(before), path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == CONDITIONS_SCHEMA
    assert payload["records"] == {
        "inputs": "H3InferenceInputs",
        "request": "ZingKVRequest",
    }
    assert type(payload["conditions"]["inputs"]) is dict
    assert type(payload["conditions"]["request"]) is dict
    after = disk(tmp_path).lookup("case").conditions
    _assert_snapshot_equal(before, after)


def test_pickle_payload_is_rejected_without_execution_or_fallback(tmp_path):
    owner = disk(tmp_path)
    disk_commit(owner, "case")
    path = owner._entry_dir("case") / "rank-0/conditions.pt"
    payload = torch.load(path, weights_only=True)
    marker = tmp_path / "pickle-executed"
    payload["conditions"]["request"] = _PicklePayload(marker)
    torch.save(payload, path)
    assert owner.lookup("case") is None
    assert not marker.exists()


@pytest.mark.parametrize(
    "corruption", ["legacy", "unknown_record", "missing_field", "tensor_type", "shape"]
)
def test_unsafe_or_invalid_conditions_are_misses(tmp_path, corruption):
    owner = disk(tmp_path)
    disk_commit(owner, "case")
    path = owner._entry_dir("case") / "rank-0/conditions.pt"
    payload = torch.load(path, weights_only=True)
    if corruption == "legacy":
        payload = payload["conditions"]
    elif corruption == "unknown_record":
        payload["records"]["inputs"] = "attacker.ChosenClass"
    elif corruption == "missing_field":
        del payload["conditions"]["inputs"]["positive"]
    elif corruption == "tensor_type":
        payload["conditions"]["inputs"]["positive"]["context"] = "not a tensor"
    else:
        payload["conditions"]["inputs"]["visual_shape"] = (1, 2)
    torch.save(payload, path)
    assert owner.lookup("case") is None


def test_legacy_conditions_manifest_is_rejected_before_deserialization(
    tmp_path, monkeypatch
):
    owner = disk(tmp_path)
    disk_commit(owner, "case")
    path = owner._entry_dir("case") / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["ranks"]["0"]["conditions_schema"]
    path.write_text(json.dumps(manifest))
    calls = []
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: calls.append(kwargs))
    assert owner.lookup("case") is None
    assert calls == []


def test_writer_rejects_custom_objects_before_serialization(tmp_path):
    owner = disk(tmp_path)
    disk_commit(owner, "case")
    conditions = owner.lookup("case").conditions
    marker = tmp_path / "pickle-executed"
    conditions["request"] = _PicklePayload(marker)
    with pytest.raises(ValueError, match="plain data"):
        encode_conditions(conditions)
    assert not marker.exists()
