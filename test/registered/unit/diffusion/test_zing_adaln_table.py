# SPDX-License-Identifier: Apache-2.0
"""CPU artifact integrity, admission and full-batch numerical contracts."""

import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

spec = importlib.util.spec_from_file_location(
    "zing_table",
    Path(__file__).resolve().parents[4]
    / "python/sglang/multimodal_gen/runtime/models/dits/zing_h3_adaln_table.py",
)
table_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(table_api)


class Embedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
        self.register_buffer("_frequency_cache", torch.tensor([1.0]), persistent=False)

    def forward(self, t):
        return self.linear(torch.stack((torch.cos(t), torch.sin(t)), dim=1))


class Projection(torch.nn.Module):
    def __init__(self, modalities, parts):
        super().__init__()
        self.modalities, self.parts = modalities, parts
        self.linear = torch.nn.Linear(4, modalities * parts * 8).bfloat16()

    def forward(self, embedding):
        return tuple(
            self.linear(embedding).view(-1, self.parts * 8).chunk(self.parts, -1)
        )


@pytest.fixture
def artifact(tmp_path):
    torch.manual_seed(257)
    arch = SimpleNamespace(
        num_layers=2,
        hidden_size=8,
        time_embed_dim=4,
        timestep_input_dim=2,
        time_embed_hidden_size=4,
        num_modalities=5,
    )
    tags = dict(video=4, text=0, audio=3, source_visual=1, condition_visual=2)
    embedder = Embedder().eval()
    projections = [Projection(5, 6).eval() for _ in range(2)] + [
        Projection(1, 2).eval()
    ]
    plan = table_api.video_plan(
        "euler",
        torch.tensor([0.0, 0.5]),
        0.0,
        0.999,
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
    )
    path = tmp_path / "table"
    table_api.write_table(
        path,
        arch=arch,
        tags=tags,
        embedder=embedder,
        projections=zip([0, 1, "final"], projections),
        plan=plan,
        identity={"synthetic": True},
        provenance={"fixture": True},
    )
    return path, arch, tags, embedder, projections, plan


def verify_synthetic_checkpoint(table, directory):
    shard, config = directory / "weights.safetensors", directory / "config.json"
    shard.write_bytes(b"synthetic shard")
    config.write_text("{}")
    table.manifest["identity"] = table_api.checkpoint_identity([shard], config)
    table.validate_checkpoint([shard], config)


def open_table(artifact, max_bytes=1 << 20, verified=True):
    path, arch, tags, *_ = artifact
    table = table_api.ZingAdalnTable(path, arch=arch, tags=tags, max_bytes=max_bytes)
    if verified:
        verify_synthetic_checkpoint(table, path.parent)
    return table


def test_full_batch_values_strides_and_shared_plan_lifetime(artifact):
    _, _, _, embedder, modules, plan = artifact
    table = open_table(artifact)
    batches = [torch.tensor(v) for v in plan["batches"]]
    first = table.prepare(
        batches, device="cpu", schedule=plan["schedule"], embedder=embedder
    )
    addresses = {name: v.data_ptr() for name, v in table.values.items()}
    second = table.prepare(
        batches, device="cpu", schedule=plan["schedule"], embedder=embedder
    )
    assert table.active_plans == 2
    for t in batches:
        with torch.no_grad():
            embedding = torch.nn.functional.silu(embedder(t)).bfloat16()
            for slot, module in zip([0, 1, "final"], modules):
                actual = table.project(slot, None, embedding, table.lookup(t))
                for a, b in zip(actual, module(embedding)):
                    assert torch.equal(a, b) and a.stride() == b.stride()
                    assert torch.isfinite(a).all()
    first.close()
    first.close()
    assert table.active_plans == 1
    second.close()
    assert table.active_plans == 0
    assert addresses == {name: v.data_ptr() for name, v in table.values.items()}
    assert table.stats()["fallback"] == 0


def test_concurrent_plan_close_is_idempotent(artifact):
    from concurrent.futures import ThreadPoolExecutor

    _, _, _, embedder, _, plan = artifact
    table = open_table(artifact)
    lease = table.prepare(
        [torch.tensor(v) for v in plan["batches"]],
        device="cpu",
        schedule=plan["schedule"],
        embedder=embedder,
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: lease.close(), range(32)))
    assert table.active_plans == 0


@pytest.mark.parametrize(
    "fault",
    [
        "schedule",
        "order",
        "sigmas",
        "missing",
        "arithmetic",
        "embedder",
        "frequencies",
        "autocast",
    ],
)
def test_admission_rejects_incompatible_plan_before_allocating(artifact, fault):
    _, _, _, embedder, _, plan = artifact
    table = open_table(artifact)
    batches = [torch.tensor(v) for v in plan["batches"]]
    schedule = dict(plan["schedule"])
    if fault == "schedule":
        schedule["solver"] = "few_step"
    elif fault == "order":
        schedule["video_order"] = table_api.tensor_description(torch.tensor([0.5, 0.0]))
    elif fault == "sigmas":
        schedule["video_sigmas"] = table_api.tensor_description(
            torch.tensor([1.0, 0.5])
        )
    elif fault == "missing":
        batches.append(torch.tensor([0.12345]))
    elif fault == "arithmetic":
        table.manifest["arithmetic"]["device"] = "cuda"
    elif fault == "embedder":
        with torch.no_grad():
            embedder.linear.weight.add_(0.1)
    elif fault == "frequencies":
        embedder._frequency_cache.add_(1)
    context = torch.autocast("cpu", dtype=torch.bfloat16, enabled=fault == "autocast")
    with context, pytest.raises(ValueError):
        table.prepare(batches, device="cpu", schedule=schedule, embedder=embedder)
    assert table.device is None and table.active_plans == 0


@pytest.mark.parametrize(
    "fault", ["file", "tensor", "shape", "missing", "tags", "schema", "raw", "budget"]
)
def test_corrupt_table_rejected(artifact, fault):
    path, arch, tags, *_ = artifact
    manifest = json.loads((path / "manifest.json").read_text())
    if fault in {"file", "tensor", "shape", "missing"}:
        values = load_file(str(path / "table.safetensors"))
        key = next(iter(values))
        if fault == "missing":
            del values[key]
        elif fault == "shape":
            values[key] = values[key][:-1].contiguous()
        else:
            values[key][0, 0] += 1
        save_file(values, str(path / "table.safetensors"))
        if fault != "file":
            manifest["table_sha256"] = table_api.file_digest(path / "table.safetensors")
    elif fault == "tags":
        tags = dict(tags, source_visual=0)
    elif fault == "schema":
        manifest["schema"] = "unrecognized"
    elif fault == "raw":
        manifest["batches"][0]["raw_hex"] = "0000803f"
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        table_api.ZingAdalnTable(
            path, arch=arch, tags=tags, max_bytes=0 if fault == "budget" else 1 << 20
        )


@pytest.mark.parametrize(
    "values",
    [
        torch.tensor([0.5, 0.0]),
        torch.tensor([0.5, 0.5]),
        torch.tensor([float("nan")]),
        torch.tensor([1.1]),
        torch.tensor([0.5], dtype=torch.float64),
    ],
)
def test_no_approximate_or_reordered_batch_key(values):
    with pytest.raises(ValueError):
        table_api.batch_key(values)


def test_loader_filter_preserves_time_embedder_and_token_refiner():
    for name in [
        "backbone.transformer_blocks.1.adaln_proj.linear.weight",
        "backbone.norm_out.linear.bias",
    ]:
        assert not table_api.keep_zing_weight(name)
    for name in [
        "backbone.time_embedder.linear_1.weight",
        "backbone.norm_out.norm.weight",
        "backbone.token_refiner.refiner_blocks.0.attn.to_q.weight",
    ]:
        assert table_api.keep_zing_weight(name)


def test_unverified_table_rejects_admission(artifact):
    _, _, _, embedder, _, plan = artifact
    table = open_table(artifact, verified=False)
    with pytest.raises(ValueError, match="identity validation"):
        table.prepare(
            [torch.tensor([0.0])],
            device="cpu",
            schedule=plan["schedule"],
            embedder=embedder,
        )
    assert table.active_plans == 0 and table.device is None


def test_concurrent_admission_hashes_embedder_once(artifact, monkeypatch):
    _, _, _, embedder, _, plan = artifact
    table = open_table(artifact)
    calls = []
    real = table_api.time_identity
    monkeypatch.setattr(
        table_api, "time_identity", lambda value: calls.append(1) or real(value)
    )
    leases = []

    def admit():
        leases.append(
            table.prepare(
                [torch.tensor([0.0])],
                device="cpu",
                schedule=plan["schedule"],
                embedder=embedder,
            )
        )

    threads = [threading.Thread(target=admit) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(leases) == 8 and len(calls) == 1 and table.active_plans == 8
    with torch.no_grad():
        next(embedder.parameters()).add_(1.0)
    with pytest.raises(ValueError, match="time embedder"):
        admit()
    [lease.close() for lease in leases]


def test_checkpoint_identity_detects_changed_shard_and_config(artifact, tmp_path):
    table = open_table(artifact)
    shard, config = tmp_path / "weights.safetensors", tmp_path / "config.json"
    shard.write_bytes(b"original")
    config.write_text("{}")
    table.manifest["identity"] = table_api.checkpoint_identity([shard], config)
    table.validate_checkpoint([shard], config)
    shard.write_bytes(b"new model")
    with pytest.raises(ValueError, match="checkpoint identity"):
        table.validate_checkpoint([shard], config)
    shard.write_bytes(b"original")
    config.write_text('{"num_modalities":4}')
    with pytest.raises(ValueError, match="checkpoint identity"):
        table.validate_checkpoint([shard], config)


def test_staleness_names_each_cause_that_requires_a_rebuild(artifact, tmp_path):
    path, *_, plan = artifact
    shard, config = tmp_path / "weights.safetensors", tmp_path / "config.json"
    shard.write_bytes(b"original")
    config.write_text("{}")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["identity"] = table_api.checkpoint_identity([shard], config)
    (path / "manifest.json").write_text(json.dumps(manifest))
    stale = lambda **kw: table_api.table_staleness(
        path, [shard], config, kw.get("deployment", {"fixture": True}), kw.get("schedule")
    )
    assert stale(schedule=plan["schedule"]) == []
    assert stale(deployment={"lora_merged": False}) == [
        "effective checkpoint or LoRA provenance changed"
    ]
    assert stale(schedule={**plan["schedule"], "condition": 1.0}) == ["schedule changed"]
    shard.write_bytes(b"merged lora")
    assert stale() == ["checkpoint shards changed: weights.safetensors"]
    config.write_text('{"x":1}')
    assert "export config changed" in stale()
    assert table_api.table_staleness(tmp_path / "none", [shard], config, {}) == [
        "table manifest missing or unreadable"
    ]


def test_table_key_changes_with_every_rebuild_trigger():
    base = dict(
        identity={"c": "1"},
        provenance={"p": 1},
        schedule={"s": [1.0]},
        arithmetic={"gpu": "L20D", "torch": "2.13"},
    )
    key = table_api.table_key(**base)
    assert key == table_api.table_key(**base)
    for field, value in [
        ("identity", {"c": "2"}),
        ("provenance", {"p": 2}),
        ("schedule", {"s": [0.5]}),
        ("arithmetic", {"gpu": "H100", "torch": "2.13"}),
    ]:
        assert table_api.table_key(**{**base, field: value}) != key


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
