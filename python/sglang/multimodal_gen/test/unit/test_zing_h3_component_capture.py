# SPDX-License-Identifier: Apache-2.0
"""Direct CPU coverage of the opt-in component capture writer."""

import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.component_capture import (
    ComponentCapture,
    pool_metadata,
)


@dataclasses.dataclass(frozen=True)
class Inputs:
    tensor: torch.Tensor
    condition_media: object


def writer(tmp_path, inputs=None):
    stage = SimpleNamespace(
        runtime={"steps": 4},
        config=SimpleNamespace(dit_config=SimpleNamespace(arch_config={})),
    )
    return ComponentCapture(tmp_path, stage, inputs or {}, {}, 42, "session")


def test_pool_capture_reconstructs_capacity_and_runtime_admission_policy():
    from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        ZingKVGeometry,
        ZingKVPoolConfig,
        ZingKVPoolPlan,
    )

    geometry = ZingKVGeometry(16, (1, 2, 2), 480, ChunkPlan((3, 2)))
    plan = ZingKVPoolPlan.build(
        ZingKVPoolConfig(),
        geometry,
        token_capacity=50000,
        runtime_budget_bytes=12 << 30,
        local_hidden_width=2048,
        prefix_hidden_width=4096,
    )
    stage = SimpleNamespace(
        kv_pool=SimpleNamespace(plan=plan), config=SimpleNamespace(dit_precision="bf16")
    )
    receipt = pool_metadata(stage)
    assert receipt["schema_version"] == 2
    assert "pool_size" not in receipt["config"]  # init=False is not a constructor arg.
    restored = ZingKVPoolPlan.build(
        ZingKVPoolConfig(**receipt["config"]),
        geometry,
        token_capacity=receipt["slot_rows"],
        **receipt["runtime_policy"],
    )
    assert restored == plan


def test_capture_owns_tensor_snapshot_without_mutating_inputs(tmp_path):
    value = torch.ones(2, requires_grad=True)
    inputs = Inputs(value, object())
    capture = writer(tmp_path, inputs)
    with torch.no_grad():
        value.add_(1)
    saved = torch.load(capture.root / "session.pt", weights_only=False)
    assert torch.equal(saved["inputs"].tensor, torch.ones(2))
    assert not saved["inputs"].tensor.requires_grad
    assert saved["inputs"].condition_media is None
    assert inputs.condition_media is not None
    with pytest.raises(FileExistsError):
        writer(tmp_path)


def test_capture_complete_receipt_matches_saved_components(tmp_path):
    capture = writer(tmp_path)
    moments = torch.ones(1, 48, 1, 1, 1)
    latent = torch.zeros(1, 24, 1, 1, 1)
    capture.append_source(
        SimpleNamespace(start=0, end=1, source_dependency_end=1, values=moments), latent
    )
    capture.append({"source": latent}, SimpleNamespace(visual=latent))
    frames = torch.arange(12, dtype=torch.uint8).reshape(3, 1, 2, 2)
    for kind in ("input", "decoder"):
        capture.append_frames(kind, frames, 0)
    assert not (capture.root / "COMPLETE.json").exists()
    capture.finish()
    assert json.loads((capture.root / "COMPLETE.json").read_text()) == {
        "schema_version": 3,
        "capture_scope": "full",
        "chunks": 1,
        "sources": 1,
        "input_frames": 1,
        "decoder_frames": 1,
    }
    saved = torch.load(capture.root / "source-00000.pt", weights_only=False)
    assert torch.equal(saved["moments"], moments)
    assert torch.equal(saved["latent"], latent)
    record = json.loads((capture.root / "frames.json").read_text())["decoder"][0]
    assert (
        record["sha256"] == hashlib.sha256(frames[:, 0].numpy().tobytes()).hexdigest()
    )


def test_capture_rejects_gaps_and_nonfinite_frames(tmp_path):
    capture = writer(tmp_path)
    values = torch.ones(3, 1, 2, 2)
    with pytest.raises(ValueError, match="contiguous"):
        capture.append_frames("input", values, 1)
    values[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        capture.append_frames("input", values, 0)
    assert capture.frames["input"] == []
