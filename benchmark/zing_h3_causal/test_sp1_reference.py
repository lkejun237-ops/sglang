# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
from sp1_reference import FRAMES, SP1Reference


def make_reference(path):
    writer = SP1Reference(path, {"source_sha": "frozen"}, write=True)
    for frames in FRAMES:
        writer.tensors(frames, torch.tensor([frames]), torch.tensor([frames + 1]))
    writer.finish()
    return path


def test_sp1_reference_roundtrip_and_binding(tmp_path):
    path = make_reference(tmp_path / "reference")
    reader = SP1Reference(path, {"source_sha": "frozen"}, write=False)
    assert reader.tensors(120)["posterior"].item() == 121
    with pytest.raises(ValueError, match="provenance"):
        SP1Reference(path, {"source_sha": "different"}, write=False)


def test_sp1_reference_nccl_tuple_survives_json_roundtrip(tmp_path):
    path = tmp_path / "reference"
    provenance = {"source_sha": "frozen", "nccl": (2, 27, 3)}
    writer = SP1Reference(path, provenance, write=True)
    for frames in FRAMES:
        writer.tensors(frames, torch.tensor([frames]), torch.tensor([frames + 1]))
    writer.finish()
    reader = SP1Reference(path, provenance, write=False)
    assert reader.tensors(120)["moments"].item() == 120
    with pytest.raises(ValueError, match="provenance"):
        SP1Reference(path, {**provenance, "nccl": (2, 27, 4)}, write=False)


@pytest.mark.parametrize("change", ["missing", "extra", "producer", "modified_tensor"])
def test_sp1_reference_rejects_invalid_evidence(tmp_path, change):
    path = make_reference(tmp_path / "reference")
    manifest = json.loads((path / "manifest.json").read_text())
    if change == "missing":
        manifest["files"].pop("1")
    elif change == "extra":
        manifest["files"]["121"] = manifest["files"]["120"]
    elif change == "producer":
        manifest["producer_world_size"] = 2
    else:
        (path / "frames-1.pt").write_bytes(b"modified")
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SP1 reference"):
        SP1Reference(path, {"source_sha": "frozen"}, write=False).tensors(1)


def test_sp1_reference_cannot_publish_incomplete_run(tmp_path):
    writer = SP1Reference(tmp_path / "reference", {}, write=True)
    writer.tensors(1, torch.ones(1), torch.ones(1))
    with pytest.raises(ValueError, match="incomplete"):
        writer.finish()
    assert not (tmp_path / "reference/manifest.json").exists()
