# SPDX-License-Identifier: Apache-2.0
"""Frozen SP1 tensor oracle for cross-SP progressive encoder acceptance."""

import hashlib
import json
from pathlib import Path

import torch

FRAMES = (*range(1, 35), 120)
PROVENANCE_KEYS = (
    "source_sha",
    "rgb_sha256",
    "weights_sha256",
    "config_sha256",
    "vae_config_sha256",
    "script_sha256",
    "stack_tiling",
    "parallel_mode",
    "torch",
    "cuda",
    "compute_capability",
    "gpu_name",
    "nccl",
    "runtime",
)


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SP1Reference:
    def __init__(self, path, provenance, *, write):
        self.path = Path(path)
        self.write = write
        # Match the on-disk representation, including NCCL's version tuple.
        provenance = json.loads(json.dumps(provenance))
        self.manifest = dict(producer_world_size=1, provenance=provenance, files={})
        if write:
            self.path.mkdir(parents=True, exist_ok=False)
        else:
            self.manifest = json.loads((self.path / "manifest.json").read_text())
            if (
                self.manifest.get("producer_world_size") != 1
                or self.manifest.get("provenance") != provenance
                or set(self.manifest.get("files", {})) != {str(n) for n in FRAMES}
            ):
                raise ValueError(
                    "SP1 reference provenance or complete frame set differs"
                )

    def tensors(self, frames, moments=None, posterior=None):
        path = self.path / f"frames-{frames}.pt"
        if self.write:
            torch.save(dict(moments=moments, posterior=posterior), path)
            self.manifest["files"][str(frames)] = dict(
                bytes=path.stat().st_size, sha256=file_digest(path)
            )
            return dict(moments=moments, posterior=posterior)
        entry = self.manifest["files"][str(frames)]
        if (
            path.stat().st_size != entry["bytes"]
            or file_digest(path) != entry["sha256"]
        ):
            raise ValueError("SP1 reference tensor integrity check failed")
        return torch.load(path, map_location="cpu", weights_only=True)

    def finish(self):
        if self.write:
            if set(self.manifest["files"]) != {str(n) for n in FRAMES}:
                raise ValueError("SP1 reference frame set is incomplete")
            (self.path / "manifest.json").write_text(
                json.dumps(self.manifest, indent=2)
            )
