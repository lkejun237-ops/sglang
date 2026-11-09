# SPDX-License-Identifier: Apache-2.0
"""Native main-compatible raw KV IO, including existing hashed cache entries."""

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache import (
    DiskEntry,
    _read_kv,
    _write_kv,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_native_kv_io_preserves_values_and_unused_slot_rows(tmp_path, dtype, strided):
    source = torch.arange(2 * 2 * 3 * 2 * 3).reshape(2, 2, 3, 2, 3).to(dtype)
    path = tmp_path / "kv.bin"
    # Upstream writes raw tensor bytes; it does not compute a payload digest.
    assert _write_kv(path, source) is None
    assert path.stat().st_size == source.numel() * source.element_size()
    backing = torch.full((2, 2, 8, 2, 3), -1, dtype=dtype)
    target = backing[:, :, :6:2] if strided else backing[:, :, :3]
    assert _read_kv(path, target) is target
    assert torch.equal(target, source)
    assert torch.all(backing[:, :, 6:] == -1)
    if strided:
        assert torch.all(backing[:, :, 1:6:2] == -1)


def test_previously_hashed_entry_loads_without_full_payload_checksum(tmp_path):
    source = torch.arange(24, dtype=torch.float32).reshape(2, 2, 3, 2)
    folder = tmp_path / "rank-0"
    folder.mkdir()
    _write_kv(folder / "kv.bin", source)
    # Older deployment metadata is tolerated, just as by main; no cache rewrite
    # or eviction is needed when removing the additional checksum extension.
    entry = DiskEntry("key", tmp_path, {"ranks": {"0": {"sha256": "legacy"}}}, 0, {})
    target = torch.empty_like(source)
    assert entry.load_kv_into(target) is target
    assert torch.equal(target, source)


@pytest.mark.parametrize("size_delta", [-1, 1])
def test_native_reader_still_rejects_wrong_file_size(tmp_path, size_delta):
    target = torch.empty((2, 2, 3, 2), dtype=torch.bfloat16)
    path = tmp_path / "kv.bin"
    path.write_bytes(b"\0" * (target.numel() * target.element_size() + size_delta))
    with pytest.raises(ValueError, match="file size"):
        _read_kv(path, target)
