# SPDX-License-Identifier: Apache-2.0
"""Paged Block0 arithmetic, file compatibility and transfer ownership on CPU."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    preparation_cache,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache import (
    _read_kv,
    _write_kv,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.paged_kv import (
    PagedKV,
)
from sglang.multimodal_gen.test.unit.realtime.test_zing_preparation_cache import inputs
from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import (  # noqa: F401
    cpu_cache,
    cpu_dma,
    dense_kv,
)


def owner(*, pages=6, page_rows=4, max_rows=16):
    stores = [
        NS(k=torch.zeros(max_rows + 1, 1, 2), v=torch.zeros(max_rows + 1, 1, 2))
        for _ in range(2)
    ]
    # Exact budget, independent of maximum entry rows.
    pool = NS(
        _pool=NS(_slots=[NS(kv_cache=stores)]),
        plan=NS(config=NS(max_prefix_rows=max_rows)),
    )
    return preparation_cache.Block0PreparationCache(
        pool,
        max_bytes=pages * 2 * 2 * page_rows * 2 * 4,
        identity={},
        disk_cache=None,
        page_rows=page_rows,
    )


def fill(cache, key, rows, value=1):
    layers = [
        (
            torch.arange(rows * 2).reshape(rows, 1, 2).float() + value + 100 * layer,
            torch.arange(rows * 2).reshape(rows, 1, 2).float() - value - 100 * layer,
        )
        for layer in range(2)
    ]
    source = NS(
        prefix=NS(positions=torch.zeros(rows, 3), layers=layers),
        _model_signature="model",
    )
    cache.fill(key, inputs(), "condition", "request", source)
    return source


def conserved(cache):
    assert len(cache.free_pages) == cache.total_pages
    ids = [page for page, free in enumerate(cache.free_pages) if free]
    ids += [page for e in cache.entries.values() for page in e.page_ids]
    if cache.pending is not None:
        ids += list(cache.pending[0].page_ids)
    assert sorted(ids) == list(range(cache.total_pages))


@pytest.mark.parametrize("rows", [2047, 2048, 2049, 2 * 2048 + 13])
@pytest.mark.parametrize("page_ids", [(0, 1, 2), (0, 1, 4), (0, 2, 4)])
def test_nonadjacent_pages_preserve_order_and_exclude_tail(
    tmp_path, cpu_dma, rows, page_ids
):
    cache = owner(pages=6, page_rows=2048, max_rows=3 * 2048)
    cache.pages.fill_(float("nan"))
    # Other live entries create holes; the allocator must preserve their pages.
    occupied = tuple(page for page in range(cache.total_pages) if page not in page_ids)
    cache.entries["occupied"] = NS(page_ids=occupied, disk_task=None)
    cache.free_pages = [page in page_ids for page in range(cache.total_pages)]
    source = fill(cache, "a", rows)
    conserved(cache)  # fill has not yet published its pending pages.
    cache.finish_fill()
    entry = cache.entries["a"]
    assert entry.page_ids == page_ids[: len(entry.page_ids)]
    slices = list(entry.data.slices())
    runs = 1 + sum(b != a + 1 for a, b in zip(entry.page_ids, entry.page_ids[1:]))
    assert len(slices) == 2 * 2 * runs
    for _, _, _, value in slices:
        assert value.is_contiguous()
        assert (
            value.untyped_storage().data_ptr()
            == cache.pages.untyped_storage().data_ptr()
        )
    expected = torch.stack([torch.stack(pair) for pair in source.prefix.layers])
    assert torch.equal(dense_kv(entry.data), expected)
    tail = rows % cache.page_rows
    if tail:
        assert cache.pages[:, :, entry.page_ids[-1], tail:].isnan().all()
    path = tmp_path / "kv.bin"
    _write_kv(path, entry.data)
    assert path.stat().st_size == expected.numel() * expected.element_size()
    dense = torch.empty_like(expected)
    _read_kv(path, dense)
    assert torch.equal(dense, expected)
    # The old dense format can also be read directly into a different page map.
    _write_kv(path, expected)
    alternate = PagedKV(
        torch.full_like(cache.pages, float("nan")),
        tuple(reversed(entry.page_ids)),
        rows,
    )
    _read_kv(path, alternate)
    assert torch.equal(dense_kv(alternate), expected)
    stores = [
        NS(k=torch.full((rows + 1, 1, 2), 77.0), v=torch.full((rows + 1, 1, 2), 77.0))
        for _ in range(2)
    ]
    destination = NS(storage=stores, _reserve_store=lambda layer, rows, k, v: (k, v))
    cache.restore(destination, entry).end.synchronize()
    for layer, store in enumerate(stores):
        assert torch.equal(store.k[:rows], expected[layer, 0])
        assert torch.equal(store.v[:rows], expected[layer, 1])
        assert torch.all(store.k[rows:] == 77) and torch.all(store.v[rows:] == 77)
    conserved(cache)


def test_lru_evicts_enough_small_entries_and_oversize_preserves_owners(cpu_dma):
    cache = owner(pages=4)
    for key in "abcd":
        fill(cache, key, 3)
        cache.finish_fill()
    cache.lookup("a")  # b,c,d,a
    fill(cache, "large", 10)
    assert list(cache.entries) == ["a"]
    assert cache.evictions == 3
    conserved(cache)
    cache.finish_fill()
    before = (list(cache.entries), list(cache.free_pages))
    # Within model row limit but greater than this host budget.
    small = owner(pages=2)
    fill(small, "a", 3)
    small.finish_fill()
    with pytest.raises(ValueError, match="page budget"):
        fill(small, "too-large", 9)
    assert list(small.entries) == ["a"]
    conserved(small)
    assert before == (list(cache.entries), list(cache.free_pages))


def test_copy_submission_failure_drains_before_returning_pages(cpu_dma, monkeypatch):
    cache = owner()
    original = torch.Tensor.copy_
    copies = 0
    drained = []

    def copy(target, source, **kwargs):
        nonlocal copies
        copies += 1
        if copies == 3:
            raise RuntimeError("copy submission failed")
        return original(target, source, **kwargs)

    def drain():
        assert cache.pending is not None
        conserved(cache)
        drained.append(True)

    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    cache.stream.synchronize.side_effect = drain
    with pytest.raises(RuntimeError, match="copy submission"):
        fill(cache, "a", 9)
    assert drained and cache.pending is None and not cache.entries
    conserved(cache)
    monkeypatch.setattr(torch.Tensor, "copy_", original)
    fill(cache, "next", 3)
    cache.finish_fill()
    conserved(cache)


def test_failed_drain_keeps_pages_owned_until_recovery(cpu_dma, monkeypatch):
    cache = owner()
    monkeypatch.setattr(
        torch.Tensor,
        "copy_",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("copy")),
    )
    cache.stream.synchronize.side_effect = RuntimeError("drain")
    with pytest.raises(RuntimeError, match="drain"):
        fill(cache, "a", 5)
    assert cache.pending is not None and not cache.entries
    conserved(cache)
    cache.stream.synchronize.side_effect = None
    cache.finish_fill()
    assert cache.pending is None
    conserved(cache)


@pytest.mark.parametrize("write_fails", [False, True])
def test_reuse_waits_for_entry_writer_and_dma(cpu_dma, monkeypatch, write_fails):
    cache = owner(pages=1)
    writer_started, release_writer, waiting = (threading.Event() for _ in range(3))
    cache.disk_cache = NS(local_rank=0, is_enabled=lambda: True)

    def write(entry):
        writer_started.set()
        assert release_writer.wait(5)
        if write_fails:
            raise OSError("disk failed")
        return True

    monkeypatch.setattr(cache, "_write_disk", write)
    fill(cache, "a", 3)
    cache.finish_fill()
    assert writer_started.wait(5)
    entry = cache.entries["a"]
    done = entry.disk_task

    def wait():
        waiting.set()
        return done.wait(5)

    entry.disk_task = NS(wait=wait)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(cache._take_pages, "b", 3)
        try:
            assert waiting.wait(5)
            assert not future.done()
            assert list(cache.entries) == ["a"] and not any(cache.free_pages)
        finally:
            release_writer.set()
        data = future.result(timeout=5)
    assert data.page_ids == entry.page_ids
    cache.stream.synchronize.assert_called()
    # Caller now owns these pages until fill/preload publishes or releases them.
    cache._return_pages(data.page_ids)
    conserved(cache)


def test_bitmap_scan_reuses_holes_in_ascending_order_and_rejects_double_free(cpu_dma):
    cache = owner(pages=6, max_rows=24)
    for key in "abcdef":
        fill(cache, key, 4)
        cache.finish_fill()
    for key in "bde":
        cache._release(key)
    assert cache.free_pages == [False, True, False, True, True, False]
    fill(cache, "new", 9)
    assert cache.pending[0].page_ids == (1, 3, 4)
    assert not any(cache.free_pages)
    conserved(cache)
    cache.finish_fill()
    cache._release("new")
    before = list(cache.free_pages)
    with pytest.raises(ValueError, match="returned more than once"):
        cache._return_pages((1, 3, 4))
    assert cache.free_pages == before
    with pytest.raises(ValueError, match="returned more than once"):
        cache._return_pages((0, 0))
    assert cache.free_pages == before
    conserved(cache)


def test_preload_protects_resident_selection_even_after_disk_delete(tmp_path, cpu_dma):
    from sglang.multimodal_gen.test.unit.realtime.test_zing_block0_cache_contract import (
        disk,
    )
    from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import disk_commit

    writer = disk(tmp_path)
    storage = [NS(k=torch.zeros(8, 2, 2, dtype=torch.bfloat16)) for _ in range(2)]
    cache = cpu_cache(storage, rows=8, entries=2, page_rows=4)
    cache.disk_cache = writer
    for key in "abc":
        disk_commit(writer, key * 64, rows=5)
    assert cache.manage("preload", ["b" * 64, "c" * 64])["all_ready"]
    writer.delete_entries(keys=["b" * 64])
    result = cache.manage("preload", ["a" * 64, "b" * 64])
    assert result["all_ready"]
    assert set(cache.entries) == {"a" * 64, "b" * 64}
    before = list(cache.entries)
    disk_commit(writer, "d" * 64, rows=5)
    result = cache.manage("preload", ["a" * 64, "b" * 64, "d" * 64])
    assert result["code"] == "cache_capacity_exceeded"
    assert result["requested_pages"] == 6 and result["total_pages"] == 4
    assert "capacity" not in result and list(cache.entries) == before
    conserved(cache)


def test_preload_read_failure_returns_all_pages(tmp_path, cpu_dma, monkeypatch):
    from sglang.multimodal_gen.test.unit.realtime.test_zing_block0_cache_contract import (
        disk,
    )
    from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import disk_commit

    writer = disk(tmp_path)
    disk_commit(writer, "a" * 64, rows=5)
    cache = cpu_cache(
        [NS(k=torch.zeros(8, 2, 2, dtype=torch.bfloat16)) for _ in range(2)],
        rows=8,
        entries=1,
        page_rows=4,
    )
    cache.disk_cache = writer

    def fail(entry, data):
        next(data.slices())[-1].fill_(99)
        raise EOFError("truncated after first page")

    original = writer.load_kv_into
    monkeypatch.setattr(writer, "load_kv_into", fail)
    assert not cache.manage("preload", ["a" * 64])["all_ready"]
    assert not cache.entries
    conserved(cache)
    monkeypatch.setattr(writer, "load_kv_into", original)
    assert cache.manage("preload", ["a" * 64])["all_ready"]
    conserved(cache)


@pytest.mark.parametrize("missing_rank", [False, True])
def test_sp_preflight_uses_identical_collective_order(
    cpu_dma, monkeypatch, missing_rank
):
    # Real rendezvous between two CPU workers: divergent early returns make the
    # peer's next collective time out, instead of silently passing a vote stub.
    condition = threading.Condition()
    rounds, local = {}, threading.local()
    caches = [owner(pages=3), owner(pages=3)]
    caches[0].entries["a"] = NS(positions=torch.zeros(8, 3))
    for rank, cache in enumerate(caches):
        cache.disk_cache = NS(
            local_rank=rank,
            prefix_rows=lambda key, rank=rank: (
                None if missing_rank and key == "a" and rank == 1 else 8
            ),
        )

    def collective(value):
        step = getattr(local, "step", 0)
        local.step = step + 1
        with condition:
            votes = rounds.setdefault(step, {})
            votes[local.rank] = value
            condition.notify_all()
            assert condition.wait_for(lambda: len(votes) == 2, timeout=5), (
                "SP collective order diverged"
            )
            return [votes[rank] for rank in range(2)]

    monkeypatch.setattr(preparation_cache, "coordinate_validation", collective)

    def run(rank):
        local.rank = rank
        return caches[rank]._preload_plan(["a", "b"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run, range(2)))
    assert results[0] == results[1]
    assert results[0] == (
        [0, 8] if missing_rank else [8, 8],
        2 if missing_rank else 4,
        3,
    )
