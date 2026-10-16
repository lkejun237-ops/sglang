# SPDX-License-Identifier: Apache-2.0
"""History RoPE correctness, including cache slices and rejected layouts."""

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits import zing_h3_rope
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    _apply_rope,
    _apply_rope_into,
)
from sglang.multimodal_gen.runtime.models.dits.zing_h3_rope import (
    CURRENT_ROPE_ROUTES,
    HISTORY_ROPE_ROUTES,
    _history_rope_rejection,
    apply_current_rope,
    apply_history_rope_batch,
    apply_history_rope_into,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("heads", [6, 12, 24, 14, 28, 56])
@pytest.mark.parametrize("rows", [1, 17, 390, 832])
@pytest.mark.parametrize("strided", [False, True])
@torch.no_grad()
def test_history_matches_eager_with_offset_and_sentinels(
    device, heads, rows, strided, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setenv("SGLANG_ZING_FUSED_HISTORY_ROPE", "1")
    torch.manual_seed(179)
    storage = torch.randn(
        rows + 4,
        heads * (3 if strided else 1),
        128,
        device=device,
        dtype=torch.bfloat16,
    )
    x = storage[2 : rows + 2, :heads]
    original = storage.clone()
    coefficients = torch.randn(rows + 4, 96, device=device, dtype=torch.bfloat16)[
        2 : rows + 2
    ]
    expected = torch.empty_like(x)
    _apply_rope_into(x, coefficients, expected)
    dst = torch.full((rows + 6, heads, 128), 37, device=device, dtype=x.dtype)
    before = HISTORY_ROPE_ROUTES.copy()
    apply_history_rope_into(x, coefficients, dst[3 : rows + 3])
    assert torch.equal(dst[3 : rows + 3], expected)
    assert torch.equal(storage, original)
    assert torch.all(dst[:3] == 37) and torch.all(dst[rows + 3 :] == 37)
    assert torch.equal(dst[3 : rows + 3, :, 96:], x[:, :, 96:])
    route = "fused" if device == "cuda" else "device"
    assert HISTORY_ROPE_ROUTES[route] == before[route] + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_rejected_alias_dtype_and_layout(monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_FUSED_HISTORY_ROPE", "1")
    x = torch.randn(17, 6, 128, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(17, 96, device="cuda", dtype=x.dtype)
    assert _history_rope_rejection(x, c, x) == "alias"
    expected = torch.empty_like(x)
    _apply_rope_into(x.clone(), c, expected)
    apply_history_rope_into(x, c, x)
    assert torch.equal(x, expected)
    for dtype in (torch.float32, torch.float16):
        y, coeff = x.to(dtype), c.to(dtype)
        out = torch.empty_like(y)
        assert _history_rope_rejection(y, coeff, out) == "dtype"
        apply_history_rope_into(y, coeff, out)
        assert torch.equal(out, _apply_rope_into(y, coeff, torch.empty_like(y)))
    y = x.transpose(0, 1).contiguous().transpose(0, 1)
    assert _history_rope_rejection(y, c, torch.empty_like(x)) == "layout"
    expected = _apply_rope_into(y, c, torch.empty_like(x))
    assert torch.equal(apply_history_rope_into(y, c, torch.empty_like(x)), expected)


@pytest.mark.parametrize("timing", ["before_source", "after_step0"])
@pytest.mark.parametrize("planned", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "window,sink,mode",
    [
        (3, 1, "block_relative"),
        (21, 3, "block_relative"),
        (None, 0, "block_relative"),
        (None, 0, "absolute"),
    ],
)
@torch.no_grad()
def test_cache_transactions_eviction_and_failure(
    device, window, sink, mode, grouped, planned, timing, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
    from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy

    monkeypatch.setenv("SGLANG_ZING_FUSED_HISTORY_ROPE", "1")
    caches = [
        ZingH3KVCache(
            1,
            window_policy=ZingWindowPolicy(window, sink),
            rope_position_mode=mode,
            window_eviction_timing=timing,
        )
        for _ in range(2)
    ]
    torch.manual_seed(179)

    def coefficients(p):
        phases = (
            p.sum(-1, keepdim=True).float()
            * torch.arange(1, 49, device=device)[None]
            / 97
        )
        return torch.cat((phases.cos(), phases.sin()), -1).bfloat16()

    def positions(n, start):
        p = torch.arange(n * 3, device=device, dtype=torch.float64).reshape(n, 3) / 7
        p[:, 0] += start
        return p

    def execute(p, block, phase):
        k = torch.randn(len(p), 14, 128, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        results = []
        for cache, rotation in zip(caches, (_apply_rope_into, apply_history_rope_into)):
            current, history = cache.attention_positions(p, prefix=phase == "prefix")
            cache.bind_attention_positions(history)

            def writer(x, value, out, out_v):
                _apply_rope_into(x, coefficients(current), out)
                out_v.copy_(value)

            if cache.raw_store:
                result = cache.raw_attention_view(
                    0,
                    k,
                    v,
                    coefficients,
                    rotation,
                    writer,
                    history_plan=(
                        cache.prepare_history_rope_plan() if planned else None
                    ),
                    rotate_batch=(
                        apply_history_rope_batch
                        if grouped and cache is caches[1]
                        else None
                    ),
                )
                committed_k = k
            else:
                result = cache.attention_view(0, k, v, writer)
                committed_k = _apply_rope_into(
                    k, coefficients(current), torch.empty_like(k)
                )
            results.append(tuple(t.clone() for t in result))
            cache.update(
                p,
                torch.full((len(p),), block, device=device),
                phase,
                0 if phase == "prefix" else 1,
                None if phase == "active" else ((committed_k, v),),
            )
        for a, b in zip(*results):
            assert torch.equal(a, b)
        for left, right in zip(caches[0]._history(), caches[1]._history()):
            assert torch.equal(left.positions, right.positions)
            assert all(
                torch.equal(a, b) for a, b in zip(left.layers[0], right.layers[0])
            )

    execute(positions(5, 0), 0, "prefix")
    for chunk in range(28):
        if chunk == 13:
            from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
                ZingKVChunk,
            )

            prefix_positions = positions(6, 0)
            prefix_k = torch.randn(6, 14, 128, device=device, dtype=torch.bfloat16)
            prefix_v = torch.randn_like(prefix_k)
            if not caches[0].raw_store:
                prefix_k = _apply_rope_into(
                    prefix_k, coefficients(prefix_positions), torch.empty_like(prefix_k)
                )
            for cache in caches:
                cache.replace_prefix(
                    ZingKVChunk(prefix_positions.clone(), 0, ((prefix_k, prefix_v),))
                )
        target = 2 * (chunk + 1)
        for cache in caches:
            cache.begin_chunk(target)
        execute(positions(7, chunk * 2 + 2), target - 1, "source")
        execute(positions(9, chunk * 2 + 2), target, "active")
        for cache in caches:
            cache.complete_step0_and_evict(target)
        execute(positions(9, chunk * 2 + 2), target, "final")
    cache = caches[1]
    if cache.raw_store:
        p = positions(3, 100)
        _, hist = cache.attention_positions(p)
        cache.bind_attention_positions(hist)
        cache._rotated_store_signatures.clear()
        k = torch.zeros(3, 14, 128, device=device, dtype=torch.bfloat16)

        def fail(*args):
            raise RuntimeError("injected rotation failure")

        with pytest.raises(RuntimeError, match="injected rotation failure"):
            cache.raw_attention_view(
                0,
                k,
                k,
                coefficients,
                fail,
                fail,
                rotate_batch=fail if grouped else None,
            )
        assert 0 not in cache._rotated_store_signatures
    for cache in caches:
        cache.clear()
        assert cache.history(0) is None and not cache._rotated_stores


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_passthrough_preserves_bfloat16_payload_bits(monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_FUSED_HISTORY_ROPE", "1")
    x = torch.zeros(17, 2, 128, device="cuda", dtype=torch.bfloat16)
    # Signed zero, subnormals, infinities and NaN payloads in non-rotary columns.
    bits = torch.tensor(
        [0, -32768, 1, -32767, 32640, -128, 32641, 32767],
        device="cuda",
        dtype=torch.int16,
    )
    x.view(torch.int16)[..., 96:] = bits.repeat(4)
    c = torch.ones(17, 96, device="cuda", dtype=x.dtype)
    out = apply_history_rope_into(x, c, torch.empty_like(x))
    assert torch.equal(out.view(torch.int16)[..., 96:], x.view(torch.int16)[..., 96:])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("heads", [14, 28, 56])
@pytest.mark.parametrize("sizes", [(1,), (17, 3, 29), (1,) * 22, (2,) * 33])
@torch.no_grad()
def test_grouped_dynamic_chunks_and_nonzero_offsets(device, heads, sizes, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setenv("SGLANG_ZING_FUSED_HISTORY_ROPE", "1")
    torch.manual_seed(182)
    stores = [
        torch.randn(n + 3, heads * 2, 128, dtype=torch.bfloat16, device=device)
        for n in sizes
    ]
    xs = [x[1 : n + 1, :heads] for x, n in zip(stores, sizes)]
    tables = [
        torch.randn(n + 2, 96, dtype=torch.bfloat16, device=device)[1 : n + 1]
        for n in sizes
    ]
    before = [x.clone() for x in stores]
    expected = torch.cat(
        [_apply_rope_into(x, c, torch.empty_like(x)) for x, c in zip(xs, tables)]
    )
    storage = torch.full(
        (sum(sizes) + 4, heads, 128), 37, device=device, dtype=torch.bfloat16
    )
    route = HISTORY_ROPE_ROUTES["grouped"]
    apply_history_rope_batch(xs, tables, storage[2:-2])
    assert torch.equal(storage[2:-2], expected)
    assert torch.all(storage[:2] == 37) and torch.all(storage[-2:] == 37)
    assert all(torch.equal(x, old) for x, old in zip(stores, before))
    assert HISTORY_ROPE_ROUTES["grouped"] == route + (device == "cuda")


@torch.no_grad()
def test_grouped_rejects_row_mismatch_without_writing(monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_FUSED_HISTORY_ROPE", "1")
    x = torch.zeros(3, 2, 128, dtype=torch.bfloat16)
    out = torch.full_like(x, 37)
    with pytest.raises(ValueError, match="rows or coefficient count"):
        apply_history_rope_batch([x], [], out)
    assert torch.all(out == 37)
    apply_history_rope_batch([], [], out[:0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rotary", [2, 32, 96, 128])
@torch.no_grad()
def test_grouped_partial_rotary_and_payload_bits(rotary, monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_FUSED_HISTORY_ROPE", "1")
    stores = [
        torch.randn(n + 4, 28, 128, device="cuda", dtype=torch.bfloat16)
        for n in (3, 17)
    ]
    xs = [t[2:-2, 7:21] for t in stores]
    patterns = torch.tensor([0, -32768, 32704, -64], device="cuda", dtype=torch.int16)
    for x in xs:
        if rotary < 128:
            tail = x.view(torch.int16)[..., rotary:]
            tail.copy_(patterns[torch.arange(tail.shape[-1], device="cuda") % 4])
    tables = [torch.randn(len(x), rotary, device="cuda", dtype=x.dtype) for x in xs]
    expected = torch.cat(
        [_apply_rope_into(x, c, torch.empty_like(x)) for x, c in zip(xs, tables)]
    )
    actual = torch.empty_like(expected)
    apply_history_rope_batch(xs, tables, actual)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.no_grad()
def test_grouped_alias_and_dtype_fallback(device, dtype, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setenv("SGLANG_ZING_FUSED_HISTORY_ROPE", "1")
    value = torch.randn(20, 14, 128, device=device, dtype=dtype)
    xs = list(value.split([3, 17]))
    tables = [torch.randn(len(x), 96, device=device, dtype=dtype) for x in xs]
    expected = torch.cat(
        [_apply_rope_into(x, c, torch.empty_like(x)) for x, c in zip(xs, tables)]
    )
    apply_history_rope_batch(xs, tables, value)
    assert torch.equal(value, expected)


@pytest.mark.parametrize("chunk_reuse", [False, True])
@torch.no_grad()
def test_history_plan_is_session_owned_and_rejects_stale_state(
    chunk_reuse, monkeypatch
):
    monkeypatch.setenv("SGLANG_ZING_CHUNK_ROPE_REUSE", "1" if chunk_reuse else "0")
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache

    first, second = [
        ZingH3KVCache(1, rope_position_mode="block_relative") for _ in range(2)
    ]
    plan = first.prepare_history_rope_plan()
    k = torch.zeros(3, 2, 128, dtype=torch.bfloat16)

    def must_not_run(*args):
        raise AssertionError("Invalid plan reached the writer")

    with pytest.raises(ValueError, match="another cache or forward state"):
        second.raw_attention_view(
            0, k, k, must_not_run, must_not_run, must_not_run, history_plan=plan
        )
    first._invalidate_rope()
    with pytest.raises(ValueError, match="another cache or forward state"):
        first.raw_attention_view(
            0, k, k, must_not_run, must_not_run, must_not_run, history_plan=plan
        )


@pytest.mark.parametrize("chunk_reuse", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_history_plan_shared_across_layers_with_store_growth(
    device, chunk_reuse, monkeypatch
):
    monkeypatch.setenv("SGLANG_ZING_CHUNK_ROPE_REUSE", "1" if chunk_reuse else "0")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import (
        ZingH3KVCache,
        ZingKVChunk,
    )

    monkeypatch.setenv("SGLANG_ZING_FUSED_HISTORY_ROPE", "1")
    torch.manual_seed(182)
    caches = [ZingH3KVCache(2, rope_position_mode="block_relative") for _ in range(2)]
    positions = torch.zeros(5, 3, device=device, dtype=torch.float64)
    pairs = tuple(
        (
            torch.randn(5, 14, 128, dtype=torch.bfloat16, device=device),
            torch.randn(5, 14, 128, dtype=torch.bfloat16, device=device),
        )
        for _ in range(2)
    )
    for cache in caches:
        cache.replace_prefix(ZingKVChunk(positions.clone(), 0, pairs))
    current_positions = torch.ones(7, 3, device=device, dtype=torch.float64)
    for cache in caches:
        _, history = cache.attention_positions(current_positions)
        cache.bind_attention_positions(history)
    plan = caches[1].prepare_history_rope_plan()

    def coefficients(p):
        phase = (
            p.sum(-1, keepdim=True).float()
            * torch.arange(1, 49, device=device)[None]
            / 97
        )
        return torch.cat((phase.cos(), phase.sin()), -1).bfloat16()

    def writer(k, v, out_k, out_v):
        _apply_rope_into(k, coefficients(current_positions), out_k)
        out_v.copy_(v)

    for layer in range(2):
        k = torch.randn(7, 14, 128, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        reference = caches[0].raw_attention_view(
            layer, k, v, coefficients, _apply_rope_into, writer
        )
        actual = caches[1].raw_attention_view(
            layer,
            k,
            v,
            coefficients,
            apply_history_rope_into,
            writer,
            rotate_batch=apply_history_rope_batch,
            history_plan=plan,
        )
        assert all(torch.equal(a, b) for a, b in zip(actual, reference))
    caches[1].prefix.positions.add_(1)
    with pytest.raises(ValueError, match="positions changed"):
        caches[1].raw_attention_view(
            0, k, v, coefficients, apply_history_rope_into, writer, history_plan=plan
        )
    for cache in caches:
        cache.clear()
        assert not list(cache.owned_tensors())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("adjacent", [False, True])
@torch.no_grad()
def test_contiguous_batch_views_raw_k_or_falls_back(device, adjacent, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_rope import (
        apply_contiguous_history_rope_batch,
    )

    monkeypatch.setenv("SGLANG_ZING_FUSED_HISTORY_ROPE", "1")
    store = torch.randn(19, 28, 128, device=device, dtype=torch.bfloat16)
    xs = list(store[2:-2, 7:21].split([3, 7, 5]))
    if not adjacent:
        xs.reverse()
    before = store.clone()
    tables = [torch.randn(len(x), 96, device=device, dtype=x.dtype) for x in xs]
    expected = torch.cat(
        [_apply_rope_into(x, c, torch.empty_like(x)) for x, c in zip(xs, tables)]
    )
    out = torch.empty_like(expected)
    route = HISTORY_ROPE_ROUTES["contiguous"]
    apply_contiguous_history_rope_batch(xs, tables, out)
    assert torch.equal(out, expected) and torch.equal(store, before)
    assert HISTORY_ROPE_ROUTES["contiguous"] == route + (device == "cuda" and adjacent)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [14, 28, 56])
@pytest.mark.parametrize("offset", [1, 8])
@torch.no_grad()
def test_grouped_unaligned_pointers_and_row_stride(heads, offset, monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_FUSED_HISTORY_ROPE", "1")
    xs, tables, stores = [], [], []
    for rows in (1, 17, 33):
        stride = heads * 128 + 3
        store = torch.randn(rows * stride + offset, device="cuda", dtype=torch.bfloat16)
        xs.append(store.as_strided((rows, heads, 128), (stride, 128, 1), offset))
        stores.append(store)
        table = torch.randn(rows * 96 + offset, device="cuda", dtype=torch.bfloat16)
        tables.append(table[offset:].view(rows, 96))
    before = [store.clone() for store in stores]
    expected = torch.cat(
        [_apply_rope_into(x, c, torch.empty_like(x)) for x, c in zip(xs, tables)]
    )
    output = torch.full(
        (expected.shape[0] + 2, heads, 128), -11, device="cuda", dtype=torch.bfloat16
    )
    route = HISTORY_ROPE_ROUTES["grouped"]
    apply_history_rope_batch(xs, tables, output[1:-1])
    assert HISTORY_ROPE_ROUTES["grouped"] == route + 1
    assert torch.equal(output[1:-1], expected)
    assert (output[0] == -11).all() and (output[-1] == -11).all()
    assert all(torch.equal(a, b) for a, b in zip(stores, before))


def _qkv_view(device, rows, heads):
    """Q as the fused projection hands it over: a row-strided QKV view."""
    qkv = torch.randn(rows, 3 * heads * 128, device=device, dtype=torch.bfloat16)
    return qkv, qkv[:, : heads * 128].view(rows, heads, 128)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("heads", [2, 28, 56])
@pytest.mark.parametrize("rows", [1, 390, 780])
@torch.no_grad()
def test_current_rope_matches_eager_without_touching_qkv(
    device, heads, rows, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setenv("SGLANG_ZING_FUSED_CURRENT_ROPE", "1")
    apply, routes = apply_current_rope, CURRENT_ROPE_ROUTES
    torch.manual_seed(211)
    qkv, q = _qkv_view(device, rows, heads)
    original = qkv.clone()
    rope = torch.randn(rows, 96, device=device, dtype=torch.bfloat16)
    expected = _apply_rope(q, rope)
    before = routes.copy()
    out = apply(q, rope)
    # The fused kernel keeps eager's BF16 product roundings, so it is exact.
    assert torch.equal(out, expected)
    assert out.is_contiguous() and out.shape == q.shape
    assert torch.equal(qkv, original)
    route = "fused" if device == "cuda" else "device"
    assert routes[route] == before[route] + 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_disabled_current_rope_is_eager(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setenv("SGLANG_ZING_FUSED_CURRENT_ROPE", "0")
    apply, routes = apply_current_rope, CURRENT_ROPE_ROUTES
    _, q = _qkv_view(device, 17, 2)
    rope = torch.randn(17, 96, device=device, dtype=torch.bfloat16)
    before = routes["disabled"]
    assert torch.equal(apply(q, rope), _apply_rope(q, rope))
    assert routes["disabled"] == before + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_unsupported_current_rope_layouts_fall_back(monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_FUSED_CURRENT_ROPE", "1")
    apply, routes = apply_current_rope, CURRENT_ROPE_ROUTES
    rope = torch.randn(17, 96, device="cuda", dtype=torch.bfloat16)
    # Ulysses hands back [rows, heads, 3 * dim] chunks: heads are not packed.
    qkv = torch.randn(17, 2, 3 * 128, device="cuda", dtype=torch.bfloat16)
    q = qkv[..., :128]
    before = routes["layout"]
    assert torch.equal(apply(q, rope), _apply_rope(q, rope))
    assert routes["layout"] == before + 1
    for dtype in (torch.float32, torch.float16):
        x, c = q.to(dtype), rope.to(dtype)
        assert torch.equal(apply(x, c), _apply_rope(x, c))


@torch.no_grad()
def test_guard_never_checks_the_input_as_the_output(monkeypatch):
    """The QKV view is never contiguous; checking it as the output rejects all."""
    monkeypatch.setenv("SGLANG_ZING_FUSED_CURRENT_ROPE", "1")
    seen = []
    guard = zing_h3_rope._history_rope_rejection

    def spy(x, coefficients, out, **kwargs):
        seen.append((x, out))
        return guard(x, coefficients, out, **kwargs)

    monkeypatch.setattr(zing_h3_rope, "_history_rope_rejection", spy)
    _, q = _qkv_view("cpu", 5, 2)
    apply_current_rope(q, torch.randn(5, 96, dtype=torch.bfloat16))
    assert len(seen) == 1
    assert seen[0][0] is q and seen[0][1] is not q


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((5, 2, 128), torch.bfloat16),
        ((1, 5, 2, 128), torch.bfloat16),
        ((5, 2, 128), torch.float32),
    ],
)
@torch.no_grad()
def test_rejected_current_rope_allocates_no_candidate_output(shape, dtype, monkeypatch):
    """A rejected input (CPU, SP 4-D, non-BF16) must not reserve a fused output."""
    monkeypatch.setenv("SGLANG_ZING_FUSED_CURRENT_ROPE", "1")
    x = torch.randn(shape, dtype=dtype)
    rope = torch.randn(5, 96, dtype=dtype)
    expected = _apply_rope(x, rope)
    allocations = []
    empty = torch.empty

    def counting_empty(*args, **kwargs):
        allocations.append(args)
        return empty(*args, **kwargs)

    monkeypatch.setattr(zing_h3_rope, "_apply_rope", lambda x, rope: expected)
    monkeypatch.setattr(torch, "empty", counting_empty)
    result = apply_current_rope(x, rope)
    monkeypatch.undo()
    assert result is expected
    assert allocations == []
