# SPDX-License-Identifier: Apache-2.0
"""Same-SP exact cache/attention oracle with actual H56/D128 CUDA collectives.

Run with torchrun --nproc-per-node={1,2,4}; BASELINE_CACHE names the frozen
main zing_h3_cache.py. No model weights or synthetic CPU emulation are used here.
Real-checkpoint validation is a separate required lane.
"""

import functools
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


class Projection:
    def __init__(self, qkv=False):
        self.qkv = qkv

    def __call__(self, x):
        return (torch.cat((x, x, x), -1) if self.qkv else x), None


@torch.inference_mode()
def main():
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        _apply_rope_into,
        _apply_rope_kv_into,
    )
    from sglang.multimodal_gen.runtime.models.dits.zing_h3 import _attention
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
    from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy

    spec = importlib.util.spec_from_file_location(
        "frozen_zing_cache", os.environ["BASELINE_CACHE"]
    )
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    init_distributed_environment(world_size=world, rank=rank, local_rank=local)
    initialize_model_parallel(
        sequence_parallel_degree=world, ulysses_degree=world, ring_degree=1
    )
    device = torch.device("cuda", local)
    module = SimpleNamespace(
        qkv_proj=Projection(True),
        out_proj=Projection(),
        q_norm=torch.nn.RMSNorm(
            128, eps=1e-6, device=device, dtype=torch.bfloat16
        ).eval(),
        k_norm=torch.nn.RMSNorm(
            128, eps=1e-6, device=device, dtype=torch.bfloat16
        ).eval(),
        num_heads=56,
        head_dim=128,
        softmax_scale=128**-0.5,
    )
    for norm in (module.q_norm, module.k_norm):
        norm.weight.fill_(1)
        norm.weight.requires_grad_(False)
    records = []

    def coeff(p):
        frequencies = torch.arange(1, 49, device=device).float() * 0.001
        angle = p.float().sum(-1, keepdim=True) * frequencies
        return torch.cat((angle.cos(), angle.sin()), -1).bfloat16()

    for window, sink, mode in [
        (21, 3, "block_relative"),
        (3, 0, "block_relative"),
        (4, 1, "absolute"),
        (None, 0, "absolute"),
    ]:
        caches = []
        for cls, flag in [(baseline.ZingH3KVCache, "0"), (ZingH3KVCache, "1")]:
            os.environ["SGLANG_ZING_CHUNK_ROPE_REUSE"] = flag
            caches.append(
                cls(
                    1,
                    window_policy=ZingWindowPolicy(window, sink),
                    rope_position_mode=mode,
                )
            )
        calls = 0

        def forward(block, phase, step=0, tail=False, override=False):
            nonlocal calls
            rows = 11 if tail else 19
            p = torch.zeros(rows, 3, device=device)
            p[:, 0] = 0 if phase == "prefix" else ((block + 1) // 2) * 20
            p[:, 1] = torch.arange(rows, device=device) % 5
            splits = [(rows + world - i - 1) // world for i in range(world)]
            start, stop = sum(splits[:rank]), sum(splits[: rank + 1])
            generator = torch.Generator().manual_seed(180 + block * 100 + step)
            hidden = torch.randn(rows, 56 * 128, generator=generator).to(
                device, torch.bfloat16
            )
            outputs = []
            for c in caches:
                current, history = c.attention_positions(p, prefix=phase == "prefix")
                if override and history is not None:
                    history = history.clone()
                    history[1, 1] += 1
                c.bind_attention_positions(history)
                rope = coeff(current)

                def writer(k, v, out_k, out_v):
                    return _apply_rope_kv_into(k, v, rope, out_k, out_v)

                def append(k, v):
                    if c.raw_store:
                        result = c.raw_attention_view(
                            0, k, v, coeff, _apply_rope_into, writer
                        )
                        authoritative = c.history(0)
                        if authoritative is not None:
                            expected = torch.empty_like(authoritative[0])
                            _apply_rope_into(authoritative[0], coeff(history), expected)
                            assert torch.equal(result[0][:-rows], expected)
                    else:
                        result = c.attention_view(0, k, v, writer)
                    return result

                output, raw = _attention(
                    module,
                    hidden[start:stop],
                    rope,
                    capture=phase != "active",
                    sequence_splits=splits if world > 1 else None,
                    attention_backend="fa4",
                    append_kv=append,
                    capture_rotated=c.single_store,
                    append_unrotated=c.raw_store,
                    capture_stored_values=c.raw_store,
                    stage_raw_k=(
                        functools.partial(c.stage_raw_k, 0) if c.raw_store else None
                    ),
                )
                outputs.append(output.clone())
                c.update(
                    p,
                    torch.full((rows,), block, device=device),
                    phase,
                    0 if phase == "prefix" else 1,
                    None if phase == "active" else [raw],
                )
            assert torch.equal(*outputs), (rank, window, block, phase, step)
            a, b = [c.history(0) for c in caches]
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            assert torch.equal(caches[0].positions(), caches[1].positions())
            calls += 1

        forward(0, "prefix")
        chunks = 85 if window == 21 else 12
        evictions = [0, 0]
        for i in range(1, chunks + 1):
            for index, c in enumerate(caches):
                before = len(c.committed)
                c.prepare_chunk(2 * i)
                evictions[index] += before - len(c.committed)
            # A short final source/target tests uneven rows on all SP degrees.
            tail = i == chunks
            forward(2 * i - 1, "source", tail=tail)
            for step in range(4):
                forward(
                    2 * i,
                    "active",
                    step,
                    tail=tail,
                    override=mode != "absolute" and i == 5 and step == 2,
                )
            forward(2 * i, "final", 4, tail=tail)
        if window == 21:
            assert min(evictions) >= 63
        for c in caches:
            c.bind_model(("weights", 0))
            try:
                c.bind_model(("weights", 1))
            except ValueError:
                pass
            else:
                raise AssertionError("Live model replacement was accepted")
            c.clear()
            assert not list(c.owned_tensors())
        records.append(
            dict(
                window=window,
                sink=sink,
                mode=mode,
                chunks=chunks,
                calls=calls,
                bitwise=True,
            )
        )
    records.extend(
        validate_boundaries(
            baseline.ZingH3KVCache,
            ZingH3KVCache,
            ZingWindowPolicy,
            device,
            rank,
            world,
            coeff,
            _apply_rope_into,
            _apply_rope_kv_into,
        )
    )
    torch.distributed.barrier()
    out = Path(os.environ["OUT"])
    (out / f"cuda-collective-sp{world}-rank{rank}.json").write_text(
        json.dumps(records, indent=2)
    )
    print("CUDA_COLLECTIVE_EXACT", world, rank, records, flush=True)


@torch.inference_mode()
def validate_boundaries(
    baseline_cls,
    candidate_cls,
    policy_cls,
    device,
    rank,
    world,
    coeff,
    rotate_into,
    write_kv,
):
    """Rank-local cache boundaries after the real-collective matrix above.

    Uses the actual 56/world head shard and original CUDA RoPE/writer. These
    checks do not claim their direct cache operations execute collectives.
    """
    heads = 56 // world
    evidence = []

    def positions(time, rows=19):
        p = torch.zeros(rows, 3, device=device)
        p[:, 0] = time
        p[:, 1] = torch.arange(rows, device=device) % 5
        p[:, 2] = torch.arange(rows, device=device) % 3
        return p

    def kv(rows, seed):
        g = torch.Generator().manual_seed(seed)
        # Take each rank's actual shard from the same explicit global tensor.
        shape = (rows, 56, 128)
        k = torch.randn(shape, generator=g)[:, rank * heads : (rank + 1) * heads]
        v = torch.randn(shape, generator=g)[:, rank * heads : (rank + 1) * heads]
        return (
            k.to(device, torch.bfloat16).contiguous(),
            v.to(device, torch.bfloat16).contiguous(),
        )

    def new(cls, flag, mode):
        os.environ["SGLANG_ZING_CHUNK_ROPE_REUSE"] = flag
        return cls(1, window_policy=policy_cls(4, 1), rope_position_mode=mode)

    def commit(c, p, block, phase, pair):
        b = torch.full((len(p),), block, device=device)
        if phase == "final":
            c.update(p, b, "active", 1)
        c.update(p, b, phase, 0 if phase == "prefix" else 1, [pair])

    def attention(c, p, pair, *, failure=None):
        if failure == "rotation" and c.rope_position_mode == "absolute":
            # Absolute mode uses the chunk's versioned positions directly.
            c._history()[0].positions[0, 1] += 7
        current, history = c.attention_positions(p)
        if failure == "rotation" and c.rope_position_mode != "absolute":
            history = history.clone()
            history[0, 1] += 7
        c.bind_attention_positions(history)
        rebuilt = 0

        def rotate(k, r, out):
            nonlocal rebuilt
            rebuilt += len(k)
            rotate_into(k, r, out)
            if failure == "rotation":
                out.fill_(float("nan"))
                raise RuntimeError("Injected history rotation failure")

        def writer(k, v, out_k, out_v):
            write_kv(k, v, coeff(current), out_k, out_v)
            if failure == "writer":
                out_k.fill_(float("nan"))
                out_v.fill_(float("nan"))
                raise RuntimeError("Injected current writer failure")

        k, v = c.raw_attention_view(0, *pair, coeff, rotate, writer)
        raw = c.history(0)
        n = 0 if raw is None else len(raw[0])
        expected = torch.empty_like(k)
        if n:
            rotate_into(raw[0], coeff(history), expected[:n])
        rotate_into(pair[0], coeff(current), expected[n:])
        expected_v = pair[1] if not n else torch.cat((raw[1], pair[1]))
        assert k.shape == expected.shape and k.dtype == expected.dtype
        assert torch.isfinite(k).all() and torch.isfinite(v).all()
        assert torch.equal(k, expected), ("boundary rotation", rank)
        assert torch.equal(v, expected_v), ("boundary V", rank)
        return k.clone(), v.clone(), rebuilt

    for mode in ("block_relative", "absolute"):
        caches = [new(baseline_cls, "0", mode), new(candidate_cls, "1", mode)]
        for c in caches:
            commit(c, positions(0), 0, "prefix", kv(19, 101))
            commit(c, positions(20), 1, "source", kv(19, 102))
            commit(c, positions(20), 2, "final", kv(19, 103))
            # Existing growth API, no alternate allocation/copy strategy.
            c._reserve_store(0, 256, *kv(11, 104))
        current, pair = positions(40, 11), kv(11, 104)

        def compare(label):
            views = [attention(c, current, pair) for c in caches]
            assert torch.equal(views[0][0], views[1][0]), label
            assert torch.equal(views[0][1], views[1][1]), label
            a, b = [c.history(0) for c in caches]
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), label
            assert torch.equal(caches[0].positions(), caches[1].positions()), label
            evidence.append(
                dict(
                    boundary=label,
                    mode=mode,
                    rank=rank,
                    head_shard=heads,
                    dtype="bfloat16",
                    bitwise=True,
                    candidate_rebuilt_rows=views[1][2],
                )
            )
            return views

        compare("initial")
        assert compare("repeat-denoise")[1][2] == 0
        for label, time, rows in [
            ("same-anchor-prefix", 0, 19),
            ("different-anchor-prefix", 100, 19),
            ("larger-prefix", 100, 23),
            ("smaller-prefix", 100, 7),
        ]:
            for cls, flag, c in zip((baseline_cls, candidate_cls), ("0", "1"), caches):
                donor = new(cls, flag, mode)
                commit(donor, positions(time, rows), 0, "prefix", kv(rows, 200 + rows))
                c.replace_prefix(donor.prefix)
                donor.clear()
                assert not list(donor.owned_tensors())
            views = compare(label)
            if label == "same-anchor-prefix":
                assert views[1][2] == rows
            assert compare(label + "-repeat")[1][2] == 0
        for c in caches:
            c.replace_prefix(None)
        compare("remove-prefix")
        for c in caches:
            old = c._stores[0][0].data_ptr()
            c._reserve_store(0, c._stores[0][0].shape[0] + 1, *pair)
            assert c._stores[0][0].data_ptr() != old
        compare("grown-storage")

        for failure in ("rotation", "writer"):
            for c in caches:
                saved = c.history(0)
                try:
                    attention(c, current, pair, failure=failure)
                except RuntimeError as exc:
                    assert str(exc).startswith("Injected ")
                else:
                    raise AssertionError("Expected injected failure did not execute")
                assert 0 not in c._rotated_store_signatures
                now = c.history(0)
                assert torch.equal(saved[0], now[0]) and torch.equal(saved[1], now[1])
            # A failed publication must not be reused even on a direct retry.
            compare(failure + "-retry")
        for c in caches:
            try:
                attention(c, current, pair, failure="writer")
            except RuntimeError:
                pass
            c.clear()
            assert not list(c.owned_tensors())
            assert not c._rotated_store_signatures
            assert c.active is None and c.pending_source is None
        evidence.append(
            dict(boundary="failed-session-clear", mode=mode, rank=rank, bitwise=True)
        )

        # The legacy inference store is intentionally not used as the mutation
        # oracle: it has no version counter. Recompute from authoritative raw K
        # with the unchanged RoPE operator instead, and disclose that boundary.
        c = new(candidate_cls, "1", mode)
        commit(c, positions(0), 0, "prefix", kv(19, 301))
        commit(c, positions(20), 2, "final", kv(19, 302))
        attention(c, current, pair)
        assert not torch.is_inference(c._stores[0][0])
        before = c._stores[0][0]._version
        c.committed[0].layers[0][0].add_(0.25)
        assert c._stores[0][0]._version > before
        assert attention(c, current, pair)[2] > 0
        c.replace_prefix(None)  # This allocation path must also stay versioned.
        attention(c, current, pair)
        assert not torch.is_inference(c._stores[0][0])
        c.committed[0].layers[0][0].add_(0.25)
        assert attention(c, current, pair)[2] > 0
        c.clear()
        assert not list(c.owned_tensors())
        evidence.append(
            dict(
                boundary="inference-raw-inplace-and-prefix-resize",
                mode=mode,
                rank=rank,
                head_shard=heads,
                oracle="original RoPE recomputation",
                bitwise=True,
            )
        )
    return evidence


if __name__ == "__main__":
    main()
