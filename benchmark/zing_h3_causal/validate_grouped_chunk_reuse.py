# SPDX-License-Identifier: Apache-2.0
"""One-GPU BF16 component oracle and warm cache-rebuild timings for #211.

No checkpoint, video, SP collectives or quality acceptance. Fails without CUDA.
"""

import json
import os
import statistics
import time

import torch

import sglang.multimodal_gen.runtime.models.dits.zing_h3  # noqa: F401
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import _apply_rope_into
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.models.dits.zing_h3_rope import (
    HISTORY_ROPE_ROUTES,
    apply_history_rope_batch,
    apply_history_rope_into,
)


@torch.no_grad()
def main():
    assert torch.cuda.is_available(), "A real CUDA device is required"
    torch.manual_seed(211)
    os.environ["SGLANG_ZING_CHUNK_ROPE_REUSE"] = "1"
    os.environ["SGLANG_ZING_FUSED_HISTORY_ROPE"] = "1"
    records = []
    for sizes in ([17, 33, 65, 129], [390] * 4):
        c = ZingH3KVCache(1, rope_position_mode="block_relative")
        for i, rows in enumerate(sizes):
            p = torch.zeros(rows, 3, device="cuda")
            p[:, 0] = i * 10
            p[:, 1] = torch.arange(rows, device="cuda")
            k = torch.randn(rows, 14, 128, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            block = torch.full((rows,), i * 2, device="cuda")
            if i:
                c.update(p, block, "active", 1)
            c.update(p, block, "final" if i else "prefix", 1 if i else 0, [(k, v)])
        current = torch.zeros(2, 3, device="cuda")
        current[:, 0] = 50
        current, history = c.attention_positions(current)
        c.bind_attention_positions(history)
        k = torch.randn(2, 14, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)

        def coeff(p):
            a = p.sum(-1, keepdim=True) * torch.arange(1, 49, device="cuda") / 97
            return torch.cat((a.cos(), a.sin()), -1).bfloat16()

        def writer(x, value, out, out_v):
            _apply_rope_into(x, coeff(current), out)
            out_v.copy_(value)

        def run(grouped):
            return c.raw_attention_view(
                0,
                k,
                v,
                coeff,
                apply_history_rope_into,
                writer,
                rotate_batch=apply_history_rope_batch if grouped else None,
            )

        hist_k, hist_v = c.history(0)
        expected_k = torch.cat(
            (
                _apply_rope_into(hist_k, coeff(history), torch.empty_like(hist_k)),
                _apply_rope_into(k, coeff(current), torch.empty_like(k)),
            )
        )
        expected_v = torch.cat((hist_v, v))
        before = HISTORY_ROPE_ROUTES.copy()
        actual_k, actual_v = run(True)
        assert torch.equal(actual_k, expected_k)
        assert torch.equal(actual_v, expected_v)
        assert HISTORY_ROPE_ROUTES["grouped"] == before["grouped"] + 1
        assert HISTORY_ROPE_ROUTES["grouped_chunks"] == before["grouped_chunks"] + 4
        before = HISTORY_ROPE_ROUTES.copy()
        run(True)
        assert HISTORY_ROPE_ROUTES == before  # all hits
        # Keep one valid middle part, invalidate both sides by coordinates.
        changed = history.clone()
        changed[: sizes[0], 0] += 1
        changed[sum(sizes[:2]) :, 0] += 1
        middle = actual_k[sizes[0] : sum(sizes[:2])].clone()
        c.bind_attention_positions(changed)
        before = HISTORY_ROPE_ROUTES.copy()
        actual_k, actual_v = run(True)
        assert HISTORY_ROPE_ROUTES["grouped"] == before["grouped"] + 2
        assert HISTORY_ROPE_ROUTES["grouped_chunks"] == before["grouped_chunks"] + 3
        assert torch.equal(actual_k[sizes[0] : sum(sizes[:2])], middle)
        expected_k[:-2] = _apply_rope_into(
            hist_k, coeff(changed), torch.empty_like(hist_k)
        )
        assert torch.equal(actual_k, expected_k)
        assert torch.equal(actual_v, expected_v)
        for grouped in (False, True):
            for _ in range(10):
                c._rotated_store_signatures.clear()
                run(grouped)
            torch.cuda.synchronize()
            wall, gpu = [], []
            before = HISTORY_ROPE_ROUTES.copy()
            for _ in range(20):
                c._rotated_store_signatures.clear()
                a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start = time.perf_counter()
                a.record()
                actual_k, actual_v = run(grouped)
                b.record()
                b.synchronize()
                wall.append((time.perf_counter() - start) * 1000)
                gpu.append(a.elapsed_time(b))
            assert torch.equal(actual_k, expected_k)
            assert torch.equal(actual_v, expected_v)
            counts = {
                key: HISTORY_ROPE_ROUTES[key] - before[key]
                for key in ("grouped", "grouped_chunks", "fused")
            }
            assert counts["grouped" if grouped else "fused"] == (20 if grouped else 80)
            records.append(
                dict(
                    sizes=sizes,
                    heads=14,
                    dim=128,
                    grouped=grouped,
                    repeats=20,
                    warmup=10,
                    routes=counts,
                    wall_median_ms=statistics.median(wall),
                    cuda_event_median_ms=statistics.median(gpu),
                )
            )
        c.clear()
        assert not list(c.owned_tensors())
    print(
        json.dumps(
            dict(
                device=torch.cuda.get_device_name(),
                capability=torch.cuda.get_device_capability(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                records=records,
                result="PASS",
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
