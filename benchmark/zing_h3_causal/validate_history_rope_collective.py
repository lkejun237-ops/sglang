# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --nproc-per-node=1/2/4; compare history RoPE through FA4/Ulysses."""

import os
import json
import torch
from types import SimpleNamespace
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.multimodal_gen.runtime.models.dits.zing_h3 import _attention
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import _apply_rope_into
from sglang.multimodal_gen.runtime.models.dits.zing_h3_rope import (
    apply_history_rope_into,
    apply_history_rope_batch,
    apply_contiguous_history_rope_batch,
    HISTORY_ROPE_ROUTES,
)


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank)
    initialize_model_parallel(
        sequence_parallel_degree=world, ulysses_degree=world, ring_degree=1
    )

    class Projection:
        def __init__(self, qkv=False):
            self.qkv = qkv

        def __call__(self, x):
            return (torch.cat((x, x, x), -1) if self.qkv else x), None

    with torch.no_grad():
        torch.manual_seed(179)
        heads = 56
        dim = 128
        rows = 19
        history_rows = 63
        h = torch.randn(rows, heads * dim, device="cuda", dtype=torch.bfloat16)
        raw = torch.randn(history_rows, heads, dim, device="cuda", dtype=torch.bfloat16)
        values = torch.randn_like(raw)
        c = torch.randn(history_rows, 96, device="cuda", dtype=torch.bfloat16)
        current = torch.randn(rows, 96, device="cuda", dtype=torch.bfloat16)
        splits = [rows // world + (i < rows % world) for i in range(world)]
        a = sum(splits[:rank])
        b = a + splits[rank]
        local_heads = heads // world
        hs = rank * local_heads
        he = hs + local_heads
        model = SimpleNamespace(
            qkv_proj=Projection(True),
            out_proj=Projection(),
            q_norm=torch.nn.RMSNorm(dim, eps=1e-6, device="cuda", dtype=torch.bfloat16),
            k_norm=torch.nn.RMSNorm(dim, eps=1e-6, device="cuda", dtype=torch.bfloat16),
            num_heads=heads,
            head_dim=dim,
            softmax_scale=dim**-0.5,
        )
        original = raw.clone()
        outputs = []
        for enabled, grouped in (
            ("0", None),
            ("1", None),
            ("1", apply_history_rope_batch),
            ("1", apply_contiguous_history_rope_batch),
        ):
            os.environ["SGLANG_ZING_FUSED_HISTORY_ROPE"] = enabled

            def append(k, v):
                keys = torch.empty(
                    history_rows + rows, local_heads, dim, device="cuda", dtype=k.dtype
                )
                if grouped:
                    sizes = [7, 11, history_rows - 18]
                    grouped(
                        list(raw[:, hs:he].split(sizes)),
                        list(c.split(sizes)),
                        keys[:history_rows],
                    )
                else:
                    apply_history_rope_into(raw[:, hs:he], c, keys[:history_rows])
                _apply_rope_into(k, current, keys[history_rows:])
                return keys, torch.cat((values[:, hs:he], v))

            y, captured = _attention(
                model,
                h[a:b],
                current,
                capture=True,
                sequence_splits=splits,
                append_kv=append,
                append_unrotated=True,
                attention_backend="fa4",
            )
            outputs.append((y, captured))
        for output, captured in outputs[1:]:
            assert torch.equal(outputs[0][0], output)
            assert all(torch.equal(a, b) for a, b in zip(outputs[0][1], captured))
        assert HISTORY_ROPE_ROUTES["grouped"] == 1
        assert HISTORY_ROPE_ROUTES["contiguous"] == 1
        assert torch.equal(raw, original)
        assert HISTORY_ROPE_ROUTES["fused"] == 2
        print(
            "COLLECTIVE_PASS "
            + json.dumps(
                dict(rank=rank, world=world, routes=dict(HISTORY_ROPE_ROUTES))
            ),
            flush=True,
        )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
