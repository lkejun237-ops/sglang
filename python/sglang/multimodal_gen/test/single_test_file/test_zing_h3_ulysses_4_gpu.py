"""Zing causal attention must preserve SP1 results under uneven Ulysses SP4."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase

_WORLD = 4


class _Projection:
    def __init__(self, *, qkv: bool = False):
        self.qkv = qkv

    def __call__(self, value):
        return (torch.cat((value, value, value), dim=-1) if self.qkv else value), None


def _worker() -> int:
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.multimodal_gen.runtime.models.dits.zing_h3 import (
        _attention,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank)
    initialize_model_parallel(
        sequence_parallel_degree=world,
        ulysses_degree=world,
        ring_degree=1,
    )

    rows, history_rows, heads, head_dim = 19, 7, 8, 16
    splits = [5, 5, 5, 4]
    start = sum(splits[:rank])
    stop = start + splits[rank]
    local_heads = heads // world
    head_start = rank * local_heads
    head_stop = head_start + local_heads
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(1234)
    hidden = torch.randn(rows, heads * head_dim, device=device, dtype=torch.bfloat16)
    history = (
        torch.randn(history_rows, heads, head_dim, device=device, dtype=torch.bfloat16),
        torch.randn(history_rows, heads, head_dim, device=device, dtype=torch.bfloat16),
    )
    rope = torch.cat(
        (
            torch.ones(rows, head_dim // 2, device=device, dtype=torch.bfloat16),
            torch.zeros(rows, head_dim // 2, device=device, dtype=torch.bfloat16),
        ),
        dim=-1,
    )
    history_rope = rope[:history_rows]
    module = SimpleNamespace(
        qkv_proj=_Projection(qkv=True),
        out_proj=_Projection(),
        q_norm=torch.nn.Identity(),
        k_norm=torch.nn.Identity(),
        num_heads=heads,
        head_dim=head_dim,
        softmax_scale=head_dim**-0.5,
    )

    reference, reference_raw = _attention(
        module,
        hidden,
        rope,
        history,
        history_rope,
        capture=True,
    )
    actual, actual_raw = _attention(
        module,
        hidden[start:stop],
        rope,
        tuple(value[:, head_start:head_stop] for value in history),
        history_rope,
        capture=True,
        sequence_splits=splits,
    )
    failures = []
    if not torch.equal(actual, reference[start:stop]):
        delta = (actual.float() - reference[start:stop].float()).abs()
        failures.append(
            f"rank {rank} output drift: mae={delta.mean():.3e}, max={delta.max():.3e}"
        )
    for name, got, expected in zip(
        ("key", "value"), actual_raw, reference_raw, strict=True
    ):
        expected = expected[:, head_start:head_stop]
        if not torch.equal(got, expected):
            failures.append(f"rank {rank} captured {name} head shard differs")
        expected_numel = rows * heads * head_dim // world
        if got.numel() != expected_numel:
            failures.append(
                f"rank {rank} captured {name} size {got.numel()} != {expected_numel}"
            )
    for failure in failures:
        print(f"FAILURE: {failure}", flush=True)
    return int(bool(failures))


class TestZingH3Ulysses(CustomTestCase):
    def test_uneven_rows_and_head_sharded_history_match_sp1(self):
        if not current_platform.is_cuda():
            self.skipTest("CUDA-only test")
        if torch.cuda.device_count() < _WORLD:
            self.skipTest(f"needs {_WORLD} GPUs")
        processes = []
        for rank in range(_WORLD):
            env = os.environ.copy()
            env.update(
                {
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                    "WORLD_SIZE": str(_WORLD),
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": "29754",
                }
            )
            processes.append(
                subprocess.Popen(
                    [sys.executable, __file__],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        outputs = [process.communicate(timeout=300)[0] for process in processes]
        if any(process.returncode for process in processes):
            self.fail("worker failed:\n" + "\n".join(outputs))


if __name__ == "__main__":
    if "RANK" in os.environ:
        sys.exit(_worker())
    unittest.main()
