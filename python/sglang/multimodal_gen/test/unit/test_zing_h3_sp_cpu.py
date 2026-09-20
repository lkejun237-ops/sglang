# SPDX-License-Identifier: Apache-2.0
"""CPU layout regression; this does not validate CUDA kernels or real weights."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, local
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
from sglang.multimodal_gen.runtime.layers import usp
from sglang.multimodal_gen.runtime.models.dits import zing_h3
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import model, request  # noqa: F401


class _CPUCollectives:
    """Replace only communication, keeping production A2A packing/unpacking."""

    def __init__(self, world):
        self.world = world
        self.thread = local()
        self.barrier = Barrier(world, timeout=20)
        self.values = [None] * world

    def exchange(self, value, consume):
        rank = self.thread.rank
        self.values[rank] = value
        self.barrier.wait()
        result = consume(self.values, rank)
        self.barrier.wait()
        return result

    def all_to_all(self, value, output_splits, input_splits):
        def receive(values, rank):
            chunks = []
            for sender, (tensor, splits) in enumerate(values):
                start = sum(splits[:rank])
                assert splits[rank] == output_splits[sender]
                chunks.append(tensor.flatten()[start : start + splits[rank]])
            return torch.cat(chunks)

        return self.exchange((value, input_splits), receive)

    def gather(self, value, splits, group, *, dim):
        assert value.shape[dim] == splits[self.thread.rank]
        return self.exchange(value, lambda values, rank: torch.cat(values, dim=dim))


@torch.no_grad()
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("reference_prefix", [False, True])
def test_real_projections_rope_and_committed_kv_match_cpu_sp1(
    model, monkeypatch, world, reference_prefix  # noqa: F811 (imported pytest fixture)
):
    if world == 4:
        arch = replace(
            model.arch,
            num_attention_heads=4,
            hidden_size=32,
            adaln_out_features=768,
            final_adaln_out_features=64,
        )
        model = zing_h3.ZingH3DiTModel(ZingH3DiTConfig(arch_config=arch), {}).eval()
        torch.manual_seed(7)
        for name, param in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                param.fill_(1)
            else:
                param.copy_(torch.randn(param.shape) * 0.08)
        model.rope.inv_freq.fill_(1)
    packer, inputs, visual, audio = request(model)
    sampler = ZingH3Sampler(model, packer, sigma_points=3)
    sampler.device = torch.device("cpu")
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, audio)
    transactions = []
    for block, phase in [
        (0, "prefix"),
        (1, "source"),
        (2, "target"),
        (3, "source"),
        (4, "target"),
    ]:
        target = phase == "target"
        packed = sampler._select_block(
            layout,
            block,
            phase=phase,
            video_t=0.25 if target else 1.0,
            audio_t=0.4 if target else 1.0,
        )
        transactions.append((packed, "active" if target else phase))
        if target:
            transactions.append(
                (
                    sampler._select_block(
                        layout,
                        block,
                        phase="target",
                        video_t=1,
                        audio_t=1,
                        final=True,
                    ),
                    "final",
                )
            )
    assert any(packed["t"].numel() % 2 for packed, _ in transactions)
    assert any(bool(packed["position_ids"].any()) for packed, _ in transactions)
    if reference_prefix:
        prefix = transactions[0][0]
        target = next(packed for packed, phase in transactions if phase == "active")
        roles = target["row_role"].clone()
        roles[roles == 10] = 4
        roles[roles == 11] = 5
        roles = torch.cat((prefix["row_role"], roles))
        rows = roles.numel()
        transactions[0] = (
            {
                **prefix,
                "x": target["x"],
                "audio_x": target["audio_x"],
                "row_role": roles,
                "t": torch.ones(rows, dtype=prefix["t"].dtype),
                "block_idx": torch.zeros(rows, dtype=prefix["block_idx"].dtype),
                "seq_lens": torch.full_like(prefix["seq_lens"], rows),
                "clean_seq_lens": torch.full_like(prefix["clean_seq_lens"], rows),
                "position_ids": torch.cat(
                    (prefix["position_ids"], target["position_ids"])
                ),
            },
            "prefix",
        )

    reference_cache = ZingH3KVCache(len(model.blocks))
    references = [
        sampler._forward(packed, reference_cache, phase)
        for packed, phase in transactions
    ]
    communication = _CPUCollectives(world)
    monkeypatch.setattr(
        zing_h3, "get_ulysses_ctx", lambda: (world, communication.thread.rank)
    )
    monkeypatch.setattr(
        zing_h3, "get_sp_group", lambda: SimpleNamespace(ulysses_group=None)
    )
    monkeypatch.setattr(zing_h3, "gather_sequence_varlen", communication.gather)
    monkeypatch.setattr(usp, "get_ulysses_parallel_world_size", lambda: world)
    monkeypatch.setattr(
        usp, "get_ulysses_parallel_rank", lambda: communication.thread.rank
    )
    monkeypatch.setattr(usp, "_ipc_ready_group", lambda: None)
    monkeypatch.setattr(usp, "_usp_all_to_all_single_varlen", communication.all_to_all)

    @torch.no_grad()
    def rank_worker(rank):
        communication.thread.rank = rank
        cache = ZingH3KVCache(len(model.blocks))
        outputs = [
            sampler._forward(packed, cache, phase) for packed, phase in transactions
        ]
        return outputs, cache

    with ThreadPoolExecutor(max_workers=world) as executor:
        futures = [executor.submit(rank_worker, rank) for rank in range(world)]
        actual, errors = [], []
        for future in futures:
            try:
                actual.append(future.result())
            except Exception as error:
                errors.append(error)
        if errors:
            pytest.fail(f"CPU SP workers failed: {errors!r}")
    for rank, (outputs, cache) in enumerate(actual):
        for got, expected in zip(outputs, references):
            for value, reference in zip(got, expected):
                torch.testing.assert_close(value, reference, atol=2e-3, rtol=0.025)
        assert cache.last_block == reference_cache.last_block == 4
        assert cache.active is None and cache.pending_source is None
        torch.testing.assert_close(cache.positions(), reference_cache.positions())
        for layer in range(len(model.blocks)):
            for value, reference in zip(
                cache.history(layer), reference_cache.history(layer)
            ):
                torch.testing.assert_close(
                    value, reference[:, rank : rank + 1], atol=2e-3, rtol=0.025
                )
