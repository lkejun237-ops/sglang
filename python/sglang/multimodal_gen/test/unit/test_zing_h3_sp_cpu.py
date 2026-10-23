# SPDX-License-Identifier: Apache-2.0
"""Layout regressions with CPU collectives and optional CUDA packing checks."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, local
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
from sglang.multimodal_gen.runtime.layers import usp
from sglang.multimodal_gen.runtime.models.dits import zing_h3
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
    model,
    request,
)

_USP = "sglang.multimodal_gen.runtime.layers.usp"


@pytest.mark.parametrize("lengths", [(5, 3), (3, 0, 5, 2)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_varlen_destination_major_preserves_tokens_and_aliases_receive(
    lengths, device, monkeypatch
):
    monkeypatch.delenv("SGLANG_ZING_SP_DESTINATION_MAJOR", raising=False)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    world, heads, dim = len(lengths), 8, 16
    # Projection slices have a 3*H*D row stride, as in the fixed Zing path.
    projection = torch.randn(sum(lengths), 3 * heads * dim, device=device)
    full = tuple(x.reshape(1, -1, heads, dim) for x in projection.chunk(3, -1))
    shards = [
        tuple(x.split(lengths, dim=1)[rank] for x in full) for rank in range(world)
    ]
    sends, receives = [], []
    recording = True

    def collective(packed, output_sizes, input_sizes):
        assert output_sizes == [n * (heads // world) * 3 * dim for n in lengths]
        if recording:
            sends.append(packed.flatten().split(input_sizes))
            return packed.new_empty(sum(output_sizes))
        return receives.pop(0)

    with (
        patch(f"{_USP}.get_ulysses_parallel_world_size", return_value=world),
        patch(f"{_USP}._usp_all_to_all_single_varlen", collective),
    ):
        for rank in range(world):
            with patch(f"{_USP}.get_ulysses_parallel_rank", return_value=rank):
                zing_h3._exchange_qkv(*(x[0] for x in shards[rank]), list(lengths))
        receives = [torch.cat([send[rank] for send in sends]) for rank in range(world)]
        recording = False
        for rank in range(world):
            received = receives[0]
            with patch(f"{_USP}.get_ulysses_parallel_rank", return_value=rank):
                actual = zing_h3._exchange_qkv(
                    *(x[0] for x in shards[rank]), list(lengths)
                )
            expected = torch.cat([x.chunk(world, dim=2)[rank] for x in full], dim=-1)
            assert torch.equal(actual, expected)
            assert actual.data_ptr() == received.data_ptr()


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

    def gather_objects(self, output, value, group=None):
        """Model rank agreement with real per-thread payloads and barriers."""
        output[:] = self.exchange(value, lambda values, rank: list(values))

    def gather(self, value, splits, group, *, dim):
        assert value.shape[dim] == splits[self.thread.rank]
        return self.exchange(value, lambda values, rank: torch.cat(values, dim=dim))


@torch.no_grad()
@pytest.mark.parametrize("count", [4, 5])
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("reference_prefix", [False, True])
def test_real_projections_rope_and_committed_kv_match_cpu_sp1(
    model,  # noqa: F811 (imported pytest fixture)
    monkeypatch,
    world,
    reference_prefix,
    count,
):
    if count == 5:
        arch = replace(
            model.arch,
            num_modalities=5,
            adaln_out_features=6 * 5 * model.arch.hidden_size,
            modality_tags=dict(
                video=4, text=0, audio=3, source_visual=1, condition_visual=2
            ),
        )
        model = zing_h3.ZingH3DiTModel(ZingH3DiTConfig(arch_config=arch), {}).eval()
        torch.manual_seed(7)
        for name, param in model.named_parameters():
            param.copy_(
                torch.ones_like(param)
                if "norm" in name and name.endswith("weight")
                else torch.randn_like(param) * 0.08
            )
        model.rope.inv_freq.fill_(1)
    if world == 4:
        arch = replace(
            model.arch,
            num_attention_heads=4,
            hidden_size=32,
            adaln_out_features=6 * count * 32,
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
