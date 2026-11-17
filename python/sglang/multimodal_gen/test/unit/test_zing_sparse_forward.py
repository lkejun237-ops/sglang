# SPDX-License-Identifier: Apache-2.0
"""Real small CPU model contracts, not CUDA throughput or image acceptance."""

import os
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
    ZingH3DiTArchConfig,
    ZingH3DiTConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.models.dits.zing_h3 import ZingH3DiTModel
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
    ZingH3Packer,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import allocate_layer_stores


@pytest.fixture
def small_model(monkeypatch):
    for key, value in {
        "SGLANG_ZING_EXPERIMENTAL_ATTENTION": "torch_sdpa",
        "SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES": "0",
        "SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN": "0",
        "SGLANG_ZING_DENOISE_CUDA_GRAPH": "0",
    }.items():
        monkeypatch.setenv(key, value)
    if not model_parallel_is_initialized():
        for key, value in dict(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29505",
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        ).items():
            os.environ.setdefault(key, value)
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    arch = ZingH3DiTArchConfig(
        num_layers=2,
        token_refiner_num_layers=1,
        hidden_size=256,
        num_attention_heads=2,
        attention_head_dim=128,
        ffn_hidden_size=512,
        latents_dim=2,
        audio_latents_dim=2,
        text_dim=12,
        timestep_input_dim=8,
        time_embed_hidden_size=16,
        time_embed_dim=8,
        adaln_out_features=6144,
        final_adaln_out_features=512,
        rope_inv_freq_len=16,
    )
    model = ZingH3DiTModel(ZingH3DiTConfig(arch_config=arch), {}).eval()
    torch.manual_seed(7)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                param.fill_(1)
            else:
                param.copy_(torch.randn(param.shape) * 0.08)
        model.rope.inv_freq.fill_(1)
    return model


def session(model, *, frames=11, retain=True):
    packer = ZingH3Packer(model.arch, num_frame_per_block=2)
    sampler = ZingH3Sampler(
        model,
        packer,
        solver="few_step",
        denoising_step_list=[1000, 500],
        window_chunks=3,
        sink_chunks=1,
        window_eviction_timing="before_source",
        rope_position_mode="block_relative",
    )
    torch.manual_seed(11)
    inputs = NS(
        positive={
            "context": torch.randn(3, 12).bfloat16(),
            "row_role": torch.tensor([0, 1, 0]),
        },
        visual={},
        audio={},
        source_visual=torch.randn(1, 2, frames, 2, 2),
        conditions=[],
        media=[],
        visual_shape=(frames, 2, 2),
    )
    storage = allocate_layer_stores(
        256,
        num_layers=2,
        local_heads=2,
        head_dim=128,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    state = sampler.initialize(
        inputs,
        torch.randn(1, 2, frames, 2, 2),
        storage=storage,
        retain_prefix_layers=retain,
    )
    return sampler, state


@torch.no_grad()
def test_real_model_prefix_export_and_windowed_continuation(small_model, monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_SPARSE_ATTN_MODE", "dense")
    monkeypatch.setenv("SGLANG_ZING_SPARSE_THREE_TIER", "0")
    sampler, baseline = session(small_model)
    while baseline.index < len(baseline.spans):
        sampler.step(baseline)
    expected = baseline.visual.clone()
    baseline.close()
    monkeypatch.setenv("SGLANG_ZING_SPARSE_ATTN_MODE", "sparse_095")
    monkeypatch.setenv("SGLANG_ZING_SPARSE_THREE_TIER", "1")
    sampler, state = session(small_model)
    prefix = tuple((k.clone(), v.clone()) for k, v in state.cache.prefix.layers)
    assert len(prefix) == 2
    try:
        while state.index < len(state.spans):
            sampler.step(state)
            assert torch.isfinite(state.visual).all()
        # This tiny fixture exercises protected 100% Refine plus actual FP8
        # round trips. Production-sized router and perceptual parity are separate.
        torch.testing.assert_close(state.visual, expected, atol=0.04, rtol=0.04)
        for actual, wanted in zip(state.cache.prefix.layers, prefix):
            for a, b in zip(actual, wanted):
                assert torch.equal(a, b)
        assert state.cache.stats["compactions"] > 0
        assert state.cache._compressed_kv_stores[0].num_blocks > 1
    finally:
        state.close()
