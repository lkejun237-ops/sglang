# SPDX-License-Identifier: Apache-2.0
"""Startup compilation must finish before KV allocation and worker readiness."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import sglang.multimodal_gen.runtime.pipelines.zing_h3_causal_pipeline as pipeline_module


@pytest.mark.parametrize("fails", [False, True])
def test_lynn_preparation_precedes_pool_and_readiness(monkeypatch, fails):
    calls = []

    def warmup(device):
        calls.append("warmup")
        if fails:
            raise RuntimeError("Injected preparation compilation failure")
        return 18

    decoder = SimpleNamespace(warmup_preparation=warmup)
    monkeypatch.setattr(
        pipeline_module, "build_decoder_modules", lambda spec: {"lynn_decoder": decoder}
    )
    stage = Mock()
    stage._build_realtime_sampler.side_effect = lambda params: calls.append("sampler")
    stage.initialize_realtime_kv_cache_pool.side_effect = lambda args: calls.append(
        "pool"
    )
    monkeypatch.setattr(
        pipeline_module, "ZingH3GenerationStage", lambda *a, **kw: stage
    )
    pipeline = object.__new__(pipeline_module.ZingH3CausalPipeline)
    pipeline.modules = {}
    pipeline.conditioning_mode = "local"
    pipeline.zing_runtime = {}
    pipeline.conditioning_loader_receipt = {}
    pipeline.add_stage = lambda stage: calls.append("stage")
    args = SimpleNamespace(
        pipeline_config=SimpleNamespace(resolve_decoder_spec=lambda: None),
        zing_conditioning_local_cache_bytes=0,
        zing_solver="few_step",
        zing_denoising_step_list=[1000, 750, 500, 250],
        zing_initial_denoising_step_list=None,
        zing_initial_block_count=0,
    )
    if fails:
        with pytest.raises(RuntimeError, match="Injected preparation"):
            pipeline.create_pipeline_stages(args)
        assert calls == ["sampler", "warmup"]
    else:
        pipeline.create_pipeline_stages(args)
        assert calls == ["sampler", "warmup", "pool", "stage"]


@pytest.mark.parametrize(
    "capability,sm_count,expected",
    [
        ((9, 0), 132, True),  # H100/H200 family, independent of SM count.
        ((9, 0), 114, True),
        ((10, 3), 148, True),  # B300 family, including other SM configurations.
        ((10, 3), 140, True),
        ((8, 9), 148, False),
        ((10, 0), 148, False),
        ((12, 0), 148, False),
    ],
)
def test_preparation_device_support_without_cuda(
    monkeypatch, capability, sm_count, expected
):
    from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.attention import (
        _supports_preparation_device,
    )

    properties = SimpleNamespace(
        major=capability[0], minor=capability[1], multi_processor_count=sm_count
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: properties)
    _supports_preparation_device.cache_clear()
    try:
        assert _supports_preparation_device(torch.device("cuda:0")) is expected
    finally:
        _supports_preparation_device.cache_clear()


def test_unsupported_device_skips_startup_preparation(monkeypatch):
    from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming import attention
    from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.model import (
        StreamingDecoder,
    )

    decoder = SimpleNamespace(
        compute_dtype=torch.bfloat16,
        config=SimpleNamespace(temporal_window=3, tile_size=64),
        spatial_ratio=4,
        architecture={
            "decoder_num_attention_heads": 32,
            "decoder_attention_head_dim": 64,
            "decoder_num_register_tokens": 4,
            "decoder_rope_dim_ratio": 0.75,
        },
    )
    monkeypatch.setattr(attention, "_supports_preparation_device", lambda device: False)
    allocate = Mock(
        side_effect=AssertionError("Unsupported device allocated CUDA memory")
    )
    monkeypatch.setattr(torch, "zeros", allocate)
    assert StreamingDecoder.warmup_preparation(decoder, "cuda:0") == 0
    allocate.assert_not_called()
