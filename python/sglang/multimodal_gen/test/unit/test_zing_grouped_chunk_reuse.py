# SPDX-License-Identifier: Apache-2.0
"""Tiny real model/cache wiring; no checkpoint or CUDA kernel claim on CPU."""

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits import zing_h3 as model_module
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.models.dits.zing_h3_rope import HISTORY_ROPE_ROUTES
from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import model  # noqa: F401
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (
    ZingH3Sampler,
    request,
)


@pytest.mark.parametrize("grouped", [None, "0", "contiguous"])
@pytest.mark.parametrize("reuse", [None, "0"])
@pytest.mark.parametrize("fused", [None, "0"])
@torch.no_grad()
def test_model_routes_multiple_invalid_parts(model, monkeypatch, grouped, reuse, fused):
    for name, value in (
        ("SGLANG_ZING_GROUPED_HISTORY_ROPE", grouped),
        ("SGLANG_ZING_CHUNK_ROPE_REUSE", reuse),
        ("SGLANG_ZING_FUSED_HISTORY_ROPE", fused),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    calls = []
    for name in ("apply_history_rope_batch", "apply_contiguous_history_rope_batch"):
        original = getattr(model_module, name)

        def record(xs, cs, out, original=original, name=name):
            calls.append((name, len(xs)))
            return original(xs, cs, out)

        monkeypatch.setattr(model_module, name, record)
    packer, inputs, visual, audio = request(model)
    sampler = ZingH3Sampler(model, packer, sigma_points=2)
    sampler.device = torch.device("cpu")
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, audio)
    c = ZingH3KVCache(
        2, window_policy=ZingWindowPolicy(4, 1), rope_position_mode="block_relative"
    )
    for block, phase in [(0, "prefix"), (1, "source")]:
        packed = sampler._select_block(layout, block, phase=phase, video_t=1, audio_t=1)
        sampler._forward(packed, c, phase)
    # Invalidate publication so prefix and source must both rebuild together.
    c._rotated_store_signatures.clear()
    calls.clear()
    before = HISTORY_ROPE_ROUTES.copy()
    packed = sampler._select_block(layout, 2, phase="target", video_t=0.25, audio_t=0.4)
    actual = sampler._forward(packed, c, "active")
    if grouped == "0":
        assert not calls
    else:
        expected = (
            "apply_contiguous_history_rope_batch"
            if grouped == "contiguous"
            else "apply_history_rope_batch"
        )
        assert calls == [(expected, 2)] * 2
        reason = "disabled" if fused == "0" else "device"
        assert (
            HISTORY_ROPE_ROUTES["grouped_fallback_" + reason]
            == before["grouped_fallback_" + reason] + 2
        )
    # Compare the same weights and raw cache with the eager per-part route.
    monkeypatch.setenv("SGLANG_ZING_GROUPED_HISTORY_ROPE", "0")
    monkeypatch.setenv("SGLANG_ZING_FUSED_HISTORY_ROPE", "0")
    c._rotated_store_signatures.clear()
    expected = sampler._forward(packed, c, "active")
    for a, b in zip(actual, expected):
        assert torch.equal(a, b)
    c.clear()
    assert not list(c.owned_tensors())
