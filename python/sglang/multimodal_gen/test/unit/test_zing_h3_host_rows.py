# SPDX-License-Identifier: Apache-2.0
"""Host row metadata must reproduce the device-metadata path bitwise."""

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.zing_h3 import ZingH3DiTModel
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    _sampling_helpers as helpers,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
    SourceReadySession,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import (
    _long_request,
    fixture_provenance,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import model as model_fixture
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    request,
)


@pytest.fixture
def model():
    return model_fixture.__wrapped__()


def _run_session(model, window, guidance):
    packer, inputs, noise, audio = _long_request(model, [3, 2])
    inputs.conditions = [{"content_id": "ref", "role": "reference"}]
    inputs.media = [{"id": "ref", "type": "video"}]
    inputs.visual = {"ref": torch.randn(1, 2, 2, 2, 2)}
    inputs.audio = {"ref": torch.randn(1, 2, 3, 2)}
    options = dict(
        solver="few_step",
        denoising_step_list=[1000, 750, 500, 250],
        guidance_scale=guidance,
    )
    if window:
        options.update(
            window_chunks=2, sink_chunks=1, rope_position_mode="block_relative"
        )
    else:
        options.update(window_chunks=None, sink_chunks=0, rope_position_mode="absolute")
    sampler = ZingH3Sampler(model, packer, **options)
    session = SourceReadySession(
        sampler,
        inputs,
        seed=31,
        max_pending_chunks=2,
        max_pending_bytes=100000,
        initial_rng_state=(torch.get_rng_state().clone(), None),
    )
    outputs, cursor = [], 0
    for chunk in packer.chunk_plan.chunks(noise.shape[2]):
        indices = packer._audio_indices_for_video_span(
            audio.shape[2], chunk.start, chunk.end, noise.shape[2]
        )
        session.append(
            inputs.source_visual[:, :, chunk.start : chunk.end],
            noise[:, :, chunk.start : chunk.end],
            audio=audio[:, :, indices],
            audio_start=cursor,
            provenance=fixture_provenance(
                100 + chunk.chunk_id, 200 + chunk.end, chunk.start, chunk.end
            ),
            final=chunk.end == noise.shape[2],
        )
        cursor += len(indices)
        result = session.step()
        outputs.append((result.visual, result.audio))
    assert session.complete
    return outputs


@torch.no_grad()
@pytest.mark.parametrize("window", [False, True])
@pytest.mark.parametrize("guidance", [1, 2])
def test_host_rows_match_device_rows_bitwise(model, monkeypatch, window, guidance):
    calls = []
    forward = ZingH3DiTModel.forward

    def record(self, *args, **kwargs):
        calls.append(kwargs.get("host_layout") is not None)
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(ZingH3DiTModel, "forward", record)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        host = _run_session(model, window, guidance)
    assert calls and all(calls)
    calls.clear()
    monkeypatch.setattr(helpers, "_host_metadata", lambda packed: None)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        device = _run_session(model, window, guidance)
    assert calls and not any(calls)
    for (host_video, host_audio), (video, audio) in zip(host, device, strict=True):
        assert torch.equal(host_video, video)
        assert torch.equal(host_audio, audio)


@torch.no_grad()
def test_changed_packed_rows_fall_back_to_device_metadata(model):
    packer, inputs, visual, _ = request(model, source=False, audio=False)
    sampler = ZingH3Sampler(model, packer, sigma_points=2)
    sampler.device = torch.device("cpu")
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, None)

    def select():
        return sampler._select_block(
            layout, 2, phase="target", video_t=0.2, audio_t=None
        )

    assert helpers._host_metadata(select()) is not None
    replaced = select()
    replaced["clean_seq_lens"] = replaced["seq_lens"]
    assert helpers._host_metadata(replaced) is None
    edited = select()
    edited["t"].fill_(1)
    assert helpers._host_metadata(edited) is None
    # Selections share static rows but never a step's t.
    first, second = select(), select()
    assert first["position_ids"] is second["position_ids"]
    assert first["t"] is not second["t"]


@torch.no_grad()
def test_forward_rejects_host_rows_of_another_selection(model):
    packer, inputs, visual, audio = request(model)
    sampler = ZingH3Sampler(model, packer, sigma_points=2)
    sampler.device = torch.device("cpu")
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, audio)
    source = sampler._select_block(layout, 1, phase="source", video_t=1, audio_t=1)
    target = sampler._select_block(layout, 2, phase="target", video_t=0.2, audio_t=0.3)
    target[helpers._HOST_LAYOUT] = source[helpers._HOST_LAYOUT]
    with pytest.raises(ValueError, match="host row metadata does not match"):
        sampler._forward(target, None, None)
    target[helpers._HOST_T] = None
    with pytest.raises(ValueError, match="requires both layout and t"):
        model(
            **{name: target[name] for name in helpers._GUARDED_ROWS},
            x=target["x"],
            audio_x=target["audio_x"],
            context=target["context"],
            host_layout=source[helpers._HOST_LAYOUT],
        )


@torch.no_grad()
def test_attention_positions_reuse_until_inputs_change():
    cache = ZingH3KVCache(1, rope_position_mode="block_relative")
    current = torch.tensor([[3.0, 0, 0], [3.0, 0, 1]])
    first = cache.attention_positions(current)
    assert cache.attention_positions(current) is first
    current.add_(1)
    second = cache.attention_positions(current)
    assert second is not first
    assert torch.equal(second[0], cache._map_attention_positions(current, False)[0])
    cache.clear()
    assert cache._attention_positions_memo is None


@torch.no_grad()
def test_active_geometry_is_checked_on_host_copies():
    cache = ZingH3KVCache(1)
    positions = torch.tensor([[1.0, 0, 0]])
    block_idx = torch.tensor([2])
    host = (positions.clone(), block_idx.clone())
    cache.update(positions, block_idx, "active", 1, validation=host)
    assert cache._active_reference is not cache.active[1]
    moved = (positions + 1, block_idx.clone())
    with pytest.raises(ValueError, match="geometry changed"):
        cache.validate(*moved, "final", 1)
    # A caller without host copies still compares the device snapshot.
    assert cache.validate(positions, block_idx, "final", 1) == 2
