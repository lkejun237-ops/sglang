# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the fixed-case key and early preparation bypass."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.conditioning.worker import WorkerConditioningState
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    source_session,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
    H3InferenceInputs,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
    inputs_on,
    preparation_key,
)


def inputs():
    return H3InferenceInputs(
        output_type="video",
        conditions=[],
        media=[],
        positive={"context": torch.ones(2, 4), "row_role": torch.zeros(2)},
        visual={"ref": torch.ones(1, 2, 1, 2, 2)},
        audio={},
        source_visual=None,
        visual_shape=(1, 2, 2),
        audio_shape=None,
        pixel_frame_count=1,
        sample_count=None,
        sample_rate=24000,
        condition_media=[{"id": "ref", "frames": torch.ones(3, 1, 2, 2)}],
    )


def test_key_uses_contents_order_s0_and_request_settings(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"first")
    b.write_bytes(b"second")
    codecs = NS(remote_read_fs=None)
    config = dict(width=32, height=32, fps=24)
    preview = dict(frames=torch.zeros(3, 1, 32, 32, dtype=torch.uint8))
    request = dict(
        prompt="scene",
        media=[dict(type="image", uri=str(a)), dict(type="image", uri=str(b))],
        max_area=1024,
    )

    def key(value=request, s0=preview):
        return preparation_key(codecs, value, s0, config, {})

    original = key()
    assert key(deepcopy(request)) == original
    assert key({**request, "prompt": "other"}) != original
    assert key({**request, "media": request["media"][::-1]}) != original
    assert key({**request, "max_area": 4096}) != original
    assert key(s0=dict(frames=torch.ones_like(preview["frames"]))) != original
    a.write_bytes(b"changed at the same path")
    assert key() != original


def test_cpu_snapshot_retains_pixels_and_owns_prepared_tensors():
    source = inputs()
    frozen = inputs_on(source, "cpu", snapshot=True)
    source.positive["context"].zero_()
    source.visual["ref"].zero_()
    source.condition_media[0]["frames"].zero_()
    assert frozen.positive["context"].sum() == 8
    assert frozen.visual["ref"].sum() == 8
    assert frozen.condition_media[0]["frames"].sum() == 12


@pytest.mark.parametrize("mode", ["late", "early", "overlap"])
def test_complete_hit_bypasses_prepare_and_reuses_pool(monkeypatch, mode):
    prepared = inputs()
    condition = WorkerConditioningState()
    entry = NS(inputs=prepared, conditioning_identity="cached", request=object())
    owner = NS(mode=mode, lookup=Mock(return_value=entry), restore=Mock())
    lease = NS(kv_cache=object())
    sampler = NS(
        window_policy=NS(window_chunks=6, sink_chunks=3),
        rope_position_mode="relative",
        rope_max_frame_gap=12,
    )
    codecs = NS(
        remote_read_fs=None,
        video_decoder_name="lynn",
        vae_frames_per_chunk=17,
        vae_latents_per_chunk=5,
        temporal_ratio=4,
        prepare=Mock(return_value=prepared),
    )
    stage = NS(
        codecs=codecs,
        block0_cache=owner,
        modules={"transformer": torch.nn.Linear(1, 1)},
        _validate_sampling_params=lambda _: None,
        _build_realtime_sampler=lambda *a, **kw: sampler,
        _build_request=lambda _: dict(prompt="scene"),
        _conditioning_consumer=Mock(),
        packer=NS(chunk_plan=NS(lengths=(1, 2, 2))),
        prompt_switch_prefix_latent_frames=5,
        kv_pool=NS(acquire=Mock(return_value=lease)),
    )
    ready = NS(_new_cache=Mock(return_value=object()))
    monkeypatch.setattr(source_session, "SourceReadySession", lambda *a, **kw: ready)
    monkeypatch.setattr(
        source_session, "SourceLedger", lambda **kw: NS(provenance_for=None)
    )
    monkeypatch.setattr(
        source_session, "SourceChunkBridge", lambda *a, **kw: NS(seed=kw["seed"])
    )
    monkeypatch.setattr(source_session, "create_source_buffer", lambda *a, **kw: NS())
    monkeypatch.setattr(source_session, "create_decoder", lambda *a, **kw: NS())
    state = source_session.ZingSourceStreamState()
    monkeypatch.setattr(state, "_encoder_use", lambda _: nullcontext())
    batch = NS(
        sampling_params=NS(prompt="scene", seed=123),
        extra={"zing_source_command": {"encoder": {"mode": "progressive"}}},
        realtime_session_id="new-session",
        realtime_event_id="new-event",
        session=NS(get_or_create_state=lambda _: condition),
    )
    state.initialize(stage, batch, dict(width=32, height=32, fps=24))
    assert codecs.prepare.call_count == int(mode == "late")
    assert stage._conditioning_consumer.call_count == int(mode == "late")
    assert stage.kv_pool.acquire.call_count == int(mode != "late")
    assert owner.restore.call_count == int(mode != "late")
    if mode != "late":
        assert condition.identity == "cached"
        assert condition.positive is state.inputs.positive
        stage.kv_pool.acquire.assert_called_once_with(batch.session, entry.request)


def test_fill_starts_on_dispatch_after_output():
    state = source_session.ZingSourceStreamState()
    prefix = object()
    state.block0_prefix = prefix
    state.block0_cache = NS(fill=Mock())
    state.block0_key = "key"
    state.block0_inputs = inputs()
    state.block0_condition_identity = "condition"
    state.block0_request = object()
    state.sampler = NS(cache=NS(prefix=prefix))
    state.ledger = NS(output_frames=0)
    state._fill_block0_after_output()
    state.block0_cache.fill.assert_not_called()
    state.ledger.output_frames = 1
    state._fill_block0_after_output()
    state.block0_cache.fill.assert_called_once()
    state._fill_block0_after_output()
    state.block0_cache.fill.assert_called_once()
