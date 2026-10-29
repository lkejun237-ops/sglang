# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the fixed-case key and early preparation bypass."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.conditioning.worker import WorkerConditioningState
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    source_session,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.media_geometry import (
    SourceGeometry,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
    H3InferenceInputs,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
    Block0PreparationCache,
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


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("joint_forward", ["off", "cross_chunk"])
def test_startup_wires_cache_without_gating_joint(
    monkeypatch, joint, cache_enabled, joint_forward
):
    prepared = inputs()
    condition = WorkerConditioningState()
    entry = NS(inputs=prepared, conditioning_identity="cached", request=object())
    owner = NS(
        mode="on",
        identity={"model": "fixture", "layout": {}},
        lookup=Mock(return_value=entry),
        restore=Mock(),
    )
    lease = NS(kv_cache=object())
    sampler = NS(
        window_policy=NS(window_chunks=6, sink_chunks=3),
        rope_position_mode="relative",
        rope_max_frame_gap=12,
        joint_forward=joint_forward,
    )
    codecs = NS(
        remote_read_fs=None,
        video_decoder_name="lynn",
        vae_frames_per_chunk=17,
        vae_latents_per_chunk=5,
        temporal_ratio=4,
        spatial_ratio=16,
        runtime={
            "h3_media_config": {
                "max_short_edge": 2048,
                "max_area_config": {"source": 1032192},
                "max_source_target_aspect_ratio_factor": 1.02,
            }
        },
        pipeline_config=NS(
            vae_config=NS(arch_config=NS(spatial_compression_ratio=16)),
            dit_config=NS(arch_config=NS(patch_size=(1, 2, 2))),
        ),
        prepare=Mock(return_value=prepared),
    )
    stage = NS(
        codecs=codecs,
        block0_cache=owner if cache_enabled else None,
        modules={"transformer": torch.nn.Linear(1, 1)},
        _validate_sampling_params=lambda _: None,
        _build_realtime_sampler=lambda *a, **kw: sampler,
        _build_request=lambda _: dict(prompt="scene"),
        _conditioning_consumer=Mock(),
        packer=NS(chunk_plan=NS(lengths=(1, 2, 2)), visual_channels=2),
        prompt_switch_prefix_latent_frames=5,
        kv_pool=NS(acquire=Mock(return_value=lease)),
    )
    ready = NS(_new_cache=Mock(return_value=object()))
    build_ready = Mock(return_value=ready)
    monkeypatch.setattr(source_session, "SourceReadySession", build_ready)
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
        extra={
            "zing_source_command": {
                "encoder": {"mode": "progressive"},
                "prompt_joint": joint,
                "source_geometry": asdict(SourceGeometry(32, 32, 32, 32, 32, 32)),
            }
        },
        realtime_session_id="new-session",
        realtime_event_id="new-event",
        session=NS(get_or_create_state=lambda _: condition),
    )
    state.initialize(stage, batch, dict(width=32, height=32, fps=24))
    assert codecs.prepare.call_count == int(not cache_enabled)
    assert stage._conditioning_consumer.call_count == int(not cache_enabled)
    assert stage.kv_pool.acquire.call_count == int(cache_enabled)
    assert owner.restore.call_count == int(cache_enabled)
    assert build_ready.call_args.kwargs["prompt_joint"] is joint
    cached_joint = joint and joint_forward == "cross_chunk"
    assert build_ready.call_args.kwargs["prefix_loader"] == (
        state._load_block0_prefix if cached_joint and cache_enabled else None
    )
    assert hasattr(ready, "prepare_prompt_prefix") is cache_enabled
    if cache_enabled:
        assert ready.prepare_prompt_prefix == state._prepare_block0_switch
    assert hasattr(ready, "restore_initial_prefix") is (
        cache_enabled and not cached_joint
    )
    if cache_enabled:
        assert condition.identity == "cached"
        assert condition.positive is state.inputs.positive
        stage.kv_pool.acquire.assert_called_once_with(batch.session, entry.request)


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("joint_forward", ["off", "cross_chunk"])
@pytest.mark.parametrize("env_joint", [False, True])
@pytest.mark.parametrize("command_joint", [None, False, True])
def test_joint_policy_is_independent_of_cache(
    monkeypatch, cache_enabled, joint_forward, env_joint, command_joint
):
    monkeypatch.setenv("SGLANG_ZING_SOURCE_PROMPT_JOINT", str(int(env_joint)))
    stage = NS(
        block0_cache=NS(identity={"model": "fixture", "layout": {}})
        if cache_enabled
        else None,
        codecs=NS(video_decoder_name="lynn"),
    )
    sampler = NS(
        joint_forward=joint_forward,
        window_policy=NS(window_chunks=21, sink_chunks=3),
        rope_position_mode="block_relative",
        rope_max_frame_gap=12,
    )
    command = {} if command_joint is None else {"prompt_joint": command_joint}
    policy, _, joint = source_session._source_key_policy(stage, sampler, 1, command)
    assert joint is (env_joint if command_joint is None else command_joint)
    assert policy["source_prompt_joint"] is joint


def test_startup_joint_consumer_commits_existing_transfer():
    state = source_session.ZingSourceStreamState()
    state.block0_entry = object()
    state.block0_restore = NS(commit=Mock())
    state.block0_cache = NS(lookup=Mock(), restore=Mock())
    state.sampler = NS(applied_prompt=None)
    cache = object()
    assert state._load_block0_prefix(cache, None, None, None)
    state.block0_restore.commit.assert_called_once_with(cache)
    state.block0_cache.lookup.assert_not_called()
    state.block0_cache.restore.assert_not_called()


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


def test_fill_accepts_host_positions_and_waits_on_kv_device(monkeypatch):
    stream = NS(device=torch.device("cuda:0"), wait_stream=Mock())
    done = NS(record=Mock(), synchronize=Mock())
    empty = torch.empty

    def allocate(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda: done)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    store = NS(k=torch.zeros(8, 2, 3))
    pool = NS(
        _pool=NS(_slots=[NS(kv_cache=[store])]),
        plan=NS(config=NS(max_prefix_rows=8)),
    )
    owner = Block0PreparationCache(
        pool, max_bytes=2 * 8 * 2 * 3 * 4, identity={}, disk_cache=None
    )
    positions = torch.arange(9).reshape(3, 3)
    cache = NS(
        prefix=NS(
            positions=positions,
            layers=[(torch.ones(3, 2, 3), torch.full((3, 2, 3), 2.0))],
        ),
        _model_signature="model",
    )

    def current_stream(device):
        if device.type != "cuda":
            raise ValueError(f"Expected a cuda device, but got: {device}")
        assert device == owner.stream.device
        return stream

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    owner.fill("host-positions", inputs(), "condition", "request", cache)
    assert not owner.entries  # Publish only after the asynchronous copy finishes.
    owner.finish_fill()
    entry = owner.entries["host-positions"]
    assert torch.equal(entry.positions, positions)
    assert torch.equal(entry.data[0, 0], cache.prefix.layers[0][0])
    assert torch.equal(entry.data[0, 1], cache.prefix.layers[0][1])
    done.synchronize.assert_called_once()
