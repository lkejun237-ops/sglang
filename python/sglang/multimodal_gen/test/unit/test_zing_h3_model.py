# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
    ZingH3DiTArchConfig,
    ZingH3DiTConfig,
)
from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams
from sglang.multimodal_gen.runtime.models.dits.zing_h3 import ZingH3DiTModel
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import ZingH3KVCache
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
    ZingH3Packer,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.test.unit.test_minimax_h3_dit_contract import (
    _ensure_single_process_parallel_runtime,
)


@pytest.fixture(autouse=True)
def cpu_execution_options(monkeypatch):
    # These tests exercise layout/arithmetic on CPU, not the default CUDA path.
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_ATTENTION", "torch_sdpa")
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES", "0")
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN", "0")


@pytest.fixture
def model():
    _ensure_single_process_parallel_runtime()
    arch = ZingH3DiTArchConfig(
        num_layers=2,
        token_refiner_num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        attention_head_dim=8,
        ffn_hidden_size=32,
        latents_dim=2,
        audio_latents_dim=2,
        text_dim=12,
        timestep_input_dim=8,
        time_embed_hidden_size=16,
        time_embed_dim=8,
        adaln_out_features=384,
        final_adaln_out_features=32,
        rope_inv_freq_len=1,
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


def request(model, *, source=True, audio=True):
    packer = ZingH3Packer(model.arch, num_frame_per_block=2)
    inputs = SimpleNamespace(
        positive={
            "context": torch.randn(3, 12).bfloat16(),
            "row_role": torch.tensor([0, 1, 0]),
        },
        visual={},
        audio={},
        source_visual=torch.randn(1, 2, 3, 2, 2) if source else None,
        conditions=[],
        media=[],
        visual_shape=(3, 2, 2),
    )
    visual = torch.randn(1, 2, 3, 2, 2)
    sound = torch.randn(1, 2, 20, 2) if audio else None
    return packer, inputs, visual, sound


def test_zing_h3_sp_request_cannot_disable_sequence_sharding(monkeypatch):
    monkeypatch.setattr(
        ZingH3SamplingParams.__mro__[1],
        "_adjust",
        lambda _self, _server_args: None,
    )
    params = ZingH3SamplingParams(enable_sequence_shard=False)
    with pytest.raises(ValueError, match="requires enable_sequence_shard=True"):
        params._adjust(SimpleNamespace(sp_degree=4))


@pytest.mark.parametrize("num_frames", [22, 240])
@pytest.mark.parametrize("sp_degree", [1, 2, 4])
def test_zing_h3_sp_request_preserves_requested_frame_count(
    monkeypatch, num_frames, sp_degree
):
    monkeypatch.setattr(
        ZingH3SamplingParams.__mro__[1],
        "_adjust",
        lambda _self, _server_args: None,
    )
    params = ZingH3SamplingParams(num_frames=num_frames)
    params.fps = 30
    params._adjust(SimpleNamespace(sp_degree=sp_degree))
    assert params.num_frames == num_frames
    assert params.fps == 30
    assert params.enable_sequence_shard is True
    assert params.adjust_frames is False


@torch.no_grad()
def test_incremental_kv_matches_explicit_clean_history(model):
    packer, inputs, visual, audio = request(model)
    sampler = ZingH3Sampler(model, packer, sigma_points=3)
    sampler.device = torch.device("cpu")
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, audio)
    cache = ZingH3KVCache(2)
    clean_parts = []
    for block, phase in [
        (0, "prefix"),
        (1, "source"),
        (2, "target"),
        (3, "source"),
        (4, "target"),
    ]:
        is_target = phase == "target"
        packed = sampler._select_block(
            layout,
            block,
            phase=phase,
            video_t=0.25 if is_target else 1.0,
            audio_t=0.4 if is_target else 1.0,
        )
        cached = sampler._forward(packed, cache, "active" if is_target else phase)
        full = {
            k: torch.cat([p[k] for p in clean_parts] + [packed[k]])
            for k in (
                "x",
                "audio_x",
                "context",
                "t",
                "block_idx",
                "position_ids",
                "row_role",
            )
        }
        full["seq_lens"] = torch.tensor([full["t"].numel()])
        full["clean_seq_lens"] = torch.tensor(
            [
                sum(p["t"].numel() for p in clean_parts)
                + (0 if is_target else packed["t"].numel())
            ]
        )
        expected = model(**full)
        for current, ref in zip(cached, expected):
            if current.shape[0]:
                torch.testing.assert_close(
                    current, ref[-current.shape[0] :], atol=2e-3, rtol=0.025
                )
        if is_target:
            packed = sampler._select_block(
                layout, block, phase="target", video_t=1, audio_t=1, final=True
            )
            before = len(cache.committed)
            sampler._forward(packed, cache, "final")
            assert len(cache.committed) == before + 1
        clean_parts.append(packed)
    assert cache.last_block == 4 and cache.active is None
    for chunk in [cache.prefix, *cache.committed]:
        for layer, (_k, v) in enumerate(chunk.layers):
            # The store owns every committed row, so a chunk must alias it
            # rather than retain a slice of the fused projection.
            assert (
                v.untyped_storage().data_ptr()
                == cache._stores[layer][1].untyped_storage().data_ptr()
            )
    cache.clear()
    assert cache.positions() is None


@torch.no_grad()
def test_causal_execution_runs_through_block_hooks(model):
    packer, inputs, visual, audio = request(model)
    sampler = ZingH3Sampler(model, packer, sigma_points=2)
    counts = [0] * len(model.blocks)
    handles = []
    for index, block in enumerate(model.blocks):
        handles.append(
            block.register_forward_pre_hook(
                lambda _module, _args, index=index: counts.__setitem__(
                    index, counts[index] + 1
                )
            )
        )
    try:
        sampler.sample(inputs, visual, audio)
    finally:
        for handle in handles:
            handle.remove()
    assert all(count > 0 for count in counts)
    assert model.layer_names == ["blocks"]


@pytest.mark.parametrize("solver", ["euler", "few_step"])
@pytest.mark.parametrize("with_audio", [False, True])
@torch.no_grad()
def test_block_sampler_runs_cfg_and_tail(model, solver, with_audio):
    packer, inputs, video, audio = request(model, audio=with_audio)
    sampler = ZingH3Sampler(
        model,
        packer,
        solver=solver,
        sigma_points=3,
        denoising_step_list=[1000, 500] if solver == "few_step" else None,
        guidance_scale=2,
    )
    output, sound = sampler.sample(inputs, video, audio)
    assert output.shape == video.shape and torch.isfinite(output).all()
    assert not torch.equal(output, video)
    assert (sound is None) == (audio is None)
    if sound is not None:
        assert sound.shape == audio.shape and torch.isfinite(sound).all()
    assert sampler.last_run_metrics == {
        "window_chunks": 21,
        "sink_chunks": 3,
        "rope_position_mode": "block_relative",
        "rope_max_frame_gap": 12,
        "block_spans": [
            {"source_block": 1, "target_block": 2, "start": 0, "end": 2, "frames": 2},
            {"source_block": 3, "target_block": 4, "start": 2, "end": 3, "frames": 1},
        ],
        "video_sigma_points": 3 if solver == "euler" else 2,
        "video_denoise_steps_per_block": 2,
        "guidance_forwards_per_step": 2,
        "forward_counts": {
            "prefix": 1,
            "source": 2,
            "denoise": 4,
            "unconditional": 4,
            "commit": 2,
        },
        "dit_forwards": 13,
    }


@torch.no_grad()
def test_cache_rejects_noisy_commit_and_duplicate_final(model):
    packer, inputs, visual, _ = request(model, source=False, audio=False)
    sampler = ZingH3Sampler(model, packer, sigma_points=2)
    sampler.device = torch.device("cpu")
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, None)
    cache = ZingH3KVCache(2)
    target = sampler._select_block(layout, 2, phase="target", video_t=0.2, audio_t=None)
    sampler._forward(target, cache, "active")
    assert cache.history(0) is None
    target["clean_seq_lens"] = target["seq_lens"]
    with pytest.raises(ValueError, match="clean t=1"):
        sampler._forward(target, cache, "final")
    target["t"].fill_(1)
    sampler._forward(target, cache, "final")
    with pytest.raises(ValueError, match="consecutive order"):
        sampler._forward(target, cache, "final")


@torch.no_grad()
def test_cache_rejects_head_shard_layout_changes():
    cache = ZingH3KVCache(2)
    prefix_positions = torch.zeros(2, 3, dtype=torch.long)
    prefix_blocks = torch.zeros(2, dtype=torch.long)
    prefix_layers = tuple(
        (torch.zeros(2, 2, 8), torch.zeros(2, 2, 8)) for _ in range(2)
    )
    cache.update(prefix_positions, prefix_blocks, "prefix", 0, prefix_layers)

    source_positions = torch.ones(3, 3, dtype=torch.long)
    source_blocks = torch.ones(3, dtype=torch.long)
    changed_layers = tuple(
        (torch.zeros(3, 1, 8), torch.zeros(3, 1, 8)) for _ in range(2)
    )
    with pytest.raises(ValueError, match="head-shard layout changed"):
        cache.update(source_positions, source_blocks, "source", 1, changed_layers)


@torch.no_grad()
def test_generation_stage_connects_prompt_dit_and_decoder(model):
    from contextlib import nullcontext

    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )

    packer, inputs, _, _ = request(model, source=False, audio=False)
    inputs.audio_shape, inputs.pixel_frame_count = None, 5
    inputs.sample_rate, inputs.output_type = 32000, "video"
    stage = object.__new__(ZingH3GenerationStage)
    stage.modules = {"transformer": model}
    stage.packer = packer
    stage.runtime = dict(
        timestep_shift=5.0, audio_timestep_shift=3.0, num_train_timestep=1000
    )
    seen = []

    def prepare(raw, use):
        seen.append(raw)
        return inputs

    def decode(latent, frames):
        assert latent.shape == (1, 2, 3, 2, 2) and frames == 5
        return torch.zeros(1, 3, 5, 32, 32)

    stage.codecs = SimpleNamespace(
        video_decoder_name="video_vae",
        prepare=prepare,
        decode_visual=decode,
        spatial_ratio=16,
        target_fps=24.0,
    )
    stage._use = lambda name, phase="encode": nullcontext(model)
    params = ZingH3SamplingParams(
        prompt="A moving scene",
        num_frames=5,
        height=32,
        width=32,
        num_inference_steps=3,
        zing_return_latents=True,
    )
    batch = SimpleNamespace(sampling_params=params, extra={}, metrics=None)
    with set_forward_context(current_timestep=0, attn_metadata=None):
        output = stage.forward(batch, None)
    assert seen[0]["messages"][0]["content"][0]["text"] == "A moving scene"
    assert output.output.shape == (1, 3, 5, 32, 32)
    assert output.trajectory_latents.shape == (1, 2, 3, 2, 2)
    assert output.audio is None
    assert output.usage["zing_h3"]["dit_forwards"] == 7
    assert output.usage["zing_h3"]["block_spans"] == [
        {"source_block": 1, "target_block": 2, "start": 0, "end": 2, "frames": 2},
        {"source_block": 3, "target_block": 4, "start": 2, "end": 3, "frames": 1},
    ]
    assert params.num_frames == 5


@torch.no_grad()
def test_request_failure_releases_session_kv(model, monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        sampling,
    )

    caches = []
    original = sampling.ZingH3KVCache

    def cache(*a, **kw):
        obj = original(*a, **kw)
        caches.append(obj)
        return obj

    monkeypatch.setattr(sampling, "ZingH3KVCache", cache)
    packer, inputs, video, audio = request(model)

    def fail(*a, **kw):
        raise RuntimeError("request cancelled")

    sampler = ZingH3Sampler(model, packer, sigma_points=3)
    monkeypatch.setattr(sampler, "_commit", fail)
    with pytest.raises(RuntimeError, match="cancelled"):
        sampler.sample(inputs, video, audio)
    assert (
        len(caches) == 1 and caches[0].positions() is None and caches[0].active is None
    )


@pytest.mark.parametrize("encode_mode", ["deterministic", "performance"])
@torch.no_grad()
def test_export_and_native_loader_keep_all_tensors_and_precision(
    model, tmp_path, encode_mode
):
    import importlib.util
    import json
    from pathlib import Path

    import yaml
    from safetensors.torch import load_file

    from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
        ComponentLoader,
    )
    from sglang.multimodal_gen.runtime.loader.fsdp_load import (
        load_model_from_full_model_state_dict,
    )
    from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
    from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

    path = (
        Path(__file__).resolve().parents[5]
        / "benchmark/zing_h3_causal/export_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location("zing_export", path)
    export_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_module)
    a = model.arch
    arch = dict(
        dim=a.hidden_size,
        num_heads=a.num_attention_heads,
        head_dim=a.attention_head_dim,
        ffn_dim=a.ffn_hidden_size,
        num_layers=a.num_layers,
        num_refiner_layers=a.token_refiner_num_layers,
        in_dim=a.latents_dim,
        audio_in_dim=a.audio_latents_dim,
        text_dim=a.text_dim,
        freq_dim=a.timestep_input_dim,
        time_embed_hidden_dim=a.time_embed_hidden_size,
        time_embed_dim=a.time_embed_dim,
        patch_size=list(a.patch_size),
        eps=a.norm_eps,
        qk_norm_eps=a.qk_norm_eps,
        final_norm_eps=a.final_norm_eps,
        num_modalities=4,
        modality_tags=dict(video=0, text=1, audio=2, source_visual=3),
        local_attn_size=-1,
        sink_size=0,
        rope_config=dict(
            freq_dim=a.rope_inv_freq_len,
            theta=10000.0,
            spatial_extent=32.0,
            frame_scale=5 / 3,
            visual_token_frame_spans=[1, 4, 4, 4, 4],
        ),
    )
    cfg = dict(
        generator_config=arch,
        causal=True,
        num_frame_per_block=2,
        split_first_latent=False,
        num_train_timestep=1000,
        timestep_shift=5.0,
        audio_timestep_shift=3.0,
        h3_media_config={},
        condition_encoder_config=dict(output_hidden_layer=50),
        vae_config={"encode_mode": encode_mode},
    )
    yaml_path = tmp_path / "effective.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg))
    components = tmp_path / "components"
    components.mkdir()
    for component in ("text_encoder", "tokenizer", "processor", "audio_vae", "vae"):
        (components / component).mkdir()
    (components / "vae/source").mkdir()
    (components / "vae/source/model.safetensors").write_bytes(b"test source weights")
    state = {
        key: torch.randn(shape) * 0.03
        for key, shape in export_module.expected_shapes(arch).items()
    }
    checkpoint = tmp_path / "model.pt"
    torch.save(state, checkpoint)
    destination = tmp_path / "export"
    result = export_module.export(
        checkpoint, yaml_path, components, destination, shard_bytes=4096
    )
    assert (
        json.loads((destination / "zing_config.json").read_text())["vae_config"][
            "encode_mode"
        ]
        == encode_mode
    )
    invalid_cfg = {**cfg, "vae_config": {"encode_mode": "fast"}}
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(yaml.safe_dump(invalid_cfg))
    with pytest.raises(ValueError, match="encode_mode"):
        export_module.export(checkpoint, invalid_path, components, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    assert result["tensors"] == 12 * a.num_layers + 10 * a.token_refiner_num_layers + 18
    index = json.loads((destination / "model_index.json").read_text())
    assert index["video_vae"] == ["diffusers", "MiniMaxH3VideoVAE"]
    assert index["audio_vae"] == ["diffusers", "MiniMaxH3AudioVAE"]
    assert (destination / "video_vae/source/model.safetensors").resolve() == (
        components / "vae/source/model.safetensors"
    ).resolve()
    for name in ("MiniMaxH3VideoVAE", "MiniMaxH3AudioVAE"):
        _, resolved_name = ModelRegistry.resolve_model_cls(name)
        assert resolved_name == name
    for alias, name in (
        ("AutoencoderKLMiniMaxH3", "MiniMaxH3VideoVAE"),
        ("AutoencoderKLMiniMaxH3Audio", "MiniMaxH3AudioVAE"),
    ):
        _, resolved_name = ModelRegistry.resolve_model_cls(alias)
        assert resolved_name == name
    for key, entry in index.items():
        if not key.startswith("_"):
            ComponentLoader.for_component_type(key, *entry)
    config = ZingH3DiTConfig()
    config.update_model_arch(
        json.loads((destination / "transformer/config.json").read_text())
    )
    with torch.device("meta"):
        loaded = ZingH3DiTModel(config, {})
    weights = {}
    for f in sorted((destination / "transformer").glob("*.safetensors")):
        weights.update(load_file(f))
    load_model_from_full_model_state_dict(
        loaded,
        loaded.preprocess_loaded_state_dict(weights.items()),
        torch.device("cpu"),
        torch.bfloat16,
        strict=True,
        param_names_mapping=get_param_names_mapping(loaded.param_names_mapping),
    )
    loaded.post_load_weights()
    assert {
        n
        for n, p in loaded.named_parameters()
        if p.dtype == torch.float32 and n != "rope.inv_freq"
    } == {n for n, p in model.named_parameters() if p.dtype == torch.float32}
    assert not any(p.is_meta for p in loaded.parameters())
    assert torch.equal(loaded.rope.inv_freq, torch.ones(1))
    with pytest.raises(FileExistsError):
        export_module.export(checkpoint, yaml_path, components, destination)
    dcp_checkpoint = tmp_path / "global_step_000001"
    dcp_checkpoint.mkdir()
    with pytest.raises(ValueError, match="DCP directories"):
        export_module.export(
            dcp_checkpoint, yaml_path, components, tmp_path / "dcp-export"
        )
    del state["backbone.context_embedder.bias"]
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="checkpoint keys"):
        export_module.export(checkpoint, yaml_path, components, tmp_path / "incomplete")
    assert not (tmp_path / "incomplete").exists()


@torch.no_grad()
@pytest.mark.parametrize("solver", ["euler", "few_step"])
def test_incremental_blocks_match_offline_sampler(model, solver):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.realtime import (
        ZingH3RealtimeState,
    )

    packer, inputs, visual, audio = request(model)
    kwargs = {"solver": solver, "sigma_points": 3}
    if solver == "few_step":
        kwargs["denoising_step_list"] = [1000, 500]
    offline = ZingH3Sampler(model, packer, **kwargs)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        expected_video, expected_audio = offline.sample(inputs, visual, audio)
    sampler = ZingH3Sampler(model, packer, **kwargs)
    state = sampler.initialize(inputs, visual, audio)
    rng = ZingH3RealtimeState()
    with rng.random_stream(torch.device("cpu"), 123):
        assert sampler.step(state) == (0, 2)
    assert state.cache.last_block == 2
    assert len(state.cache.committed) == 1
    assert state.cache.active is state.cache.pending_source is None
    torch.testing.assert_close(state.visual[:, :, 2:], visual[:, :, 2:])
    # An unrelated request must not perturb the second block's re-noise RNG.
    torch.randn(99)
    with rng.random_stream(torch.device("cpu"), 123):
        assert sampler.step(state) == (2, 3)
    assert state.cache.last_block == 4
    assert len(state.cache.committed) == 2
    torch.testing.assert_close(state.visual, expected_video, rtol=0, atol=0)
    torch.testing.assert_close(state.audio, expected_audio, rtol=0, atol=0)
    state.close()
    assert state.cache.history(0) is None
    assert state.visual is state.positive is None


@torch.no_grad()
@pytest.mark.parametrize("incremental", [False, True])
def test_sampler_preserves_diagnostics_across_realtime_split(
    model, monkeypatch, incremental
):
    from contextlib import nullcontext

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        sampling,
    )

    completed, phases, regions = [], [], []
    diagnostic = SimpleNamespace(phase=None)
    diagnostic.record = lambda name, **values: completed.append(
        (name, diagnostic.phase, values["video"].clone())
    )
    monkeypatch.setattr(sampling, "current_diagnostic", lambda: diagnostic)
    monkeypatch.setattr(
        sampling,
        "maybe_record_function",
        lambda name: (regions.append(name), nullcontext())[1],
    )
    packer, inputs, visual, audio = request(model)
    sampler = ZingH3Sampler(model, packer, sigma_points=3)
    original = sampler._forward

    def forward(*args, **kwargs):
        phases.append(diagnostic.phase)
        return original(*args, **kwargs)

    monkeypatch.setattr(sampler, "_forward", forward)
    if incremental:
        state = sampler.initialize(inputs, visual, audio)
        try:
            while state.index < len(state.spans):
                sampler.step(state)
            actual = state.visual.clone()
        finally:
            state.close()
    else:
        actual, _ = sampler.sample(inputs, visual, audio)
    assert phases == [
        "prefix",
        "block1.source",
        "block2.step0",
        "block2.step1",
        "block2.commit",
        "block3.source",
        "block4.step0",
        "block4.step1",
        "block4.commit",
    ]
    assert [(name, phase) for name, phase, _ in completed] == [
        ("block_complete", "block2.commit"),
        ("block_complete", "block4.commit"),
    ]
    torch.testing.assert_close(completed[-1][2], actual, rtol=0, atol=0)
    assert sum(name.startswith("Zing.causal_block") for name in regions) == 2
    assert sum(name.startswith("Zing.denoise_step") for name in regions) == 4
    assert sampler.video_scheduler.diagnostic_modality == "video"
    assert sampler.audio_scheduler.diagnostic_modality == "audio"


@torch.no_grad()
@pytest.mark.parametrize("failure_phase", ["active", "final"])
def test_second_block_failure_clears_history_and_closes_state(
    model, monkeypatch, failure_phase
):
    packer, inputs, visual, audio = request(model)
    sampler = ZingH3Sampler(model, packer, sigma_points=3)
    state = sampler.initialize(inputs, visual, audio)
    sampler.step(state)
    original = sampler._forward

    def failing(packed, cache, update):
        if update == failure_phase:
            assert cache.pending_source is not None
            raise RuntimeError("cancelled block")
        return original(packed, cache, update)

    monkeypatch.setattr(sampler, "_forward", failing)
    with pytest.raises(RuntimeError, match="cancelled"):
        sampler.step(state)
    assert state.closed
    assert not state.cache.committed and state.cache.history(0) is None
    assert state.cache.pending_source is state.cache.active is None
    with pytest.raises(ValueError, match="closed"):
        sampler.step(state)
    state.close()


@torch.no_grad()
def test_prompt_updates_keep_absolute_video_positions_and_committed_kv(model):
    packer, inputs, visual, _ = request(model, audio=False)
    sampler = ZingH3Sampler(model, packer, sigma_points=2)
    state = sampler.initialize(inputs, visual)
    sampler.step(state)
    committed = state.cache.committed[0]
    positions = state.positive["position_ids"][state.positive["block_idx"] > 0].clone()
    for length in (7, 2):
        old_prefix = state.cache.prefix
        inputs.positive = {
            "context": torch.randn(length, 12).bfloat16(),
            "row_role": torch.zeros(length, dtype=torch.long),
        }
        sampler.update_condition(state, inputs)
        assert state.cache.prefix is not old_prefix
        assert state.cache.committed[0] is committed
        torch.testing.assert_close(
            state.positive["position_ids"][state.positive["block_idx"] > 0],
            positions,
            rtol=0,
            atol=0,
        )
    sampler.step(state)
    assert state.cache.last_block == 4
    state.close()


@pytest.fixture
def realtime_stage(model):
    from contextlib import nullcontext

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )

    packer, inputs, _, _ = request(model, audio=False)
    inputs.audio_shape, inputs.pixel_frame_count = None, 9
    inputs.sample_rate, inputs.output_type = 32000, "video"
    inputs.condition_media = []
    stage = object.__new__(ZingH3GenerationStage)
    stage.modules = {"transformer": model}
    stage.packer = packer
    stage.runtime = {
        "timestep_shift": 5.0,
        "audio_timestep_shift": 3.0,
        "num_train_timestep": 1000,
    }
    seen = []

    def prepare(raw, use, **kwargs):
        seen.append(raw)
        return inputs

    def decode(latent, frames):
        # Encode temporal index into pixels to catch repeated/skipped frame slices.
        return (
            torch.arange(frames)
            .view(1, 1, frames, 1, 1)
            .expand(1, 3, frames, 32, 32)
            .float()
        )

    stage.codecs = SimpleNamespace(
        video_decoder_name="video_vae",
        prepare=prepare,
        decode_visual_range=lambda latent, start, end: decode(latent, end)[
            :, :, start:end
        ],
        spatial_ratio=16,
        temporal_ratio=4,
        target_fps=24.0,
        vae_latents_per_chunk=5,
        vae_frames_per_chunk=17,
        _parse_inference_request=lambda raw: (
            raw["messages"][0]["content"][0]["text"],
        ),
        encode_condition=lambda prompt, media: inputs.positive,
    )
    stage._use = lambda name, phase="encode": nullcontext(model)
    return stage, seen


def _realtime_batch(index, session):
    return SimpleNamespace(
        sampling_params=ZingH3SamplingParams(
            prompt="A moving scene",
            num_frames=9,
            height=32,
            width=32,
            num_inference_steps=3,
            seed=42,
            zing_realtime_decode_mode="immediate",
        ),
        extra={},
        metrics=None,
        realtime_session_id="zing-session",
        block_idx=index,
        session=session,
    )


@torch.no_grad()
def test_realtime_stage_returns_first_block_before_second_and_releases_final(
    realtime_stage,
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.realtime import (
        ZingH3RealtimeState,
    )
    from sglang.multimodal_gen.runtime.realtime.session import RealtimeSession

    stage, seen = realtime_stage
    session = RealtimeSession()
    first = stage.forward(_realtime_batch(0, session), None)
    state = session.get_state(ZingH3RealtimeState)
    cache = state.sampling.cache
    assert state.sampling.index == 1 and cache.last_block == 2
    assert first.output.shape[2] == 5
    second = stage.forward(_realtime_batch(1, session), None)
    assert len(seen) == 1
    joined = torch.cat((first.output, second.output), dim=2)
    torch.testing.assert_close(joined[0, 0, :, 0, 0], torch.arange(9).float())
    assert state.sampling is state.inputs is state.cpu_rng is None
    assert cache.history(0) is None


@torch.no_grad()
@pytest.mark.parametrize("end", ["disconnect", "decode_failure", "out_of_order"])
def test_realtime_session_cleanup(realtime_stage, end):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.realtime import (
        ZingH3RealtimeState,
    )
    from sglang.multimodal_gen.runtime.realtime.session import RealtimeSessionCache

    stage, _ = realtime_stage
    sessions = RealtimeSessionCache()
    first = _realtime_batch(0, None)
    sessions.attach(first)
    stage.forward(first, None)
    state = first.session.get_state(ZingH3RealtimeState)
    sampling = state.sampling
    if end == "disconnect":
        assert sessions.release(first.realtime_session_id)
    elif end == "out_of_order":
        with pytest.raises(ValueError, match="in order"):
            stage.forward(_realtime_batch(3, first.session), None)
    else:

        def fail(*args):
            raise RuntimeError("decoder failed")

        stage.codecs.decode_visual_range = fail
        with pytest.raises(RuntimeError, match="decoder failed"):
            stage.forward(_realtime_batch(1, first.session), None)
    assert sampling.closed and sampling.cache.history(0) is None
    assert state.inputs is state.sampling is state.cpu_rng is state.cuda_rng is None


@torch.no_grad()
def test_stable_decode_waits_for_complete_windows_and_preserves_block_events(
    realtime_stage,
):
    from sglang.multimodal_gen.runtime.realtime.session import RealtimeSession

    stage, _ = realtime_stage
    stage.packer.num_frame_per_block = 5
    inputs = stage.codecs.prepare(None, None)
    inputs.visual_shape = (12, 2, 2)
    inputs.source_visual = torch.randn(1, 2, 12, 2, 2)
    inputs.pixel_frame_count = 39
    stage.codecs.va = SimpleNamespace(vae_token_drop=3)
    session = RealtimeSession()
    outputs, events = [], []
    for index in range(3):
        batch = _realtime_batch(index, session)
        batch.sampling_params.zing_realtime_decode_mode = "stable"
        batch.realtime_event_id = index + 1
        batch.request_id = f"block-{index}"
        result = stage.forward(batch, None)
        if index == 0:
            assert result.output is None
            assert result.raw_frame_batches == []
        else:
            outputs.extend(result.output)
            events.extend(result.usage["zing_realtime_blocks"])
            assert len(result.output) == (1 if index == 1 else 2)
    assert [x.shape[1] for x in outputs] == [17, 17, 5]
    assert [x["event_id"] for x in events] == [1, 2, 3]
    joined = torch.cat(outputs, dim=1)
    torch.testing.assert_close(joined[0, :, 0, 0], torch.arange(39).float())


@pytest.mark.parametrize("pattern", [(5,), (1, 2, 2)])
@pytest.mark.parametrize("latent_end,frames", [(1, 1), (3, 9), (7, 22), (12, 39)])
def test_taeh3_stage_preserves_frames_and_block_provenance(pattern, latent_end, frames):
    """Feed fixed latents on different AR boundaries, including early EOS."""
    from contextlib import nullcontext

    from sglang.multimodal_gen.runtime.models.vaes.taeh3 import TAEH3Decoder
    from sglang.multimodal_gen.runtime.models.vaes.taeh3.upstream import TAEHV
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.realtime import (
        ZingH3RealtimeState,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )

    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9)
            upstream = TAEHV(None, arch_name="taeh3").eval()
            latent = torch.randn(1, 24, 12, 1, 2)
        decoder = object.__new__(TAEH3Decoder)
        torch.nn.Module.__init__(decoder)
        decoder.decoder = upstream.decoder
        stage = object.__new__(ZingH3GenerationStage)
        stage.modules = {"taeh3_decoder": decoder}
        stage._use = lambda *args: nullcontext(decoder)
        stage.codecs = SimpleNamespace(
            video_decoder_name="taeh3_decoder",
            spatial_ratio=16,
            temporal_ratio=4,
            vae_latents_per_chunk=5,
            vae_frames_per_chunk=17,
            va=SimpleNamespace(vae_token_drop=3),
        )
        state = ZingH3RealtimeState()
        state.inputs = SimpleNamespace(visual_shape=(12, 1, 2), pixel_frame_count=39)
        state.sampling = SimpleNamespace(
            visual=latent, index=0, spans=[None] * 100, close=lambda: None
        )
        pieces, events, cursor, index = [], [], 0, 0
        while cursor < latent_end:
            cursor = min(latent_end, cursor + pattern[index % len(pattern)])
            state.sampling.index += 1
            batch = SimpleNamespace(
                block_idx=index, request_id=f"r{index}", realtime_event_id=index
            )
            final = cursor == latent_end
            ready = stage._queue_realtime_output(batch, state, cursor, "stable", final)
            if ready:
                outputs, height, width = stage._decode_realtime_output(
                    state, cursor, ready, final=final
                )
                assert (height, width) == (16, 32)
                pieces.extend(outputs)
                events.extend(ready)
            index += 1
        joined = torch.cat(pieces, dim=1).unsqueeze(0)
        expected = decoder.decode(latent[:, :, :latent_end], frames)
        torch.testing.assert_close(joined, expected, atol=0, rtol=0)
        assert [event["block_idx"] for event in events] == list(range(index))
        assert [event["event_id"] for event in events] == list(range(index))
        assert [event["request_id"] for event in events] == [
            f"r{i}" for i in range(index)
        ]
        assert events[-1]["frame_end"] == frames
        assert state.emitted_frames == frames and not state.pending_blocks
        state.dispose()
        assert state.video_decoder is None and state.decoder_frames is None
    finally:
        torch.set_num_threads(threads)


@torch.no_grad()
@pytest.mark.parametrize("count", [4, 5])
def test_permuted_modality_routing_preserves_forward_and_committed_kv(model, count):
    from dataclasses import replace

    roles = ["video", "text", "audio", "source_visual"]
    if count == 5:
        roles.append("condition_visual")
    canonical = dict(zip(roles, range(count)))
    permuted = (
        dict(video=2, text=0, audio=3, source_visual=1, condition_visual=2)
        if count == 4
        else dict(video=4, text=0, audio=3, source_visual=1, condition_visual=2)
    )
    arch = replace(
        model.arch,
        num_modalities=count,
        adaln_out_features=6 * count * model.arch.hidden_size,
        modality_tags=canonical,
        ar_condition_timestep=0.875,
        reference_visual_timestep=0.999,
    )
    reference = ZingH3DiTModel(ZingH3DiTConfig(arch_config=arch), {}).eval()
    # Initialize every parameter, including the independent fifth branch.
    torch.manual_seed(7)
    for name, param in reference.named_parameters():
        param.copy_(
            torch.ones_like(param)
            if "norm" in name and name.endswith("weight")
            else torch.randn_like(param) * 0.08
        )
    reference.rope.inv_freq.fill_(1)
    candidate = ZingH3DiTModel(
        ZingH3DiTConfig(arch_config=replace(arch, modality_tags=permuted)), {}
    ).eval()
    weights = {key: value.clone() for key, value in reference.state_dict().items()}
    for layer in range(arch.num_layers):
        prefix = f"blocks.{layer}.adaln_proj.linear."
        for suffix in ("weight", "bias"):
            parts = weights[prefix + suffix].chunk(count, dim=0)
            weights[prefix + suffix] = torch.cat(
                [
                    parts[canonical[role]]
                    for role in sorted(roles, key=permuted.__getitem__)
                ],
                dim=0,
            )
    candidate.load_state_dict(weights)
    packer, inputs, visual, audio = request(reference)
    inputs.visual = {"ref": torch.randn(1, 2, 1, 2, 2)}
    inputs.media = [{"id": "ref", "type": "image"}]
    inputs.conditions = [{"content_id": "ref", "role": "reference"}]
    layout = packer.pack_inference_layout(inputs, inputs.positive, visual, audio)
    samplers = [
        ZingH3Sampler(
            current,
            packer,
            sigma_points=3,
            ar_condition_timestep=0.875,
            reference_visual_timestep=0.999,
        )
        for current in (reference, candidate)
    ]
    for sampler in samplers:
        sampler.device = torch.device("cpu")
    caches = [ZingH3KVCache(arch.num_layers) for _ in samplers]
    # Reuse one host layout across both models to catch derived-cache tag aliasing.
    for block, phase in [(0, "prefix"), (1, "source"), (2, "active"), (2, "final")]:
        packed = samplers[0]._select_block(
            layout,
            block,
            phase="target" if phase in ("active", "final") else phase,
            video_t=0.25,
            audio_t=0.4,
            final=phase == "final",
        )
        outputs = [
            sampler._forward(packed, cache, phase)
            for sampler, cache in zip(samplers, caches)
        ]
        for actual, expected in zip(outputs[1], outputs[0]):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        if phase != "active":
            for layer in range(arch.num_layers):
                actual, expected = caches[1].history(layer), caches[0].history(layer)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want, atol=0, rtol=0)
        if phase == "prefix":
            roles_tensor = packed["row_role"]
            assert bool(((roles_tensor == 0) | (roles_tensor == 1)).any())
            assert bool((roles_tensor == 4).any())
            assert bool(
                (packed["t"][(roles_tensor == 0) | (roles_tensor == 1)] == 0.875).all()
            )
            assert bool((packed["t"][roles_tensor == 4] == 0.999).all())
            bad = dict(packed, t=torch.ones_like(packed["t"]))
            with pytest.raises(ValueError, match="checkpoint-defined prefix"):
                samplers[0]._forward(bad, ZingH3KVCache(arch.num_layers), "prefix")


@pytest.mark.parametrize(
    "count,tags",
    [
        (3, None),
        (4, {}),
        (4, dict(video=0, text=1, audio=2, source_visual=3, condition_visual=4)),
        (5, dict(video=0, text=1, audio=2, source_visual=3, condition_visual=0)),
        (4, dict(video=False, text=1, audio=2, source_visual=3)),
    ],
)
def test_invalid_modality_routing_rejected_by_model(model, count, tags):
    from dataclasses import replace

    with pytest.raises(ValueError):
        ZingH3DiTModel(
            ZingH3DiTConfig(
                arch_config=replace(
                    model.arch, num_modalities=count, modality_tags=tags
                )
            ),
            {},
        )


def test_rational_rope_scale_matches_numeric_geometry(model):
    rational = ZingH3Packer(model.arch, frame_scale="5/3")
    numeric = ZingH3Packer(model.arch, frame_scale=5 / 3)
    assert rational.rope_frame_scale == numeric.rope_frame_scale
    with pytest.raises(ValueError, match="geometry"):
        ZingH3Packer(model.arch, frame_scale="5/0")


def verify_synthetic_table(table, directory):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import (
        checkpoint_identity,
    )

    shard, config = directory / "synthetic.safetensors", directory / "config.json"
    shard.write_bytes(b"synthetic shard")
    config.write_text("{}")
    table.manifest["identity"] = checkpoint_identity([shard], config)
    table.validate_checkpoint([shard], config)


@torch.no_grad()
@pytest.mark.parametrize(
    "count,tags",
    [
        (4, None),
        (4, dict(video=2, text=0, audio=3, source_visual=1, condition_visual=2)),
        (5, dict(video=4, text=0, audio=3, source_visual=1, condition_visual=2)),
    ],
)
@pytest.mark.parametrize("solver", ["euler", "few_step"])
def test_weights_free_table_matches_resident_latents_and_kv(
    model, tmp_path, count, tags, solver
):
    from dataclasses import replace

    from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import (
        ZingAdalnTable,
        video_plan,
        write_table,
    )

    arch = replace(
        model.arch,
        num_modalities=count,
        modality_tags=tags,
        adaln_out_features=6 * count * model.arch.hidden_size,
        ar_condition_timestep=0,
        reference_visual_timestep=0.999,
    )
    resident = ZingH3DiTModel(ZingH3DiTConfig(arch_config=arch), {}).eval()
    torch.manual_seed(257)
    for name, p in resident.named_parameters():
        p.copy_(
            torch.ones_like(p)
            if "norm" in name and name.endswith("weight")
            else torch.randn_like(p) * 0.08
        )
    resident.rope.inv_freq.fill_(1)
    packer, inputs, noise, _ = request(resident, audio=False)
    inputs.visual = {f"ref{i}": torch.randn(1, 2, 1, 2, 2) for i in range(6)}
    inputs.media = [{"id": key, "type": "image"} for key in inputs.visual]
    inputs.conditions = [
        {"content_id": key, "role": "reference"} for key in inputs.visual
    ]
    options = dict(
        solver=solver,
        sigma_points=3,
        denoising_step_list=[1000, 500] if solver == "few_step" else None,
        ar_condition_timestep=0,
        reference_visual_timestep=0.999,
        guidance_scale=2,
    )
    baseline = ZingH3Sampler(resident, packer, **options)
    plan = video_plan(
        solver,
        baseline.video_scheduler.timesteps,
        0,
        0.999,
        sigmas=baseline.video_scheduler.sigmas,
    )
    path = tmp_path / "adaln"
    write_table(
        path,
        arch=arch,
        tags=resident.modality_tags,
        embedder=resident.time_embedder,
        projections=[
            *((i, b.adaln_proj) for i, b in enumerate(resident.blocks)),
            ("final", resident.final_layer.adaln_proj),
        ],
        plan=plan,
        identity={"synthetic": True},
        provenance={"fixture": True},
    )
    table = ZingAdalnTable(
        path, arch=arch, tags=resident.modality_tags, max_bytes=1 << 20
    )
    verify_synthetic_table(table, tmp_path)
    # Meta construction proves omitted parameters are absent before loading.
    with torch.device("meta"):
        candidate = ZingH3DiTModel(
            ZingH3DiTConfig(arch_config=replace(arch)), {}, adaln_table=table
        )
    assert not any("adaln_proj" in name for name, _ in candidate.named_parameters())
    assert all(block.adaln_proj is None for block in candidate.blocks)
    assert candidate.final_layer.adaln_proj is None and candidate.adaln_cache is None
    candidate.to_empty(device="cpu").eval()
    candidate.load_state_dict(
        {k: v for k, v in resident.state_dict().items() if "adaln_proj" not in k}
    )
    sampler = ZingH3Sampler(candidate, packer, **options)
    states = [s.initialize(inputs, noise) for s in (baseline, sampler)]
    assert table.active_plans == 1
    try:
        for i in range(len(states[0].spans)):
            # Pair identical CD re-noise RNG draws in both modes.
            torch.manual_seed(100 + i)
            baseline.step(states[0])
            torch.manual_seed(100 + i)
            sampler.step(states[1])
            assert torch.equal(states[0].visual, states[1].visual)
            assert torch.isfinite(states[1].visual).all()
            for layer in range(arch.num_layers):
                for a, b in zip(
                    states[0].cache.history(layer), states[1].cache.history(layer)
                ):
                    assert torch.equal(a, b)
        sampler.update_condition(states[1], inputs)
        assert table.active_plans == 1
    finally:
        for state in states:
            state.close()
    assert table.active_plans == 0 and table.hits > 0
    with pytest.raises(ValueError, match="offline"):
        candidate.refresh_weight_derived_caches(weights_path=None)
    with pytest.raises(ValueError, match="offline"):
        candidate.prepare_lora_adapter({})
    for source in [None, "unused-checkpoint-path"]:
        with pytest.raises(ValueError, match="offline"):
            candidate.validate_weight_update_source(weights_path=source)
    # A missing plan must reject before pool acquisition or prefix work.
    bad = ZingH3Sampler(
        candidate,
        packer,
        sigma_points=4,
        ar_condition_timestep=0,
        reference_visual_timestep=0.999,
    )

    def forbidden_acquire(_):
        pytest.fail("incompatible AdaLN plan acquired KV storage")

    with pytest.raises(ValueError, match="schedule|timestep batch"):
        bad.initialize(inputs, noise, acquire_storage=forbidden_acquire)
    assert table.active_plans == 0

    def failed_acquire(_):
        raise RuntimeError("synthetic KV pool exhaustion")

    with pytest.raises(RuntimeError, match="pool exhaustion"):
        sampler.initialize(inputs, noise, acquire_storage=failed_acquire)
    assert table.active_plans == 0


@torch.no_grad()
def test_offline_adaln_builder_streams_exported_projection_keys(
    model, tmp_path, monkeypatch
):
    import importlib.util
    import json
    from dataclasses import asdict
    from pathlib import Path

    from safetensors.torch import save_file

    from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import (
        ZingAdalnTable,
        video_plan,
    )

    spec = importlib.util.spec_from_file_location(
        "zing_builder",
        Path(__file__).resolve().parents[5]
        / "benchmark/zing_h3_causal/build_adaln_table.py",
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    import weakref

    observed = []

    class AuditedProjection(builder.MiniMaxH3AdalnProj):
        def __init__(self, *args, **kwargs):
            assert all(value() is None for value in observed), (
                "previous layer retained projection storage"
            )
            super().__init__(*args, **kwargs)

        def forward(self, embedding):
            observed.extend(weakref.ref(p) for p in self.parameters())
            return super().forward(embedding)

    monkeypatch.setattr(builder, "MiniMaxH3AdalnProj", AuditedProjection)
    root = tmp_path / "export"
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text(
        json.dumps(asdict(model.arch) | {"_class_name": "ZingH3DiTModel"})
    )
    (root / "zing_config.json").write_text(
        json.dumps(
            dict(
                checkpoint_sha256="0" * 64,
                provenance=dict(
                    native_sha="0" * 40,
                    checkpoint_source="generator",
                    component_versions={"fixture": "cpu"},
                ),
            )
        )
    )
    weights = {}
    for i, target in [(1, "in"), (2, "out")]:
        for part in ["weight", "bias"]:
            weights[f"backbone.time_embedder.linear_{i}.{part}"] = model.state_dict()[
                f"time_embedder.proj_{target}.{part}"
            ]
    for i, block in enumerate(model.blocks):
        for part in ["weight", "bias"]:
            weights[f"backbone.transformer_blocks.{i}.adaln_proj.linear.{part}"] = (
                block.adaln_proj.state_dict()[f"linear.{part}"]
            )
    for part in ["weight", "bias"]:
        weights[f"backbone.norm_out.linear.{part}"] = (
            model.final_layer.adaln_proj.state_dict()[f"linear.{part}"]
        )
    save_file(weights, str(transformer / "weights.safetensors"))
    plan = video_plan(
        "euler", torch.tensor([0, 0.5]), sigmas=torch.tensor([1.0, 0.5, 0.0])
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    builder.build(root, plan_path, tmp_path / "table", "cpu")
    assert observed and all(value() is None for value in observed)
    assert not (tmp_path / "bad-table").exists()
    wrong_plan = dict(plan, schedule=dict(plan["schedule"], condition=0.1))
    (tmp_path / "wrong-plan.json").write_text(json.dumps(wrong_plan))
    with pytest.raises(ValueError, match="prefix timesteps"):
        builder.build(root, tmp_path / "wrong-plan.json", tmp_path / "bad-table", "cpu")
    assert not (tmp_path / "bad-table").exists()
    table = ZingAdalnTable(
        tmp_path / "table", arch=model.arch, tags=model.modality_tags, max_bytes=1 << 20
    )
    table.validate_checkpoint(
        [transformer / "weights.safetensors"], transformer / "config.json"
    )
    lease = table.prepare(
        [torch.tensor([0.0])],
        device="cpu",
        schedule=plan["schedule"],
        embedder=model.time_embedder,
    )
    t = torch.tensor([0.0])
    embedding = torch.nn.functional.silu(model.time_embedder(t)).bfloat16()
    for i, block in enumerate(model.blocks):
        for actual, expected in zip(
            table.project(i, None, embedding, table.lookup(t)),
            block.adaln_proj(embedding),
        ):
            assert torch.equal(actual, expected)
    lease.close()


@torch.no_grad()
def test_weights_free_reader_never_materializes_projection_tensors(
    model, tmp_path, monkeypatch
):
    from safetensors.torch import save_file

    from sglang.multimodal_gen.runtime.loader.weight_readers import safetensors_mmap
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import (
        keep_zing_weight,
    )

    path = tmp_path / "weights.safetensors"
    retained = {
        "backbone.time_embedder.linear_1.weight": torch.randn(8, 8),
        "backbone.norm_out.norm.weight": torch.randn(16),
    }
    excluded = {
        "backbone.transformer_blocks.0.adaln_proj.linear.weight": torch.randn(384, 8),
        "backbone.norm_out.linear.bias": torch.randn(32),
    }
    save_file(retained | excluded, str(path))
    real_open = safetensors_mmap.safe_open
    read_keys = []

    class AuditedFile:
        def __init__(self, *args, **kwargs):
            self.handle = real_open(*args, **kwargs)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def keys(self):
            return self.handle.keys()

        def get_tensor(self, key):
            read_keys.append(key)
            return self.handle.get_tensor(key)

    monkeypatch.setattr(safetensors_mmap, "safe_open", AuditedFile)
    loaded = dict(
        safetensors_mmap.SafetensorsMmapReader().iter_weights(
            [str(path)],
            device="cpu",
            to_cpu=True,
            key_filter=keep_zing_weight,
            show_progress=False,
        )
    )
    assert set(read_keys) == set(loaded) == set(retained)
    assert not set(read_keys) & set(excluded)
    for name in retained:
        assert torch.equal(loaded[name], retained[name])
