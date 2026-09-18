# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
    ZingH3DiTArchConfig,
    ZingH3DiTConfig,
)
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
        for k, v in chunk.layers:
            assert v.untyped_storage().nbytes() == v.numel() * v.element_size()
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
def test_generation_stage_connects_prompt_dit_and_decoder(model):
    from contextlib import nullcontext

    from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams
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
        prepare=prepare, decode_visual=decode, spatial_ratio=16, target_fps=24.0
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


@torch.no_grad()
def test_export_and_native_loader_keep_all_tensors_and_precision(model, tmp_path):
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
        vae_config=dict(encode_mode="deterministic"),
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
