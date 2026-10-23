"""CPU routing/loading tests; these do not establish CUDA numerical parity."""

import itertools
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

import sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl as h3
import sglang.multimodal_gen.runtime.models.encoders.qwen3vl as qwen
import sglang.multimodal_gen.runtime.models.encoders.qwen3vl_ops as ops
from sglang.multimodal_gen.configs.models.encoders.qwen3vl import (
    Qwen3VLExecutionOptions,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.srt.runtime_context import get_parallel


def tiny_config():
    text = Qwen3VLTextConfig(
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        vocab_size=32,
        pad_token_id=0,
        max_position_embeddings=128,
        rope_scaling={"rope_type": "default", "mrope_section": [4, 6, 6]},
    )
    vision = SimpleNamespace(
        hidden_size=144,
        intermediate_size=192,
        hidden_act="gelu_pytorch_tanh",
        num_heads=2,
        depth=1,
        patch_size=2,
        temporal_patch_size=1,
        in_channels=3,
        num_position_embeddings=16,
        spatial_merge_size=2,
        out_hidden_size=64,
        deepstack_visual_indexes=[0],
    )
    return SimpleNamespace(
        text_config=text,
        vision_config=vision,
        num_hidden_layers=1,
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=28,
        conditioning_projection_path=None,
        _fsdp_shard_conditions=[],
        stacked_params_mapping=[],
    )


@pytest.fixture(autouse=True)
def baseline_options(monkeypatch):
    monkeypatch.setenv("SGLANG_H3_QWEN_FP8", "0")
    for field in fields(Qwen3VLExecutionOptions):
        monkeypatch.setenv(f"SGLANG_H3_QWEN_{field.name.upper()}", "0")


@pytest.fixture
def fa4_available(monkeypatch):
    import sglang.kernels.ops.attention.flash_attention_v4 as fa4

    monkeypatch.setattr(fa4, "is_flash_attention_v4_available", lambda: True)


def make_h3(quant_config=None):
    with get_parallel().override(tp_size=1, tp_rank=0):
        return h3.MiniMaxH3Qwen3VLEncoder(
            SimpleNamespace(arch_config=tiny_config(), quant_config=quant_config)
        )


@pytest.mark.parametrize("values", itertools.product((False, True), repeat=5))
@pytest.mark.parametrize("fp8", [False, True])
def test_independent_h3_construction(values, fp8, monkeypatch, fa4_available):
    for field, enabled in zip(fields(Qwen3VLExecutionOptions), values):
        monkeypatch.setenv(f"SGLANG_H3_QWEN_{field.name.upper()}", str(int(enabled)))
    quant_config = None
    if fp8:
        from sglang.multimodal_gen.runtime.layers.quantization import fp8 as native_fp8

        monkeypatch.setattr(
            native_fp8, "get_tensor_model_parallel_world_size", lambda: 1
        )
        quant_config = native_fp8.Fp8Config()
        monkeypatch.setenv("SGLANG_H3_QWEN_FP8", "1")
    encoder = make_h3(quant_config)
    assert encoder.execution_options == Qwen3VLExecutionOptions(*values)
    layer = encoder.model.language_model.layers[0]
    assert isinstance(layer.input_layernorm, ops.Qwen3VLHFRMSNorm) == values[0]
    assert isinstance(layer.self_attn.q_norm, ops.Qwen3VLHFRMSNorm) == values[0]
    assert (layer.self_attn.fused_rope is not None) == values[1]
    assert hasattr(layer.mlp, "gate_up_proj") == values[2]
    assert isinstance(layer.self_attn.attn, ops.Qwen3VLFA4Attention) == values[3]
    if not values[3]:
        assert layer.self_attn.attn.backend == AttentionBackendEnum.TORCH_SDPA
    visual = encoder.model.visual.blocks[0].attn
    assert visual.head_dim == 72
    assert isinstance(visual._attention_impl, ops.Qwen3VLFA4Attention) == values[4]
    assert isinstance(encoder.model.language_model.norm, nn.Identity)
    if fp8:
        quantized = [
            module
            for module in encoder.model.language_model.modules()
            if isinstance(
                getattr(module, "quant_method", None), native_fp8.Fp8LinearMethod
            )
        ]
        assert len(quantized) == (6 if values[2] else 7)
        assert all(
            not getattr(getattr(module, "quant_method", None), "quant_config", None)
            for module in encoder.model.visual.modules()
        )
    # Environment changes after construction do not reconfigure this instance.
    monkeypatch.setenv("SGLANG_H3_QWEN_FUSED_GATE_UP", str(int(not values[2])))
    assert layer.mlp.fused_gate_up == values[2]


@pytest.mark.parametrize("invalid", ["", "true", "2", " 1"])
def test_invalid_option_rejected(invalid, monkeypatch):
    monkeypatch.setenv("SGLANG_H3_QWEN_TEXT_FA4", invalid)
    with pytest.raises(ValueError, match="SGLANG_H3_QWEN_TEXT_FA4"):
        h3._execution_options_from_env()


def test_default_and_generic_instance_isolation(monkeypatch, fa4_available):
    from sglang.multimodal_gen.runtime.layers.quantization import fp8 as native_fp8

    for field in fields(Qwen3VLExecutionOptions):
        monkeypatch.delenv(f"SGLANG_H3_QWEN_{field.name.upper()}")
    monkeypatch.delenv("SGLANG_H3_QWEN_FP8")
    monkeypatch.setattr(native_fp8, "get_tensor_model_parallel_world_size", lambda: 1)
    assert h3._execution_options_from_env() == Qwen3VLExecutionOptions(
        True, True, True, True, True
    )
    enabled = make_h3(native_fp8.Fp8Config())
    layer = enabled.model.language_model.layers[0]
    assert isinstance(layer.input_layernorm, ops.Qwen3VLHFRMSNorm)
    assert layer.self_attn.fused_rope is not None
    assert hasattr(layer.mlp, "gate_up_proj")
    assert isinstance(layer.mlp.gate_up_proj.quant_method, native_fp8.Fp8LinearMethod)
    assert isinstance(
        enabled.model.visual.blocks[0].attn._attention_impl, ops.Qwen3VLFA4Attention
    )
    with get_parallel().override(tp_size=1, tp_rank=0):
        generic = qwen.Qwen3VLModel(tiny_config())
    assert isinstance(
        enabled.model.language_model.layers[0].self_attn.attn, ops.Qwen3VLFA4Attention
    )
    assert not isinstance(
        generic.language_model.layers[0].self_attn.attn, ops.Qwen3VLFA4Attention
    )
    assert not isinstance(
        generic.language_model.layers[0].input_layernorm, ops.Qwen3VLHFRMSNorm
    )


def test_packed_checkpoint_loading_and_mlp_output(monkeypatch):
    native = make_h3()
    monkeypatch.setenv("SGLANG_H3_QWEN_FUSED_GATE_UP", "1")
    packed = make_h3()
    weights = [
        (name, value.detach().clone()) for name, value in native.named_parameters()
    ]
    # Reversed order also exercises up-before-gate loading with mmap retention.
    packed._keep_checkpoint_mapping = True
    loaded = packed.load_weights(reversed(weights))
    assert loaded == set(dict(packed.named_parameters()))
    a = native.model.language_model.layers[0].mlp
    b = packed.model.language_model.layers[0].mlp
    torch.testing.assert_close(
        b.gate_up_proj.weight, torch.cat((a.gate_proj.weight, a.up_proj.weight))
    )
    assert sum(p.numel() for p in a.parameters()) == sum(
        p.numel() for p in b.parameters()
    )
    # A CPU oracle checks packing/order; the CUDA rounded kernel is tested separately.
    monkeypatch.setattr(
        qwen,
        "rounded_swiglu",
        lambda x: nn.functional.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1],
    )
    x = torch.randn(2, 7, 64)
    torch.testing.assert_close(a(x), b(x))
    missing_up = [(n, w) for n, w in weights if not n.endswith("mlp.up_proj.weight")]
    with pytest.raises(ValueError, match="Incomplete packed"):
        packed.load_weights(missing_up)


def test_all_off_keeps_native_forward_and_checkpoint_names(monkeypatch):
    # SRT RowParallelLinear enters its allocation context even at TP=1.
    # No collective is used by this test; no distributed runtime is required.
    monkeypatch.setattr("sglang.srt.layers.linear.get_tp_group", lambda: None)
    with monkeypatch.context() as baseline:
        baseline.setattr(h3, "_execution_options_from_env", Qwen3VLExecutionOptions)
        native = make_h3().eval()
    # SRT projections allocate empty storage for checkpoint loading. Populate
    # every parameter instead of accidentally comparing uninitialized memory.
    generator = torch.Generator().manual_seed(29)
    with torch.no_grad():
        for parameter in native.parameters():
            parameter.uniform_(-0.1, 0.1, generator=generator)
    for field in fields(Qwen3VLExecutionOptions):
        monkeypatch.setenv(f"SGLANG_H3_QWEN_{field.name.upper()}", "0")
    disabled = make_h3().eval()
    disabled.load_state_dict(native.state_dict(), strict=True)
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        a = native(ids).last_hidden_state
        b = disabled(ids).last_hidden_state
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    pixels = torch.randn(16, 3 * 1 * 2 * 2, generator=generator)
    grid = torch.tensor([[1, 4, 4]])
    with torch.no_grad():
        a = native.model.visual(pixels, grid)
        b = disabled.model.visual(pixels, grid)
    torch.testing.assert_close(a.pooler_output, b.pooler_output, rtol=0, atol=0)
    assert len(a.deepstack_features) == len(b.deepstack_features) == 1
    torch.testing.assert_close(
        a.deepstack_features[0], b.deepstack_features[0], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quant_config": SimpleNamespace(get_name=lambda: "other")},
        {"use_weight_only_fp8": True},
        {"use_tensor_parallel": True},
    ],
)
def test_packing_rejects_unsupported_layout(kwargs, monkeypatch):
    monkeypatch.setattr(qwen, "_tp_world_size", lambda: 2)
    with pytest.raises(ValueError, match="encoder TP=1"):
        qwen.Qwen3VLTextMLP(
            tiny_config().text_config,
            execution_options=Qwen3VLExecutionOptions(fused_gate_up=True),
            **kwargs,
        )


def test_fa4_dense_gqa_and_packed_72_metadata(fa4_available, monkeypatch):
    calls = []
    monkeypatch.setattr(ops, "_require_cuda_half", lambda *_: None)

    def record(q, k, v, **kwargs):
        calls.append((q.shape, k.shape, kwargs))
        return q

    language = ops.Qwen3VLFA4Attention(softmax_scale=0.125, causal=True)
    language.kernel = record
    q, k = torch.zeros(2, 7, 4, 64), torch.zeros(2, 7, 2, 64)
    assert language(q, k, k).shape == q.shape
    assert calls[-1][2]["causal"] is True
    assert calls[-1][0][-2] == 2 * calls[-1][1][-2]
    vision = ops.Qwen3VLFA4Attention(softmax_scale=72**-0.5, causal=False)
    vision.kernel = record
    q = torch.zeros(12, 2, 72)
    cu = torch.tensor([0, 4, 12], dtype=torch.int32)
    vision.forward_varlen(
        q, q, q, cu_seqlens=cu, cu_seqlens_host=(0, 4, 12), max_seqlen=8
    )
    args = calls[-1][2]
    assert args["cu_seqlens_q"] is cu and args["cu_seqlens_k"] is cu
    assert args["max_seqlen_q"] == args["max_seqlen_k"] == 8
    assert args["causal"] is False and args["num_splits"] == 1


def test_fa4_unavailable_does_not_fall_back(monkeypatch):
    import sglang.kernels.ops.attention.flash_attention_v4 as fa4

    monkeypatch.setattr(fa4, "is_flash_attention_v4_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        ops.Qwen3VLFA4Attention(softmax_scale=1.0, causal=True)


@pytest.mark.parametrize(
    "text_fa4,vision_fa4", itertools.product((False, True), repeat=2)
)
@pytest.mark.parametrize(
    "requested", [None, AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA]
)
def test_attention_switches_obey_component_contract(
    text_fa4, vision_fa4, requested, monkeypatch, fa4_available
):
    from contextlib import nullcontext

    from sglang.multimodal_gen.runtime.layers.attention.selector import (
        ComponentAttentionBackendNotAppliedError,
        component_attn_backend_context_manager,
        get_component_attn_backend_context,
    )

    monkeypatch.setenv("SGLANG_H3_QWEN_TEXT_FA4", str(int(text_fa4)))
    monkeypatch.setenv("SGLANG_H3_QWEN_VISION_FA4", str(int(vision_fa4)))
    expected = {"fa" if enabled else "torch_sdpa" for enabled in (text_fa4, vision_fa4)}
    conflict = requested is not None and expected != {requested.name.lower()}
    with (
        pytest.raises(ComponentAttentionBackendNotAppliedError)
        if conflict
        else nullcontext()
    ):
        with component_attn_backend_context_manager(
            requested, "text_encoder", require_backend_selection=requested is not None
        ):
            make_h3()
            assert (
                set(get_component_attn_backend_context().selected_backends) == expected
            )


def test_fused_rope_cache_shared_within_encoder_only(monkeypatch):
    monkeypatch.setattr("sglang.srt.layers.rotary_embedding.factory._ROPE_DICT", {})
    config = tiny_config().text_config
    config.num_hidden_layers = 2
    options = Qwen3VLExecutionOptions(fused_mrope=True, text_fa4=False)
    first = qwen.Qwen3VLTextModel(config, execution_options=options)
    second = qwen.Qwen3VLTextModel(config, execution_options=options)
    layers = [layer.self_attn for layer in first.layers]
    assert layers[0].rotary_emb is layers[1].rotary_emb
    assert layers[0].fused_rope is layers[1].fused_rope
    assert layers[0].fused_rope is not second.layers[0].self_attn.fused_rope
    assert not any("cos_sin_cache" in name for name in first.state_dict())


@pytest.mark.parametrize(
    "flag,explicit,expected",
    [
        (None, None, "fp8"),
        (None, "fp8", "fp8"),
        ("0", None, None),
        ("1", None, "fp8"),
        ("1", "fp8", "fp8"),
        ("0", "kitchen_int8", "kitchen_int8"),
    ],
)
def test_fp8_switch_resolves_native_online_config(
    flag, explicit, expected, monkeypatch
):
    from sglang.multimodal_gen.configs.models.encoders.minimax_h3_qwen3vl import (
        resolve_h3_qwen_online_quantization,
    )

    if flag is not None:
        monkeypatch.setenv("SGLANG_H3_QWEN_FP8", flag)
    else:
        monkeypatch.delenv("SGLANG_H3_QWEN_FP8")
    assert resolve_h3_qwen_online_quantization(explicit) == expected


@pytest.mark.parametrize(
    "flag,explicit", [("bad", None), ("1", "mxfp4"), ("0", "fp8"), (None, "mxfp4")]
)
def test_fp8_switch_rejects_conflicts(flag, explicit, monkeypatch):
    from sglang.multimodal_gen.configs.models.encoders.minimax_h3_qwen3vl import (
        resolve_h3_qwen_online_quantization,
    )

    if flag is None:
        monkeypatch.delenv("SGLANG_H3_QWEN_FP8")
    else:
        monkeypatch.setenv("SGLANG_H3_QWEN_FP8", flag)
    with pytest.raises(ValueError, match="SGLANG_H3_QWEN_FP8"):
        resolve_h3_qwen_online_quantization(explicit)


@pytest.mark.parametrize("flag", [None, "1"])
def test_fp8_switch_enters_shared_loader_only_for_h3(monkeypatch, flag):
    import sglang.multimodal_gen.runtime.layers.quantization as quantization
    from sglang.multimodal_gen.configs.models.encoders.minimax_h3_qwen3vl import (
        MiniMaxH3Qwen3VLConfig,
    )
    from sglang.multimodal_gen.runtime.loader.component_loaders import (
        text_encoder_loader as loader,
    )

    class ConfigRecorder:
        def __init__(self, ignored_layers):
            self.ignored_layers = ignored_layers

    monkeypatch.setattr(loader, "_get_encoder_quant_config", lambda *args: None)

    def resolve(name):
        assert name == "fp8"
        return ConfigRecorder

    monkeypatch.setattr(quantization, "get_quantization_config", resolve)
    if flag is None:
        monkeypatch.delenv("SGLANG_H3_QWEN_FP8")
    else:
        monkeypatch.setenv("SGLANG_H3_QWEN_FP8", flag)
    config = MiniMaxH3Qwen3VLConfig()
    loader._configure_encoder_quantization(
        config, h3.MiniMaxH3Qwen3VLEncoder, {}, "/unused", "/unused", "text_encoder"
    )
    assert isinstance(config.quant_config, ConfigRecorder)
    unrelated = SimpleNamespace(quant_config=None)
    loader._configure_encoder_quantization(
        unrelated, h3.MiniMaxH3Qwen3VLEncoder, {}, "/unused", "/unused", "text_encoder"
    )
    assert unrelated.quant_config is None
    monkeypatch.setenv("SGLANG_H3_QWEN_FP8", "0")
    monkeypatch.setattr(
        loader,
        "_get_encoder_quant_config",
        lambda *args: SimpleNamespace(get_name=lambda: "fp8"),
    )
    with pytest.raises(
        loader.ComponentCheckpointUnsupportedError, match="non-FP8 checkpoint"
    ):
        loader._configure_encoder_quantization(
            config, h3.MiniMaxH3Qwen3VLEncoder, {}, "/unused", "/unused", "text_encoder"
        )


def test_packed_projection_uses_native_fp8_linear(monkeypatch):
    # Keep the real LinearBase/parameter loader; substitute only the optional
    # hardware-specific quant method. This verifies construction, not FP8 math.
    from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod

    config = SimpleNamespace(
        get_name=lambda: "fp8",
        is_checkpoint_fp8_serialized=False,
        activation_scheme="dynamic",
        weight_block_size=None,
        ignored_layers=[],
        get_quant_method=lambda layer, prefix: UnquantizedLinearMethod(),
    )
    mlp = qwen.Qwen3VLTextMLP(
        tiny_config().text_config,
        quant_config=config,
        execution_options=Qwen3VLExecutionOptions(fused_gate_up=True),
    )
    assert isinstance(mlp.gate_up_proj, qwen.Qwen3VLQuantizedLinear)
    assert mlp.gate_up_proj.quant_config is config
    assert mlp.gate_up_proj.weight.shape == (192, 64)


def test_h3_defaults_and_explicit_switches_prevent_transformers_fallback(monkeypatch):
    from sglang.multimodal_gen.configs.models.encoders.minimax_h3_qwen3vl import (
        MiniMaxH3Qwen3VLConfig,
    )
    from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
        ComponentLoader,
    )
    from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
        TextEncoderLoader,
    )

    monkeypatch.setattr(
        ComponentLoader, "should_raise_customized_load_error", lambda *args: False
    )
    args = SimpleNamespace(
        component_quantizations={},
        pipeline_config=SimpleNamespace(
            text_encoder_configs=[MiniMaxH3Qwen3VLConfig()]
        ),
    )
    loader = TextEncoderLoader()
    assert loader.should_raise_customized_load_error(args, "text_encoder")
    for field in fields(Qwen3VLExecutionOptions):
        monkeypatch.delenv(f"SGLANG_H3_QWEN_{field.name.upper()}")
    monkeypatch.delenv("SGLANG_H3_QWEN_FP8")
    assert loader.should_raise_customized_load_error(args, "text_encoder")
    for flag in ("SGLANG_H3_QWEN_TEXT_FA4", "SGLANG_H3_QWEN_FP8"):
        monkeypatch.setenv(flag, "1")
        assert loader.should_raise_customized_load_error(args, "text_encoder")
        monkeypatch.delenv(flag)
    monkeypatch.setenv("SGLANG_H3_QWEN_FP8", "1")
    args.pipeline_config.text_encoder_configs = [SimpleNamespace()]
    assert not loader.should_raise_customized_load_error(args, "text_encoder")


def test_default_fp8_requires_native_loader(monkeypatch):
    monkeypatch.delenv("SGLANG_H3_QWEN_FP8")
    with pytest.raises(ValueError, match="native component loader"):
        make_h3()


@pytest.mark.parametrize("flag", [None, "1"])
def test_default_and_explicit_fp8_reject_marlin(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("SGLANG_H3_QWEN_FP8")
    else:
        monkeypatch.setenv("SGLANG_H3_QWEN_FP8", flag)
    model = nn.Module()
    model.language_model = nn.Sequential(nn.Linear(2, 2))
    model.language_model[0].quant_method = SimpleNamespace(use_marlin=True)
    monkeypatch.setattr(h3, "Qwen3VLModel", lambda *args, **kwargs: model)
    with pytest.raises(ValueError, match="Marlin weight-only"):
        make_h3(SimpleNamespace(get_name=lambda: "fp8"))
