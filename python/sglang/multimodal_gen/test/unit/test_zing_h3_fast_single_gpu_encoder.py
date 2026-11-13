# SPDX-License-Identifier: Apache-2.0
"""Contracts, fallbacks, transactional safety, and CUDA parity for fast single-GPU H3 encoding."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.conditioning.cache import ConditioningCache
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.fast_encoder import (
    _build_tile_geometry_plan,
    encoder_numerical_optimizations_enabled,
    is_fast_single_gpu_model_supported,
    prepare_fast_single_gpu_encoder,
    triton_available,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae import (
    AutoencoderKL,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.norm import (
    FusedGroupNorm3D,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.progressive import (
    CausalEncoderCache,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_cnn import (
    EncoderFCN3D,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    codecs as codecs_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
    ZingH3Codecs,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
    create_session,
)

_CUDA_TRITON = torch.cuda.is_available() and triton_available()


class _TiledH3Fixture(nn.Module):
    split_tiles = AutoencoderKL.split_tiles
    _assemble_tiles = AutoencoderKL._assemble_tiles
    blend = AutoencoderKL.blend
    _local_tile_indices = staticmethod(AutoencoderKL._local_tile_indices)

    def __init__(
        self,
        *,
        ch: int = 32,
        ch_mult=(1, 2, 2, 2, 4, 4),
        space_down=(2, 2, 2, 2, 1, 1),
        time_down=(1, 2, 2, 1, 1, 1),
        num_res_blocks: int = 1,
        tile_size: int = 64,
        tile_overlap_min: int = 16,
        padding_mode: str = "reflect",
        causal: bool = True,
        use_t_isolated_gn: bool = True,
    ):
        super().__init__()
        self.encoder = EncoderFCN3D(
            ch=ch,
            ch_mult=list(ch_mult),
            space_down=list(space_down),
            time_down=list(time_down),
            num_res_blocks=num_res_blocks,
            in_channels=3,
            z_channels=24,
            double_z=True,
            padding_mode=padding_mode,
            causal=causal,
            use_t_isolated_gn=use_t_isolated_gn,
        ).eval()
        self.quant_conv = nn.Conv3d(48, 48, kernel_size=1).eval()
        self.encoder_tiling = True
        self.stack_tiling = False
        self.tile_size = tile_size
        self.tile_overlap_min = tile_overlap_min
        self.decoder_tile_size = tile_size
        self.decoder_tile_overlap_min = tile_overlap_min
        self.vae_ratio = 16
        self.vae_ratio_t = 4
        self._blend_weight_cache = {}

    def encode(self, x):
        return self.quant_conv(self.encoder(x))

    def tiled_encode(self, x):
        yi, yl, yo = self.split_tiles(x.shape[-2], False)
        xi, xl, xo = self.split_tiles(x.shape[-1], False)
        rows = [
            [self.encode(x[..., y : y + h, z : z + w]) for z, w in zip(xi, xl)]
            for y, h in zip(yi, yl)
        ]
        return self._assemble_tiles(
            rows,
            [v // self.vae_ratio for v in yo],
            [v // self.vae_ratio for v in xo],
        )

    def _adaptive_encode(self, x):
        return self.tiled_encode(x) if self.encoder_tiling else self.encode(x)


def _make_codecs_fixture(
    model: _TiledH3Fixture,
    *,
    encode_dtype: str = "bfloat16",
    encode_mode: str = "performance",
):
    codecs = object.__new__(ZingH3Codecs)
    codecs.modules = {"video_vae": model}
    codecs.vae_frames_per_chunk = 17
    codecs.vae_latents_per_chunk = 5
    codecs.temporal_ratio = 4
    codecs.spatial_ratio = 16
    codecs.va = SimpleNamespace(
        latents_mean=[0.1 * (i % 5) for i in range(24)],
        latents_std=[1.0 + 0.05 * (i % 4) for i in range(24)],
    )
    codecs.runtime = {
        "vae_config": {
            "encode_dtype": encode_dtype,
            "encode_mode": encode_mode,
            "encode_tile_batch_size": 2,
            "posterior_seed": 1234,
            "cudnn_benchmark_limit": (
                1 if torch.backends.cudnn.benchmark_limit is not None else None
            ),
        }
    }
    codecs.reference_cache = ConditioningCache(4 << 20)
    return codecs


@pytest.mark.parametrize("value, expected", [(None, True), ("1", True), ("0", False)])
def test_numerical_optimization_policy(monkeypatch, value, expected):
    name = "MINIMAX_H3_ENCODER_NUMERICAL_OPTIMIZATIONS"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    assert encoder_numerical_optimizations_enabled() is expected
    cache = CausalEncoderCache(
        _TiledH3Fixture(), tile_mode="replicated", encode_mode="performance"
    )
    assert cache.numerical_optimizations is expected
    assert cache.fused_norm_input is expected
    assert cache.fused_conv_input  # Copy/history fusion remains independently enabled.
    # Policy is frozen for a session, just like existing execution choices.
    monkeypatch.setenv(name, "0" if expected else "1")
    assert cache.numerical_optimizations is expected
    assert cache.fused_norm_input is expected


@pytest.mark.parametrize("value", ["true", "", "2"])
def test_numerical_optimization_policy_rejects_invalid_value(monkeypatch, value):
    monkeypatch.setenv("MINIMAX_H3_ENCODER_NUMERICAL_OPTIMIZATIONS", value)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        encoder_numerical_optimizations_enabled()


def test_numerical_optimization_off_blocks_preparation_and_visual_fast(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_ENCODER_NUMERICAL_OPTIMIZATIONS", "0")
    monkeypatch.setenv("MINIMAX_H3_ENCODER_SINGLE_GPU_FAST", "1")
    monkeypatch.setenv("MINIMAX_H3_FAST_VISUAL_ENCODE", "1")
    # Guards run before model/device access, including previously prepared models.
    assert prepare_fast_single_gpu_encoder(object()) is None
    assert not ZingH3Codecs._can_fast_visual_encode_profile(object(), object())


@pytest.mark.skipif(not _CUDA_TRITON, reason="Requires CUDA and Triton")
def test_cuda_numerical_optimization_off_matches_legacy(monkeypatch):
    torch.manual_seed(83)
    model = _TiledH3Fixture().to(device="cuda", dtype=torch.bfloat16)
    codecs = _make_codecs_fixture(model)
    monkeypatch.setenv("MINIMAX_H3_ENCODER_NUMERICAL_OPTIMIZATIONS", "1")
    codecs.prepare_visual_encoder_weights()
    assert model._fast_single_gpu_encoder is not None
    frames = torch.randint(0, 256, (3, 9, 64, 64), dtype=torch.uint8)

    def run():
        session = create_session(codecs, capacity_frames=17)
        try:
            session.append(frames, frame_start=0)
            outputs = [session.pop().values.clone(), session.pop().values.clone()]
        finally:
            session.dispose()
        assert session.state_usage().retained_bytes == 0
        return outputs

    # Explicit pre-691 arithmetic: old GN fusion off, original convolution path.
    monkeypatch.setenv("MINIMAX_H3_ENCODER_SINGLE_GPU_FAST", "0")
    monkeypatch.setenv("MINIMAX_H3_PROGRESSIVE_FUSED_NORM_INPUT", "0")
    baseline = run()
    monkeypatch.setenv("MINIMAX_H3_ENCODER_SINGLE_GPU_FAST", "1")
    monkeypatch.setenv("MINIMAX_H3_PROGRESSIVE_FUSED_NORM_INPUT", "1")
    monkeypatch.setenv("MINIMAX_H3_ENCODER_NUMERICAL_OPTIMIZATIONS", "0")
    candidate = run()
    assert all(torch.equal(a, b) for a, b in zip(baseline, candidate))
    assert not codecs._can_fast_visual_encode_profile(frames)


def test_fast_encoder_model_admission_and_rejections():
    torch.manual_seed(11)
    valid = _TiledH3Fixture()
    assert is_fast_single_gpu_model_supported(valid)

    fused_gn = _TiledH3Fixture()
    fused_gn.encoder.norm_out = FusedGroupNorm3D(
        32, 128, affine=True, use_t_isolated_gn=True
    )
    assert is_fast_single_gpu_model_supported(fused_gn)

    for activation in ("relu", "identity"):
        fused_gn.encoder.norm_out.activation = activation
        assert not is_fast_single_gpu_model_supported(fused_gn)

    non_causal = _TiledH3Fixture(causal=False)
    assert not is_fast_single_gpu_model_supported(non_causal)

    non_isolated_gn = _TiledH3Fixture(use_t_isolated_gn=False)
    assert not is_fast_single_gpu_model_supported(non_isolated_gn)

    bad_quant = _TiledH3Fixture()
    bad_quant.quant_conv = nn.Conv3d(48, 48, kernel_size=3, padding=1)
    assert not is_fast_single_gpu_model_supported(bad_quant)

    # CPU models never allocate CUDA fast encoder assets
    assert prepare_fast_single_gpu_encoder(valid) is None


@pytest.mark.parametrize(
    "height,width,tile_size,overlap,expected_tiles",
    [
        (480, 832, 256, 64, 12),
        (736, 1280, 256, 64, 28),
        (736, 1312, 256, 64, 28),
        (256, 256, 256, 64, 1),
        (112, 160, 64, 16, 6),
    ],
)
def test_tile_geometry_plan_and_blend_lut_exactness(
    height, width, tile_size, overlap, expected_tiles
):
    torch.manual_seed(29)
    model = _TiledH3Fixture(tile_size=tile_size, tile_overlap_min=overlap)
    plan = _build_tile_geometry_plan(
        model, height, width, device=torch.device("cpu"), dtype=torch.bfloat16
    )
    assert plan.num_tiles == expected_tiles
    assert plan.tile_h == min(height, tile_size)
    assert plan.tile_w == min(width, tile_size)

    n_rows, n_cols = len(plan.ys), len(plan.xs)
    th, tw = plan.blend_lut.tile_h, plan.blend_lut.tile_w
    t_lat = 3
    c_lat = 48
    tiles_ncdhw = torch.randn(
        (expected_tiles, c_lat, t_lat, th, tw), dtype=torch.bfloat16
    )
    rows = [
        [tiles_ncdhw[r * n_cols + c : r * n_cols + c + 1] for c in range(n_cols)]
        for r in range(n_rows)
    ]
    ref = model._assemble_tiles(
        rows,
        [v // model.vae_ratio for v in plan.yo],
        [v // model.vae_ratio for v in plan.xo],
    )

    # Emulate _tile_blend_nthwc_to_ncdhw_kernel arithmetic in BF16 and verify 0-ulp match
    tiles_nthwc = tiles_ncdhw.permute(0, 2, 3, 4, 1).contiguous()
    lut = plan.blend_lut
    raw_flat = tiles_nthwc.permute(1, 0, 2, 3, 4).reshape(
        t_lat, expected_tiles * th * tw, c_lat
    )
    v_cur = raw_flat[:, lut.idx_cur.long(), :]
    v_up = raw_flat[:, lut.idx_up.long(), :]
    v_y = torch.where(
        lut.has_y[None, :, None],
        v_up * lut.wy0[None, :, None] + v_cur * lut.wy1[None, :, None],
        v_cur,
    )
    v_left = raw_flat[:, lut.idx_left.long(), :]
    h_val = torch.where(
        lut.has_x[None, :, None],
        v_left * lut.wx0[None, :, None] + v_y * lut.wx1[None, :, None],
        v_y,
    )
    out = h_val.permute(2, 0, 1).reshape(1, c_lat, t_lat, lut.out_h, lut.out_w)
    assert torch.equal(out, ref)


@pytest.mark.parametrize(
    "height,width,tile_size,overlap",
    [(65, 96, 64, 16), (64, 97, 64, 16), (481, 833, 256, 64)],
)
def test_fast_geometry_rejects_tiles_clipped_by_native_slicing(
    height, width, tile_size, overlap
):
    model = _TiledH3Fixture(tile_size=tile_size, tile_overlap_min=overlap)
    ys, hs, _ = model.split_tiles(height, False)
    xs, ws, _ = model.split_tiles(width, False)
    assert any(y + h > height for y, h in zip(ys, hs)) or any(
        x + w > width for x, w in zip(xs, ws)
    )
    with pytest.raises(ValueError, match="tiles within the canvas"):
        _build_tile_geometry_plan(
            model, height, width, device=torch.device("cpu"), dtype=torch.bfloat16
        )


def test_cpu_and_env_fallbacks_preserve_existing_contracts(monkeypatch):
    torch.manual_seed(43)
    model = _TiledH3Fixture(tile_size=64, tile_overlap_min=16).to(torch.bfloat16)
    codecs = _make_codecs_fixture(model)
    codecs.prepare_visual_encoder_weights()
    assert getattr(model, "_fast_single_gpu_encoder", None) is None

    cache = CausalEncoderCache(
        model,
        tile_mode="replicated",
        encode_mode="performance",
        encode_dtype="bfloat16",
    )
    assert not cache.fast_single_gpu
    assert not cache.can_encode_uint8_fast(64, 64)

    frames = torch.randint(0, 256, (3, 9, 64, 64), dtype=torch.uint8)
    session = create_session(codecs, capacity_frames=17)
    session.append(frames, frame_start=0)
    first = session.pop()
    second = session.pop()
    assert (first.start, first.end) == (0, 1)
    assert (second.start, second.end) == (1, 3)
    session.dispose()
    assert session.state_usage().retained_bytes == 0

    # Verify monkeypatching codecs_module.encode_clips bypasses fast visual encode check
    called = []
    orig_encode_clips = codecs_module.encode_clips

    def custom_encode_clips(m, clips, bs):
        called.append(len(clips))
        return orig_encode_clips(m, clips, bs)

    monkeypatch.setattr(codecs_module, "encode_clips", custom_encode_clips)
    assert not codecs._can_fast_visual_encode_profile(
        frames[:, :1], check_encode_clips_hook=True
    )
    lat = codecs.encode_visual(frames[:, :1])
    assert called == [1]
    assert lat.shape == (1, 24, 1, 4, 4)


@pytest.mark.skipif(not _CUDA_TRITON, reason="Requires CUDA and Triton")
def test_cuda_fast_kernels_and_progressive_parity(monkeypatch):
    torch.manual_seed(101)
    model = (
        _TiledH3Fixture(tile_size=64, tile_overlap_min=16)
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    param_snapshot = {
        name: (p.data_ptr(), p.stride(), p.detach().clone())
        for name, p in model.named_parameters()
    }
    codecs = _make_codecs_fixture(model)
    codecs.prepare_visual_encoder_weights()
    assert getattr(model, "_fast_single_gpu_encoder", None) is not None
    codecs.warmup_source_vae()

    # Shared model parameters must never be mutated or re-strided in place
    for name, p in model.named_parameters():
        ptr, stride, orig = param_snapshot[name]
        assert p.data_ptr() == ptr, name
        assert p.stride() == stride, name
        assert torch.equal(p, orig), name

    total_frames = 34
    video = torch.randint(0, 256, (3, total_frames, 112, 160), dtype=torch.uint8)

    def _run_stream(fast_flag: str, schedule: list[int], src: torch.Tensor):
        monkeypatch.setenv("MINIMAX_H3_ENCODER_SINGLE_GPU_FAST", fast_flag)
        sess = create_session(codecs, capacity_frames=total_frames)
        try:
            assert sess.cache.fast_single_gpu is (fast_flag == "1")
            cursor = 0
            emitted = []
            for step in schedule:
                sess.append(src[:, cursor : cursor + step], frame_start=cursor)
                cursor += step
                while sess.ready:
                    emitted.append(sess.pop().values.detach().clone())
            sess.finish()
            while sess.ready:
                emitted.append(sess.pop().values.detach().clone())
            return torch.cat(emitted, dim=2), dict(sess.cache.last_execution)
        finally:
            sess.dispose()

    base_out, base_exec = _run_stream("0", [total_frames], video)
    fast_whole, fast_exec = _run_stream("1", [total_frames], video)
    fast_single, _ = _run_stream("1", [1] * total_frames, video)
    fast_irreg, _ = _run_stream("1", [5, 4, 8, 3, 6, 8], video)

    assert not base_exec["fast_single_gpu"]
    assert fast_exec["fast_single_gpu"]
    assert fast_whole.shape == base_out.shape == (1, 48, 10, 7, 10)
    # Packetization invariance: exact 0-ulp match across arbitrary arrival schedules
    assert torch.equal(fast_whole, fast_single)
    assert torch.equal(fast_whole, fast_irreg)

    diff = (fast_whole.float() - base_out.float()).abs()
    rrmse = diff.pow(2).mean().sqrt() / base_out.float().pow(2).mean().sqrt()
    assert float(diff.max()) < 0.08
    assert float(rrmse) < 0.03

    # Verify FP32-accumulated fast path is at least as close to exact FP32 ground truth as native BF16
    model_fp32 = (
        _TiledH3Fixture(tile_size=64, tile_overlap_min=16).cuda().float().eval()
    )
    model_fp32.load_state_dict({k: v.float() for k, v in model.state_dict().items()})
    codecs_fp32 = _make_codecs_fixture(model_fp32, encode_dtype="float32")
    monkeypatch.setenv("MINIMAX_H3_FAST_VISUAL_ENCODE", "0")
    fp32_out = torch.cat(
        [
            codecs_fp32.encode_visual_moments(video[:, :17]).detach().float(),
            codecs_fp32.encode_visual_moments(video[:, 17:34]).detach().float(),
        ],
        dim=2,
    )
    fast_vs_fp32 = (fast_whole.float() - fp32_out).pow(2).mean().sqrt() / fp32_out.pow(
        2
    ).mean().sqrt()
    base_vs_fp32 = (base_out.float() - fp32_out).pow(2).mean().sqrt() / fp32_out.pow(
        2
    ).mean().sqrt()
    assert float(fast_vs_fp32) < 0.022
    assert float(fast_vs_fp32) <= float(base_vs_fp32) * 1.1

    # Strict causality: corrupting frames 9..16 must have 0-ulp effect on latents 0..2
    perturbed = video.clone()
    perturbed[:, 9:17] = 255 - perturbed[:, 9:17]
    fast_perturbed, _ = _run_stream("1", [9, 8, 17], perturbed)
    assert torch.equal(fast_perturbed[:, :, :3], fast_whole[:, :, :3])


@pytest.mark.skipif(not _CUDA_TRITON, reason="Requires CUDA and Triton")
def test_cuda_transactional_fault_injection_and_zero_retained_bytes(monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_ENCODER_SINGLE_GPU_FAST", "1")
    torch.manual_seed(202)
    model = (
        _TiledH3Fixture(tile_size=64, tile_overlap_min=16)
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    codecs = _make_codecs_fixture(model)
    codecs.prepare_visual_encoder_weights()

    frames = torch.randint(0, 256, (3, 17, 112, 160), dtype=torch.uint8)

    # Clean reference run
    ref_sess = create_session(codecs, capacity_frames=17)
    ref_sess.append(frames, frame_start=0)
    ref_parts = [ref_sess.pop().values.clone() for _ in range(3)]
    assert ref_sess.state_usage().retained_bytes == 0
    ref_sess.dispose()

    # Fault-injected run: fail inside quant_conv on the second pop (frames 1..8)
    sess = create_session(codecs, capacity_frames=17)
    sess.append(frames, frame_start=0)
    p0 = sess.pop().values.clone()
    assert sess.state_usage().retained_bytes > 0
    assert torch.equal(p0, ref_parts[0])

    def _boom(_x):
        raise RuntimeError("Injected quant_conv failure")

    model.quant_conv.forward = _boom
    with pytest.raises(RuntimeError, match="Injected quant_conv failure"):
        sess.pop()
    del model.quant_conv.__dict__["forward"]

    # State must remain clean; retry succeeds and matches uninterrupted run bit-for-bit
    p1 = sess.pop().values.clone()
    p2 = sess.pop().values.clone()
    assert torch.equal(p1, ref_parts[1])
    assert torch.equal(p2, ref_parts[2])
    # Clip boundary (17 frames) resets retained bytes to 0
    assert sess.state_usage().retained_bytes == 0

    sess.reset()
    assert sess.state_usage().retained_bytes == 0
    sess.dispose()
    assert sess.state_usage().retained_bytes == 0


@pytest.mark.skipif(not _CUDA_TRITON, reason="Requires CUDA and Triton")
@pytest.mark.parametrize("frame_count", [1, 5, 17, 20])
def test_cuda_fast_visual_encode_and_moments_parity(monkeypatch, frame_count):
    torch.manual_seed(303 + frame_count)
    model = (
        _TiledH3Fixture(tile_size=64, tile_overlap_min=16)
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    codecs = _make_codecs_fixture(model)
    codecs.prepare_visual_encoder_weights()

    frames = torch.randint(0, 256, (3, frame_count, 112, 160), dtype=torch.uint8)

    monkeypatch.setenv("MINIMAX_H3_FAST_VISUAL_ENCODE", "0")
    base_lat = codecs.encode_visual(frames)
    monkeypatch.setenv("MINIMAX_H3_FAST_VISUAL_ENCODE", "1")
    fast_lat = codecs.encode_visual(frames)

    assert fast_lat.shape == base_lat.shape
    assert fast_lat.dtype == base_lat.dtype == torch.float16
    diff = (fast_lat.detach().float() - base_lat.detach().float()).abs()
    rrmse = diff.pow(2).mean().sqrt() / base_lat.detach().float().pow(2).mean().sqrt()
    assert float(rrmse) < 0.02

    if frame_count in (1, 17):
        monkeypatch.setenv("MINIMAX_H3_FAST_VISUAL_ENCODE", "0")
        base_m = codecs.encode_visual_moments(frames).detach()
        monkeypatch.setenv("MINIMAX_H3_FAST_VISUAL_ENCODE", "1")
        fast_m = codecs.encode_visual_moments(frames).detach()
        assert fast_m.shape == base_m.shape
        m_rrmse = (fast_m.float() - base_m.float()).pow(
            2
        ).mean().sqrt() / base_m.float().pow(2).mean().sqrt()
        assert float(m_rrmse) < 0.03

        model_fp32 = (
            _TiledH3Fixture(tile_size=64, tile_overlap_min=16).cuda().float().eval()
        )
        model_fp32.load_state_dict(
            {k: v.float() for k, v in model.state_dict().items()}
        )
        codecs_fp32 = _make_codecs_fixture(model_fp32, encode_dtype="float32")
        monkeypatch.setenv("MINIMAX_H3_FAST_VISUAL_ENCODE", "0")
        fp32_m = codecs_fp32.encode_visual_moments(frames).detach().float()
        fast_err = (fast_m.float() - fp32_m).pow(2).mean().sqrt() / fp32_m.pow(
            2
        ).mean().sqrt()
        base_err = (base_m.float() - fp32_m).pow(2).mean().sqrt() / fp32_m.pow(
            2
        ).mean().sqrt()
        assert float(fast_err) < 0.022
        assert float(fast_err) <= float(base_err) * 1.1

    if frame_count == 1:
        item = {"type": "image", "frames": frames}
        monkeypatch.setenv("MINIMAX_H3_FAST_VISUAL_ENCODE", "0")
        r0 = codecs.encode_reference(item)
        monkeypatch.setenv("MINIMAX_H3_FAST_VISUAL_ENCODE", "1")
        r1 = codecs.encode_reference(item)
        assert codecs.reference_cache.snapshot(tenant="vae")["miss"] == 2
        assert r0.shape == r1.shape
        _ = json.dumps({"ok": True})


def test_fast_encoder_tiny_canvas_and_uninitialized_sp_fallback(monkeypatch):
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.fast_encoder import (
        _build_tile_geometry_plan,
    )

    model = _TiledH3Fixture(tile_size=64, tile_overlap_min=16).eval()
    model.encoder_tiling = False
    with pytest.raises(ValueError, match="Tile dimensions"):
        _build_tile_geometry_plan(
            model,
            16,
            64,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

    codecs = _make_codecs_fixture(model)
    frames = torch.zeros((3, 1, 64, 64), dtype=torch.uint8)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.distributed.model_parallel_is_initialized",
        lambda: False,
    )
    assert codecs._can_fast_visual_encode_profile(frames) is False


@pytest.mark.skipif(not _CUDA_TRITON, reason="Requires CUDA and Triton")
def test_fast_encoder_dynamic_dtype_hook_and_chunking_guards(monkeypatch):
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.fast_encoder import (
        _conv3d_nthwc,
        stage_uint8_to_gpu,
    )
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.progressive import (
        CausalEncoderCache,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
        create_session,
    )

    torch.manual_seed(99)
    model = (
        _TiledH3Fixture(tile_size=64, tile_overlap_min=16)
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    codecs = _make_codecs_fixture(model)
    codecs.prepare_visual_encoder_weights()

    cache = CausalEncoderCache(
        model, encode_mode="performance", encode_dtype="bfloat16"
    )
    assert cache.fast_single_gpu is True

    # 1. Dynamic cast to FP32 invalidates fast_single_gpu cleanly
    model.float()
    assert cache.fast_single_gpu is False
    assert cache.can_encode_uint8_fast(64, 64) is False
    model.to(torch.bfloat16)
    assert cache.fast_single_gpu is True

    # 2. Submodule forward hook / cache.forward monkeypatch falls back cleanly
    handle = model.encoder.conv_in.register_forward_hook(lambda m, i, o: o)
    assert cache.fast_single_gpu is False
    handle.remove()
    assert cache.fast_single_gpu is True

    called = {"forward": 0}
    session = create_session(codecs, capacity_frames=8)
    orig_forward = session.cache.forward

    def _wrapped_forward(*args, **kwargs):
        called["forward"] += 1
        return orig_forward(*args, **kwargs)

    session.cache.forward = _wrapped_forward
    session.append(
        torch.randint(0, 256, (3, 1, 64, 64), dtype=torch.uint8), frame_start=0
    )
    _ = session.pop()
    assert called["forward"] > 0
    session.dispose()

    # 3. Output-element 32-bit overflow chunking (3 -> 128 channel expansion)
    assets = model._fast_single_gpu_encoder
    x_pad = torch.randn((4, 3, 18, 18, 3), device="cuda", dtype=torch.bfloat16)
    w_expand = torch.randn((128, 3, 3, 3, 3), device="cuda", dtype=torch.bfloat16).to(
        memory_format=torch.channels_last_3d
    )
    full_out = _conv3d_nthwc(x_pad, w_expand, (1, 1, 1), (1, 1, 1), 1)
    # Input numel is 11,664 (<= 40,000), while output numel is 131,072 (> 40,000) -> 4 chunks of b=1
    conv_calls = {"n": 0}
    orig_conv3d = torch.nn.functional.conv3d

    def _counting_conv3d(*args, **kwargs):
        conv_calls["n"] += 1
        return orig_conv3d(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "conv3d", _counting_conv3d)
    monkeypatch.setenv("MINIMAX_H3_FAST_CONV_MAX_ELEMS", "40000")
    chunked_out = _conv3d_nthwc(x_pad, w_expand, (1, 1, 1), (1, 1, 1), 1)
    assert conv_calls["n"] == 4
    torch.testing.assert_close(chunked_out, full_out, rtol=2e-2, atol=0.1)

    # 4. Pinned H2D buffer growth synchronizes in-flight events safely
    small_cpu = torch.randint(0, 256, (3, 1, 32, 32), dtype=torch.uint8)
    _ = stage_uint8_to_gpu(assets, small_cpu)
    assets.pinned_slots = [
        torch.empty((1024,), dtype=torch.uint8, pin_memory=True),
        torch.empty((1024,), dtype=torch.uint8, pin_memory=True),
    ]
    larger_cpu = torch.randint(0, 256, (3, 1, 64, 64), dtype=torch.uint8)
    gpu_out = stage_uint8_to_gpu(assets, larger_cpu)
    assert torch.equal(gpu_out.cpu(), larger_cpu)
