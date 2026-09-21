# SPDX-License-Identifier: Apache-2.0
"""Numerical contracts at the H3/Zing sharing boundary (no model weights)."""

import math
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
    minimax_h3_encode_audio_mean,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
    ZingH3Codecs,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
    _patchify_video_latents,
    _unpatchify_video_rows,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3EulerSchedule,
)
from sglang.srt.model_loader.weight_loading import default_weight_loader


@pytest.mark.parametrize(
    "shape,patch", [((1, 3, 2, 4, 6), (1, 2, 2)), ((2, 3, 4, 6, 8), (2, 2, 2))]
)
def test_patch_rows_keep_patch_major_channels(shape, patch):
    x = torch.arange(math.prod(shape)).reshape(shape)
    rows = _patchify_video_latents(x, patch)
    # Enumerate patches independently: each row's columns must be the channels
    # of the same pixel, rather than adjacent pixels of the same channel.
    expected = []
    pt, ph, pw = patch
    for batch in x:
        for t in range(0, shape[2], pt):
            for h in range(0, shape[3], ph):
                for w in range(0, shape[4], pw):
                    expected.append(
                        batch[:, t : t + pt, h : h + ph, w : w + pw].flatten(1).T
                    )
    assert torch.equal(rows, torch.stack(expected))
    assert torch.equal(_unpatchify_video_rows(rows, shape[0], shape[2:], patch), x)
    with pytest.raises(ValueError, match="row count"):
        _unpatchify_video_rows(rows, shape[0] + 1, shape[2:], patch)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shift", [3.0, 5.0])
def test_euler_sharing_preserves_every_step_and_rng(dtype, shift):
    schedule = ZingH3EulerSchedule(50, shift)
    sample = torch.linspace(-2, 2, 35).reshape(5, 7).to(dtype)
    prediction = torch.linspace(0.3, -0.7, 35).reshape(5, 7).to(dtype)
    rng = torch.get_rng_state()
    for i in range(len(schedule.timesteps)):
        # The pre-refactor Zing contract includes FP32 x0 and interpolation,
        # followed by one cast back to the sample dtype.
        clean = sample.float() + (1 - schedule.timesteps[i]) * prediction.float()
        ratio = schedule.sigmas[i + 1] / schedule.sigmas[i]
        expected = (ratio * sample.float() + (1 - ratio) * clean).to(dtype)
        actual, master = schedule.step(prediction, sample, step_index=i)
        assert torch.equal(actual, expected)
        assert torch.equal(master, clean)
        assert master.dtype == torch.float32
        sample = actual
    assert torch.equal(rng, torch.get_rng_state())


class AudioVAE(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.attn_proj = attention

    def preprocess(self, waveform, rate):
        assert rate == 32000
        assert waveform.shape == (2, 1, 7)
        return waveform

    def encoder(self, x):
        return torch.cat((x, 2 * x, 3 * x), dim=1)

    def pre_block(self, x):
        assert x.shape == (2, 7, 3)
        return x + 0.25

    def mean_proj(self, x):
        return x.to(torch.bfloat16)


@pytest.mark.parametrize("attention", [False, True])
def test_audio_mean_sharing_preserves_zing_normalization(attention):
    vae = AudioVAE(attention)
    waveform = torch.linspace(-1, 1, 14).reshape(2, 7)
    expected = torch.stack((waveform, 2 * waveform, 3 * waveform), dim=1)
    expected = (expected + (0.25 if attention else 0)).bfloat16().float()
    assert torch.equal(minimax_h3_encode_audio_mean(vae, waveform, 32000), expected)
    codecs = object.__new__(ZingH3Codecs)
    codecs.modules = {"audio_vae": vae}
    codecs.audio_sample_rate = 32000
    codecs.aa = SimpleNamespace(
        latent_channels=3, latents_mean=[0.1, 0.2, 0.3], latents_std=[0.7, 0.8, 0.9]
    )
    normalized = (
        expected.permute(0, 2, 1).unsqueeze(0) - torch.tensor([0.1, 0.2, 0.3])
    ) / torch.tensor([0.7, 0.8, 0.9])
    actual = codecs.encode_audio(waveform)
    assert actual.shape == (1, 2, 7, 3)
    assert actual.dtype == torch.float16
    assert torch.equal(actual, normalized.half())


def test_shared_weight_loader_preserves_storage_scalars_and_shape_errors():
    param = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.bfloat16))
    ptr = param.data_ptr()
    source = torch.arange(6.0).reshape(3, 2).T
    default_weight_loader(param, source)
    assert param.data_ptr() == ptr
    assert torch.equal(param, source.bfloat16())
    scalar = torch.nn.Parameter(torch.zeros(1, 1))
    default_weight_loader(scalar, torch.tensor(3.5))
    assert scalar.item() == 3.5
    with pytest.raises(AssertionError, match="Attempted to load weight"):
        default_weight_loader(param, torch.zeros(3, 2))


def test_h3_public_request_fields_resolve_same_canvas_without_frame_alignment():
    from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
        minimax_h3_validate_canonical_request,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
        minimax_h3_resolve_plan,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.request_adapter import (
        build_h3_request,
    )

    fields = dict(
        task="ref2va",
        prompt="A moving scene",
        conditions=[{"type": "image", "uri": "ref.png", "role": "reference"}],
        target={"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 5},
    )
    source = {"type": "video", "uri": "source.mp4"}
    params = ZingH3SamplingParams(**fields, source=source)
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )

    stage = object.__new__(ZingH3GenerationStage)
    stage.codecs = SimpleNamespace(target_fps=24)
    request = stage._build_request(
        SimpleNamespace(sampling_params=params, extra=params.build_request_extra())
    )
    plan = minimax_h3_resolve_plan(minimax_h3_validate_canonical_request(**fields))
    target = request["messages"][1]
    assert target["source"] == source
    assert target["content"][0]["output"] == {
        "height": plan.shape["height"],
        "width": plan.shape["width"],
        "frames": 120,
    }
    assert target["conditions"] == [
        {
            "content_id": "reference-0",
            "role": "reference",
            "latent_modalities": ["visual"],
        }
    ]
    assert params.build_request_extra()["target"] == fields["target"]
    assert params.build_request_extra()["source"] == source
    params.zing_request = {"messages": []}
    with pytest.raises(ValueError, match="not both"):
        build_h3_request(params, {}, media_fps=24)


def test_h3_request_video_reference_keeps_audio_routing_and_start_time():
    from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.request_adapter import (
        build_h3_request,
    )

    params = ZingH3SamplingParams(
        task="ref2va",
        prompt="A moving scene",
        conditions=[
            {"type": "image", "uri": "ref.png", "role": "reference"},
            {
                "type": "video",
                "uri": "ref.mp4",
                "role": "reference",
                "start_time_seconds": 1.25,
            },
        ],
        target={"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5},
        zing_generate_audio=True,
    )
    request = build_h3_request(params, {}, media_fps=24)
    assert request["messages"][0]["content"][2]["clip"] == {"start": 1.25}
    assert request["messages"][1]["conditions"][1]["latent_modalities"] == [
        "visual",
        "audio",
    ]
    assert request["messages"][1]["content"][0]["type"] == "video_audio"


@pytest.mark.parametrize("task", ["t2va", "fl2va", "unknown"])
def test_zing_rejects_other_h3_tasks(task):
    from sglang.multimodal_gen.configs.sample.zing_h3 import ZingH3SamplingParams
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.request_adapter import (
        build_h3_request,
    )

    params = ZingH3SamplingParams(task=task, prompt="A scene")
    with pytest.raises(ValueError, match="only supports task='ref2va'"):
        build_h3_request(params, {}, media_fps=24)


@pytest.mark.parametrize("time_major", [False, True])
def test_h3_audio_entrypoint_keeps_normalization_in_inference_mode(
    monkeypatch, time_major
):
    from contextlib import nullcontext

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        reference_encoding,
    )

    waveform = torch.linspace(-1, 1, 14).reshape(2, 7).requires_grad_()
    monkeypatch.setattr(
        reference_encoding, "_load_waveform", lambda *a, **kw: (waveform, 32000)
    )
    monkeypatch.setattr(
        reference_encoding, "set_forward_context", lambda **kw: nullcontext()
    )
    config = SimpleNamespace(
        latent_channels=3, latents_mean=[0.1, 0.2, 0.3], latents_std=[0.7, 0.8, 0.9]
    )

    class LayoutAudioVAE(AudioVAE):
        def mean_proj(self, x):
            z = super().mean_proj(x)
            return z.transpose(1, 2).contiguous() if time_major else z

    with torch.enable_grad():
        result = reference_encoding.minimax_h3_encode_reference_audio_rows(
            LayoutAudioVAE(False), "unused.wav", config
        )
    expected = (
        torch.stack((waveform, 2 * waveform, 3 * waveform), dim=1)
        .bfloat16()
        .float()
        .permute(0, 2, 1)
    )
    expected = (
        (expected - torch.tensor(config.latents_mean))
        / torch.tensor(config.latents_std)
    ).reshape(-1, 3)
    assert torch.equal(result["rows"], expected)
    assert not result["rows"].requires_grad
    assert result["ref_audio_t"] == 7
