# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for Zing media routing; model inference is mocked."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    codecs as codecs_module,
)


@pytest.fixture
def codecs():
    instance = object.__new__(codecs_module.ZingH3Codecs)
    instance.spatial_ratio = 16
    instance.patch_size = (1, 2, 2)
    instance.temporal_ratio = 4
    instance.vae_frames_per_chunk = 17
    instance.vae_latents_per_chunk = 5
    instance.target_fps = 24.0
    instance.max_short_edge = 32
    instance.canvas_multiple = 32
    instance.max_area_config = dict(
        reference_image=1024, reference_video=1024, source=1024
    )
    instance.max_source_target_aspect_ratio_factor = 2.0
    instance.audio_sample_rate = 32000
    return instance


def make_request(kind, *, references=0, source=True):
    count = 1 if kind == "image" else 125
    media = [
        {"type": "image", "id": f"ref-{i}", "uri": f"reference-{i}.png"}
        for i in range(references)
    ]
    target = {
        "role": "target",
        "content": [
            {"type": kind, "output": {"frames": count, "height": 32, "width": 32}}
        ],
        "conditions": [
            {
                "content_id": item["id"],
                "role": "reference",
                "latent_modalities": ["visual"],
            }
            for item in media
        ],
    }
    if source:
        target["source"] = {
            "type": "image" if kind == "image" else "video",
            "uri": "source",
        }
    return {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Edit the scene"}, *media],
            },
            target,
        ]
    }


@pytest.mark.parametrize(
    "kind,references,source",
    [
        ("image", 0, True),
        ("image", 3, True),
        ("video", 0, True),
        ("video", 3, True),
        ("video_audio", 0, True),
    ],
    ids=[
        "image-edit",
        "multi-reference-image",
        "video-edit",
        "reference-source-video",
        "video-audio-edit",
    ],
)
@pytest.mark.parametrize("retain_media", [False, True])
def test_prepare_routes_reference_and_source_media(
    codecs, monkeypatch, kind, references, source, retain_media
):
    loaded, decoded, encoded_media, audio_samples = [], [], [], []

    @contextmanager
    def use_component(name):
        loaded.append(name)
        yield

    def decode(item, *, need_audio):
        decoded.append((item["uri"], need_audio))
        frames = 1 if item["type"] == "image" else 125
        return {
            **item,
            "frames": torch.zeros(3, frames, 32, 32, dtype=torch.uint8),
            "fps": 24.0,
        }

    def condition(prompt, media):
        encoded_media.extend(item["id"] for item in media)
        return {
            "context": torch.zeros(1, 4),
            "row_role": torch.zeros(1, dtype=torch.long),
        }

    def audio(waveform):
        audio_samples.append(tuple(waveform.shape))
        return torch.zeros(1, 2, 10, 4)

    monkeypatch.setattr(codecs, "_decode_media", decode)
    monkeypatch.setattr(codecs, "encode_condition", condition)
    monkeypatch.setattr(
        codecs, "encode_visual", lambda frames: torch.zeros(1, 2, 1, 2, 2)
    )
    monkeypatch.setattr(codecs, "encode_audio", audio)
    result = codecs.prepare(
        make_request(kind, references=references, source=source),
        use_component,
        retain_condition_media=retain_media,
    )
    assert (result.condition_media is not None) is retain_media
    if retain_media:
        assert [item["id"] for item in result.condition_media] == encoded_media
    assert result.output_type == kind
    assert result.pixel_frame_count == (1 if kind == "image" else 125)
    assert result.visual_shape == ((1 if kind == "image" else 38), 2, 2)
    assert (result.source_visual is not None) is source
    assert list(result.visual) == [f"ref-{i}" for i in range(references)]
    assert [item["id"] for item in result.media] == encoded_media == list(result.visual)
    assert len(result.conditions) == references
    assert result.audio == {}
    assert all(not need_audio for _, need_audio in decoded)
    assert loaded == ["text_encoder", "video_vae"] + (
        ["audio_vae"] if kind == "video_audio" else []
    )
    if kind == "video_audio":
        # 125 frames / 24 fps, truncated to whole samples at 32 kHz.
        assert result.sample_count == 166666
        assert result.sample_rate == 32000
        assert result.audio_shape == (2, 10, 4)
        assert audio_samples == [(2, 166666)]
    else:
        assert result.sample_count is None
        assert result.audio_shape is None
        assert audio_samples == []


def test_source_video_length_must_match_long_target(codecs, monkeypatch):
    monkeypatch.setattr(
        codecs,
        "_decode_media",
        lambda item, **kwargs: {
            **item,
            "frames": torch.zeros(3, 124, 32, 32, dtype=torch.uint8),
            "fps": 24.0,
        },
    )
    with pytest.raises(ValueError, match="source frames must equal"):
        codecs.prepare(
            make_request("video"), lambda name: pytest.fail("No model should load")
        )


@pytest.mark.parametrize(
    "source_fps,expected",
    [(1.0, [0.5, 2.5, 4.0]), (3.0, [1 / 3, 7 / 6])],
    ids=["deduplicated-low-fps", "rounded-noninteger-stride"],
)
def test_condition_video_timestamps_follow_source_frames(
    codecs, monkeypatch, source_fps, expected
):
    seen = {}

    class Processor:
        temporal_patch_size = 2

        def __call__(self, *, videos, do_sample_frames, return_tensors):
            assert not do_sample_frames
            assert return_tensors == "pt"
            seen["frames"] = len(videos[0])
            return {
                "pixel_values_videos": torch.zeros(1),
                "video_grid_thw": torch.tensor([[len(videos[0]) // 2, 2, 2]]),
            }

    def presentation(tokenizer, **kwargs):
        seen.update(kwargs)
        return torch.tensor([1]), torch.tensor([0])

    codecs.modules = {
        "tokenizer": object(),
        "processor": SimpleNamespace(
            video_processor=Processor(), image_processor=SimpleNamespace(merge_size=2)
        ),
        "text_encoder": SimpleNamespace(
            encode_ids=lambda ids, **kwargs: torch.zeros(1, 4)
        ),
    }
    codecs.runtime = {"condition_encoder_config": {"video_sample_fps": 2.0}}
    monkeypatch.setattr(
        codecs_module, "minimax_h3_ref2va_video_presentation", presentation
    )
    codecs.encode_condition(
        "A video",
        [
            {
                "type": "video",
                "fps": source_fps,
                "frames": torch.zeros(3, 5, 2, 2, dtype=torch.uint8),
            }
        ],
    )
    assert seen["video_block_timestamps"][0] == pytest.approx(expected)
    assert seen["frames"] == len(expected) * 2


@pytest.fixture
def temporal_codecs(codecs):
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae import (
        AutoencoderKL,
    )

    # Exercise the actual H3 temporal assembly/blending, replacing only the
    # expensive learned per-window decoder with a context-sensitive CPU kernel.
    vae = AutoencoderKL.__new__(AutoencoderKL)
    torch.nn.Module.__init__(vae)
    vae.register_parameter("anchor", torch.nn.Parameter(torch.zeros(())))
    vae.vae_ratio_t = 4
    vae.clip_length = 17
    vae.token_drop = 3
    vae.tokens_chunk_size = 5
    vae.token_overlap = 2
    vae.frame_pre_padding = 3
    vae.frame_overlap = 5
    vae.isolated_first_frame = vae.isolated_last_frame = False
    vae.use_3d_conv = True
    vae._blend_weight_cache = {}
    vae.eval()
    calls = []

    def decode(z):
        calls.append(z.detach().clone())
        return z.repeat_interleave(4, dim=2) + z.mean(dim=2, keepdim=True)

    vae._adaptive_decode = decode
    codecs.modules = {"video_vae": vae}
    codecs.va = SimpleNamespace(
        vae_token_drop=3, latents_mean=[0, 0, 0], latents_std=[1, 1, 1]
    )
    return codecs, calls


@pytest.mark.parametrize("stream_cat", ["0", "1"])
@pytest.mark.parametrize("frames", [1, 2, 17, 18, 22, 34, 39, 243])
def test_decode_range_matches_full_h3_temporal_assembly(
    temporal_codecs, monkeypatch, stream_cat, frames
):
    import math

    codecs, calls = temporal_codecs
    monkeypatch.setenv("MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT", stream_cat)
    full, tail = divmod(frames - 1, 17)
    tokens = full * 5 + 1 + math.ceil(tail / 4)
    latent = torch.linspace(-0.4, 0.4, tokens).view(1, 1, tokens, 1, 1)
    latent = latent.expand(1, 3, tokens, 1, 1)
    expected = codecs.decode_visual(latent, frames)
    # Every start offset includes clip boundaries and the blended overlap.
    for start in range(frames):
        actual = codecs.decode_visual_range(latent, start, frames)
        torch.testing.assert_close(actual, expected[:, :, start:], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["immediate", "stable"])
def test_streaming_ranges_match_previous_prefix_semantics_with_bounded_work(
    temporal_codecs, mode
):
    codecs, calls = temporal_codecs
    latent = torch.linspace(-0.4, 0.4, 72).view(1, 1, 72, 1, 1).expand(1, 3, 72, 1, 1)
    emitted = 0
    for end in [*range(5, 72, 5), 72]:
        full, tail = divmod(end - 1, 5)
        frame_end = min(243, full * 17 + 1 + tail * 4)
        if mode == "stable" and end != 72:
            frame_end = max(0, (end - 2) // 5) * 17
        if frame_end <= emitted:
            continue
        expected = codecs.decode_visual(latent[:, :, :end], frame_end)[:, :, emitted:]
        calls.clear()
        actual = codecs.decode_visual_range(latent[:, :, :end], emitted, frame_end)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Work per output remains bounded even late in a 10-second sequence.
        assert len(calls) <= 3
        assert all(z.shape[2] <= 7 for z in calls)
        emitted = frame_end
    assert emitted == 243


@pytest.mark.parametrize("height,width", [(31, 32), (32, 31), (0, 32), (64, 64)])
def test_source_request_rejects_invalid_target_geometry_before_loading(
    codecs, monkeypatch, height, width
):
    request = make_request("video")
    request["messages"][1]["content"][0]["output"].update(height=height, width=width)
    monkeypatch.setattr(
        codecs,
        "_decode_media",
        lambda item, **kw: {
            **item,
            "frames": torch.zeros(3, 125, 32, 32, dtype=torch.uint8),
            "fps": 24.0,
        },
    )
    with pytest.raises(ValueError, match="geometry"):
        codecs.prepare(request, lambda name: pytest.fail("No model should load"))


def test_source_request_rejects_incompatible_aspect_before_loading(codecs, monkeypatch):
    monkeypatch.setattr(
        codecs,
        "_decode_media",
        lambda item, **kw: {
            **item,
            "frames": torch.zeros(3, 125, 32, 128, dtype=torch.uint8),
            "fps": 24.0,
        },
    )
    with pytest.raises(ValueError, match="aspect ratio"):
        codecs.prepare(
            make_request("video"), lambda name: pytest.fail("No model should load")
        )


@pytest.mark.parametrize(
    "uri", ["image.png", "https://example.test/ref.png", "s3://bucket/ref.png"]
)
def test_h3_and_zing_share_reference_descriptor_validation(codecs, uri):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
        minimax_h3_validate_canonical_request,
    )

    request = make_request("image", references=1, source=False)
    request["messages"][0]["content"][1]["uri"] = uri
    _, media, _, _, _ = codecs._parse_inference_request(request)
    canonical = minimax_h3_validate_canonical_request(
        task="ref2va",
        prompt="Edit the scene",
        conditions=[{"type": "image", "uri": uri, "role": "reference"}],
        target={"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 5},
    )
    assert {key: media[0][key] for key in ("type", "uri")} == {
        key: canonical["conditions"][0][key] for key in ("type", "uri")
    }
    request["messages"][0]["content"][1]["uri"] = ""
    with pytest.raises(ValueError, match="uri must be a non-empty string"):
        codecs._parse_inference_request(request)
    with pytest.raises(ValueError, match="uri must be a non-empty string"):
        minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="Edit the scene",
            conditions=[{"type": "image", "uri": "", "role": "reference"}],
            target={"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 5},
        )


@pytest.mark.parametrize(
    "clip",
    [{"start": -1}, {"start": True}, {"start": float("nan")}, {"start": 2, "end": 1}],
)
def test_source_clip_rejected_before_decoding(codecs, clip):
    request = make_request("video")
    request["messages"][1]["source"]["clip"] = clip
    with pytest.raises(ValueError):
        codecs._parse_inference_request(request)


@pytest.mark.parametrize("mode", ["deterministic", "performance"])
@pytest.mark.parametrize("outer_deterministic", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_visual_encode_policy_restores_backend_state(mode, outer_deterministic, fail):
    def state():
        return (
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )

    original = state()
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = outer_deterministic
        torch.use_deterministic_algorithms(outer_deterministic, warn_only=True)
        before = state()
        try:
            with codecs_module._visual_encode_policy(mode):
                assert state()[:3] == (
                    mode == "performance",
                    mode == "deterministic",
                    mode == "deterministic",
                )
                if fail:
                    raise RuntimeError("encoder failed")
        except RuntimeError as exc:
            assert fail and str(exc) == "encoder failed"
        assert state() == before
        with (
            pytest.raises(ValueError, match="encode_mode"),
            codecs_module._visual_encode_policy("fast"),
        ):
            pytest.fail("invalid mode must not enter encoding")
        assert state() == before
    finally:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = original[
            :2
        ]
        torch.use_deterministic_algorithms(original[2], warn_only=original[3])


@pytest.mark.parametrize("mode", ["deterministic", "performance"])
@pytest.mark.parametrize("frames", [1, 17, 22])
def test_visual_encode_forwards_mode_and_preserves_seed(
    codecs, monkeypatch, mode, frames
):
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(1, 1)
    model.quant_conv = torch.nn.Linear(1, 1)
    codecs.modules = {"video_vae": model}
    codecs.runtime = {
        "vae_config": {
            "encode_mode": mode,
            "encode_dtype": "float32",
            "encode_tile_batch_size": 2,
            "posterior_seed": 42,
        }
    }
    codecs.va = SimpleNamespace(latents_mean=[0.0, 0.0], latents_std=[1.0, 1.0])
    calls = []

    def encode_clips(model, clips, batch_size):
        assert torch.backends.cudnn.benchmark == (mode == "performance")
        assert torch.backends.cudnn.deterministic == (mode == "deterministic")
        assert torch.are_deterministic_algorithms_enabled() == (mode == "deterministic")
        assert batch_size == 2
        calls.append([clip.shape[2] for clip in clips])
        return [
            torch.zeros(1, 4, 1 if clip.shape[2] == 1 else 5, 1, 1) for clip in clips
        ]

    monkeypatch.setattr(codecs_module, "encode_clips", encode_clips)
    pixels = torch.zeros(3, frames, 1, 1, dtype=torch.uint8)
    first = codecs.encode_visual(pixels)
    second = codecs.encode_visual(pixels)
    assert torch.equal(first, second)
    expected_frames = {1: 1, 17: 5, 22: 7}[frames]
    assert first.shape == (1, 2, expected_frames, 1, 1)
    assert first.dtype == torch.float16
    assert calls == [[1] if frames == 1 else [17] * ((frames + 16) // 17)] * 2


@pytest.mark.parametrize("count", [5, 9])
def test_stream_prompt_preserves_reference_order_without_extra_video(
    codecs, monkeypatch, count
):
    from contextlib import nullcontext

    seen = []
    monkeypatch.setattr(
        codecs,
        "_decode_media",
        lambda item, **kw: {
            **item,
            "frames": torch.full(
                (3, 1, 32, 32), int(item["id"].split("-")[1]), dtype=torch.uint8
            ),
            "fps": 24,
        },
    )
    monkeypatch.setattr(
        codecs, "encode_condition", lambda prompt, media: seen.extend(media) or {}
    )
    encoded = []
    monkeypatch.setattr(
        codecs,
        "encode_visual",
        lambda pixels: encoded.append(pixels) or torch.zeros(1, 2, 1, 2, 2),
    )
    request = make_request("video", references=count, source=False)
    # Use deliberately non-sorted IDs and distinct pixels; preserve wire order.
    request["messages"][0]["content"][1:] = reversed(
        request["messages"][0]["content"][1:]
    )
    request["messages"][1]["conditions"].reverse()
    result = codecs.prepare(
        request,
        lambda name: nullcontext(),
        retain_condition_media=True,
    )
    assert [item["id"] for item in seen] == [f"ref-{i}" for i in reversed(range(count))]
    assert all(item["type"] == "image" for item in seen)
    assert len(seen) == len(encoded) == len(result.visual) == count
    assert seen[0]["frames"].eq(count - 1).all()
    assert encoded[0].eq(count - 1).all()
    assert result.source_visual is None
    assert result.condition_media == seen
