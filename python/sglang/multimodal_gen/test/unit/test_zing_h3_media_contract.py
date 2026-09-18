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
        ("video_audio", 0, True),
    ],
    ids=["image-edit", "multi-reference-image", "video-edit", "video-audio-edit"],
)
def test_prepare_routes_four_media_cases(codecs, monkeypatch, kind, references, source):
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
        make_request(kind, references=references, source=source), use_component
    )
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
