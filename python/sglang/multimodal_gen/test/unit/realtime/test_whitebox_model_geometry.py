# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for preserving native source pixels at the model boundary."""

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.model_geometry import (
    ModelGeometry,
    resolution_spec,
)


def test_resolution_selector_is_exact_and_omission_preserves_the_host_default():
    default = VideoSpec(32, 32)
    assert resolution_spec(None, default) is default
    assert resolution_spec("480p", default) == VideoSpec(832, 480)
    assert resolution_spec("720p", default) == VideoSpec(1280, 720)
    for invalid in ("736p", "1280x720", "720", 720, True, {}):
        with pytest.raises(ValueError) as failure:
            resolution_spec(invalid, default)
        assert failure.value.code == "INVALID_RESOLUTION"
        assert failure.value.stage == "validation"


def test_720_source_rows_survive_model_padding_and_crop_exactly():
    geometry = ModelGeometry(VideoSpec(1280, 720))
    rows = [bytes([row % 251, 83, 194]) * 1280 for row in range(720)]
    source = b"".join(rows)
    padded = geometry.pad_rgb(source)
    assert geometry.model == VideoSpec(1280, 736)
    assert len(padded) == 2_826_240
    assert padded[:2_764_800] == source
    assert padded[2_764_800:] == rows[-1] * 16
    assert geometry.crop_rgb(padded) == source


def test_crop_uses_each_model_frame_boundary_for_a_packed_batch():
    geometry = ModelGeometry(VideoSpec(1280, 720))
    first = b"a" * 2_764_800
    second = b"b" * 2_764_800
    model = first + b"x" * 61_440 + second + b"y" * 61_440
    assert geometry.crop_rgb(model, num_frames=2) == first + second
    assert geometry.pad_rgb(first + second, num_frames=2) == (
        first + b"a" * 61_440 + second + b"b" * 61_440
    )


def test_aligned_geometry_preserves_the_original_bytes_object():
    geometry = ModelGeometry(VideoSpec(832, 480))
    source = bytes(1_198_080)
    assert geometry.model == geometry.source
    assert geometry.pad_rgb(source) is source
    assert geometry.crop_rgb(source) is source


def test_model_init_preserves_public_initialization_and_unrelated_fields():
    geometry = ModelGeometry(VideoSpec(1280, 720))
    stream = dict(format="rgb24", width=1280, height=720, fps=24)
    references = [dict(id="reference", payload=b"image")]
    initial = dict(
        size="1280x720", source_stream=stream, references=references, seed=42
    )
    model = geometry.model_init(initial)
    assert model == dict(
        size="1280x736",
        source_stream=dict(format="rgb24", width=1280, height=736, fps=24),
        references=references,
        seed=42,
    )
    assert initial["size"] == "1280x720"
    assert initial["source_stream"] is stream and stream["height"] == 720
    assert model is not initial and model["source_stream"] is not stream
    assert model["references"] is references


def test_receipt_explains_public_and_private_pixel_geometry():
    receipt = ModelGeometry(VideoSpec(1280, 720)).receipt()
    assert receipt == dict(
        protocol="whitebox-model-geometry-v1",
        source=dict(width=1280, height=720, fps=24),
        model=dict(width=1280, height=736, fps=24),
        padding=dict(mode="replicate", left=0, top=0, right=0, bottom=16),
        output_crop=dict(x=0, y=0, width=1280, height=720),
    )


def test_unaligned_width_replicates_only_the_last_pixel_in_each_row():
    geometry = ModelGeometry(VideoSpec(2, 2))
    source = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    row1 = source[:6] + bytes([4, 5, 6]) * 30
    row2 = source[6:] + bytes([10, 11, 12]) * 30
    assert geometry.model == VideoSpec(32, 32)
    assert geometry.pad_rgb(source) == row1 + row2 * 31
    assert geometry.crop_rgb(row1 + row2 * 31) == source


@pytest.mark.parametrize("operation", ["pad_rgb", "crop_rgb"])
@pytest.mark.parametrize("data", [b"", b"invalid", bytearray(3)])
def test_invalid_rgb_lengths_and_types_are_rejected(operation, data):
    with pytest.raises(ValueError, match="RGB"):
        getattr(ModelGeometry(VideoSpec(32, 32)), operation)(data)


@pytest.mark.parametrize("operation", ["pad_rgb", "crop_rgb"])
@pytest.mark.parametrize("num_frames", [0, -1, True, 1.0])
def test_frame_count_must_be_a_positive_integer(operation, num_frames):
    with pytest.raises(ValueError, match="frame count"):
        getattr(ModelGeometry(VideoSpec(32, 32)), operation)(
            bytes(3072), num_frames=num_frames
        )


@pytest.mark.parametrize(
    "spec",
    [
        VideoSpec(0, 720),
        VideoSpec(1280, -1),
        VideoSpec(True, 32),
        VideoSpec(32, 32, 30),
    ],
)
def test_geometry_rejects_invalid_dimensions_and_non_24_fps(spec):
    with pytest.raises(ValueError, match="dimensions|24 FPS"):
        ModelGeometry(spec)


def test_independent_480_input_736_output_never_upscales_or_crops_rgb():
    source, target = VideoSpec(832, 480), VideoSpec(1280, 736)
    geometry = ModelGeometry(source, target)
    rgb = bytes([37]) * source.frame_bytes
    generated = bytes([71]) * target.frame_bytes + bytes([129]) * target.frame_bytes
    assert geometry.ingress == source
    assert geometry.model == geometry.output == target
    assert geometry.pad_rgb(rgb) is rgb
    assert geometry.crop_rgb(generated, num_frames=2) is generated
    init = geometry.model_init(dict(seed=42, source_stream={"width": 832}))
    assert init["size"] == "1280x736"
    assert init["source_stream"] == dict(format="rgb24", width=832, height=480, fps=24)
    receipt = geometry.receipt()
    assert receipt["padding"]["right"] == receipt["padding"]["bottom"] == 0
    assert receipt["model_input"]["width"] == 832
    assert receipt["output_crop"] == dict(x=0, y=0, width=1280, height=736)


def test_independent_target_requires_its_own_rgb_length_and_aligned_grid():
    geometry = ModelGeometry(VideoSpec(832, 480), VideoSpec(1280, 736))
    with pytest.raises(ValueError, match="RGB"):
        geometry.crop_rgb(bytes(832 * 480 * 3))
    with pytest.raises(ValueError, match="RGB"):
        geometry.pad_rgb(bytes(1280 * 736 * 3))
    with pytest.raises(ValueError, match="grid-aligned"):
        ModelGeometry(VideoSpec(832, 480), VideoSpec(1280, 720))
