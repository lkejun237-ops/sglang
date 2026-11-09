# SPDX-License-Identifier: Apache-2.0
"""Applicable pre-RPC regressions retained under the approved v6 redesign."""

import importlib
from pathlib import Path

import pytest
import torch

MODULE = "sglang.multimodal_gen.runtime.conditioning.contract"
SOURCE = Path(__file__).parents[2] / "runtime/conditioning/inputs.py"


def test_input_contract_is_present():
    assert SOURCE.is_file(), "The shared conditioning input contract is required"


@pytest.fixture
def api():
    if not SOURCE.is_file():
        pytest.skip("Feature assertion above owns the initial red gate")
    return importlib.import_module(MODULE)


def image(value=1, **kw):
    return dict(
        id="<Image 1>",
        type="image",
        frames=torch.full((3, 1, 2, 4), value, dtype=torch.uint8),
        fps=1.0,
        **kw,
    )


def test_21_images_and_source_preview_identity_preserves_full_snapshot(api):
    from sglang.multimodal_gen.runtime.conditioning.inputs import (
        content_identity,
        validate_condition_media,
    )
    from sglang.multimodal_gen.runtime.conditioning.resources import ResourceLimits

    media = [dict(image(), type="video")] + [image(i) for i in range(1, 22)]
    profile = "a" * 64
    original = content_identity("prompt", media, profile)
    changed = [*media[:-1], image(99)]
    assert original != content_identity("prompt", changed, profile)
    assert original != content_identity("prompt", media[::-1], profile)
    with pytest.raises(api.ConditionError, match="Media count"):
        validate_condition_media([*media, image()], max_media=22)
    with pytest.raises(api.ConditionError, match="Media count"):
        validate_condition_media(media, max_media=21)
    with pytest.raises(api.ConditionError, match="Decoded media"):
        validate_condition_media(media, max_input_bytes=media[0]["frames"].numel())
    assert ResourceLimits().max_media == 22
    assert ResourceLimits(max_media=33).max_media == 33


def _cpu_encoder(monkeypatch):
    # Isolate the existing CPU presentation module from pipelines_core's CUDA
    # package initializer; execute its real source, rather than a test double.
    import importlib.util
    import sys

    name = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation"
    path = (
        Path(__file__).parents[2]
        / "runtime/pipelines_core/stages/model_specific_stages/minimax_h3/presentation.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    presentation = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, presentation)
    spec.loader.exec_module(presentation)
    return importlib.import_module(
        "sglang.multimodal_gen.runtime.conditioning.encoder"
    ).encode_condition


def test_shared_encoder_statement_identity_matches_frozen_native(api, monkeypatch):
    import ast
    import hashlib
    import inspect

    encode_condition = _cpu_encoder(monkeypatch)
    assert (
        inspect.signature(encode_condition).parameters["max_prompt_tokens"].default
        is None
    )
    function = ast.parse(inspect.getsource(encode_condition)).body[0]
    # The optional budget preflight does not alter native encoding statements.
    # Pin its exact invocation before normalizing it; token-budget contracts
    # independently verify rejection before media preprocessing or GPU work.
    preflight = [
        item
        for item in function.body
        if isinstance(item, ast.Expr)
        and isinstance(item.value, ast.Call)
        and isinstance(item.value.func, ast.Name)
        and item.value.func.id == "validate_prompt_token_budget"
    ]
    assert len(preflight) == 1
    assert ast.dump(preflight[0], include_attributes=False) == ast.dump(
        ast.parse(
            "validate_prompt_token_budget(tokenizer, prompt, max_prompt_tokens)"
        ).body[0],
        include_attributes=False,
    )
    body = [
        item
        for item in function.body
        if item is not preflight[0]
        and not (isinstance(item, ast.If) and "max_rows" in ast.unparse(item.test))
    ]
    tree = ast.Module(body=body, type_ignores=[])

    class DefaultPresentation(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            # Per-call dependency injection leaves the default native function
            # unchanged; its behavior is also verified independently below.
            if (
                isinstance(node.func, ast.BoolOp)
                and isinstance(node.func.op, ast.Or)
                and [ast.unparse(value) for value in node.func.values]
                == ["presentation_fn", "minimax_h3_ref2va_video_presentation"]
            ):
                node.func = node.func.values[1]
            return node

    tree = DefaultPresentation().visit(tree)
    # Python 3.14 ast.dump omits empty list fields by default. Normalize 3.12
    # to the same representation without changing the frozen native assertion.
    for node in ast.walk(tree):
        node._fields = tuple(
            name for name in node._fields if getattr(node, name, None) != []
        )
    fingerprint = hashlib.sha256(
        ast.dump(tree, include_attributes=False).encode()
    ).hexdigest()
    assert (
        fingerprint
        == "c6ab4b92243aa1c80c91e69756615a3eb5810d16150a19af96bdd0e47f53f1e7"
    )


def test_native_encoder_six_images_video_sampling_padding_and_token_cap(
    api, monkeypatch
):
    from types import SimpleNamespace

    encode_condition = _cpu_encoder(monkeypatch)

    class Tokenizer:
        def __init__(self):
            self.texts = []

        def __call__(self, text, **kwargs):
            self.texts.append(text)
            return {"input_ids": [ord(c) for c in text]}

        def convert_tokens_to_ids(self, token):
            return {
                "<|vision_start|>": 1,
                "<|vision_end|>": 2,
                "<|image_pad|>": 3,
                "<|video_pad|>": 4,
            }[token]

    class Images:
        merge_size = 1

        def __call__(self, images, **kwargs):
            self.images = images
            return {
                "pixel_values": torch.tensor([int(i.flat[0]) for i in images]),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
            }

    class Videos:
        temporal_patch_size = 2

        def __call__(self, videos, **kwargs):
            self.videos, self.options = videos, kwargs
            return {
                "pixel_values_videos": torch.tensor(
                    [int(f.flat[0]) for f in videos[0]]
                ),
                "video_grid_thw": torch.tensor([[len(videos[0]) // 2, 2, 2]]),
            }

    class Encoder:
        calls = 0

        def encode_ids(self, ids, **features):
            self.calls += 1
            self.ids, self.features = ids, features
            return ids[:, None].expand(-1, 5120).to(torch.bfloat16).contiguous()

    tokenizer, images, videos, encoder = Tokenizer(), Images(), Videos(), Encoder()
    modules = {
        "tokenizer": tokenizer,
        "processor": SimpleNamespace(image_processor=images, video_processor=videos),
        "text_encoder": encoder,
    }
    runtime = {"condition_encoder_config": {"video_sample_fps": 8}}
    video = dict(
        id="v",
        type="video",
        fps=30,
        frames=torch.arange(10, dtype=torch.uint8)[None, :, None, None].expand(
            3, -1, 2, 4
        ),
    )
    media = [video] + [image(i) | {"id": f"i{i}"} for i in range(1, 7)]
    result = encode_condition(modules, runtime, "  完整 prompt é  ", media)
    assert [int(i.flat[0]) for i in images.images] == [1, 2, 3, 4, 5, 6]
    assert [int(f.flat[0]) for f in videos.videos[0]] == [0, 4, 8, 8]
    assert videos.options["do_sample_frames"] is False
    assert "  完整 prompt é  " in tokenizer.texts
    assert result["context"].shape[0] == result["row_role"].shape[0]
    assert set(result["row_role"].tolist()) == {0, 1}
    before = encoder.calls
    with pytest.raises(api.ConditionError):
        encode_condition(modules, runtime, "p", media, max_rows=1)
    assert encoder.calls == before


def test_encoder_presentation_dependency_is_per_call(api, monkeypatch):
    from types import SimpleNamespace

    encode_condition = _cpu_encoder(monkeypatch)
    encoder = SimpleNamespace(
        encode_ids=lambda ids, **_: torch.zeros(ids.numel(), 5120, dtype=torch.bfloat16)
    )
    modules = {
        "tokenizer": object(),
        "processor": SimpleNamespace(image_processor=SimpleNamespace(merge_size=1)),
        "text_encoder": encoder,
    }
    calls = []

    def presentation(tokenizer, **kwargs):
        calls.append(kwargs)
        return torch.tensor([1, 2]), torch.tensor([1, 0])

    import inspect

    assert "presentation_fn" in inspect.signature(encode_condition).parameters
    result = encode_condition(modules, {}, " p ", [], presentation_fn=presentation)
    assert calls[0]["prompt"] == " p "
    assert result["row_role"].tolist() == [0, 1]


@pytest.mark.parametrize("count", [1, 10, 22, 33])
def test_media_rgb_limits_derive_from_count_without_large_allocations(api, count):
    from sglang.multimodal_gen.runtime.conditioning.inputs import (
        content_identity,
        validate_condition_media,
    )
    from sglang.multimodal_gen.runtime.conditioning.resources import ResourceLimits

    limits = ResourceLimits(max_media=count)
    assert limits.max_input_bytes == count * (48 << 20)
    assert limits.max_decoded_pixels == count * 16_777_216
    assert limits.max_request_bytes == limits.max_input_body_bytes
    assert limits.max_condition_bytes == limits.output_reservation_bytes
    assert (
        limits.max_response_bytes
        == limits.max_condition_bytes + limits.max_header_bytes + 1024
    )
    assert (
        limits.max_inflight_bytes_per_tenant
        >= limits.max_request_bytes + limits.output_reservation_bytes
    )
    # Stride-zero views represent maximum-sized RGB images without allocating
    # count * 48 MiB; these checks read metadata only, not tensor content.
    frames = torch.zeros((), dtype=torch.uint8).expand(3, 1, 4096, 4096)
    media = [dict(type="image", frames=frames) for _ in range(count)]
    assert (
        len(
            validate_condition_media(
                media,
                max_media=count,
                max_side=limits.max_side,
                max_input_bytes=limits.max_input_bytes,
                max_decoded_pixels=limits.max_decoded_pixels,
            )
        )
        == count
    )
    with pytest.raises(api.ConditionError, match="Decoded media"):
        validate_condition_media(media, max_input_bytes=limits.max_input_bytes - 1)
    # Callers that already hashed frames must not meet another hidden 512MiB cap.
    assert content_identity("prompt", media, "a" * 64, item_hashes=["b" * 64] * count)


def test_local_cache_keeps_all_32_references_and_source(api):
    from sglang.multimodal_gen.runtime.conditioning.worker import LocalConditioner

    calls = []

    def encode(prompt, media):
        calls.append(len(media))
        return dict(
            context=torch.zeros(2, 5120, dtype=torch.bfloat16),
            row_role=torch.zeros(2, dtype=torch.int64),
        )

    conditioner = LocalConditioner(encode, "a" * 64, cache_bytes=1 << 20)
    media = [dict(image(), type="video")] + [image(i) for i in range(32)]
    conditioner("prompt", media)
    conditioner("prompt", media)
    assert calls == [33]
    media[-1]["frames"].add_(1)
    conditioner("prompt", media)
    assert calls == [33, 33]
