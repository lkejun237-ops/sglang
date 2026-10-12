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
    function = ast.parse(inspect.getsource(encode_condition)).body[0]
    body = [
        item
        for item in function.body
        if not (isinstance(item, ast.If) and "max_rows" in ast.unparse(item.test))
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
