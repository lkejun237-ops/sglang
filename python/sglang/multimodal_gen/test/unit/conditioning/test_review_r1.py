# SPDX-License-Identifier: Apache-2.0
"""Applicable pre-RPC regressions retained under the approved v6 redesign."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.conditioning import native


def condition():
    return {
        "context": torch.ones(2, 5120, dtype=torch.bfloat16),
        "row_role": torch.tensor([0, 1]),
    }


@pytest.mark.parametrize("raises", [False, True])
def test_native_drains_selected_device_on_success_and_exception(monkeypatch, raises):
    events = []
    encoder = object.__new__(native.NativeConditionEncoder)
    encoder.gpu_id, encoder.max_rows = 3, 10
    encoder.max_prompt_tokens = None
    encoder.pipeline = SimpleNamespace(modules={}, zing_runtime={})

    def encode(*args, **kwargs):
        assert kwargs == {"max_prompt_tokens": None}
        events.append("encode")
        if raises:
            raise RuntimeError("private GPU error")
        return condition()

    monkeypatch.setattr(
        torch.cuda, "set_device", lambda device: events.append(("device", device))
    )
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: events.append(("drain", device))
    )
    from sglang.multimodal_gen.runtime.conditioning import encoder as encoder_module

    monkeypatch.setattr(encoder_module, "encode_condition", encode)
    value = SimpleNamespace(prompt="prompt", media=[])
    if raises:
        with pytest.raises(RuntimeError, match="private GPU error"):
            encoder(value)
    else:
        assert encoder(value)["context"].device.type == "cpu"
    assert events[-1] == ("drain", 3), (
        "Native failure returned before selected-device CUDA drain"
    )


@pytest.mark.parametrize("over_budget", [False, True])
def test_native_prompt_budget_with_prefix_task(monkeypatch, over_budget):
    from unittest.mock import Mock

    from sglang.multimodal_gen.runtime.conditioning import encoder as encoder_module
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.conditioning.inputs import media_identity

    cache = SimpleNamespace(clear=Mock())
    task = SimpleNamespace(cache=cache)
    forward = Mock(return_value=condition()["context"])
    presentation = Mock(return_value=(torch.tensor([1, 2]), torch.tensor([0, 1])))
    encoder = object.__new__(native.NativeConditionEncoder)
    encoder.gpu_id, encoder.max_rows, encoder.max_prompt_tokens = 3, 10, 8
    encoder.pipeline = SimpleNamespace(
        modules={
            "tokenizer": lambda text, **kwargs: {"input_ids": list(text)},
            "processor": SimpleNamespace(image_processor=SimpleNamespace(merge_size=1)),
            "text_encoder": SimpleNamespace(encode_ids=forward),
        },
        zing_runtime={},
    )
    monkeypatch.setattr(
        encoder_module, "minimax_h3_ref2va_video_presentation", presentation
    )
    monkeypatch.setattr(torch.cuda, "set_device", Mock())
    drain = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", drain)
    value = SimpleNamespace(prompt="a" * (9 if over_budget else 8), media=[], task=task)

    if over_budget:
        with pytest.raises(ConditionError, match="Prompt tokens"):
            encoder.encode_device(value)
        presentation.assert_not_called()
        forward.assert_not_called()
        cache.clear.assert_called_once_with()
    else:
        result = encoder.encode_device(value)
        assert torch.equal(result["context"], condition()["context"])
        assert forward.call_args.kwargs == {
            "prefix_cache": cache,
            "media_identity": media_identity([]),
        }
        cache.clear.assert_not_called()
    drain.assert_called_once_with(3)
