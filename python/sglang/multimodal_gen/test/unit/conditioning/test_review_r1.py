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
