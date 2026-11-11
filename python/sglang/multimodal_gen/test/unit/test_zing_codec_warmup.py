# SPDX-License-Identifier: Apache-2.0
"""Worker startup probes execute real codec paths and release request state."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
    get_forward_context_or_none,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codec_warmup import (
    WARMUP_PROMPT,
    warmup_decoder,
    warmup_encoder,
)
from sglang.multimodal_gen.test.unit.test_zing_lynn_stream import model  # noqa: F401


def test_encoder_warms_source_vae_before_one_text_request_with_forward_context():
    calls = []
    before = get_forward_context_or_none()

    def encode(prompt, media):
        assert torch.is_inference_mode_enabled()
        assert get_forward_context().current_timestep == 0
        calls.append((prompt, media))
        return {
            "context": torch.ones(3, 8),
            "row_role": torch.zeros(3, dtype=torch.long),
        }

    warmup_encoder(
        SimpleNamespace(
            warmup_source_vae=lambda: calls.append("source_vae"),
            encode_condition=encode,
        )
    )
    assert calls == ["source_vae", (WARMUP_PROMPT, [])]
    assert get_forward_context_or_none() is before


@pytest.mark.parametrize("failure", ["source_vae", "forward", "nonfinite", "empty"])
def test_encoder_failure_propagates_and_restores_forward_context(failure):
    before = get_forward_context_or_none()
    calls = []

    def source_warmup():
        calls.append("source_vae")
        if failure == "source_vae":
            raise RuntimeError("Injected source VAE failure")

    def encode(prompt, media):
        calls.append("qwen")
        if failure == "forward":
            raise RuntimeError("Injected Qwen failure")
        rows = 0 if failure == "empty" else 3
        return {
            "context": torch.full((rows, 8), float("nan")),
            "row_role": torch.zeros(rows, dtype=torch.long),
        }

    with pytest.raises(RuntimeError):
        warmup_encoder(
            SimpleNamespace(warmup_source_vae=source_warmup, encode_condition=encode)
        )
    assert calls == (
        ["source_vae"] if failure == "source_vae" else ["source_vae", "qwen"]
    )
    assert get_forward_context_or_none() is before


def test_decoder_runs_one_real_latent_and_releases_stream(model, monkeypatch):
    streams, inputs = [], []
    original = model.new_stream

    def new_stream(**kwargs):
        assert kwargs == {"frame_limit": 1, "group": None}
        stream = original(**kwargs)
        push = stream.push

        def record(latent):
            assert not torch.is_grad_enabled()
            assert not torch.is_inference_mode_enabled()
            assert not torch.is_inference(latent)
            inputs.append(latent.clone())
            output = push(latent)
            assert not torch.is_inference(output)
            return output

        stream.push = record
        streams.append(stream)
        return stream

    monkeypatch.setattr(model, "new_stream", new_stream)
    random_state = torch.random.get_rng_state().clone()
    warmup_decoder(model, width=12, height=12)
    assert len(inputs) == len(streams) == 1
    assert inputs[0].shape == (1, 24, 1, 3, 3)
    assert streams[0].latents_seen == streams[0].frames_emitted == 1
    assert streams[0].closed and streams[0].state is None
    assert torch.equal(torch.random.get_rng_state(), random_state)
    # The next user stream starts with no KV/phase state from warmup.
    stream = original()
    assert stream.latents_seen == 0 and stream.state.tiles == {}
    stream.close()


@pytest.mark.parametrize("failure", [None, "forward", "nonfinite", "peer_prepare"])
def test_decoder_consensus_and_cleanup(failure):
    events = []
    group = object()

    class Stream:
        def push(self, latent):
            events.append("push")
            assert latent.shape == (1, 24, 1, 3, 3)
            if failure == "forward":
                raise RuntimeError("Injected decode failure")
            value = float("nan") if failure == "nonfinite" else 0.5
            return torch.full((1, 3, 1, 12, 12), value)

        def finish(self, frames):
            assert frames == 1
            events.append("finish")

        def close(self):
            events.append("close")

    def new_stream(**kwargs):
        assert kwargs == {"frame_limit": 1, "group": group}
        return Stream()

    def agree(stage, value, *, error):
        events.append((stage, error is not None))
        if failure == "peer_prepare" and stage == "decoder_warmup_prepare":
            raise RuntimeError("Injected peer preparation failure")

    decoder = SimpleNamespace(
        spatial_ratio=4,
        architecture={"latent_channels": 24},
        latent_mean=torch.zeros(24),
        new_stream=new_stream,
    )
    if failure is None:
        warmup_decoder(decoder, group=group, agree=agree, width=12, height=12)
    else:
        with pytest.raises(RuntimeError):
            warmup_decoder(decoder, group=group, agree=agree, width=12, height=12)
    assert events[0] == ("decoder_warmup_prepare", False)
    assert events[-1] == "close"
    if failure == "peer_prepare":
        assert "push" not in events
    else:
        assert events[-2] == ("decoder_warmup_complete", failure is not None)


def test_decoder_invalid_canvas_is_reported_before_any_decode():
    votes = []

    def unexpected_stream(**kwargs):
        pytest.fail("Invalid input entered the decoder")

    decoder = SimpleNamespace(spatial_ratio=4, new_stream=unexpected_stream)
    with pytest.raises(ValueError, match="latent aligned"):
        warmup_decoder(
            decoder,
            width=13,
            height=12,
            agree=lambda stage, value, **kwargs: votes.append((stage, kwargs["error"])),
        )
    assert len(votes) == 1
    assert votes[0][0] == "decoder_warmup_prepare"
    assert isinstance(votes[0][1], ValueError)
