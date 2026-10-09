# SPDX-License-Identifier: Apache-2.0
"""Profiler-only remote trace configuration must not pollute ordinary requests."""

from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    remote_encoder,
    source_encoder,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
    PROGRESSIVE_ENCODER_VERSION,
)


@pytest.mark.parametrize("profiling", [False, True])
def test_remote_trace_directory_is_used_only_during_profile(monkeypatch, profiling):
    seen = []

    def create_session(codecs, *, capacity_frames, config):
        seen.append(config)
        return SimpleNamespace(
            contract_version=PROGRESSIVE_ENCODER_VERSION,
            minimum_credit_frames=8,
            accepted_frames=0,
            emitted_latents=0,
            dispose=lambda: None,
        )

    monkeypatch.setattr(remote_encoder, "create_session", create_session)
    profile = dict(
        mode="progressive",
        contract_version=PROGRESSIVE_ENCODER_VERSION,
        factory="sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder:create_session",
        remote=dict(port=31001, mode="overlap", profile_dir="/traces"),
    )
    codecs = SimpleNamespace(
        vae_frames_per_chunk=17, vae_latents_per_chunk=5, temporal_ratio=4
    )

    def create():
        source_encoder.create_source_buffer(codecs, profile, capacity_frames=51)

    if profiling:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
            create()
    else:
        create()
    assert ("profile_dir" in seen[0]) is profiling
    assert profile["remote"]["profile_dir"] == "/traces"
