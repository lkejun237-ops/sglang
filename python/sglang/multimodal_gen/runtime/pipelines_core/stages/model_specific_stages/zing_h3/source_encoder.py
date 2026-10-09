# SPDX-License-Identifier: Apache-2.0
"""Select a source codec from the trusted deployment profile, never the wire.

Here profile means deployment configuration. Remote trace capture is enabled
only during an actual Torch profiler request; ordinary requests do not capture traces.
"""

import importlib

from .streaming_codecs import SourcePixelBuffer


def create_source_buffer(codecs, profile, *, capacity_frames):
    if profile == {"mode": "clip17"}:
        return SourcePixelBuffer(codecs, capacity_frames=capacity_frames)
    # #142 owns the shared contract and encoder implementation. Import only
    # when selected so the unchanged clip baseline remains independently usable.
    from .progressive_encoder import PROGRESSIVE_ENCODER_VERSION

    if (
        profile.get("mode") != "progressive"
        or profile.get("contract_version") != PROGRESSIVE_ENCODER_VERSION
    ):
        raise ValueError("Unsupported progressive source encoder profile")
    if (
        codecs.vae_frames_per_chunk,
        codecs.vae_latents_per_chunk,
        codecs.temporal_ratio,
    ) != (17, 5, 4):
        raise ValueError("Progressive source encoder requires clip17 VAE geometry")
    path = profile.get("factory", "")
    module, separator, name = path.partition(":")
    if not separator or not module or not name:
        raise ValueError("A progressive source encoder factory is required")
    factory = getattr(importlib.import_module(module), name)
    if "remote" in profile:
        import torch
        from .remote_encoder import create_session

        config = dict(profile["remote"])
        if not torch.autograd.profiler._is_profiler_enabled:
            config.pop("profile_dir", None)
        session = create_session(codecs, capacity_frames=capacity_frames, config=config)
    else:
        session = factory(codecs, capacity_frames=capacity_frames)
    try:
        if session.contract_version != PROGRESSIVE_ENCODER_VERSION:
            raise ValueError("Source encoder contract version does not match")
        if (
            type(session.minimum_credit_frames) is not int
            or not 1 <= session.minimum_credit_frames <= 8
        ):
            raise ValueError("Source encoder cannot advance with progressive credit")
        if session.accepted_frames or session.emitted_latents:
            raise ValueError("Source encoder must start with empty session state")
    except BaseException:
        session.dispose()
        raise
    return session
