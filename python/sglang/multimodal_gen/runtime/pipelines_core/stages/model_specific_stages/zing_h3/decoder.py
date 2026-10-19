# SPDX-License-Identifier: Apache-2.0
"""Decoder registry: per-class construction and capability declarations for Zing H3.

Callers read capabilities from the selected adapter and never infer the class
from a module name; `module_key` only names the component for residency.
"""

import torch

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3_decoder import DecoderSpec

from .source_decoder import SourceLynnDecoder
from .streaming_codecs import StableVideoDecoder


def sp_tile_group():
    """SP group whose ranks split Lynn tiles; None without distributed SP>1."""
    from sglang.multimodal_gen.runtime.distributed import (
        get_sp_group,
        model_parallel_is_initialized,
    )

    if not torch.distributed.is_initialized() or not model_parallel_is_initialized():
        return None
    group = get_sp_group()
    return group if group.world_size > 1 else None


class OfficialH3Adapter:
    """Exported H3 video_vae; overlap-window decode in immediate or stable mode."""

    module_key = "video_vae"
    stateful = False
    realtime_modes = ("immediate", "stable")
    source_decoder = StableVideoDecoder

    def build(self, spec: DecoderSpec) -> dict:
        # The generic component loader already owns the exported video_vae.
        return {}


class LynnStreamingAdapter:
    module_key = "lynn_decoder"
    stateful = True
    realtime_modes = ("stable",)

    def build(self, spec: DecoderSpec) -> dict:
        from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.loader import (
            load_lynn_decoder,
        )

        # Built on CPU; residency moves it to the GPU and keeps the mixed dtypes.
        return {self.module_key: load_lynn_decoder(spec, device="cpu")}

    def stable_frames(self, module, latent_count):
        # Each decode_step finalizes its frames; there is no lookahead tail.
        return module.protocol.frame_count(latent_count)

    def new_stream(self, module, *, frame_limit):
        return module.new_stream(frame_limit=frame_limit, group=sp_tile_group())

    def decode(self, module, latent, frames):
        return module.decode(latent, frames, group=sp_tile_group())

    source_decoder = SourceLynnDecoder


DECODERS = {
    "official_h3": OfficialH3Adapter(),
    "lynn_streaming": LynnStreamingAdapter(),
}


def decoder_adapter(spec: DecoderSpec):
    """Return the adapter for `spec`; unknown classes never fall back."""
    try:
        return DECODERS[spec.type]
    except KeyError:
        raise ValueError(f"Unknown Zing decoder type: {spec.type!r}") from None


def build_decoder_modules(spec: DecoderSpec) -> dict:
    """Return the extra pipeline modules for `spec`."""
    return decoder_adapter(spec).build(spec)
