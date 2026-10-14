# SPDX-License-Identifier: Apache-2.0
"""Decoder registry: per-class construction and capability declarations for Zing H3.

Callers read capabilities from the selected adapter and never infer the class
from a module name; `module_key` only names the component for residency.
"""

import torch

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3_decoder import DecoderSpec

from .source_decoder import SourceLynnDecoder, SourceTAEH3Decoder
from .streaming_codecs import StableVideoDecoder

_TAEH3_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class OfficialH3Adapter:
    """Exported H3 video_vae; overlap-window decode in immediate or stable mode."""

    module_key = "video_vae"
    stateful = False
    realtime_modes = ("immediate", "stable")
    source_decoder = StableVideoDecoder

    def build(self, spec: DecoderSpec) -> dict:
        # The generic component loader already owns the exported video_vae.
        return {}


class TAEH3Adapter:
    module_key = "taeh3_decoder"
    stateful = True
    realtime_modes = ("stable",)

    def build(self, spec: DecoderSpec) -> dict:
        from sglang.multimodal_gen.runtime.models.vaes.taeh3 import (
            WEIGHTS_SHA256,
            TAEH3Decoder,
        )

        decoder = TAEH3Decoder(spec.checkpoint_path, WEIGHTS_SHA256)
        return {self.module_key: decoder.to(dtype=_TAEH3_DTYPES[spec.precision])}

    def stable_frames(self, module, latent_count):
        # Five-latent decode groups minus the upstream twelve-frame tail.
        return max(0, (latent_count // 5) * 17 - 12)

    def new_stream(self, module, *, frame_limit):
        return module.new_stream()

    source_decoder = SourceTAEH3Decoder


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
        return module.new_stream(frame_limit=frame_limit)

    source_decoder = SourceLynnDecoder


DECODERS = {
    "official_h3": OfficialH3Adapter(),
    "taeh3": TAEH3Adapter(),
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
