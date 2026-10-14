# SPDX-License-Identifier: Apache-2.0
"""Output-frame schedule of the Lynn streaming decoder.

Ported from the Zing reference (seedleap/zing @ 8eda9c22c6, src/h3/vae_decoder/temporal.py).
"""

import math


class TemporalProtocol:
    def __init__(self, architecture):
        # Head slots per latent (4 for H3) and valid RGB frames per phase group (17).
        self.ratio = math.prod(architecture["temporal_downsample_factors"])
        self.clip_length = architecture["clip_length"]
        self.phases = math.ceil(self.clip_length / self.ratio)
        self.pre_padding = (-self.clip_length) % self.ratio
        self.overlap = (-architecture["token_drop"]) % self.phases
        self.teacher_window = self.phases + self.overlap

    def frame_count(self, latent_count):
        groups, remainder = divmod(latent_count, self.phases)
        tail = remainder * self.ratio - self.pre_padding if remainder else 0
        return groups * self.clip_length + tail

    def new_frames(self, latent_index):
        """Frames emitted by the step that consumes the latent at `latent_index`."""
        return self.frame_count(latent_index + 1) - self.frame_count(latent_index)
