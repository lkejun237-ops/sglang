# SPDX-License-Identifier: Apache-2.0
"""Worker-owned Zing state; the shared session cache controls its lifetime."""

from contextlib import contextmanager

import torch

from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState


class ZingH3RealtimeState(BaseRealtimeState):
    def __init__(self):
        self.sampler = self.sampling = self.inputs = None
        self.prompt = None
        self.cpu_rng = self.cuda_rng = None
        self.emitted_frames = 0
        self.pending_blocks = []
        self.video_decoder = self.decoder_frames = None
        self.decoded_latents = 0

    @contextmanager
    def random_stream(self, device, seed):
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            if self.cpu_rng is None:
                torch.manual_seed(int(seed))
            else:
                torch.set_rng_state(self.cpu_rng)
                if devices:
                    torch.cuda.set_rng_state(self.cuda_rng, devices[0])
            yield
            self.cpu_rng = torch.get_rng_state()
            if devices:
                self.cuda_rng = torch.cuda.get_rng_state(devices[0])

    def dispose(self):
        if self.video_decoder is not None:
            self.video_decoder.close()
        if self.sampling is not None:
            self.sampling.close()
        self.sampler = self.sampling = self.inputs = None
        self.cpu_rng = self.cuda_rng = None
        self.prompt = None
        self.emitted_frames = 0
        self.pending_blocks = []
        self.video_decoder = self.decoder_frames = None
        self.decoded_latents = 0
