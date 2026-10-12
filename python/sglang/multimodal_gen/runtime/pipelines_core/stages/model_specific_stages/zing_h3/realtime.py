# SPDX-License-Identifier: Apache-2.0
"""Worker-owned Zing state; the shared session cache controls its lifetime."""

from contextlib import contextmanager

import torch

from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState


class ZingH3RealtimeState(BaseRealtimeState):
    def __init__(self):
        self.sampler = self.sampling = self.inputs = None
        self.prompt = self.prompt_event = None
        self.failed_update = self.conditioning_notice = None
        self.effective_prompt_version = None
        self.cpu_rng = self.cuda_rng = None
        self.emitted_frames = 0
        self.pending_blocks = []
        self.video_decoder = self.decoder_frames = None
        self.decoded_latents = 0
        self.kv_lease = None

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
        try:
            if self.video_decoder is not None:
                self.video_decoder.close()
        finally:
            try:
                if self.sampling is not None:
                    self.sampling.close()
            finally:
                # Return the slot only after the cache dropped its views.
                if self.kv_lease is not None:
                    self.kv_lease.release()
                    self.kv_lease = None
        self.sampler = self.sampling = self.inputs = None
        self.cpu_rng = self.cuda_rng = None
        self.prompt = self.prompt_event = None
        self.failed_update = self.conditioning_notice = None
        self.effective_prompt_version = None
        self.emitted_frames = 0
        self.pending_blocks = []
        self.video_decoder = self.decoder_frames = None
        self.decoded_latents = 0
