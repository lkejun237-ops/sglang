# SPDX-License-Identifier: Apache-2.0
"""Worker-owned Zing state; the shared session cache controls its lifetime."""

from contextlib import contextmanager

import torch

from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState


class ZingH3RealtimeState(BaseRealtimeState):
    def __init__(self):
        self.sampler = self.sampling = self.inputs = None
        self.prompt = self.prompt_event = self.requested_prompt = None
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
        self.prompt = self.prompt_event = self.requested_prompt = None
        self.failed_update = self.conditioning_notice = None
        self.effective_prompt_version = None
        self.emitted_frames = 0
        self.pending_blocks = []
        self.video_decoder = self.decoder_frames = None
        self.decoded_latents = 0


class ZingH3BlockSession:
    """Feed a finite encoded request through the shared rolling-prefix session.

    Keep the original full noise/output tensor for block decoding. The ready
    session owns only the next block and its last five generated latents.
    """

    def __init__(
        self,
        sampler,
        inputs,
        visual,
        *,
        seed,
        initial_rng_state,
        acquire_storage=None,
        prompt_event_id=None,
        prompt_version=None,
    ):
        from .incremental import SourceReadySession

        self.visual = visual
        self.source = inputs.source_visual
        self.index = 0
        self.prompt_updated = False
        self.spans = []
        rows = max(sampler.processor.chunk_plan.lengths)
        tensors = (visual, self.source)
        budget = sum(
            value[:, :, :rows].numel() * value.element_size()
            for value in tensors
            if value is not None
        )
        self.ready = SourceReadySession(
            sampler,
            inputs,
            seed=seed,
            max_pending_chunks=1,
            max_pending_bytes=budget,
            total_latent_frames=visual.shape[2],
            initial_rng_state=initial_rng_state,
            acquire_storage=(
                None
                if acquire_storage is None
                else lambda positive, ready: acquire_storage(positive)
            ),
            initial_prompt_event_id=prompt_event_id,
            initial_prompt_version=prompt_version,
        )
        self._next_span()

    def _next_span(self):
        start = self.ready.next_latent
        self.spans.append((0, 0, start, start + self.ready.next_chunk_size))

    def update_prompt(self, condition, event_id, prompt_version):
        self.ready.update_prompt(condition, event_id, prompt_version)

    def discard_pending_prompt(self):
        self.ready.discard_pending_prompt()

    def retag_pending_prompt(self, event_id, prompt_version):
        pending = self.ready.pending_prompt
        if pending is not None:
            self.ready.pending_prompt = pending._replace(
                event_id=event_id, prompt_version=prompt_version
            )

    def step(self):
        self.prompt_updated = (
            self.ready.pending_prompt is not None and self.ready.prompt_switch_ready
        )
        start = self.ready.next_latent
        end = start + self.ready.next_chunk_size
        source = None if self.source is None else self.source[:, :, start:end]
        self.ready.append_block(
            source, self.visual[:, :, start:end], final=end == self.visual.shape[2]
        )
        result = self.ready.step()
        self.visual[:, :, start:end].copy_(result.visual)
        self.spans[-1] = (
            result.chunk.source_block,
            result.chunk.target_block,
            start,
            end,
        )
        self.index += 1
        if not self.ready.complete:
            self._next_span()
        return result

    def close(self):
        self.ready.dispose()
        self.visual = self.source = None
        self.spans = []
