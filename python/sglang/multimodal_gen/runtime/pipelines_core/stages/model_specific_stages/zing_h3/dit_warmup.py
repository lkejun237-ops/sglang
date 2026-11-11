# SPDX-License-Identifier: Apache-2.0
"""One local request through the loaded DiT's ordinary input service."""

import asyncio
import logging
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import fields
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.realtime.worker_parallel import ParallelChannel

from .dit_link import DiTInputService, DiTLinkClient

logger = logging.getLogger(__name__)


class _LocalChannel:
    """Run the existing service synchronously, without external E/D transports."""

    def __init__(self, create_session, parallel):
        self.replies = deque()
        self.service = DiTInputService(
            SimpleNamespace(send=self.replies.append, group=parallel), create_session
        )

    def send(self, message):
        self.service.queue.append(message)
        asyncio.run(self.service.queue.run_once())

    def receive(self):
        return self.replies.popleft()

    def close(self):
        if self.service.worker is not None:
            self.service.worker.close()
        self.service.worker = self.service.owner = None
        self.service.queue.clear()
        self.replies.clear()


def _mock_inputs(model, codecs, height, width):
    if any(type(n) is not int or n < 1 for n in (height, width)) or any(
        n % codecs.canvas_multiple for n in (height, width)
    ):
        raise ValueError("DiT warmup requires a positive, latent-aligned canvas")
    device = next(model.parameters()).device
    # A private CPU generator gives every SP rank identical inputs without
    # changing the random state of subsequent real requests.
    generator = torch.Generator(device="cpu").manual_seed(0)
    condition = dict(
        context=torch.randn(32, model.arch.text_dim, generator=generator).to(
            device=device, dtype=torch.bfloat16
        ),
        row_role=torch.zeros(32, dtype=torch.long, device=device),
    )
    shape = (
        1,
        codecs.va.latent_channels,
        1,
        height // codecs.spatial_ratio,
        width // codecs.spatial_ratio,
    )
    mean = torch.randn(shape, generator=generator)
    # A bounded posterior with deterministic sampling through the real codec.
    moments = torch.cat((mean, torch.zeros_like(mean)), dim=1).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return condition, moments


@torch.inference_mode()
def warmup_dit(model, codecs, create_session, *, parallel=None, height=480, width=832):
    """Replay one T=1 source and EOS on every SP rank, then release its session."""
    started = time.monotonic()
    consensus = ParallelChannel(parallel)
    error = tensors = None
    try:
        tensors = _mock_inputs(model, codecs, height, width)
    except Exception as exc:
        error = exc
    # Do not let a peer enter preparation/model collectives after a local
    # allocation failure on another rank.
    consensus.check_errors(error)
    condition, moments = tensors
    request = dict(
        messages=[
            dict(role="user", content=[dict(type="text", text="A quiet landscape.")]),
            dict(
                role="target",
                content=[
                    dict(
                        type="video", output=dict(frames=1, height=height, width=width)
                    )
                ],
            ),
        ]
    )
    with set_forward_context(current_timestep=0, attn_metadata=None):
        inputs = codecs.prepare(
            request, lambda _: nullcontext(), conditioning=lambda *_: condition
        )
        metadata = {
            field.name: getattr(inputs, field.name)
            for field in fields(inputs)
            if field.name not in {"positive", "source_visual"}
        }
        channel = _LocalChannel(create_session, parallel)
        client = DiTLinkClient(channel)
        try:
            client.open(
                dict(
                    session_id="startup-warmup",
                    generation_id="warmup",
                    seed=0,
                    width=width,
                    height=height,
                    fps=int(codecs.target_fps),
                )
            )
            replies = client.send(
                "conditioning",
                tensors=condition,
                inputs=metadata,
                event_id=1,
                prompt_version=1,
            )
            if any(item["type"] == "conditioning.rejected" for item in replies):
                raise RuntimeError("DiT warmup conditioning was rejected")
            replies.extend(
                client.send(
                    "source.moments",
                    tensor=moments,
                    latent_start=0,
                    latent_end=1,
                    source_dependency_end=1,
                    source_records=[[1, 0, 1]],
                    final_source_frames=1,
                )
            )
            replies.extend(client.send("source.end", latent_frames=1))
            client.release("startup-warmup")
            error = None
            try:
                outputs = [item for item in replies if item["type"] == "dit.latents"]
                completed = [
                    item for item in replies if item["type"] == "session.completed"
                ]
                expected = (
                    1,
                    model.arch.latents_dim,
                    1,
                    height // codecs.spatial_ratio,
                    width // codecs.spatial_ratio,
                )
                if (
                    len(outputs) != 1
                    or len(completed) != 1
                    or completed[0]["latent_frames"] != 1
                    or tuple(outputs[0]["tensor"].shape) != expected
                    or not torch.isfinite(outputs[0]["tensor"]).all().item()
                ):
                    raise RuntimeError("DiT warmup did not complete one finite latent")
                device = next(model.parameters()).device
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            except Exception as exc:
                error = exc
            consensus.check_errors(error)
        finally:
            # EOS closes the request, and cancel clears the service owner.
            # Direct cleanup also covers a failed link that cannot send cancel.
            channel.close()
    logger.info(
        "DiT startup warmup completed: %sx%s T=1 elapsed_s=%.3f",
        width,
        height,
        time.monotonic() - started,
    )
