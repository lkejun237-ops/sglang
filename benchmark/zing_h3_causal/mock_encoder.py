# SPDX-License-Identifier: Apache-2.0
"""Mock Encoder producer for trusted local component captures (pickle-bearing).

Sends actual captured moments and Qwen features; it does not load an Encoder or
synthesize tensors. The consumer runs the existing posterior sampler and DiT.
"""

import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
    ZingH3Codecs,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.component_capture import (
    map_tensors,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
    SourceReadySession,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.queued_dit import (
    QueuedDiTSession,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_bridge import (
    SourceChunkBridge,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.streaming_noise import (
    STREAMING_NOISE_VERSION,
)
from sglang.multimodal_gen.runtime.realtime.zing_contracts import Provenance, SourceSpan


def read_encoder_capture(
    directory,
    *,
    session_id,
    generation_id,
    device="cpu",
    prompt_at_latents=(),
    clip_frames=17,
    clip_latents=5,
    temporal_ratio=4,
):
    """Yield queue messages from existing Encoder test-Master output files.

    The Master receipt does not record prompt/source arrival ordering. Supply
    one explicit source-latent boundary per replacement prompt (initial features
    always come first). No prompt or source tensors are synthesized here.
    """
    root = Path(directory)
    receipt = json.loads((root / "summary.json").read_text())
    conditions, chunks = receipt["conditions"], receipt["chunks"]
    boundaries = (0, *(row["latent_end"] for row in chunks))
    if (
        not conditions
        or not chunks
        or len(prompt_at_latents) != len(conditions) - 1
        or any(type(n) is not int or n not in boundaries for n in prompt_at_latents)
        or tuple(sorted(prompt_at_latents)) != tuple(prompt_at_latents)
    ):
        raise ValueError("Captured prompt results require ordered source boundaries")
    # Earlier lifecycle harnesses saved MomentsSink.summary() directly, without
    # the Master's envelope. Their final dependency is the accepted frame count.
    source_frames = receipt.get("source_frames", chunks[-1]["source_dependency_end"])
    identity = dict(session_id=session_id, generation_id=generation_id)

    def tensor(payload, spec):
        if (
            len(payload) != spec["bytes"]
            or hashlib.sha256(payload).hexdigest() != spec["sha256"]
        ):
            raise ValueError("Encoder capture payload hash does not match")
        value = torch.frombuffer(
            bytearray(payload), dtype=getattr(torch, spec["dtype"])
        )
        return value.reshape(spec["shape"]).to(device)

    def condition(row):
        return dict(
            type="conditioning",
            **identity,
            tensors={
                name: tensor((root / spec["file"]).read_bytes(), spec)
                for name, spec in row["tensors"].items()
            },
            event_id=row["event_id"],
            prompt_version=row.get("prompt_version"),
        )

    yield condition(conditions[0])
    updates = iter(zip(prompt_at_latents, conditions[1:]))
    pending = next(updates, None)
    cursor = 0
    with (root / "output.moments").open("rb") as stream:
        for row in chunks:
            while pending is not None and pending[0] == cursor:
                yield condition(pending[1])
                pending = next(updates, None)
            if row["latent_start"] != cursor:
                raise ValueError("Encoder capture source ranges are not contiguous")
            stream.seek(row["byte_offset"])
            value = tensor(stream.read(row["bytes"]), row)
            cursor = row["latent_end"]
            clip, offset = divmod(cursor - 1, clip_latents)
            yield dict(
                type="source.moments",
                **identity,
                tensor=value,
                latent_start=row["latent_start"],
                latent_end=cursor,
                pixel_end=min(
                    source_frames,
                    clip * clip_frames + 1 + offset * temporal_ratio,
                ),
                source_dependency_end=row["source_dependency_end"],
            )
    while pending is not None and pending[0] == cursor:
        yield condition(pending[1])
        pending = next(updates, None)
    if pending is not None or cursor != receipt["latent_frames"]:
        raise ValueError("Encoder capture ended before all queued inputs")
    yield dict(type="source.end", latent_frames=cursor, **identity)


async def replay_queued_encoder(capture, session, frozen, new_sampler, pipeline_config):
    """Reuse the original replay loader, model, sampler and captured prefix."""
    capture = Path(capture)
    inputs = session["inputs"]
    device = inputs.positive["context"].device
    codecs = ZingH3Codecs({}, pipeline_config, session["runtime"])
    sampler = new_sampler()
    results = []
    spans = [
        span for row in frozen for span in row["chunk"].provenance.source_alignment
    ]

    def provenance(pixels, dependency):
        selected = []
        for span in spans:
            start = max(pixels.start, span.output_frame_start)
            end = min(pixels.end, span.output_frame_start + span.num_frames)
            if start < end:
                selected.append(SourceSpan(span.source_id, start, start, end - start))
        return Provenance(pixels, tuple(selected), dependency)

    bridge = SourceChunkBridge(
        sampler.processor.chunk_plan,
        clip_frames=codecs.vae_frames_per_chunk,
        clip_latents=codecs.vae_latents_per_chunk,
        temporal_ratio=codecs.temporal_ratio,
        capacity_latents=max(sampler.processor.chunk_plan.lengths)
        + codecs.vae_latents_per_chunk,
        capacity_bytes=256 << 20,
        seed=session["seed"],
        provenance_for=provenance,
    )

    def create(condition, event, version):
        return SourceReadySession(
            sampler,
            replace(inputs, positive=condition),
            seed=session["seed"],
            max_pending_chunks=1,
            max_pending_bytes=256 << 20,
            initial_prompt_event_id=event,
            initial_prompt_version=version,
        )

    def sample(moments, *, latent_start):
        return codecs.sample_streaming_visual_moments(
            moments, latent_start=latent_start, noise_version=STREAMING_NOISE_VERSION
        )

    def validate(condition):
        _, height, width = inputs.visual_shape
        visual = condition["context"].new_zeros(
            (1, sampler.processor.visual_channels, 1, height, width)
        )
        sampler.processor.pack_inference_layout(inputs, condition, visual, None)

    async def output(result):
        results.append(result)

    identity = dict(session_id="mock-encoder", generation_id="replay")
    worker = QueuedDiTSession(
        **identity,
        create_session=create,
        bridge=bridge,
        sample_moments=sample,
        validate_condition=validate,
        on_output=output,
    )
    consumer = asyncio.create_task(worker.run())
    try:
        await worker.put(dict(type="conditioning", tensors=inputs.positive, **identity))
        await asyncio.sleep(0)
        files = sorted(capture.glob("source-*.pt"))
        if not files:
            raise ValueError("Mock Encoder requires captured source moments")
        total_frames = frozen[-1]["chunk"].provenance.pixels.end
        for path in files:
            encoded = torch.load(path, map_location="cpu", weights_only=False)
            encoded = map_tensors(encoded, lambda t: t.to(device))
            end = encoded["end"]
            clip, offset = divmod(end - 1, codecs.vae_latents_per_chunk)
            pixel_end = min(
                total_frames,
                clip * codecs.vae_frames_per_chunk + 1 + offset * codecs.temporal_ratio,
            )
            await worker.put(
                dict(
                    type="source.moments",
                    tensor=encoded["moments"],
                    latent_start=encoded["start"],
                    latent_end=end,
                    pixel_end=pixel_end,
                    source_dependency_end=encoded["source_dependency_end"],
                    **identity,
                )
            )
            await asyncio.sleep(0)
        await worker.put(dict(type="source.end", latent_frames=end, **identity))
        await consumer
        if not worker.complete:
            raise RuntimeError("Queued DiT replay did not complete")
        return results
    finally:
        if not consumer.done():
            consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        worker.close()
