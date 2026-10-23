# SPDX-License-Identifier: Apache-2.0
"""Six-branch component inputs; execution uses the native serving implementation."""

import torch

from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
    ZingChunkGraphs,
)
from sglang.multimodal_gen.runtime.models.dits.zing_h3_joint import (
    PREFIX_PHASES as PHASES,
)


def packs_for(sampler, inputs, condition, history, current):
    packs, _, _ = sampler.pack_prompt_prefix(inputs, condition, history)
    packs.extend(current_packs(sampler, current))
    return packs


def current_packs(sampler, current):
    source, target, *_ = current.spans[0]
    vs, _ = (
        sampler.schedules_for(current)
        if hasattr(sampler, "schedules_for")
        else (sampler.video_scheduler, sampler.audio_scheduler)
    )
    return [
        sampler._select_block(
            current.positive, source, phase="source", video_t=1, audio_t=None
        ),
        sampler._select_block(
            current.positive,
            target,
            phase="target",
            video_t=sampler._host_list(vs.timesteps)[0],
            audio_t=None,
        ),
    ]


def joint(sampler, packs, cache):
    # Reject a nonempty cache before entering the failure-cleanup scope.
    if (
        cache.prefix is not None
        or cache.committed
        or cache.active is not None
        or cache.pending_source is not None
    ):
        raise ValueError("The prefix component probe requires a cleared cache")
    packed = ZingChunkGraphs._joint_packed(packs, PHASES)
    try:
        video, audio = sampler._forward(packed, cache, "joint")
    except BaseException:
        cache.clear()
        raise
    return video[-packs[-1]["x"].shape[0] :], audio


def host_positions(cache):
    """Snapshot metadata for accuracy only; Graph may mix host/device chunks."""
    chunks = cache._history()
    return torch.cat([c.positions.detach().cpu() for c in chunks]) if chunks else None
