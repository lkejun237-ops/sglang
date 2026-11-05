# SPDX-License-Identifier: Apache-2.0
"""Prompt preparation and admission shared by integrated and external inputs."""

from copy import copy

from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import ZingKVPoolAdmissionError
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
_CAPACITY_MESSAGE = "This request is too large. Try fewer images or a shorter prompt."


def prepare_prompt_update(
    codecs,
    inputs,
    prompt,
    command,
    *,
    use_component,
    conditioning,
    prefix_latent_frames=5,
):
    """Return Qwen features and an optional complete reference replacement."""
    if "references" in command:
        if prefix_latent_frames != 5:
            raise ConditionError(
                "CONDITION_INVALID",
                "Reference updates require prompt prefix rebuilding",
            )
        candidate = codecs.prepare_reference_update(
            inputs,
            command["references"],
            prompt,
            use_component,
            conditioning=conditioning,
        )
        return candidate.positive, candidate
    return conditioning(prompt, inputs.condition_media), None


def admit_prompt_update(
    stage,
    sampler,
    inputs,
    condition,
    *,
    source_shape,
    total_frames=None,
    prompt_update=True,
):
    """Check capacity before a prepared condition can replace active KV."""
    if getattr(stage, "kv_pool", None) is None:
        return
    inputs = copy(inputs)
    inputs.source_visual = None
    packer = stage.packer
    _, height, width = inputs.visual_shape
    visual = condition["context"].new_zeros(
        (1, packer.visual_channels, 1, height, width)
    )
    layout = packer.pack_inference_layout(inputs, condition, visual, None)
    request = stage.kv_request(
        sampler,
        inputs,
        layout,
        source_shape=source_shape,
        total_frames=total_frames,
    )
    try:
        from .packing import ZingH3Packer

        stage.kv_pool.plan.admit(
            request,
            prompt_prefix_frames=ZingH3Packer.PROMPT_SWITCH_PREFIX_FRAMES
            if prompt_update
            else 0,
            prompt_joint=getattr(sampler, "prompt_joint", False),
        )
    except ZingKVPoolAdmissionError as exc:
        logger.warning("Prompt update rejected by KV pool: %s", exc.detail)
        raise ConditionError("CONDITION_INVALID", _CAPACITY_MESSAGE) from exc
    return request


def coordinate_prompt_admission(admit):
    """Validate locally, then agree before publishing any candidate state.

    The callback must only prepare/validate a candidate and must not contain
    collectives: every rank must reach the vote even when local validation fails.
    """
    from sglang.multimodal_gen.runtime.conditioning.worker import (
        coordinate_validation_decision,
    )

    error = request = None
    try:
        request = admit()
    except ValueError as exc:
        # ConditionError is also a ValueError. Packing/row_role validation uses
        # plain ValueError and must not bypass the same rank-wide decision.
        logger.warning("Condition admission rejected locally", exc_info=True)
        error = exc
    if not coordinate_validation_decision(error is None):
        raise ConditionError("CONDITION_INVALID", _CAPACITY_MESSAGE) from error
    return request
