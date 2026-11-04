# SPDX-License-Identifier: Apache-2.0
"""Prompt preparation and admission shared by integrated and external inputs."""

from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import ZingKVPoolAdmissionError
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


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


def admit_prompt_update(stage, sampler, inputs, condition, *, source_shape=None):
    """Check capacity before a prepared condition can replace active KV."""
    if getattr(stage, "kv_pool", None) is None:
        return
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
        source_shape=(height, width) if source_shape is None else source_shape,
        total_frames=None,
    )
    try:
        stage.kv_pool.plan.admit(request)
    except ZingKVPoolAdmissionError as exc:
        logger.warning("Prompt update rejected by KV pool: %s", exc.detail)
        raise ConditionError("CONDITION_INVALID", exc.message) from exc
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
        raise ConditionError(
            "CONDITION_INVALID", "Condition validation failed on a DiT rank"
        ) from error
    return request
