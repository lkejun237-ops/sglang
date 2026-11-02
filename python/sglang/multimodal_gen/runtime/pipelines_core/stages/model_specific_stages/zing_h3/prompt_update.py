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
    """Keep the original all-rank decision before any cache is changed."""
    from sglang.multimodal_gen.runtime.conditioning.worker import coordinate_validation

    error = request = None
    try:
        request = admit()
    except ConditionError as exc:
        error = exc
    if not all(coordinate_validation(error is None)):
        raise ConditionError(
            "CONDITION_INVALID", "Reference update exceeds KV capacity"
        ) from error
    return request
