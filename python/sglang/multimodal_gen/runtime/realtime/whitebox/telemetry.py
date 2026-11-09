# SPDX-License-Identifier: Apache-2.0
"""Numeric allowlist for model telemetry; never retain prompts or raw usage."""

import math

HOST_FIELDS = (
    "prepare",
    "qwen_init",
    "prompt",
    "input_copy",
    "encoder",
    "bridge",
    "dit",
    "decoder",
)
CHUNK_FIELDS = frozenset(
    {
        "request_prepare_ms",
        "scheduler_forward_ms",
        "chunk_total_ms",
        "output_pace_ms",
        "transport_encode_ms",
        "transport_write_ms",
        "model_denoise_ms",
        "model_vae_encode_ms",
        "model_vae_decode_ms",
        "model_post_decode_ms",
        "vae_queue_wait_ms",
        "vae_decode_ms",
        "raw_frame_async_enqueue_ms",
    }
    | {"h3_" + name + "_host_ms" for name in HOST_FIELDS}
)


def numeric_durations(values, allowed):
    if not isinstance(values, dict):
        return {}
    return {
        key: round(value, 3)
        for key, value in values.items()
        if key in allowed
        and type(value) in (int, float)
        and math.isfinite(value)
        and 0 <= value < 86_400_000
    }


def h3_host_durations(usage):
    if not isinstance(usage, dict):
        return {}
    values = numeric_durations(
        usage.get("zing_source_stage_host_ms"),
        {name + "_host_ms" for name in HOST_FIELDS},
    )
    return {"h3_" + name: value for name, value in values.items()}
