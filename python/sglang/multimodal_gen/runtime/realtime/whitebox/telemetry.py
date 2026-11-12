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
        "encoder_wait_ms",
        "encoder_process_ms",
        "encoder_return_ms",
        "dit_wait_ms",
        "dit_process_ms",
        "dit_return_ms",
        "decoder_wait_ms",
        "decoder_process_ms",
        "decoder_return_ms",
    }
    | {"h3_" + name + "_host_ms" for name in HOST_FIELDS}
)
STAGE_ROLE_NAMES = ("encoder", "dit", "decoder")
STAGE_METRIC_FIELDS = frozenset({"wait_ms", "process_ms", "return_ms"})


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


def sanitize_frame_stage_breakdown(items, *, start_frame=None, num_frames=None):
    """Sanitize per-frame Encoder/DiT/Decoder wait/process/return timings."""
    if not isinstance(items, list) or not items:
        return []
    if num_frames is not None and len(items) != num_frames:
        return []
    cleaned = []
    for idx, entry in enumerate(items):
        if not isinstance(entry, dict):
            return []
        frame_idx = entry.get("output_frame")
        if type(frame_idx) is not int or frame_idx < 0:
            if start_frame is None:
                return []
            frame_idx = start_frame + idx
        elif start_frame is not None and frame_idx != start_frame + idx:
            return []
        row = {"output_frame": frame_idx}
        has_stage = False
        for role in STAGE_ROLE_NAMES:
            stage_vals = numeric_durations(entry.get(role), STAGE_METRIC_FIELDS)
            if stage_vals:
                row[role] = stage_vals
                has_stage = True
        if not has_stage:
            return []
        cleaned.append(row)
    return cleaned


def h3_host_durations(usage):
    if not isinstance(usage, dict):
        return {}
    values = numeric_durations(
        usage.get("zing_source_stage_host_ms"),
        {name + "_host_ms" for name in HOST_FIELDS},
    )
    return {"h3_" + name: value for name, value in values.items()}
