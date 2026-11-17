# SPDX-License-Identifier: Apache-2.0
"""Numeric allowlist for model telemetry; never retain prompts or raw usage."""

import math

# Fixed role/stage/clock combinations. These also bound Prometheus cardinality;
# never derive stage labels from arbitrary model receipts.
TTFF_HOST_STAGES = {
    "room": frozenset(
        "source_session_to_first_rgb segment_setup_to_first_rgb "
        "segment_setup_to_source_dispatch condition_dequeue_to_first_rgb "
        "room_accept_to_first_rgb room_accept_to_condition_dequeue "
        "condition_dequeue_to_model_send room_accept_to_model_send "
        "sink_idle_wait effective_boundary_wait reference_resolve_store "
        "send_lock_wait model_wire_lock_wait prompt_send_call "
        "prompt_send_to_accepted accepted_to_condition_ready old_clip_input_wait "
        "old_clip_dispatch_to_condition_ready condition_ready_to_resume "
        "resume_to_fresh_source_dispatch source_dispatch_to_first_rgb "
        "resume_to_first_rgb".split()
    ),
    "encoder": frozenset(
        "encoder_queue_wait condition_cache_lookup media_prepare qwen_prepare "
        "qwen_remote_prepare reference_vae source_prepare source_vae_append "
        "source_vae_pop".split()
    ),
    "dit": frozenset(
        "condition_queue_wait condition_admission condition_activate "
        "history_kv_rebuild condition_ready_wait condition_to_source_ready "
        "first_output_compute initial_joint_step0 prompt_joint_step0 "
        "initial_joint_graph resume_joint_graph source_step0_graph "
        "denoise_graph denoise_steps condition_prefix_apply".split()
    ),
    "decoder": frozenset("decoder_queue_wait vae_decode rgb_copy_publish".split()),
}
TTFF_CUDA_STAGES = {
    "room": frozenset(),
    "encoder": frozenset(
        "reference_vae source_prepare source_vae_append source_vae_pop".split()
    ),
    "dit": TTFF_HOST_STAGES["dit"]
    - {"condition_queue_wait", "condition_ready_wait", "condition_to_source_ready"},
    "decoder": frozenset("vae_decode rgb_copy_publish".split()),
}
TTFF_METRIC_STAGES = frozenset(
    f"whitebox_ttff_{phase}_{role}_{clock}_{stage}"
    for phase in ("initial", "prompt_switch")
    for clock, roles in (("host", TTFF_HOST_STAGES), ("cuda", TTFF_CUDA_STAGES))
    for role, stages in roles.items()
    for stage in stages
)
CLI_STARTUP_STAGES = frozenset(
    "whitebox_cli_" + key
    for key in (
        "preparationMs",
        "browserLaunchMs",
        "runtimePrepareMs",
        "browserContextMs",
        "worldLoadMs",
        "environmentInspectionMs",
        "producerStartMs",
    )
)

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
