# SPDX-License-Identifier: Apache-2.0
"""Lower the H3 public request fields into Zing's causal execution layout."""

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    minimax_h3_validate_canonical_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    minimax_h3_resolve_plan,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    minimax_h3_task_profile,
)


def build_h3_request(params, extra, *, media_fps):
    """Return None for the legacy flat/messages API; never align target frames.

    H3 owns validation, reference routing and spatial policy. Zing owns the
    causal source and exact frame count. media_fps is a timeline, not a rate
    limiter: real-time generation emits each completed block immediately.
    """
    fields = {
        name: extra.get(name, getattr(params, name, None))
        for name in ("task", "conditions", "target")
    }
    if all(value is None for value in fields.values()):
        return None
    if (
        extra.get("zing_request") is not None
        or getattr(params, "zing_request", None) is not None
    ):
        raise ValueError(
            "Use either H3 task/conditions/target or zing_request, not both"
        )
    if params.image_path or params.video_path:
        raise ValueError(
            "Use conditions/source instead of image_path/video_path with H3 request fields"
        )
    if fields["task"] not in (None, "ref2va"):
        raise ValueError("Zing H3 only supports task='ref2va'")
    canonical = minimax_h3_validate_canonical_request(
        prompt=params.prompt,
        task="ref2va",
        conditions=fields["conditions"],
        target=fields["target"],
    )
    if any(c["role"] != "reference" for c in canonical["conditions"]):
        raise ValueError(
            "Zing ref2va supports references; target keyframes are not supported"
        )
    if "duration_seconds" not in canonical["target"]:
        raise ValueError("Zing target.duration_seconds is required")
    plan = minimax_h3_resolve_plan(canonical)
    profile = minimax_h3_task_profile(canonical["task"])
    content = [{"type": "text", "text": canonical["prompt"]}]
    conditions = []
    for index, condition in enumerate(canonical["conditions"]):
        media_id = f"reference-{index}"
        item = {"id": media_id, "type": condition["type"], "uri": condition["uri"]}
        if "start_time_seconds" in condition:
            item["clip"] = {"start": condition["start_time_seconds"]}
        content.append(item)
        rule = profile.rule_for(
            role=condition["role"], condition_type=condition["type"]
        )
        modalities = []
        if rule.visual_tokenizer_encode:
            modalities.append("visual")
        if rule.audio_tokenizer_encode:
            modalities.append("audio")
        conditions.append(
            {
                "content_id": media_id,
                "role": "reference",
                "latent_modalities": modalities,
            }
        )
    target = {
        "role": "target",
        "content": [
            {
                "type": "video_audio" if params.zing_generate_audio else "video",
                "output": {
                    "frames": round(
                        canonical["target"]["duration_seconds"] * media_fps
                    ),
                    "height": plan.shape["height"],
                    "width": plan.shape["width"],
                },
            }
        ],
        "conditions": conditions,
    }
    source = extra.get("source", getattr(params, "source", None))
    if source is not None:
        target["source"] = source
    return {"messages": [{"role": "user", "content": content}, target]}
