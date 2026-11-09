# SPDX-License-Identifier: Apache-2.0
"""Native Zing Qwen conditioning without VAE preprocessing or generation."""

import numpy as np

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation import (
    minimax_h3_ref2va_video_presentation,
)

from .contract import ConditionError


def validate_prompt_token_budget(tokenizer, prompt, max_tokens):
    """Count only verbatim prompt tokens before media preprocessing or GPU work."""
    if max_tokens is None:
        return
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ConditionError("CONDITION_INVALID", "Invalid prompt token budget")
    if len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) > max_tokens:
        raise ConditionError(
            "CONDITION_INVALID", "Prompt tokens exceed the encoder budget"
        )


def encode_condition(
    modules,
    runtime,
    prompt,
    media,
    max_rows=None,
    *,
    presentation_fn=None,
    max_prompt_tokens=None,
):
    tokenizer, processor = modules["tokenizer"], modules["processor"]
    validate_prompt_token_budget(tokenizer, prompt, max_prompt_tokens)
    features, labels, images, videos, timestamps = {}, [], [], [], []
    counts = {"image": 0, "video": 0, "audio": 0}
    for item in media:
        kind = item["type"]
        if kind in {"audio", "video_audio"}:
            counts["audio"] += 1
            labels.append(("audio", counts["audio"]))
        if kind == "image":
            counts["image"] += 1
            labels.append(("image", counts["image"]))
            images.append(item["frames"][:, 0].permute(1, 2, 0).numpy())
        elif kind in {"video", "video_audio"}:
            counts["video"] += 1
            labels.append(("video", counts["video"]))
            fps = runtime["condition_encoder_config"]["video_sample_fps"]
            source_fps = float(item["fps"])
            cursor, indices = 0.0, []
            while round(cursor) < item["frames"].shape[1]:
                if not indices or round(cursor) > indices[-1]:
                    indices.append(round(cursor))
                cursor += source_fps / fps
            frames = [item["frames"][:, i].permute(1, 2, 0).numpy() for i in indices]
            # Rounding and deduplication can change the effective sample rate.
            ts = [i / source_fps for i in indices]
            temporal_patch = processor.video_processor.temporal_patch_size
            frames += [frames[-1]] * (-len(frames) % temporal_patch)
            ts += [ts[-1]] * (-len(ts) % temporal_patch)
            videos.append(np.stack(frames))
            timestamps.append(
                [
                    (ts[i] + ts[i + temporal_patch - 1]) / 2
                    for i in range(0, len(ts), temporal_patch)
                ]
            )
    image_counts, video_counts = [], []
    merge = processor.image_processor.merge_size**2
    if images:
        f = processor.image_processor(images=images, return_tensors="pt")
        features.update(
            pixel_values=f["pixel_values"], image_grid_thw=f["image_grid_thw"]
        )
        image_counts = [int(g.prod()) // merge for g in f["image_grid_thw"]]
    if videos:
        f = processor.video_processor(
            videos=videos, do_sample_frames=False, return_tensors="pt"
        )
        features.update(
            pixel_values_videos=f["pixel_values_videos"],
            video_grid_thw=f["video_grid_thw"],
        )
        for grid, ts in zip(f["video_grid_thw"], timestamps):
            if int(grid[0]) != len(ts):
                raise ValueError(
                    "Zing Qwen video timestamps do not match processor tokens"
                )
            video_counts.append([int(grid[1] * grid[2]) // merge] * len(ts))
    ids, tags = (presentation_fn or minimax_h3_ref2va_video_presentation)(
        tokenizer,
        prompt=prompt,
        condition_labels=labels,
        image_token_count=image_counts,
        video_block_token_counts=video_counts,
        video_block_timestamps=timestamps,
    )
    # None keeps the original unbounded local behavior; services pass a limit.
    if max_rows is not None and (
        type(max_rows) is not int or max_rows <= 0 or ids.numel() > max_rows
    ):
        raise ConditionError(
            "CONDITION_INVALID", "Condition token rows exceed the encoder budget"
        )
    context = modules["text_encoder"].encode_ids(ids, **features)
    return {"context": context, "row_role": (tags == 0).long().to(context.device)}
