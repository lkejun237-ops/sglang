# Copyright 2025 The MiniMax Team and The HuggingFace Team.
# SPDX-License-Identifier: Apache-2.0
"""Inference row geometry adapted from seedleap/zing a0f45389.

Only codec-independent packing is shared with the training reference. Runtime
model loading, device placement, codecs and sampling belong to SGLang.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
import torch


def _patchify_video_latents(
    latents: torch.Tensor, patch_size: tuple[int, int, int]
) -> torch.Tensor:
    """按 T/H/W patch 展开 visual latent，空间遍历顺序与模型 patch embedding 保持一致。"""
    if latents.ndim != 5:
        raise ValueError(
            f"video latents must be [B,C,T,H,W], got {list(latents.shape)}"
        )
    patch_t, patch_h, patch_w = patch_size
    batch, channels, frames, height, width = latents.shape
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            f"video latents {list(latents.shape)} are not divisible by patch {patch_size}"
        )
    rows = latents.reshape(
        batch,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    rows = rows.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(
        -1, channels, math.prod(patch_size)
    )
    return rows.permute(0, 2, 1).contiguous()


def _unpatchify_video_rows(
    rows: torch.Tensor,
    batch_size: int,
    latent_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """严格逆转 ``_patchify_video_latents``，主要供推理阶段从生成 rows 恢复 VAE latent。"""
    if rows.ndim != 3 or rows.shape[1] != math.prod(patch_size):
        raise ValueError("H3 video rows must be [S,patch_volume,C]")
    frames, height, width = latent_shape
    patch_t, patch_h, patch_w = patch_size
    expected = batch_size * frames // patch_t * height // patch_h * width // patch_w
    if (
        frames % patch_t
        or height % patch_h
        or width % patch_w
        or rows.shape[0] != expected
    ):
        raise ValueError("H3 video row count does not match latent shape")
    channels = rows.shape[-1]
    latents = rows.permute(0, 2, 1).reshape(
        batch_size,
        frames // patch_t,
        height // patch_h,
        width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    return (
        latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
        .reshape(batch_size, channels, frames, height, width)
        .contiguous()
    )


class H3RowRole(IntEnum):
    """逻辑 row 的语义标签；数值须与 H3 checkpoint 的 AdaLN/路由约定一致。"""

    CONDITION_TEXT = 0
    CONDITION_VISUAL = 1
    REFERENCE_VISUAL = 4
    REFERENCE_AUDIO = 5
    SOURCE_VISUAL = 7
    TARGET_VISUAL = 10
    TARGET_AUDIO = 11


@dataclass
class H3InferenceInputs:
    """一次生成会话中固定不变的 condition、几何和 clean source。"""

    output_type: str
    conditions: list[dict[str, Any]]
    media: list[dict[str, Any]]  # 独立归一后的 VAE reference 轨道
    positive: dict[str, torch.Tensor]
    visual: dict[str, torch.Tensor]  # reference ID -> visual latent
    audio: dict[str, torch.Tensor]  # reference ID -> audio latent
    source_visual: torch.Tensor | None
    visual_shape: tuple[int, int, int]
    audio_shape: tuple[int, int, int] | None
    pixel_frame_count: int
    sample_count: int | None
    sample_rate: int


class ZingH3Packer:
    def __init__(
        self,
        arch,
        *,
        num_frame_per_block=5,
        split_first_latent=False,
        spatial_extent=32.0,
        frame_scale=5.0 / 3.0,
        visual_token_frame_spans=(1, 4, 4, 4, 4),
    ):
        self.causal = True
        self.patch_size = tuple(arch.patch_size)
        self.visual_channels, self.audio_channels = (
            arch.latents_dim,
            arch.audio_latents_dim,
        )
        self.num_frame_per_block = num_frame_per_block
        self.split_first_latent = split_first_latent
        self.rope_spatial_extent, self.rope_frame_scale = spatial_extent, frame_scale
        self.visual_token_frame_spans = tuple(visual_token_frame_spans)
        if (
            self.patch_size[0] != 1
            or num_frame_per_block < 1
            or not self.visual_token_frame_spans
            or not all(math.isfinite(v) for v in (spatial_extent, frame_scale))
            or min(spatial_extent, frame_scale, *visual_token_frame_spans) <= 0
        ):
            raise ValueError("Invalid Zing block or RoPE geometry")

    def _spatial_grid(self, height: int, width: int) -> torch.Tensor:
        """生成面积归一的二维 RoPE 网格，使不同长宽比共享相近的坐标尺度。"""
        sqrt_area = math.sqrt(height * width)

        def axis(dim: int, patch: int) -> torch.Tensor:
            ratio = dim / sqrt_area
            return torch.from_numpy(
                np.linspace(
                    (1.0 - ratio) / 2.0,
                    (1.0 + ratio) / 2.0,
                    dim // patch,
                    endpoint=False,
                )
                * self.rope_spatial_extent
            ).to(torch.float64)

        hh, ww = torch.meshgrid(
            axis(height, self.patch_size[1]),
            axis(width, self.patch_size[2]),
            indexing="ij",
        )
        return torch.stack((hh.reshape(-1), ww.reshape(-1)), dim=-1)

    def _video_t_grid(
        self, length: int, origin: float, offset: int = 0
    ) -> torch.Tensor:
        # H3 的视觉帧并非恒定占一个时间单位；周期 span 来自 checkpoint rope_config。
        spans = torch.tensor(
            [
                self.rope_frame_scale
                * self.visual_token_frame_spans[
                    index % len(self.visual_token_frame_spans)
                ]
                for index in range(offset + length)
            ],
            dtype=torch.float64,
        )
        starts = torch.cat((torch.zeros(1, dtype=torch.float64), spans.cumsum(0)))
        return origin + starts[offset : offset + length]

    def _video_t_span(self, length: int) -> float:
        return sum(
            self.rope_frame_scale
            * self.visual_token_frame_spans[index % len(self.visual_token_frame_spans)]
            for index in range(length)
        )

    def _audio_indices_for_video_span(
        self, audio_length: int, start: int, end: int, total_frames: int
    ) -> list[int]:
        """把一个 causal visual block 的 H3 时间区间映射到应与其同块出现的 audio 帧。"""
        start_t, end_t = self._video_t_span(start), self._video_t_span(end)
        return [
            index
            for index in range(audio_length)
            if index >= start_t and (index < end_t or end == total_frames)
        ]

    def patchify_visual(self, latents: torch.Tensor) -> torch.Tensor:
        return _patchify_video_latents(latents, self.patch_size)

    def unpatchify_visual(
        self, rows: torch.Tensor, batch_size: int, latent_shape: tuple[int, int, int]
    ) -> torch.Tensor:
        return _unpatchify_video_rows(rows, batch_size, latent_shape, self.patch_size)

    def pack_inference_layout(
        self,
        inputs: H3InferenceInputs,
        condition: dict[str, torch.Tensor] | None,
        visual_target: torch.Tensor,
        audio_target: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """只编译完整 condition/reference/target layout；当前 block 选择和 cache 状态归 pipeline。"""
        if condition is None:
            context = inputs.positive["context"]
            encoded = {
                "context": context.new_empty((0, context.shape[1])),
                "row_role": torch.empty(0, dtype=torch.long, device=context.device),
                "reference_visual": {},
                "reference_audio": {},
                "source_visual": None,
                "target_visual": visual_target,
                "target_audio": audio_target,
            }
            return self._pack_layout([], [], encoded)
        encoded = {
            **condition,
            "reference_visual": inputs.visual,
            "reference_audio": inputs.audio,
            "source_visual": inputs.source_visual,
            "target_visual": visual_target,
            "target_audio": audio_target,
        }
        return self._pack_layout(inputs.conditions, inputs.media, encoded)

    def _pack_layout(
        self,
        conditions: list[dict[str, Any]],
        media_items: list[dict[str, Any]],
        encoded: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """把三类物理存储编译为共享的逻辑 row 顺序；训练与推理均走此唯一实现。"""
        reference_visual, reference_audio = (
            encoded["reference_visual"],
            encoded["reference_audio"],
        )
        source_visual, target_visual, target_audio = (
            encoded["source_visual"],
            encoded["target_visual"],
            encoded["target_audio"],
        )
        self._validate_visual(target_visual)
        if target_audio is not None:
            self._validate_audio(target_audio)
        context, context_roles = encoded["context"], encoded["row_role"]
        if (
            context.ndim != 2
            or context_roles.ndim != 1
            or context.shape[0] != context_roles.shape[0]
            or not torch.all(
                (context_roles == int(H3RowRole.CONDITION_TEXT))
                | (context_roles == int(H3RowRole.CONDITION_VISUAL))
            )
        ):
            raise ValueError(
                "H3 context row_role must align with context and contain only condition roles"
            )
        # 官方 H3 对完整 Qwen context（含 vision rows）统一使用 (row_index,0,0)，role 不参与 position 生成。
        condition_positions = torch.zeros(context.shape[0], 3, dtype=torch.float64)
        condition_positions[:, 0] = torch.arange(context.shape[0], dtype=torch.float64)
        segments = [
            self._segment("context", context, condition_positions, context_roles, 0)
        ]
        condition_ids = [condition["content_id"] for condition in conditions]
        if any(condition.get("role") != "reference" for condition in conditions):
            raise ValueError("H3 schema-v4 conditions only support reference")
        if len(set(condition_ids)) != len(condition_ids) or condition_ids != [
            media["id"] for media in media_items if media["id"] in set(condition_ids)
        ]:
            raise ValueError(
                "H3 conditions must be a unique subset of user media in the same order"
            )
        media_by_id = {media["id"]: media for media in media_items}
        # reference 按用户声明顺序推进共同时间原点；音视频 reference 同时存在时取较长跨度。
        target_origin = float(context.shape[0])
        for condition in conditions:
            target_origin = self._append_reference_segments(
                segments,
                media_by_id[condition["content_id"]],
                reference_visual,
                reference_audio,
                target_visual.shape[3:],
                target_origin,
            )
        if source_visual is not None:
            self._validate_visual(source_visual)
            if source_visual.shape[2] != target_visual.shape[2]:
                raise ValueError(
                    "H3 source and target visual latents must have identical temporal length"
                )
        if self.causal:
            # condition/reference 共用 block0；生成区按 source 奇数、target 偶数交错推进。
            total_frames = int(target_visual.shape[2])
            block_boundaries = list(
                range(0, total_frames, self.num_frame_per_block)
            ) + [total_frames]
            if (
                self.split_first_latent
                and min(total_frames, self.num_frame_per_block) > 1
            ):
                block_boundaries.insert(
                    1, 1
                )  # 只拆原首块，保留后续边界；单帧或原块长为 1 时无需再拆
            for block_number, (start, end) in enumerate(
                zip(block_boundaries, block_boundaries[1:])
            ):
                source_block, target_block = 2 * block_number + 1, 2 * block_number + 2
                if source_visual is not None:
                    segments.append(
                        self._visual_segment(
                            source_visual[:, :, start:end],
                            target_origin,
                            H3RowRole.SOURCE_VISUAL,
                            source_block,
                            temporal_offset=start,
                        )
                    )
                if target_audio is not None:
                    # audio 与当前 visual 时间窗同块出现，尾块吸收所有剩余 audio latent。
                    indices = self._audio_indices_for_video_span(
                        target_audio.shape[2], start, end, total_frames
                    )
                    if indices:
                        segments.append(
                            self._audio_segment(
                                target_audio[:, :, indices],
                                target_origin,
                                target_visual.shape[3:],
                                H3RowRole.TARGET_AUDIO,
                                target_block,
                                time_offset=indices[0],
                            )
                        )
                segments.append(
                    self._visual_segment(
                        target_visual[:, :, start:end],
                        target_origin,
                        H3RowRole.TARGET_VISUAL,
                        target_block,
                        temporal_offset=start,
                    )
                )
        else:
            if source_visual is not None:
                segments.append(
                    self._visual_segment(
                        source_visual, target_origin, H3RowRole.SOURCE_VISUAL, 1
                    )
                )
            if target_audio is not None:
                segments.append(
                    self._audio_segment(
                        target_audio,
                        target_origin,
                        target_visual.shape[3:],
                        H3RowRole.TARGET_AUDIO,
                        2,
                    )
                )
            segments.append(
                self._visual_segment(
                    target_visual, target_origin, H3RowRole.TARGET_VISUAL, 2
                )
            )
        return self._finish_segments(segments)

    def _append_reference_segments(
        self,
        segments,
        media,
        reference_visual,
        reference_audio,
        target_visual_shape,
        origin: float,
    ) -> float:
        """追加一个 reference 的可选音/视频轨道，并返回下一段可用的时间 origin。"""
        visual_latent, audio_latent = (
            reference_visual.get(media["id"]),
            reference_audio.get(media["id"]),
        )
        if audio_latent is not None:
            spatial_shape = (
                visual_latent.shape[3:]
                if visual_latent is not None
                else target_visual_shape
            )
            segments.append(
                self._audio_segment(
                    audio_latent, origin, spatial_shape, H3RowRole.REFERENCE_AUDIO, 0
                )
            )
        if visual_latent is not None:
            segments.append(
                self._visual_segment(
                    visual_latent,
                    origin,
                    H3RowRole.REFERENCE_VISUAL,
                    0,
                    fixed_t=origin if media["type"] == "image" else None,
                )
            )
        audio_span = 0.0 if audio_latent is None else float(audio_latent.shape[2])
        visual_span = (
            0.0
            if visual_latent is None
            else (
                1.0
                if media["type"] == "image"
                else self._video_t_span(visual_latent.shape[2])
            )
        )
        return origin + max(audio_span, visual_span)

    def _validate_visual(self, visual: torch.Tensor) -> None:
        if (
            visual.ndim != 5
            or visual.shape[0] != 1
            or visual.shape[1] != self.visual_channels
        ):
            raise ValueError(
                "H3 visual latent must be [1,C,T,H,W] with checkpoint channels"
            )
        if any(visual.shape[index] % self.patch_size[index - 2] for index in (2, 3, 4)):
            raise ValueError(
                "H3 visual latent grid must be divisible by checkpoint patch_size"
            )

    def _validate_audio(self, audio: torch.Tensor) -> None:
        if (
            audio.ndim != 4
            or audio.shape[0] != 1
            or audio.shape[1] != 2
            or audio.shape[-1] != self.audio_channels
        ):
            raise ValueError(
                "H3 audio latent must be [1,2,T,C] with checkpoint channels"
            )

    def _visual_segment(
        self, latent, origin, role, block, fixed_t=None, temporal_offset=0
    ):
        """将 visual latent patchify，并为每个 patch row 配置 ``(t,h,w)``。"""
        self._validate_visual(latent)
        rows = self.patchify_visual(latent)
        temporal, height, width = latent.shape[2:]
        frame = self._spatial_grid(height, width)
        positions = torch.zeros(rows.shape[0], 3, dtype=torch.float64).view(
            temporal, -1, 3
        )
        positions[:, :, 0] = (
            torch.full((temporal,), float(fixed_t), dtype=torch.float64)
            if fixed_t is not None
            else self._video_t_grid(temporal, float(origin), int(temporal_offset))
        )[:, None]
        positions[:, :, 1:] = frame[None]
        return self._segment("visual", rows, positions.view(-1, 3), role, block)

    def _audio_segment(self, latent, origin, visual_shape, role, block, time_offset=0):
        """将双声道 audio latent 展平；左右声道分别锚定视觉 w 轴两端。"""
        self._validate_audio(latent)
        rows = latent[0].reshape(-1, latent.shape[-1])
        length = latent.shape[2]
        positions = torch.zeros(rows.shape[0], 3, dtype=torch.float64)
        positions[:, 0] = (
            float(origin)
            + float(time_offset)
            + torch.arange(length, dtype=torch.float64)
        ).repeat(2)
        w_grid = self._spatial_grid(int(visual_shape[0]), int(visual_shape[1]))[
            :, 1
        ].unique(sorted=True)
        positions[:length, 2], positions[length:, 2] = w_grid[0], w_grid[-1]
        return self._segment("audio", rows, positions, role, block)

    @staticmethod
    def _segment(storage, values, positions, role, block):
        """构造一个尚未合并的逻辑片段；storage 指明 values 最终进入哪个物理张量。"""
        roles = (
            role.detach().cpu().to(torch.long)
            if isinstance(role, torch.Tensor)
            else torch.full((values.shape[0],), int(role), dtype=torch.long)
        )
        if roles.shape != (values.shape[0],):
            raise ValueError("H3 segment row_role must align with values")
        return {
            "storage": storage,
            "values": values,
            "positions": positions,
            "row_role": roles,
            "block": block,
        }

    def _finish_segments(self, segments) -> dict[str, torch.Tensor]:
        """合并片段并交叉校验逻辑 row 数与三类物理存储的计数。"""
        cursor = sum(int(segment["values"].shape[0]) for segment in segments)
        by_storage = {
            name: [segment for segment in segments if segment["storage"] == name]
            for name in ("context", "visual", "audio")
        }
        positions = torch.cat([segment["positions"] for segment in segments])
        roles = torch.cat([segment["row_role"] for segment in segments])
        visual_values = torch.cat(
            [segment["values"] for segment in by_storage["visual"]]
        )
        audio_values = (
            torch.cat([segment["values"] for segment in by_storage["audio"]])
            if by_storage["audio"]
            else visual_values.new_empty((0, self.audio_channels))
        )
        context = torch.cat([segment["values"] for segment in by_storage["context"]])
        if (
            positions.shape != (cursor, 3)
            or roles.shape != (cursor,)
            or not set(roles.tolist()) <= {int(role) for role in H3RowRole}
        ):
            raise ValueError("H3 packed structural tensors are not row-aligned")
        context_roles = torch.tensor(
            [int(H3RowRole.CONDITION_TEXT), int(H3RowRole.CONDITION_VISUAL)],
            dtype=torch.long,
        )
        visual_roles = torch.tensor(
            [
                int(H3RowRole.REFERENCE_VISUAL),
                int(H3RowRole.SOURCE_VISUAL),
                int(H3RowRole.TARGET_VISUAL),
            ],
            dtype=torch.long,
        )
        audio_roles = torch.tensor(
            [int(H3RowRole.REFERENCE_AUDIO), int(H3RowRole.TARGET_AUDIO)],
            dtype=torch.long,
        )
        if (
            context.shape[0] != int(torch.isin(roles, context_roles).sum())
            or visual_values.shape[0] != int(torch.isin(roles, visual_roles).sum())
            or audio_values.shape[0] != int(torch.isin(roles, audio_roles).sum())
        ):
            raise ValueError("H3 packed values do not match row_role storage counts")
        return {
            "clean_visual": visual_values,
            "clean_audio": audio_values,
            "context": context,
            "position_ids": positions,
            "row_role": roles,
            "block_idx": torch.cat(
                [
                    torch.full(
                        (segment["values"].shape[0],),
                        int(segment["block"]),
                        dtype=torch.long,
                    )
                    for segment in segments
                ]
            ),
            "seq_lens": torch.tensor([cursor], dtype=torch.long),
        }
