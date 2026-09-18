# Copyright 2025 The MiniMax Team and The HuggingFace Team.
# SPDX-License-Identifier: Apache-2.0
"""Packed block selection adapted from seedleap/zing a0f45389."""

import torch

from .packing import H3RowRole


class ZingBlockSamplingHelpers:
    causal = True

    def _predict(
        self,
        packed: dict[str, torch.Tensor],
        latent_shape: tuple[int, int, int],
        cache,
        update: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        video_rows, audio_rows = self._forward(packed, cache, update)
        roles = packed["row_role"]
        visual_rows = (
            (roles == int(H3RowRole.REFERENCE_VISUAL))
            | (roles == int(H3RowRole.SOURCE_VISUAL))
            | (roles == int(H3RowRole.TARGET_VISUAL))
        )
        audio_storage_rows = (roles == int(H3RowRole.REFERENCE_AUDIO)) | (
            roles == int(H3RowRole.TARGET_AUDIO)
        )
        video_mask = (roles == int(H3RowRole.TARGET_VISUAL))[visual_rows].to(
            self.device
        )
        video = self.processor.unpatchify_visual(
            video_rows[video_mask], 1, latent_shape
        )
        audio_mask = (roles == int(H3RowRole.TARGET_AUDIO))[audio_storage_rows].to(
            self.device
        )
        audio = (
            audio_rows[audio_mask].reshape(1, 2, -1, self.processor.audio_channels)
            if bool(audio_mask.any())
            else None
        )
        return video, audio

    def _commit(self, layout, block_number, video, audio, cache) -> None:
        self._set_block_latents(layout, block_number, video, audio)
        packed = self._select_block(
            layout,
            block_number,
            phase="target",
            video_t=1.0,
            audio_t=1.0 if audio is not None else None,
            final=True,
        )
        self._forward(packed, cache, "final")

    def _forward(
        self, packed: dict[str, torch.Tensor], cache, update: str | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        names = (
            "x",
            "audio_x",
            "context",
            "t",
            "seq_lens",
            "clean_seq_lens",
            "block_idx",
            "position_ids",
            "row_role",
        )
        kwargs = {name: packed[name].to(self.device) for name in names}
        if cache is not None:
            kwargs.update(cache=cache, self_cache_update=update)
        elif update is not None:
            raise ValueError("H3 cache update requires a cache")
        return self.generator(**kwargs)

    def _block_spans(
        self, layout: dict[str, torch.Tensor], visual_shape: tuple[int, int, int]
    ) -> list[tuple[int, int, int, int]]:
        if not self.causal:
            return [(1, 2, 0, visual_shape[0])]
        spatial_rows = (
            visual_shape[1]
            // self.processor.patch_size[1]
            * (visual_shape[2] // self.processor.patch_size[2])
        )
        visual_targets = layout["row_role"] == int(H3RowRole.TARGET_VISUAL)
        spans, cursor = [], 0
        target_blocks = layout["block_idx"][visual_targets].unique(sorted=True).tolist()
        if target_blocks != [2 * index + 2 for index in range(len(target_blocks))]:
            raise ValueError(
                "H3 processor target blocks must be consecutive positive even ids"
            )
        for index, target_block in enumerate(target_blocks):
            rows = int(
                (visual_targets & (layout["block_idx"] == target_block)).sum().item()
            )
            if rows % spatial_rows:
                raise ValueError(
                    "H3 processor emitted invalid visual target block geometry"
                )
            frames = rows // spatial_rows
            spans.append((2 * index + 1, target_block, cursor, cursor + frames))
            cursor += frames
        if cursor != visual_shape[0]:
            raise ValueError(
                "H3 processor target block ids do not cover the requested video"
            )
        return spans

    def _audio_indices(
        self, layout: dict[str, torch.Tensor], target_block: int
    ) -> list[int]:
        target = layout["row_role"] == int(H3RowRole.TARGET_AUDIO)
        if not bool(target.any()):
            return []  # video-only 输出仍可携带 audio reference，但没有待采样 audio latent
        all_times = layout["position_ids"][target, 0]
        block_times = layout["position_ids"][
            target & (layout["block_idx"] == target_block), 0
        ]
        if not block_times.numel():
            raise ValueError("H3 video_audio target block has no aligned audio rows")
        return (
            (block_times.unique(sorted=True) - all_times.min())
            .round()
            .to(torch.long)
            .tolist()
        )

    def _set_block_latents(
        self,
        layout: dict[str, torch.Tensor],
        target_block: int,
        video: torch.Tensor,
        audio: torch.Tensor | None,
    ) -> None:
        roles = layout["row_role"]
        visual_rows = (
            (roles == int(H3RowRole.REFERENCE_VISUAL))
            | (roles == int(H3RowRole.SOURCE_VISUAL))
            | (roles == int(H3RowRole.TARGET_VISUAL))
        )
        target = (
            (roles == int(H3RowRole.TARGET_VISUAL))
            & (layout["block_idx"] == target_block)
        )[visual_rows]
        video_rows = self.processor.patchify_visual(video).to(layout["clean_visual"])
        if int(target.sum().item()) != video_rows.shape[0]:
            raise ValueError(
                "H3 current video block does not match processor target rows"
            )
        layout["clean_visual"][target.to(layout["clean_visual"].device)] = video_rows
        if audio is None:
            return
        audio_storage_rows = (roles == int(H3RowRole.REFERENCE_AUDIO)) | (
            roles == int(H3RowRole.TARGET_AUDIO)
        )
        target = (
            (roles == int(H3RowRole.TARGET_AUDIO))
            & (layout["block_idx"] == target_block)
        )[audio_storage_rows]
        audio_rows = audio[0].reshape(-1, audio.shape[-1]).to(layout["clean_audio"])
        if int(target.sum().item()) != audio_rows.shape[0]:
            raise ValueError(
                "H3 current audio block does not match processor target rows"
            )
        layout["clean_audio"][target.to(layout["clean_audio"].device)] = audio_rows

    def _select_block(
        self,
        layout: dict[str, torch.Tensor],
        block_number: int,
        *,
        phase: str,
        video_t: float,
        audio_t: float | None,
        final: bool = False,
    ) -> dict[str, torch.Tensor]:
        rows = int(layout["seq_lens"].sum().item())
        roles, blocks = layout["row_role"], layout["block_idx"]
        if roles.shape != (rows,) or blocks.shape != (rows,):
            raise ValueError(
                "H3 inference layout row_role/block_idx must cover seq_lens"
            )
        if phase not in {"prefix", "source", "target"}:
            raise ValueError("H3 inference phase must be prefix, source, or target")
        target_roles = (roles == int(H3RowRole.TARGET_VISUAL)) | (
            roles == int(H3RowRole.TARGET_AUDIO)
        )
        if not self.causal:
            keep = torch.ones(rows, dtype=torch.bool, device=blocks.device)
        elif phase == "prefix":
            keep = blocks == 0
        elif phase == "source":
            keep = (blocks == block_number) & (roles == int(H3RowRole.SOURCE_VISUAL))
        else:
            keep = (blocks == block_number) & target_roles
        if not bool(keep.any()):
            raise ValueError(f"H3 inference {phase} phase selected no rows")
        context_rows = (roles == int(H3RowRole.CONDITION_TEXT)) | (
            roles == int(H3RowRole.CONDITION_VISUAL)
        )
        visual_rows = (
            (roles == int(H3RowRole.REFERENCE_VISUAL))
            | (roles == int(H3RowRole.SOURCE_VISUAL))
            | (roles == int(H3RowRole.TARGET_VISUAL))
        )
        audio_rows = (roles == int(H3RowRole.REFERENCE_AUDIO)) | (
            roles == int(H3RowRole.TARGET_AUDIO)
        )
        selected_roles = roles[keep]
        selected_context = keep[context_rows]
        selected_visual = keep[visual_rows]
        selected_audio = keep[audio_rows]
        packed = {
            "x": layout["clean_visual"][
                selected_visual.to(layout["clean_visual"].device)
            ],
            "audio_x": layout["clean_audio"][
                selected_audio.to(layout["clean_audio"].device)
            ],
            "context": layout["context"][selected_context.to(layout["context"].device)],
            "position_ids": layout["position_ids"][keep],
            "row_role": selected_roles,
            "block_idx": blocks[keep],
            "seq_lens": torch.tensor(
                [int(keep.sum().item())], dtype=torch.long, device=keep.device
            ),
        }
        target_rows = (selected_roles == int(H3RowRole.TARGET_VISUAL)) | (
            selected_roles == int(H3RowRole.TARGET_AUDIO)
        )
        update_rows = torch.zeros_like(target_rows) if final else target_rows
        packed["clean_seq_lens"] = (
            torch.zeros_like(packed["seq_lens"])
            if self.causal and phase == "target" and not final
            else packed["seq_lens"].clone()
        )
        packed["t"] = torch.ones(
            (int(keep.sum().item()),), dtype=torch.float32, device=keep.device
        )
        selected_visual_rows = (
            (selected_roles == int(H3RowRole.REFERENCE_VISUAL))
            | (selected_roles == int(H3RowRole.SOURCE_VISUAL))
            | (selected_roles == int(H3RowRole.TARGET_VISUAL))
        )
        selected_audio_rows = (selected_roles == int(H3RowRole.REFERENCE_AUDIO)) | (
            selected_roles == int(H3RowRole.TARGET_AUDIO)
        )
        packed["t"][selected_visual_rows] = torch.where(
            update_rows[selected_visual_rows],
            packed["t"].new_full((int(selected_visual_rows.sum()),), float(video_t)),
            packed["t"].new_ones((int(selected_visual_rows.sum()),)),
        )
        if bool(selected_audio_rows.any()):
            audio_values = packed["t"].new_full(
                (int(selected_audio_rows.sum()),),
                1.0 if audio_t is None else float(audio_t),
            )
            packed["t"][selected_audio_rows] = torch.where(
                update_rows[selected_audio_rows],
                audio_values,
                torch.ones_like(audio_values),
            )
        return packed
