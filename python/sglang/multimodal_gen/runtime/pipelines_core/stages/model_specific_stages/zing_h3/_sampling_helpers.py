# Copyright 2025 The MiniMax Team and The HuggingFace Team.
# SPDX-License-Identifier: Apache-2.0
"""Packed block selection adapted from seedleap/zing a0f45389."""

from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.runtime.models.dits.zing_h3 import (
    ZingHostLayout,
    host_to_device,
)
from sglang.multimodal_gen.runtime.models.dits.zing_h3_cache import tensor_identity
from sglang.multimodal_gen.runtime.utils.profiler import maybe_record_function
from sglang.multimodal_gen.runtime.utils.zing_diagnostics import current_diagnostic

from .packing import H3RowRole

_LAYOUT_INDEX = "_zing_layout_index"
_HOST_LAYOUT = "_zing_host_layout"
_HOST_T = "_zing_host_t"
_PREDICT_ROWS = "_zing_predict_rows"
_HOST_GUARD = "_zing_host_guard"
_GUARDED_ROWS = (
    "t",
    "seq_lens",
    "clean_seq_lens",
    "block_idx",
    "position_ids",
    "row_role",
)
_HOST_LAYOUT_KEYS = ("row_role", "block_idx", "position_ids", "seq_lens")


def _guard_host_metadata(packed):
    packed[_HOST_GUARD] = tuple(
        (name, packed[name], tensor_identity(packed[name])) for name in _GUARDED_ROWS
    )


def _host_metadata(packed):
    """Host copies of a selection's rows, unless the caller changed its tensors.

    A replaced or in-place edited metadata tensor makes the forward read the
    device tensors instead. Inference tensors have no version counter, so only
    their replacement is detected.
    """
    host_layout = packed.get(_HOST_LAYOUT)
    if host_layout is None:
        return None
    for name, tensor, identity in packed[_HOST_GUARD]:
        if packed.get(name) is not tensor or tensor_identity(tensor) != identity:
            return None
    return {"host_layout": host_layout, "host_t": packed[_HOST_T]}


def _positions(mask, device):
    return host_to_device(mask.nonzero().squeeze(1), device)


def _visual_roles(roles):
    return (
        (roles == int(H3RowRole.REFERENCE_VISUAL))
        | (roles == int(H3RowRole.SOURCE_VISUAL))
        | (roles == int(H3RowRole.TARGET_VISUAL))
    )


def _audio_roles(roles):
    return (roles == int(H3RowRole.REFERENCE_AUDIO)) | (
        roles == int(H3RowRole.TARGET_AUDIO)
    )


@dataclass(eq=False)
class _BlockSelection:
    """One block phase's rows: host facts plus reusable device tensors."""

    visual_source: torch.Tensor
    audio_source: torch.Tensor
    context_source: torch.Tensor
    device_rows: dict
    # Host masks that build each step's t.
    visual_rows: torch.Tensor
    audio_rows: torch.Tensor
    target_rows: torch.Tensor
    host_layouts: dict
    # Host masks of target video/audio rows within the compact outputs.
    predict_video: torch.Tensor
    predict_audio: torch.Tensor | None
    predict_rows: dict = field(default_factory=dict)

    def prediction_rows(self, device):
        """Device indices of the target rows, built once per output device."""
        rows = self.predict_rows.get(device)
        if rows is None:
            rows = (
                _positions(self.predict_video, device),
                None
                if self.predict_audio is None
                else _positions(self.predict_audio, device),
            )
            self.predict_rows[device] = rows
        return rows


@dataclass(eq=False)
class _LayoutIndex:
    """Host copies and cached row selections of one packed layout.

    Packing fixes a layout's rows; while blocks are sampled only its clean
    latents change. One device-to-host copy per layout lets every block
    selection, latent write and prediction split use host facts and cached
    device index tensors, so a denoising step reads nothing back.
    """

    signature: tuple
    host: dict
    selections: dict = field(default_factory=dict)
    latent_targets: dict = field(default_factory=dict)


class ZingBlockSamplingHelpers:
    causal = True

    def _layout_index(self, layout: dict) -> _LayoutIndex:
        signature = tuple(tensor_identity(layout[key]) for key in _HOST_LAYOUT_KEYS)
        index = layout.get(_LAYOUT_INDEX)
        if index is None or index.signature != signature:
            index = _LayoutIndex(
                signature, {key: layout[key].cpu() for key in _HOST_LAYOUT_KEYS}
            )
            layout[_LAYOUT_INDEX] = index
        return index

    def _host_list(self, tensor: torch.Tensor) -> list:
        """Host values of a small schedule tensor, read once per tensor object."""
        cache = self.__dict__.setdefault("_host_lists", {})
        entry = cache.get(id(tensor))
        if (
            entry is None
            or entry[0] is not tensor
            or entry[1] != tensor_identity(tensor)
        ):
            # The entry holds the tensor, so its id cannot be reused meanwhile.
            entry = (tensor, tensor_identity(tensor), tensor.tolist())
            cache[id(tensor)] = entry
        return entry[2]

    def _predict(
        self,
        packed: dict[str, torch.Tensor],
        latent_shape: tuple[int, int, int],
        cache,
        update: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        video_rows, audio_rows = self._forward(packed, cache, update)
        if _host_metadata(packed) is not None:
            video_index, audio_index = packed[_PREDICT_ROWS].prediction_rows(
                video_rows.device
            )
            video = self.processor.unpatchify_visual(
                video_rows.index_select(0, video_index), 1, latent_shape
            )
            audio = (
                None
                if audio_index is None
                else audio_rows.index_select(0, audio_index).reshape(
                    1, 2, -1, self.processor.audio_channels
                )
            )
            return video, audio
        roles = packed["row_role"]
        visual_rows = _visual_roles(roles)
        audio_storage_rows = _audio_roles(roles)
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

    def _predict_joint(self, packs, phases, latent_shape, cache):
        """Concatenate packed storage while keeping branch timesteps/modalities."""
        names = (
            "x",
            "audio_x",
            "context",
            "t",
            "block_idx",
            "position_ids",
            "row_role",
        )
        packed = {name: torch.cat([p[name] for p in packs]) for name in names}
        packed["seq_lens"] = sum(p["seq_lens"] for p in packs)
        packed["clean_seq_lens"] = sum(p["clean_seq_lens"] for p in packs)
        metadata = [_host_metadata(p) for p in packs]
        if all(m is not None for m in metadata):
            packed[_HOST_LAYOUT] = ZingHostLayout(
                **{
                    name: (
                        sum(getattr(m["host_layout"], name) for m in metadata)
                        if name in {"seq_lens", "clean_seq_lens"}
                        else torch.cat(
                            [getattr(m["host_layout"], name) for m in metadata]
                        )
                    )
                    for name in (
                        "seq_lens",
                        "clean_seq_lens",
                        "block_idx",
                        "position_ids",
                        "row_role",
                    )
                }
            )
            packed[_HOST_T] = torch.cat([m["host_t"] for m in metadata])
            _guard_host_metadata(packed)
        from sglang.multimodal_gen.runtime.models.dits.zing_h3_joint import (
            ZingJointUnsupported,
            joint_transaction,
        )

        host = _host_metadata(packed)
        layout = packed if host is None else vars(host["host_layout"])
        unsupported = error = None
        try:
            joint_transaction(
                cache,
                phases,
                layout["position_ids"],
                layout["block_idx"],
                layout["row_role"],
                packed["t"] if host is None else host["host_t"],
                int(layout["clean_seq_lens"].sum()),
            )
        except ZingJointUnsupported as failure:
            unsupported = str(failure)
        except (ValueError, RuntimeError) as failure:
            error = failure
        eligible = self._agree_sp_boundary(
            cache,
            int(layout["block_idx"][-1]),
            phase="joint_precheck",
            cross=phases[0] == "final",
            joint_eligible=unsupported is None and error is None,
            required_rows=int(layout["block_idx"].numel()),
            precheck_error=error is not None,
            local_error=error,
            layout_signature=self._collective_layout_signature(
                packed, int(layout["block_idx"][-1]) - 1, int(layout["block_idx"][-1])
            ),
        )
        if error is not None:
            raise error
        if not eligible:
            reasons = self.last_run_metrics["joint_fallbacks"]
            reason = unsupported or "peer_joint_precheck"
            reasons[reason] = reasons.get(reason, 0) + 1
            return None
        packed["_zing_joint_phases"] = phases
        video, _ = self._forward(packed, cache, "joint")
        kind = (
            "initial_joint"
            if tuple(phases) == ("prefix", "source", "active")
            else "source_first"
            if len(phases) == 2
            else "prompt_joint"
            if len(phases) in (4, 5, 6)
            else "cross_chunk"
        )
        shapes = self.last_run_metrics.setdefault("joint_row_shapes", {}).setdefault(
            kind, {}
        )
        shape = "+".join(str(p["x"].shape[0]) for p in packs)
        shapes[shape] = shapes.get(shape, 0) + 1
        return self.processor.unpatchify_visual(
            video[-packs[-1]["x"].shape[0] :], 1, latent_shape
        ), None

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

    def _commit_clean_prefix(self, layout, block_number, video, cache) -> None:
        """Commit an already-clean target without running a denoising forward."""
        self._set_block_latents(layout, block_number, video, None)
        packed = self._select_block(
            layout,
            block_number,
            phase="target",
            video_t=1.0,
            audio_t=None,
            final=True,
        )
        metadata = _host_metadata(packed)
        validation = (
            None
            if metadata is None
            else (
                metadata["host_layout"].position_ids,
                metadata["host_layout"].block_idx,
            )
        )
        cache.update(
            packed["position_ids"],
            packed["block_idx"],
            "active",
            video.shape[2],
            validation=validation,
        )
        self._forward(packed, cache, "final")

    def _forward(
        self, packed: dict[str, torch.Tensor], cache, update: str | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = getattr(self, "_forward_counts", None)
        if counts is not None:
            kind = {
                None: "unconditional",
                "active": "denoise",
                "final": "commit",
            }.get(update, update)
            if update == "joint":
                kind = (
                    "initial_joint"
                    if tuple(packed["_zing_joint_phases"])
                    == ("prefix", "source", "active")
                    else "source_first"
                    if len(packed["_zing_joint_phases"]) == 2
                    else "prompt_joint"
                    if len(packed["_zing_joint_phases"]) in (4, 5, 6)
                    else "cross_chunk"
                )
            counts[kind] = counts.get(kind, 0) + 1
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
        kwargs = {name: host_to_device(packed[name], self.device) for name in names}
        if cache is not None:
            kwargs.update(cache=cache, self_cache_update=update)
        elif update is not None:
            raise ValueError("H3 cache update requires a cache")
        kwargs.update(_host_metadata(packed) or {})
        if getattr(cache, "_compressed_kv_enabled", False):
            kwargs["sparse_sampler"] = self
        if update == "joint":
            kwargs["joint_phases"] = packed["_zing_joint_phases"]
        with maybe_record_function(f"Zing.forward.{update or 'unconditional'}"):
            diagnostic = current_diagnostic()
            if diagnostic:
                diagnostic.record(
                    "forward_input",
                    update=update,
                    packed={name: kwargs[name] for name in names},
                )
            result = self.generator(**kwargs)
            if diagnostic:
                diagnostic.record(
                    "forward_output", update=update, video=result[0], audio=result[1]
                )
                diagnostic.cache(cache)
            return result

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
        planned = [
            (c.source_block, c.target_block, c.start, c.end)
            for c in self.processor.chunk_plan.chunks(visual_shape[0])
        ]
        if spans != planned:
            raise ValueError("H3 packed chunks do not match the checkpoint chunk plan")
        return planned

    def _audio_indices(
        self, layout: dict[str, torch.Tensor], target_block: int
    ) -> list[int]:
        host = self._layout_index(layout).host
        target = host["row_role"] == int(H3RowRole.TARGET_AUDIO)
        if not bool(target.any()):
            return []  # video-only 输出仍可携带 audio reference，但没有待采样 audio latent
        all_times = host["position_ids"][target, 0]
        block_times = host["position_ids"][
            target & (host["block_idx"] == target_block), 0
        ]
        if not block_times.numel():
            raise ValueError("H3 video_audio target block has no aligned audio rows")
        return (
            (block_times.unique(sorted=True) - all_times.min())
            .round()
            .to(torch.long)
            .tolist()
        )

    def _has_block(self, layout: dict[str, torch.Tensor], block_number: int) -> bool:
        return bool(
            (self._layout_index(layout).host["block_idx"] == block_number).any()
        )

    def _set_block_latents(
        self,
        layout: dict[str, torch.Tensor],
        target_block: int,
        video: torch.Tensor,
        audio: torch.Tensor | None,
    ) -> None:
        index = self._layout_index(layout)
        targets = index.latent_targets.get(target_block)
        if targets is None:
            roles, blocks = index.host["row_role"], index.host["block_idx"]
            in_block = blocks == target_block
            visual = ((roles == int(H3RowRole.TARGET_VISUAL)) & in_block)[
                _visual_roles(roles)
            ]
            audio_rows = ((roles == int(H3RowRole.TARGET_AUDIO)) & in_block)[
                _audio_roles(roles)
            ]
            targets = (
                _positions(visual, layout["clean_visual"].device),
                _positions(audio_rows, layout["clean_audio"].device),
            )
            index.latent_targets[target_block] = targets
        visual_target, audio_target = targets
        video_rows = self.processor.patchify_visual(video).to(layout["clean_visual"])
        if visual_target.numel() != video_rows.shape[0]:
            raise ValueError(
                "H3 current video block does not match processor target rows"
            )
        layout["clean_visual"].index_copy_(0, visual_target, video_rows)
        if audio is None:
            return
        audio_rows = audio[0].reshape(-1, audio.shape[-1]).to(layout["clean_audio"])
        if audio_target.numel() != audio_rows.shape[0]:
            raise ValueError(
                "H3 current audio block does not match processor target rows"
            )
        layout["clean_audio"].index_copy_(0, audio_target, audio_rows)

    def _block_selection(
        self, layout: dict[str, torch.Tensor], block_number: int, phase: str
    ) -> _BlockSelection:
        index = self._layout_index(layout)
        selection = index.selections.get((block_number, phase))
        if selection is None:
            selection = self._new_block_selection(layout, index, block_number, phase)
            index.selections[(block_number, phase)] = selection
        return selection

    def _new_block_selection(self, layout, index, block_number, phase):
        host = index.host
        rows = int(host["seq_lens"].sum().item())
        roles, blocks = host["row_role"], host["block_idx"]
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
            keep = torch.ones(rows, dtype=torch.bool)
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
        selected_roles = roles[keep]
        metadata_device = layout["row_role"].device
        kept = _positions(keep, metadata_device)
        count = int(keep.sum().item())
        seq_lens = torch.tensor([count], dtype=torch.long)
        device_seq_lens = host_to_device(seq_lens, metadata_device)
        device_rows = {
            "position_ids": layout["position_ids"].index_select(0, kept),
            "row_role": layout["row_role"].index_select(0, kept),
            "block_idx": layout["block_idx"].index_select(0, kept),
            "seq_lens": device_seq_lens,
        }
        host_rows = dict(
            seq_lens=seq_lens,
            block_idx=blocks[keep],
            position_ids=host["position_ids"][keep],
            row_role=selected_roles,
        )
        clean_seq_lens = {}
        host_layouts = {}
        for final in (False, True):
            unmarked = self.causal and phase == "target" and not final
            clean_seq_lens[final] = (
                torch.zeros_like(device_seq_lens)
                if unmarked
                else device_seq_lens.clone()
            )
            host_layouts[final] = ZingHostLayout(
                clean_seq_lens=(
                    torch.zeros_like(seq_lens) if unmarked else seq_lens.clone()
                ),
                **host_rows,
            )
        device_rows["clean_seq_lens"] = clean_seq_lens
        target_video = selected_roles == int(H3RowRole.TARGET_VISUAL)
        target_audio = selected_roles == int(H3RowRole.TARGET_AUDIO)
        selected_visual_rows = _visual_roles(selected_roles)
        selected_audio_rows = _audio_roles(selected_roles)
        predict_audio = target_audio[selected_audio_rows]
        return _BlockSelection(
            visual_source=_positions(
                keep[_visual_roles(roles)], layout["clean_visual"].device
            ),
            audio_source=_positions(
                keep[_audio_roles(roles)], layout["clean_audio"].device
            ),
            context_source=_positions(keep[context_rows], layout["context"].device),
            device_rows=device_rows,
            visual_rows=selected_visual_rows,
            audio_rows=selected_audio_rows,
            target_rows=target_video | target_audio,
            host_layouts=host_layouts,
            predict_video=target_video[selected_visual_rows],
            predict_audio=(predict_audio if bool(predict_audio.any()) else None),
        )

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
        selection = self._block_selection(layout, block_number, phase)
        device_rows = selection.device_rows
        roles = selection.host_layouts[final].row_role
        t = self._block_timesteps(roles, video_t=video_t, audio_t=audio_t, final=final)
        packed = {
            "x": layout["clean_visual"].index_select(0, selection.visual_source),
            "audio_x": layout["clean_audio"].index_select(0, selection.audio_source),
            "context": layout["context"].index_select(0, selection.context_source),
            "position_ids": device_rows["position_ids"],
            "row_role": device_rows["row_role"],
            "block_idx": device_rows["block_idx"],
            "seq_lens": device_rows["seq_lens"],
            "clean_seq_lens": device_rows["clean_seq_lens"][final],
            "t": host_to_device(t, device_rows["seq_lens"].device),
            _HOST_LAYOUT: selection.host_layouts[final],
            _HOST_T: t,
            _PREDICT_ROWS: selection,
        }
        _guard_host_metadata(packed)
        return packed

    def _deferred_final(self, cache, recent_visual, layout):
        """Reconstruct the active clean block from the existing prompt history.

        The cache owns its block/position identity. The session's five-frame
        history owns its values; neither needs an additional pending record.
        """
        block, positions = cache.active
        host_positions = cache._active_reference
        if host_positions is None or host_positions.device.type != "cpu":
            raise ValueError("Deferred final requires validated host cache positions")
        frames = int(host_positions[:, 0].unique().numel())
        if recent_visual is None or recent_visual.shape[2] < frames:
            raise ValueError("Deferred final exceeds retained visual history")
        x = self.processor.patchify_visual(recent_visual[:, :, -frames:]).to(
            layout["clean_visual"]
        )
        rows = positions.shape[0]
        if x.shape[0] != rows:
            raise ValueError("Deferred final history does not match active geometry")
        host = ZingHostLayout(
            position_ids=host_positions,
            block_idx=torch.full((rows,), block, dtype=torch.long),
            row_role=torch.full(
                (rows,), int(H3RowRole.TARGET_VISUAL), dtype=torch.long
            ),
            seq_lens=torch.tensor([rows], dtype=torch.long),
            clean_seq_lens=torch.tensor([rows], dtype=torch.long),
        )
        t = torch.ones(rows, dtype=torch.float32)
        # Layout metadata can remain on CPU while visual payloads live on CUDA.
        # Match ordinary _select_block packs before either joint path cats them.
        metadata_device = layout["row_role"].device
        packed = {
            "x": x,
            "audio_x": layout["clean_audio"][:0],
            "context": layout["context"][:0],
            "position_ids": host_to_device(host_positions, metadata_device),
            **{
                name: host_to_device(getattr(host, name), metadata_device)
                for name in ("block_idx", "row_role", "seq_lens", "clean_seq_lens")
            },
            "t": host_to_device(t, metadata_device),
            _HOST_LAYOUT: host,
            _HOST_T: t,
        }
        _guard_host_metadata(packed)
        return packed

    def _block_timesteps(self, roles, *, video_t, audio_t=None, final=False):
        # One implementation for request admission and actual packed forwards.
        t = torch.ones(roles.shape, dtype=torch.float32)
        if not final:
            t[roles == int(H3RowRole.TARGET_VISUAL)] = float(video_t)
            t[roles == int(H3RowRole.TARGET_AUDIO)] = (
                1.0 if audio_t is None else float(audio_t)
            )
        t[(roles == 0) | (roles == 1)] = self.ar_condition_timestep
        t[roles == 4] = self.reference_visual_timestep
        return t
