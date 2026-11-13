# SPDX-License-Identifier: Apache-2.0
"""Progressive encoder contract and real session factory for issue #142.

A factory receives ``codecs, capacity_frames=...`` and returns a session. Shared
read-only weights are allowed; caches, stride phase and cursors are session-owned.
See docs/docs/sglang-diffusion/zing/specs/PROGRESSIVE_ENCODER.md for the acceptance contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from sglang.multimodal_gen.runtime.realtime.zing_component_metrics import (
    component_timer,
)

if TYPE_CHECKING:
    from .streaming_codecs import CodecSlice

PROGRESSIVE_ENCODER_VERSION = "h3-clip17-progressive-v1"


@dataclass(frozen=True)
class EncoderStateUsage:
    # Include queued inputs/outputs and all cache storages, without double counting.
    retained_bytes: int
    # Cumulative input temporal positions evaluated by temporal convolutions,
    # across layers and spatial tiles. Include recomputation, exclude padding.
    temporal_positions: int


class ProgressiveEncoderSession(Protocol):
    contract_version: str
    accepted_frames: int
    emitted_latents: int
    buffered_frames: int
    minimum_credit_frames: int
    ready: bool
    ended: bool
    disposed: bool

    def append(self, pixels: torch.Tensor, *, frame_start: int) -> None: ...

    def pop(self) -> CodecSlice | None: ...

    def finish(self) -> None: ...

    def reset(self) -> None: ...

    def dispose(self) -> None: ...

    def state_usage(self) -> EncoderStateUsage: ...


def expected_progressive_slices(frames: int) -> list[tuple[int, int, int]]:
    """EOS-inclusive (latent_start, latent_end, real pixel watermark) oracle.

    1+2+2 counts output latents, NOT transport pixel batch sizes. Candidate
    causal watermarks are 1,9,17, reset at each legacy clip boundary. EOS alone
    permits right replication and partial final groups, following legacy ceil.
    """
    if type(frames) is not int or frames < 1:
        raise ValueError("Frame count must be a positive integer")
    result = []
    for pixel_start in range(0, frames, 17):
        count = min(17, frames - pixel_start)
        tokens = 1 + (count - 1 + 3) // 4
        for start, stop, dependency in ((0, 1, 1), (1, 3, 9), (3, 5, 17)):
            if start < tokens:
                base = pixel_start // 17 * 5
                result.append(
                    (
                        base + start,
                        base + min(stop, tokens),
                        pixel_start + min(count, dependency),
                    )
                )
    return result


class H3ProgressiveEncoderSession:
    """Pull-based clip17 encoder with bounded per-layer causal state."""

    contract_version = PROGRESSIVE_ENCODER_VERSION
    minimum_credit_frames = 8

    def __init__(self, codecs, *, capacity_frames, cache=None):
        from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.progressive import (
            CausalEncoderCache,
        )

        if type(capacity_frames) is not int or capacity_frames < 8:
            raise ValueError("Progressive capacity must hold eight input frames")
        if (
            codecs.vae_frames_per_chunk,
            codecs.vae_latents_per_chunk,
            codecs.temporal_ratio,
        ) != (17, 5, 4):
            raise ValueError("Progressive encoding requires clip17 and ratio4")
        self.codecs = codecs
        self.capacity_frames = capacity_frames
        self.encode_mode = codecs.runtime["vae_config"]["encode_mode"]
        self.encode_dtype = codecs.runtime["vae_config"]["encode_dtype"]
        self.cudnn_benchmark_limit = codecs.runtime["vae_config"].get(
            "cudnn_benchmark_limit"
        )
        self.cache = (
            cache
            if cache is not None
            else CausalEncoderCache(
                codecs.modules["video_vae"],
                encode_mode=self.encode_mode,
                encode_dtype=self.encode_dtype,
            )
        )
        self.disposed = False
        self.reset()

    def _open(self):
        if self.disposed:
            raise ValueError("Progressive encoder is disposed")

    def reset(self):
        self._open()
        self._pixels = None
        self._signature = None
        self._clip_frames = 0
        self.accepted_frames = self.emitted_latents = self._consumed = 0
        self.ended = False
        self.cache.clear()
        self.cache.temporal_positions = 0

    @property
    def buffered_frames(self):
        return 0 if self._pixels is None else self._pixels.shape[1]

    @property
    def ready(self):
        return not self.disposed and (
            self.buffered_frames >= (1 if self._clip_frames == 0 else 8)
            or (self.ended and self.buffered_frames > 0)
        )

    def append(self, pixels, *, frame_start):
        self._open()
        if self.ended:
            raise ValueError("Progressive encoder has ended")
        if (
            pixels.ndim != 4
            or pixels.shape[0] != 3
            or pixels.shape[1] < 1
            or min(pixels.shape[2:]) < 1
            or pixels.dtype != torch.uint8
        ):
            raise ValueError("Source pixels must be nonempty RGB bytes [3,T,H,W]")
        if type(frame_start) is not int or frame_start != self.accepted_frames:
            raise ValueError("Source frames must be contiguous")
        if self.buffered_frames + pixels.shape[1] > self.capacity_frames:
            raise ValueError("Source pixel capacity exceeded")
        signature = (*pixels.shape[2:], pixels.device)
        if self._signature is not None and signature != self._signature:
            raise ValueError("Source pixel format cannot change")
        joined = (
            pixels.clone()
            if self._pixels is None
            else torch.cat((self._pixels, pixels), 1)
        )
        self._pixels = joined
        self._signature = signature
        self.accepted_frames += pixels.shape[1]

    @torch.no_grad()
    def pop(self):
        from .codecs import _visual_encode_policy
        from .streaming_codecs import CodecSlice

        # EOS is independent of buffered length: a partial tail is ready only
        # after finish(). Agree before any rank can return or enter encoding.
        self.cache.agree(
            (
                "pop",
                self.cache.session_sequence,
                self.disposed,
                self.ended,
                self.accepted_frames,
                self.emitted_latents,
                self.buffered_frames,
                self._clip_frames,
            )
        )
        self._open()
        if not self.ready:
            return None
        # A local preparation failure must rendezvous with peers before their
        # device collective; pending cache state is committed only at the end.
        error = None
        use_fast_uint8 = False
        try:
            needed = 1 if self._clip_frames == 0 else 8
            count = min(needed, self.buffered_frames)
            length = 1 if needed == 1 else (count + 3) // 4
            frames = self._pixels[:, :count]
            model = self.cache.model
            parameter = next(model.encoder.parameters())
            config = self.codecs.runtime["vae_config"]
            if (
                config["encode_mode"],
                config["encode_dtype"],
                config.get("cudnn_benchmark_limit"),
            ) != (
                self.encode_mode,
                self.encode_dtype,
                self.cudnn_benchmark_limit,
            ):
                raise ValueError(
                    "Progressive encoder policy cannot change within a session"
                )
            dtype = getattr(torch, self.encode_dtype)
            if (
                parameter.dtype != dtype
                or next(model.quant_conv.parameters()).dtype != dtype
            ):
                raise ValueError(
                    "Encoder weights must be loaded in the configured dtype"
                )
            can_fast = getattr(self.cache, "can_encode_uint8_fast", None)
            use_fast_uint8 = bool(
                callable(can_fast)
                and can_fast(int(frames.shape[2]), int(frames.shape[3]))
            )
            if not use_fast_uint8:
                if count < needed:
                    frames = torch.cat(
                        (frames, frames[:, -1:].expand(-1, needed - count, -1, -1)), 1
                    )
                pixels = frames[None].to(device=parameter.device, dtype=dtype).div(255)
                mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
                std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        except Exception as exc:
            error = exc
        self.cache.agree("prepared", error=error)
        with (
            component_timer(
                "encoder",
                operation="source",
                frames=count,
                height=frames.shape[-2],
                width=frames.shape[-1],
            ),
            _visual_encode_policy(
                self.encode_mode, self.cudnn_benchmark_limit
            ) as encode_policy,
        ):
            if use_fast_uint8:
                moments, states, work = self.cache.encode_uint8_frames(
                    frames, needed_frames=needed, valid_latents=length
                )
            else:
                moments, states, work = self.cache.encode((pixels - mean) / std)
            self.cache.last_execution["encode_policy"] = dict(
                encode_policy, encode_dtype=self.encode_dtype
            )
        error = None
        try:
            values = moments if use_fast_uint8 else moments[:, :, :length].clone()
            remaining = (
                self._pixels[:, count:].clone()
                if count < self.buffered_frames
                else None
            )
            result = CodecSlice(
                values,
                self.emitted_latents,
                self.emitted_latents + length,
                self._consumed + count,
            )
        except Exception as exc:
            error = exc
        self.cache.agree("slice_ready", error=error)
        # Tensor-free evidence for dependency/index and actual sharding checks.
        self.cache.last_execution.update(
            source_dependency_end=self._consumed + count,
            latent_start=self.emitted_latents,
            latent_end=self.emitted_latents + length,
        )
        self.cache.commit(states, work)
        self._pixels = remaining
        self._consumed += count
        self.emitted_latents += length
        self._clip_frames += needed
        if (
            self._clip_frames == 17
            or count < needed
            or (self.ended and not self.buffered_frames)
        ):
            self.cache.clear()
            self._clip_frames = 0
        return result

    def finish(self):
        self._open()
        if self.ended or not self.accepted_frames:
            raise ValueError(
                "Progressive encoder requires a nonempty unfinished stream"
            )
        self.ended = True
        if not self.buffered_frames:
            self.cache.clear()

    def dispose(self):
        self._pixels = None
        self.cache.clear()
        self.disposed = True

    def state_usage(self):
        tensors = [v[0] for v in self.cache.states.values() if v[0] is not None]
        if self._pixels is not None:
            tensors.append(self._pixels)
        storages = {
            t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
            for t in tensors
        }
        return EncoderStateUsage(sum(storages.values()), self.cache.temporal_positions)


def create_session(codecs, *, capacity_frames):
    return H3ProgressiveEncoderSession(codecs, capacity_frames=capacity_frames)
