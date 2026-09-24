# SPDX-License-Identifier: Apache-2.0
"""Launch contract, capacity and admission for the Zing realtime KV pool.

Zing stores K/V as linear rows ``[rows, local heads, head dim]``; a slot is not
tied to a spatial shape. A slot is sized once for the configured upper bounds
and any realtime session whose peak row count fits is admitted, so smaller
resolutions reuse the same slot. Every target chunk pairs with a same-size
source chunk (or none) and, when the session generates audio, with the stereo
audio latents that packing assigns to its time span. Slots always reserve that
target audio. Reference audio lives in the block-zero prefix and is covered by
``max_prefix_rows``.

Peak occupancy is reached while a target block is being denoised: eviction has
already run (``ZingH3KVCache.prepare_chunk`` precedes the source forward), the
current source is committed as pending, and the target occupies the scratch
tail. Unconditional CFG forwards run without a cache and are not counted.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.runtime.realtime.causal_kv_cache_pool import (
    RealtimeKVCachePool,
    RealtimeKVCachePoolSlot,
    allocated_cache_bytes,
)
from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
from sglang.multimodal_gen.runtime.realtime.zing_window import ZingWindowPolicy

# K/V/rotated-K per row: a finite window keeps raw K, V and a rotated-K scratch.
ZING_KV_POOL_TENSORS_PER_ROW = 3


class ZingKVCapacityError(RuntimeError):
    """A pooled cache would exceed its fixed slot; pooled storage never grows."""


@dataclass(slots=True)
class ZingLayerStore:
    """One layer of a slot: raw K, V and the rotated-K attention scratch."""

    k: torch.Tensor
    v: torch.Tensor
    rotated_k: torch.Tensor

    @property
    def rows(self) -> int:
        return int(self.k.shape[0])

    @property
    def layout(self):
        return tuple(self.k.shape[1:]), self.k.dtype, self.k.device


def allocate_layer_stores(
    rows, *, num_layers, local_heads, head_dim, dtype, device
) -> list[ZingLayerStore]:
    """Allocate one slot (or the prefix workspace) before readiness."""
    _positive_int("rows", rows)
    shape = (rows, local_heads, head_dim)
    return [
        ZingLayerStore(
            torch.empty(shape, dtype=dtype, device=device),
            torch.empty(shape, dtype=dtype, device=device),
            torch.empty(shape, dtype=dtype, device=device),
        )
        for _ in range(num_layers)
    ]


class ZingKVPoolAdmissionError(ValueError):
    """Rejected by the startup pool contract.

    ``code`` is a stable ASCII identifier and ``message`` is browser-safe plain
    English. ``detail`` carries technical context for server logs only.
    """

    def __init__(self, code: str, message: str, detail: str):
        # Worker errors cross processes as their first text line, so that line
        # carries only the stable code and the browser-safe message.
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.detail = code, message, detail


def _unsupported(detail: str) -> ZingKVPoolAdmissionError:
    return ZingKVPoolAdmissionError(
        "kv_pool_request_unsupported",
        "This session configuration is not supported by the server.",
        detail,
    )


def _positive_int(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class ZingKVGeometry:
    """Checkpoint grid needed to turn a pixel canvas into packed visual rows."""

    spatial_ratio: int
    patch_size: tuple[int, int, int]
    max_short_edge: int
    chunk_plan: ChunkPlan
    # ZingH3Packer._video_t_span: checkpoint time covered by N latent frames.
    # Audio latent index i sits at time i, so this maps chunks to audio rows.
    video_t_span: Callable[[int], float] | None = field(default=None, compare=False)
    # Period of the per-frame time spans (rope visual_token_frame_spans).
    time_span_period: int = 1

    def __post_init__(self):
        _positive_int("spatial_ratio", self.spatial_ratio)
        _positive_int("max_short_edge", self.max_short_edge)
        if len(self.patch_size) != 3 or any(
            type(p) is not int or p < 1 for p in self.patch_size
        ):
            raise ValueError("patch_size must be three positive integers")
        # Packing lays out one spatial grid per latent frame.
        if self.patch_size[0] != 1:
            raise ValueError("Zing KV pool requires temporal patch size 1")
        _positive_int("time_span_period", self.time_span_period)

    def audio_frames(self, start, end, *, tail, audio_length=None):
        """Audio latents packed with latent frames [start, end).

        Mirrors ZingH3Packer._audio_indices_for_video_span: indices i with
        span(start) <= i < span(end); the final chunk of a finite session also
        absorbs every remaining index. A continuous session has no tail.
        """
        if self.video_t_span is None:
            raise ValueError("Zing KV pool geometry has no audio time mapping")
        low = math.ceil(self.video_t_span(start))
        high = math.ceil(self.video_t_span(end))
        if audio_length is not None:
            high = audio_length if tail else min(high, audio_length)
        return max(0, high - low)

    @property
    def canvas_multiples(self) -> tuple[int, int]:
        return (
            self.spatial_ratio * self.patch_size[1],
            self.spatial_ratio * self.patch_size[2],
        )

    def rows_per_frame(self, height: int, width: int) -> int:
        mh, mw = self.canvas_multiples
        if height % mh or width % mw:
            raise ValueError("Canvas is not aligned to the generator spatial grid")
        return (height // mh) * (width // mw)


@dataclass(frozen=True)
class ZingKVPoolConfig:
    pool_size: int
    max_target_area: int
    max_target_long_edge: int
    max_prefix_rows: int
    window_chunks: int = 21
    sink_chunks: int = 3
    rope_position_mode: str = "block_relative"
    rope_max_frame_gap: int = 12

    def __post_init__(self):
        for name in (
            "pool_size",
            "max_target_area",
            "max_target_long_edge",
            "max_prefix_rows",
        ):
            _positive_int(name, getattr(self, name))
        if self.window_chunks is None:
            raise ValueError("Zing KV pool requires a finite attention window")
        ZingWindowPolicy(self.window_chunks, self.sink_chunks)
        ZingWindowPolicy.validate_rope(self.rope_position_mode, self.rope_max_frame_gap)

    @property
    def window_policy(self) -> ZingWindowPolicy:
        return ZingWindowPolicy(self.window_chunks, self.sink_chunks)

    @classmethod
    def from_pipeline_config(cls, pipeline_config, *, max_sessions: int):
        """Return None when the pool is disabled; reject partial contracts."""
        prefix = "zing_kv_pool_"
        names = (
            "max_target_area",
            "max_target_long_edge",
            "max_prefix_rows",
            "window_chunks",
            "sink_chunks",
            "rope_position_mode",
            "rope_max_frame_gap",
        )
        size = getattr(pipeline_config, prefix + "size", None)
        values = {n: getattr(pipeline_config, prefix + n, None) for n in names}
        if size is None:
            configured = [prefix + n for n in names[:3] if values[n] is not None]
            if configured:
                raise ValueError(
                    f"{', '.join(configured)} requires {prefix}size to be set"
                )
            return None
        missing = [prefix + n for n in names[:3] if values[n] is None]
        if missing:
            raise ValueError(f"Zing KV pool requires {', '.join(missing)}")
        config = cls(
            pool_size=size, **{n: v for n, v in values.items() if v is not None}
        )
        _positive_int("max_sessions", max_sessions)
        if config.pool_size < max_sessions:
            raise ValueError(
                "Zing KV pool must cover every admitted worker session: "
                f"pool_size={config.pool_size} "
                f"realtime_max_sessions_per_worker={max_sessions}"
            )
        return config


def peak_rows(
    geometry: ZingKVGeometry,
    window: ZingWindowPolicy,
    *,
    prefix_rows: int,
    rows_per_frame: int,
    has_source: bool,
    total_frames: int | None = None,
    has_audio: bool = False,
    audio_length: int | None = None,
) -> int:
    """Maximum rows held by one session at any denoising step.

    ``window_chunks`` already includes sinks and the current chunk. Chunk
    lengths follow the ChunkPlan (including a split first latent) and audio
    rows follow the fractional time spans, so rows per chunk are periodic but
    not constant. The maximum is taken over chunk indices until one full
    period has slid through the window, not approximated as window x longest
    chunk. ``audio_length`` bounds a finite session and makes its last chunk
    absorb the remaining audio.
    """
    if window.window_chunks is None:
        raise ValueError("Zing KV pool capacity requires a finite window")
    plan = geometry.chunk_plan
    per_frame = rows_per_frame * (2 if has_source else 1)
    split = int(plan.split_first_latent and plan.lengths[0] > 1)
    steady = window.window_chunks + 2 + split
    rows, best, seen, stop, current = [], 0, {}, None, 0
    while True:
        chunk = plan.chunk(current, total_frames)
        if chunk is None:
            break
        count = per_frame * (chunk.end - chunk.start)
        if has_audio:
            tail = total_frames is not None and chunk.end == total_frames
            # Two stereo channels become two rows per audio latent.
            count += 2 * geometry.audio_frames(
                chunk.start, chunk.end, tail=tail, audio_length=audio_length
            )
        rows.append(count)
        live = sum(rows[i] for i in range(current) if window.retains(i, current))
        best = max(best, live + rows[current])
        # A finite session is enumerated to its end: its last chunk is
        # truncated and absorbs the remaining audio, so it breaks the period.
        if total_frames is None and stop is None and current >= steady:
            signature = ((current - split) % len(plan.lengths),)
            if has_audio:
                span = geometry.video_t_span(chunk.start)
                signature += (
                    chunk.start % geometry.time_span_period,
                    round(span - math.floor(span), 9),
                )
            if signature in seen:
                # Every window after one more full period repeats a seen one.
                stop = current + window.window_chunks + current - seen[signature]
            seen[signature] = current
        if stop is not None and current >= stop:
            break
        current += 1
        if current > 1_000_000:
            raise ValueError("Zing KV pool chunk rows never become periodic")
    return prefix_rows + best


@dataclass(frozen=True)
class ZingKVRequest:
    """Normalized realtime session geometry, known after prefix packing."""

    target_height: int
    target_width: int
    source_shape: tuple[int, int] | None
    prefix_rows: int
    window_chunks: int | None
    sink_chunks: int
    rope_position_mode: str
    rope_max_frame_gap: int
    generates_audio: bool = False
    # Audio latent length of a finite session; None for a continuous one.
    audio_length: int | None = None
    total_frames: int | None = None


@dataclass(frozen=True)
class ZingKVPoolPlan:
    """Physical slot capacity derived once from the launch contract."""

    config: ZingKVPoolConfig
    geometry: ZingKVGeometry
    slot_rows: int

    @classmethod
    def build(cls, config: ZingKVPoolConfig, geometry: ZingKVGeometry):
        mh, mw = geometry.canvas_multiples
        # Every admitted canvas is aligned, so its row count is at most this.
        max_rows_per_frame = config.max_target_area // (mh * mw)
        if max_rows_per_frame < 1:
            raise ValueError("zing_kv_pool_max_target_area is below one grid cell")
        slot_rows = peak_rows(
            geometry,
            config.window_policy,
            prefix_rows=config.max_prefix_rows,
            rows_per_frame=max_rows_per_frame,
            has_source=True,
            # Reserve target audio for every session; it is a small fraction
            # of the visual rows and avoids a second slot class.
            has_audio=geometry.video_t_span is not None,
        )
        return cls(config, geometry, slot_rows)

    def slot_bytes(self, *, num_layers, local_heads, head_dim, element_size):
        return (
            ZING_KV_POOL_TENSORS_PER_ROW
            * self.slot_rows
            * num_layers
            * local_heads
            * head_dim
            * element_size
        )

    def prefix_workspace_bytes(
        self, *, num_layers, local_heads, head_dim, element_size
    ):
        # A prompt update runs the new prefix into a donor cache before the
        # in-place replacement; that donor needs raw K, V and rotated K.
        return (
            ZING_KV_POOL_TENSORS_PER_ROW
            * self.config.max_prefix_rows
            * num_layers
            * local_heads
            * head_dim
            * element_size
        )

    def admit(self, request: ZingKVRequest) -> int:
        """Validate a request against the contract and return its peak rows.

        Only deterministic request inputs are read, so every SP rank reaches
        the same decision.
        """
        c, g = self.config, self.geometry
        policy = (
            request.window_chunks,
            request.sink_chunks,
            request.rope_position_mode,
            request.rope_max_frame_gap,
        )
        expected = (
            c.window_chunks,
            c.sink_chunks,
            c.rope_position_mode,
            c.rope_max_frame_gap,
        )
        if policy != expected:
            raise _unsupported(
                f"window/rope policy {policy} differs from pool contract {expected}"
            )
        if request.generates_audio and g.video_t_span is None:
            raise _unsupported("pool geometry has no audio time mapping")
        h, w = request.target_height, request.target_width
        mh, mw = g.canvas_multiples
        if type(h) is not int or type(w) is not int or h < 1 or w < 1:
            raise _unsupported(f"invalid target canvas {h}x{w}")
        if h % mh or w % mw:
            raise _unsupported(
                f"target canvas {h}x{w} is not aligned to the {mh}x{mw} grid"
            )
        if min(h, w) > g.max_short_edge:
            raise _unsupported(
                f"target short edge {min(h, w)} exceeds {g.max_short_edge}"
            )
        if max(h, w) > c.max_target_long_edge:
            raise _unsupported(
                f"target long edge {max(h, w)} exceeds {c.max_target_long_edge}"
            )
        if h * w > c.max_target_area:
            raise _unsupported(f"target area {h * w} exceeds {c.max_target_area}")
        if request.source_shape is not None and tuple(request.source_shape) != (h, w):
            raise _unsupported(
                f"source canvas {request.source_shape} must equal target {(h, w)}"
            )
        if type(request.prefix_rows) is not int or request.prefix_rows < 0:
            raise _unsupported(f"invalid prefix rows {request.prefix_rows}")
        if request.prefix_rows > c.max_prefix_rows:
            raise _unsupported(
                f"prefix rows {request.prefix_rows} exceed {c.max_prefix_rows}"
            )
        rows = peak_rows(
            g,
            c.window_policy,
            prefix_rows=request.prefix_rows,
            rows_per_frame=g.rows_per_frame(h, w),
            has_source=request.source_shape is not None,
            total_frames=request.total_frames,
            has_audio=request.generates_audio,
            audio_length=request.audio_length if request.generates_audio else None,
        )
        if rows > self.slot_rows:
            # Only a finite session's audio tail can exceed the continuous
            # per-chunk audio the slot reserves.
            raise _unsupported(f"peak rows {rows} exceed slot rows {self.slot_rows}")
        return rows


logger = logging.getLogger(__name__)

# The session attribute holding its single pooled slot lease.
ZING_KV_SESSION_LEASE = "zing_kv_lease"


def _sp_agree(ok: bool) -> bool:
    """Return True only when every sequence-parallel rank reports success.

    Every rank must call this on every acquisition, whatever its local
    outcome, so a local failure can never leave the others in a collective.
    """
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_sp_group,
        get_ulysses_parallel_world_size,
    )

    if not torch.distributed.is_initialized() or get_ulysses_parallel_world_size() == 1:
        return ok
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    torch.distributed.all_reduce(
        flag, op=torch.distributed.ReduceOp.MIN, group=get_sp_group().cpu_group
    )
    return bool(flag.item())


class ZingKVPool:
    """Startup-allocated realtime slots plus one prompt-update prefix workspace.

    The worker executes sessions on one serialized compute stream, so one
    workspace serves every prompt update; it is borrowed only for the donor
    prefix forward and cleared before the next session step.
    """

    def __init__(self, plan, slots, workspace, *, agree=_sp_agree):
        self.plan = plan
        self._pool = RealtimeKVCachePool(
            [
                RealtimeKVCachePoolSlot(slot_id=i, kv_cache=stores, crossattn_cache=[])
                for i, stores in enumerate(slots)
            ]
        )
        self.workspace = workspace
        self._agree = agree

    @classmethod
    def allocate(
        cls, plan, *, num_layers, local_heads, head_dim, dtype, device, **kwargs
    ):
        layout = dict(
            num_layers=num_layers,
            local_heads=local_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        slots = [
            allocate_layer_stores(plan.slot_rows, **layout)
            for _ in range(plan.config.pool_size)
        ]
        workspace = allocate_layer_stores(plan.config.max_prefix_rows, **layout)
        return cls(plan, slots, workspace, **kwargs)

    @property
    def size(self):
        return self._pool.size

    @property
    def available(self):
        return self._pool.available

    def allocated_bytes(self):
        slots = list(self._pool._slots)
        workspace = RealtimeKVCachePoolSlot(-1, self.workspace, [])
        return allocated_cache_bytes(slots), allocated_cache_bytes([workspace])

    def acquire(self, session, request):
        """Admit ``request`` and lease one slot to ``session`` on every rank.

        A session holds at most one unreleased lease. Admission reads only
        deterministic inputs; slot availability is agreed across SP ranks and
        a partial success is rolled back, so all ranks raise together.
        """
        error, lease, rows = None, None, None
        held = getattr(session, ZING_KV_SESSION_LEASE, None)
        try:
            if held is not None and not held.released:
                raise _unsupported("session already holds a Zing KV pool slot")
            rows = self.plan.admit(request)
        except ZingKVPoolAdmissionError as rejected:
            error = rejected
        if error is None and self._pool.available:
            lease = self._pool.acquire()
        if not self._agree(lease is not None):
            if lease is not None:
                lease.release()
            error = error or ZingKVPoolAdmissionError(
                "kv_pool_exhausted",
                "The server is busy. Please try again later.",
                f"no free Zing KV pool slot on at least one rank "
                f"(local available={self._pool.available}, size={self.size})",
            )
            logger.warning("Zing KV pool rejected a session: %s", error.detail)
            raise error
        lease.peak_rows = rows
        setattr(session, ZING_KV_SESSION_LEASE, lease)
        logger.info(
            "ZING_KV_CACHE_POOL_LEASE slot=%d peak_rows=%d slot_rows=%d available=%d",
            lease.slot_id,
            rows,
            self.plan.slot_rows,
            self._pool.available,
        )
        return lease
