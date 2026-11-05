# SPDX-License-Identifier: Apache-2.0
"""Launch contract, capacity and admission for the Zing realtime KV pool.

Zing stores K/V as linear rows ``[rows, local heads, head dim]``; a slot is not
tied to a spatial shape. A single slot is sized once from the static memory budget
and any realtime session whose peak row count fits is admitted, so smaller
resolutions reuse the same slot. Every target chunk pairs with a temporally aligned
source chunk (or none), with an independent spatial grid, and with the stereo
audio latents that packing assigns to its time span. Prefix, source, target and
audio share the same token capacity without independent quotas.

Peak occupancy is enumerated through source and target stages. With
``before_source`` the retained window is selected before source; with
``after_step0`` the previous live window survives source and target step zero.
The target occupies the scratch tail until final commits it. Unconditional
CFG forwards run without a cache and are not counted.
"""

import gc
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

# Runtime memory is separate from the persistent KV rows. Keep a floor even
# when a caller sets mem_fraction_static close to one. Admission below also
# accounts for the actual live forward, including prompt replacement.
ZING_MIN_RUNTIME_BYTES = 8 * 1024**3
ZING_FIXED_WORKSPACE_BYTES = 3 * 1024**3
ZING_WORKSPACE_BYTES_PER_FEATURE = 64


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
    """Allocate storage for a realtime session slot."""
    _positive_int("rows", rows)
    shape = (rows, local_heads, head_dim)
    # Fixed-shape attention may read rows past the
    # live length, so every unwritten row must be finite (0 * NaN V is NaN).
    # Raw K must expose mutations to per-chunk RoPE cache signatures, even
    # when allocation runs inside inference_mode.
    with torch.inference_mode(False):
        keys = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]
    return [
        ZingLayerStore(
            k,
            torch.zeros(shape, dtype=dtype, device=device),
            torch.zeros(shape, dtype=dtype, device=device),
        )
        for k in keys
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
    # These are compute geometry limits, independent of token capacity.
    max_target_area: int = 1280 * 736
    max_target_long_edge: int = 1280
    window_chunks: int = 21
    sink_chunks: int = 3
    rope_position_mode: str = "block_relative"
    rope_max_frame_gap: int = 12
    window_eviction_timing: str = "after_step0"
    pool_size: int = field(default=1, init=False)

    def __post_init__(self):
        for name in ("max_target_area", "max_target_long_edge"):
            _positive_int(name, getattr(self, name))
        if self.window_chunks is None:
            raise ValueError("Zing KV pool requires a finite attention window")
        if self.window_eviction_timing not in ("before_source", "after_step0"):
            raise ValueError("Invalid Zing window eviction timing")
        ZingWindowPolicy(self.window_chunks, self.sink_chunks)
        ZingWindowPolicy.validate_rope(self.rope_position_mode, self.rope_max_frame_gap)

    @property
    def window_policy(self):
        return ZingWindowPolicy(self.window_chunks, self.sink_chunks)

    @classmethod
    def from_pipeline_config(
        cls, pipeline_config, *, max_sessions: int, mem_fraction_static=None
    ):
        if mem_fraction_static is None:
            return None
        if max_sessions != 1:
            raise ValueError("Zing KV pool requires one active session per worker")
        names = (
            "window_chunks",
            "sink_chunks",
            "rope_position_mode",
            "rope_max_frame_gap",
            "window_eviction_timing",
        )
        return cls(**{n: getattr(pipeline_config, "zing_kv_pool_" + n) for n in names})


@dataclass(frozen=True)
class ZingComputeLimits:
    """Metadata-only limits, also usable by an Encoder without a local KV pool."""

    canvas_multiples: tuple[int, int]
    max_short_edge: int
    max_target_area: int = ZingKVPoolConfig.max_target_area
    max_target_long_edge: int = ZingKVPoolConfig.max_target_long_edge

    def validate_canvas(self, height, width):
        mh, mw = self.canvas_multiples
        if (
            type(height) is not int
            or type(width) is not int
            or min(height, width) < 1
            or height % mh
            or width % mw
            or min(height, width) > self.max_short_edge
            or max(height, width) > self.max_target_long_edge
            or height * width > self.max_target_area
        ):
            raise _unsupported(
                f"canvas {height}x{width} exceeds compute geometry limits"
            )


def cleaned_free_memory(device):
    """Use driver free bytes after releasing unreachable objects and idle cache."""
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info(device)[0]


def token_capacity(
    *,
    before_load,
    after_load,
    mem_fraction_static,
    num_layers,
    local_heads,
    head_dim,
    element_size,
    additional_runtime_reserve_bytes=0,
):
    """Same static-memory definition as the LLM pool; no cached-byte add-back."""
    if not 0 < mem_fraction_static < 1:
        raise ValueError("mem_fraction_static must be between 0 and 1")
    row_bytes = (
        ZING_KV_POOL_TENSORS_PER_ROW
        * num_layers
        * local_heads
        * head_dim
        * element_size
    )
    _positive_int("row_bytes", row_bytes)
    if additional_runtime_reserve_bytes < 0:
        raise ValueError("additional runtime reserve must be nonnegative")
    budget = math.floor(
        after_load
        - before_load * (1 - mem_fraction_static)
        - additional_runtime_reserve_bytes
    )
    rows = budget // row_bytes
    if rows < 1:
        raise ValueError(
            "Insufficient memory for Zing KV pool after runtime reserve; reduce resident weights or use a larger device"
        )
    return rows, budget


def peak_rows(
    geometry: ZingKVGeometry,
    window: ZingWindowPolicy,
    *,
    prefix_rows: int,
    rows_per_frame: int,
    has_source: bool,
    source_rows_per_frame: int | None = None,
    total_frames: int | None = None,
    has_audio: bool = False,
    audio_length: int | None = None,
    window_eviction_timing: str = "after_step0",
    prompt_prefix_frames: int = 0,
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
    if window_eviction_timing not in ("before_source", "after_step0"):
        raise ValueError(
            "Zing window eviction timing must be before_source or after_step0"
        )
    committed = {}
    plan = geometry.chunk_plan
    current = 0
    if prompt_prefix_frames and (
        total_frames is None or total_frames > prompt_prefix_frames
    ):
        # Rebuilding clean target history commits every prefix chunk without
        # normal eviction. The next source/target must also fit before step0.
        for chunk in plan.chunks(prompt_prefix_frames):
            committed[chunk.chunk_id] = rows_per_frame * (chunk.end - chunk.start)
            current = chunk.chunk_id + 1
    if source_rows_per_frame is not None:
        _positive_int("source_rows_per_frame", source_rows_per_frame)
        if not has_source:
            raise ValueError("Source rows require a source chunk")
    per_frame = rows_per_frame
    if has_source:
        per_frame += (
            rows_per_frame if source_rows_per_frame is None else source_rows_per_frame
        )
    split = int(plan.split_first_latent and plan.lengths[0] > 1)
    steady = current + window.window_chunks + 2 + split
    best, seen, stop = sum(committed.values()), {}, None
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
        retained = {i: n for i, n in committed.items() if window.retains(i, current)}
        old = (
            sum(committed.values())
            if window_eviction_timing == "after_step0"
            else sum(retained.values())
        )
        # Source rows are persistent, target rows occupy the tail until final.
        # Cross-chunk previous-final replaces previous-target scratch, so the
        # union tail reaches this same old-history + source + target peak.
        best = max(best, old + count)
        committed = {**retained, current: count}
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
    window_eviction_timing: str = "after_step0"


@dataclass(frozen=True)
class ZingKVPoolPlan:
    """Physical slot capacity derived once from the launch contract."""

    config: ZingKVPoolConfig
    geometry: ZingKVGeometry
    slot_rows: int
    runtime_budget_bytes: int | None = None
    local_hidden_width: int | None = None
    prefix_hidden_width: int | None = None

    @classmethod
    def build(
        cls,
        config: ZingKVPoolConfig,
        geometry: ZingKVGeometry,
        *,
        token_capacity: int,
        runtime_budget_bytes: int | None = None,
        local_hidden_width: int | None = None,
        prefix_hidden_width: int | None = None,
    ):
        _positive_int("token_capacity", token_capacity)
        widths = (local_hidden_width, prefix_hidden_width)
        if any((runtime_budget_bytes is None) != (width is None) for width in widths):
            raise ValueError(
                "runtime budget and joint/prefix hidden widths must be provided together"
            )
        if runtime_budget_bytes is not None:
            _positive_int("runtime_budget_bytes", runtime_budget_bytes)
            _positive_int("local_hidden_width", local_hidden_width)
            _positive_int("prefix_hidden_width", prefix_hidden_width)
        return cls(
            config,
            geometry,
            token_capacity,
            runtime_budget_bytes,
            local_hidden_width,
            prefix_hidden_width,
        )

    def slot_bytes(self, *, num_layers, local_heads, head_dim, element_size):
        return (
            ZING_KV_POOL_TENSORS_PER_ROW
            * self.slot_rows
            * num_layers
            * local_heads
            * head_dim
            * element_size
        )

    def validate_canvas(self, height, width):
        c, g = self.config, self.geometry
        ZingComputeLimits(
            g.canvas_multiples,
            g.max_short_edge,
            c.max_target_area,
            c.max_target_long_edge,
        ).validate_canvas(height, width)

    def admit(
        self, request: ZingKVRequest, *, prompt_prefix_frames=0, prompt_joint=False
    ) -> int:
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
            request.window_eviction_timing,
        )
        expected = (
            c.window_chunks,
            c.sink_chunks,
            c.rope_position_mode,
            c.rope_max_frame_gap,
            c.window_eviction_timing,
        )
        if policy != expected:
            raise _unsupported(
                f"window/rope policy {policy} differs from pool contract {expected}"
            )
        if request.generates_audio and g.video_t_span is None:
            raise _unsupported("pool geometry has no audio time mapping")
        h, w = request.target_height, request.target_width
        self.validate_canvas(h, w)
        mh, mw = g.canvas_multiples
        source_rows = None
        if request.source_shape is not None:
            shape = request.source_shape
            if len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
                raise _unsupported(f"invalid source canvas {shape}")
            sh, sw = shape
            if (
                sh % mh
                or sw % mw
                or min(sh, sw) > g.max_short_edge
                or max(sh, sw) > c.max_target_long_edge
                or sh * sw > c.max_target_area
            ):
                raise _unsupported(
                    f"source canvas {shape} exceeds pool geometry limits"
                )
            source_rows = g.rows_per_frame(sh, sw)
        if type(request.prefix_rows) is not int or request.prefix_rows < 0:
            raise _unsupported(f"invalid prefix rows {request.prefix_rows}")
        rows = max(
            peak_rows(
                g,
                c.window_policy,
                prefix_rows=request.prefix_rows,
                rows_per_frame=g.rows_per_frame(h, w),
                has_source=request.source_shape is not None,
                source_rows_per_frame=source_rows,
                total_frames=request.total_frames,
                window_eviction_timing=request.window_eviction_timing,
                has_audio=request.generates_audio,
                audio_length=request.audio_length if request.generates_audio else None,
                prompt_prefix_frames=prefix_frames,
            )
            for prefix_frames in {0, prompt_prefix_frames}
        )
        if rows > self.slot_rows:
            raise _unsupported(f"peak rows {rows} exceed slot rows {self.slot_rows}")
        if self.runtime_budget_bytes is not None:
            # Prefix preparation and the joint source/target forward do not
            # scale with retained KV occupancy. A replacement can additionally
            # replay clean target frames. Use the longest chunk for both paths.
            chunk = max(g.chunk_plan.lengths)
            target_rows = g.rows_per_frame(h, w)
            current_rows = chunk * (target_rows + (source_rows or 0))
            replay_rows = prompt_prefix_frames * target_rows
            forward_rows = max(
                request.prefix_rows
                + (
                    current_rows + replay_rows
                    if prompt_joint
                    else max(current_rows, replay_rows)
                ),
                # Cross-chunk joint execution also includes the previous final.
                chunk * (2 * target_rows + (source_rows or 0)),
            )
            workspace = (
                ZING_FIXED_WORKSPACE_BYTES
                + ZING_WORKSPACE_BYTES_PER_FEATURE
                * max(
                    self.local_hidden_width * forward_rows,
                    # Stable prefix prefill/rebuild replicates every row and
                    # the full hidden width on each SP rank. Only stored KV
                    # heads are sharded; the prefix FFN workspace is not.
                    self.prefix_hidden_width * request.prefix_rows,
                )
            )
            if workspace > self.runtime_budget_bytes:
                raise _unsupported(
                    f"forward workspace {workspace} bytes exceeds runtime budget "
                    f"{self.runtime_budget_bytes} bytes for {forward_rows} rows "
                    f"(joint width={self.local_hidden_width}, "
                    f"prefix rows={request.prefix_rows}, "
                    f"prefix width={self.prefix_hidden_width})"
                )
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
    """Startup-allocated realtime session slots, without a resident donor."""

    def __init__(self, plan, slots, *, agree=_sp_agree):
        self.plan = plan
        self._pool = RealtimeKVCachePool(
            [
                RealtimeKVCachePoolSlot(slot_id=i, kv_cache=stores, crossattn_cache=[])
                for i, stores in enumerate(slots)
            ]
        )
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
        return cls(plan, slots, **kwargs)

    @property
    def size(self):
        return self._pool.size

    @property
    def available(self):
        return self._pool.available

    def allocated_bytes(self):
        return allocated_cache_bytes(list(self._pool._slots))

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
