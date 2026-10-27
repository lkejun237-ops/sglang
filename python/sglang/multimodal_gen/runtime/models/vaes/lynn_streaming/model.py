# SPDX-License-Identifier: Apache-2.0
"""Inference-only Lynn StreamingDecoder: one latent in, 1 or 4 RGB frames out.

Ported from the Zing reference (seedleap/zing @ 8eda9c22c6, src/h3/vae_decoder/models/student.py).
Parameter names match the delivered checkpoint; training, packed and window paths are not carried over.
"""

import math

import msgspec
import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
    MiniMaxH3VideoViTDecoder3d,
)
from torch import nn

from .attention import ATTENTION_BACKENDS, streaming_block
from .geometry import split_tiles, tile_shard
from .stream import LynnStream
from .temporal import TemporalProtocol


class LynnDecoderConfig(msgspec.Struct, frozen=True):
    """Decoder-section fields of the training config that inference needs."""

    temporal_window: int
    attention_backend: str
    tile_size: int
    tile_overlap: int
    tile_batch_size: int
    pixel_mean: tuple[float, ...]
    pixel_std: tuple[float, ...]


class StreamState(msgspec.Struct):
    latent_index: int = 0
    geometry: tuple | None = None
    # Request-private, flat FP32 storage sized for the largest temporal phase.
    pixel_send: torch.Tensor | None = None
    pixel_recv: torch.Tensor | None = None
    # tile id -> per-layer (K before RoPE, V), each [B, history spatial tokens, heads, head_dim]
    tiles: dict = {}


class StreamingDecoder(nn.Module):
    def __init__(self, config: LynnDecoderConfig, architecture: dict):
        super().__init__()
        self.config = config
        self.architecture = a = architecture
        self.protocol = TemporalProtocol(a)
        self.spatial_ratio = math.prod(a["spatial_downsample_factors"])
        if config.temporal_window < 1 or config.tile_batch_size < 0:
            raise ValueError("invalid temporal window or tile batch size")
        if not 0 <= config.tile_overlap < config.tile_size:
            raise ValueError("invalid tile geometry")
        if (
            config.tile_size % self.spatial_ratio
            or config.tile_overlap % self.spatial_ratio
        ):
            raise ValueError("tile geometry must be latent aligned")
        self.post_quant_conv = nn.Conv3d(a["latent_channels"], a["latent_channels"], 1)
        self.decoder = MiniMaxH3VideoViTDecoder3d(
            in_channels=a["latent_channels"],
            out_channels=a["out_channels"],
            patch_size=self.spatial_ratio,
            patch_size_t=self.protocol.ratio,
            num_layers=a["decoder_num_layers"],
            num_attention_heads=a["decoder_num_attention_heads"],
            attention_head_dim=a["decoder_attention_head_dim"],
            num_register_tokens=a["decoder_num_register_tokens"],
            ffn_mult=a["decoder_ffn_mult"],
            rope_theta=a["decoder_rope_theta"],
            rope_dim_ratio=a["decoder_rope_dim_ratio"],
            norm_eps=a["decoder_norm_eps"],
        )
        width = a["decoder_num_attention_heads"] * a["decoder_attention_head_dim"]
        self.phase_embedding = nn.Parameter(torch.zeros(self.protocol.phases, width))
        # Constants from config, not weights: they are absent from the checkpoint state_dict.
        for name, values in (
            ("latent_mean", a["latents_mean"]),
            ("latent_std", a["latents_std"]),
            ("pixel_mean", config.pixel_mean),
            ("pixel_std", config.pixel_std),
        ):
            self.register_buffer(
                name, torch.tensor(values).view(1, -1, 1, 1, 1), persistent=False
            )
        self.attention = ATTENTION_BACKENDS[config.attention_backend]()

    @torch.no_grad()
    def warmup_preparation(self, device):
        """Compile supported preparation shapes before the worker accepts users."""
        device = torch.device(device)
        a = self.architecture
        if (
            device.type != "cuda"
            or self.compute_dtype != torch.bfloat16
            or self.config.temporal_window != 3
            or self.config.tile_size // self.spatial_ratio != 16
            or a["decoder_num_attention_heads"] != 32
            or a["decoder_attention_head_dim"] != 64
            or a["decoder_num_register_tokens"] != 4
            or a["decoder_rope_dim_ratio"] != 0.75
        ):
            return 0
        from .attention import _supports_preparation_device

        if not _supports_preparation_device(device):
            return 0
        from .fused_preparation import prepare

        # Use the model's real RoPE layout, even while its weights reside on CPU.
        rope_device = self.phase_embedding.device
        count = 0
        for batch in (6, 12, 14, 28, 30):
            q = torch.zeros(batch, 261, 32, 64, device=device, dtype=torch.bfloat16)
            for history in (0, 256, 512):
                past = q.new_zeros(batch, history, 32, 64)
                for dtype in (torch.float32, torch.bfloat16):
                    ropes = [
                        tuple(
                            t.to(device=device, dtype=dtype)
                            for t in self.decoder.rope(
                                torch.zeros(batch, length, 3, device=rope_device)
                            )
                        )
                        for length in (261, history + 261)
                    ]
                    prepare(q, q, q, past, past, *ropes, 256, 512)
                    count += 1
        torch.cuda.synchronize(device)
        return count

    def to_compute(self, device, dtype):
        # Autocast dtype for decode_step; residency must move devices only, never recast.
        self.compute_dtype = dtype
        self.to(device=device)
        self.post_quant_conv.to(dtype=dtype)
        for name, module in self.decoder.named_modules():
            # proj_in, norms and the residual stream stay FP32.
            if isinstance(module, nn.Linear) and name != "proj_in":
                module.to(dtype=dtype)
        self.decoder.register_tokens.data = self.decoder.register_tokens.data.to(dtype)
        return self

    def init_stream(self):
        return StreamState()

    def new_stream(self, *, frame_limit=None, group=None):
        return LynnStream(self, frame_limit=frame_limit, group=group)

    def decode(self, latent, frames, *, group=None):
        return self.new_stream(frame_limit=frames, group=group).decode(latent, frames)

    def _project_pixels(self, hidden, grid, start):
        b, frames = hidden.shape[:2]
        h, w = grid
        ratio, decoder, t_ratio = self.spatial_ratio, self.decoder, self.protocol.ratio
        normed = decoder.norm_out(hidden.float()).to(decoder.proj_out.weight.dtype)
        pixels = decoder.proj_out(normed)[:, :, : h * w]
        pixels = pixels.reshape(
            b, frames, h, w, decoder.out_channels, t_ratio, ratio, ratio
        )
        pixels = pixels.permute(0, 4, 1, 5, 2, 6, 3, 7)
        pixels = pixels.reshape(
            b, decoder.out_channels, frames, t_ratio, h * ratio, w * ratio
        )
        # Phase 0 drops the pre-padding slots; phase follows the absolute latent index.
        parts = [
            pixels[:, :, i, self.protocol.pre_padding :]
            if (start + i) % self.protocol.phases == 0
            else pixels[:, :, i]
            for i in range(frames)
        ]
        return torch.cat(parts, dim=2).float()

    @staticmethod
    def _blend(a, b, extent, dim):
        if extent == 0:
            return b
        shape = [1] * b.ndim
        shape[dim] = extent
        weight = (torch.arange(extent, device=b.device, dtype=b.dtype) / extent).view(
            shape
        )
        tail, front, rest = (
            [slice(None)] * b.ndim,
            [slice(None)] * b.ndim,
            [slice(None)] * b.ndim,
        )
        tail[dim], front[dim], rest[dim] = (
            slice(-extent, None),
            slice(extent),
            slice(extent, None),
        )
        faded = a[tuple(tail)] * (1 - weight) + b[tuple(front)] * weight
        return torch.cat([faded, b[tuple(rest)]], dim=dim)

    def _stitch_tiles(self, rows, oy, ox):
        stitched = []
        for i, row in enumerate(rows):
            parts = []
            for j, tile in enumerate(row):
                if i:
                    tile = self._blend(rows[i - 1][j], tile, oy[i - 1], -2)
                if j:
                    # Reference order: blend with the tile above first, then the one on the left.
                    tile = self._blend(row[j - 1], tile, ox[j - 1], -1)
                if i < len(rows) - 1 and oy[i]:
                    tile = tile[..., : -oy[i], :]
                if j < len(row) - 1 and ox[j]:
                    tile = tile[..., : -ox[j]]
                parts.append(tile)
            stitched.append(torch.cat(parts, dim=-1))
        return torch.cat(stitched, dim=-2) * self.pixel_std + self.pixel_mean

    def _tile_layout(self, z_t):
        cfg, ratio = self.config, self.spatial_ratio
        kwargs = dict(size=cfg.tile_size, overlap=cfg.tile_overlap, ratio=ratio)
        ys, hs, oy = split_tiles(z_t.shape[-2] * ratio, **kwargs)
        xs, ws, ox = split_tiles(z_t.shape[-1] * ratio, **kwargs)
        return ys, hs, oy, xs, ws, ox

    def decode_step(self, z_t, state, group=None):
        """Consume one normalized latent [B, C, 1, H, W]; returns (RGB frames, state).

        With an SP `group` (world size > 1) each rank decodes and keeps K/V only
        for its contiguous tile range, then all ranks gather every tile and
        stitch the full frame locally. Every rank must call this in lockstep.
        Not retryable: a failure part-way leaves `state` half-updated.
        """
        if (
            z_t.ndim != 5
            or z_t.shape[2] != 1
            or z_t.shape[1] != self.architecture["latent_channels"]
        ):
            raise ValueError("decode_step expects normalized [B,24,1,H,W]")
        cfg = self.config
        rank, world_size = (
            (0, 1) if group is None else (group.rank_in_group, group.world_size)
        )
        geometry = (
            tuple(z_t.shape),
            z_t.device,
            z_t.dtype,
            cfg.tile_size,
            cfg.tile_overlap,
            cfg.temporal_window,
            rank,
            world_size,
        )
        if state.geometry is not None and state.geometry != geometry:
            raise ValueError(
                "stream geometry changed; start a new stream for a new video"
            )
        state.geometry = geometry
        phase = state.latent_index % self.protocol.phases
        # The only place the latent is de-normalized.
        z_t = z_t.float() * self.latent_std + self.latent_mean
        ys, hs, oy, xs, ws, ox = self._tile_layout(z_t)
        tile_ids = [(y, x, hp, wp) for y, hp in zip(ys, hs) for x, wp in zip(xs, ws)]
        if world_size == 1:
            decoded = self._decode_tiles(z_t, tile_ids, state, phase)
        else:
            # Allocate before tile decoding, while activation memory is still free.
            max_shape = (
                z_t.shape[0],
                self.decoder.out_channels,
                self.protocol.ratio,
                hs[0],
                ws[0],
            )
            self._init_pixel_workspace(
                state, group, len(tile_ids), max_shape, z_t.device
            )
            start, end = tile_shard(len(tile_ids), rank, world_size)
            try:
                decoded, error = (
                    self._decode_tiles(z_t, tile_ids[start:end], state, phase),
                    None,
                )
            except Exception as exc:
                # Still join the gather so peers fail instead of hanging in it.
                decoded, error = {}, exc
            # Tiles on one axis share a size; phase 0 drops the pre-padding frames.
            count = self.protocol.ratio - (
                self.protocol.pre_padding if phase == 0 else 0
            )
            shape = (z_t.shape[0], self.decoder.out_channels, count, hs[0], ws[0])
            decoded = self._gather_tiles(group, tile_ids, decoded, error, shape, state)
        rows = [
            [decoded[y, x, hp, wp] for x, wp in zip(xs, ws)] for y, hp in zip(ys, hs)
        ]
        frames = self._stitch_tiles(rows, oy, ox)
        state.latent_index += 1
        return frames, state

    @staticmethod
    def _init_pixel_workspace(state, group, tile_count, shape, device):
        """Agree on allocation success before any rank enters the pixel collective.

        This guards synchronous workspace allocation errors, not arbitrary
        asynchronous CUDA failures or a process that has already exited.
        """
        if state.pixel_send is not None:
            return
        # Keep the control flag on CPU so reporting a GPU allocation failure
        # does not itself require a new GPU allocation.
        failed = torch.zeros(1, dtype=torch.int32, device="cpu")
        width = ((tile_count + group.world_size - 1) // group.world_size) * math.prod(
            shape
        ) + 1
        reason = None
        try:
            state.pixel_send = torch.empty(width, dtype=torch.float32, device=device)
            state.pixel_recv = torch.empty(
                group.world_size * width, dtype=torch.float32, device=device
            )
        except Exception as exc:
            # Do not keep an exception/traceback that owns partially allocated
            # tensors alive while peers wait for the allocation consensus.
            reason = str(exc)
            state.pixel_send = state.pixel_recv = None
            failed.fill_(1)
        torch.distributed.all_reduce(
            failed, op=torch.distributed.ReduceOp.MAX, group=group.cpu_group
        )
        if failed.item():
            state.pixel_send = state.pixel_recv = None
            detail = f": {reason}" if reason is not None else " on an SP peer"
            raise RuntimeError(f"Lynn pixel workspace initialization failed{detail}")

    @staticmethod
    def _gather_tiles(group, tile_ids, decoded, error, shape, state):
        """Pack directly into reusable storage and gather only the current phase."""
        world_size, numel = group.world_size, math.prod(shape)
        bounds = [tile_shard(len(tile_ids), r, world_size) for r in range(world_size)]
        width = max(end - start for start, end in bounds) * numel
        local = state.pixel_send[: width + 1]
        local.zero_()
        start, end = bounds[group.rank_in_group]
        if error is not None:
            local[-1] = 1
        else:
            for index, tile in enumerate(tile_ids[start:end]):
                local[index * numel : (index + 1) * numel].view(shape).copy_(
                    decoded[tile]
                )
        # The whole flat prefix is contiguous; slicing each maximal-phase row
        # separately would leave gaps between rows during a shorter phase.
        received = state.pixel_recv[: world_size * (width + 1)]
        group.all_gather_into(received, local)
        gathered = received.view(world_size, width + 1)
        failed = gathered[:, -1].nonzero().flatten().tolist()
        if error is not None:
            local = received = gathered = None
            raise error
        if failed:
            local = received = gathered = None
            raise RuntimeError(f"Lynn tile decode failed on SP rank(s) {failed}")
        result = {}
        for rank, (start, end) in enumerate(bounds):
            tiles = gathered[rank, : (end - start) * numel].view(end - start, *shape)
            result.update(zip(tile_ids[start:end], tiles))
        return result

    def _decode_tiles(self, z_t, tile_ids, state, phase):
        """Decode `tile_ids` of the de-normalized latent; updates their K/V in `state`."""
        cfg, decoder, ratio = self.config, self.decoder, self.spatial_ratio
        batch_size = cfg.tile_batch_size or max(1, len(tile_ids))
        video_batch, decoded = z_t.shape[0], {}
        window = self.protocol.teacher_window
        keep_frames = cfg.temporal_window - 1
        for start in range(0, len(tile_ids), batch_size):
            batch_tiles = tile_ids[start : start + batch_size]
            z = torch.cat(
                [
                    z_t[
                        ...,
                        y // ratio : (y + hp) // ratio,
                        x // ratio : (x + wp) // ratio,
                    ]
                    for y, x, hp, wp in batch_tiles
                ]
            )
            tile_batch, _, _, h, w = z.shape
            z = self.post_quant_conv(z.to(self.post_quant_conv.weight.dtype))
            spatial_tokens = h * w
            gy, gx = torch.meshgrid(
                2 * (torch.arange(h, device=z.device).float() + 0.5) / h - 1,
                2 * (torch.arange(w, device=z.device).float() + 0.5) / w - 1,
                indexing="ij",
            )
            positions = (
                torch.stack([torch.zeros_like(gy), gy, gx], dim=-1)
                .flatten(0, 1)[None]
                .expand(tile_batch, -1, -1)
            )
            suffix = positions.new_zeros(
                (tile_batch, decoder.num_register_tokens + 1, 3)
            )
            blocks = decoder.transformer_blocks
            if state.latent_index:
                layers = [
                    (
                        torch.cat(
                            [state.tiles[tile][layer][0] for tile in batch_tiles]
                        ),
                        torch.cat(
                            [state.tiles[tile][layer][1] for tile in batch_tiles]
                        ),
                    )
                    for layer in range(len(blocks))
                ]
            else:
                layers = [None] * len(blocks)
            with torch.autocast(device_type=z.device.type, enabled=False):
                tokens = z[:, :, 0].permute(0, 2, 3, 1).flatten(1, 2).float()
                hidden = decoder.proj_in(tokens) + self.phase_embedding[phase]
            # Register and zero-cls tokens are rebuilt every step and never cached.
            hidden = torch.cat(
                [
                    hidden,
                    decoder.register_tokens.expand(tile_batch, -1, -1).to(hidden.dtype),
                    torch.zeros_like(hidden[:, :1]),
                ],
                dim=1,
            )
            query_rope = decoder.rope(torch.cat([positions, suffix], dim=1))
            # Every layer has the same history geometry. Reuse RoPE within this
            # tile batch only; relative times must age again on the next latent.
            key_ropes = {0: query_rope}
            updated = []
            for block, history in zip(blocks, layers):
                if history is None:
                    if torch.is_autocast_enabled(z.device.type):
                        dtype = torch.get_autocast_dtype(z.device.type)
                    else:
                        dtype = block.attn.to_q.weight.dtype
                    empty = hidden.new_empty(
                        tile_batch,
                        0,
                        block.attn.heads,
                        block.attn.dim_head,
                        dtype=dtype,
                    )
                    history_k, history_v = empty, empty
                else:
                    history_k, history_v = history
                history_frames = history_k.shape[1] // spatial_tokens
                if history_frames not in key_ropes:
                    past_positions = []
                    for distance in range(history_frames, 0, -1):
                        past = positions.clone()
                        past[..., 0] = 2 * -distance / window
                        past_positions.append(past)
                    key_ropes[history_frames] = decoder.rope(
                        torch.cat([*past_positions, positions, suffix], dim=1)
                    )
                key_rope = key_ropes[history_frames]
                hidden, k, v = streaming_block(
                    block,
                    hidden,
                    history_k,
                    history_v,
                    query_rope=query_rope,
                    key_rope=key_rope,
                    spatial_tokens=spatial_tokens,
                    cache_tokens=keep_frames * spatial_tokens,
                    attention=self.attention,
                )
                updated.append((k, v))
            for index, tile in enumerate(batch_tiles):
                lo, hi = index * video_batch, (index + 1) * video_batch
                state.tiles[tile] = [(k[lo:hi], v[lo:hi]) for k, v in updated]
            pixels = self._project_pixels(hidden[:, None], (h, w), state.latent_index)
            decoded.update(zip(batch_tiles, pixels.split(video_batch)))
        return decoded
