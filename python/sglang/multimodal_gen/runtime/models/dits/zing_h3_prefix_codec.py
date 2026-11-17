# SPDX-License-Identifier: Apache-2.0
"""Versioned complete FP8 Block0 payloads, without tensor reconstruction.

The wire schema is fixed by the server, never supplied as arbitrary field names
or shapes by a cache object. One aligned byte buffer per layer contains every
resident field. Copying strided means/scales into that buffer performs no
dequantization or requantization. BF16 remains the independent legacy format.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch

from .zing_h3_compressed_view import LAYOUT_VERSION

BF16 = "bf16-raw-v1"
FP8 = "fp8-residual-v1"


def encoding_for_identity(identity):
    attention = identity.get("execution", {}).get("attention", {})
    if not attention.get("three_tier", False):
        return BF16
    if attention.get("compressed_layout") != LAYOUT_VERSION:
        raise ValueError("Unsupported compressed Block0 execution layout")
    return FP8


@dataclass(frozen=True)
class Field:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset: int
    size: int


@lru_cache(maxsize=64)
def fields(rows, heads, dim):
    if any(type(n) is not int or n <= 0 for n in (rows, heads, dim)):
        raise ValueError("Compressed Block0 geometry must be positive integers")
    blocks, sub = (rows + 63) // 64, 4 * ((rows + 63) // 64)
    specs = [
        ("k_res_fp8", (blocks, heads, 64, dim), torch.float8_e4m3fn),
        ("v_res_fp8", (blocks, heads, dim, 64), torch.float8_e4m3fn),
        ("k_res_scale", (heads, blocks), torch.float32),
        ("v_res_scale", (heads, blocks), torch.float32),
        ("k_blk_mean", (heads, blocks, dim), torch.bfloat16),
        ("v_blk_mean", (heads, blocks, dim), torch.bfloat16),
        ("k_sub_mean_fp8", (heads, sub, dim), torch.float8_e4m3fn),
        ("v_sub_mean_fp8", (heads, dim, sub), torch.float8_e4m3fn),
        ("k_sub_scale", (heads, sub), torch.float32),
        ("v_sub_scale", (heads, sub), torch.float32),
        ("k_blk_mean_fp8", (heads, blocks, dim), torch.float8_e4m3fn),
        ("v_blk_mean_fp8", (heads, dim, blocks), torch.float8_e4m3fn),
        ("k_blk_scale", (heads, blocks), torch.float32),
        ("v_blk_scale", (heads, blocks), torch.float32),
        ("tau_k", (heads, blocks), torch.float32),
        ("rho_kv", (heads, blocks), torch.float32),
        ("tau_k_between", (heads, blocks), torch.float32),
        ("tau_k_within", (heads, blocks), torch.float32),
        ("alpha_k", (heads, blocks), torch.float32),
        ("block_valid_len", (blocks,), torch.int32),
        ("block_chunk_id", (blocks,), torch.int32),
        ("block_delta_t", (blocks,), torch.float32),
        ("block_is_shifted", (blocks,), torch.int32),
    ]
    result, offset = [], 0
    import math

    for name, shape, dtype in specs:
        offset = (offset + 15) // 16 * 16
        size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        result.append(Field(name, shape, dtype, offset, size))
        offset += size
    return tuple(result)


def layer_bytes(rows, heads, dim):
    last = fields(rows, heads, dim)[-1]
    return (last.offset + last.size + 15) // 16 * 16


def _view(buffer, field):
    return (
        buffer[field.offset : field.offset + field.size]
        .view(field.dtype)
        .view(field.shape)
    )


def _resident_view(store, field):
    tensor = getattr(store, field.name)
    # fp8-residual-v1 stores this flag as int32. copy_ converts it at the
    # storage boundary so the attention kernel always receives a bool mask.
    resident_dtype = torch.bool if field.name == "block_is_shifted" else field.dtype
    if tensor.dtype != resident_dtype or len(tensor.shape) != len(field.shape):
        raise ValueError("Compressed Block0 field dtype/rank mismatch")
    if any(capacity < size for capacity, size in zip(tensor.shape, field.shape)):
        raise ValueError("Compressed Block0 exceeds resident field capacity")
    return tensor[tuple(slice(0, size) for size in field.shape)]


def allocate(rows, heads, dim, *, device):
    # Clear alignment padding as well: no uninitialized GPU bytes reach storage.
    return torch.zeros(layer_bytes(rows, heads, dim), dtype=torch.uint8, device=device)


def pack_layer(store, rows):
    count = (rows + 63) // 64
    if not 0 < count <= store.num_blocks:
        raise ValueError("Compressed Block0 is not fully published")
    buffer = allocate(rows, store.num_heads, store.head_dim, device=store.device)
    for field in fields(rows, store.num_heads, store.head_dim):
        _view(buffer, field).copy_(_resident_view(store, field))
    return buffer


def unpack_layer(buffer, store, rows):
    expected = layer_bytes(rows, store.num_heads, store.head_dim)
    if (
        buffer.dtype != torch.uint8
        or buffer.ndim != 1
        or not buffer.is_contiguous()
        or buffer.numel() != expected
        or buffer.device != store.device
    ):
        raise ValueError("Invalid compressed Block0 byte buffer")
    for field in fields(rows, store.num_heads, store.head_dim):
        _resident_view(store, field).copy_(_view(buffer, field))


def snapshot(cache):
    prefix = cache.prefix
    stores = cache._compressed_kv_stores
    if (
        prefix is None
        or prefix.offset != 0
        or prefix.block != 0
        or not stores
        or len(stores) != cache.num_layers
    ):
        raise ValueError("Compressed Block0 requires a complete prefix")
    rows = int(prefix.positions.shape[0])
    counts = cache._prefix_compressed_blocks
    if counts != tuple((rows + 63) // 64 for _ in stores):
        raise ValueError("Compressed Block0 prefix boundary is incompatible")
    return tuple(pack_layer(store, rows) for store in stores)


def restore(cache, positions, blocks, buffers, host_positions):
    """Restore private resident arrays directly; no quantizer is called."""
    if (
        not getattr(cache, "_compressed_kv_enabled", False)
        or cache.prefix is not None
        or cache._length
    ):
        raise ValueError("Compressed restore requires an empty compressed cache")
    if len(buffers) != cache.num_layers:
        raise ValueError("Compressed restore requires every transformer layer")
    if any(store.num_blocks for store in cache._compressed_kv_stores):
        raise ValueError("Compressed restore cannot overwrite published blocks")
    rows = int(positions.shape[0])
    shadow = cache.planning_copy()
    shadow.retain_prefix_layers = False
    # Shape-only clean K/V views advance the existing validated host transaction.
    # No raw data is read or committed in bookkeeping-only mode.
    layers = [(slot.k[:rows], slot.v[:rows]) for slot in cache.storage]
    shadow.update(
        positions,
        blocks,
        "prefix",
        0,
        layers=layers,
        validation=(host_positions, torch.zeros(rows, dtype=torch.long)),
    )
    try:
        for buffer, store in zip(buffers, shadow._compressed_kv_stores, strict=True):
            unpack_layer(buffer, store, rows)
        cache.publish_plan(shadow)
    except Exception:
        cache.clear()
        raise
