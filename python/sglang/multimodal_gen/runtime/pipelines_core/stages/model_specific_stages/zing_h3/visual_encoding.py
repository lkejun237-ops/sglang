# SPDX-License-Identifier: Apache-2.0
"""Match native Zing's shape-bucketed, tile-major visual encoding batches."""

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def replicate(value, padding):
    for dim, (before, after) in zip((4, 3, 2), zip(padding[::2], padding[1::2])):
        parts = []
        if before:
            shape = list(value.shape)
            shape[dim] = before
            parts.append(value.narrow(dim, 0, 1).expand(shape))
        parts.append(value)
        if after:
            shape = list(value.shape)
            shape[dim] = after
            parts.append(value.narrow(dim, value.shape[dim] - 1, 1).expand(shape))
        if len(parts) > 1:
            value = torch.cat(parts, dim)
    return value


class SpatialReflectionPad(TorchDispatchMode):
    """Avoid 3D padding's 32-bit indexing limit for native-sized tile batches.

    The mode is scoped to encoding. Spatial reflection is independent for each
    N/C/T slice; replication copies boundary slices without changing values.
    """

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if (
            func is torch.ops.aten.pad.default
            and args[0].ndim == 5
            and len(args[1]) == 6
        ):
            value, padding = args[:2]
            mode = args[2] if len(args) > 2 else (kwargs or {}).get("mode", "constant")
            if not any(padding):
                return value
            if mode == "reflect" and not any(padding[4:]):
                padded = torch.ops.aten.reflection_pad2d.default(
                    value.flatten(0, -3), padding[:4]
                )
                return padded.unflatten(0, value.shape[:3])
            if mode == "replicate" and min(padding) >= 0:
                return replicate(value, padding)
        if (
            func is torch.ops.aten.replication_pad3d.default
            and args[0].ndim == 5
            and min(args[1]) >= 0
        ):
            return replicate(*args)

        if func is not torch.ops.aten.reflection_pad3d.default:
            return func(*args, **(kwargs or {}))
        hidden, padding = args
        if not any(padding):
            return hidden
        left, right, top, bottom, front, back = padding
        if hidden.ndim != 5 or not hidden.is_contiguous() or front or back:
            return func(*args, **(kwargs or {}))
        padded = torch.ops.aten.reflection_pad2d.default(
            hidden.flatten(0, -3), (left, right, top, bottom)
        )
        return padded.unflatten(0, hidden.shape[:3])


def encode_batch(model, values):
    with SpatialReflectionPad():
        return model.encode(values)


def encode_clips(model, clips, batch_size):
    """Encode single-item clips in native tile-major order, then restore clip order."""
    if batch_size < 1:
        raise ValueError("encode_tile_batch_size must be positive")
    if not model.encoder_tiling:
        return [
            part
            for start in range(0, len(clips), batch_size)
            for part in encode_batch(
                model, torch.cat(clips[start : start + batch_size])
            ).split(1)
        ]
    height, width = clips[0].shape[-2:]
    yi, yl, yo = model.split_tiles(height, False)
    xi, xl, xo = model.split_tiles(width, False)
    specs = [(y, h, x, w) for y, h in zip(yi, yl) for x, w in zip(xi, xl)]
    # BF16 convolution results depend on batch shape: preserve native order.
    groups = {}
    for tile, (y, h, x, w) in enumerate(specs):
        for clip_id, clip in enumerate(clips):
            value = clip[..., y : y + h, x : x + w]
            groups.setdefault(value.shape[2:], []).append(
                (clip_id * len(specs) + tile, value)
            )
    encoded = [None] * (len(specs) * len(clips))
    for group in groups.values():
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            result = encode_batch(model, torch.cat([v for _, v in batch])).split(1)
            for (idx, _), moments in zip(batch, result):
                encoded[idx] = moments
    outputs = []
    for clip_id in range(len(clips)):
        start = clip_id * len(specs)
        rows = [
            encoded[start + y * len(xi) : start + (y + 1) * len(xi)]
            for y in range(len(yi))
        ]
        outputs.append(
            model._assemble_tiles(
                rows,
                [v // model.vae_ratio for v in yo],
                [v // model.vae_ratio for v in xo],
            )
        )
    return outputs
