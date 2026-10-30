# SPDX-License-Identifier: Apache-2.0
"""Spatial geometry shared by offline media and continuous source ingress."""

import math
from dataclasses import asdict, dataclass

from sglang.multimodal_gen.configs.zing_media import ZING_SOURCE_RESIZE

SOURCE_RESIZE_VERSION = "h3-lanczos-v1"


@dataclass(frozen=True)
class SourceGeometry:
    input_height: int
    input_width: int
    encoded_height: int
    encoded_width: int
    target_height: int
    target_width: int

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("Source geometry requires positive integer dimensions")
        if (
            self.encoded_height > self.input_height
            or self.encoded_width > self.input_width
        ):
            raise ValueError("Source encoding must not upscale input pixels")

    @property
    def input_shape(self):
        return self.input_height, self.input_width

    @property
    def encoded_shape(self):
        return self.encoded_height, self.encoded_width

    @property
    def use_target_736_media_path(self):
        return (self.target_height, self.target_width) == (736, 1280)

    def resize(self, pixels):
        if tuple(pixels.shape[-2:]) != self.input_shape:
            raise ValueError("Source pixels differ from the negotiated input geometry")
        return _resize_frames(
            pixels,
            *self.encoded_shape,
            contiguous_hwc=self.use_target_736_media_path,
        )

    def receipt(self):
        return dict(
            width=self.encoded_width,
            height=self.encoded_height,
            resize_applied=self.input_shape != self.encoded_shape,
            algorithm=SOURCE_RESIZE_VERSION,
        )


def resolve_source_geometry(input_shape, target_shape, pipeline_config, media):
    """Resolve once per session, independently of transport and model execution."""
    h, w = input_shape
    th, tw = target_shape
    if any(type(v) is not int or v < 1 for v in (h, w, th, tw)):
        raise ValueError("Source and target dimensions must be positive integers")
    vae = pipeline_config.vae_config.arch_config
    patch = pipeline_config.dit_config.arch_config.patch_size
    mh, mw = (
        vae.spatial_compression_ratio * patch[1],
        vae.spatial_compression_ratio * patch[2],
    )
    limits = ZING_SOURCE_RESIZE
    if mh != mw:
        raise ValueError("Source resize requires a square model spatial grid")
    if th % mh or tw % mw or min(th, tw) > media["max_short_edge"]:
        raise ValueError("Target canvas does not match the model spatial limits")
    factor = max((w / h) / (tw / th), (tw / th) / (w / h))
    if factor > media["max_source_target_aspect_ratio_factor"]:
        raise ValueError("Source and target aspect ratios are incompatible")
    eh, ew = _select_h3_bucket(
        h,
        w,
        multiple=mh,
        max_short_edge=min(limits.max_short_edge, media["max_short_edge"]),
        max_area=min(limits.max_area, media["max_area_config"]["source"]),
    )
    factor = max((ew / eh) / (tw / th), (tw / th) / (ew / eh))
    if factor > media["max_source_target_aspect_ratio_factor"]:
        raise ValueError("Encoded source and target aspect ratios are incompatible")
    return SourceGeometry(h, w, eh, ew, th, tw)


def _select_h3_bucket(
    height: int, width: int, *, multiple: int, max_short_edge: int, max_area: int
) -> tuple[int, int]:
    # 每条媒体独立选桶，候选始终不得超过自身原始尺寸。
    if min(height, width, multiple, max_short_edge, max_area) < 1:
        raise ValueError("H3 media geometry and max_area must be positive")
    source_ratio = width / height
    if not 0.25 <= source_ratio <= 4.0:
        raise ValueError(
            f"H3 supports image/video aspect ratios in [1:4,4:1], got {width}:{height}"
        )
    candidates = set()
    max_height = min(height, max_area // multiple)
    for bucket_h in range(multiple, max_height + 1, multiple):
        width_limit = min(width, max_area // bucket_h)
        if width_limit < multiple:
            continue
        ideal = source_ratio * bucket_h
        for bucket_w in {
            math.floor(ideal / multiple) * multiple,
            math.ceil(ideal / multiple) * multiple,
            width_limit // multiple * multiple,
        }:
            if (
                bucket_w < multiple
                or bucket_w > width_limit
                or min(bucket_h, bucket_w) > max_short_edge
            ):
                continue
            ratio = bucket_w / bucket_h
            if 0.25 <= ratio <= 4.0:
                candidates.add((bucket_h, bucket_w))
    if not candidates:
        raise ValueError(
            f"H3 media {height}x{width} has no non-upscaling bucket under max_area={max_area}"
        )
    if height * width <= max_area:
        # 原图与候选桶共左下角；按矩形 IoU 最大选桶，不用 H/W 欧氏距离混淆几何重叠。
        ranked_iou = []
        for bucket_h, bucket_w in candidates:
            intersection = min(height, bucket_h) * min(width, bucket_w)
            ranked_iou.append(
                (
                    intersection
                    / (height * width + bucket_h * bucket_w - intersection),
                    bucket_h * bucket_w,
                    -bucket_h,
                    -bucket_w,
                )
            )
        _, _, neg_h, neg_w = max(ranked_iou)
        return -neg_h, -neg_w
    ranked = [
        (
            max(
                (bucket_w / bucket_h) / source_ratio,
                source_ratio / (bucket_w / bucket_h),
            ),
            bucket_h,
            bucket_w,
        )
        for bucket_h, bucket_w in candidates
    ]
    near_aspect = [
        item for item in ranked if item[0] <= 1.05
    ]  # 与 Wan 一致：比例形变不超过 5% 时优先吃满面积预算
    if near_aspect:
        _, bucket_h, bucket_w = min(
            near_aspect, key=lambda item: (-(item[1] * item[2]), item)
        )
        return bucket_h, bucket_w
    _, bucket_h, bucket_w = min(
        ranked, key=lambda item: (item[0], -(item[1] * item[2]), item[1], item[2])
    )  # 多媒体共享画布受限时退回最接近比例桶
    return bucket_h, bucket_w


def _resize_frames(frames, height: int, width: int, *, contiguous_hwc: bool = False):
    """在 CPU 上逐帧做 Lanczos resize，并维持 dataloader 的 uint8 ``[C,T,H,W]`` 契约。"""
    import numpy as np
    import torch
    from PIL import Image

    if tuple(frames.shape[-2:]) == (height, width):
        return frames
    images = [
        np.asarray(
            Image.fromarray(
                np.ascontiguousarray(frame.permute(1, 2, 0).numpy())
                if contiguous_hwc
                else frame.permute(1, 2, 0).numpy()
            ).resize((width, height), Image.Resampling.LANCZOS)
        )
        for frame in frames.permute(1, 0, 2, 3)
    ]
    return (
        torch.from_numpy(np.stack(images))
        .permute(3, 0, 1, 2)
        .contiguous()
        .to(torch.uint8)
    )
