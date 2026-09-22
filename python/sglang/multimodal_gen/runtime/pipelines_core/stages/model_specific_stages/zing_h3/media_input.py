# Copyright 2025 The MiniMax Team and The HuggingFace Team.
# SPDX-License-Identifier: Apache-2.0
"""Zing inference message/media geometry adapted from seedleap/zing a0f45389."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from typing import Any

import fsspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.media_request import (
    optional_nonnegative_media_time,
    parse_h3_media_descriptor,
)


@contextmanager
def _local_media(uri: str, *, fs: Any | None = None):
    if not uri.startswith(("s3://", "http://", "https://")):
        yield uri.removeprefix("file://"), fs
        return
    # PyAV/PIL 统一消费本地路径；远端对象只在解码期间落临时文件，退出上下文立即删除。
    suffix = os.path.splitext(uri.split("?", 1)[0])[1] or ".bin"
    temp_path, read_fs, previous_exc = None, fs, None
    try:
        for delay in (0.5, 0.5, 0.5, 1, None):
            try:
                if fs is not None and previous_exc is None:
                    strip_protocol = getattr(fs, "_strip_protocol", None)
                    path = (
                        str(strip_protocol(uri))
                        if callable(strip_protocol)
                        else fsspec.core.url_to_fs(uri)[1]
                    )
                    read_fs = fs
                else:
                    open_kwargs = (
                        {"skip_instance_cache": True}
                        if previous_exc is not None and uri.startswith("s3://")
                        else {}
                    )
                    read_fs, path = fsspec.core.url_to_fs(uri, **open_kwargs)
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as target:
                    temp_path = target.name
                    with read_fs.open(path, "rb") as source:
                        shutil.copyfileobj(source, target)
                break
            except Exception as exc:
                if temp_path is not None and os.path.exists(temp_path):
                    os.unlink(temp_path)
                temp_path = None
                if previous_exc is not None:
                    exc.__context__ = previous_exc
                if delay is None:
                    raise
                previous_exc = exc
                time.sleep(delay)
        yield temp_path, read_fs
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)


def _clip_bounds(item: dict[str, Any]) -> tuple[float, float | None]:
    """把可选闭区间 ``[start,end]`` clip 归一成秒；end=None 表示一直读到媒体末尾。"""
    clip = item.get("clip")
    if clip is None:
        return 0.0, None
    if item.get("type") == "image":
        raise ValueError("H3 media clip is only valid for media with a timeline")
    if (
        not isinstance(clip, dict)
        or "start" not in clip
        or set(clip) - {"start", "end"}
    ):
        raise ValueError("H3 media clip requires start and optional end")
    start = optional_nonnegative_media_time(clip["start"], "clip.start")
    end = optional_nonnegative_media_time(clip.get("end"), "clip.end")
    if (
        start is None
        or ("end" in clip and end is None)
        or (end is not None and end < start)
    ):
        raise ValueError("H3 media clip requires start >= 0 and end >= start")
    return start, end


def _clip_indices(
    start: float, end: float | None, rate: float
) -> tuple[int, int | None]:
    """把闭区间秒边界转成 Python 半开索引；start=end 恰好保留一个帧/采样点。"""
    return round(start * rate), None if end is None else round(end * rate) + 1


def _sample_max_area_config(
    defaults: dict[str, int], sample: dict[str, Any]
) -> dict[str, int]:
    """用请求的非空局部字段覆盖 YAML 默认值，不修改常驻 max_area_config。"""
    override = sample.get("h3_max_area_config") if isinstance(sample, dict) else None
    if override is None:
        return dict(defaults)
    if not isinstance(override, dict) or not override:
        raise ValueError("h3_max_area_config must be a non-empty object")
    unknown = set(override) - set(defaults)
    if unknown:
        raise ValueError(f"h3_max_area_config has unknown fields: {sorted(unknown)}")
    if any(type(value) is not int or value < 1 for value in override.values()):
        raise ValueError("h3_max_area_config values must be positive integers")
    return {**defaults, **override}


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


def _resize_frames(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """在 CPU 上逐帧做 Lanczos resize，并维持 dataloader 的 uint8 ``[C,T,H,W]`` 契约。"""
    if tuple(frames.shape[-2:]) == (height, width):
        return frames
    images = [
        np.asarray(
            Image.fromarray(frame.permute(1, 2, 0).numpy()).resize(
                (width, height), Image.Resampling.LANCZOS
            )
        )
        for frame in frames.permute(1, 0, 2, 3)
    ]
    return (
        torch.from_numpy(np.stack(images))
        .permute(3, 0, 1, 2)
        .contiguous()
        .to(torch.uint8)
    )


def _resample_video(
    frames: torch.Tensor, source_fps: float, target_fps: float
) -> torch.Tensor:
    """用最近时间槽复制/丢弃帧；不插值像素，保证 reference/target 的内容不被混帧。"""
    if source_fps <= 0:
        raise ValueError("H3 video FPS must be positive")
    if source_fps == target_fps:
        return frames
    scale = target_fps / source_fps
    slots = np.floor(np.arange(frames.shape[1]) * scale + 0.5).astype(np.int64)
    counts = np.diff(slots, append=math.floor(frames.shape[1] * scale + 0.5))
    return torch.repeat_interleave(frames, torch.from_numpy(counts), dim=1)


def _trimmed_vae_reference_frames(
    num_frames: int, frames_per_chunk: int, temporal_ratio: int
) -> int:
    if (
        min(num_frames, frames_per_chunk, temporal_ratio) < 1
        or (frames_per_chunk - 1) % temporal_ratio
    ):
        raise ValueError("H3 VAE encode chunk must contain 4k+1 frames")
    if num_frames <= temporal_ratio:
        return num_frames  # 过短视频原样交给 VAE 内部 padding
    tail = (num_frames - 1) % frames_per_chunk + 1
    return num_frames - (tail - 1) % temporal_ratio  # reference 尾块向下截到 4k+1


class ZingH3MediaInput:
    def _validate_paired_aspect(
        self, source_frames: torch.Tensor, target_height: int, target_width: int
    ) -> None:
        source_ratio, target_ratio = (
            source_frames.shape[-1] / source_frames.shape[-2],
            target_width / target_height,
        )
        factor = max(source_ratio / target_ratio, target_ratio / source_ratio)
        if factor > self.max_source_target_aspect_ratio_factor:
            raise ValueError(
                f"H3 source/target aspect ratio factor {factor:.4f} exceeds h3_media_config.max_source_target_aspect_ratio_factor={self.max_source_target_aspect_ratio_factor}"
            )

    def _inference_geometry(self, target: dict[str, Any]) -> tuple[int, int, int, int]:
        """验证用户要求的像素几何，并换算 generator 实际生成的 latent 帧数。"""
        output = target.get("output")
        if not isinstance(output, dict) or set(output) != {"frames", "height", "width"}:
            raise ValueError(
                "H3 inference target.output requires exactly frames, height and width"
            )
        if any(type(output[name]) is not int for name in ("frames", "height", "width")):
            raise ValueError("H3 inference target.output values must be integers")
        pixel_frames, height, width = (
            output[name] for name in ("frames", "height", "width")
        )
        if (
            min(pixel_frames, height, width) < 1
            or height % (self.spatial_ratio * self.patch_size[1])
            or width % (self.spatial_ratio * self.patch_size[2])
        ):
            raise ValueError(
                "H3 inference output geometry must be positive and divisible by the VAE/generator spatial grid"
            )
        if min(height, width) > self.max_short_edge:
            raise ValueError(
                "H3 inference output geometry exceeds h3_media_config.max_short_edge"
            )
        if target["type"] == "image" and pixel_frames != 1:
            raise ValueError("H3 image inference target requires output.frames=1")
        chunks, tail = divmod(pixel_frames - 1, self.vae_frames_per_chunk)
        latent_frames = (
            chunks * self.vae_latents_per_chunk
            + 1
            + math.ceil(tail / self.temporal_ratio)
        )
        if latent_frames % self.patch_size[0]:
            raise ValueError(
                "H3 inference latent frames must be divisible by generator temporal patch size"
            )
        return height, width, latent_frames, pixel_frames

    def _parse_inference_request(
        self, request: dict[str, Any]
    ) -> tuple[
        str,
        list[dict[str, Any]],
        dict[str, Any],
        list[dict[str, Any]],
        dict[str, Any] | None,
    ]:
        """校验推理专用的两消息协议，返回扁平且有明确类型的运行期字段。"""
        if (
            not isinstance(request, dict)
            or not isinstance(request.get("messages"), list)
            or len(request["messages"]) != 2
        ):
            raise ValueError(
                "H3 inference request requires exactly one user and one target message"
            )
        if any(
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or message["role"] not in {"user", "target"}
            for message in request["messages"]
        ):
            raise ValueError("H3 inference messages must have user and target roles")
        by_role = {message["role"]: message for message in request["messages"]}
        if set(by_role) != {"user", "target"}:
            raise ValueError("H3 inference messages must have user and target roles")
        user, target_message = by_role["user"], by_role["target"]
        user_content, target_content = (
            user.get("content"),
            target_message.get("content"),
        )
        if (
            not isinstance(user_content, list)
            or not isinstance(target_content, list)
            or len(target_content) != 1
        ):
            raise ValueError(
                "H3 inference user/target content must be lists and target content must contain one output descriptor"
            )
        prompts = [
            item.get("text")
            for item in user_content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if (
            len(prompts) != 1
            or not isinstance(prompts[0], str)
            or not prompts[0].strip()
        ):
            raise ValueError(
                "H3 inference user content requires exactly one non-empty text prompt"
            )
        target = target_content[0]
        if (
            not isinstance(target, dict)
            or not isinstance(target.get("type"), str)
            or target.get("type") not in {"image", "video", "video_audio"}
            or "uri" in target
        ):
            raise ValueError(
                "H3 inference target must be a URI-free image, video or video_audio output descriptor"
            )
        self._inference_geometry(target)
        media = []
        for index, item in enumerate(user_content):
            if not isinstance(item, dict):
                raise ValueError("H3 inference user content items must be objects")
            if item.get("type") == "text":
                continue
            descriptor = parse_h3_media_descriptor(item, path=f"user.content[{index}]")
            if not isinstance(item.get("id"), str) or not item["id"]:
                raise ValueError("H3 inference media require a non-empty id")
            _clip_bounds(item)
            media.append({**item, **descriptor})
        media_by_id = {item["id"]: item for item in media}
        if len(media_by_id) != len(media):
            raise ValueError("H3 inference media ids must be unique")
        conditions = target_message.get("conditions", [])
        if not isinstance(conditions, list) or any(
            not isinstance(condition, dict) for condition in conditions
        ):
            raise ValueError("H3 inference target.conditions must be a list")
        condition_ids = [condition.get("content_id") for condition in conditions]
        if (
            any(not isinstance(content_id, str) for content_id in condition_ids)
            or len(set(condition_ids)) != len(condition_ids)
            or any(content_id not in media_by_id for content_id in condition_ids)
        ):
            raise ValueError(
                "H3 inference conditions must be a unique subset of user media"
            )
        if condition_ids != [
            item["id"] for item in media if item["id"] in set(condition_ids)
        ]:
            raise ValueError("H3 inference conditions must preserve user media order")
        for condition in conditions:
            self._validate_inference_condition(
                condition, media_by_id[condition["content_id"]]
            )
        source = target_message.get("source")
        if source is not None:
            descriptor = parse_h3_media_descriptor(source, path="target.source")
            source_type = "image" if target["type"] == "image" else "video"
            if (
                set(source) - {"type", "uri", "clip"}
                or descriptor["type"] != source_type
            ):
                raise ValueError(
                    f"H3 {target['type']} inference target.source must be a {source_type} URI descriptor"
                )
            _clip_bounds(source)
            source = {**source, **descriptor}
        return (
            prompts[0],
            media,
            dict(target),
            [dict(condition) for condition in conditions],
            None if source is None else dict(source),
        )

    @staticmethod
    def _validate_inference_condition(
        condition: dict[str, Any], media: dict[str, Any]
    ) -> None:
        role, modalities = condition.get("role"), condition.get("latent_modalities")
        if (
            set(condition) != {"content_id", "role", "latent_modalities"}
            or role != "reference"
            or not isinstance(modalities, list)
            or not modalities
            or any(not isinstance(modality, str) for modality in modalities)
            or len(set(modalities)) != len(modalities)
            or not set(modalities) <= {"visual", "audio"}
        ):
            raise ValueError(
                "H3 inference condition must be one reference with unique visual/audio latent_modalities"
            )
        if media["type"] == "image" and set(modalities) != {"visual"}:
            raise ValueError("H3 image condition only supports visual latent modality")
        if media["type"] == "audio" and set(modalities) != {"audio"}:
            raise ValueError("H3 audio condition only supports audio latent modality")

    def _normalize_inference_media(
        self,
        media: list[dict[str, Any]],
        source: dict[str, Any] | None,
        conditions: list[dict[str, Any]],
        target: dict[str, Any],
        *,
        max_area_config: dict[str, int],
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int, int, int, int]:
        """reference/source 各自选桶；target 输出尺寸由请求显式指定。"""
        media, source = (
            [{**item} for item in media],
            None if source is None else {**source},
        )
        output_height, output_width, num_frames, pixel_frames = (
            self._inference_geometry(target)
        )
        media_by_id = {item["id"]: item for item in media}
        target_height, target_width = output_height, output_width
        for condition in conditions:
            item = media_by_id[condition["content_id"]]
            if item["type"] == "image":
                height, width = _select_h3_bucket(
                    item["frames"].shape[-2],
                    item["frames"].shape[-1],
                    multiple=self.canvas_multiple,
                    max_short_edge=self.max_short_edge,
                    max_area=max_area_config["reference_image"],
                )
                item["frames"] = _resize_frames(item["frames"], height, width)
            elif "frames" in item:
                item["frames"] = _resample_video(
                    item["frames"], float(item["fps"]), self.target_fps
                )
                frames = _trimmed_vae_reference_frames(
                    item["frames"].shape[1],
                    self.vae_frames_per_chunk,
                    self.temporal_ratio,
                )
                height, width = _select_h3_bucket(
                    item["frames"].shape[-2],
                    item["frames"].shape[-1],
                    multiple=self.canvas_multiple,
                    max_short_edge=self.max_short_edge,
                    max_area=max_area_config["reference_video"],
                )
                item["frames"] = _resize_frames(
                    item["frames"][:, :frames], height, width
                )
                item["fps"] = self.target_fps
            if "waveform" in item and "frames" in item:
                frame_count = (
                    item["frames"].shape[1] if "frames" in item else pixel_frames
                )
                sample_count = int(
                    frame_count / self.target_fps * int(item["sample_rate"])
                )
                item["waveform"] = F.pad(
                    item["waveform"][:, :sample_count],
                    (0, max(0, sample_count - item["waveform"].shape[1])),
                )
        if source is not None:
            if target["type"] == "image":
                if source["frames"].shape[1] != 1:
                    raise ValueError(
                        "H3 image inference source must decode to exactly one frame"
                    )
                source["fps"] = 1.0
            else:
                source["frames"] = _resample_video(
                    source["frames"], float(source["fps"]), self.target_fps
                )
                if source["frames"].shape[1] != pixel_frames:
                    raise ValueError(
                        "H3 inference source frames must equal target.output.frames after FPS normalization"
                    )
                source["fps"] = self.target_fps
            self._validate_paired_aspect(source["frames"], target_height, target_width)
            source_height, source_width = _select_h3_bucket(
                source["frames"].shape[-2],
                source["frames"].shape[-1],
                multiple=self.canvas_multiple,
                max_short_edge=self.max_short_edge,
                max_area=max_area_config["source"],
            )
            source["frames"] = _resize_frames(
                source["frames"], source_height, source_width
            )
        return media, source, target_height, target_width, num_frames, pixel_frames

    def _decode_media(
        self, item: dict[str, Any], *, need_audio: bool
    ) -> dict[str, Any]:
        """按 URI 解码一个媒体描述；仅在 condition/target 真正需要时读取音轨。"""
        import av
        from PIL import Image, UnidentifiedImageError

        result = {
            key: item[key] for key in ("id", "type", "uri", "clip") if key in item
        }
        start, end = _clip_bounds(item)
        with _local_media(str(item["uri"]), fs=self.remote_read_fs) as (
            path,
            remote_read_fs,
        ):
            self.remote_read_fs = remote_read_fs
            frames, fps = None, None
            if item["type"] == "image":
                try:
                    with Image.open(path) as image:
                        frame = np.array(
                            image.convert("RGB"), dtype=np.uint8, copy=True
                        )
                except (UnidentifiedImageError, OSError):
                    with av.open(path) as container:
                        frame = next(container.decode(video=0)).to_rgb().to_ndarray()
                frames, fps = frame[None], 1.0
            elif item["type"] in {"video", "video_audio"}:
                with av.open(path) as container:
                    stream = container.streams.video[0]
                    fps = float(stream.average_rate or stream.base_rate)
                    first, last = _clip_indices(start, end, fps)
                    selected = []
                    for index, frame in enumerate(container.decode(stream)):
                        if last is not None and index >= last:
                            break
                        if index >= first:
                            selected.append(frame.to_rgb().to_ndarray())
                    frames = (
                        np.stack(selected)
                        if selected
                        else np.empty((0,), dtype=np.uint8)
                    )
                if not len(frames):
                    raise ValueError(
                        f"H3 media clip produced no video frames: {item['uri']}"
                    )
            if frames is not None:
                result["frames"] = (
                    torch.from_numpy(np.ascontiguousarray(frames))
                    .permute(3, 0, 1, 2)
                    .to(torch.uint8)
                )
                result["fps"] = fps
            if need_audio:
                # 音频单独重新打开容器并重采样；cursor 始终以目标采样率计数，便于精确裁 clip。
                chunks, cursor = [], 0
                first, last = _clip_indices(start, end, self.audio_sample_rate)
                with av.open(path) as container:
                    if not container.streams.audio:
                        raise ValueError(
                            f"H3 media requires an audio stream: {item['uri']}"
                        )
                    stream = container.streams.audio[0]
                    resampler = av.AudioResampler(
                        format="fltp", layout="stereo", rate=self.audio_sample_rate
                    )
                    finished = False
                    for audio_frame in container.decode(stream):
                        for output in resampler.resample(audio_frame):
                            values = output.to_ndarray().reshape(2, -1)
                            end = cursor + values.shape[1]
                            if end > first and (last is None or cursor < last):
                                chunks.append(
                                    values[
                                        :,
                                        max(0, first - cursor) : values.shape[1]
                                        if last is None
                                        else min(values.shape[1], last - cursor),
                                    ]
                                )
                            cursor = end
                            if last is not None and cursor >= last:
                                finished = True
                                break
                        if finished:
                            break
                    if not finished:
                        for output in resampler.resample(None):
                            values = output.to_ndarray().reshape(2, -1)
                            end = cursor + values.shape[1]
                            if end > first and (last is None or cursor < last):
                                chunks.append(
                                    values[
                                        :,
                                        max(0, first - cursor) : values.shape[1]
                                        if last is None
                                        else min(values.shape[1], last - cursor),
                                    ]
                                )
                            cursor = end
                if not chunks:
                    raise ValueError(
                        f"H3 media produced no decoded audio: {item['uri']}"
                    )
                waveform = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
                if not waveform.shape[1]:
                    raise ValueError(
                        f"H3 media clip produced no audio samples: {item['uri']}"
                    )
                result["waveform"] = torch.from_numpy(np.ascontiguousarray(waveform))
                result["sample_rate"] = self.audio_sample_rate
        return result
