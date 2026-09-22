# SPDX-License-Identifier: Apache-2.0
"""Zing request preparation using SGLang's shared H3 Qwen and VAE modules."""

import math
from contextlib import contextmanager

import numpy as np
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation import (
    minimax_h3_ref2va_video_presentation,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
    _AudioVAEDeterminismContext,
    minimax_h3_encode_audio_mean,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.decoding import (
    _deterministic_audio_decode_context,
)

from .media_input import ZingH3MediaInput, _sample_max_area_config
from .packing import H3InferenceInputs
from .visual_encoding import encode_clips


@contextmanager
def _visual_encode_policy(mode):
    if mode not in ("deterministic", "performance"):
        raise ValueError(
            "Unsupported Zing visual encode_mode; expected deterministic or performance"
        )
    deterministic = mode == "deterministic"
    saved = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = saved[:2]
        torch.use_deterministic_algorithms(saved[2], warn_only=saved[3])


class ZingH3Codecs(ZingH3MediaInput):
    def __init__(self, modules, pipeline_config, runtime):
        self.modules, self.pipeline_config, self.runtime = (
            modules,
            pipeline_config,
            runtime,
        )
        self.patch_size = tuple(pipeline_config.dit_config.arch_config.patch_size)
        self.va = pipeline_config.vae_config.arch_config
        self.aa = pipeline_config.audio_vae_config.arch_config
        self.temporal_ratio = self.va.temporal_compression_ratio
        self.spatial_ratio = self.va.spatial_compression_ratio
        self.vae_frames_per_chunk = self.va.vae_clip_length
        self.vae_latents_per_chunk = math.ceil(
            self.vae_frames_per_chunk / self.temporal_ratio
        )
        self.canvas_multiple = self.spatial_ratio * self.patch_size[2]
        media = runtime["h3_media_config"]
        self.target_fps = float(media["fps"])
        self.max_short_edge = int(media["max_short_edge"])
        self.max_area_config = media["max_area_config"]
        self.max_source_target_aspect_ratio_factor = float(
            media["max_source_target_aspect_ratio_factor"]
        )
        self.audio_sample_rate = self.aa.sample_rate
        self.remote_read_fs = None

    def prepare_visual_encoder_weights(self):
        """Prepare encoder precision at startup, before cached sessions exist."""
        model = self.modules["video_vae"]
        dtype = getattr(torch, self.runtime["vae_config"]["encode_dtype"])
        model.encoder.to(dtype=dtype)
        model.quant_conv.to(dtype=dtype)

    @staticmethod
    def _stats(config, latent, audio=False):
        shape = (1, 1, 1, -1) if audio else (1, -1, 1, 1, 1)
        mean = latent.new_tensor(config.latents_mean, dtype=torch.float32).view(shape)
        std = latent.new_tensor(config.latents_std, dtype=torch.float32).view(shape)
        return mean, std

    def encode_condition(self, prompt, media):
        tokenizer, processor = self.modules["tokenizer"], self.modules["processor"]
        features, labels, images, videos, timestamps = {}, [], [], [], []
        counts = {"image": 0, "video": 0, "audio": 0}
        for item in media:
            kind = item["type"]
            if kind in {"audio", "video_audio"}:
                counts["audio"] += 1
                labels.append(("audio", counts["audio"]))
            if kind == "image":
                counts["image"] += 1
                labels.append(("image", counts["image"]))
                images.append(item["frames"][:, 0].permute(1, 2, 0).numpy())
            elif kind in {"video", "video_audio"}:
                counts["video"] += 1
                labels.append(("video", counts["video"]))
                fps = self.runtime["condition_encoder_config"]["video_sample_fps"]
                source_fps = float(item["fps"])
                cursor, indices = 0.0, []
                while round(cursor) < item["frames"].shape[1]:
                    if not indices or round(cursor) > indices[-1]:
                        indices.append(round(cursor))
                    cursor += source_fps / fps
                frames = [
                    item["frames"][:, i].permute(1, 2, 0).numpy() for i in indices
                ]
                # Rounding and deduplication can change the effective sample rate.
                ts = [i / source_fps for i in indices]
                temporal_patch = processor.video_processor.temporal_patch_size
                frames += [frames[-1]] * (-len(frames) % temporal_patch)
                ts += [ts[-1]] * (-len(ts) % temporal_patch)
                videos.append(np.stack(frames))
                timestamps.append(
                    [
                        (ts[i] + ts[i + temporal_patch - 1]) / 2
                        for i in range(0, len(ts), temporal_patch)
                    ]
                )
        image_counts, video_counts = [], []
        merge = processor.image_processor.merge_size**2
        if images:
            f = processor.image_processor(images=images, return_tensors="pt")
            features.update(
                pixel_values=f["pixel_values"], image_grid_thw=f["image_grid_thw"]
            )
            image_counts = [int(g.prod()) // merge for g in f["image_grid_thw"]]
        if videos:
            f = processor.video_processor(
                videos=videos, do_sample_frames=False, return_tensors="pt"
            )
            features.update(
                pixel_values_videos=f["pixel_values_videos"],
                video_grid_thw=f["video_grid_thw"],
            )
            for grid, ts in zip(f["video_grid_thw"], timestamps):
                if int(grid[0]) != len(ts):
                    raise ValueError(
                        "Zing Qwen video timestamps do not match processor tokens"
                    )
                video_counts.append([int(grid[1] * grid[2]) // merge] * len(ts))
        ids, tags = minimax_h3_ref2va_video_presentation(
            tokenizer,
            prompt=prompt,
            condition_labels=labels,
            image_token_count=image_counts,
            video_block_token_counts=video_counts,
            video_block_timestamps=timestamps,
        )
        context = self.modules["text_encoder"].encode_ids(ids, **features)
        return {"context": context, "row_role": (tags == 0).long().to(context.device)}

    def encode_visual(self, frames):
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

        model = self.modules["video_vae"]
        device = next(model.parameters()).device
        dtype = getattr(torch, self.runtime["vae_config"]["encode_dtype"])
        # Only the encoder changes precision. Decoder weights remain FP32.
        model.encoder.to(dtype=dtype)
        model.quant_conv.to(dtype=dtype)
        pixels = frames[None].to(device=device, dtype=dtype).div(255)
        mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        pixels = (pixels - mean) / std
        count = pixels.shape[2]
        if count != 1 and count % self.vae_frames_per_chunk:
            pixels = torch.cat(
                (
                    pixels,
                    pixels[:, :, -1:].expand(
                        -1, -1, -count % self.vae_frames_per_chunk, -1, -1
                    ),
                ),
                2,
            )
        with _visual_encode_policy(self.runtime["vae_config"]["encode_mode"]):
            moments = torch.cat(
                encode_clips(
                    model,
                    list(pixels.split(self.vae_frames_per_chunk, dim=2)),
                    self.runtime["vae_config"]["encode_tile_batch_size"],
                ),
                2,
            )
            chunks, tail = divmod(count - 1, self.vae_frames_per_chunk)
            length = (
                chunks * self.vae_latents_per_chunk
                + 1
                + math.ceil(tail / self.temporal_ratio)
            )
            posterior = DiagonalGaussianDistribution(moments[:, :, :length])
            generator = torch.Generator(device="cpu").manual_seed(
                self.runtime["vae_config"]["posterior_seed"]
            )
            latent = posterior.sample(generator=generator).half().float()
        mean, std = self._stats(self.va, latent)
        return ((latent - mean) / std).half()

    def encode_visual_moments(self, frames):
        """Encode one complete native clip (or an EOS single image), without RNG.

        Padding and posterior sampling belong to the stream owner. Do not use
        this helper to independently pad arbitrary transport append batches.
        """
        if frames.shape[1] not in (1, self.vae_frames_per_chunk):
            raise ValueError("H3 encoding requires one image or a complete clip")
        model = self.modules["video_vae"]
        device = next(model.parameters()).device
        dtype = getattr(torch, self.runtime["vae_config"]["encode_dtype"])
        model.encoder.to(dtype=dtype)
        model.quant_conv.to(dtype=dtype)
        pixels = frames[None].to(device=device, dtype=dtype).div(255)
        mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        with _visual_encode_policy(self.runtime["vae_config"]["encode_mode"]):
            return model._adaptive_encode((pixels - mean) / std)

    def sample_streaming_visual_moments(self, moments, *, latent_start, noise_version):
        """Opt-in streaming posterior; legacy encode_visual keeps its RNG mapping."""
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

        from .streaming_noise import indexed_video_noise

        posterior = DiagonalGaussianDistribution(moments)
        noise = indexed_video_noise(
            tuple(posterior.mean.shape),
            seed=self.runtime["vae_config"]["posterior_seed"],
            latent_start=latent_start,
            purpose="source_posterior",
            device=moments.device,
            dtype=moments.dtype,
            version=noise_version,
        )
        latent = (posterior.mean + posterior.std * noise).half().float()
        mean, std = self._stats(self.va, latent)
        return ((latent - mean) / std).half()

    def encode_audio(self, waveform):
        model = self.modules["audio_vae"]
        device = next(model.parameters()).device
        with _AudioVAEDeterminismContext():
            latent = minimax_h3_encode_audio_mean(
                model, waveform.to(device).reshape(2, -1), self.audio_sample_rate
            )
        # Native audio mean_proj returns [stereo, channels, time].
        latent = latent.reshape(1, 2, self.aa.latent_channels, -1).permute(0, 1, 3, 2)
        mean, std = self._stats(self.aa, latent, audio=True)
        return ((latent - mean) / std).half()

    def prepare(
        self,
        request,
        use_component,
        *,
        retain_condition_media=False,
    ):
        # Reuse decoded references for prompt refreshes; offline calls release them.
        if isinstance(request, dict) and "cache_uri" in request:
            raise ValueError(
                "Zing runtime accepts media messages, not pickled request caches"
            )
        # 请求可局部覆盖 source/reference 面积；必须在解码前失败，且不改常驻配置。
        max_area_config = _sample_max_area_config(self.max_area_config, request)
        prompt, items, target, conditions, source = self._parse_inference_request(
            request
        )
        audio_ids = {
            c["content_id"] for c in conditions if "audio" in c["latent_modalities"]
        }
        media = [
            self._decode_media(item, need_audio=item["id"] in audio_ids)
            for item in items
        ]
        source = (
            None if source is None else self._decode_media(source, need_audio=False)
        )
        refs, source, h, w, t, frames = self._normalize_inference_media(
            media, source, conditions, target, max_area_config=max_area_config
        )
        with use_component("text_encoder"):
            positive = self.encode_condition(prompt, media)
        visual, audio = {}, {}
        modes = {c["content_id"]: c["latent_modalities"] for c in conditions}
        visual_refs = [item for item in refs if "visual" in modes.get(item["id"], [])]
        source_latent = None
        if visual_refs or source is not None:
            with use_component("video_vae"):
                for item in visual_refs:
                    visual[item["id"]] = self.encode_visual(item["frames"])
                if source is not None:
                    source_latent = self.encode_visual(source["frames"])
        audio_refs = [item for item in refs if "audio" in modes.get(item["id"], [])]
        generate_audio = target["type"] == "video_audio"
        samples, audio_shape = None, None
        if audio_refs or generate_audio:
            with use_component("audio_vae"):
                for item in audio_refs:
                    audio[item["id"]] = self.encode_audio(item["waveform"])
                if generate_audio:
                    samples = int(frames / self.target_fps * self.audio_sample_rate)
                    audio_shape = tuple(
                        self.encode_audio(torch.zeros(2, samples)).shape[1:]
                    )
        return H3InferenceInputs(
            target["type"],
            conditions,
            [{"id": item["id"], "type": item["type"]} for item in refs],
            positive,
            visual,
            audio,
            source_latent,
            (t, h // self.spatial_ratio, w // self.spatial_ratio),
            audio_shape,
            frames,
            samples,
            self.audio_sample_rate,
            condition_media=media if retain_condition_media else None,
        )

    def decode_visual_range(self, latent, frame_start, frame_end):
        """Decode new pixels plus one preceding clip for H3 overlap blending.

        Starting at a clip boundary preserves temporal padding. The preceding
        clip supplies the blend into the first requested clip; earlier clips
        cannot affect that range. Immediate mode may revisit the unfinished
        boundary window, but neither mode re-decodes the growing full prefix.
        Slice before copying to the VAE device to bound the transfer as well.
        """
        if not 0 <= frame_start < frame_end:
            raise ValueError("Invalid Zing video decode range")
        model = self.modules["video_vae"]
        if getattr(model, "isolated_first_frame", False) or getattr(
            model, "isolated_last_frame", False
        ):
            raise ValueError(
                "Zing streaming decoding requires the standard H3 temporal layout"
            )
        clip = max(0, frame_start // self.vae_frames_per_chunk - 1)
        offset = clip * self.vae_frames_per_chunk
        window = latent[:, :, clip * self.vae_latents_per_chunk :].to(
            next(model.parameters()).device
        )
        decoded = self.decode_visual(window, frame_end - offset)
        return decoded[:, :, frame_start - offset : frame_end - offset]

    @property
    def video_decoder_name(self):
        return "taeh3_decoder" if "taeh3_decoder" in self.modules else "video_vae"

    def decode_visual(self, latent, frames):
        if self.video_decoder_name == "taeh3_decoder":
            return self.modules["taeh3_decoder"].decode(latent, frames)
        model = self.modules["video_vae"]
        single = frames == 1 and latent.shape[2] == 1
        chunk, drop = self.vae_latents_per_chunk, self.va.vae_token_drop
        padded = (
            chunk
            if single
            else max(
                2 * chunk - drop, latent.shape[2] + (-(latent.shape[2] + drop)) % chunk
            )
        )
        latent = torch.cat(
            (
                latent,
                latent[:, :, -1:].expand(-1, -1, padded - latent.shape[2], -1, -1),
            ),
            2,
        )
        mean, std = self._stats(self.va, latent)
        latent = latent.float() * std + mean
        decoded = (
            model._adaptive_decode(latent) if single else model.decode_base(latent)
        )
        if single:
            start = (-self.vae_frames_per_chunk) % self.temporal_ratio
            decoded = decoded[:, :, start : start + 1]
        if decoded.shape[2] < frames:
            raise ValueError("Zing VAE decoded fewer frames than requested")
        mean = decoded.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        std = decoded.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        return (decoded[:, :, :frames].float() * std + mean).clamp(0, 1)

    def decode_audio(self, latent, samples):
        mean, std = self._stats(self.aa, latent, audio=True)
        latent = (
            (latent.float() * std + mean)
            .permute(0, 1, 3, 2)
            .reshape(2, self.aa.latent_channels, -1)
        )
        with _deterministic_audio_decode_context():
            waveform = (
                self.modules["audio_vae"].decode(latent).reshape(1, 2, -1).float()
            )
        if samples > waveform.shape[-1]:
            raise ValueError("Zing audio VAE decoded fewer samples than requested")
        return waveform[..., :samples]
