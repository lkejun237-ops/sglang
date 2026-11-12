# SPDX-License-Identifier: Apache-2.0
"""Zing request preparation using SGLang's shared H3 Qwen and VAE modules."""

import hashlib
import json
import math
import os
import threading
import time
import weakref
from contextlib import contextmanager, nullcontext
from copy import copy

import torch

from sglang.multimodal_gen.runtime.conditioning.cache import ConditioningCache
from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
from sglang.multimodal_gen.runtime.conditioning.encoder import encode_condition
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
from sglang.multimodal_gen.runtime.realtime.zing_component_metrics import (
    component_timer,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.zing_stage_timing import handoff, stage_span

from .decoder import decoder_adapter
from .media_input import ZingH3MediaInput, _sample_max_area_config
from .packing import H3InferenceInputs
from .visual_encoding import encode_clips

# Default checkpoint reference preprocessing is larger than the output canvas.
_STATIC_REFERENCE_MAX_AREA = 1032192


def _source_compute_canvas_supported(height, width):
    # Source encoding uses video chunks; the larger still-reference envelope
    # does not establish a safe bound for this path.
    return (
        0 < min(height, width) <= 480
        and max(height, width) <= 1280
        and height * width <= 832 * 480
    )


_VISUAL_ENCODE_POLICY_LOCK = threading.RLock()
logger = init_logger(__name__)


def _validate_encoder_benchmark_limit(value):
    # Only the explicitly measured experimental candidate is admitted. Default
    # None leaves cuDNN's existing candidate policy unchanged.
    if value is not None and (type(value) is not int or value != 1):
        raise ValueError("Encoder cudnn_benchmark_limit must be null or 1")
    return value


@contextmanager
def _visual_encode_policy(mode, cudnn_benchmark_limit=None):
    if mode not in ("deterministic", "performance"):
        raise ValueError(
            "Unsupported Zing visual encode_mode; expected deterministic or performance"
        )
    _validate_encoder_benchmark_limit(cudnn_benchmark_limit)
    deterministic = mode == "deterministic"
    # PyTorch backend flags are process-wide. Serialize visual encoder contexts,
    # including nested callers; dedicated overlap uses separate worker processes.
    # The serving owner must not launch unrelated cuDNN work concurrently on a
    # different host thread while this context is active.
    with _VISUAL_ENCODE_POLICY_LOCK:
        previous_limit = torch.backends.cudnn.benchmark_limit
        if cudnn_benchmark_limit is not None and previous_limit is None:
            raise RuntimeError(
                "This runtime does not support Encoder cudnn_benchmark_limit"
            )
        saved = (
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        try:
            if cudnn_benchmark_limit is not None:
                torch.backends.cudnn.benchmark_limit = cudnn_benchmark_limit
            torch.backends.cudnn.benchmark = not deterministic
            torch.backends.cudnn.deterministic = deterministic
            torch.use_deterministic_algorithms(deterministic)
            yield dict(
                encode_mode=mode,
                cudnn_benchmark_limit=torch.backends.cudnn.benchmark_limit,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            )
        finally:
            if cudnn_benchmark_limit is not None:
                torch.backends.cudnn.benchmark_limit = previous_limit
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = saved[
                :2
            ]
            torch.use_deterministic_algorithms(saved[2], warn_only=saved[3])


def _replicated_visual_latent(latent, encode_mode):
    """Keep replicated SP inputs identical when encoder algorithms can differ.

    Benchmark-selected convolutions need not agree between rank processes.
    Broadcasting the sampled, normalized latent preserves the posterior RNG
    mapping. This does not make performance encoding reproducible across runs;
    use deterministic encoding for that requirement.
    """
    if encode_mode != "performance" or not torch.distributed.is_initialized():
        return latent
    from sglang.multimodal_gen.runtime.distributed import get_sp_group

    return get_sp_group().broadcast(latent, src=0)


class ZingH3Codecs(ZingH3MediaInput):
    def __init__(
        self, modules, pipeline_config, runtime, *, reference_cache_bytes=256 << 20
    ):
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
        from sglang.multimodal_gen.configs.zing_media import ZingResizeLimits

        self.reference_resize = ZingResizeLimits.parse(
            getattr(pipeline_config, "zing_reference_resize", None),
            multiple=self.canvas_multiple,
        )
        self.max_source_target_aspect_ratio_factor = float(
            media["max_source_target_aspect_ratio_factor"]
        )
        self.audio_sample_rate = self.aa.sample_rate
        self.remote_read_fs = None
        # CPU tensors only; lifetime is the loaded codecs/model, not a session.
        # Both preparation products share one byte-accounted LRU budget.
        self.reference_cache = ConditioningCache(reference_cache_bytes)
        self.decoder = decoder_adapter(pipeline_config.resolve_decoder_spec())

    def prepare_visual_encoder_weights(self):
        """Prepare shared precision, weights and layout strategies without inference."""
        from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.progressive_layout import (
            make_encoder_layout,
        )
        from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_cnn import (
            EncoderFCN3D,
        )

        _validate_encoder_benchmark_limit(
            self.runtime["vae_config"].get("cudnn_benchmark_limit")
        )
        model = self.modules["video_vae"]
        dtype = getattr(torch, self.runtime["vae_config"]["encode_dtype"])
        model.encoder.to(dtype=dtype)
        model.quant_conv.to(dtype=dtype)
        # Sessions share prepared strategies; singleton startup keeps native arithmetic.
        native = make_encoder_layout(model)
        model._progressive_encoder_layouts = (native, native)
        if (
            os.environ.get("MINIMAX_H3_PROGRESSIVE_CHANNELS_LAST", "1") == "1"
            and self.runtime["vae_config"].get("encode_mode") == "performance"
            and dtype == torch.bfloat16
            and isinstance(model.encoder, EncoderFCN3D)
            and model.encoder_tiling
            and next(model.encoder.parameters()).is_cuda
            and torch.version.cuda is not None
        ):
            # Keep whole-clip/reference encoding on its original weights/layout.
            # These immutable serving copies are shared by progressive sessions.
            model._progressive_encoder_weights = {
                layer: layer.weight.detach().clone(memory_format=torch.channels_last_3d)
                for layer in model.encoder.modules()
                if isinstance(layer, torch.nn.Conv3d) and layer.kernel_size != (1, 1, 1)
            }
            model._progressive_encoder_layouts = (
                native,
                make_encoder_layout(model, weights=model._progressive_encoder_weights),
            )
        else:
            model._progressive_encoder_weights = {}

    def warmup_source_vae(self):
        """Warm prepared source tile kernels before the caller advertises ready."""
        model = self.modules["video_vae"]
        layouts = getattr(model, "_progressive_encoder_layouts", None)
        if layouts is None or not layouts[1].channels_last:
            return
        from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.progressive import (
            CausalEncoderCache,
        )

        # This temporary replicated cache joins no collectives or real session IDs.
        with (
            torch.no_grad(),
            _visual_encode_policy(
                "performance", self.runtime["vae_config"].get("cudnn_benchmark_limit")
            ),
        ):
            warmup = CausalEncoderCache(
                model,
                tile_mode="replicated",
                encode_mode="performance",
                encode_dtype="bfloat16",
            )
            try:
                for frames in (1, 8):
                    pixels = next(model.encoder.parameters()).new_zeros(
                        (1, 3, frames, model.tile_size, model.tile_size)
                    )
                    _, pending, work = warmup.forward(pixels, "startup")
                    warmup.commit(pending, work)
            finally:
                warmup.clear()
            torch.cuda.synchronize(next(model.encoder.parameters()).device)

    @staticmethod
    def _stats(config, latent, audio=False):
        shape = (1, 1, 1, -1) if audio else (1, -1, 1, 1, 1)
        mean = latent.new_tensor(config.latents_mean, dtype=torch.float32).view(shape)
        std = latent.new_tensor(config.latents_std, dtype=torch.float32).view(shape)
        return mean, std

    def validate_compute_inputs(
        self,
        prompt,
        media,
        *,
        target_shape=None,
        target_frames=None,
        source=None,
        output_type=None,
    ):
        """CPU metadata gate before Qwen/VAE, also used by prompt/reference updates.

        This compute envelope is separate from shared KV capacity. It bounds
        Qwen input and reference encoding; exact packed KV is checked later.
        """
        plan = getattr(self, "compute_plan", None)
        if plan is None:
            return
        if output_type == "video_audio":
            raise ConditionError(
                "CONDITION_INVALID",
                "This server supports video-only output",
            )
        if target_frames is not None and target_frames > 720:
            raise ConditionError(
                "CONDITION_INVALID", "This video is too long for this server"
            )
        if source is not None:
            frames = source["frames"]
            if (
                frames.ndim != 4
                or frames.shape[1] > 720
                or not _source_compute_canvas_supported(*frames.shape[-2:])
            ):
                raise ConditionError(
                    "CONDITION_INVALID",
                    "This source video is too large or too long for this server",
                )
        if len(prompt.encode("utf-8")) > 65536:
            raise ConditionError(
                "CONDITION_INVALID", "This prompt is too long for this server"
            )
        # Continuous source presentation adds one image outside the reference set.
        references = [item for item in media if item.get("origin") != "source_s0"]
        if len(references) > self.pipeline_config.zing_reference_max_images:
            raise ConditionError(
                "CONDITION_INVALID", "This request has too many reference images"
            )
        if target_shape is not None:
            self._validate_compute_canvas(*target_shape)
        for item in media:
            frames = item.get("frames")
            if frames is None or frames.ndim != 4 or frames.shape[1] != 1:
                raise ConditionError(
                    "CONDITION_INVALID",
                    "This media configuration is not supported",
                )
            h, w = map(int, frames.shape[-2:])
            if min(h, w) > self.max_short_edge or h * w > _STATIC_REFERENCE_MAX_AREA:
                raise ConditionError(
                    "CONDITION_INVALID",
                    "These reference images are too large for this server",
                )

    def _validate_compute_canvas(self, height, width):
        from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
            ZingKVPoolAdmissionError,
        )

        try:
            self.compute_plan.validate_canvas(height, width)
        except ZingKVPoolAdmissionError as exc:
            logger.warning("Compute canvas rejected: %s", exc.detail)
            raise ConditionError(
                "CONDITION_INVALID", "This video size is not supported"
            ) from exc

    def validate_prepared_compute_inputs(self, inputs):
        """Check persisted preparation metadata before copying it back to CUDA."""
        plan = getattr(self, "compute_plan", None)
        if plan is None:
            return
        if inputs.audio_shape is not None or inputs.audio:
            raise ConditionError(
                "CONDITION_INVALID",
                "This server supports video-only output",
            )
        if inputs.pixel_frame_count > 720:
            raise ConditionError(
                "CONDITION_INVALID", "This video is too long for this server"
            )
        source = getattr(inputs, "source_visual", None)
        if source is not None and not _source_compute_canvas_supported(
            source.shape[-2] * self.spatial_ratio,
            source.shape[-1] * self.spatial_ratio,
        ):
            raise ConditionError(
                "CONDITION_INVALID", "This source video is too large for this server"
            )
        _, h, w = inputs.visual_shape
        self._validate_compute_canvas(h * self.spatial_ratio, w * self.spatial_ratio)
        if inputs.positive["context"].shape[0] > 16384:
            raise ConditionError(
                "CONDITION_INVALID",
                "This prompt and its reference images are too large for this server",
            )
        if len(inputs.conditions) > self.pipeline_config.zing_reference_max_images:
            raise ConditionError(
                "CONDITION_INVALID", "This request has too many reference images"
            )
        for value in inputs.visual.values():
            if (
                value.shape[2] != 1
                or value.shape[-2] * value.shape[-1] * self.spatial_ratio**2
                > _STATIC_REFERENCE_MAX_AREA
            ):
                raise ConditionError(
                    "CONDITION_INVALID",
                    "These reference images are too large for this server",
                )

    def encode_condition(self, prompt, media):
        if getattr(self, "conditioning_mode", "local") != "local":
            raise ConditionError(
                "CONDITION_INVALID", "External conditioning cannot use a local encoder"
            )
        self.validate_compute_inputs(prompt, media)
        return encode_condition(
            self.modules,
            self.runtime,
            prompt,
            media,
            max_rows=16384 if getattr(self, "compute_plan", None) is not None else None,
            presentation_fn=minimax_h3_ref2va_video_presentation,
        )

    def encode_visual(self, frames, *, metric_operation="source"):
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

        encode_mode = self.runtime["vae_config"]["encode_mode"]
        err_payload = None
        caught = None
        normalized = None
        try:
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
            with (
                component_timer(
                    "encoder",
                    operation=metric_operation,
                    frames=count,
                    height=frames.shape[-2],
                    width=frames.shape[-1],
                ),
                _visual_encode_policy(
                    encode_mode,
                    self.runtime["vae_config"].get("cudnn_benchmark_limit"),
                ),
            ):
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
            normalized = ((latent - mean) / std).half()
        except BaseException as exc:
            caught = exc
            err_payload = (type(exc).__name__, str(exc))

        if encode_mode == "performance" and torch.distributed.is_initialized():
            from sglang.multimodal_gen.runtime.distributed import get_sp_group

            sp_group = get_sp_group()
            if (
                getattr(sp_group, "world_size", 1) > 1
                and getattr(sp_group, "cpu_group", None) is not None
            ):
                gathered = [None] * sp_group.world_size
                torch.distributed.all_gather_object(
                    gathered, err_payload, group=sp_group.cpu_group
                )
                if caught is not None:
                    raise caught
                for item in gathered:
                    if item is not None:
                        err_type, err_msg = item
                        raise RuntimeError(
                            f"Visual encoding failed on a peer SP rank ({err_type}): {err_msg}"
                        )
        if caught is not None:
            raise caught
        return _replicated_visual_latent(normalized, encode_mode)

    def encode_reference(self, item):
        cache = getattr(self, "reference_cache", None)
        if item["type"] != "image" or cache is None or cache.max_bytes == 0:
            return self.encode_visual(item["frames"], metric_operation="reference")
        from sglang.multimodal_gen.runtime.conditioning.worker import (
            coordinate_validation,
        )

        frames = item["frames"]
        model = self.modules["video_vae"]
        profile = json.dumps(
            {
                "shape": list(frames.shape),
                "dtype": str(frames.dtype),
                "vae": self.runtime["vae_config"],
                "mean": self.va.latents_mean,
                "std": self.va.latents_std,
                "cudnn_benchmark_limit": torch.backends.cudnn.benchmark_limit,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(profile.encode())
        digest.update(frames.contiguous().numpy().tobytes())
        identity = digest.hexdigest()
        key = ("vae", weakref.ref(model), identity)
        cached = cache.get(key)
        receipts = coordinate_validation((identity, cached is not None))
        if any(receipt[0] != identity for receipt in receipts):
            raise ValueError("Reference encoding inputs differ across SP ranks")
        # A partial hit must not skip encode_visual's replicated-latent broadcast.
        if all(receipt[1] for receipt in receipts):
            return cached["latent"].to(device=next(model.parameters()).device)
        latent = self.encode_visual(frames, metric_operation="reference")
        cache.put(key, {"latent": latent.detach().cpu()})
        return latent

    def _log_reference_cache(self):
        cache = getattr(self, "reference_cache", None)
        if cache is not None:
            logger.info(
                "Zing reference cache: %s",
                json.dumps(
                    {
                        "total": cache.snapshot(),
                        "preprocess": cache.snapshot(tenant="preprocess"),
                        "vae": cache.snapshot(tenant="vae"),
                    },
                    sort_keys=True,
                ),
            )

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
        with _visual_encode_policy(
            self.runtime["vae_config"]["encode_mode"],
            self.runtime["vae_config"].get("cudnn_benchmark_limit"),
        ):
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
        normalized = ((latent - mean) / std).half()
        return _replicated_visual_latent(
            normalized, self.runtime["vae_config"]["encode_mode"]
        )

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
        source_preview=None,
        conditioning=None,
        stage_trace=None,
        preparation_metrics=None,
    ):
        media_started = time.perf_counter()
        # Reuse decoded references for prompt refreshes; offline calls release them.
        if isinstance(request, dict) and "cache_uri" in request:
            raise ValueError(
                "Zing runtime accepts media messages, not pickled request caches"
            )
        if conditioning is not None and not callable(conditioning):
            raise ConditionError(
                "CONDITION_INVALID",
                "External conditioning requires a trusted worker consumer",
            )
        if (
            conditioning is None
            and getattr(self, "conditioning_mode", "local") != "local"
        ):
            raise ConditionError(
                "CONDITION_INVALID", "External conditioning descriptor is missing"
            )
        error = None
        try:
            # 请求可局部覆盖 source/reference 面积；必须在解码前失败，且不改常驻配置。
            max_area_config = _sample_max_area_config(self.max_area_config, request)
            prompt, items, target, conditions, source = self._parse_inference_request(
                request
            )
            audio_ids = {
                c["content_id"] for c in conditions if "audio" in c["latent_modalities"]
            }
            reference_ids = {c["content_id"] for c in conditions}
            media = [
                (
                    self._decode_reference_image(item, max_area_config=max_area_config)
                    if item["type"] == "image" and item["id"] in reference_ids
                    else self._decode_media(item, need_audio=item["id"] in audio_ids)
                )
                for item in items
            ]
            if source_preview is not None:
                # Internal decoded S0 from the accepted source command, never wire
                # media or a URI. Keep it on the Qwen track, outside VAE conditions.
                if any(
                    item["type"] in {"video", "video_audio"} for item in items
                ) or any(item["id"] == source_preview["id"] for item in items):
                    raise ValueError(
                        "Source presentation conflicts with explicit media"
                    )
                media.insert(0, dict(source_preview, origin="source_s0"))
            source = (
                None if source is None else self._decode_media(source, need_audio=False)
            )
            normalized_media, source, h, w, t, frames = self._normalize_inference_media(
                media, source, conditions, target, max_area_config=max_area_config
            )
            self.validate_compute_inputs(
                prompt,
                normalized_media,
                target_shape=(h, w),
                target_frames=frames,
                source=source,
                output_type=target["type"],
            )
        except (ValueError, OSError, ConditionError) as exc:
            if conditioning is None:
                raise
            error = exc
        if conditioning is not None:
            from sglang.multimodal_gen.runtime.conditioning.worker import (
                coordinate_validation,
            )

            # Agree on CPU preparation before any rank enters Qwen/VAE collectives.
            if not all(coordinate_validation(error is None)):
                if isinstance(error, ConditionError):
                    raise error
                raise ConditionError(
                    "CONDITION_INVALID", "Media normalization failed"
                ) from error
            if preparation_metrics is not None:
                preparation_metrics.record(
                    "media_prepare", (time.perf_counter() - media_started) * 1000
                )
            with (
                preparation_metrics.measure("qwen_remote_prepare")
                if preparation_metrics is not None
                else nullcontext()
            ):
                positive = conditioning(prompt, normalized_media)
            if positive is None:
                raise ConditionError(
                    "CONDITION_INVALID",
                    "The external conditioning provider returned no condition",
                )
        else:
            positive = None
            if preparation_metrics is not None:
                preparation_metrics.record(
                    "media_prepare", (time.perf_counter() - media_started) * 1000
                )
        request_id = stage_trace.identity["request_id"] if stage_trace else None
        measure_cuda = stage_trace is not None and stage_trace.measure_cuda
        if positive is None:
            with stage_span(
                stage_trace,
                "Encoder",
                component="qwen",
                cuda=measure_cuda,
                input_ids=[f"{request_id}:payload"],
                output_ids=[f"{request_id}:positive"],
            ) as span:
                with (
                    (
                        preparation_metrics.measure("qwen_prepare")
                        if preparation_metrics is not None
                        else nullcontext()
                    ),
                    use_component("text_encoder"),
                ):
                    positive = self.encode_condition(prompt, normalized_media)
            handoff(span)
        visual, audio = {}, {}
        modes = {c["content_id"]: c["latent_modalities"] for c in conditions}
        visual_refs = [
            item for item in normalized_media if "visual" in modes.get(item["id"], [])
        ]
        source_latent = None
        if visual_refs or source is not None:
            with use_component("video_vae"):
                for item in visual_refs:
                    with stage_span(
                        stage_trace,
                        "Encoder",
                        component="reference_vae",
                        cuda=measure_cuda,
                        media_id=item["id"],
                        input_ids=[f"{request_id}:media:{item['id']}"],
                        output_ids=[f"{request_id}:reference:{item['id']}"],
                    ) as span:
                        with (
                            preparation_metrics.measure("reference_vae")
                            if preparation_metrics is not None
                            else nullcontext()
                        ):
                            visual[item["id"]] = self.encode_reference(item)
                    handoff(span)
                if source is not None:
                    with stage_span(
                        stage_trace,
                        "Encoder",
                        component="source_vae",
                        cuda=measure_cuda,
                        input_ids=[f"{request_id}:source_pixels"],
                        output_ids=[f"{request_id}:source_latents"],
                        source_frame_range=[0, int(source["frames"].shape[1])],
                    ) as span:
                        source_latent = self.encode_visual(source["frames"])
                    handoff(span)
        audio_refs = [
            item for item in normalized_media if "audio" in modes.get(item["id"], [])
        ]
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
        self._log_reference_cache()
        return H3InferenceInputs(
            target["type"],
            conditions,
            [{"id": item["id"], "type": item["type"]} for item in normalized_media],
            positive,
            visual,
            audio,
            source_latent,
            (t, h // self.spatial_ratio, w // self.spatial_ratio),
            audio_shape,
            frames,
            samples,
            self.audio_sample_rate,
            condition_media=normalized_media if retain_condition_media else None,
            max_area_config=max_area_config,
        )

    def prepare_reference_media(self, active, references):
        """CPU-only preparation shared by early Qwen and boundary admission."""
        old_ids = {item["content_id"] for item in active.conditions}
        media = [item for item in active.condition_media if item["id"] not in old_ids]
        ids = [item["id"] for item in media + references]
        if len(ids) != len(set(ids)) or any(
            item["type"] != "image" for item in references
        ):
            raise ValueError(
                "Reference IDs must not conflict with retained presentation"
            )
        max_area_config = (
            getattr(active, "max_area_config", None) or self.max_area_config
        )
        decoded_references = [
            self._decode_reference_image(item, max_area_config=max_area_config)
            for item in references
        ]
        conditions = [
            dict(
                content_id=item["id"],
                role="reference",
                latent_modalities=["visual"],
            )
            for item in references
        ]
        target = dict(
            type=active.output_type,
            output=dict(
                frames=active.pixel_frame_count,
                height=active.visual_shape[1] * self.spatial_ratio,
                width=active.visual_shape[2] * self.spatial_ratio,
            ),
        )
        normalized, _, h, w, t, _ = self._normalize_inference_media(
            decoded_references,
            None,
            conditions,
            target,
            max_area_config=max_area_config,
        )
        if (
            t,
            h // self.spatial_ratio,
            w // self.spatial_ratio,
        ) != active.visual_shape:
            raise ValueError("Reference updates cannot change output geometry")
        return media + normalized, normalized, conditions

    def prepare_reference_update(
        self, active, references, prompt, use_component, *, conditioning
    ):
        """Build a complete candidate without changing active media or stream state.

        CPU decode/geometry failures must agree across SP ranks before any Qwen
        or VAE collective. GPU/runtime failures propagate and close the session.
        """
        from sglang.multimodal_gen.runtime.conditioning.worker import (
            coordinate_validation,
        )

        error = None
        try:
            media, normalized, conditions = self.prepare_reference_media(
                active, references
            )
        except (ValueError, OSError) as exc:
            error = exc
        if not all(coordinate_validation(error is None)):
            raise ConditionError(
                "CONDITION_INVALID", "Reference media preparation failed"
            ) from error
        # Retained presentation is already normalized; only new references change.
        self.validate_compute_inputs(prompt, media)
        positive = conditioning(prompt, media)
        candidate = copy(active)
        candidate.conditions = conditions
        candidate.condition_media = media
        candidate.media = [{"id": item["id"], "type": item["type"]} for item in media]
        candidate.positive = positive
        candidate.visual, candidate.audio = {}, {}
        candidate.source_visual = None
        reference_ids = {item["id"] for item in references}
        if reference_ids:
            with use_component("video_vae"):
                for item in normalized:
                    if item["id"] in reference_ids:
                        candidate.visual[item["id"]] = self.encode_reference(item)
        self._log_reference_cache()
        return candidate

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
        return self.decoder.module_key

    def decode_visual(self, latent, frames):
        if self.decoder.stateful:
            return self.decoder.decode(
                self.modules[self.decoder.module_key], latent, frames
            )
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
