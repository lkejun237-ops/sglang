# SPDX-License-Identifier: Apache-2.0
"""Small local requests for already loaded Encoder and Decoder workers."""

import json
import logging
import time

import torch

logger = logging.getLogger(__name__)

WARMUP_PROMPT = "A quiet landscape in daylight."


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def warmup_encoder(codecs):
    """Warm the source VAE, then run one text-only request through Qwen."""
    started = time.monotonic()
    codecs.warmup_source_vae()
    warmup_text_encoder(codecs)
    elapsed_s = round(time.monotonic() - started, 3)
    summary = {
        "role": "encoder",
        "warmed_up": True,
        "elapsed_s": elapsed_s,
        "stages": ["source_vae", "qwen_text_encoder"],
    }
    from sglang.multimodal_gen.runtime.realtime.worker_http import publish_role_warmup

    publish_role_warmup("encoder", summary)
    logger.info("Encoder warmup completed in %.3fs", elapsed_s)
    print("WARMUP_COMPLETED " + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def warmup_text_encoder(codecs):
    """Run one text-only request through the worker's existing Qwen encoder."""
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )

    started = time.monotonic()
    with (
        torch.inference_mode(),
        set_forward_context(current_timestep=0, attn_metadata=None),
    ):
        result = codecs.encode_condition(WARMUP_PROMPT, [])
        context, roles = result["context"], result["row_role"]
        if (
            context.ndim != 2
            or context.shape[0] == 0
            or roles.shape != (context.shape[0],)
            or not torch.isfinite(context).all()
        ):
            raise RuntimeError("Encoder warmup returned invalid text features")
        _synchronize(context.device)
    logger.info("Encoder text warmup completed in %.3fs", time.monotonic() - started)


def warmup_decoder(model, *, group=None, agree=None, width=832, height=480):
    """Decode one temporal latent, using the ordinary request-owned stream.

    ``agree`` has the existing RoleCommandControl.agree signature. Distributed
    workers use it to stop every rank before decoding if input preparation fails,
    and to keep readiness behind all ranks' completed requests.
    """
    started = time.monotonic()
    latent = stream = None
    signature = None
    error = None
    try:
        ratio = model.spatial_ratio
        if (
            type(width) is not int
            or type(height) is not int
            or min(width, height) < 1
            or width % ratio
            or height % ratio
        ):
            raise ValueError(
                "Decoder warmup canvas must be positive and latent aligned"
            )
        device = model.latent_mean.device
        shape = (
            1,
            model.architecture["latent_channels"],
            1,
            height // ratio,
            width // ratio,
        )
        # A private CPU RNG gives every SP rank the same input without touching
        # the RNG used by subsequent real requests.
        latent = torch.randn(
            shape,
            dtype=torch.float32,
            generator=torch.Generator(device="cpu").manual_seed(0),
        ).to(device)
        stream = model.new_stream(frame_limit=1, group=group)
        _synchronize(device)
        signature = (shape, width, height)
    except Exception as exc:
        error = exc
    try:
        if agree is not None:
            agree("decoder_warmup_prepare", signature, error=error)
        if error is not None:
            raise error
        try:
            # Match DecoderService's execution mode so compiled kernels can be
            # reused by normal requests without an inference-tensor variant.
            with torch.no_grad():
                output = stream.push(latent)
                stream.finish(1)
                if (
                    output.shape != (1, 3, 1, height, width)
                    or not torch.isfinite(output).all()
                ):
                    raise RuntimeError("Decoder warmup returned invalid pixels")
                _synchronize(device)
        except Exception as exc:
            error = exc
        if agree is not None:
            agree("decoder_warmup_complete", signature, error=error)
        if error is not None:
            raise error
    finally:
        if stream is not None:
            stream.close()
    elapsed_s = round(time.monotonic() - started, 3)
    summary = {
        "role": "decoder",
        "warmed_up": True,
        "elapsed_s": elapsed_s,
        "width": width,
        "height": height,
        "frames": 1,
    }
    from sglang.multimodal_gen.runtime.realtime.worker_http import publish_role_warmup

    publish_role_warmup("decoder", summary)
    logger.info(
        "Decoder latent warmup completed at %sx%s in %.3fs",
        width,
        height,
        elapsed_s,
    )
    print("WARMUP_COMPLETED " + json.dumps(summary, sort_keys=True), flush=True)
    return summary
