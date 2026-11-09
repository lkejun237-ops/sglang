# SPDX-License-Identifier: Apache-2.0
"""Administrator deployment profile, RPC service and pixel-exact warmup tools."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import fields
from pathlib import Path


def parser():
    value = argparse.ArgumentParser(description="Native TP1 Zing conditioning RPC")
    commands = value.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build-profile")
    builder.add_argument("--model", required=True)
    builder.add_argument("--numerical-environment", required=True)
    builder.add_argument("--projection", default=None)
    builder.add_argument("--output", required=True)
    for name in ("serve", "warmup"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--profile", required=True)
        if name == "serve":
            command.add_argument(
                "--conditioning-result-transport",
                choices=("http", "nccl"),
                default=None,
                help="Override config result_transport (nccl if absent); must match the DiT client.",
            )
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8093)
        else:
            command.add_argument(
                "--input",
                required=True,
                help="JSON list: prompt, ordered images, optional RGB s0_image",
            )
            command.add_argument("--url", default=None)
    return value


def load_configuration(args):
    from .contract import DeploymentProfile, safe_json
    from .resources import ResourceLimits

    config = safe_json(Path(args.config).read_bytes())
    transport = getattr(args, "conditioning_result_transport", None)
    if transport is not None:
        if transport not in {"http", "nccl"}:
            raise ValueError("Unknown conditioning result transport")
        if type(config) is dict:
            config["result_transport"] = transport
    required_limits = {field.name for field in fields(ResourceLimits)}
    if (
        type(config) is not dict
        or type(config.get("limits")) is not dict
        or set(config["limits"]) != required_limits
    ):
        raise ValueError("All conditioning limits must be explicitly configured")
    limits = ResourceLimits(**config["limits"])
    if not isinstance(config.get("tokens"), dict) or not config["tokens"]:
        raise ValueError("Conditioning tokens must be explicitly configured")
    for token, principal in config["tokens"].items():
        if (
            type(token) is not str
            or not token
            or not token.isascii()
            or type(principal) is not dict
            or set(principal) - {"tenant", "admin"}
            or type(principal.get("tenant")) is not str
            or not principal["tenant"]
            or type(principal.get("admin", False)) is not bool
        ):
            raise ValueError("Invalid conditioning token principal")
    native = config.get("native")
    if (
        type(native) is not dict
        or type(native.get("model_path")) is not str
        or not native["model_path"]
        or type(native.get("gpu_id")) is not int
        or native["gpu_id"] < 0
    ):
        raise ValueError(
            "Conditioning native model path and GPU must be explicitly configured"
        )
    prompt_tokens = native.get("max_prompt_tokens")
    if prompt_tokens is not None and (
        type(prompt_tokens) is not int or prompt_tokens <= 0
    ):
        raise ValueError("Prompt token budget must be a positive integer")
    return config, DeploymentProfile.load(args.profile), limits


logger = logging.getLogger(__name__)


def load_warmup_input(document):
    import numpy as np
    import torch
    from PIL import Image

    from .contract import _invalid
    from .inputs import validate_condition_media, validate_prompt

    if type(document) is not dict or set(document) - {"prompt", "images", "s0_image"}:
        _invalid("Invalid warmup entry fields")
    prompt = validate_prompt(document.get("prompt"))
    images = document.get("images", [])
    if (
        type(images) is not list
        or len(images) > 16
        or any(type(path) is not str or not path for path in images)
    ):
        _invalid("Invalid warmup image paths")
    paths = (
        [("video", document["s0_image"])]
        if document.get("s0_image") is not None
        else []
    )
    paths.extend(("image", path) for path in images)
    if len(paths) > 16:
        _invalid("Warmup media count exceeds the input budget")
    media = []
    for kind, path in paths:
        if type(path) is not str or not path:
            _invalid("Invalid warmup image path")
        try:
            with Image.open(path) as image:
                if not 0 < image.width <= 8192 or not 0 < image.height <= 8192:
                    _invalid("Warmup media dimensions exceed the input budget")
                array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        except (OSError, ValueError):
            _invalid("Warmup RGB image is unavailable")
        media.append(
            dict(
                type=kind,
                fps=1.0,
                frames=torch.from_numpy(array)
                .permute(2, 0, 1)
                .unsqueeze(1)
                .contiguous(),
            )
        )
    return prompt, validate_condition_media(media)


async def warmup(args, config, profile, limits):
    import httpx

    from .client import validate_service_url
    from .contract import ConditionError, safe_json
    from .inputs import content_identity
    from .wire import decode_response, encode_request, unpack_message

    document = safe_json(Path(args.input).read_bytes())
    if type(document) is not list:
        raise ValueError("Warmup input must be a JSON list")
    url = args.url or config.get("service_url")
    url = validate_service_url(url)
    token = config.get("bearer_token")
    if not token:
        candidates = [
            token
            for token, principal in config["tokens"].items()
            if not principal.get("admin")
        ]
        if len(candidates) != 1:
            raise ValueError("Warmup requires one explicit bearer token")
        token = candidates[0]
    counts = {"hit": 0, "miss": 0, "joined": 0}
    async with httpx.AsyncClient(
        base_url=url.rstrip("/"),
        headers={"Authorization": "Bearer " + token},
        timeout=limits.request_timeout_s,
    ) as client:
        handshake = await client.get("/v1/conditioning/profile")
        if (
            handshake.status_code != 200
            or handshake.json().get("profile_id") != profile.profile_id
        ):
            raise ConditionError("CONDITION_PROFILE_MISMATCH")
        for entry in document:
            prompt, media = load_warmup_input(entry)
            digest = content_identity(
                prompt,
                media,
                profile.profile_id,
                profile.document["preprocess_version"],
            )
            body = encode_request(
                prompt,
                media,
                profile.profile_id,
                deadline_ms=limits.request_timeout_s * 1000,
                input_digest=digest,
            )
            if len(body) > limits.max_request_bytes:
                raise ConditionError(
                    "CONDITION_INVALID", "Warmup request exceeds the byte budget"
                )
            async with client.stream(
                "POST",
                "/v1/conditioning/encode",
                content=body,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(body)),
                },
            ) as response:
                parts, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > limits.max_response_bytes:
                        raise ConditionError(
                            "CONDITION_INVALID",
                            "Warmup response exceeds the byte budget",
                        )
                    parts.append(chunk)
                data = b"".join(parts)
                if response.status_code != 200:
                    error = safe_json(data)
                    raise ConditionError(error.get("code", "CONDITION_UNAVAILABLE"))
            decode_response(
                data,
                profile.profile_id,
                digest,
                limits.max_condition_rows,
                limits.max_response_bytes,
            )
            header, _ = unpack_message(data, max_bytes=limits.max_response_bytes)
            counts[header["cache"]] += 1
    print(json.dumps(counts, sort_keys=True))
    return counts


def is_loopback_host(host):
    return host in ("127.0.0.1", "::1", "localhost")


def validate_bind(host, config):
    """Reject an external plaintext bind; runs before the model is loaded."""
    if (
        not is_loopback_host(host)
        and not config.get("tls")
        and config.get("internal_auth_proxy_verified") is not True
    ):
        raise ValueError(
            "External bind requires TLS/mTLS or a verified internal authentication proxy"
        )


def warm_encoder(encoder):
    """Run one synthetic encode so kernel compilation finishes before readiness."""
    from types import SimpleNamespace

    import numpy as np
    import torch

    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )

    row = np.arange(832, dtype=np.uint8)[None, :].repeat(480, 0)
    image = np.stack([row, row[::-1], row // 2], -1)
    frames = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[:, None]
    media = [{"id": "<Image 1>", "type": "image", "frames": frames.contiguous()}]
    with (
        torch.inference_mode(),
        set_forward_context(current_timestep=0, attn_metadata=None),
    ):
        encoder(SimpleNamespace(prompt="warmup", media=media))


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "build-profile":
        from .contract import canonical_json, safe_json
        from .native import build_profile

        profile = build_profile(
            args.model,
            safe_json(Path(args.numerical_environment).read_bytes()),
            projection=args.projection,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_json(profile.document))
        print(
            json.dumps({"profile_id": profile.profile_id, "gpu_validation": "NOT_RUN"})
        )
        return
    config, profile, limits = load_configuration(args)
    if args.command == "warmup":
        return asyncio.run(warmup(args, config, profile, limits))
    import uvicorn

    from .native import NativeConditionEncoder
    from .service import ConditioningService

    validate_bind(args.host, config)
    if config.get("tls") or not is_loopback_host(args.host):
        logger.warning(
            "Non-loopback bind or TLS is experimental; only loopback single-node "
            "deployment is validated"
        )
    result_transport = config.get("result_transport", "nccl")
    if result_transport not in {"http", "nccl"}:
        raise ValueError("Unknown conditioning result transport")
    if result_transport == "nccl" and args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("NCCL conditioning requires an IPv4 loopback bind")
    native = config["native"]
    encoder = NativeConditionEncoder(
        native["model_path"],
        profile,
        max_rows=limits.max_condition_rows,
        gpu_id=native["gpu_id"],
        max_prompt_tokens=native.get("max_prompt_tokens"),
    )
    # The first encode compiles kernels (~30 s on L20D), far above request_timeout_s.
    warm_encoder(encoder)
    tls = config.get("tls")
    options = (
        dict(
            ssl_certfile=tls["certificate"],
            ssl_keyfile=tls["private_key"],
            ssl_ca_certs=tls["ca"],
            ssl_cert_reqs=2,
        )
        if tls
        else {}
    )
    uvicorn.run(
        ConditioningService(
            profile=profile,
            encoder=encoder,
            tokens=config["tokens"],
            limits=limits,
            result_transport=result_transport,
        ),
        host=args.host,
        port=args.port,
        limit_concurrency=limits.max_active_requests,
        h11_max_incomplete_event_size=limits.max_header_bytes,
        **options,
    )


if __name__ == "__main__":
    main()
