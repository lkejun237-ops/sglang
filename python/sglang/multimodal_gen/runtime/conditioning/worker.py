# SPDX-License-Identifier: Apache-2.0
"""Worker conditioners with CPU consensus before any GPU broadcast."""

import logging
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState
from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
    DEFAULT_MAX_REFERENCE_IMAGES,
)

from .contract import ConditionError, DeploymentProfile, safe_json
from .inputs import (
    PREPROCESS_VERSION,
    content_identity,
    hash_frame,
    validate_condition_media,
)

logger = logging.getLogger(__name__)
EXCLUDED_ENCODER_COMPONENTS = frozenset({"text_encoder", "tokenizer", "processor"})


class WorkerConditioningState(BaseRealtimeState):
    def __init__(self):
        self.identity = self.positive = None
        self.media_hashes = {}
        self.prefetch = None

    def dispose(self):
        try:
            if self.prefetch is not None:
                self.prefetch.close()
                self.prefetch = None
        finally:
            self.identity = self.positive = None
            self.media_hashes.clear()


def _identity(prompt, media, profile_id, state, preprocess_version):
    items = validate_condition_media(media)
    hashes, current = [], {}
    for original, item in zip(media, items):
        tensor = original["frames"]
        key = id(tensor)
        # Inference-mode tensors have no version counter, so in-place rewrites
        # are invisible; only reuse a hash when a version can prove no mutation.
        try:
            signature = (tensor.data_ptr(), tuple(tensor.shape), tensor._version)
        except RuntimeError:
            signature = None
        previous = state.media_hashes.get(key)
        if (
            signature is not None
            and previous is not None
            and previous[0] is tensor
            and previous[1] == signature
        ):
            digest = previous[2]
        else:
            digest = hash_frame(item["frames"])
        current[key] = (tensor, signature, digest)
        hashes.append(digest)
    state.media_hashes = current
    return content_identity(
        prompt, items, profile_id, preprocess_version, item_hashes=hashes
    )


def _sp_group():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return None
    from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_group

    return get_sp_group()


def coordinate_validation(receipt):
    group = _sp_group()
    if group is None or group.world_size == 1:
        return [receipt]
    values = [None] * group.world_size
    torch.distributed.all_gather_object(values, receipt, group=group.cpu_group)
    return values


def broadcast_receipt(receipt):
    group = _sp_group()
    if group is None or group.world_size == 1:
        return receipt
    values = [receipt]
    torch.distributed.broadcast_object_list(
        values, src=group.ranks[0], group=group.cpu_group
    )
    return values[0]


def _broadcast_tensor(tensor, name):
    group = _sp_group()
    if group is not None and group.world_size > 1:
        torch.distributed.broadcast(
            tensor, src=group.ranks[0], group=group.device_group
        )
    return tensor


def _allocate(shapes, device, positive):
    return {
        name: positive[name].to(device=device)
        if positive is not None
        else torch.empty(
            shape,
            dtype=torch.bfloat16 if name == "context" else torch.int64,
            device=device,
        )
        for name, shape in shapes.items()
    }


def _failure(receipts):
    for receipt in receipts:
        if type(receipt) is not dict or receipt.get("ok") is not True:
            return (
                receipt.get("code", "CONDITION_INVALID")
                if type(receipt) is dict
                else "CONDITION_INVALID"
            )
    return None


def _cpu_timeout(group):
    if group is None:
        return None
    backend = group.cpu_group._get_backend(torch.device("cpu"))
    timeout = getattr(backend.options, "_timeout", None)
    if timeout is None:
        raise ConditionError(
            "CONDITION_INVALID", "Cannot determine the SP CPU process group timeout"
        )
    return timeout.total_seconds()


class RemoteConditioner:
    def __init__(
        self,
        client,
        profile_id,
        *,
        rank=None,
        coordinate=coordinate_validation,
        broadcast_receipt=broadcast_receipt,
        allocate=_allocate,
        broadcast_tensor=_broadcast_tensor,
        preprocess_version=PREPROCESS_VERSION,
        validate_input=None,
    ):
        group = _sp_group() if rank is None else None
        self.rank = (
            (0 if group is None else group.rank_in_group) if rank is None else rank
        )
        self.client, self.profile_id = client, profile_id
        self.coordinate, self.broadcast_receipt = coordinate, broadcast_receipt
        self.allocate, self.broadcast_tensor = allocate, broadcast_tensor
        self.preprocess_version = preprocess_version
        self.validate_input = validate_input

    @classmethod
    def from_server_args(cls, args):
        from .client import ConditioningClient
        from .resources import ResourceLimits

        group = _sp_group()
        rank = 0 if group is None else group.rank_in_group
        client = profile = None
        try:
            path = Path(args.conditioning_config)
            if path.stat().st_size > 1 << 20:
                raise ConditionError(
                    "CONDITION_INVALID",
                    "Conditioning client configuration exceeds the byte limit",
                )
            config = safe_json(path.read_bytes())
            profile = DeploymentProfile.load(args.conditioning_expected_profile)
            max_media = (
                getattr(
                    getattr(args, "pipeline_config", None),
                    "zing_reference_max_images",
                    DEFAULT_MAX_REFERENCE_IMAGES,
                )
                + 1
            )
            limits = ResourceLimits(
                **{"max_media": max_media, **config.get("limits", {})}
            )
            client = ConditioningClient(
                args.conditioning_service_url,
                config.get("bearer_token", config.get("token")),
                profile.profile_id,
                timeout_s=config.get("conditioning_timeout_s", 10),
                sp_timeout_s=_cpu_timeout(group),
                max_rows=limits.max_condition_rows,
                max_request_bytes=limits.max_request_bytes,
                max_bytes=limits.max_response_bytes,
                media_limits=dict(
                    max_media=limits.max_media,
                    max_side=limits.max_side,
                    max_input_bytes=limits.max_input_bytes,
                    max_decoded_pixels=limits.max_decoded_pixels,
                ),
                result_transport=config.get("result_transport", "http"),
                retry_initial_s=config.get("retry_initial_s", 0.05),
                retry_max_s=config.get("retry_max_s", 0.5),
            )
            receipt = {"ok": True, "digest": profile.profile_id, "cached": False}
        except Exception as exc:
            logger.exception("Conditioning client configuration failed")
            receipt = {
                "ok": False,
                "code": exc.code
                if isinstance(exc, ConditionError)
                else "CONDITION_UNAVAILABLE",
            }
        receipts = coordinate_validation(receipt)
        error = _failure(receipts)
        if error or len({x.get("digest") for x in receipts}) != 1:
            raise ConditionError(error or "CONDITION_INVALID")
        handshake = None
        if rank == 0:
            try:
                client.handshake()
                handshake = {"ok": True}
            except Exception as exc:
                logger.exception("Conditioning profile handshake failed")
                handshake = {
                    "ok": False,
                    "code": exc.code
                    if isinstance(exc, ConditionError)
                    else "CONDITION_UNAVAILABLE",
                }
        error = _failure([broadcast_receipt(handshake)])
        if error:
            raise ConditionError(error)
        return cls(
            client,
            profile.profile_id,
            rank=rank,
            preprocess_version=profile.document["preprocess_version"],
            validate_input=client.validate_input,
        )

    def __call__(self, prompt, media, *, device, state=None):
        state = state if state is not None else WorkerConditioningState()
        try:
            if self.validate_input is not None:
                self.validate_input(prompt, media)
            digest = _identity(
                prompt, media, self.profile_id, state, self.preprocess_version
            )
            receipt = {
                "ok": True,
                "digest": digest,
                "cached": state.identity == digest and state.positive is not None,
            }
        except Exception as exc:
            logger.exception("Conditioning input validation failed")
            receipt = {
                "ok": False,
                "code": exc.code
                if isinstance(exc, ConditionError)
                else "CONDITION_INVALID",
                "digest": None,
                "cached": False,
            }
        receipts = self.coordinate(receipt)
        error = _failure(receipts)
        if (
            error
            or len({x.get("digest") for x in receipts}) != 1
            or len({x.get("cached") for x in receipts}) != 1
        ):
            raise ConditionError(error or "CONDITION_INVALID")
        if receipt["cached"]:
            return state.positive
        positive = announcement = None
        if self.rank == 0:
            try:
                if getattr(self.client, "result_transport", "http") == "nccl":
                    positive = self.client.encode_device(
                        prompt, media, input_digest=digest, device=device
                    )
                else:
                    encoder = state.prefetch or self.client
                    positive = encoder.encode(prompt, media, input_digest=digest)
                announcement = {
                    "ok": True,
                    "shapes": {
                        name: tuple(tensor.shape) for name, tensor in positive.items()
                    },
                }
            except Exception as exc:
                logger.exception("Conditioning RPC failed")
                announcement = {
                    "ok": False,
                    "code": exc.code
                    if isinstance(exc, ConditionError)
                    else "CONDITION_UNAVAILABLE",
                }
        announcement = self.broadcast_receipt(announcement)
        error = _failure([announcement])
        if error:
            raise ConditionError(error)
        transferred = None
        try:
            transferred = self.allocate(announcement["shapes"], device, positive)
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            transfer = {"ok": True}
        except Exception:
            logger.exception("Conditioning allocation or host transfer failed")
            transfer = {"ok": False, "code": "CONDITION_UNAVAILABLE"}
        error = _failure(self.coordinate(transfer))
        if error:
            raise ConditionError(error)
        for name in ("context", "row_role"):
            self.broadcast_tensor(transferred[name], name)
        state.identity, state.positive = digest, transferred
        return transferred


class LocalConditioner:
    def __init__(
        self,
        encode,
        profile_id,
        *,
        cache_bytes=256 << 20,
        preprocess_version=PREPROCESS_VERSION,
    ):
        from .cache import ConditioningCache

        self.encode, self.profile_id, self.preprocess_version = (
            encode,
            profile_id,
            preprocess_version,
        )
        self.cache = ConditioningCache(cache_bytes)
        self.identity_state = WorkerConditioningState()
        self.device = torch.device("cpu")

    def __call__(self, prompt, media, *, state=None):
        # Preserve native local multi-frame and audio inputs; cache the single
        # frame subset only, without imposing remote output-row limits.
        if self.cache.max_bytes == 0 or any(
            item.get("type") not in {"image", "video"}
            or not isinstance(item.get("frames"), torch.Tensor)
            or item["frames"].ndim != 4
            or item["frames"].shape[1] != 1
            for item in media
        ):
            return self.encode(prompt, media)
        try:
            digest = _identity(
                prompt,
                media,
                self.profile_id,
                state or self.identity_state,
                self.preprocess_version,
            )
        except ConditionError:
            # Local native inputs may exceed the RPC validator's bounds.
            # Such inputs retain the original encoder path without caching.
            return self.encode(prompt, media)

        original = []

        def encode():
            result = self.encode(prompt, media)
            self.device = result["context"].device
            original.append(result)
            return {
                name: tensor.detach().to(device="cpu")
                for name, tensor in result.items()
            }

        result = self.cache.get_or_encode(digest, encode)
        if original:
            # The encoding caller already owns its native-device result. Cache
            # CPU copies independently; avoid an unnecessary second H2D copy.
            return original[0]
        return {name: tensor.to(device=self.device) for name, tensor in result.items()}
