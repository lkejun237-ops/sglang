# SPDX-License-Identifier: Apache-2.0
"""Adapt the existing Block0 owner to Encoder/DiT control messages."""

import time
from copy import deepcopy
from dataclasses import fields
from types import SimpleNamespace

from .packing import H3InferenceInputs
from .preparation_cache import fill_prepared_prefix, inputs_on


def prepared_message(inputs):
    return {field.name: getattr(inputs, field.name) for field in fields(inputs)}


def prepared_inputs(value):
    value = dict(value)
    for name in ("visual_shape", "audio_shape"):
        if value[name] is not None:
            value[name] = tuple(value[name])
    return H3InferenceInputs(**value)


class DiTPreparation:
    """Lives with the model, independently of individual communication groups."""

    def __init__(self, stage, sampler_factory):
        self.stage, self.sampler_factory = stage, sampler_factory

    @property
    def cache(self):
        return self.stage.block0_cache

    def query_prompt(self, context, command):
        """Read published CPU conditions without model admission or collectives."""
        import hashlib

        from .block0_identity import fingerprint

        prompt = command.get("text")
        from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
            validate_prompt_text,
        )

        validate_prompt_text(prompt)
        if not context and self.cache is None:
            return dict(hit=False, cache_key=None, lookup={"cache_result": "disabled"})
        identity = deepcopy(context["identity"])
        request = identity["request"]
        request["prompt"] = prompt
        if "references" in command or "reference_digests" in command:
            digests = "reference_digests" in command
            field = "sha256" if digests else "payload"
            refs = command["reference_digests" if digests else "references"]
            if not isinstance(refs, list):
                raise ValueError("Invalid prompt references")
            media = []
            seen = {"<Video 1>"}
            for ref in refs:
                if (
                    not isinstance(ref, dict)
                    or set(ref) != {"id", "format", field}
                    or not isinstance(ref["id"], str)
                    or not ref["id"]
                    or ref["id"] in seen
                    or ref["format"] not in {"png", "jpeg", "webp"}
                    or (not digests and not isinstance(ref[field], bytes))
                    or (
                        digests
                        and (
                            not isinstance(ref[field], str)
                            or len(ref[field]) != 64
                            or any(c not in "0123456789abcdef" for c in ref[field])
                        )
                    )
                ):
                    raise ValueError("Invalid prompt reference")
                seen.add(ref["id"])
                media.append(
                    dict(
                        id=ref["id"],
                        type="image",
                        format=ref["format"],
                        payload={
                            "sha256": ref[field]
                            if digests
                            else hashlib.sha256(ref[field]).hexdigest()
                        },
                    )
                )
            old_ids = set(context["reference_ids"])
            request["media"] = [
                m for m in request["media"] if m["id"] not in old_ids
            ] + media
            request["conditions"] = [
                dict(content_id=m["id"], role="reference", latent_modalities=["visual"])
                for m in media
            ]
        key = fingerprint(identity)
        started = time.perf_counter()
        cache = self.cache
        # A disabled cache is a genuine miss, not a reason to join a GPU queue.
        entry = None
        detail = {}
        if cache is not None:
            if getattr(cache, "backend", None) == "mooncake":
                entry, detail = cache.query_published(key)
            elif hasattr(cache, "entries"):
                # Entries are published only after their CPU snapshot is complete.
                # Retain the condition object, not the evictable KV allocation.
                entry = cache.entries.get(key)
        return dict(
            cache_key=key,
            hit=entry is not None,
            cache_query_context=dict(
                identity=identity,
                reference_ids=[
                    item["content_id"] for item in request.get("conditions", [])
                ],
            ),
            lookup={
                "cache_result": "l1_hit" if entry is not None else "miss",
                "lookup_host_ms": (time.perf_counter() - started) * 1000,
                "query_path": "independent",
                **detail,
            },
            **(
                dict(
                    inputs=prepared_message(entry.inputs),
                    admission_request=entry.request,
                    conditioning_identity=entry.conditioning_identity,
                )
                if entry is not None
                else {}
            ),
        )

    def cached_condition(self, message, *, initial, snapshot=None):
        entry = self.cache.lookup(message["cache_key"])
        if entry is None and snapshot is None:
            raise ValueError("The selected cache entry is no longer available")
        admission = (
            entry
            if entry is not None
            else SimpleNamespace(
                inputs=prepared_inputs(snapshot["inputs"]),
                request=snapshot["admission_request"],
            )
        )
        self.stage.admit_cached_preparation(admission, prompt_update=not initial)
        device = next(self.stage.modules["transformer"].parameters()).device
        inputs = inputs_on(
            admission.inputs,
            device,
        )
        message = dict(message, cache_hit=entry is not None)
        return dict(
            message,
            tensors=inputs.positive,
            inputs={
                field.name: getattr(inputs, field.name)
                for field in fields(inputs)
                if field.name not in ("positive", "source_visual")
            },
        )

    def receive_candidate(self, message):
        """Device adoption runs only on the model owner, after CPU transport."""
        import torch

        device = next(self.stage.modules["transformer"].parameters()).device

        def adopt(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, dict):
                return {k: adopt(v) for k, v in value.items()}
            if isinstance(value, list):
                return [adopt(v) for v in value]
            if isinstance(value, tuple):
                return tuple(adopt(v) for v in value)
            return value

        return adopt(message)

    def control(self, operation, arguments, *, busy):
        from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

        from .cache_logging import cache_scope

        arguments = dict(arguments)
        context = arguments.pop("_trace", None)
        with cache_scope(context):
            try:
                return self._control(operation, arguments, busy=busy)
            except ConditionError as exc:
                # Admission is a request result, not a failed worker/transport.
                return {"code": exc.code}

    def _control(self, operation, arguments, *, busy):
        from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
            ZingCacheReq,
        )

        cache = self.cache
        if cache is None:
            return {"code": "cache_unavailable"}
        if operation == "identity":
            return {"identity": getattr(cache, "key_identity", cache.identity)}
        if operation in ("preload", "unload", "status"):
            return cache.control(ZingCacheReq(operation, **arguments), busy=busy)
        if operation == "lookup":
            key = arguments["cache_key"]
            if arguments.get("world_id") is not None:
                cache.bind_world(
                    key,
                    arguments["world_id"],
                    s0_sha256=arguments.get("s0_sha256"),
                    observer_token=arguments.get("cache_recording"),
                    context=arguments.get("_trace"),
                )
            if (
                not isinstance(key, str)
                or len(key) != 64
                or any(c not in "0123456789abcdef" for c in key)
            ):
                return {"code": "invalid_cache_selection"}
            entry = (
                None
                if arguments.get("force", False)
                else cache.lookup(key, allow_disk=arguments.get("prefill", False))
            )
            if entry is not None:
                self.stage.admit_cached_preparation(
                    entry, prompt_update=arguments["prompt_update"]
                )
            from .preparation_cache import coordinate_validation

            lookup = (
                {
                    "cache_result": "miss",
                    "miss_reason": "forced",
                    "lookup_host_ms": None,
                }
                if arguments.get("force", False)
                else dict(cache.last_lookup)
            )
            return {
                "hit": entry is not None,
                "lookup": lookup,
                "rank_lookups": coordinate_validation(lookup),
                **(
                    {
                        "inputs": prepared_message(entry.inputs),
                        "conditioning_identity": entry.conditioning_identity,
                    }
                    if entry is not None
                    else {}
                ),
            }
        if operation == "prefill":
            from sglang.multimodal_gen.runtime.realtime.prefill_persistence import (
                persistence_descriptor,
            )

            if busy:
                return {"code": "cache_busy"}
            if getattr(cache, "tiers", None) is not None and not cache.tiers.l1:
                return {"code": "cache_unavailable"}
            if arguments.get("world_id") is not None:
                cache.bind_world(
                    arguments["cache_key"],
                    arguments["world_id"],
                    observer_token=arguments.get("cache_recording"),
                    context=arguments.get("_trace"),
                )
            from .block0_telemetry import Block0Telemetry
            from .preparation_cache import coordinate_validation

            telemetry = Block0Telemetry()
            began = time.perf_counter()
            if arguments.get("force", False):
                existing = None
            elif arguments.get("resident_only") and cache.backend == "mooncake":
                existing = cache.lookup(arguments["cache_key"], resident_only=True)
            else:
                existing = cache.lookup(arguments["cache_key"], allow_disk=True)
            lookup = (
                {
                    "cache_result": "miss",
                    "miss_reason": "forced",
                    "lookup_host_ms": None,
                }
                if arguments.get("force", False)
                else dict(cache.last_lookup)
            )
            telemetry.cache_result = lookup["cache_result"]
            telemetry.lookup_details = lookup
            telemetry.miss_reason = lookup.get("miss_reason")
            if arguments.get("force", False):
                telemetry.skip("cache_lookup")
            else:
                telemetry.record("cache_lookup", (time.perf_counter() - began) * 1000)

            def timings():
                telemetry.skip(
                    "restore"
                )  # Prefill publishes Host KV, not a live generation.
                telemetry.finish()
                snapshot = telemetry.snapshot()
                return dict(
                    timings=snapshot,
                    rank_timings=[
                        dict(rank=rank, **value)
                        for rank, value in enumerate(coordinate_validation(snapshot))
                    ],
                )

            if existing is not None:
                self.stage.admit_cached_preparation(existing)
                cache.register(
                    existing.key,
                    game_id=arguments.get("game_id"),
                    group_id=arguments.get("group_id"),
                )
                telemetry.skip("block0_prepare_and_l1_fill")
                return {
                    "cache_key": existing.key,
                    "_disk_descriptor": persistence_descriptor(
                        cache, existing, requested=arguments.get("persist", False)
                    ),
                    "hit": True,
                    "entries": cache.entry_count,
                    "evictions": cache.evictions,
                    **timings(),
                }
            if arguments.get("inputs") is None:
                return {"code": "cache_unavailable"}
            inputs = prepared_inputs(arguments["inputs"])
            from .prompt_update import coordinate_prompt_admission

            coordinate_prompt_admission(
                lambda: self.stage.codecs.validate_prepared_compute_inputs(inputs)
            )
            sampler = self.sampler_factory()
            admitted = self.stage.admit_realtime_condition(
                sampler,
                inputs,
                inputs.positive,
                source_shape=tuple(arguments["source_shape"]),
                total_frames=None,
                prompt_update=False,
            )
            if admitted is not None and not cache.can_store(admitted.prefix_rows):
                return {"code": "cache_capacity_exceeded"}
            device = next(self.stage.modules["transformer"].parameters()).device
            with telemetry.measure("block0_prepare_and_l1_fill"):
                entry = fill_prepared_prefix(
                    self.stage,
                    sampler,
                    inputs_on(inputs, device),
                    owner=SimpleNamespace(),
                    key=arguments["cache_key"],
                    identity=arguments.get("conditioning_identity"),
                    request=arguments["request"],
                    source_shape=tuple(arguments["source_shape"]),
                    force=arguments.get("force", False),
                )
            if not all(coordinate_validation(entry is not None)):
                return {
                    "code": "cache_capacity_busy"
                    if cache.backend == "local"
                    else "cache_unavailable"
                }
            cache.register(
                entry.key,
                game_id=arguments.get("game_id"),
                group_id=arguments.get("group_id"),
            )
            return {
                "cache_key": entry.key,
                "_disk_descriptor": persistence_descriptor(
                    cache, entry, requested=arguments.get("persist", False)
                ),
                "hit": False,
                "entries": cache.entry_count,
                "evictions": cache.evictions,
                **timings(),
            }
        if operation not in ("list_entries", "delete_entries"):
            return {"code": "invalid_cache_selection"}
        if getattr(cache, "backend", "local") == "mooncake":
            return {"code": "cache_operation_unsupported"}
        if operation == "delete_entries":
            from .preparation_cache import coordinate_validation

            if any(coordinate_validation(busy)):
                return {"code": "cache_busy"}
            target_keys = arguments.get("keys")
            if target_keys is not None and (
                not isinstance(target_keys, (list, tuple))
                or any(
                    not isinstance(k, str)
                    or len(k) != 64
                    or any(c not in "0123456789abcdef" for c in k)
                    for k in target_keys
                )
            ):
                return {"code": "invalid_cache_selection"}
            resident_keys = list(getattr(cache, "entries", {}).keys())
            all_resident = list(
                dict.fromkeys(
                    k
                    for rank_keys in coordinate_validation(resident_keys)
                    for k in rank_keys
                )
            )
            evict_keys = (
                [k for k in all_resident if k in target_keys]
                if target_keys is not None
                else all_resident
            )
            if hasattr(cache, "manage"):
                unload = cache.manage("unload", evict_keys)
                if isinstance(unload, dict) and unload.get("code"):
                    return unload
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_world_group,
        )
        from sglang.multimodal_gen.runtime.realtime.worker_parallel import (
            ParallelChannel,
        )

        return ParallelChannel(get_world_group()).leader_call(
            lambda: getattr(cache.disk_cache, operation)(**arguments)
        )
