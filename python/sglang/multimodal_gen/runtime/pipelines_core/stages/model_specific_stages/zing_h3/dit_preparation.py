# SPDX-License-Identifier: Apache-2.0
"""Adapt the existing Block0 owner to Encoder/DiT control messages."""

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

    def cached_condition(self, message, *, initial):
        entry = self.cache.lookup(message["cache_key"])
        if entry is None:
            raise ValueError("The selected cache entry is no longer available")
        self.stage.admit_cached_preparation(entry, prompt_update=not initial)
        device = next(self.stage.modules["transformer"].parameters()).device
        inputs = inputs_on(entry.inputs, device)
        return dict(
            message,
            tensors=inputs.positive,
            inputs={
                field.name: getattr(inputs, field.name)
                for field in fields(inputs)
                if field.name not in ("positive", "source_visual")
            },
        )

    def control(self, operation, arguments, *, busy):
        from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

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
            return {"identity": cache.identity}
        if operation in ("preload", "unload", "status"):
            return cache.control(ZingCacheReq(operation, **arguments), busy=busy)
        if operation == "lookup":
            key = arguments["cache_key"]
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
            return {
                "hit": entry is not None,
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
            if busy:
                return {"code": "cache_busy"}
            existing = (
                None
                if arguments.get("force", False)
                else cache.lookup(arguments["cache_key"], allow_disk=True)
            )
            if existing is not None:
                self.stage.admit_cached_preparation(existing)
                cache.register(
                    existing.key,
                    game_id=arguments.get("game_id"),
                    group_id=arguments.get("group_id"),
                )
                return {
                    "cache_key": existing.key,
                    "hit": True,
                    "entries": len(cache.entries),
                    "evictions": cache.evictions,
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
            cache.register(
                entry.key,
                game_id=arguments.get("game_id"),
                group_id=arguments.get("group_id"),
            )
            return {
                "cache_key": entry.key,
                "hit": False,
                "entries": len(cache.entries),
                "evictions": cache.evictions,
            }
        if operation not in ("list_entries", "delete_entries"):
            return {"code": "invalid_cache_selection"}
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_world_group,
        )
        from sglang.multimodal_gen.runtime.realtime.worker_parallel import (
            ParallelChannel,
        )

        return ParallelChannel(get_world_group()).leader_call(
            lambda: getattr(cache.disk_cache, operation)(**arguments)
        )
