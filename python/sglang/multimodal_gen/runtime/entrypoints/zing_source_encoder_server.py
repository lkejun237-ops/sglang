# SPDX-License-Identifier: Apache-2.0
"""Serve source VAE moments and Qwen conditions on the realtime WebSocket API.

Single GPU, one active session per service. Reuses native loaders and conditioning; no
DiT or decoder model is loaded by this endpoint.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, nullcontext
from dataclasses import fields
from pathlib import Path


def tensor_payload(tensor):
    import torch

    tensor = tensor.detach().cpu().contiguous()
    return dict(
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=list(tensor.shape),
        payload=tensor.view(torch.uint8).numpy().tobytes(),
    )


class EncoderConditioning:
    """Keep the original prepared media across source dispatches/prompt events."""

    def __init__(
        self,
        codecs,
        *,
        cache_bytes=256 << 20,
        encode_tensor=tensor_payload,
        prompt_switch_prefix_latent_frames=5,
    ):
        from sglang.multimodal_gen.runtime.conditioning.worker import LocalConditioner

        self.codecs = codecs
        self.encode_tensor = encode_tensor
        self.prompt_switch_prefix_latent_frames = prompt_switch_prefix_latent_frames
        self.conditioner = LocalConditioner(
            codecs.encode_condition,
            hashlib.sha256(str(id(codecs)).encode()).hexdigest(),
            cache_bytes=cache_bytes,
        )
        self.dispose()

    def dispose(self):
        self.inputs = self.prompt = self.event_id = self.failed_update = None
        self.cache_key = None
        self.cache_lookup = {}
        self.cache_hit = False
        self.cache_write = False
        self.cache_request = self.cache_preview = None
        self.cache_candidates = {}
        self.pending_inputs = None
        self.candidates = {}
        self.conditioner.identity_state.dispose()

    def confirm_applied(self, event_id):
        # Match SourceStreamingSession: a prepared reference candidate becomes
        # active only when the sampler publishes its prompt epoch.
        if self.pending_inputs is not None and self.pending_inputs[0] == event_id:
            _, self.inputs = self.pending_inputs
            self.pending_inputs = None
        if event_id in self.candidates:
            self.inputs = self.candidates[event_id]
            self.candidates = {
                key: value for key, value in self.candidates.items() if key > event_id
            }
        if event_id in self.cache_candidates:
            self.cache_request = self.cache_candidates[event_id]
            self.cache_candidates = {
                key: value
                for key, value in self.cache_candidates.items()
                if key > event_id
            }

    def reject(self, event_id):
        self.candidates.pop(event_id, None)
        self.cache_candidates.pop(event_id, None)
        if self.pending_inputs is not None and self.pending_inputs[0] == event_id:
            self.pending_inputs = None
        if event_id == self.event_id:
            self.failed_update = (self.prompt, event_id)

    def _lookup(self, batch, preview=None, update=None):
        query = getattr(self, "cache_query", None)
        if query is None:
            return None
        identity = query("identity")
        if identity is None or "identity" not in identity:
            self.cache_key, self.cache_hit = None, False
            return None
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_preparation import (
            prepared_inputs,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
            inputs_on,
            preparation_key,
            preparation_request,
            updated_preparation_request,
        )

        command = batch.extra["zing_source_command"]
        self.cache_write = bool(command.get("cache_write", False))
        if self.cache_request is None:
            self.cache_request = preparation_request(
                self.codecs, batch.sampling_params.zing_request
            )
            self.cache_preview = preview
        request = self.cache_request
        if update is not None:
            request = updated_preparation_request(
                self.codecs,
                request,
                self.inputs,
                update.get("prompt", batch.sampling_params.prompt),
                update,
            )
            self.cache_candidates[update["event_id"]] = request
        self.cache_key = preparation_key(
            self.codecs,
            request,
            self.cache_preview,
            command["config"],
            identity["identity"],
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.cache_logging import (
            batch_context,
        )

        response = query(
            "lookup",
            cache_key=self.cache_key,
            prefill=bool(batch.extra.get("zing_prefill")),
            force=batch.extra.get("zing_prefill", {}).get("force", False),
            prompt_update=update is not None,
            _trace=batch_context(batch),
        )
        if response.get("code") == "CONDITION_INVALID":
            from sglang.multimodal_gen.runtime.conditioning.contract import (
                ConditionError,
            )

            raise ConditionError(
                "CONDITION_INVALID", "This request is too large for this server"
            )
        self.cache_lookup = {
            key: response[key] for key in ("lookup", "rank_lookups") if key in response
        }
        self.cache_hit = bool(response.get("hit"))
        if not self.cache_hit:
            return None
        # Retain the original CPU condition media for later prompt-only changes.
        inputs = prepared_inputs(response["inputs"])
        if getattr(self.codecs, "compute_plan", None) is not None:
            self.codecs.validate_prepared_compute_inputs(inputs)
        inputs.condition_media = [
            {k: v.cpu() if hasattr(v, "cpu") else v for k, v in item.items()}
            for item in (inputs.condition_media or [])
        ]
        device = next(iter(self.codecs.modules["video_vae"].parameters())).device
        self.conditioner.identity_state.identity = response.get("conditioning_identity")
        self.conditioner.identity_state.positive = inputs.positive
        return inputs_on(inputs, device)

    def _condition(self, inputs, event_id, version, *, include_inputs=False):
        """One event carries all its prepared changes in one transport message.

        A cache hit sends only its key and event identity. On a miss,
        ``tensors`` contains the complete Qwen condition. ``inputs`` is
        present for initialization or reference replacement, including an empty
        visual map for clearing references; omission preserves active references.
        Even a reference-only edit recomputes Qwen against the new images.
        """
        if self.cache_key is not None and self.cache_hit:
            return dict(
                event_id=event_id,
                prompt_version=version,
                cache_key=self.cache_key,
                cache_hit=True,
                cache_lookup=self.cache_lookup.get("lookup", {}),
            )
        output = dict(
            event_id=event_id,
            prompt_version=version,
            tensors={
                name: self.encode_tensor(value)
                for name, value in inputs.positive.items()
            },
        )
        if self.cache_key is not None:
            output.update(
                cache_key=self.cache_key,
                cache_hit=False,
                conditioning_identity=self.conditioner.identity_state.identity,
                cache_lookup=self.cache_lookup.get("lookup", {}),
            )
        if include_inputs:
            output["inputs"] = {
                field.name: getattr(inputs, field.name)
                for field in fields(inputs)
                if field.name
                not in (
                    ("positive", "source_visual")
                    if self.cache_key is not None and self.cache_write
                    else ("positive", "source_visual", "condition_media")
                )
            }
        return output

    def forward(self, batch, on_condition=None, *, async_updates=False):
        from copy import copy

        from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.prompt_update import (
            prepare_prompt_update,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
            source_preview,
        )

        command = batch.extra["zing_source_command"]
        update_command = batch.extra.get("zing_prompt_update", {})
        update = (
            update_command.get("prompt", batch.sampling_params.prompt),
            update_command.get("event_id", batch.realtime_event_id),
        )
        version = getattr(batch, "realtime_prompt_version", None)
        output = notice = None
        if self.inputs is None:
            startup = command.get("startup")
            preview_config = dict(command["config"])
            if startup is not None:
                preview_config.update(
                    {key: startup[key] for key in ("width", "height") if key in startup}
                )
            preview = source_preview(
                preview_config,
                startup["rgb"]
                if startup is not None
                else command["appends"][0]["payload"]["data"],
            )
            self.inputs = self._lookup(batch, preview)
            if self.inputs is None:
                self.inputs = self.codecs.prepare(
                    batch.sampling_params.zing_request,
                    lambda _: nullcontext(),
                    retain_condition_media=True,
                    source_preview=preview,
                    conditioning=self.conditioner,
                )
            self.prompt = batch.sampling_params.prompt
            self.event_id = batch.realtime_event_id
            output = self._condition(
                self.inputs,
                self.event_id,
                version,
                include_inputs=on_condition is not None,
            )
            if on_condition is not None:
                # An early reference event must not replace the initial epoch.
                on_condition("conditioning", **output)
                output = None
        if update != (self.prompt, self.event_id) and update != self.failed_update:
            # A newer command supersedes a candidate not yet applied by DiT.
            self.pending_inputs = None
            sent = False
            previous = self.prompt, self.event_id
            if update_command.get("rejected"):
                self.failed_update = update
            else:
                try:
                    candidate = self._lookup(batch, update=update_command)
                    if candidate is not None:
                        positive = candidate.positive
                    else:
                        positive, candidate = prepare_prompt_update(
                            self.codecs,
                            self.inputs,
                            update[0],
                            update_command,
                            use_component=lambda _: nullcontext(),
                            conditioning=self.conditioner,
                            prefix_latent_frames=self.prompt_switch_prefix_latent_frames,
                        )
                    if candidate is None:
                        candidate = copy(self.inputs)
                        candidate.positive = positive
                    if on_condition is not None:
                        # Publish the candidate before dispatch: the reply may
                        # already contain the first output using this event.
                        self.pending_inputs = (update[1], candidate)
                        if async_updates:
                            if len(self.candidates) >= 32:
                                raise ConditionError(
                                    "CONDITION_BUSY",
                                    "Too many pending condition updates",
                                )
                            self.candidates[update[1]] = candidate
                    output = self._condition(
                        candidate,
                        update[1],
                        version,
                        include_inputs=on_condition is not None
                        and (async_updates or "references" in update_command),
                    )
                    if on_condition is not None:
                        sent = True
                        on_condition("conditioning", **output)
                        output = None
                except ConditionError as exc:
                    logging.warning("Encoder condition update rejected", exc_info=True)
                    self.prompt, self.event_id = previous
                    self.pending_inputs = None
                    self.candidates.pop(update[1], None)
                    self.cache_candidates.pop(update[1], None)
                    self.failed_update = update
                    output = None
                    notice = dict(code=exc.code, event_id=update[1])
                else:
                    self.prompt, self.event_id = update
                    self.failed_update = None
                    if on_condition is None:
                        self.inputs = candidate
                        self.cache_request = self.cache_candidates.pop(
                            update[1], self.cache_request
                        )
            if self.failed_update == update and on_condition is not None and not sent:
                # DiT already discarded pending state when it rejected a sent
                # candidate. Only local preparation failures need cancellation.
                on_condition("conditioning.cancel", event_id=update[1])
        return output, notice


class SourceEncoderExecutor:
    """Adapt existing pull-based codec sessions to the realtime execution seam.

    All CUDA work, including release, runs on one owned thread. The API shields an
    in-flight forward before cleanup; cancelling a socket cannot release its cache
    while kernels are still running. Each dispatch owns at most ten output slices.
    """

    def __init__(self, codecs, *, conditioning=None, dit_link=None):
        self.codecs = codecs
        self.conditioning = conditioning
        self.dit_link = dit_link
        if self.conditioning is not None:
            self.conditioning.cache_query = self._cache_query
        self._dit_final_frames = None
        self._dit_latents = 0
        self._dit_committed_frames = 0
        self.owner = self.encoder = None
        self.command_index = 0
        self.consumed = 0
        self.startup_preparation = None
        self.startup_attached = False

    def _prefill(self, batch):
        import torch

        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_preparation import (
            prepared_message,
        )

        if self.owner is not None:
            raise ValueError("The encoder is busy")
        if self.dit_link is None:
            raise ValueError("The worker connections are not ready")
        if "identity" not in self._cache_query("identity"):
            raise ValueError("Block0 caching is not enabled")
        options = batch.extra["zing_prefill"]
        command = batch.extra["zing_source_command"]
        command["appends"] = [{"payload": {"data": options["source_frame"]}}]
        try:
            with (
                torch.inference_mode(),
                set_forward_context(current_timestep=0, attn_metadata=None),
            ):
                began = time.perf_counter()
                self.conditioning.forward(batch)
                encoder_ms = (time.perf_counter() - began) * 1000
                if self.conditioning.cache_key is None:
                    raise ValueError("Block0 caching is not enabled")
                from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.cache_logging import (
                    batch_context,
                )

                labels = {name: options.get(name) for name in ("game_id", "group_id")}
                labels["_trace"] = batch_context(batch)
                if self.conditioning.cache_hit:
                    # Reuse prefill to register labels even on an existing hit.
                    receipt = self._cache_query(
                        "prefill",
                        cache_key=self.conditioning.cache_key,
                        inputs=None,
                        request=None,
                        source_shape=None,
                        force=False,
                        **labels,
                    )
                else:
                    geometry = command.get("source_geometry")
                    shape = (
                        [geometry["encoded_height"], geometry["encoded_width"]]
                        if geometry
                        else [command["config"][k] for k in ("height", "width")]
                    )
                    receipt = self._cache_query(
                        "prefill",
                        cache_key=self.conditioning.cache_key,
                        inputs=prepared_message(self.conditioning.inputs),
                        conditioning_identity=self.conditioning.conditioner.identity_state.identity,
                        request=self.conditioning.cache_request,
                        source_shape=[n // self.codecs.spatial_ratio for n in shape],
                        force=options.get("force", False),
                        **labels,
                    )
                if receipt.get("code"):
                    from sglang.multimodal_gen.runtime.conditioning.contract import (
                        ConditionError,
                    )

                    raise ConditionError(
                        "CONDITION_BUSY", "The prefix cache could not be prepared"
                    )
                timing = receipt.setdefault("timings", {})
                timing.setdefault("stages", {})["encoder_conditioning_and_lookup"] = {
                    "host_ms": encoder_ms,
                    "status": "executed",
                }
                lookup_receipt = getattr(self.conditioning, "cache_lookup", {})
                origin = lookup_receipt.get("lookup", {})
                # A first lookup may promote L2; a second bookkeeping lookup
                # must not turn that request into an advertised L1 hit.
                if receipt.get("hit") and origin.get("cache_result") == "l2_hit":
                    timing["cache_result"] = "l2_hit"
                    timing["lookup"] = origin
                    for rank, value in zip(
                        receipt.get("rank_timings", []),
                        lookup_receipt.get("rank_lookups", []),
                    ):
                        rank["cache_result"], rank["lookup"] = (
                            value["cache_result"],
                            value,
                        )
                if timing.get("cache_result") is not None:
                    receipt["cache_result"] = timing["cache_result"]
                return OutputBatch(
                    raw_frame_batches=[], usage={"zing_prefill": receipt}
                )
        finally:
            self.conditioning.dispose()

    def _cache_query(self, operation, **arguments):
        if self.dit_link is None:
            return None
        try:
            return self.dit_link.cache_control(operation, **arguments)
        except BaseException:
            controller = getattr(self, "connection_controller", None)
            if controller is not None:
                controller.failed()
            raise

    def forward(self, request):
        from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
        from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
            ReleaseRealtimeSessionReq,
            ZingCacheReq,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_encoder import (
            create_source_buffer,
        )

        if isinstance(request, ZingCacheReq):
            controller = getattr(self, "connection_controller", None)
            if self.dit_link is None or (
                controller is not None and controller.state != "connected"
            ):
                return OutputBatch(output={"code": "connection_not_ready"})
            return OutputBatch(
                output=self._cache_query(
                    request.operation,
                    game_id=request.game_id,
                    group_id=request.group_id,
                    cache_key=request.cache_key,
                )
            )
        if isinstance(request, ReleaseRealtimeSessionReq):
            if self.dit_link is not None:
                self.dit_link.release(request.session_id)
            if self.owner is not None and request.session_id == self.owner[0]:
                self.encoder.dispose()
                if self.conditioning is not None:
                    self.conditioning.dispose()
                self.owner = self.encoder = None
                self._dit_final_frames = None
                self._dit_latents = 0
                self._dit_committed_frames = 0
                self.startup_preparation = None
                self.startup_attached = False
            return OutputBatch(output=[])
        if not isinstance(request, list) or len(request) != 1:
            raise ValueError("Source encoding requires one request")
        batch = request[0]
        if "zing_prefill" in batch.extra:
            return self._prefill(batch)
        owner = (batch.realtime_session_id, batch.realtime_generation_id)
        command = batch.extra["zing_source_command"]
        if self.owner is None:
            if batch.block_idx != 0:
                raise ValueError("Source encoding must start at the first chunk")
            # Deployment profile chooses the algorithm. This process owns the
            # model, so no remote peer/bridge is needed for its source buffer.
            profile = dict(command["encoder"])
            profile.pop("remote", None)
            self.encoder = create_source_buffer(
                self.codecs,
                profile,
                capacity_frames=3 * self.codecs.vae_frames_per_chunk,
            )
            # Group destruction can close an already-completed DiT link after
            # its owner is gone, so no per-owner Release request reaches here.
            # A new codec owner must never inherit the previous player's source
            # watermark, latent cursor, EOS target or startup preparation.
            self._dit_final_frames = None
            self._dit_latents = 0
            self._dit_committed_frames = 0
            self.startup_preparation = None
            self.startup_attached = False
            self.owner, self.command_index, self.consumed = owner, 0, 0
        if owner != self.owner or batch.block_idx != self.command_index:
            raise ValueError("Source session identity or chunk sequence does not match")
        config = command["config"]

        dit_outputs = []
        notices = []
        condition_ready = []
        condition_prepared = []
        cache_reports = []

        def consume_results(results, kind=None, payload=None):
            rejection = None
            for result in results:
                if self.conditioning is not None and result["type"] in (
                    "dit.latents",
                    "decoder.frames",
                    "decoder.output_queued",
                    "decoder.input_sent",
                ):
                    self.conditioning.confirm_applied(result.get("event_id"))
                    self._dit_committed_frames = max(
                        self._dit_committed_frames,
                        result["output_frame_start"] + result["num_frames"],
                    )
                if result["type"] == "dit.latents":
                    value = result.pop("tensor")
                    result.pop("transfer_id", None)
                    dit_outputs.append({**result, **tensor_payload(value)})
                elif result["type"] in (
                    "decoder.frames",
                    "decoder.output_queued",
                    "decoder.input_sent",
                ):
                    result.pop("transfer_id", None)
                    dit_outputs.append(result)
                elif result["type"] == "conditioning.rejected":
                    if result.get("initial"):
                        raise ConditionError(
                            result["code"], "This request is too large for this server"
                        )
                    if getattr(self.dit_link, "async_inputs", False):
                        self.conditioning.reject(result["event_id"])
                        notices.append(result)
                        continue
                    if (
                        kind != "conditioning"
                        or result["event_id"] != payload["event_id"]
                    ):
                        raise ValueError(
                            "DiT condition rejection does not match the event"
                        )
                    rejection = result
                elif result["type"] == "conditioning.ready":
                    command = batch.extra.get("zing_prompt_update", {})
                    if (
                        result.get("event_id") != command.get("event_id")
                        or result.get("prompt_version") != command.get("prompt_version")
                        or result.get("source_frame_index")
                        != command.get("source_boundary")
                        or condition_ready
                    ):
                        raise ValueError(
                            "DiT condition readiness does not match the request"
                        )
                    self.conditioning.confirm_applied(result["event_id"])
                    condition_ready.append(
                        dict(
                            type="control_ack",
                            kind="prompt.update",
                            state="kv_ready",
                            event_id=result["event_id"],
                            prompt_version=result["prompt_version"],
                            source_frame_index=result["source_frame_index"],
                        )
                    )
                elif result["type"] == "conditioning.prepared":
                    if (
                        result.get("preparation") != self.startup_preparation
                        or condition_prepared
                    ):
                        raise ValueError(
                            "DiT preparation does not match the source capture"
                        )
                    condition_prepared.append(result["preparation"])
                elif result["type"] == "model.cache":
                    cache_reports.append(result["usage"]["zing_block0_cache"])
                elif result["type"] != "session.completed":
                    raise ValueError("Unexpected DiT output")

            if rejection is not None:
                raise ConditionError(rejection["code"], "Condition update was rejected")

        def send_dit(kind, **payload):
            if kind == "conditioning":
                payload.update(
                    {
                        name: command.get(name)
                        for name in ("cache_write", "game_id", "group_id")
                    }
                )
                boundary = batch.extra.get("zing_prompt_update", {}).get(
                    "source_boundary"
                )
                if boundary is not None:
                    payload["source_boundary"] = boundary
                if command.get("startup", {}).get("op") == "prepare":
                    payload["source_prepare"] = self.startup_preparation
            consume_results(
                self.dit_link.send(
                    kind,
                    request_id=batch.request_id,
                    chunk_index=batch.block_idx,
                    **payload,
                ),
                kind,
                payload,
            )

        try:
            if self.dit_link is not None and getattr(
                self.dit_link, "async_inputs", False
            ):
                # Apply earlier activation/rejection notices before preparing a
                # new prompt. A full prepared epoch travels with every async
                # update so in-flight activation cannot mix Qwen and references.
                consume_results(self.dit_link.collect())
            if command.get("startup") is not None:
                result = self._forward_startup(
                    batch, command, send_dit, condition_prepared
                )
            else:
                result = self._forward_source(
                    batch, command, config, outputs_state=(dit_outputs, send_dit)
                )
            if notices:
                result.realtime_handoff["conditioning_notices"] = notices
                result.conditioning_notice = notices[-1]
            result.usage["zing_source_committed_end"] = self._dit_committed_frames
            result.usage["zing_realtime_blocks"] = [
                {
                    "event_id": item.get("event_id"),
                    "prompt_version": item.get("prompt_version"),
                    "num_frames": item["num_frames"],
                }
                for item in dit_outputs
            ]
            if condition_ready:
                result.usage["zing_condition_ready"] = condition_ready[0]
            if cache_reports:
                result.realtime_handoff["cache_reports"] = cache_reports
            return result
        except BaseException as exc:
            controller = getattr(self, "connection_controller", None)
            if controller is not None and not isinstance(exc, ConditionError):
                if self.dit_link is not None:
                    self.dit_link.failed = True
                controller.failed()
            from sglang.multimodal_gen.runtime.utils.realtime_video import (
                cancel_async_raw_rgb_frame_reference,
            )

            for item in dit_outputs:
                if item.get("reference") is not None:
                    cancel_async_raw_rgb_frame_reference(item["reference"])
            raise

    def _forward_startup(self, batch, command, send_dit, confirmations):
        import torch

        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )
        from sglang.multimodal_gen.runtime.realtime.source_prepare import (
            PROTOCOL,
            V2_PROTOCOL,
            preparation_identity,
            validate_preview_rgb,
        )

        startup = command["startup"]
        identity = preparation_identity(startup.get("identity"))
        protocol = startup.get("protocol", PROTOCOL)
        if (
            protocol not in {PROTOCOL, V2_PROTOCOL}
            or command["appends"]
            or command.get("end") is not None
            or self.command_index
            or self._dit_latents
            or self.consumed
            or self.startup_preparation is not None
            and (
                self.startup_preparation["identity"] != identity
                or self.startup_preparation["protocol"] != protocol
            )
        ):
            raise ValueError("Invalid source preparation transition")
        if startup["op"] == "prepare":
            width, height = (command["config"][k] for k in ("width", "height"))
            if protocol == V2_PROTOCOL:
                width, height = startup.get("width"), startup.get("height")
            validate_preview_rgb(startup["rgb"], width, height)
            expected = dict(
                identity=identity,
                protocol=protocol,
                preview_sha256=hashlib.sha256(startup["rgb"]).hexdigest(),
                preview_size=[width, height],
            )
            if self.startup_preparation is None:
                self.startup_preparation = expected
                with (
                    torch.inference_mode(),
                    set_forward_context(current_timestep=0, attn_metadata=None),
                ):
                    self.conditioning.forward(batch, on_condition=send_dit)
                if confirmations != [expected]:
                    raise ValueError("DiT did not confirm source preparation")
            elif expected != self.startup_preparation:
                raise ValueError("Source preparation pixels changed")
            receipt = dict(type="source.prepared", identity=identity, level="condition")
        elif startup["op"] == "attach" and self.startup_preparation is not None:
            self.startup_attached = True
            receipt = dict(type="source.attached", identity=identity)
        else:
            raise ValueError("Invalid source preparation transition")
        return OutputBatch(
            raw_frame_batches=[],
            realtime_handoff={"dit_latents": []},
            usage=dict(
                zing_startup_ready=receipt,
                zing_source_consumed_end=0,
                zing_source_committed_end=0,
                zing_source_complete=False,
                zing_source_more_work=False,
            ),
        )

    def _forward_source(self, batch, command, config, *, outputs_state):
        import torch

        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        dit_outputs, send_dit = outputs_state
        if self.startup_preparation is not None:
            from sglang.multimodal_gen.runtime.realtime.source_prepare import PROTOCOL

            if not self.startup_attached:
                raise ValueError("Source preparation has not been attached")
            if (
                self.command_index == 0
                and self.startup_preparation["protocol"] == PROTOCOL
            ):
                data = command["appends"][0]["payload"]["data"][
                    : config["width"] * config["height"] * 3
                ]
                if (
                    hashlib.sha256(data).hexdigest()
                    != self.startup_preparation["preview_sha256"]
                ):
                    raise ValueError(
                        "The first source frame differs from its preparation"
                    )
        geometry = command.get("source_geometry")
        condition = notice = None
        with (
            torch.inference_mode(),
            set_forward_context(current_timestep=0, attn_metadata=None),
        ):
            if self.conditioning is not None:
                condition, notice = self.conditioning.forward(
                    batch,
                    on_condition=send_dit if self.dit_link is not None else None,
                    **(
                        {"async_updates": True}
                        if getattr(self.dit_link, "async_inputs", False)
                        else {}
                    ),
                )
            sink_command = batch.extra.get("zing_sink_update")
            if sink_command is not None:
                if self.dit_link is None:
                    raise ValueError("Sink updates require a connected DiT worker")
                send_dit("sink.update", **sink_command)
            for item in command["appends"]:
                payload = item["payload"]
                pixels = torch.frombuffer(bytearray(payload["data"]), dtype=torch.uint8)
                pixels = pixels.reshape(
                    payload["num_frames"], config["height"], config["width"], 3
                ).permute(3, 0, 1, 2)
                if geometry is not None:
                    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.media_geometry import (
                        SourceGeometry,
                    )

                    pixels = SourceGeometry(**geometry).resize(pixels)
                self.encoder.append(pixels, frame_start=payload["source_frame_start"])
            if command["end"] is not None:
                self._dit_final_frames = command["end"]["frame_end"]
                self.encoder.finish()
            outputs = []
            while self.encoder.ready and len(outputs) < 10:
                value = self.encoder.pop()
                if self.dit_link is not None:
                    send_dit(
                        "source.moments",
                        tensor=value.values,
                        latent_start=value.start,
                        latent_end=value.end,
                        source_dependency_end=value.source_dependency_end,
                        source_records=batch.extra["zing_dit_source_records"],
                        final_source_frames=self._dit_final_frames,
                    )
                    self._dit_latents = value.end
                    # Retain the existing ten-slice dispatch bound.
                    outputs.append(None)
                else:
                    outputs.append(
                        dict(
                            format="vae_posterior_moments",
                            **tensor_payload(value.values),
                            latent_start=value.start,
                            latent_end=value.end,
                            source_dependency_end=value.source_dependency_end,
                        )
                    )
                self.consumed = value.source_dependency_end
                if any(item["type"] == "decoder.frames" for item in dit_outputs):
                    # Return references before reserving another batch of RGB.
                    # The API can consume them while the next dispatch runs.
                    break
            if (
                self.dit_link is not None
                and self.encoder.ended
                and not self.encoder.ready
            ):
                send_dit("source.end", latent_frames=self._dit_latents)
        self.command_index += 1
        return OutputBatch(
            # An empty RGB list means no media postprocessing in the shared
            # scheduler-client wrapper. Actual output is an explicit handoff.
            raw_frame_batches=[],
            realtime_handoff=(
                {"dit_latents": dit_outputs}
                if self.dit_link is not None
                else {"source_moments": outputs, "conditioning": condition}
            ),
            conditioning_notice=notice,
            usage=dict(
                zing_source_consumed_end=self.consumed,
                zing_source_more_work=self.encoder.ready,
                zing_source_complete=self.encoder.ended and not self.encoder.ready,
            ),
        )


class EncoderClient:
    def __init__(self, executor, pool):
        self.executor, self.pool = executor, pool

    async def forward(self, request):
        return await asyncio.get_running_loop().run_in_executor(
            self.pool, self._forward, request
        )

    async def cache_disk(self, operation, **arguments):
        from fastapi import HTTPException

        def call():
            controller = getattr(self.executor, "connection_controller", None)
            if self.executor.dit_link is None or (
                controller is not None and controller.state != "connected"
            ):
                raise HTTPException(409, detail={"code": "connection_not_ready"})
            return self.executor._cache_query(operation, **arguments)

        return await asyncio.get_running_loop().run_in_executor(self.pool, call)

    async def open_dit(self, message):
        return await asyncio.get_running_loop().run_in_executor(
            self.pool, self._open_dit, message
        )

    def _open_dit(self, message):
        from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

        controller = getattr(self.executor, "connection_controller", None)
        if controller is not None:
            # The queued init may have waited behind other model work. Check at
            # execution, serialized with quiesce's state transition, not enqueue.
            with controller.lock:
                if controller.state != "connected":
                    raise ConditionError(
                        "CONDITION_BUSY",
                        "The connection is closing. Please wait before starting again.",
                    )
        try:
            return self.executor.dit_link.open(message)
        except BaseException:
            if controller is not None:
                controller.failed()
            raise

    def _forward(self, request):
        import torch

        from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

        try:
            result = self.executor.forward(request)
            torch.cuda.synchronize()
        except BaseException as exc:
            from sglang.multimodal_gen.runtime.realtime.cleanup_watchdog import (
                CleanupWatchdog,
            )

            # Publish failure before waiting on a possibly broken CUDA stream.
            with CleanupWatchdog(120):
                controller = getattr(self.executor, "connection_controller", None)
                if controller is not None and not isinstance(exc, ConditionError):
                    controller.failed()
                try:
                    torch.cuda.synchronize()
                except BaseException:
                    if controller is not None:
                        controller.failed()
                    raise
            raise
        return result


def create_apps(
    model_path,
    *,
    dit_port=None,
    service_count=1,
    connection_ports=None,
    pipeline_config=None,
    session_max_lifetime_s=None,
):
    from sglang.multimodal_gen.runtime.realtime.worker_services import service_ports

    peers = service_ports(service_count, dit=dit_port).get(
        "dit", [None] * service_count
    )
    # Use the same bootstrap, model loader and warmup as encoder_worker.
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker import (
        initialize_runtime_imports,
        load_encoder,
    )

    ServerArgs, set_global_server_args = initialize_runtime_imports()
    from fastapi import FastAPI

    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_source_encoder_adapter import (
        ZingSourceEncoderAdapter,
    )
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_video_api import (
        router,
    )

    args = ServerArgs.from_kwargs(
        model_path=str(model_path),
        pipeline_config=pipeline_config,
        pipeline_class_name="ZingH3CausalPipeline",
        num_gpus=1,
        video_decoder_backend="native",
        attention_backend="torch_sdpa",
        warmup_mode="off",
        realtime_max_sessions=1,
        realtime_max_sessions_per_worker=1,
        realtime_session_max_lifetime_s=(
            ServerArgs.realtime_session_max_lifetime_s
            if session_max_lifetime_s is None
            else session_max_lifetime_s
        ),
        zing_source_work_quantum=1,
        zing_source_joint_forward=None,
        text_encoder_cpu_offload=False,
        zing_conditioning_mode="local",
        zing_conditioning_prefetch=False,
    )
    set_global_server_args(args)

    @asynccontextmanager
    async def lifespan(app):
        with (
            tempfile.TemporaryDirectory(prefix="zing-encoder-tp1-") as rendezvous,
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="encoder") as pool,
        ):
            executors = []

            def load():
                from sglang.multimodal_gen.runtime.conditioning.native import (
                    build_encoder_pipeline,
                )
                from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                    maybe_init_distributed_environment_and_model_parallel,
                )
                from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
                    ZingH3Codecs,
                )

                # Qwen uses the existing TP1 loader/attention implementation.
                maybe_init_distributed_environment_and_model_parallel(
                    tp_size=1,
                    sp_size=1,
                    distributed_init_method="file://" + rendezvous + "/group",
                )
                qwen = build_encoder_pipeline(str(model_path), args)
                source = load_encoder(Path(model_path))
                # Preserve checkpoint metadata populated by the original loader,
                # including the statistics used to normalize reference latents.
                args.pipeline_config.vae_config.arch_config = source.va
                codecs = ZingH3Codecs(
                    {**qwen.modules, **source.modules},
                    args.pipeline_config,
                    source.runtime,
                )
                from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
                    ZingComputeLimits,
                )

                # This service must reject unsafe encoding inputs before it sends
                # tensors to a DiT, independently of that peer's token capacity.
                codecs.compute_plan = ZingComputeLimits(
                    (codecs.canvas_multiple,) * 2, codecs.max_short_edge
                )
                codecs.warmup_encoder()
                if connection_ports is not None:
                    connection_ports.initialize_stores()
                for peer in peers:
                    link = None
                    if peer is not None:
                        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_io import (
                            DiTIOChannel,
                        )
                        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_link import (
                            DiTLinkClient,
                        )
                        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_transport import (
                            EncoderDiTChannel,
                        )

                        link = DiTLinkClient(
                            DiTIOChannel(
                                lambda peer=peer: EncoderDiTChannel.connect(
                                    peer, rank=0, device="cuda:0"
                                ),
                                cleanup_timeout=120,
                            ).start()
                        )
                    executors.append(
                        SourceEncoderExecutor(
                            codecs,
                            conditioning=EncoderConditioning(
                                codecs,
                                prompt_switch_prefix_latent_frames=args.pipeline_config.prompt_switch_prefix_latent_frames,
                                cache_bytes=args.zing_conditioning_local_cache_bytes,
                                encode_tensor=(lambda value: value.detach())
                                if link is not None
                                else tensor_payload,
                            ),
                            dit_link=link,
                        )
                    )

            def shutdown():
                from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                    cleanup_dist_env_and_memory,
                )

                # Qwen owns TP1 groups even without a DiT peer. Destroy them
                # explicitly before interpreter teardown, after forwards drain.
                failure = None
                try:
                    for executor in executors:
                        if executor.dit_link is not None:
                            try:
                                if connection_ports is not None:
                                    executor.dit_link.disconnect()
                                else:
                                    executor.dit_link.shutdown()
                            except Exception as exc:
                                logging.exception(
                                    "Encoder service link shutdown failed"
                                )
                                failure = failure or exc
                finally:
                    cleanup_dist_env_and_memory()
                if failure is not None:
                    raise failure

            try:
                await asyncio.get_running_loop().run_in_executor(pool, load)
                for service_app, executor in zip(apps, executors):
                    service_app.state.realtime_execution_client = EncoderClient(
                        executor, pool
                    )
                if connection_ports is not None:
                    from sglang.multimodal_gen.runtime.realtime.worker_group_encoder import (
                        install_encoder_group,
                    )

                    for index, (service_app, executor) in enumerate(
                        zip(apps, executors)
                    ):
                        install_encoder_group(
                            service_app,
                            executor,
                            pool,
                            connection_ports.services[index],
                            connection_ports.stores[index],
                        )
                    controllers = [
                        service_app.state.worker_connections for service_app in apps
                    ]
                    for service_app in apps:
                        service_app.state.worker_connections.ready = (
                            lambda service_app=service_app: (
                                getattr(service_app.state, "connection_ready", False)
                                and not any(
                                    controller.restart_required
                                    for controller in controllers
                                )
                            )
                        )
                yield
            finally:
                await asyncio.get_running_loop().run_in_executor(pool, shutdown)

    from sglang.multimodal_gen.runtime.realtime.admission import (
        InMemorySessionLeaseStore,
        RealtimeAdmissionController,
    )

    apps = []
    for index, peer in enumerate(peers):
        app = FastAPI(lifespan=lifespan if index == 0 else None)
        # Admission and mutable request state belong to a service, not the
        # process-global controller used by the original single endpoint.
        app.state.realtime_admission_controller = RealtimeAdmissionController(
            InMemorySessionLeaseStore(1, args.realtime_session_lease_ttl_s),
            wait_timeout_s=0,
        )
        if peer is None and connection_ports is None:
            app.state.realtime_adapter_factory = lambda _: ZingSourceEncoderAdapter()
        else:
            from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_encoder_dit_adapter import (
                ZingEncoderDiTAdapter,
            )

            def adapter_factory(_, app=app):
                adapter = ZingEncoderDiTAdapter()
                if connection_ports is not None:
                    adapter.connection_app = app
                return adapter

            app.state.realtime_adapter_factory = adapter_factory
        app.include_router(router)
        from sglang.multimodal_gen.runtime.entrypoints.openai.zing_cache_api import (
            router as cache_router,
        )

        app.include_router(cache_router)
        from sglang.multimodal_gen.runtime.realtime.role_pool.native_http import (
            install_managed_transport,
        )

        # The input WS and management HTTP are different ASGI applications.
        # Bind authorization to the business app before its middleware stack is
        # built; the native controller is attached later by the model lifespan.
        install_managed_transport(app)
        apps.append(app)
    return apps, args


def create_app(model_path, *, dit_port=None):
    apps, args = create_apps(model_path, dit_port=dit_port)
    return apps[0], args


async def serve_services(apps, args, *, host, ports):
    import json
    import signal

    from sglang.multimodal_gen.runtime.realtime.source_transport import (
        SourceTransportBudget,
    )
    from sglang.multimodal_gen.runtime.realtime.worker_http import WorkerHTTPServer

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    health, business = [], []
    try:
        for port in ports["http"]:
            health.append(WorkerHTTPServer("encoder", host, port))
        # All apps share this one model lifespan and CUDA execution thread.
        async with apps[0].router.lifespan_context(apps[0]):
            try:
                for app, server in zip(apps, health):
                    controller = getattr(app.state, "worker_connections", None)
                    if controller is not None:
                        server.connections = controller
                        server.connection_peers = [
                            a.state.worker_connections for a in apps
                        ]
                        controller.install(server.app)
                    from sglang.multimodal_gen.runtime.entrypoints.openai.zing_cache_api import (
                        router as cache_router,
                    )

                    server.app.state.realtime_execution_client = (
                        app.state.realtime_execution_client
                    )
                    server.app.state.realtime_admission_controller = (
                        app.state.realtime_admission_controller
                    )
                    server.app.include_router(cache_router)
                for app, port in zip(apps, ports["ws"]):
                    business.append(
                        WorkerHTTPServer(
                            "encoder",
                            host,
                            port,
                            app=app,
                            **SourceTransportBudget.from_args(args).websocket_options(),
                        )
                    )
                for app in apps:
                    app.state.connection_ready = True
                for server in health:
                    server.ready = True
                print("ENCODER_SERVICES_READY " + json.dumps(ports), flush=True)
                await stop.wait()
            finally:
                for app in apps:
                    app.state.connection_ready = False
                for server in health:
                    server.ready = False
                for server in business:
                    server.server.should_exit = True
                for server in reversed(business):
                    await asyncio.to_thread(server.close)
    finally:
        for server in reversed(health):
            await asyncio.to_thread(server.close)
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pipeline-config-path", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--advertise-host", default="127.0.0.1")
    parser.add_argument("--worker-id")
    parser.add_argument("--rendezvous-port", type=int, default=32000)
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Base input WS port; service i uses port + i",
    )
    parser.add_argument("--service-count", type=int, default=1)
    parser.add_argument(
        "--session-max-lifetime-s",
        type=float,
        help="Hard session lifetime from first server output dispatch; defaults to the standard realtime limit",
    )
    parser.add_argument(
        "--http-port", type=int, help="Base management port; defaults to port + 1000"
    )
    cli = parser.parse_args()

    from sglang.multimodal_gen.runtime.realtime.zing_nccl import configure_zing_nccl

    configure_zing_nccl()

    from sglang.multimodal_gen.runtime.realtime.role_pool.reporting import (
        validate_worker_configuration,
    )

    validate_worker_configuration("encoder", cli.service_count)

    import uuid

    from sglang.multimodal_gen.runtime.realtime.worker_connection_ports import (
        ConnectionPorts,
    )

    connection_ports = ConnectionPorts(
        "encoder",
        cli.service_count,
        worker_id=cli.worker_id or "encoder-" + uuid.uuid4().hex,
        host=cli.advertise_host,
        bind_host=cli.host,
        input_ws=cli.port,
        http=cli.http_port if cli.http_port is not None else cli.port + 1000,
        rendezvous=cli.rendezvous_port,
    )
    ports = dict(
        ws=connection_ports.ports["input_ws"], http=connection_ports.ports["http"]
    )
    try:
        apps, args = create_apps(
            cli.model,
            service_count=cli.service_count,
            connection_ports=connection_ports,
            session_max_lifetime_s=cli.session_max_lifetime_s,
            pipeline_config=json.loads(cli.pipeline_config_path.read_text())
            if cli.pipeline_config_path
            else None,
        )
        asyncio.run(serve_services(apps, args, host=cli.host, ports=ports))
    finally:
        connection_ports.close()


if __name__ == "__main__":
    from sglang.multimodal_gen.runtime.realtime.worker_process import run_worker

    run_worker(main)
