# SPDX-License-Identifier: Apache-2.0
"""Queue-driven DiT ingress for an externally encoded, single active session.

The owner supplies the existing initialized sampler/session factory, posterior
sampler and source bridge. No encoder, Qwen, decoder or model loader runs here.
All methods belong to one event loop; model calls remain single-writer.
"""

import asyncio

import torch

from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
from sglang.multimodal_gen.runtime.realtime.polling_task_queue import PollingTaskQueue


class QueuedDiTSession:
    def __init__(
        self,
        *,
        session_id,
        generation_id,
        create_session,
        bridge,
        sample_moments,
        on_output,
        validate_condition=None,
        prepare_condition=None,
        release_storage=None,
        poll_interval_s=0.005,
        max_control_tasks=10,
        max_condition_bytes=256 << 20,
    ):
        if not session_id or not generation_id:
            raise ValueError("DiT session identity is required")
        if max_control_tasks < 1 or max_condition_bytes < 1:
            raise ValueError("DiT queue budgets must be positive")
        self.identity = session_id, generation_id
        self.create_session = create_session
        self.bridge, self.sample_moments = bridge, sample_moments
        self.on_output, self.validate_condition = on_output, validate_condition
        self.prepare_condition = prepare_condition
        self.release_storage = release_storage
        self.max_control_tasks = max_control_tasks
        self.max_condition_bytes = max_condition_bytes
        self.condition_bytes = self.control_tasks = 0
        self.initial_received = self.eos_received = self.closed = False
        self.complete = self.in_flight = False
        self.last_event_id = None
        self.state = None
        self.conditioning_notice = None
        self._admission = asyncio.Lock()
        self.queue = PollingTaskQueue(
            self._process,
            arrange=self._arrange,
            ready=self._queue_ready,
            poll_interval_s=poll_interval_s,
        )

    def _queue_ready(self, queue):
        if self.closed:
            raise asyncio.CancelledError("DiT input session is closed")
        return bool(queue)

    def _open(self):
        if self.closed:
            raise ValueError("DiT input session is closed")

    async def put(self, message):
        """Accept decoded tensors from the transport, preserving session identity.

        Admission waits for bounded staging space. Conditions bypass incomplete
        frame groups, but never overtake an already selected model task.
        """
        async with self._admission:
            await self._put(message)

    async def _put(self, message):
        self._open()
        if (message.get("session_id"), message.get("generation_id")) != self.identity:
            raise ValueError("DiT input session or generation does not match")
        if self.eos_received:
            raise ValueError("DiT input has ended")
        kind = message.get("type")
        if kind in {"conditioning", "conditioning.cancel"}:
            cancelled = kind == "conditioning.cancel"
            if cancelled and not self.initial_received:
                raise ValueError("Initial conditioning must precede cancellation")
            tensors = message.get("tensors", {})
            if not cancelled:
                if set(tensors) != {"context", "row_role"}:
                    raise ValueError("DiT conditioning requires context and row_role")
                context, roles = tensors["context"], tensors["row_role"]
                if (
                    not isinstance(context, torch.Tensor)
                    or not isinstance(roles, torch.Tensor)
                    or context.ndim != 2
                    or roles.ndim != 1
                    or context.shape[0] != roles.shape[0]
                    or min(context.shape) < 1
                    or not context.is_floating_point()
                    or roles.dtype != torch.int64
                    or roles.device != context.device
                ):
                    raise ValueError("Invalid DiT conditioning tensor geometry")
            elif tensors or "inputs" in message:
                raise ValueError(
                    "Condition cancellation cannot carry tensors or inputs"
                )
            event_id = message.get("event_id")
            if self.initial_received and (
                type(event_id) is not int
                or event_id < 1
                or (self.last_event_id is not None and event_id <= self.last_event_id)
            ):
                raise ValueError("DiT prompt events must be strictly increasing")
            if (
                not self.initial_received
                and event_id is not None
                and (type(event_id) is not int or event_id < 0)
            ):
                raise ValueError("Invalid initial conditioning event identity")
            reference_tensors = []
            if "inputs" in message:
                if not isinstance(message["inputs"], dict):
                    raise ValueError("Prepared DiT inputs must be an object")
                for name in ("visual", "audio"):
                    values = message["inputs"].get(name, {})
                    if not isinstance(values, dict) or any(
                        not isinstance(value, torch.Tensor)
                        or not value.is_floating_point()
                        or value.device != context.device
                        for value in values.values()
                    ):
                        raise ValueError("Invalid DiT reference tensors")
                    reference_tensors.extend(values.values())
            nbytes = sum(
                t.numel() * t.element_size()
                for t in [*tensors.values(), *reference_tensors]
            )
            if nbytes > self.max_condition_bytes:
                raise ValueError("DiT condition exceeds queue capacity")
            while (
                self.control_tasks >= self.max_control_tasks
                or self.condition_bytes + nbytes > self.max_condition_bytes
            ):
                await asyncio.sleep(self.queue.poll_interval_s)
                self._open()
            task = dict(
                message,
                tensors={k: v.detach().clone() for k, v in tensors.items()},
                initial=not self.initial_received,
                nbytes=nbytes,
            )
            if "inputs" in message:
                task["inputs"] = dict(message["inputs"])
                for name in ("visual", "audio"):
                    task["inputs"][name] = {
                        key: value.detach().clone()
                        for key, value in message["inputs"].get(name, {}).items()
                    }
            self.initial_received = True
            self.last_event_id = event_id
            self.control_tasks += 1
            self.condition_bytes += nbytes
        elif kind in {"source.moments", "source.latents"}:
            if not self.initial_received:
                raise ValueError("Initial conditioning must precede source data")
            value = message.get("tensor")
            start, end = message.get("latent_start"), message.get("latent_end")
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 5
                or min(value.shape) < 1
                or not value.is_floating_point()
                or type(start) is not int
                or type(end) is not int
                or start != self.bridge.accepted_latents
                or end != start + value.shape[2]
            ):
                raise ValueError("Invalid or out-of-order DiT source tensor")
            # SourceChunkBridge validates clip geometry, capacity and ownership.
            # Bound admission before posterior sampling allocates another tensor.
            nbytes = value.numel() * value.element_size()
            if (
                value.shape[2] > self.bridge.capacity_latents
                or nbytes > self.bridge.capacity_bytes
            ):
                raise ValueError("DiT source exceeds queue capacity")
            while (
                self.bridge.buffered_latents + value.shape[2]
                > self.bridge.capacity_latents
                or self.bridge.buffered_bytes + nbytes > self.bridge.capacity_bytes
            ):
                await asyncio.sleep(self.queue.poll_interval_s)
                self._open()
            with torch.no_grad():
                source = (
                    self.sample_moments(value, latent_start=start)
                    if kind == "source.moments"
                    else value
                )
            task = dict(message, tensor=source)
        elif kind == "source.end":
            if (
                not self.initial_received
                or type(message.get("latent_frames")) is not int
                or message.get("latent_frames") != self.bridge.accepted_latents
            ):
                raise ValueError("DiT EOS does not cover accepted source latents")
            task = dict(message)
        else:
            raise ValueError("Unsupported DiT input message")
        self.queue.append(task)
        if kind == "source.end":
            self.eos_received = True

    def _arrange(self, message):
        if message is not None:
            kind = message["type"]
            if kind in {"conditioning", "conditioning.cancel"}:
                return [message]
            if kind == "source.end":
                self.bridge.finish()
            else:
                self.bridge.append(
                    message["tensor"],
                    latent_start=message["latent_start"],
                    pixel_end=message["pixel_end"],
                    source_dependency_end=message["source_dependency_end"],
                )
        # Select at most one frame task. A queued prompt must be consumed before
        # consulting next_chunk_size, since it can reset the model phase.
        if self.queue or self.in_flight or self.state is None:
            return []
        if self.bridge.ended and not self.bridge.buffered_latents:
            return [dict(type="source.end")]
        count = self.state.next_chunk_size
        if count is None or not self.bridge.ready_for(count):
            return []
        return [dict(type="source.group", chunk=self.bridge.pop(count))]

    async def _process(self, queue):
        task = queue.popleft()
        self.in_flight = True
        try:
            if task["type"] in {"conditioning", "conditioning.cancel"}:
                try:
                    if task["type"] == "conditioning.cancel":
                        self.state.discard_pending_prompt()
                    else:
                        # A newer event supersedes the previous pending update,
                        # including when admission rejects the new candidate.
                        if not task["initial"]:
                            self.state.discard_pending_prompt()
                        candidate = None
                        if self.prepare_condition is not None:
                            candidate = self.prepare_condition(task)
                        if self.validate_condition is not None:
                            self.validate_condition(task["tensors"])
                        if task["initial"]:
                            self.state = self.create_session(
                                task["tensors"],
                                task.get("event_id"),
                                task.get("prompt_version"),
                            )
                        else:
                            # The existing session rebuilds rolling prefix KV at the
                            # original safe boundary; it never runs Qwen here.
                            self.state.update_prompt(
                                task["tensors"],
                                task["event_id"],
                                task.get("prompt_version"),
                                **(
                                    {"inputs": candidate}
                                    if candidate is not None
                                    else {}
                                ),
                            )
                except ConditionError as exc:
                    if task["initial"]:
                        raise
                    self.conditioning_notice = {
                        "code": exc.code,
                        "event_id": task["event_id"],
                    }
                finally:
                    self.control_tasks -= 1
                    self.condition_bytes -= task["nbytes"]
            elif task["type"] == "source.group":
                chunk = task["chunk"]
                self.state.append(
                    chunk.source,
                    chunk.noise,
                    provenance=chunk.provenance,
                    final=chunk.final,
                )
                result = self.state.step()
                await self.on_output(result)
            else:
                if not self.state.ended:
                    self.state.finish()
                if not self.state.complete:
                    raise ValueError("DiT did not drain its source before EOS")
                self.complete = True
        except BaseException:
            self.close()
            raise
        finally:
            self.in_flight = False
        if not self.complete:
            # Reconsider buffered frames without waiting for any new arrival.
            self.queue.append(None)

    async def run(self):
        try:
            while not self.complete:
                self._open()
                await self.queue.run_once()
        finally:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.queue.clear()
            self.bridge.dispose()
            if self.state is not None:
                self.state.dispose()
        finally:
            if self.release_storage is not None:
                self.release_storage()
            self.condition_bytes = self.control_tasks = 0
