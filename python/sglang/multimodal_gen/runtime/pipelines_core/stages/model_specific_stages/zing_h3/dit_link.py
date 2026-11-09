# SPDX-License-Identifier: Apache-2.0
"""Single-session queued DiT input with independent data sends and result notices.

Control and tensor descriptors use ZMQ; tensor payloads use NCCL. DiT's I/O
owners receive independently of the model loops. Direct Decoder output permits
send-and-continue; init, EOS drain and teardown retain completion barriers.
"""

import asyncio
from contextlib import nullcontext

from sglang.multimodal_gen.runtime.realtime.polling_task_queue import PollingTaskQueue
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from .source_bridge import source_pixel_end

logger = init_logger(__name__)


class DiTLinkClient:
    def __init__(self, channel):
        self.channel = channel
        self.owner = None
        self.failed = False
        self.sequence = 0
        self.async_inputs = False
        self.pending_results = []
        self.cache_identity = None

    def _validate_reply(self, reply):
        transfer = reply.get("transfer_id")
        if type(transfer) is not int or not 0 <= transfer < self.sequence:
            raise ValueError("DiT reply does not match a sent input")
        if reply["type"] == "worker.error":
            raise RuntimeError("DiT processing failed; restart the worker")
        if reply["type"] == "input.rejected":
            raise ValueError("DiT rejected the session identity")
        if (
            reply["type"] != "input.processed"
            and self.owner is not None
            and (reply.get("session_id"), reply.get("generation_id")) != self.owner
        ):
            raise ValueError("DiT output belongs to another session")

    def _collect_available(self):
        while self.channel.poll():
            reply = self.channel.receive()
            self._validate_reply(reply)
            if reply["type"] != "input.processed":
                self.pending_results.append(reply)
                if len(self.pending_results) > 128:
                    raise ValueError("DiT notifications exceed their bounded budget")

    def collect(self):
        if self.failed:
            raise RuntimeError("Encoder/DiT link requires restart")
        try:
            self._collect_available()
            results, self.pending_results = self.pending_results, []
            return results
        except BaseException:
            self.failed = True
            raise

    def exchange(self, message):
        if self.failed:
            raise RuntimeError("Encoder/DiT link requires restart")
        sequence = self.sequence
        self.sequence += 1
        outputs, self.pending_results = self.pending_results, []
        try:
            self.channel.send(dict(message, transfer_id=sequence))
            while True:
                reply = self.channel.receive()
                if not self.async_inputs and reply.get("transfer_id") != sequence:
                    raise ValueError("DiT reply does not match the input message")
                self._validate_reply(reply)
                if reply["type"] in ("input.processed", "input.rejected"):
                    if reply["type"] == "input.rejected":
                        raise ValueError("DiT rejected the session identity")
                    if reply["transfer_id"] == sequence:
                        return outputs
                    continue
                if reply["type"] == "worker.error":
                    raise RuntimeError("DiT processing failed; restart the worker")
                if (
                    self.owner is not None
                    and (reply.get("session_id"), reply.get("generation_id"))
                    != self.owner
                ):
                    raise ValueError("DiT output belongs to another session")
                outputs.append(reply)
                if len(outputs) > (128 if self.async_inputs else 10):
                    raise ValueError("DiT reply exceeds the output budget")
        except BaseException:
            from sglang.multimodal_gen.runtime.utils.realtime_video import (
                cancel_async_raw_rgb_frame_reference,
            )

            for item in outputs:
                if item.get("reference") is not None:
                    cancel_async_raw_rgb_frame_reference(item["reference"])
            self.failed = True
            raise

    def open(self, message):
        if self.owner is not None:
            raise ValueError("An Encoder/DiT session is already active")
        self.owner = message["session_id"], message["generation_id"]
        try:
            opened = self.exchange(dict(message, type="session.init"))
            self.async_inputs = any(r.get("async_inputs", False) for r in opened)
            return opened
        except BaseException:
            self.owner = None
            raise

    def send(self, kind, **fields):
        if self.owner is None:
            raise ValueError("Encoder/DiT session is not initialized")
        message = dict(
            type=kind,
            session_id=self.owner[0],
            generation_id=self.owner[1],
            **fields,
        )
        if not self.async_inputs or kind in ("source.end", "session.cancel"):
            return self.exchange(message)
        try:
            message["transfer_id"] = self.sequence
            self.sequence += 1
            # The communication owner keeps receiving notifications during the
            # next model invocation. Full queues still apply bounded backpressure.
            self.channel.send(message, on_wait=self._collect_available)
            return self.collect()
        except BaseException:
            self.failed = True
            raise

    def cache_control(self, operation, **arguments):
        if operation == "identity" and self.cache_identity is not None:
            return self.cache_identity
        message = dict(type="cache.control", operation=operation, arguments=arguments)
        if self.owner is not None:
            message.update(session_id=self.owner[0], generation_id=self.owner[1])
        replies = self.exchange(message)
        results = [item["result"] for item in replies if item["type"] == "cache.result"]
        self.pending_results.extend(
            item for item in replies if item["type"] != "cache.result"
        )
        if len(results) != 1:
            raise RuntimeError("Invalid cache control response")
        if operation == "identity":
            self.cache_identity = results[0]
        return results[0]

    def release(self, session_id):
        if self.owner is not None and session_id == self.owner[0]:
            try:
                self.send("session.cancel")
            finally:
                self.owner = None
                self.async_inputs = False

    def disconnect(self):
        """Close this link without terminating the remote model worker."""
        try:
            if self.owner is not None and not self.failed:
                self.release(self.owner[0])
        finally:
            if self.failed:
                self.channel.close(fault=True)
            else:
                self.channel.close()

    def shutdown(self):
        try:
            if not self.failed:
                if self.owner is not None:
                    self.release(self.owner[0])
                self.exchange(dict(type="worker.shutdown"))
        finally:
            self.channel.close()


class DiTInputService:
    def __init__(
        self,
        channel,
        create_session,
        create_decoder=None,
        *,
        on_failure=None,
        cleanup_guard=nullcontext,
        abort_transport=None,
    ):
        self.channel, self.create_session = channel, create_session
        self.create_decoder = create_decoder
        self.on_failure, self.cleanup_guard = on_failure, cleanup_guard
        self.abort_transport = abort_transport
        self.decoder = None
        self.transport_failed = False
        self.worker = self.owner = None
        self.stopped = False
        self.queue = PollingTaskQueue(self._process)
        self.sequence = 0

    async def _process(self, queue):
        from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError

        message = queue.popleft()
        sequence = message["transfer_id"]
        if sequence != self.sequence:
            raise ValueError("Encoder input sequence does not match")
        self.sequence += 1
        identity = message.get("session_id"), message.get("generation_id")
        kind = message["type"]

        def reply(kind, **fields):
            self.channel.send(dict(type=kind, transfer_id=sequence, **fields))

        def _sync_step_error(err=None):
            group = getattr(self.channel, "group", None)
            if group is None or getattr(group, "world_size", 1) <= 1:
                if err is not None:
                    raise err
                return
            local_payload = (
                None
                if err is None
                else (type(err).__name__, getattr(err, "code", None), str(err))
            )
            errors = [
                group.broadcast_object(
                    local_payload if group.rank_in_group == rank else None,
                    src=rank,
                )
                for rank in range(group.world_size)
            ]
            if err is not None:
                raise err
            for item in errors:
                if item is not None:
                    err_type, err_code, err_msg = item
                    if err_type == "ConditionError":
                        raise ConditionError(err_code or "CONDITION_INVALID", err_msg)
                    raise RuntimeError(
                        f"DiT rank operation failed ({err_type}): {err_msg}"
                    )

        if kind == "cache.control":
            if self.owner is not None and identity != self.owner:
                reply("input.rejected")
                return
            result = None
            step_err = None
            try:
                control = getattr(self.create_session, "cache_control", None)
                result = (
                    {"code": "cache_unavailable"}
                    if control is None
                    else control(
                        message["operation"],
                        message.get("arguments", {}),
                        busy=self.owner is not None,
                    )
                )
            except BaseException as exc:
                step_err = exc
            try:
                _sync_step_error(step_err)
            except BaseException:
                reply("worker.error")
                raise
            reply(
                "cache.result",
                **(
                    {"session_id": self.owner[0], "generation_id": self.owner[1]}
                    if self.owner is not None
                    else {}
                ),
                result=result,
            )
            reply("input.processed")
            return

        # Reject another owner without destroying the active session's state.
        if (kind == "session.init" and self.owner is not None) or (
            kind not in ("session.init", "worker.shutdown") and identity != self.owner
        ):
            reply("input.rejected")
            return
        try:
            if kind == "worker.shutdown":
                if self.worker is not None:
                    self.worker.close()
                self._close_decoder()
                self.stopped = True
            elif kind == "session.init":
                if not all(isinstance(v, str) and v for v in identity):
                    raise ValueError("Session identity is required")

                async def output(result):
                    item = dict(
                        type="dit.latents",
                        format="dit_latents",
                        transfer_id=self.current_transfer,
                        session_id=self.owner[0],
                        generation_id=self.owner[1],
                        tensor=result.visual,
                        latent_start=result.chunk.start,
                        latent_end=result.chunk.end,
                        event_id=result.prompt_event_id,
                        prompt_version=result.prompt_version,
                        sink_event_id=getattr(result, "sink_event_id", None),
                        sink_chunks=getattr(result, "sink_chunks", 0),
                        sink_chunk_ids=getattr(result, "sink_chunk_ids", ()),
                        **result.provenance.metadata(),
                    )
                    if self.decoder is not None:
                        item = self.decoder.output(result, self.current_message, item)
                    try:
                        self.channel.send(item)
                    except BaseException:
                        if item.get("reference") is not None:
                            from sglang.multimodal_gen.runtime.utils.realtime_video import (
                                cancel_async_raw_rgb_frame_reference,
                            )

                            cancel_async_raw_rgb_frame_reference(item["reference"])
                        raise

                self.worker = self.create_session(message, output)
                self.owner = identity
                if self.create_decoder is not None:
                    self.decoder = self.create_decoder(message)
                    reply(
                        "session.opened",
                        session_id=identity[0],
                        generation_id=identity[1],
                        output_kind="rgb",
                        async_inputs=bool(
                            getattr(
                                getattr(self.decoder, "remote", None), "oneway", False
                            )
                        ),
                        **(
                            {"output_attachment": self.decoder.remote.output_attachment}
                            if getattr(
                                getattr(self.decoder, "remote", None),
                                "output_attachment",
                                None,
                            )
                            else {}
                        ),
                    )
            elif kind == "session.cancel":
                self._close_decoder()
                self.worker.close()
                self.worker = self.owner = None
            else:
                # A rejected initial condition waits for the matching cancel;
                # already-sent source payloads must still receive their ACKs.
                if self.worker.closed:
                    reply("input.processed")
                    return
                self.current_transfer = sequence
                self.current_message = message
                initial_condition = (
                    kind == "conditioning" and not self.worker.initial_received
                )
                if kind == "source.moments":
                    # Identical boundary calculation to the integrated pipeline.
                    message["pixel_end"] = source_pixel_end(
                        message["latent_end"],
                        self.worker.bridge,
                        frame_end=message.get("final_source_frames"),
                    )
                    self.worker.source_records = message["source_records"]
                if kind == "conditioning" and message.get("cache_hit"):
                    expand_err = None
                    try:
                        expand = getattr(self.create_session, "cached_condition", None)
                        if expand is None:
                            raise ValueError("Cached conditioning is unavailable")
                        message = expand(message, initial=initial_condition)
                    except BaseException as exc:
                        expand_err = exc
                    _sync_step_error(expand_err)
                step_err = None
                try:
                    await self.worker.put(message)
                    while self.worker.queue:
                        await self.worker.queue.run_once()
                    if self.decoder is not None and kind in (
                        "source.moments",
                        "source.end",
                    ):
                        self.decoder.source_progress(
                            self.worker.bridge.accepted_pixels,
                            self.worker.source_records,
                        )
                except BaseException as exc:
                    step_err = exc
                _sync_step_error(step_err)
                notice = getattr(self.worker, "conditioning_notice", None)
                if notice is not None:
                    self.worker.conditioning_notice = None
                    reply(
                        "conditioning.rejected",
                        session_id=self.owner[0],
                        generation_id=self.owner[1],
                        **notice,
                    )
                if self.worker.complete:
                    if self.decoder is not None:
                        self.decoder.finish(self.worker.bridge.accepted_pixels)
                        self._close_decoder()
                    reply(
                        "session.completed",
                        session_id=self.owner[0],
                        generation_id=self.owner[1],
                        latent_frames=self.worker.bridge.accepted_latents,
                        source_frames=self.worker.bridge.accepted_pixels,
                    )
                    self.worker.close()
            reply("input.processed")
        except BaseException as exc:
            if isinstance(exc, ConditionError) and kind == "conditioning":
                if initial_condition:
                    self._close_decoder()
                    self.worker.close()
                elif self.worker.state is not None:
                    self.worker.state.discard_pending_prompt()
                reply(
                    "conditioning.rejected",
                    session_id=self.owner[0],
                    generation_id=self.owner[1],
                    event_id=message.get("event_id"),
                    initial=initial_condition,
                    code=exc.code,
                )
                reply("input.processed")
                return
            import logging as _logging

            (globals().get("logger") or _logging.getLogger(__name__)).exception(
                "DiT input processing failed"
            )
            # Teardown clears decoder; preserve the failure category for the
            # group owner before deciding whether model restart is required.
            remote = getattr(self.decoder, "remote", None)
            self.transport_failed = remote is not None and remote.failure is not None
            if self.on_failure is not None:
                self.on_failure()
            with self.cleanup_guard():
                if self.abort_transport is not None:
                    self.abort_transport()
                if remote is not None:
                    remote.failure = remote.failure or RuntimeError("DiT input failed")
                self._close_decoder()
                if self.worker is not None:
                    self.worker.close()
                # The channel may already be broken. Preserve the original
                # failure rather than blocking recovery on an error receipt.
                try:
                    reply("worker.error")
                except Exception:
                    pass
            raise

    def _close_decoder(self):
        if self.decoder is not None:
            self.decoder.close()
            self.decoder = None

    async def run(self):
        try:
            while not self.stopped:
                # Consume the I/O owner's completed-message inbox. Socket/NCCL
                # reception continues on its own thread during model execution.
                if self.queue:
                    await self.queue.run_once()
                elif self.channel.poll():
                    self.queue.append(self.channel.receive())
                else:
                    await asyncio.sleep(self.queue.poll_interval_s)
        finally:
            self._close_decoder()
            if self.worker is not None:
                self.worker.close()
