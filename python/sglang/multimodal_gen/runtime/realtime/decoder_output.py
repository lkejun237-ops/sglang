# SPDX-License-Identifier: Apache-2.0
"""Single-owner Decoder output mailbox, independent of the model thread.

Only local shared-memory references enter this mailbox. RGB stays on Decoder
and is consumed by the existing realtime output adapter.
"""

import logging
import queue
import secrets
import threading


class DecoderOutputSession:
    def __init__(self, config, cancel_reference, *, capacity=10):
        self.config = dict(config)
        self.token = secrets.token_urlsafe(32)
        self.queue = queue.Queue(maxsize=capacity)
        self.cancel_reference = cancel_reference
        self.lock = threading.Lock()
        self.attached = False
        self.stopped = threading.Event()
        self.done = threading.Event()
        self.completed = False
        self.inflight = None
        self.owned_frame_end = self.consumed_frames = 0

    def source_progress(self, owned_frame_end, records, timeout):
        from sglang.multimodal_gen.runtime.realtime.zing_source_stream import (
            source_consumed_message,
        )

        if self.stopped.is_set():
            return
        if type(owned_frame_end) is not int or owned_frame_end < self.owned_frame_end:
            raise ValueError("Invalid Decoder source ownership watermark")
        previous = None
        boundaries = []
        # Credit acknowledges wire input, independently of source resizing and
        # output resolution. Older same-size callers omit source_stream.
        source = self.config.get("source_stream", self.config)
        if not isinstance(source, dict) or any(
            type(source.get(k)) is not int or source[k] < 1 for k in ("width", "height")
        ):
            raise ValueError("Invalid Decoder source stream dimensions")
        frame_bytes = source["width"] * source["height"] * 3
        for record in records:
            if not isinstance(record, (list, tuple)) or len(record) != 3:
                raise ValueError("Invalid source packet boundary")
            source_id, start, end = record
            if (
                any(type(v) is not int for v in record)
                or source_id < 1
                or not 0 <= start < end
                or (
                    previous is not None
                    and (source_id != previous[0] + 1 or start != previous[1])
                )
            ):
                raise ValueError("Invalid source packet sequence")
            previous = source_id, end
            boundaries.append((end, end * frame_bytes))
        if (
            not boundaries
            or owned_frame_end > boundaries[-1][0]
            or records[0][1] > self.consumed_frames
        ):
            raise ValueError("Source packets do not cover the ownership watermark")
        message = source_consumed_message(
            boundaries, self.consumed_frames, owned_frame_end
        )
        if message is not None:
            # Same FIFO as RGB: current dispatch output is sent before credit.
            # Progress without an RGB block still grants complete input packets.
            self.put(message, timeout, discard_if_stopped=True)
            self.consumed_frames = message["source_frame_end"]
        self.owned_frame_end = owned_frame_end

    def attach(self, message):
        with self.lock:
            if (
                self.stopped.is_set()
                or self.attached
                or message.get("type") != "output.attach"
                or any(
                    message.get(k) != self.config[k]
                    for k in ("session_id", "generation_id")
                )
                or not secrets.compare_digest(str(message.get("token", "")), self.token)
            ):
                raise ValueError(
                    "Output connection does not match an available session"
                )
            self.attached = True

    def put(self, item, timeout, *, discard_if_stopped=False):
        # Do not hold a lock across a bounded put: disconnect must unblock it.
        import time

        deadline = time.monotonic() + timeout
        while True:
            with self.lock:
                if self.stopped.is_set():
                    if discard_if_stopped:
                        break
                    raise RuntimeError("Decoder output connection is closed")
                try:
                    self.queue.put_nowait(item)
                    return
                except queue.Full:
                    pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Decoder output queue is full")
            self.stopped.wait(0.005)
        # The input owner cancels behind already-sent NCCL payloads. Continue
        # receiving those payloads after output disconnect, without retaining
        # their RGB references or treating cancellation as a model failure.
        reference = item.get("reference")
        if reference is not None:
            self.cancel_reference(reference)

    def take(self):
        with self.lock:
            if self.stopped.is_set():
                raise RuntimeError("Decoder output connection is closed")
            item = self.queue.get_nowait()
            self.inflight = item.get("reference")
            return item

    def sent(self):
        with self.lock:
            self.inflight = None

    def abort(self):
        with self.lock:
            self.stopped.set()
            refs = [self.inflight]
            self.inflight = None
            while not self.queue.empty():
                refs.append(self.queue.get_nowait().get("reference"))
        for ref in refs:
            if ref is not None:
                try:
                    self.cancel_reference(ref)
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Decoder output cleanup failed"
                    )
        self.done.set()


class DecoderOutputBroker:
    def __init__(self, cancel_reference):
        self.cancel_reference = cancel_reference
        self.lock = threading.Lock()
        self.session = None

    def open(self, config):
        if not all(
            isinstance(config.get(k), str) and config[k]
            for k in ("session_id", "generation_id")
        ):
            raise ValueError("Decoder output session identity is required")
        with self.lock:
            if self.session is not None:
                raise ValueError("Decoder output session is already active")
            self.session = DecoderOutputSession(config, self.cancel_reference)
            return self.session

    def attach(self, message):
        with self.lock:
            if self.session is None:
                raise ValueError("Decoder output session is unavailable")
            self.session.attach(message)
            return self.session

    def close(self, session):
        if session is None:
            return
        session.abort()
        with self.lock:
            if self.session is session:
                self.session = None

    def install(self, app):
        # Imports are local so mailbox lifecycle tests do not need GPU packages.
        import asyncio
        import time
        from types import SimpleNamespace

        import msgspec.msgpack
        from fastapi import WebSocket, WebSocketDisconnect

        from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
            RawRGBRealtimeOutputAdapter,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )
        from sglang.multimodal_gen.runtime.realtime.websocket_transport import (
            _close_realtime_websocket,
            _drain_realtime_cleanup,
            accept_realtime_websocket,
            cancel_realtime_tasks,
            wait_realtime_tasks,
            write_error_msg,
        )
        from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
            PixelRange,
            Provenance,
            SourceSpan,
        )

        @app.websocket("/v1/realtime_video/output")
        async def output(websocket: WebSocket):
            ws = await accept_realtime_websocket(websocket)
            session = None
            watcher = sender = None

            async def watch_disconnect():
                while True:
                    message = await ws.receive()
                    if message["type"] == "websocket.disconnect":
                        session.abort()
                        return
                    raise ValueError(
                        "Decoder output connection only accepts an attachment"
                    )

            async def send_outputs():
                adapter = RawRGBRealtimeOutputAdapter()
                config = session.config
                pending_timings = []

                async def send_timings(final=False):
                    retained = []
                    for timing in pending_timings:
                        if not timing.finished and not final:
                            retained.append(timing)
                            continue
                        report = timing.snapshot(final=final)
                        if report["status"] == "pending" and not final:
                            retained.append(timing)
                            continue
                        if final and report["status"] == "pending":
                            report["status"] = "incomplete"
                        await ws.send_bytes(
                            msgspec.msgpack.encode(
                                dict(
                                    type="model.timing",
                                    model_session_id=config["session_id"],
                                    generation_id=config["generation_id"],
                                    timing=report,
                                )
                            )
                        )
                    pending_timings[:] = retained

                identity = dict(
                    session_id=config["session_id"],
                    generation_id=config["generation_id"],
                )
                await ws.send_bytes(
                    msgspec.msgpack.encode(dict(type="output.ready", **identity))
                )
                last_decoder_timing = None
                while True:
                    await send_timings()
                    try:
                        item = session.take()
                    except queue.Empty:
                        await asyncio.sleep(0.005)
                        continue
                    if item["type"] == "session.completed":
                        await send_timings(final=True)
                        await ws.send_bytes(
                            msgspec.msgpack.encode(dict(item, **identity))
                        )
                        session.completed = True
                        session.done.set()
                        return
                    if item["type"] == "source.consumed":
                        await ws.send_bytes(msgspec.msgpack.encode(item))
                        continue
                    if item["type"] == "decoder.timing":
                        if not config.get("return_stage_timing", False):
                            continue
                        dec_info = last_decoder_timing or {}
                        last_decoder_timing = None
                        dec_stage = {
                            "wait_ms": round(
                                max(0.0, float(dec_info.get("wait_ms", 0.0))), 3
                            ),
                            "process_ms": round(
                                max(0.0, float(dec_info.get("process_ms", 0.0))), 3
                            ),
                            "return_ms": round(
                                max(0.0, float(dec_info.get("return_ms", 0.0))), 3
                            ),
                        }
                        frames_breakdown = [
                            dict(entry, decoder=dict(dec_stage))
                            for entry in (item.get("frame_stage_breakdown") or ())
                            if isinstance(entry, dict)
                        ]
                        num_frames = item.get("num_frames")
                        if type(num_frames) is not int or num_frames < 1:
                            num_frames = len(frames_breakdown)
                        summary = {}
                        if frames_breakdown:
                            for role in ("encoder", "dit", "decoder"):
                                for metric in ("wait_ms", "process_ms", "return_ms"):
                                    vals = [
                                        float(row[role][metric])
                                        for row in frames_breakdown
                                        if isinstance(row.get(role), dict)
                                        and type(row[role].get(metric)) in (int, float)
                                    ]
                                    if vals:
                                        summary[f"{role}_{metric}"] = round(
                                            sum(vals) / len(vals), 3
                                        )
                        for extra_key in ("transport_encode_ms", "transport_write_ms"):
                            if type(dec_info.get(extra_key)) in (int, float):
                                summary[extra_key] = round(
                                    max(0.0, float(dec_info[extra_key])), 3
                                )
                        await ws.send_bytes(
                            msgspec.msgpack.encode(
                                dict(
                                    type="chunk_telemetry",
                                    **identity,
                                    request_id=item.get("request_id"),
                                    chunk_index=item.get("chunk_index"),
                                    output_frame_start=item.get("output_frame_start"),
                                    num_frames=num_frames,
                                    frame_stage_breakdown=frames_breakdown,
                                    **summary,
                                )
                            )
                        )
                        continue
                    meta = item["output_metadata"]
                    provenance = Provenance(
                        PixelRange(
                            meta["output_frame_start"],
                            meta["output_frame_start"] + meta["num_frames"],
                        ),
                        tuple(SourceSpan(**v) for v in meta["source_alignment"]),
                        meta["source_dependency_end"],
                    )
                    ref = item["reference"]
                    batch = SimpleNamespace(
                        request_id=ref["request_id"],
                        block_idx=ref["chunk_index"],
                        width=config["width"],
                        height=config["height"],
                        fps=config["fps"],
                        realtime_output_format=config.get("realtime_output_format"),
                        output_compression=config.get("output_compression"),
                        realtime_preview_max_width=config.get(
                            "realtime_preview_max_width"
                        ),
                        extra={
                            "zing_source_output_blocks": [
                                dict(
                                    meta,
                                    provenance=provenance,
                                    request_id=ref["request_id"],
                                )
                            ]
                        },
                    )
                    result = OutputBatch(
                        raw_frame_shared_memory_ref=ref,
                        raw_frame_metadata=item["metadata"],
                    )
                    stats = None
                    try:
                        async with ws.send_group():
                            stats = await adapter.send(
                                ws,
                                SimpleNamespace(
                                    id=config["session_id"],
                                    generation_id=config["generation_id"],
                                    trace_id=config.get("trace_id"),
                                ),
                                result,
                                batch,
                            )
                        timing = item.get("timing")
                        if timing is not None:
                            pending_timings.append(timing)
                            if len(pending_timings) >= 64:
                                await send_timings(final=True)
                    finally:
                        session.sent()
                    if not config.get("return_stage_timing", False):
                        continue
                    sent_ns = time.perf_counter_ns()
                    tstate = item.get("timing_state")
                    stats_dict = stats if isinstance(stats, dict) else {}
                    encode_ms = float(
                        stats_dict.get("raw_payload_build_ms", 0.0)
                        + stats_dict.get("msgpack_encode_ms", 0.0)
                    )
                    write_ms = float(stats_dict.get("ws_send_ms", 0.0))
                    if isinstance(tstate, dict):
                        done_ns = tstate.get("done_mono_ns")
                        ret_ms = (
                            max(0.0, (sent_ns - done_ns) / 1e6)
                            if type(done_ns) is int
                            else max(0.0, encode_ms + write_ms)
                        )
                        last_decoder_timing = dict(
                            wait_ms=max(0.0, float(tstate.get("wait_ms", 0.0))),
                            process_ms=max(0.0, float(tstate.get("process_ms", 0.0))),
                            return_ms=ret_ms,
                            transport_encode_ms=max(0.0, encode_ms),
                            transport_write_ms=max(0.0, write_ms),
                        )
                    else:
                        last_decoder_timing = dict(
                            wait_ms=0.0,
                            process_ms=0.0,
                            return_ms=max(0.0, encode_ms + write_ms),
                            transport_encode_ms=max(0.0, encode_ms),
                            transport_write_ms=max(0.0, write_ms),
                        )

            try:
                message = msgspec.msgpack.decode(
                    await asyncio.wait_for(ws.receive_bytes(), 30)
                )
                session = self.attach(message)
                watcher = asyncio.create_task(watch_disconnect())
                sender = asyncio.create_task(send_outputs())
                done, _ = await wait_realtime_tasks([watcher, sender])
                for task in done:
                    task.result()
                if (
                    not session.completed
                    and not session.stopped.is_set()
                    and getattr(self, "on_send_failure", None) is not None
                ):
                    self.on_send_failure()
            except WebSocketDisconnect:
                if session is not None:
                    session.abort()
            except Exception:
                # An owner abort ends only its request, including initial capacity
                # refusal. Invalid output protocol still quarantines below; client disconnect cancels.
                if session is not None and session.stopped.is_set():
                    return
                logging.getLogger(__name__).exception(
                    "Decoder output connection failed"
                )
                if session is not None:
                    callback = getattr(self, "on_send_failure", None)
                    if callback is not None:
                        callback()
                try:
                    await write_error_msg(
                        "Video output could not be completed.",
                        ws,
                        code="DECODER_OUTPUT_FAILED",
                        fatal=True,
                    )
                except Exception:
                    pass
            finally:

                async def cleanup():
                    if session is not None:
                        session.abort()
                    await cancel_realtime_tasks(watcher, sender)
                    await _close_realtime_websocket(
                        ws, code=1000, reason="Output connection closed"
                    )

                await _drain_realtime_cleanup(cleanup())
