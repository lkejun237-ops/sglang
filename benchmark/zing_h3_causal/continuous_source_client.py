#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Send video or an RGB pipe progressively over the production MessagePack API.

No total length is sent. EOF causes source.end, never a successful bare close.
Use a live engine pipe, or a documented, non-looping raw export of real media.
The receiver records local WebSocket delivery, not browser presentation or WAN
capture latency. Hashing and bounded sampling are included; video encoding runs
after request completion. Full raw RGB output requires --media-mode raw.
"""

import argparse
import asyncio
import hashlib
import json
import threading
import time
from collections import deque
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path

import msgpack
import websockets
from bench_client import calculate_metrics, distribution
from benchmark_artifacts import TIMING_SCOPE, MediaCapture, check_retention
from continuous_source_input import open_source, source_dimensions
from continuous_source_media import load_prompt_updates, reference_payload
from decoder_output_client import (
    ThreadCredit,
    consume_source_credit,
    retire_source_boundaries,
)

from sglang.multimodal_gen.runtime.realtime.source_transport import (
    DEFAULT_MESSAGE_BYTES,
    PROTOCOL_OVERHEAD_BYTES,
    SourceTransportBudget,
)
from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
    DEFAULT_MAX_REFERENCE_IMAGES,
)

CLIENT_TIMING = {
    "clock": "CLOCK_MONOTONIC",
    "origin": "client_start_before_connect_and_source_decode",
    "init_origin": "serialized_init_immediately_before_websocket_send",
    "arrival": "complete_rgb_payload_received_after_message_unpack",
    "wire_arrival": "complete_rgb_wire_received_before_message_unpack",
    "e2e_end": "last_valid_rgb_batch_received",
    "elapsed_end": "receiver_and_source_tasks_finished_after_server_close",
    "scope": TIMING_SCOPE,
}


@asynccontextmanager
async def connection_preserving_error(connector):
    """Legacy WebSockets may raise again while closing an aborted transport."""
    ws = await connector.__aenter__()
    try:
        yield ws
    except BaseException as exc:
        try:
            await connector.__aexit__(type(exc), exc, exc.__traceback__)
        except Exception as close_error:
            exc.add_note(f"Connection cleanup failed: {type(close_error).__name__}")
        raise
    else:
        await connector.__aexit__(None, None, None)


def websocket_options(*, profile, timeout, message_budget=DEFAULT_MESSAGE_BYTES):
    return dict(
        max_size=message_budget,
        max_queue=1,
        compression=None,
        # Total request timeout remains bounded independently of keepalive.
        ping_timeout=timeout if profile else 20,
    )


async def run(args, *, output_sink=None):
    source_width, source_height = source_dimensions(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    # Custom sinks own their latent/codec artifacts and retain their existing API.
    capture = (
        None
        if output_sink is not None
        else MediaCapture(
            output,
            args.width,
            args.height,
            args.expected_frames,
            getattr(args, "media_mode", "representative"),
            args.ffmpeg,
        )
    )
    try:
        async with asyncio.timeout(args.timeout):
            with (
                (
                    (output / output_sink.filename).open("wb")
                    if output_sink is not None
                    else nullcontext(capture)
                ) as received_file,
                (output / "events.jsonl").open("w") as trace,
            ):
                async with open_source(
                    args, source_width * source_height * 3, output / "source-decode.log"
                ) as read_source:
                    summary = await exchange(
                        args,
                        output,
                        started,
                        read_source,
                        received_file,
                        trace,
                        output_sink=output_sink,
                    )
    finally:
        try:
            if capture is not None:
                receipt = await asyncio.to_thread(capture.finish)
        finally:
            check_retention(output)
    if capture is not None:
        summary.update(
            media=receipt, media_path=receipt["media_path"], media_mode=capture.mode
        )
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


async def exchange(
    args, output, started, read_source, received_file, trace, *, output_sink=None
):
    timing = (
        {**CLIENT_TIMING, "scope": received_file.timing_scope}
        if output_sink is None
        else CLIENT_TIMING
    )
    prompt_updates = deque(load_prompt_updates(args.prompt_updates))
    message_budget = getattr(args, "max_message_bytes", DEFAULT_MESSAGE_BYTES)
    SourceTransportBudget(message_bytes=message_budget)
    source_width, source_height = source_dimensions(args)
    frame_bytes = source_width * source_height * 3
    output_frame_bytes = args.width * args.height * 3
    source_format = dict(
        format="rgb24", width=source_width, height=source_height, fps=24
    )
    input_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    state = dict(
        submitted=0,
        accepted=0,
        consumed=0,
        consumed_bytes=0,
        received=0,
        sequence=0,
        event_id=0,
        ended=False,
        completed=False,
        first_output=None,
        last_output=None,
        stall_seconds=0.0,
        next_playback=None,
        max_pending=0,
        first_source_send=None,
        last_source_send=None,
        source_end_send=None,
        sink_confirmed_bytes=0,
        sink_semantics="processed_rgb_bytes; persistence_is_recorded_in_media_receipt",
        credit_wait_seconds=0.0,
    )
    prompt_results = {}
    boundaries, accepted, capture, sent = deque(), deque(), deque(), deque()
    latency_samples, delivery_times, wire_delivery_times = [], [], []
    dependency_latency_samples = []
    block_capture_latency_samples = []
    credit, writer = ThreadCredit(), asyncio.Lock()
    trace_lock = threading.Lock()
    submitted_packets = deque()

    def record(kind, **fields):
        with trace_lock:
            trace.write(
                json.dumps(
                    dict(kind=kind, seconds=time.monotonic() - started, **fields)
                )
                + "\n"
            )
            trace.flush()

    def record_applied_prompt(message):
        update = prompt_results.get(message.get("event_id"))
        if update is not None and update["status"] == "pending":
            received = time.monotonic() - started
            update.update(
                status="applied",
                received_seconds=received,
                latency_ms=(received - update["submitted_seconds"]) * 1000,
                output_frame_start=message["output_frame_start"],
                pts_us=message.get("pts_us"),
            )
            # Only observed output of a newer epoch proves an older
            # pending event was superseded; submission/ACK cannot.
            for older in list(prompt_results.values()):
                if (
                    older["event_id"] < update["event_id"]
                    and older["status"] == "pending"
                ):
                    older["status"] = "superseded"
            record("prompt_applied", **update)

    async with (
        connection_preserving_error(
            websockets.connect(
                args.url,
                **websocket_options(
                    profile=args.profile,
                    timeout=args.timeout,
                    message_budget=message_budget,
                ),
            )
        ) as ws,
    ):

        async def send(message):
            wire = msgpack.packb(message, use_bin_type=True)
            if len(wire) > message_budget:
                raise ValueError("Message exceeds the configured transport size limit")
            async with writer:
                if message.get("type") == "init":
                    state["init_send"] = time.monotonic()
                await ws.send(wire)
            return len(wire)

        init = dict(
            type="init",
            zing_realtime_api_version=1,
            prompt=args.prompt,
            size=f"{args.width}x{args.height}",
            fps=24,
            seed=args.seed,
            zing_realtime_decode_mode="stable",
            realtime_output_format="raw",
            source_stream=source_format,
            source_message_budget_bytes=message_budget,
        )
        if getattr(args, "cache_write", False):
            init["cache_write"] = True
        if getattr(args, "receive_stage_timing", False):
            init["return_stage_timing"] = True
        if args.profile:
            init.update(profile=True, profile_all_stages=True)
        if args.reference:
            refs = []
            for i, name in enumerate(args.reference):
                media_id = (
                    args.reference_id[i] if args.reference_id else f"reference-{i}"
                )
                refs.append(reference_payload(name, media_id))
            init["references"] = refs
        await send(init)
        record("init_sent", init_send_seconds=state["init_send"] - started)
        (output / "client-timing.json").write_text(
            json.dumps(
                {
                    **timing,
                    "init_send_seconds": state["init_send"] - started,
                    "client_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                },
                indent=2,
            )
        )
        ready = msgpack.unpackb(
            await asyncio.wait_for(ws.recv(), args.timeout), raw=False
        )
        if (
            ready.get("type") != "session.ready"
            or ready.get("zing_realtime_api_version") != 1
            or ready.get("source_stream") != source_format
            or type(ready.get("num_inference_steps")) is not int
            or ready["num_inference_steps"] < 1
            or ready.get("realtime_output_format") != "raw"
            or ready.get("max_message_bytes") != message_budget
            or not 0 < ready.get("max_source_message_bytes", 0) <= message_budget
        ):
            raise ValueError("Continuous source negotiation failed")
        max_references = ready.get("max_references", DEFAULT_MAX_REFERENCE_IMAGES)
        if type(max_references) is not int or max_references < 0:
            raise ValueError("The service advertised an invalid reference count limit")
        for update in prompt_updates:
            if "references" in update and len(update["references"]) > max_references:
                raise ValueError(
                    f"The service accepts at most {max_references} reference images"
                )
        if output_sink is not None:
            output_sink.negotiate(ready)
        decoder_url = getattr(args, "decoder_url", None)
        split_output = ready.get("output_transport") == "decoder_websocket"
        if split_output:
            if not decoder_url or output_sink is not None:
                raise ValueError(
                    "Direct Decoder RGB output requires --decoder-url and --output-kind rgb"
                )
        elif decoder_url:
            raise ValueError("The service did not negotiate direct Decoder output")
        record("ready", message=ready)
        needs_reference_updates = any(
            "references" in update for update in prompt_updates
        )
        # Includes [] (clear). Missing capability means unsupported, not an
        # activation status. The server requires prefix5 / [3,2] / split_first_latent.
        supports_reference_updates = ready.get("prompt_reference_updates", False)
        if needs_reference_updates and not supports_reference_updates:
            raise ValueError("The service did not advertise prompt reference updates")
        (output / "request.json").write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "ready": ready,
                    "source_mode": (
                        "video-decode"
                        if getattr(args, "source_video", None)
                        else "live-pipe"
                        if args.source_rgb == "-"
                        else "raw-replay"
                    ),
                    "scope": "client WebSocket delivery; not browser presentation",
                },
                indent=2,
            )
        )
        packet_frames = min(
            ready["max_source_batch_frames"],
            ready["max_pending_source_frames"],
            ready["max_pending_source_bytes"] // frame_bytes,
            (ready["max_source_message_bytes"] - PROTOCOL_OVERHEAD_BYTES)
            // frame_bytes,
        )
        if packet_frames < 1:
            raise ValueError("Negotiated source budget cannot hold one RGB frame")

        async def produce():
            source_started = time.monotonic()
            pattern_index = 0
            while True:
                if (
                    prompt_updates
                    and prompt_updates[0]["source_frame_start"] == state["submitted"]
                ):
                    update = prompt_updates.popleft()
                    state["event_id"] += 1
                    event = dict(
                        type="event",
                        kind="prompt",
                        event_id=state["event_id"],
                        payload=update["prompt"],
                    )
                    if "references" in update:
                        event["references"] = [
                            reference_payload(item["path"], item["id"])
                            for item in update["references"]
                        ]
                    reference_assets = (
                        [
                            dict(
                                id=item["id"],
                                format=item["format"],
                                bytes=len(item["payload"]),
                                sha256=hashlib.sha256(item["payload"]).hexdigest(),
                            )
                            for item in event.get("references", [])
                        ]
                        if "references" in event
                        else None
                    )
                    # Register before awaiting send: includes serialization,
                    # writer waiting and upload, not local image-file reads.
                    submitted = time.monotonic() - started
                    prompt_results[event["event_id"]] = dict(
                        event_id=event["event_id"],
                        status="pending",
                        source_frame_start=state["submitted"],
                        submitted_seconds=submitted,
                        received_seconds=None,
                        latency_ms=None,
                        output_frame_start=None,
                        pts_us=None,
                    )
                    record(
                        "prompt_send",
                        submitted_seconds=submitted,
                        event_id=event["event_id"],
                        source_frame_start=state["submitted"],
                        prompt=update["prompt"],
                        references=update.get("references"),
                        reference_assets=reference_assets,
                    )
                    await send(event)
                count = min(
                    packet_frames,
                    args.batch_pattern[pattern_index % len(args.batch_pattern)],
                )
                if prompt_updates:
                    count = min(
                        count,
                        prompt_updates[0]["source_frame_start"] - state["submitted"],
                    )
                pattern_index += 1
                data = await read_source(count * frame_bytes)
                if not data:
                    break
                if len(data) % frame_bytes:
                    raise ValueError("Source contains an incomplete RGB frame")
                count = len(data) // frame_bytes
                start = state["submitted"]
                if args.paced:
                    await asyncio.sleep(
                        max(
                            0,
                            source_started
                            + (start + count - 1) / 24
                            - time.monotonic(),
                        )
                    )
                available_at = time.monotonic()
                # In paced replay the intended capture time is independent of
                # credit stalls. Using read time would hide upstream backlog.
                capture_at = source_started + start / 24 if args.paced else available_at
                await credit.wait_for(
                    lambda: (
                        start + count - state["consumed"]
                        <= ready["max_pending_source_frames"]
                        and (start + count) * frame_bytes - state["consumed_bytes"]
                        <= ready["max_pending_source_bytes"]
                    )
                )
                credit_wait = time.monotonic() - available_at
                state["credit_wait_seconds"] += credit_wait
                state["event_id"] += 1
                event_id = state["event_id"]
                payload = dict(
                    client_seq=state["sequence"],
                    source_frame_start=start,
                    num_frames=count,
                    pts_us=start * 1_000_000 // 24,
                    data=data,
                )
                message = dict(
                    type="event",
                    kind="source.append",
                    event_id=event_id,
                    payload=payload,
                )
                if (
                    len(msgpack.packb(message, use_bin_type=True))
                    > ready["max_source_message_bytes"]
                ):
                    raise ValueError(
                        "Source append exceeds the negotiated message size"
                    )
                with credit.lock:
                    boundaries.append(
                        (start + count, (start + count) * frame_bytes, event_id)
                    )
                    if split_output:
                        # Registered before upload; Decoder output never waits
                        # for the independent Encoder acknowledgement stream.
                        submitted_packets.append(
                            dict(
                                source_id=state["sequence"] + 1,
                                source_frame_start=start,
                                source_frame_end=start + count,
                            )
                        )
                    capture.append((start, start + count, capture_at))
                    state["submitted"] += count
                    state["sequence"] += 1
                    state["max_pending"] = max(
                        state["max_pending"], state["submitted"] - state["consumed"]
                    )
                input_hash.update(data)
                send_started = time.monotonic()
                if state["first_source_send"] is None:
                    state["first_source_send"] = send_started
                state["last_source_send"] = send_started
                record(
                    "source_send",
                    start=start,
                    count=count,
                    event_id=event_id,
                    available_seconds=available_at - started,
                    capture_seconds=capture_at - started,
                    source_schedule_lag_seconds=(
                        max(
                            0,
                            available_at - (source_started + (start + count - 1) / 24),
                        )
                        if args.paced
                        else None
                    ),
                    credit_wait_seconds=credit_wait,
                )
                sent.append((start, start + count, send_started))
                await send(message)
            expected = getattr(args, "expected_frames", None)
            if prompt_updates:
                raise ValueError("Source ended before all scheduled prompt updates")
            if state["submitted"] == 0 or (
                expected is not None and state["submitted"] != expected
            ):
                raise ValueError("Source frame count does not match the requested case")
            state["ended"] = True
            state["source_end_send"] = time.monotonic()
            state["event_id"] += 1
            await send(
                dict(
                    type="event",
                    kind="source.end",
                    event_id=state["event_id"],
                    payload=dict(source_frame_end=state["submitted"]),
                )
            )
            record("source_end", frames=state["submitted"])

        async def receive(receive_ws, *, channel="combined"):
            next_source_id = 1
            profile_received = False
            async for wire in receive_ws:
                wire_received = time.monotonic()
                message = msgpack.unpackb(wire, raw=False)
                kind = message.get("type")
                if channel == "output" and kind not in (
                    "frame_batch",
                    "frame_batch_header",
                    "source.consumed",
                    "session.completed",
                    "error",
                ):
                    raise ValueError("Unexpected Decoder output message")
                if channel == "input" and kind in (
                    "source.consumed",
                    "frame_batch",
                    "frame_batch_header",
                    "session.completed",
                ):
                    raise ValueError("Encoder unexpectedly returned video output")
                if kind == "error":
                    if message.get("fatal") is False:
                        update = prompt_results.get(message.get("event_id"))
                        if update is not None and update["status"] == "pending":
                            update.update(
                                status="rejected", error_code=message.get("code")
                            )
                        record("error", message=message)
                        continue
                    raise RuntimeError(message.get("content", "Generation failed"))
                if kind == "source.accepted":
                    with credit.lock:
                        end = message["source_frame_end"]
                        if (
                            message["source_id"] != next_source_id
                            or message["client_seq"] != next_source_id - 1
                            or message["source_frame_start"] != state["accepted"]
                            or not state["accepted"] < end <= state["submitted"]
                            or (end, end * frame_bytes, message["event_id"])
                            not in boundaries
                        ):
                            raise ValueError("Invalid source acceptance")
                        next_source_id += 1
                        accepted.append(message)
                        state["accepted"] = end
                        retire_source_boundaries(state, boundaries)
                elif kind == "source.consumed":
                    with credit.lock:
                        consume_source_credit(
                            state, boundaries, message, from_decoder=channel == "output"
                        )
                    credit.notify()
                elif output_sink is not None and kind == "conditioning":
                    receipt = await output_sink.receive_conditioning(
                        message, received_file
                    )
                    update = prompt_results.get(message.get("event_id"))
                    if update is not None and update["status"] == "pending":
                        received = time.monotonic() - started
                        update.update(
                            status="encoded",
                            received_seconds=received,
                            latency_ms=(received - update["submitted_seconds"]) * 1000,
                        )
                        for older in list(prompt_results.values()):
                            if (
                                older["event_id"] < update["event_id"]
                                and older["status"] == "pending"
                            ):
                                older["status"] = "superseded"
                    record(kind, message=receipt)
                    continue
                elif output_sink is not None and kind == output_sink.message_type:
                    state["received"] = await output_sink.receive(
                        message, received_file, accepted_frames=state["accepted"]
                    )
                    if kind == "dit.latents":
                        record_applied_prompt(message)
                    while (
                        accepted
                        and accepted[0]["source_frame_end"] <= state["received"]
                    ):
                        accepted.popleft()
                    while sent and sent[0][1] <= state["received"]:
                        sent.popleft()
                    record(
                        kind,
                        message={k: v for k, v in message.items() if k != "payload"},
                    )
                    if args.consumer_delay:
                        await asyncio.sleep(args.consumer_delay)
                    continue
                elif kind in ("frame_batch", "frame_batch_header"):
                    if output_sink is not None:
                        raise ValueError(
                            "Unexpected RGB output from the encoding endpoint"
                        )
                    payload = (
                        await receive_ws.recv()
                        if kind == "frame_batch_header"
                        else message.pop("payload")
                    )
                    if kind == "frame_batch_header":
                        wire_received = time.monotonic()
                    now = time.monotonic()
                    start, count = message["output_frame_start"], message["num_frames"]
                    if (
                        start != state["received"]
                        or count < 1
                        or len(payload) != count * output_frame_bytes
                        or len(payload) != message["total_size"]
                        or message.get("encoding") not in (None, "raw")
                        or message.get("format") != "rgb24"
                        or message.get("channels") != 3
                        or (message.get("width"), message.get("height"))
                        != (args.width, args.height)
                        or message["pts_us"] != start * 1_000_000 // 24
                        or not start + count
                        <= message["source_dependency_end"]
                        <= state["submitted" if split_output else "accepted"]
                    ):
                        raise ValueError("Invalid delivered frame batch")
                    cursor = start
                    for span in message["source_alignment"]:
                        owner = next(
                            (
                                a
                                for a in (
                                    submitted_packets.copy()
                                    if split_output
                                    else accepted
                                )
                                if a["source_id"] == span["source_id"]
                            ),
                            None,
                        )
                        n = span["num_frames"]
                        if (
                            owner is None
                            or n < 1
                            or span["source_frame_start"] != cursor
                            or span["output_frame_start"] != cursor
                            or owner["source_frame_start"] > cursor
                            or cursor + n > owner["source_frame_end"]
                        ):
                            raise ValueError("Invalid delivered source alignment")
                        cursor += n
                    if cursor != start + count:
                        raise ValueError("Incomplete delivered source alignment")
                    dependency = message["source_dependency_end"]
                    owner = next(
                        (
                            entry
                            for entry in sent.copy()
                            if entry[0] < dependency <= entry[1]
                        ),
                        None,
                    )
                    if owner is None:
                        raise ValueError("Missing source dependency send timestamp")
                    dependency_latency = now - owner[2]
                    dependency_latency_samples.append(dependency_latency)
                    if state["first_output"] is None:
                        state["first_output"] = now
                        state["next_playback"] = now + args.startup_buffer_seconds
                    stall = max(0, now - state["next_playback"])
                    state["stall_seconds"] += stall
                    state["next_playback"] += stall + count / 24
                    for frame in range(start, start + count):
                        while capture and capture[0][1] <= frame:
                            capture.popleft()
                        if not capture or not capture[0][0] <= frame < capture[0][1]:
                            raise ValueError("Missing source availability timestamp")
                        capture_time = capture[0][2]
                        if args.paced:
                            capture_time += (frame - capture[0][0]) / 24
                        latency_samples.append(now - capture_time)
                    block_capture_latency = now - capture_time
                    block_capture_latency_samples.append(block_capture_latency)
                    state["last_output"] = now
                    record_applied_prompt(message)
                    state["received"] += count
                    delivery_times.append((now - started, count))
                    wire_delivery_times.append(
                        (wire_received - state["init_send"], count)
                    )
                    output_hash.update(payload)
                    record(
                        "frames",
                        header=message,
                        wire_received_seconds=wire_received - started,
                        init_wire_received_seconds=wire_received - state["init_send"],
                        received_seconds=now - started,
                        stall_seconds=stall,
                        dependency_to_output_seconds=dependency_latency,
                        block_capture_to_output_seconds=block_capture_latency,
                    )
                    written = (
                        await asyncio.to_thread(received_file.write, payload)
                        if received_file.mode == "raw"
                        else received_file.write(payload)
                    )
                    if written != len(payload):
                        raise OSError("Incomplete RGB output write")
                    state["sink_confirmed_bytes"] += written
                    # Keep any packet covering the next frame; a lookahead
                    # dependency may legitimately be reused by multiple outputs.
                    while sent and sent[0][1] <= state["received"]:
                        sent.popleft()
                    while accepted and accepted[0]["source_frame_end"] <= min(
                        state["received"], state["consumed"]
                    ):
                        accepted.popleft()
                    while (
                        submitted_packets
                        and submitted_packets[0]["source_frame_end"]
                        <= state["received"]
                    ):
                        submitted_packets.popleft()
                    if args.consumer_delay:
                        await asyncio.sleep(args.consumer_delay)
                elif kind == "profile_result":
                    if (
                        message.get("session_id") != ready["session_id"]
                        or not state["completed"]
                        or profile_received
                        or any(
                            row.get("request_id") != ready["session_id"]
                            for row in message.get("stages", [])
                        )
                    ):
                        raise ValueError(
                            "Stage profile does not match the completed session"
                        )
                    await asyncio.to_thread(
                        (output / "profile.json").write_text, json.dumps(message)
                    )
                    await send(dict(type="profile_ack", session_id=ready["session_id"]))
                    profile_received = True
                    record(
                        kind,
                        session_id=ready["session_id"],
                        status=message.get("status"),
                        stage_count=len(message.get("stages", [])),
                    )
                    continue
                elif kind == "session.completed":
                    if channel == "output" and any(
                        message.get(k) != ready[k]
                        for k in ("session_id", "generation_id")
                    ):
                        raise ValueError("Decoder completion identity does not match")
                    if output_sink is not None:
                        output_sink.complete(message, state)
                        state["completed"] = True
                        record(kind, message=message)
                        continue
                    if (
                        not state["ended"]
                        or any(
                            state[key] != state["submitted"]
                            for key in (
                                ("received",)
                                if split_output
                                else ("accepted", "consumed", "received")
                            )
                        )
                        or message["output_frames"] != state["received"]
                    ):
                        raise ValueError("The completed session is missing frames")
                    state["completed"] = True
                elif kind == "input.completed":
                    state["input_completed"] = True
                record(kind, message=message)
            if not state.get("input_completed" if channel == "input" else "completed"):
                raise RuntimeError("Connection closed before session completion")
            # A normal server close follows scheduler cleanup. Wait for it before
            # the next serial request; receipt of all RGB alone is not release.

        async def heartbeat():
            while True:
                await asyncio.sleep(5)
                if state.get("input_completed"):
                    return
                await send(dict(type="event", kind="heartbeat"))

        output_thread = None
        try:
            if split_output:
                from decoder_output_client import (
                    OutputReceiverThread,
                    receive_decoder_output,
                    validate_split_completion,
                )

                output_thread = OutputReceiverThread(
                    lambda: receive_decoder_output(
                        ready,
                        decoder_url,
                        websocket_options(
                            profile=args.profile,
                            timeout=args.timeout,
                            message_budget=message_budget,
                        ),
                        receive,
                    )
                )
            async with asyncio.TaskGroup() as group:
                h = group.create_task(heartbeat())
                producer = group.create_task(produce())
                receiver = group.create_task(
                    receive(ws, channel="input" if split_output else "combined")
                )
                if output_thread is not None:
                    group.create_task(output_thread.wait())
                await receiver
                await producer
                h.cancel()
            if split_output:
                validate_split_completion(state)
        except BaseException as exc:
            if output_thread is not None:
                await output_thread.close()
            # The cancelled receiver no longer drains the bounded incoming queue.
            # Abort a failed session rather than wait for a close handshake whose
            # reads may be paused; preserve the original error or cancellation.
            try:
                ws.transport.abort()
            except Exception as abort_error:
                exc.add_note(f"Transport abort failed: {type(abort_error).__name__}")
            try:
                (output / "failure-summary.json").write_text(
                    json.dumps(
                        {
                            **state,
                            "success": False,
                            "session_id": ready["session_id"],
                            "elapsed_seconds": time.monotonic() - started,
                            "timing_boundaries": timing,
                            "input_sha256": input_hash.hexdigest(),
                            "output_sha256": output_hash.hexdigest(),
                            "delivery_times": delivery_times,
                            "init_wire_delivery_times": wire_delivery_times,
                        },
                        indent=2,
                    )
                )
            except OSError as receipt_error:
                exc.add_note(
                    f"Failure receipt write failed: {type(receipt_error).__name__}"
                )
            raise
        finally:
            if output_thread is not None:
                await output_thread.close()
            for update in prompt_results.values():
                if update["status"] == "pending":
                    update["status"] = (
                        "not_observed" if state["completed"] else "interrupted"
                    )
            updates = list(prompt_results.values())
            prompt_metrics = dict(
                prompt_updates=updates,
                prompt_switch_ttff_ms=distribution(
                    [row["latency_ms"] for row in updates if row["status"] == "applied"]
                ),
                prompt_update_status_counts={
                    status: sum(row["status"] == status for row in updates)
                    for status in (
                        "applied",
                        "encoded",
                        "rejected",
                        "superseded",
                        "not_observed",
                        "interrupted",
                    )
                },
                prompt_update_timing_scope="client monotonic before prompt send to complete first matching frame-batch payload; includes serialization, writer waiting and upload; excludes image-file reads, browser decoding and rendering",
            )
            if output_sink is not None and output_sink.message_type == "dit.latents":
                prompt_metrics["prompt_activation_latency_ms"] = prompt_metrics.pop(
                    "prompt_switch_ttff_ms"
                )
                prompt_metrics["prompt_update_timing_scope"] = (
                    "Client prompt send to complete first generated latent using that event_id; no RGB decoding"
                )
            elif output_sink is not None:
                prompt_metrics.pop("prompt_switch_ttff_ms")
                prompt_metrics["prompt_encode_latency_ms"] = distribution(
                    [row["latency_ms"] for row in updates if row["status"] == "encoded"]
                )
                prompt_metrics["prompt_update_timing_scope"] = (
                    "Client prompt send to encoded condition receipt; not DiT activation"
                )
            (output / "prompt-updates.json").write_text(
                json.dumps(prompt_metrics, indent=2)
            )
        elapsed = time.monotonic() - started
        if output_sink is not None:
            summary = dict(
                completed=state["completed"],
                session_id=ready["session_id"],
                input_sha256=input_hash.hexdigest(),
                source_frames=state["submitted"],
                elapsed_seconds=elapsed,
                **output_sink.summary(),
            )
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            return summary
        source_latency = distribution(latency_samples)
        dependency_latency = distribution(dependency_latency_samples)
        block_latency = distribution(block_capture_latency_samples)
        metrics = calculate_metrics(
            [
                {"arrival_seconds": at, "num_frames": count}
                for at, count in delivery_times
            ]
        )
        init_metrics = calculate_metrics(
            [
                {"arrival_seconds": at, "num_frames": count}
                for at, count in wire_delivery_times
            ]
        )
        boot = Path("/proc/sys/kernel/random/boot_id")
        summary = dict(
            state,
            **prompt_metrics,
            schema_version=2,
            session_id=ready["session_id"],
            client_clock_domain=(
                f"CLOCK_MONOTONIC:{boot.read_text().strip()}" if boot.exists() else None
            ),
            output_batch_interval_p95_seconds=metrics["icl_seconds"]["p95"],
            output_batch_interval_max_seconds=metrics["icl_seconds"]["max"],
            elapsed_seconds=elapsed,
            effective_delivery_fps=state["received"] / elapsed,
            completion_overhead_seconds=elapsed - metrics["e2e_seconds"],
            init_send_to_rgb=init_metrics,
            timing_boundaries=timing,
            client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            **metrics,
            timing_origin="client_start_before_connect_and_source_decode",
            input_mode="paced" if args.paced else "unpaced",
            source_to_first_output_seconds=state["first_output"]
            - state["first_source_send"],
            tail_drain_seconds=state["last_output"] - state["last_source_send"],
            first_output_before_source_end=state["first_output"]
            < state["source_end_send"],
            dependency_to_output_seconds=dependency_latency,
            source_to_output_seconds=source_latency,
            ifl_seconds=None,
            ifl_status="not_measured_no_individual_frame_delivery_timestamps",
            source_to_output_p50_seconds=source_latency["median"],
            source_to_output_p95_seconds=source_latency["p95"],
            source_to_output_max_seconds=source_latency["max"],
            dependency_to_output_p50_seconds=dependency_latency["median"],
            dependency_to_output_p95_seconds=dependency_latency["p95"],
            block_capture_to_output_p50_seconds=block_latency["median"],
            block_capture_to_output_p95_seconds=block_latency["p95"],
            input_sha256=input_hash.hexdigest(),
            output_sha256=output_hash.hexdigest(),
            delivery_times=delivery_times,
            init_wire_delivery_times=wire_delivery_times,
        )
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(
            json.dumps(
                {k: v for k, v in summary.items() if k != "delivery_times"},
                indent=2,
            )
        )
        return summary


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="ws://127.0.0.1:30000/v1/realtime_video/generate"
    )
    parser.add_argument(
        "--decoder-url",
        help="Decoder output WebSocket URL when RGB uses a separate connection",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-rgb", help="RGB24 file or - for an engine pipe")
    source.add_argument(
        "--source-video",
        help="24 FPS video decoded incrementally; no looping or FPS conversion",
    )
    parser.add_argument(
        "--ffmpeg", default="ffmpeg", help="FFmpeg executable for --source-video"
    )
    parser.add_argument(
        "--expected-frames", type=int, help="Reject incomplete or extra source frames"
    )
    parser.add_argument("--output", required=True, help="New result directory")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--reference-id", action="append", default=[])
    parser.add_argument(
        "--prompt-updates",
        help="JSON list of prompt updates at source frame positions, with optional complete reference lists",
    )

    parser.add_argument("--max-message-bytes", type=int, default=DEFAULT_MESSAGE_BYTES)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--source-width", type=int, help="Source RGB width; defaults to output width"
    )
    parser.add_argument(
        "--source-height", type=int, help="Source RGB height; defaults to output height"
    )
    parser.add_argument(
        "--media-mode",
        choices=("metrics", "representative", "raw"),
        default="representative",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paced", action="store_true")
    parser.add_argument(
        "--cache-write",
        action="store_true",
        help="Allow this request to populate the server's enabled Block0 cache",
    )
    parser.add_argument("--receive-stage-timing", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Diagnostic run only; excludes this run from throughput acceptance",
    )
    parser.add_argument("--batch-pattern", type=int, nargs="+", default=[1])
    parser.add_argument("--startup-buffer-seconds", type=float, default=0.5)
    parser.add_argument("--consumer-delay", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=900)
    return parser


def main():
    cli = parser()
    args = cli.parse_args()
    if args.reference_id and len(args.reference_id) != len(args.reference):
        cli.error("Provide one reference ID per reference image.")
    if (
        min(args.width, args.height, *args.batch_pattern) < 1
        or args.startup_buffer_seconds < 0
    ):
        cli.error("Dimensions, batch lengths and startup buffering must be valid")
    try:
        source_dimensions(args)
    except ValueError as exc:
        cli.error(str(exc))
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
