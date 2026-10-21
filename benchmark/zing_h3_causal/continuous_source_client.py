#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Send video or an RGB pipe progressively over the production MessagePack API.

No total length is sent. EOF causes source.end, never a successful bare close.
Use a live engine pipe, or a documented, non-looping raw export of real media.
The receiver records local WebSocket delivery, not browser presentation or WAN
capture latency. File writing is included and should use local fast storage.
"""

import argparse
import asyncio
import hashlib
import json
import time
from collections import deque
from pathlib import Path

import msgpack
import websockets
from bench_client import calculate_metrics, distribution
from continuous_source_input import open_source
from continuous_source_media import load_prompt_updates, reference_payload

from sglang.multimodal_gen.runtime.realtime.source_transport import (
    DEFAULT_MESSAGE_BYTES,
    PROTOCOL_OVERHEAD_BYTES,
    SourceTransportBudget,
)
from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
    DEFAULT_MAX_REFERENCE_IMAGES,
)


def websocket_options(*, profile, timeout, message_budget=DEFAULT_MESSAGE_BYTES):
    return dict(
        max_size=message_budget,
        max_queue=1,
        compression=None,
        # Total request timeout remains bounded independently of keepalive.
        ping_timeout=timeout if profile else 20,
    )


async def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    async with asyncio.timeout(args.timeout):
        with (
            (output / "output.rgb").open("wb") as received_file,
            (output / "events.jsonl").open("w") as trace,
        ):
            async with open_source(
                args, args.width * args.height * 3, output / "source-decode.log"
            ) as read_source:
                return await exchange(
                    args, output, started, read_source, received_file, trace
                )


async def exchange(args, output, started, read_source, received_file, trace):
    prompt_updates = deque(load_prompt_updates(args.prompt_updates))
    message_budget = getattr(args, "max_message_bytes", DEFAULT_MESSAGE_BYTES)
    SourceTransportBudget(message_bytes=message_budget)
    frame_bytes = args.width * args.height * 3
    source_format = dict(format="rgb24", width=args.width, height=args.height, fps=24)
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
        credit_wait_seconds=0.0,
    )
    prompt_results = {}
    boundaries, accepted, capture, sent = deque(), deque(), deque(), deque()
    latency_samples, delivery_times = [], []
    dependency_latency_samples = []
    block_capture_latency_samples = []
    credit, writer = asyncio.Condition(), asyncio.Lock()

    def record(kind, **fields):
        trace.write(
            json.dumps(dict(kind=kind, seconds=time.monotonic() - started, **fields))
            + "\n"
        )
        trace.flush()

    async with websockets.connect(
        args.url,
        **websocket_options(
            profile=args.profile, timeout=args.timeout, message_budget=message_budget
        ),
    ) as ws:

        async def send(message):
            wire = msgpack.packb(message, use_bin_type=True)
            if len(wire) > message_budget:
                raise ValueError("Message exceeds the configured transport size limit")
            async with writer:
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
        if args.user_media:
            media = json.loads(Path(args.user_media).read_text())
            for item in media:
                item["payload"] = Path(item.pop("path")).read_bytes()
            init["user_media"] = media
        await send(init)
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
                async with credit:
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
                    boundaries.append(
                        (start + count, (start + count) * frame_bytes, event_id)
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

        async def receive():
            next_source_id = 1
            profile_received = False
            async for wire in ws:
                message = msgpack.unpackb(wire, raw=False)
                kind = message.get("type")
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
                elif kind == "source.consumed":
                    end, nbytes = (
                        message["source_frame_end"],
                        message["source_bytes_end"],
                    )
                    if not state["consumed"] <= end <= state["accepted"] or (
                        end,
                        nbytes,
                    ) not in [b[:2] for b in boundaries]:
                        raise ValueError("Invalid source credit watermark")
                    async with credit:
                        state["consumed"], state["consumed_bytes"] = end, nbytes
                        while boundaries and boundaries[0][0] <= end:
                            boundaries.popleft()
                        credit.notify_all()
                elif kind in ("frame_batch", "frame_batch_header"):
                    payload = (
                        await ws.recv()
                        if kind == "frame_batch_header"
                        else message.pop("payload")
                    )
                    now = time.monotonic()
                    start, count = message["output_frame_start"], message["num_frames"]
                    if (
                        start != state["received"]
                        or count < 1
                        or len(payload) != count * frame_bytes
                        or len(payload) != message["total_size"]
                        or message.get("encoding") not in (None, "raw")
                        or message.get("format") != "rgb24"
                        or message.get("channels") != 3
                        or (message.get("width"), message.get("height"))
                        != (args.width, args.height)
                        or message["pts_us"] != start * 1_000_000 // 24
                        or not start + count
                        <= message["source_dependency_end"]
                        <= state["accepted"]
                    ):
                        raise ValueError("Invalid delivered frame batch")
                    cursor = start
                    for span in message["source_alignment"]:
                        owner = next(
                            (
                                a
                                for a in accepted
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
                        (entry for entry in sent if entry[0] < dependency <= entry[1]),
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
                    update = prompt_results.get(message.get("event_id"))
                    if update is not None and update["status"] == "pending":
                        received = now - started
                        update.update(
                            status="applied",
                            received_seconds=received,
                            latency_ms=(received - update["submitted_seconds"]) * 1000,
                            output_frame_start=start,
                            pts_us=message["pts_us"],
                        )
                        # Only observed output of a newer epoch proves an older
                        # pending event was superseded; submission/ACK cannot.
                        for older in prompt_results.values():
                            if (
                                older["event_id"] < update["event_id"]
                                and older["status"] == "pending"
                            ):
                                older["status"] = "superseded"
                        record("prompt_applied", **update)
                    state["received"] += count
                    delivery_times.append((now - started, count))
                    output_hash.update(payload)
                    await asyncio.to_thread(received_file.write, payload)
                    record(
                        "frames",
                        header=message,
                        received_seconds=now - started,
                        stall_seconds=stall,
                        dependency_to_output_seconds=dependency_latency,
                        block_capture_to_output_seconds=block_capture_latency,
                    )
                    # Keep any packet covering the next frame; a lookahead
                    # dependency may legitimately be reused by multiple outputs.
                    while sent and sent[0][1] <= state["received"]:
                        sent.popleft()
                    while accepted and accepted[0]["source_frame_end"] <= min(
                        state["received"], state["consumed"]
                    ):
                        accepted.popleft()
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
                    if (
                        not state["ended"]
                        or any(
                            state[key] != state["submitted"]
                            for key in ("accepted", "consumed", "received")
                        )
                        or message["output_frames"] != state["received"]
                    ):
                        raise ValueError("The completed session is missing frames")
                    state["completed"] = True
                record(kind, message=message)
            if not state["completed"]:
                raise RuntimeError("Connection closed before session.completed")
            # A normal server close follows scheduler cleanup. Wait for it before
            # the next serial request; receipt of all RGB alone is not release.

        async def heartbeat():
            while True:
                await asyncio.sleep(5)
                await send(dict(type="event", kind="heartbeat"))

        try:
            async with asyncio.TaskGroup() as group:
                h = group.create_task(heartbeat())
                producer = group.create_task(produce())
                receiver = group.create_task(receive())
                await receiver
                await producer
                h.cancel()
        except BaseException:
            # The cancelled receiver no longer drains the bounded incoming queue.
            # Abort a failed session rather than wait for a close handshake whose
            # reads may be paused; preserve the original error or cancellation.
            ws.transport.abort()
            raise
        finally:
            for update in prompt_results.values():
                if update["status"] == "pending":
                    update["status"] = (
                        "not_observed" if state["completed"] else "interrupted"
                    )
            updates = list(prompt_results.values())
            prompt_metrics = dict(
                prompt_updates=updates,
                prompt_update_latency_ms=distribution(
                    [row["latency_ms"] for row in updates if row["status"] == "applied"]
                ),
                prompt_update_status_counts={
                    status: sum(row["status"] == status for row in updates)
                    for status in (
                        "applied",
                        "rejected",
                        "superseded",
                        "not_observed",
                        "interrupted",
                    )
                },
                prompt_update_timing_scope="client before send to complete first matching frame payload; excludes image-file reads and browser rendering",
            )
            (output / "prompt-updates.json").write_text(
                json.dumps(prompt_metrics, indent=2)
            )
        elapsed = time.monotonic() - started
        source_latency = distribution(latency_samples)
        dependency_latency = distribution(dependency_latency_samples)
        block_latency = distribution(block_capture_latency_samples)
        metrics = calculate_metrics(
            [
                {"arrival_seconds": at, "num_frames": count}
                for at, count in delivery_times
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
    parser.add_argument(
        "--user-media", help="JSON list of typed media with local path fields"
    )

    parser.add_argument("--max-message-bytes", type=int, default=DEFAULT_MESSAGE_BYTES)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paced", action="store_true")
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
    if args.user_media and args.reference:
        cli.error("Use either --user-media or --reference.")
    if args.reference_id and len(args.reference_id) != len(args.reference):
        cli.error("Provide one reference ID per reference image.")
    if (
        min(args.width, args.height, *args.batch_pattern) < 1
        or args.startup_buffer_seconds < 0
    ):
        cli.error("Dimensions, batch lengths and startup buffering must be valid")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
