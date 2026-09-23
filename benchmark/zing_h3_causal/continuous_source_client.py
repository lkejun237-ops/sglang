#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay an RGB pipe progressively over the production MessagePack API.

No total length is sent. EOF causes source.end, never a successful bare close.
Use a live engine pipe, or a documented, non-looping raw export of real media.
The receiver records local WebSocket delivery, not browser presentation or WAN
capture latency. File writing is included and should use local fast storage.
"""

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections import deque
from pathlib import Path

import msgpack
import websockets
from continuous_source_media import reference_payload


async def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    frame_bytes = args.width * args.height * 3
    source_format = dict(format="rgb24", width=args.width, height=args.height, fps=24)
    source = sys.stdin.buffer if args.source_rgb == "-" else open(args.source_rgb, "rb")
    received_file = (output / "output.rgb").open("wb")
    trace = (output / "events.jsonl").open("w")
    input_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    started = time.monotonic()
    state = dict(
        submitted=0,
        accepted=0,
        consumed=0,
        consumed_bytes=0,
        received=0,
        sequence=0,
        event_id=0,
        ended=False,
        first_output=None,
        last_output=None,
        stall_seconds=0.0,
        next_playback=None,
        max_pending=0,
    )
    boundaries, accepted, capture = deque(), deque(), deque()
    latency_samples, delivery_times = [], []
    credit, writer = asyncio.Condition(), asyncio.Lock()

    def record(kind, **fields):
        trace.write(
            json.dumps(dict(kind=kind, seconds=time.monotonic() - started, **fields))
            + "\n"
        )
        trace.flush()

    async with websockets.connect(
        args.url, max_size=32 << 20, max_queue=1, compression=None
    ) as ws:

        async def send(message):
            wire = msgpack.packb(message, use_bin_type=True)
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
            num_inference_steps=4,
            zing_realtime_decode_mode="stable",
            realtime_output_format="raw",
            source_stream=source_format,
        )
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
            or ready.get("num_inference_steps") != 4
            or ready.get("realtime_output_format") != "raw"
        ):
            raise ValueError("Continuous source negotiation failed")
        record("ready", message=ready)
        (output / "request.json").write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "ready": ready,
                    "source_mode": "live-pipe"
                    if args.source_rgb == "-"
                    else "raw-replay",
                    "scope": "client WebSocket delivery; not browser presentation",
                },
                indent=2,
            )
        )
        if max(args.batch_pattern) > ready["max_source_batch_frames"]:
            raise ValueError("Batch pattern exceeds negotiated size")

        async def produce():
            source_started = time.monotonic()
            pattern_index = 0
            while True:
                count = args.batch_pattern[pattern_index % len(args.batch_pattern)]
                pattern_index += 1
                data = await asyncio.to_thread(source.read, count * frame_bytes)
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
                record(
                    "source_send",
                    start=start,
                    count=count,
                    event_id=event_id,
                    available_seconds=available_at - started,
                    capture_seconds=capture_at - started,
                    source_schedule_lag_seconds=max(
                        0, available_at - (source_started + (start + count - 1) / 24)
                    )
                    if args.paced
                    else None,
                    credit_wait_seconds=time.monotonic() - available_at,
                )
                await send(message)
            state["ended"] = True
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
            async for wire in ws:
                message = msgpack.unpackb(wire, raw=False)
                kind = message.get("type")
                if kind == "error":
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
                    state["last_output"] = now
                    state["received"] += count
                    delivery_times.append((now - started, count))
                    output_hash.update(payload)
                    await asyncio.to_thread(received_file.write, payload)
                    record(
                        "frames",
                        header=message,
                        received_seconds=now - started,
                        stall_seconds=stall,
                    )
                    while accepted and accepted[0]["source_frame_end"] <= min(
                        state["received"], state["consumed"]
                    ):
                        accepted.popleft()
                    if args.consumer_delay:
                        await asyncio.sleep(args.consumer_delay)
                elif kind == "session.completed":
                    if (
                        not state["ended"]
                        or state["received"] != state["submitted"]
                        or message["output_frames"] != state["received"]
                    ):
                        raise ValueError("The completed session is missing frames")
                    record(kind, message=message)
                    return
                record(kind, message=message)
            raise RuntimeError("Connection closed before session.completed")

        async def heartbeat():
            while True:
                await asyncio.sleep(5)
                await send(dict(type="event", kind="heartbeat"))

        try:
            async with asyncio.timeout(args.timeout):
                async with asyncio.TaskGroup() as group:
                    h = group.create_task(heartbeat())
                    producer, receiver = (
                        group.create_task(produce()),
                        group.create_task(receive()),
                    )
                    await receiver
                    await producer
                    h.cancel()
            elapsed = time.monotonic() - started
            samples = sorted(latency_samples)
            summary = dict(
                state,
                elapsed_seconds=elapsed,
                effective_delivery_fps=state["received"] / elapsed,
                ttff_seconds=state["first_output"] - started,
                source_to_output_p50_seconds=samples[len(samples) // 2],
                source_to_output_p95_seconds=samples[
                    min(len(samples) - 1, int(len(samples) * 0.95))
                ],
                source_to_output_max_seconds=max(samples),
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
        finally:
            if source is not sys.stdin.buffer:
                source.close()
            received_file.close()
            trace.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="ws://127.0.0.1:30000/v1/realtime_video/generate"
    )
    parser.add_argument(
        "--source-rgb", required=True, help="RGB24 file or - for an engine pipe"
    )
    parser.add_argument("--output", required=True, help="New result directory")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--reference-id", action="append", default=[])
    parser.add_argument(
        "--user-media", help="JSON list of typed media with local path fields"
    )

    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paced", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Diagnostic run only; excludes this run from throughput acceptance",
    )
    parser.add_argument("--batch-pattern", type=int, nargs="+", default=[1])
    parser.add_argument("--startup-buffer-seconds", type=float, default=0.5)
    parser.add_argument("--consumer-delay", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.user_media and args.reference:
        parser.error("Use either --user-media or --reference.")
    if args.reference_id and len(args.reference_id) != len(args.reference):
        parser.error("Provide one reference ID per reference image.")
    if (
        min(args.width, args.height, *args.batch_pattern) < 1
        or args.startup_buffer_seconds < 0
    ):
        parser.error("Dimensions, batch lengths and startup buffering must be valid")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
