# SPDX-License-Identifier: Apache-2.0
"""Single-active-session rejection, disconnect and next-session recovery.

Only target an isolated test instance. This test intentionally disconnects its
own session; it never sends restart/admin operations to the deployment.
"""

import argparse
import asyncio
import contextlib
import json
import time
from pathlib import Path

import continuous_source_client as client
import msgpack
import websockets
from bench_streaming_source import case_geometry, prepare_case
from case_sets import load_case_set
from frozen_rgb import bind_frozen_sources
from websockets.exceptions import ConnectionClosed


async def rejected_session(second, init):
    # Admission may reject and close immediately after the WebSocket handshake.
    # Still read the real queued error frame; a close alone is insufficient.
    with contextlib.suppress(ConnectionClosed):
        await second.send(init)
    reply = msgpack.unpackb(await asyncio.wait_for(second.recv(), 10), raw=False)
    await asyncio.wait_for(second.wait_closed(), 10)
    if (
        reply.get("type") != "error"
        or second.close_code != 1008
        or second.close_reason not in {"CAPACITY_EXHAUSTED", "USER_SESSION_LIMIT"}
    ):
        raise RuntimeError(
            "Second session did not return an explicit admission rejection"
        )
    return reply


def frame_count(path):
    if not path.exists():
        return 0
    return sum(
        1
        for line in path.read_text().split("\n")[:-1]
        if json.loads(line)["kind"] == "frames"
    )


async def wait_for_progress(task, path, previous=0):
    async with asyncio.timeout(900):
        while frame_count(path) <= previous:
            if task.done():
                await task
                raise RuntimeError(
                    "Active test session ended before progress checkpoint"
                )
            await asyncio.sleep(0.05)


async def check(case, root, url):
    def arguments(output):
        command = [
            "--media-mode",
            "metrics",
            "--url",
            url,
            "--source-rgb",
            case["source_rgb"],
            "--output",
            str(output),
            "--width",
            str(case["width"]),
            "--height",
            str(case["height"]),
            "--source-width",
            str(case["source_width"]),
            "--source-height",
            str(case["source_height"]),
            "--expected-frames",
            str(case["frames"]),
            "--prompt",
            case["prompt"],
            "--seed",
            "123",
            "--batch-pattern",
            "1",
            "--paced",
            "--timeout",
            "900",
        ]
        for path, identity in zip(
            case["references"], case["reference_ids"], strict=True
        ):
            command += ["--reference", path, "--reference-id", identity]
        return client.parser().parse_args(command)

    initial = {}
    original_pack = client.msgpack.packb

    def pack(message, *args, **kwargs):
        wire = original_pack(message, *args, **kwargs)
        if (
            isinstance(message, dict)
            and message.get("type") == "init"
            and "wire" not in initial
        ):
            initial["wire"] = wire
        return wire

    client.msgpack.packb = pack
    active = asyncio.create_task(client.run(arguments(root / "active")))
    events = root / "active/events.jsonl"
    receipt = dict(
        case_id=case["case_id"], input_sha256=case["frozen_source"]["sha256"]
    )
    try:
        await wait_for_progress(active, events)
        before = frame_count(events)
        async with websockets.connect(
            url, max_size=64 << 20, compression=None, open_timeout=10
        ) as second:
            reply = await rejected_session(second, initial["wire"])
            receipt["second_session_reply"] = reply
            receipt["second_close_code"] = second.close_code
            receipt["second_close_reason"] = second.close_reason
        await wait_for_progress(active, events, before)
        receipt["active_progress_after_rejection"] = frame_count(events) > before
        receipt["disconnect_monotonic_ns"] = time.monotonic_ns()
        active.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await active
        # One fixed grace interval, no retry loop to hide slow cleanup.
        await asyncio.sleep(2)
        receipt["recovery_started_monotonic_ns"] = time.monotonic_ns()
        result = await client.run(arguments(root / "recovery"))
        receipt.update(
            recovery_completed=result["completed"],
            recovery_frames=result["received"],
            recovery_ttff_seconds=result["ttff_seconds"],
            passed=result["completed"] and result["received"] == case["frames"],
        )
        if not receipt["passed"]:
            raise RuntimeError("Next session did not complete after disconnect")
        return receipt
    finally:
        client.msgpack.packb = original_pack
        if not active.done():
            active.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await active
        (root / "lifecycle.json").write_text(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="ws://127.0.0.1:30000/v1/realtime_video/generate"
    )
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--frozen-inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    member = load_case_set()["members"][0]
    prepared = prepare_case(
        member["case_id"],
        args.media_root,
        args.output_dir / "case",
        manifest=member["manifest"],
        case_id=member["case_id"],
    )
    for width, height in [(832, 480), (1280, 736)]:
        root = args.output_dir / f"{width}x{height}"
        root.mkdir()
        case = bind_frozen_sources(
            [
                case_geometry(
                    prepared,
                    width=width,
                    height=height,
                    source_width=width,
                    source_height=height,
                )
            ],
            args.frozen_inputs,
        )[0]
        asyncio.run(check(case, root, args.url))


if __name__ == "__main__":
    main()
