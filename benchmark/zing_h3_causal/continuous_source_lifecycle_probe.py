# SPDX-License-Identifier: Apache-2.0
"""Real source lifecycle probes; admission retries are recorded, never hidden."""

import argparse
import asyncio
import hashlib
import json
import pathlib
import time
from contextlib import asynccontextmanager, suppress

import msgpack
import websockets
from websockets.exceptions import ConnectionClosed

from continuous_source_media import reference_payload


CASES = [
    "invalid_sequence",
    "cancel_inflight",
    "reconnect_after_cancel",
    "cancel_end_race",
    "reconnect_after_end_race",
    "concurrent_rejection",
    "pause_prompt_resume",
    "invalid_after_output",
    "session_isolation",
    "reference",
]


def source_payload(pixels, stride, start, count, seq):
    return dict(
        client_seq=seq,
        source_frame_start=start,
        num_frames=count,
        pts_us=start * 1_000_000 // 24,
        data=pixels[start * stride : (start + count) * stride],
    )


@asynccontextmanager
async def keepalive(ws, interval=5):
    async def heartbeat():
        while True:
            await asyncio.sleep(interval)
            try:
                await ws.send(
                    msgpack.packb(
                        dict(type="event", kind="heartbeat"), use_bin_type=True
                    )
                )
            except ConnectionClosed:
                return

    task = asyncio.create_task(heartbeat())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def negotiate(ws, init, send, recv):
    """Admission may close the socket before the client can send init."""
    try:
        await send(ws, init)
        return await recv(ws)
    except ConnectionClosed as exc:
        if exc.rcvd is None or (exc.rcvd.code, exc.rcvd.reason) != (
            1008,
            "USER_SESSION_LIMIT",
        ):
            raise
        return dict(type="error", content="USER_SESSION_LIMIT", close_code=1008)


async def wait_for_scheduler_inflight(path, session_id, timeout=30):
    """Observe a real pending scheduler RPC; this is not a CUDA-start event."""
    deadline = time.monotonic() + timeout
    pending = {}
    with pathlib.Path(path).open() as stream:
        while time.monotonic() < deadline:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line or not line.endswith("\n"):
                    stream.seek(offset)
                    break
                if "realtime_trace " not in line:
                    continue
                event = json.loads(line.split("realtime_trace ", 1)[1])
                if event.get("session_id") != session_id:
                    continue
                request_id = event.get("request_id")
                if event.get("event") == "server.scheduler_forward_start":
                    pending[request_id] = event
                elif event.get("event") == "server.scheduler_forward_done":
                    pending.pop(request_id, None)
            if pending:
                return next(iter(pending.values()))
            await asyncio.sleep(0.01)
    raise TimeoutError("No in-flight scheduler request was observed for this session")


async def main(args):
    pixels = pathlib.Path(args.rgb).read_bytes()
    stride = 832 * 480 * 3
    assert len(pixels) >= 51 * stride
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    case_request = (
        json.loads(pathlib.Path(args.case_request).read_text())
        if args.case_request
        else None
    )
    case_content = case_request["messages"][0]["content"] if case_request else []
    case_prompt = next(
        (x["text"] for x in case_content if x["type"] == "text"),
        "Preserve the subjects and camera motion",
    )
    results = []
    attempts = []

    async def send(ws, obj):
        await ws.send(msgpack.packb(obj, use_bin_type=True))

    async def recv(ws):
        item = msgpack.unpackb(await asyncio.wait_for(ws.recv(), 180), raw=False)
        if item["type"] == "frame_batch_header":
            payload = await asyncio.wait_for(ws.recv(), 180)
            assert len(payload) == item["total_size"] == item["num_frames"] * stride
            item["sha256"] = hashlib.sha256(payload).hexdigest()
        return item

    async def connect(reference=False, user=None):
        started = time.monotonic()
        while True:
            ws = await websockets.connect(
                args.url + (f"?user_id={user}" if user else ""),
                max_size=32 << 20,
                max_queue=1,
                compression=None,
            )
            init = dict(
                type="init",
                zing_realtime_api_version=1,
                prompt=case_prompt,
                size="832x480",
                fps=24,
                seed=42,
                num_inference_steps=4,
                zing_realtime_decode_mode="stable",
                realtime_output_format="raw",
                source_stream=dict(format="rgb24", width=832, height=480, fps=24),
            )
            if case_request:
                from urllib.parse import urlparse, unquote

                init["references"] = [
                    reference_payload(unquote(urlparse(item["uri"]).path), item["id"])
                    for item in case_content
                    if item["type"] == "image"
                ]
            elif reference:
                init["references"] = [
                    reference_payload(args.reference, "source-first-frame")
                ]
            try:
                item = await negotiate(ws, init, send, recv)
            except Exception:
                await ws.close()
                raise
            attempts.append(dict(elapsed=time.monotonic() - started, response=item))
            if item["type"] == "session.ready":
                return ws, item["session_id"]
            await ws.close()
            assert item["type"] == "error" and "USER_SESSION_LIMIT" in item.get(
                "content", ""
            ), item
            assert time.monotonic() - started < 30, attempts
            await asyncio.sleep(0.1)

    async def append(ws, start=0, count=17, seq=0, event=1):
        await send(
            ws,
            dict(
                type="event",
                kind="source.append",
                event_id=event,
                payload=source_payload(pixels, stride, start, count, seq),
            ),
        )

    async def finish(ws, total, event):
        await send(
            ws,
            dict(
                type="event",
                kind="source.end",
                event_id=event,
                payload=dict(source_frame_end=total),
            ),
        )

    async def collect(ws, messages, until):
        while not until(messages):
            item = await recv(ws)
            messages.append(item)
            assert item["type"] != "error", item

    async def run_case(name):
        inflight_witness = None
        started = time.monotonic()
        ws, session = await connect(
            reference=name == "reference",
            user="isolation-a" if name == "session_isolation" else None,
        )
        messages = []
        async with ws, keepalive(ws):
            if name == "session_isolation":
                other, other_id = await connect(user="isolation-b")
                assert session != other_id
                other_messages = []
                async with other, keepalive(other):
                    await append(ws)
                    await append(other)
                    await collect(
                        ws,
                        messages,
                        lambda xs: any(x["type"] == "frame_batch_header" for x in xs),
                    )
                    await collect(
                        other,
                        other_messages,
                        lambda xs: any(x["type"] == "frame_batch_header" for x in xs),
                    )
                    await append(other, start=17, count=17, seq=1, event=2)
                    await finish(ws, 17, 2)
                    await finish(other, 34, 3)
                    await collect(
                        ws,
                        messages,
                        lambda xs: any(x["type"] == "session.completed" for x in xs),
                    )
                    await collect(
                        other,
                        other_messages,
                        lambda xs: any(x["type"] == "session.completed" for x in xs),
                    )
                    for own, total in [(messages, 17), (other_messages, 34)]:
                        assert (
                            sum(
                                x["num_frames"]
                                for x in own
                                if x["type"] == "frame_batch_header"
                            )
                            == total
                        )
                        assert own[-1]["output_frames"] == total
                    (out / "isolation-peer.json").write_text(
                        json.dumps(
                            dict(session_id=other_id, messages=other_messages), indent=2
                        )
                    )
            elif name == "invalid_after_output":
                await append(ws)
                await collect(
                    ws,
                    messages,
                    lambda xs: any(x["type"] == "frame_batch_header" for x in xs),
                )
                await append(ws, start=17, count=1, seq=3, event=2)
                while True:
                    item = await recv(ws)
                    messages.append(item)
                    if item["type"] == "error":
                        assert item["content"] == "Please send a valid streaming event."
                        break
            elif name == "invalid_sequence":
                await append(ws, seq=1)
                item = await recv(ws)
                messages.append(item)
                assert (
                    item["type"] == "error"
                    and item["content"] == "Please send a valid streaming event."
                ), item
            elif name in ("cancel_inflight", "cancel_end_race"):
                await append(ws)
                await collect(
                    ws,
                    messages,
                    lambda xs: any(x["type"] == "source.accepted" for x in xs),
                )
                if name == "cancel_end_race":
                    await finish(ws, 17, 2)
                if args.server_log:
                    inflight_witness = await wait_for_scheduler_inflight(
                        args.server_log, session
                    )
                else:
                    # Compatibility only: this delay is not an in-flight witness.
                    await asyncio.sleep(0.05)
                # Closing the transport cancels this generation; no resume token exists.
            else:
                if name == "concurrent_rejection":
                    async with websockets.connect(
                        args.url, max_size=32 << 20, compression=None
                    ) as other:
                        rejection = await recv(other)
                        assert rejection[
                            "type"
                        ] == "error" and "USER_SESSION_LIMIT" in rejection.get(
                            "content", ""
                        ), rejection
                await append(ws)
                if name == "pause_prompt_resume":
                    await collect(
                        ws,
                        messages,
                        lambda xs: any(x["type"] == "frame_batch_header" for x in xs),
                    )
                    pause_started = time.monotonic()
                    # Keep reading while input is paused: cached real output is
                    # allowed, but no frame may exceed the accepted source.
                    pending_read = asyncio.create_task(recv(ws))
                    while time.monotonic() - pause_started < 2:
                        done, _ = await asyncio.wait(
                            [pending_read],
                            timeout=max(0, 2 - (time.monotonic() - pause_started)),
                        )
                        if not done:
                            break
                        item = pending_read.result()
                        assert item["type"] not in ("error", "session.completed")
                        if item["type"] == "frame_batch_header":
                            assert item["output_frame_start"] + item["num_frames"] <= 17
                        messages.append(item)
                        pending_read = asyncio.create_task(recv(ws))
                    pending_read.cancel()
                    with suppress(asyncio.CancelledError):
                        await pending_read
                    await send(
                        ws,
                        dict(
                            type="event",
                            kind="prompt",
                            event_id=2,
                            payload=case_prompt + "\nUse slightly warmer colors.",
                        ),
                    )
                    await collect(
                        ws,
                        messages,
                        lambda xs: any(
                            x["type"] == "control_ack" and x.get("event_id") == 2
                            for x in xs
                        ),
                    )
                    await append(ws, start=17, count=17, seq=1, event=3)
                    await append(ws, start=34, count=17, seq=2, event=4)
                    await finish(ws, 51, 5)
                    assert time.monotonic() - pause_started >= 2
                    total = 51
                else:
                    total = 17
                    await finish(ws, total, 2)
                await collect(
                    ws,
                    messages,
                    lambda xs: any(x["type"] == "session.completed" for x in xs),
                )
                headers = [x for x in messages if x["type"] == "frame_batch_header"]
                assert sum(x["num_frames"] for x in headers) == total, messages
                assert messages[-1]["output_frames"] == total, messages[-1]
                if name == "pause_prompt_resume":
                    assert any(x.get("event_id") == 2 for x in headers), headers
                    assert headers[0].get("event_id") is None, headers[0]
        results.append(
            dict(
                case=name,
                passed=True,
                scheduler_inflight_witness=inflight_witness,
                session_id=session,
                elapsed_seconds=time.monotonic() - started,
                messages=messages,
            )
        )
        (out / "results.json").write_text(json.dumps(results, indent=2))
        (out / "admission-attempts.json").write_text(json.dumps(attempts, indent=2))

    try:
        for case in args.cases:
            await run_case(case)
    finally:
        (out / "admission-attempts.json").write_text(json.dumps(attempts, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--reference")
    parser.add_argument(
        "--server-log", help="Local server log required to prove in-flight cancellation"
    )
    parser.add_argument("--case-request")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--url", default="ws://127.0.0.1:30000/v1/realtime_video/generate"
    )
    args = parser.parse_args()
    if not args.reference and not args.case_request:
        parser.error("Provide --reference or --case-request")
    asyncio.run(main(args))
