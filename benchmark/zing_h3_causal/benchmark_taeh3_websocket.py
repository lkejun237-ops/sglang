# SPDX-License-Identifier: Apache-2.0
"""Exercise the real HTTP/WebSocket adapter against an already loaded scheduler.

Loopback transport includes RGB framing and receipt, but excludes WAN and browser
rendering. No preview resizing or output pacing is permitted in this benchmark.
"""

import asyncio
import contextlib
import copy
import json
import socket
import threading
import time


def run_websocket_benchmark(generator, request, output, frames):
    import msgspec.msgpack
    import uvicorn
    import websockets

    from sglang.multimodal_gen.runtime.entrypoints.http_server import create_app
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args

    set_global_server_args(generator.server_args)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(generator.server_args), log_level="warning")
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 60
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("WebSocket benchmark server did not start")
        time.sleep(0.1)
    report = {"scope": __doc__, "runs": []}

    async def session(label, count, cancel=False, update_prompt=False):
        req = copy.deepcopy(request)
        target = req["messages"][1]["content"][0]
        target["type"] = "video"
        target["output"]["frames"] = count
        width, height = target["output"]["width"], target["output"]["height"]
        init = {
            "type": "init",
            "prompt": req["messages"][0]["content"][0]["text"],
            "size": f"{width}x{height}",
            "num_frames": count,
            "fps": 24,
            "seed": 42,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "zing_request": req,
            "zing_solver": "few_step",
            "zing_denoising_step_list": [1000, 750, 500, 250],
            "zing_realtime_decode_mode": "stable",
            "realtime_output_format": "raw",
            "realtime_output_pacing": False,
            "realtime_causal_sink_size": 0,
            "realtime_causal_kv_cache_num_frames": -1,
        }
        rows, emitted, first, play_until, stalls = [], 0, None, None, 0.0
        prompt_sent = False
        prompt_sent_at = None
        started = time.perf_counter()
        async with websockets.connect(
            f"ws://127.0.0.1:{port}/v1/realtime_video/generate",
            max_size=None,
            open_timeout=60,
            ping_interval=20,
            ping_timeout=60,
        ) as ws:
            await ws.send(msgspec.msgpack.encode(init))

            async def heartbeat():
                while True:
                    await asyncio.sleep(5)
                    await ws.send(
                        msgspec.msgpack.encode(
                            {"type": "event", "kind": "heartbeat", "payload": None}
                        )
                    )

            keepalive = asyncio.create_task(heartbeat())
            pending = None
            try:
                while emitted < count:
                    packed = await asyncio.wait_for(ws.recv(), timeout=300)
                    if pending is None:
                        message = msgspec.msgpack.decode(packed)
                        if message.get("type") == "error":
                            raise RuntimeError(
                                message.get("content", "WebSocket generation failed")
                            )
                        if message.get("type") == "frame_batch_header":
                            pending = message
                            continue
                        if message.get("type") != "frame_batch":
                            continue
                        payload = message["payload"]
                    else:
                        message, pending, payload = pending, None, packed
                    n = message["num_frames"]
                    assert len(payload) == n * width * height * 3, (
                        message,
                        len(payload),
                    )
                    now = time.perf_counter() - started
                    if first is None:
                        first = play_until = now
                    stall = max(0.0, now - play_until)
                    stalls += stall
                    play_until = max(play_until, now) + n / 24
                    emitted += n
                    rows.append(
                        {
                            "chunk_index": message["chunk_index"],
                            "event_id": message.get("event_id"),
                            "frames": n,
                            "arrival_seconds": now,
                            "cumulative_frames": emitted,
                            "stall_seconds": stall,
                            "buffer_seconds": emitted / 24 - (now - first),
                        }
                    )
                    if update_prompt and not prompt_sent:
                        await ws.send(
                            msgspec.msgpack.encode(
                                {
                                    "type": "event",
                                    "kind": "prompt",
                                    "event_id": 123,
                                    "payload": init["prompt"]
                                    + " The camera slowly moves forward.",
                                }
                            )
                        )
                        prompt_sent = True
                        prompt_sent_at = time.perf_counter() - started
                    if cancel:
                        break
            finally:
                keepalive.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await keepalive
                (output / f"websocket-{label}-batches.json").write_text(
                    json.dumps(rows, indent=2)
                )
                print(
                    "BATCHES",
                    label,
                    json.dumps(
                        [
                            [r["chunk_index"], r["event_id"], round(r["arrival_seconds"], 3)]
                            for r in rows
                        ]
                    ),
                    flush=True,
                )
        if not cancel:
            assert emitted == count, (emitted, count)
        if update_prompt:
            assert any(row["event_id"] == 123 for row in rows), rows
            observed = [row["event_id"] == 123 for row in rows]
            assert observed == sorted(observed), "Prompt provenance moved backwards"
        duration = rows[-1]["arrival_seconds"]
        report["runs"].append(
            {
                "name": label,
                "frames": emitted,
                "requested_frames": count,
                "cancelled": cancel,
                "first_frame_seconds": first,
                "total_seconds": duration,
                "fps": emitted / duration,
                "playback_stall_seconds": stalls,
                "prompt_sent_seconds": prompt_sent_at,
            }
        )
        (output / "websocket-report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report["runs"][-1]), flush=True)
        # Disconnect cleanup can wait for the in-flight GPU block. This delay
        # is outside every measured request and is not hidden in latency.
        await asyncio.sleep(5)

    async def run():
        await session("cold_10s", 240)
        await session("warm_sustained", frames)
        await session("cancel", 240, cancel=True)
        await session("after_cancel", 22)
        await session("prompt_update", 85, update_prompt=True)

    try:
        asyncio.run(run())
    finally:
        server.should_exit = True
        thread.join(timeout=60)
        listener.close()
