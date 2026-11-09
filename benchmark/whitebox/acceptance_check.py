#!/usr/bin/env python3
"""End-to-end acceptance against the isolated acceptance_server.py only."""

import asyncio
import io
import json
import time
from uuid import uuid4

import av
import httpx
import msgspec
from PIL import Image
from websockets.asyncio.client import connect

BASE = "http://127.0.0.1:18087"


async def session(name, ready, results, *, finish_after=None):
    first = None
    count = 0
    applied = -1
    access = None
    async with connect(
        BASE.replace("http", "ws") + "/v1/realtime_video/generate?user_id=" + name,
        max_size=8 * 1024 * 1024,
    ) as ws:

        async def send(**message):
            await ws.send(msgspec.msgpack.encode(dict(version=2, **message)))

        await send(
            type="init",
            whitebox={"world_id": "acceptance-world"},
            prompt="Preserve world motion",
            references=[],
        )
        async for wire in ws:
            message = msgspec.msgpack.decode(wire)
            kind = message.get("type")
            if kind == "error":
                results[name] = {"error": message, "frames": count}
                ready.set()
                return
            if kind == "session.ready":
                access = (message["recording_id"], message["report_token"])
                results[name] = {"ready": True, "play_seconds": message["play_seconds"]}
            elif kind == "whitebox.control":
                applied = message["sequence"]
            elif kind == "whitebox.capture":
                image = Image.new(
                    "RGB",
                    (message["width"], message["height"]),
                    (message["index"] % 180, 100, 130),
                )
                file = io.BytesIO()
                image.save(file, format="JPEG")
                await send(
                    type="event",
                    kind="mock.frame",
                    payload=dict(
                        index=message["index"],
                        applied_sequence=applied,
                        data=file.getvalue(),
                    ),
                )
            elif kind == "frame_batch":
                assert (
                    message["encoding"] == "h264"
                    and message["output_frame_start"] == count
                )
                assert message["pts_us"] == count * 1000000 // 24
                count += 1
                if first is None:
                    first = time.monotonic()
                    ready.set()
                    await send(
                        type="event",
                        kind="input",
                        sequence=0,
                        payload={
                            "held_keys": ["KeyW"],
                            "events": [{"type": "keydown", "code": "KeyW"}],
                        },
                    )
                if count == 30:
                    await send(type="event", kind="prompt", payload="Changed prompt")
                if count % 24 == 0:
                    await send(type="event", kind="heartbeat", payload={})
                    await send(
                        type="client_metric",
                        stage="client_video_decode",
                        duration_ms=1.5,
                    )
                if (
                    finish_after is not None
                    and time.monotonic() - first >= finish_after
                ):
                    await send(type="event", kind="finish", payload={})
                    finish_after = None
            elif kind == "session.completed":
                assert count == message["frames"] and count > 1
                results[name].update(
                    frames=count, elapsed=time.monotonic() - first, access=access
                )
                return
    raise AssertionError("Socket ended before completed")


async def main():
    async with httpx.AsyncClient(timeout=20) as client:
        initial = (await client.get(BASE + "/v1/slots")).json()
        assert initial["total_slots"] == initial["free_slots"] == 2, initial
        results = {}
        names = ["acceptance-" + uuid4().hex for _ in range(3)]
        ready = [asyncio.Event() for _ in names]
        tasks = [
            asyncio.create_task(session(names[i], ready[i], results)) for i in range(2)
        ]
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in ready[:2])), 15
        )
        assert all(results[name].get("ready") for name in names[:2]), results
        for _ in range(30):
            full = (await client.get(BASE + "/v1/slots")).json()
            if full["used_slots"] == 2:
                break
            await asyncio.sleep(0.2)
        assert full["used_slots"] == 2 and full["free_slots"] == 0, full
        await session(names[2], ready[2], results)
        assert (
            results[names[2]]["error"].get("code")
            or results[names[2]]["error"].get("reason")
        ) == "CAPACITY_EXHAUSTED", results[names[2]]
        print(
            json.dumps({"two_active": full, "third_rejected": results[names[2]]}),
            flush=True,
        )
        await asyncio.sleep(
            18
        )  # exceeds the test lease TTL; renewal must preserve both slots
        renewed = (await client.get(BASE + "/v1/slots")).json()
        assert renewed["used_slots"] == 2, renewed
        print("Lease renewal passed beyond the 15-second TTL", flush=True)
        await asyncio.wait_for(asyncio.gather(*tasks), 60)
        for name in names[:2]:
            result = results[name]
            assert 59 <= result["elapsed"] <= 64, result
            recording, token = result.pop("access")
            url = BASE + "/v1/whitebox/recordings/" + recording
            assert (await client.get(url)).status_code == 404
            for _ in range(50):
                report = (
                    await client.get(url, headers={"Authorization": "Bearer " + token})
                ).json()
                if report["upload_status"] == "complete" and len(report["videos"]) == 3:
                    break
                await asyncio.sleep(0.2)
            assert (
                report["upload_status"] == "complete"
                and report["dataset_status"] == "excluded_mock"
            ), report
            assert len(report["videos"]) == 3 and len(report["timings"]) >= 10, report
            timelines = []
            for video in report["videos"]:
                ranged = await client.get(video["url"], headers={"Range": "bytes=0-99"})
                assert ranged.status_code == 206
                content = (await client.get(video["url"])).content
                with av.open(io.BytesIO(content)) as container:
                    assert container.streams.video[0].codec_context.name == "h264"
                    pts = [
                        frame.pts * frame.time_base
                        for frame in container.decode(video=0)
                    ]
                assert len(pts) == result["frames"]
                timelines.append(pts)
            assert timelines[0] == timelines[1] == timelines[2]
            result.update(
                recording_id=recording,
                trace_stages=len(report["timings"]),
                aligned=True,
            )
        await asyncio.sleep(3)
        released = (await client.get(BASE + "/v1/slots")).json()
        assert released["free_slots"] == 2, released
        followup = "acceptance-" + uuid4().hex
        await session(followup, asyncio.Event(), results, finish_after=1)
        assert results[followup]["frames"] > 1
        results[followup].pop("access")
        print(
            json.dumps(
                {
                    "sessions": results,
                    "released": released,
                    "store": "LOCAL MOCK; real OSS acceptance remains separate",
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
