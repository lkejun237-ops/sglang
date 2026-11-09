# SPDX-License-Identifier: Apache-2.0
"""SDK boundary failures must not silently corrupt three-track alignment."""

import asyncio
import copy
import json
import threading

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox.cli_compositor import (
    StreamingCompositor,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import (
    EngineFrame,
    VideoSpec,
)


def setup():
    spec = VideoSpec(160, 96)
    source = {"sessionId": "source", "epoch": 2}
    renderer = StreamingCompositor(source, spec)
    header = {
        "source": {"sourceFrameId": 17, "sourceTimeUs": 900000},
        "uiRevision": 3,
        "uiCompleteThroughUs": 900000,
        "outputFrameId": 1,
        "outputPtsUs": 0,
    }
    frame = EngineFrame(
        0, 0, bytes(spec.frame_bytes), {}, provenance={"cli_frame_header": header}
    )
    renderer.configured.set()
    renderer.encode = lambda rgb, index: b"\0VP8"
    renderer.decode = lambda payload: bytes(spec.frame_bytes)
    return renderer, frame, spec


@pytest.mark.parametrize(
    "field",
    ["source", "uiRevision", "uiCompleteThroughUs", "outputFrameId", "outputPtsUs"],
)
def test_rejects_misaligned_composition(field):
    async def run():
        renderer, frame, spec = setup()

        class Peer:
            async def send(self, wire):
                header = copy.deepcopy(frame.provenance["cli_frame_header"])
                header[field] = {} if field == "source" else 999
                renderer.output.put_nowait((header, b"output"))

        renderer.upstream["video"] = Peer()
        with pytest.raises(ValueError, match="does not match"):
            await renderer.render(frame.rgb, frame, spec)
        assert renderer.count == 0
        await renderer.close()

    asyncio.run(run())


def test_compositor_failure_interrupts_pending_render():
    async def run():
        renderer, frame, spec = setup()

        class Peer:
            async def send(self, wire):
                renderer.fail(ValueError("COMPOSE_PLAYER_FAILED"))

        renderer.upstream["video"] = Peer()
        with pytest.raises(ValueError, match="compositor failed"):
            await renderer.render(frame.rgb, frame, spec)
        await renderer.close()
        await renderer.close()

    asyncio.run(run())


def test_eof_requires_acknowledgement_of_exact_last_frame():
    async def run():
        renderer, _, _ = setup()
        renderer.count = 3
        renderer.composition = {"compositionId": "composition"}
        renderer.finishing = True

        class Control:
            async def __aiter__(self):
                yield json.dumps(
                    {
                        "type": "media.ended",
                        "compositionId": "composition",
                        "epoch": 2,
                        "lastOutputFrameId": 2,
                    }
                )

        renderer.control = Control()
        await renderer.read_control()
        assert renderer.failure.is_set()
        assert not renderer.ended.is_set()

    asyncio.run(run())


def test_cancelled_codec_work_finishes_before_contexts_are_released():
    async def run():
        renderer = StreamingCompositor(
            {"sessionId": "source", "epoch": 2}, VideoSpec(160, 96)
        )
        started, release = threading.Event(), threading.Event()
        context = object()
        renderer.encoder = context

        def encode(rgb, index):
            started.set()
            assert release.wait(5)
            assert renderer.encoder is context
            return b"encoded"

        renderer._encode = encode
        pending = asyncio.create_task(asyncio.to_thread(renderer.encode, b"", 0))
        assert await asyncio.to_thread(started.wait, 5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        closing = asyncio.create_task(renderer.close())
        await asyncio.sleep(0.02)
        try:
            assert renderer.closed
            assert not closing.done()
        finally:
            release.set()
            await closing
        assert renderer.encoder is None
        with pytest.raises(ValueError, match="closed"):
            await asyncio.to_thread(renderer.encode, b"", 1)

    asyncio.run(run())
