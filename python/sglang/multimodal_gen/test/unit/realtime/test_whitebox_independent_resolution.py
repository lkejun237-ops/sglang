"""Real Room codecs with a mock model: source bytes and output canvases differ."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS

import av
import msgspec
from test_whitebox_interactive import BrowserUI, Engine, Worker

from sglang.multimodal_gen.runtime.realtime.whitebox.async_recording import (
    AsyncArchiveManager,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec
from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
    RenderJournal,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.render_source import SourceSession
from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import MemoryRoomStore
from sglang.multimodal_gen.runtime.realtime.whitebox.session import WhiteboxConfig


def test_room_accepts_small_source_receipts_and_emits_full_model_rgb(tmp_path):
    async def run():
        small, large = VideoSpec(32, 32), VideoSpec(64, 64)
        engine = Engine(small)
        engine.stream_id = "view"
        workers = []

        class MixedWorker(Worker):
            def __init__(self, spec):
                super().__init__(spec)
                self.prompt_id = None

            async def emit(self, message):
                if message["type"] == "session.ready":
                    message["source_prompt_protocol"] = "ordered-boundary-v2"
                if message["type"] == "frame_batch":
                    message.update(
                        width=large.width,
                        height=large.height,
                        payload=bytes([117]) * large.frame_bytes,
                    )
                await super().emit(message)

        class Leases:
            active = 0

            @asynccontextmanager
            async def open(self):
                self.active += 1
                worker = MixedWorker(small)
                workers.append(worker)
                try:
                    yield (
                        worker,
                        NS(
                            session_id="model",
                            generation_id="generation",
                            denoiser=NS(worker_id="exclusive", model_revision="test"),
                        ),
                    )
                finally:
                    await worker.close()
                    self.active -= 1

        leases = Leases()
        journal = RenderJournal(
            MemoryRoomStore(),
            user_id="user",
            source_session_id="source",
            source_epoch="epoch",
            owner_epoch="room",
        )
        await journal.create()
        archives = AsyncArchiveManager(recorder_slots=3)

        async def factory(**_):
            return engine

        source = SourceSession(
            NS(runtime=NS(players={}), make_segment_leases=lambda *_: leases),
            journal,
            source_epoch="epoch",
            user_id="user",
            session_id="source",
            trace_id="mixed",
            config=WhiteboxConfig(
                engine_factory=factory,
                ui_factory=lambda **_: BrowserUI(),
                spool=tmp_path,
                spec=small,
                model_target=large,
                room_owned=True,
                capture_clock=True,
                startup_seconds=5,
                drain_seconds=3,
            ),
            archives=archives,
        )
        messages = []

        async def consume():
            while True:
                m = msgspec.msgpack.decode(await source.recv())
                messages.append(m)
                if m.get("type") in {"whitebox.frame", "frame_batch"}:
                    source.media_delivered()

        async def produce():
            while True:
                await engine.push()
                await asyncio.sleep(0.025)

        async def wait(predicate):
            async with asyncio.timeout(8):
                while not predicate():
                    await asyncio.sleep(0.01)

        tasks = []
        try:
            await source.send(
                msgspec.msgpack.encode(
                    dict(type="init", whitebox={"world_id": "world"}, prompt="test")
                )
            )
            tasks = [asyncio.create_task(consume()), asyncio.create_task(produce())]
            await wait(lambda: source.ready.is_set())
            await source.send(
                msgspec.msgpack.encode(
                    dict(
                        type="command",
                        op="render.start",
                        request_id="r1",
                        source_session_id="source",
                        source_epoch="epoch",
                        expected_render_revision=0,
                        prompt="test",
                        seed=42,
                        references=[],
                    )
                )
            )
            await wait(
                lambda: sum(m.get("type") == "frame_batch" for m in messages) >= 4
            )
            assert workers[0].init["size"] == "64x64"
            assert workers[0].init["source_stream"] == dict(
                format="rgb24", width=32, height=32, fps=24
            )
            sent = [m for m in workers[0].events if m.get("kind") == "source.append"]
            assert sent and all(
                len(m["payload"]["data"]) == small.frame_bytes for m in sent
            )
            assert all(
                (m["width"], m["height"]) == (32, 32)
                for m in messages
                if m.get("type") == "whitebox.frame"
            )
            output = [m for m in messages if m.get("type") == "frame_batch"]
            assert all((m["width"], m["height"]) == (64, 64) for m in output)
            decoder = av.CodecContext.create("h264", "r")
            decoded = [
                f for m in output for f in decoder.decode(av.Packet(m["payload"]))
            ]
            assert decoded and all((f.width, f.height) == (64, 64) for f in decoded)
            assert any(
                m.get("type") == "render.stream"
                and m.get("video_stream", {}).get("width") == 64
                for m in messages
            )
        finally:
            await source.close()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await archives.close()
        assert leases.active == 0

    asyncio.run(run())
