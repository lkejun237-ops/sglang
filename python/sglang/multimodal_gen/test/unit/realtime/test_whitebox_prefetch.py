import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import msgspec
from fastapi import FastAPI

from sglang.multimodal_gen.runtime.realtime.whitebox.prefetch import install_prefetch
from sglang.multimodal_gen.runtime.realtime.whitebox.room_proxy import RoomProxy


def test_prefetch_reuses_gateway_boundary_without_allocating_game_or_model(monkeypatch):
    async def run():
        monkeypatch.setenv("WHITEBOX_PREFETCH_ENABLED", "1")
        directory = AsyncMock()
        directory.resolve.side_effect = AssertionError(
            "A prefetch must not allocate a game"
        )
        connect = AsyncMock(
            side_effect=AssertionError("A prefetch must not connect a model")
        )
        messages = []

        class Socket:
            received = False

            async def accept(self):
                pass

            async def receive_bytes(self):
                if self.received:
                    await asyncio.Event().wait()
                self.received = True
                return msgspec.msgpack.encode(
                    {
                        "type": "world.prepare",
                        "whitebox": {"source_id": "s", "world_id": "w"},
                    }
                )

            async def send_bytes(self, value):
                messages.append(msgspec.msgpack.decode(value))

            async def close(self, **_):
                pass

        def transport(request):
            assert request.url == "http://room/internal/worlds/prepare"
            assert request.headers["authorization"] == "Bearer internal-test"
            assert json.loads(request.content) == {"world_id": "s/w"}
            return httpx.Response(
                200,
                text='{"type":"world.preparation","state":"preparing"}\n{"type":"world.preparation","state":"ready"}\n',
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(transport)
        ) as client:
            proxy = RoomProxy(
                directory, {"single": "http://room"}, "internal-test", client, connect
            )
            await proxy.serve(Socket(), user_id="test", trace_id="trace")
        assert [x["state"] for x in messages] == ["preparing", "ready"]
        directory.resolve.assert_not_called()
        connect.assert_not_called()

    asyncio.run(run())


def test_internal_prefetch_requires_token_and_explicit_capability(monkeypatch):
    async def run():
        app = FastAPI()
        install_prefetch(app, "internal-test")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://room"
        ) as client:
            bad = await client.post(
                "/internal/worlds/prepare", json={"world_id": "s/w"}
            )
            assert bad.status_code in {401, 403}
            monkeypatch.delenv("WHITEBOX_PREFETCH_ENABLED", raising=False)
            disabled = await client.post(
                "/internal/worlds/prepare",
                json={"world_id": "s/w"},
                headers={"Authorization": "Bearer internal-test"},
            )
            assert disabled.status_code == 404

    asyncio.run(run())


def test_resource_preparation_is_bounded_and_releases_on_completion(
    tmp_path, monkeypatch
):
    async def run():
        monkeypatch.setenv("WHITEBOX_PREFETCH_ENABLED", "1")
        monkeypatch.setenv("WHITEBOX_CLI_RESOURCE_TRANSPORT", "cdn")
        entry = tmp_path / "sdk/packages/cli/src/main.ts"
        entry.parent.mkdir(parents=True)
        entry.write_text("")
        monkeypatch.setenv("WHITEBOX_CLI_ENTRYPOINT", str(entry))
        monkeypatch.setenv("WHITEBOX_CLI_CACHE_DIRECTORY", str(tmp_path / "cache"))
        started, finish = asyncio.Event(), asyncio.Event()
        spawned = []

        class Input:
            def write(self, value):
                assert json.loads(value) == {"worldId": "s/w"}

            async def drain(self):
                pass

            def close(self):
                pass

        class Process:
            returncode = 0
            stdin = Input()

            @property
            def stdout(self):
                async def lines():
                    started.set()
                    await finish.wait()
                    yield b'{"type":"world.preparation","state":"ready"}\n'

                return lines()

            async def wait(self):
                return 0

        async def spawn(*args, **kwargs):
            spawned.append(args)
            assert args[3].endswith("worldkit_cdn_prepare.mjs")
            return Process()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        app = FastAPI()
        install_prefetch(app, "internal-test")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://room"
        ) as client:
            headers = {"Authorization": "Bearer internal-test"}
            first = asyncio.create_task(
                client.post(
                    "/internal/worlds/prepare",
                    json={"world_id": "s/w"},
                    headers=headers,
                )
            )
            await started.wait()
            other = await client.post(
                "/internal/worlds/prepare", json={"world_id": "s/w"}, headers=headers
            )
            assert other.json()["code"] == "PREPARATION_BUSY"
            assert len(spawned) == 1
            finish.set()
            assert (await first).json()["state"] == "ready"
            assert (
                await client.post(
                    "/internal/worlds/prepare",
                    json={"world_id": "s/w"},
                    headers=headers,
                )
            ).json()["state"] == "ready"
            assert len(spawned) == 2

    asyncio.run(run())


def test_gateway_disconnect_cancels_resource_request_without_a_lease(monkeypatch):
    async def run():
        monkeypatch.setenv("WHITEBOX_PREFETCH_ENABLED", "1")
        started, closed = asyncio.Event(), asyncio.Event()
        directory, connect = AsyncMock(), AsyncMock()

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                started.set()
                await asyncio.Event().wait()
                yield b""

            async def aclose(self):
                closed.set()

        class Socket:
            received = False

            async def accept(self):
                pass

            async def receive_bytes(self):
                if self.received:
                    await started.wait()
                    raise ConnectionError("closed")
                self.received = True
                return msgspec.msgpack.encode(
                    {
                        "type": "world.prepare",
                        "whitebox": {"source_id": "s", "world_id": "w"},
                    }
                )

            async def send_bytes(self, value):
                raise AssertionError("No ready before preparation")

            async def close(self, **kwargs):
                pass

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, stream=Stream())
            )
        ) as client:
            proxy = RoomProxy(
                directory, {"single": "http://room"}, "internal-test", client, connect
            )
            await asyncio.wait_for(
                proxy.serve(Socket(), user_id="test", trace_id="trace"), 2
            )
        assert closed.is_set()
        directory.resolve.assert_not_called()
        connect.assert_not_called()

    asyncio.run(run())


def test_room_cancellation_terminates_worker_and_releases_preparation_lock(
    tmp_path, monkeypatch
):
    async def run():
        monkeypatch.setenv("WHITEBOX_PREFETCH_ENABLED", "1")
        monkeypatch.setenv("WHITEBOX_CLI_RESOURCE_TRANSPORT", "cdn")
        entry = tmp_path / "sdk/packages/cli/src/main.ts"
        entry.parent.mkdir(parents=True)
        entry.write_text("")
        monkeypatch.setenv("WHITEBOX_CLI_ENTRYPOINT", str(entry))
        monkeypatch.setenv("WHITEBOX_CLI_CACHE_DIRECTORY", str(tmp_path / "cache"))
        started = asyncio.Event()
        spawned = []
        signals = []

        class Input:
            def write(self, value):
                pass

            async def drain(self):
                pass

            def close(self):
                pass

        class Process:
            pid = 12345
            returncode = None
            stdin = Input()

            @property
            def stdout(self):
                async def lines():
                    started.set()
                    await asyncio.Event().wait()
                    yield b""

                return lines()

            async def wait(self):
                self.returncode = -15
                return -15

        async def spawn(*args, **kwargs):
            p = Process()
            spawned.append(p)
            return p

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        monkeypatch.setattr("os.killpg", lambda pid, sig: signals.append((pid, sig)))
        app = FastAPI()
        install_prefetch(app, "internal-test")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://room"
        ) as client:
            for _ in range(2):
                started.clear()
                request = asyncio.create_task(
                    client.post(
                        "/internal/worlds/prepare",
                        json={"world_id": "s/w"},
                        headers={"Authorization": "Bearer internal-test"},
                    )
                )
                await asyncio.wait_for(started.wait(), 1)
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)
        assert len(spawned) == 2 and len(signals) == 2
        assert all(p.returncode is not None for p in spawned)

    asyncio.run(run())
