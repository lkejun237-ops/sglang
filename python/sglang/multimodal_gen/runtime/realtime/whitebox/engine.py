# SPDX-License-Identifier: Apache-2.0
"""WebSocket transport for the forthcoming CLI, without assuming its wire schema."""

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from websockets.asyncio.client import connect

from .contracts import EngineFrame, VideoSpec


@dataclass(frozen=True)
class EngineEnd:
    frame_end: int


class EngineCodec(Protocol):
    def input(self, sequence: int, payload: dict[str, Any]) -> bytes | str: ...
    def finish(self) -> bytes | str: ...
    def receive(self, wire: bytes | str) -> EngineFrame | EngineEnd | None:
        """Translate the CLI protocol; None is a non-frame acknowledgement."""
        ...


@dataclass
class EngineEndpoint:
    url: str
    headers: dict[str, str] | None = None
    # Optional async cleanup for a CLI process/session launched by the resolver.
    cleanup: Any = None


class WebSocketEngineSession:
    def __init__(self, websocket, codec: EngineCodec, endpoint: EngineEndpoint):
        self.websocket, self.codec, self.endpoint = websocket, codec, endpoint
        self.finishing = self.closed = False
        self.writer = asyncio.Lock()

    async def frames(self):
        expected = 0
        # The CLI adapter must signal EOF explicitly. A network close is partial.
        while True:
            message = self.codec.receive(await self.websocket.recv())
            if isinstance(message, EngineEnd):
                if message.frame_end != expected:
                    raise ValueError("Engine EOF does not match delivered frames")
                return
            if isinstance(message, EngineFrame):
                if message.index != expected:
                    raise ValueError("Engine frames are not contiguous")
                expected += 1
                yield message

    async def send_input(self, sequence: int, payload: dict[str, Any]):
        async with self.writer:
            if not self.finishing and not self.closed:
                await self.websocket.send(self.codec.input(sequence, payload))

    async def finish(self):
        async with self.writer:
            if not self.finishing and not self.closed:
                self.finishing = True
                await self.websocket.send(self.codec.finish())

    async def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            await self.websocket.close()
        finally:
            if self.endpoint.cleanup is not None:
                await self.endpoint.cleanup()


class WebSocketEngineFactory:
    def __init__(
        self,
        resolver,
        codec_factory,
        *,
        max_message_bytes=16 * 1024 * 1024,
        connect_factory=connect,
    ):
        self.resolver, self.codec_factory = resolver, codec_factory
        self.max_message_bytes, self.connect = max_message_bytes, connect_factory

    async def __call__(self, *, world_id: str, spec: VideoSpec, trace_id: str):
        endpoint = await self.resolver(world_id=world_id, spec=spec, trace_id=trace_id)
        try:
            codec = self.codec_factory(spec)
            websocket = await self.connect(
                endpoint.url,
                additional_headers=endpoint.headers,
                max_size=self.max_message_bytes,
                max_queue=1,
                compression=None,
                open_timeout=15,
                close_timeout=2,
            )
            return WebSocketEngineSession(websocket, codec, endpoint)
        except BaseException:
            if endpoint.cleanup is not None:
                await endpoint.cleanup()
            raise
