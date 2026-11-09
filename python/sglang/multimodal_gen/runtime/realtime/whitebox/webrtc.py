# SPDX-License-Identifier: Apache-2.0
"""Optional lossy WebRTC media, reliable WebSocket signaling and commands.

Not an RTP track: complete self-describing Annex-B access units use an unordered
maxRetransmits=0 data channel. Bounded fragmentation preserves frame/state identity.
Every access unit must be an IDR. WS remains available as the comparison baseline.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import struct
import time
from uuid import uuid4

import msgspec

FRAGMENT_BYTES = 12 * 1024
MAX_FRAME_BYTES = 4 * 1024 * 1024
BUFFER_LIMIT = 512 * 1024
HEADER = struct.Struct("!4sIHHI")


def ice_configuration():
    servers = json.loads(os.getenv("WHITEBOX_RTC_ICE_SERVERS", "[]"))
    urls = os.getenv("WHITEBOX_RTC_TURN_URLS")
    secret = os.getenv("WHITEBOX_RTC_TURN_SECRET")
    if urls and secret:
        username = f"{int(time.time()) + 3600}:{uuid4().hex}"
        credential = base64.b64encode(
            hmac.new(secret.encode(), username.encode(), hashlib.sha1).digest()
        ).decode()
        servers.append(
            dict(urls=urls.split(","), username=username, credential=credential)
        )
    return servers


class WebRTCMedia:
    def __init__(self):
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection

        self.servers = ice_configuration()
        self.pc = RTCPeerConnection(
            RTCConfiguration(iceServers=[RTCIceServer(**s) for s in self.servers])
        )
        self.channel = None
        self.connected = asyncio.Event()
        self.sequence = 0
        self.dropped = 0

        @self.pc.on("datachannel")
        def channel_open(channel):
            if (
                self.channel is not None
                or channel.label != "whitebox-media-v1"
                or channel.ordered
                or channel.maxRetransmits != 0
            ):
                channel.close()
                return
            self.channel = channel
            if channel.readyState == "open":
                self.connected.set()

            @channel.on("open")
            def opened():
                self.connected.set()

            @channel.on("message")
            def unexpected(_):
                # Media channel is server -> browser only; controls are reliable.
                channel.close()

    async def answer(self, offer):
        from aiortc import RTCSessionDescription

        if (
            not isinstance(offer, str)
            or len(offer) > 128 * 1024
            or self.pc.signalingState != "stable"
        ):
            raise ValueError("Invalid WebRTC offer.")
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer, type="offer")
        )
        await self.pc.setLocalDescription(await self.pc.createAnswer())
        return self.pc.localDescription.sdp

    async def wait_connected(self):
        await asyncio.wait_for(self.connected.wait(), 20)

    async def send(self, message):
        channel = self.channel
        if channel is None or channel.readyState != "open":
            raise RuntimeError("WebRTC media is disconnected.")
        if not message.get("keyframe"):
            raise ValueError("Lossy transport requires independent H.264 access units.")
        wire = msgspec.msgpack.encode(message)
        if len(wire) > MAX_FRAME_BYTES:
            raise ValueError("Media frame exceeds the transport limit.")
        if channel.bufferedAmount + len(wire) > BUFFER_LIMIT:
            self.dropped += 1
            return False
        self.sequence += 1
        count = (len(wire) + FRAGMENT_BYTES - 1) // FRAGMENT_BYTES
        for index in range(count):
            channel.send(
                HEADER.pack(b"WBV1", self.sequence, index, count, len(wire))
                + wire[index * FRAGMENT_BYTES : (index + 1) * FRAGMENT_BYTES]
            )

        return True

    async def close(self):
        await self.pc.close()
