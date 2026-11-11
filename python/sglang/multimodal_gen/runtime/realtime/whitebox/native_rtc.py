# SPDX-License-Identifier: Apache-2.0
"""Bounded IPC to an optional native RTP process; never used by WS sessions."""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
import os
import re
import struct

HEADER = struct.Struct("!II")
MAX_AU = 1 << 20
_endpoint_leases = {}


def configured_rtc_endpoints():
    """Optional fixed listen/public port pairs, one native process per source."""
    value = os.environ.get("WHITEBOX_NATIVE_RTC_ENDPOINTS")
    if value is None:
        return ()
    try:
        rows = json.loads(value)
        if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
            raise ValueError()
        result, listen_ports, public_ports = [], set(), set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"listen", "public_port"}:
                raise ValueError()
            listen, public = row["listen"], row["public_port"]
            if not isinstance(listen, str) or type(public) is not int:
                raise ValueError()
            host, port = listen.rsplit(":", 1)
            ipaddress.IPv4Address(host)
            port = int(port)
            if not 1 <= port <= 65535 or not 1 <= public <= 65535:
                raise ValueError()
            if port in listen_ports or public in public_ports:
                raise ValueError()
            listen_ports.add(port)
            public_ports.add(public)
            result.append((listen, public, port))
        return tuple(result)
    except (ValueError, TypeError, KeyError):
        raise ValueError("Invalid native RTC endpoint configuration") from None


def validate_rtc_source_capacity(capacity):
    endpoints = configured_rtc_endpoints()
    if type(capacity) is not int or capacity < 1 or capacity > (len(endpoints) or 1):
        raise ValueError("Each RTC source requires a dedicated configured endpoint")


class NativeRTC:
    def __init__(
        self,
        on_event,
        *,
        executable=None,
        listen=None,
        public_ip=None,
        public_port=None,
        source_codec=None,
        source_h264_profile=None,
    ):
        self.on_event = on_event
        if source_codec is not None and source_codec not in {"h264", "vp9", "av1"}:
            raise ValueError("Unsupported source codec")
        if source_h264_profile is not None:
            import re

            if source_codec != "h264" or not re.fullmatch(
                "[a-f0-9]{6}", source_h264_profile
            ):
                raise ValueError("Invalid source H264 profile")
        self.source_codec = source_codec
        self.source_h264_profile = source_h264_profile
        self.executable = executable or os.environ.get(
            "WHITEBOX_NATIVE_RTC_BINARY", "/app/whitebox-rtc"
        )
        self.listen = listen or os.environ.get(
            "WHITEBOX_NATIVE_RTC_LISTEN", "0.0.0.0:40000"
        )
        self.public_ip = (
            public_ip
            if public_ip is not None
            else os.environ.get("WHITEBOX_NATIVE_RTC_PUBLIC_IP", "")
        )
        self.public_port = (
            public_port
            if public_port is not None
            else int(os.environ.get("WHITEBOX_NATIVE_RTC_PUBLIC_PORT", "443"))
        )
        self.process = self.reader = self.stderr = None
        self.pending = {}
        self.frames = {}
        self.sequence = 0
        self.closed = False
        self.ready = asyncio.Event()
        self.connected = asyncio.Event()
        self.write_lock = asyncio.Lock()
        self.error = None
        self.delivery_stats_version = 0
        self.close_task = None
        self.started = False
        self.endpoint_lease = None
        self.endpoint_candidates = (
            configured_rtc_endpoints() if listen is None and public_port is None else ()
        )

    async def start(self):
        if self.started or self.closed:
            raise RuntimeError("The media endpoint cannot be started twice")
        self.started = True
        if self.endpoint_candidates:
            for listen, public, port in self.endpoint_candidates:
                if port not in _endpoint_leases:
                    marker = object()
                    _endpoint_leases[port] = marker
                    self.endpoint_lease = (port, marker)
                    self.listen, self.public_port = listen, public
                    break
            else:
                raise RuntimeError("No native RTC endpoint is available")
        try:
            selected_args = []
            if self.source_codec is not None:
                selected_args = ["--source-codec", self.source_codec]
                if self.source_h264_profile is not None:
                    selected_args += ["--source-h264-profile", self.source_h264_profile]
            self.process = await asyncio.create_subprocess_exec(
                self.executable,
                "--listen",
                self.listen,
                "--public-ip",
                self.public_ip,
                "--public-port",
                str(self.public_port),
                *selected_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            self.reader = asyncio.create_task(self._read(), name="rtc-native-events")
            self.stderr = asyncio.create_task(
                self._stderr(), name="rtc-native-diagnostics"
            )
            await asyncio.wait_for(self.ready.wait(), 5)
            self._check()
        except BaseException:
            await self.close()
            raise

    def _check(self):
        if self.closed or self.error:
            raise RuntimeError("The media endpoint is unavailable") from self.error

    async def _stderr(self):
        async for line in self.process.stderr:
            text = line[:4096].decode(errors="replace").rstrip()
            logger = logging.getLogger(__name__)
            if text.startswith("WHITEBOX_RTC_CC "):
                logging.getLogger("uvicorn.error").info("%s", text)
            elif text.startswith("RTC command "):
                logging.getLogger("uvicorn.error").warning("Native RTC: %s", text[:512])
            else:
                logger.debug("Native RTC: %s", text)

    async def _read(self):
        try:
            while True:
                size, payload_size = HEADER.unpack(
                    await self.process.stdout.readexactly(8)
                )
                if not 0 < size <= 256 << 10 or payload_size:
                    raise ValueError("Invalid native media event")
                message = json.loads(await self.process.stdout.readexactly(size))
                kind, request_id = message.get("type"), message.get("id")
                if kind == "ready":
                    self.ready.set()
                elif kind == "state" and message.get("state") == "connected":
                    self.connected.set()
                if kind == "mapped":
                    frame = self.frames.get(request_id)
                    if frame is not None:
                        self.on_event(
                            {
                                "type": "frame.meta",
                                "track": message["track"],
                                "rtp_timestamp": message["rtp_timestamp"],
                                "frame": frame,
                            }
                        )
                elif request_id in self.pending and kind in {
                    "answer",
                    "ok",
                    "sent",
                    "error",
                }:
                    future = self.pending[request_id]
                    if not future.done():
                        if kind == "error":
                            logging.getLogger("uvicorn.error").warning(
                                "WHITEBOX_RTC_NATIVE_ERROR code=%s request_id=%s",
                                message.get("code"),
                                request_id,
                            )
                            error = RuntimeError(
                                "The media endpoint could not complete this operation"
                            )
                            code = message.get("code")
                            if isinstance(code, str) and re.fullmatch(
                                r"[A-Za-z][A-Za-z0-9_.:-]{0,95}", code
                            ):
                                error.upstream_code = code
                            future.set_exception(error)
                        else:
                            future.set_result(message)
                else:
                    self.on_event(message)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.error = error
            self.ready.set()
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(RuntimeError("The media endpoint stopped"))
            if not self.closed:
                self.on_event({"type": "state", "state": "failed"})

    async def exchange(self, kind, *, payload=b"", metadata=None, **fields):
        self._check()
        if len(self.pending) >= 8 or len(payload) > MAX_AU:
            raise ValueError("Native media operation budget exceeded")
        self.sequence += 1
        request_id = self.sequence
        raw = json.dumps(
            dict(type=kind, id=request_id, **fields), separators=(",", ":")
        ).encode()
        if len(raw) > 256 << 10:
            raise ValueError("Native media control budget exceeded")
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        if metadata is not None:
            self.frames[request_id] = metadata
        try:
            async with self.write_lock:
                self._check()
                self.process.stdin.write(
                    HEADER.pack(len(raw), len(payload)) + raw + payload
                )
                await asyncio.wait_for(self.process.stdin.drain(), 2)
            return await asyncio.wait_for(future, 5 if kind == "offer" else 2)
        finally:
            self.pending.pop(request_id, None)
            self.frames.pop(request_id, None)

    async def offer(self, sdp):
        return await self.exchange("offer", sdp=sdp)

    async def candidate(self, value):
        return await self.exchange("candidate", candidate=value)

    async def send_frame(self, message):
        # RTCSubscription owns bounded per-track credits. IPC writes stay atomic,
        # but a frame waits only for its own native departure, never the full queue.
        # UI events already travel reliably as whitebox.states.
        metadata = {
            key: value
            for key, value in message.items()
            if key not in {"payload", "states", "rtc_pts_us"}
        }
        return await self.exchange(
            "frame",
            payload=message["payload"],
            metadata=metadata,
            track=message["stream"],
            epoch=message["media_stream_id"],
            pts_us=message.get("rtc_pts_us", message["pts_us"]),
        )

    async def close(self):
        if self.close_task is None:
            self.closed = True
            self.close_task = asyncio.create_task(
                self._close(), name="rtc-native-close"
            )
        await asyncio.shield(self.close_task)

    async def _close(self):
        if self.process:
            if self.process.stdin:
                self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), 2)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), 2)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
        tasks = [task for task in (self.reader, self.stderr) if task]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for future in self.pending.values():
            if not future.done():
                future.cancel()
        self.pending.clear()
        self.frames.clear()
        # A port remains leased until its native process has actually exited.
        # Failed cleanup therefore cannot hand a live socket to another source.
        if self.endpoint_lease is not None:
            port, marker = self.endpoint_lease
            if _endpoint_leases.get(port) is marker:
                del _endpoint_leases[port]
            self.endpoint_lease = None
