# SPDX-License-Identifier: Apache-2.0
"""RGB adapter to the public Worldkit streaming compositor.

VP8 is confined to the local SDK contract. Public playback/recordings remain
H.264. One in-flight model frame prevents the realtime player from dropping
intermediate presentations; every returned frame must match its source evidence.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
import threading
from fractions import Fraction
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

from .cli_room import decode_packet
from .metrics import stage
from .rooms import new_id


class StreamingCompositor:
    metadata = {
        "mode": "worldkit_stream_compositor",
        "training_eligible": True,
        "mapping": "source_frame_and_ui_revision",
        "capture": "chromium_self_tab_vp8",
        "local_transport_codec": "vp8",
    }

    def __init__(self, descriptor, spec):
        self.source, self.spec = descriptor, spec
        self.stream_id, self.token = new_id(), secrets.token_urlsafe(32)
        self.closed = self.finishing = False
        self.count = 0
        self.process = self.server = self.control = self.media = None
        self.upstream = {}
        self.connected = {key: asyncio.Event() for key in ("control", "video")}
        self.tasks = []
        self.failure = asyncio.Event()
        self.error = None
        self.ended = asyncio.Event()
        self.configured = asyncio.Event()
        self.output = asyncio.Queue(maxsize=1)
        self.lock = asyncio.Lock()
        self.composition = None
        self.encoder = self.decoder = None
        self.codec_lock = threading.Lock()

    def fail(self, error):
        if not self.closed and not self.failure.is_set():
            self.error = error
            self.failure.set()

    async def wait(self, operation, timeout=30):
        task = asyncio.ensure_future(operation)
        failed = asyncio.create_task(self.failure.wait())
        try:
            done, _ = await asyncio.wait(
                (task, failed), timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if failed in done:
                raise ValueError("The streaming compositor failed") from self.error
            if task not in done:
                raise TimeoutError("The streaming compositor timed out")
            return task.result()
        finally:
            for pending in (task, failed):
                if not pending.done():
                    pending.cancel()
            await asyncio.gather(task, failed, return_exceptions=True)

    async def serve_model(self, socket):
        url = urlsplit(socket.request.path)
        channel = url.path.lstrip("/")
        if (
            parse_qs(url.query).get("token") != [self.token]
            or channel not in self.connected
            or channel in self.upstream
        ):
            await socket.close(code=1008, reason="Invalid composition connection")
            return
        self.upstream[channel] = socket
        try:
            if channel == "control":
                await socket.send(
                    json.dumps(
                        {
                            "type": "media.config",
                            "streamId": self.stream_id,
                            "epoch": self.source["epoch"],
                            "mediaGeneration": 1,
                            "config": {
                                "codec": "vp8",
                                "codedWidth": self.spec.width,
                                "codedHeight": self.spec.height,
                                "colorSpace": {
                                    "primaries": "bt709",
                                    "transfer": "bt709",
                                    "matrix": "bt709",
                                    "fullRange": False,
                                },
                            },
                        }
                    )
                )
            self.connected[channel].set()
            async for wire in socket:
                value = json.loads(wire)
                if (
                    channel != "control"
                    or value.get("type") != "media.keyframe"
                    or value.get("streamId") != self.stream_id
                    or value.get("epoch") != self.source["epoch"]
                ):
                    raise ValueError("Invalid model control message")
                # Every local model frame is independently decodable.
        except Exception as exc:
            if not self.finishing:
                self.fail(exc)
        finally:
            if not self.closed and not self.finishing:
                self.fail(ValueError("The model observer disconnected"))

    async def start(self):
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve

        self.server = await serve(
            self.serve_model,
            "127.0.0.1",
            0,
            max_size=65536,
            compression=None,
            close_timeout=1,
        )
        port = self.server.sockets[0].getsockname()[1]
        model = {
            "kind": "worldkit-model-video-stream",
            "schemaVersion": 1,
            "streamId": self.stream_id,
            "sourceSessionId": self.source["sessionId"],
            "sourceEpoch": self.source["epoch"],
            "uiBundleHash": self.source["uiBundleHash"],
            "codec": "vp8",
            "mapping": "exact-frame",
            "streams": {
                key: {"url": f"ws://127.0.0.1:{port}/{key}?token={self.token}"}
                for key in ("control", "video")
            },
        }
        cli = Path(os.environ["WHITEBOX_CLI_ENTRYPOINT"])
        root = cli.parents[3]
        self.process = await asyncio.create_subprocess_exec(
            os.environ.get("WHITEBOX_NODE", "node"),
            "--import",
            "tsx",
            str(Path(__file__).with_name("worldkit_compositor.mjs")),
            cwd=root,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=1048576,
            start_new_session=True,
        )
        browser = {"rendering": os.environ.get("WHITEBOX_CLI_RENDERING", "auto")}
        if os.environ.get("WHITEBOX_CLI_BROWSER_EXECUTABLE"):
            browser["executablePath"] = os.environ["WHITEBOX_CLI_BROWSER_EXECUTABLE"]
        request = {
            "source": {
                key: self.source[key] for key in ("sessionId", "baseUrl", "accessToken")
            },
            "videoInput": model,
            "browser": browser,
            "video": {
                "width": self.spec.width,
                "height": self.spec.height,
                "fps": self.spec.fps,
                "bitrate": 8000000,
            },
        }
        self.process.stdin.write((json.dumps(request) + "\n").encode())
        await self.process.stdin.drain()
        line = await self.wait(self.process.stdout.readline())
        if not line.endswith(b"\n"):
            raise ValueError("The composition host stopped before startup")
        descriptor = json.loads(line)
        base = urlsplit(descriptor["baseUrl"])
        if (
            descriptor.get("kind") != "worldkit-composition"
            or descriptor.get("status") != "starting"
            or base.hostname != "127.0.0.1"
            or base.scheme != "http"
            or descriptor.get("sourceSessionId") != self.source["sessionId"]
            or descriptor.get("sourceEpoch") != self.source["epoch"]
            or descriptor.get("uiBundleHash") != self.source["uiBundleHash"]
        ):
            raise ValueError("Invalid composition identity")
        self.composition = descriptor
        for key in ("control", "video"):
            endpoint = urlsplit(descriptor["streams"][key]["url"])
            expected = "control" if key == "control" else "media"
            if (
                endpoint.scheme != "ws"
                or endpoint.netloc != base.netloc
                or endpoint.query
                or endpoint.path
                != f"/v1/compositions/{descriptor['compositionId']}/{expected}"
            ):
                raise ValueError("Invalid composition endpoint")
            url = (
                endpoint.geturl()
                + "?"
                + urlencode(
                    {"token": descriptor["accessToken"], "clientId": self.stream_id}
                )
            )
            connection = await connect(
                url, max_size=4259844, max_queue=4, compression=None, close_timeout=1
            )
            if key == "control":
                self.control = connection
            else:
                self.media = connection
        self.tasks = [
            asyncio.create_task(self.read_control()),
            asyncio.create_task(self.read_media()),
        ]
        # The public player opens model sockets only after source ledger and UI
        # history exist. Do not capture H3 source frames before that boundary.
        await self.wait(
            asyncio.gather(*(e.wait() for e in self.connected.values())), timeout=120
        )

    async def read_control(self):
        try:
            async for wire in self.control:
                value = json.loads(wire)
                if (
                    value.get("compositionId") != self.composition["compositionId"]
                    or value.get("epoch") != self.source["epoch"]
                ):
                    raise ValueError("The composition identity changed")
                if value["type"] == "media.config":
                    config = value["config"]
                    if (
                        value["mediaGeneration"] != 1
                        or config.get("codec") != "vp8"
                        or (config["codedWidth"], config["codedHeight"])
                        != (self.spec.width, self.spec.height)
                    ):
                        raise ValueError("The composition video configuration changed")
                    self.configured.set()
                elif value["type"] == "media.ended":
                    if not self.finishing or value["lastOutputFrameId"] != self.count:
                        raise ValueError("The composition ended at an unexpected frame")
                    self.ended.set()
                    return
                else:
                    code = str(value.get("code", "UNKNOWN"))
                    safe_code = (
                        code
                        if code.startswith("COMPOSE_")
                        and code.replace("_", "").isalnum()
                        else "UNKNOWN"
                    )
                    raise ValueError("The composition reported " + safe_code[:128])
            if not self.ended.is_set():
                raise ValueError("The composition control stream ended early")
        except Exception as exc:
            self.fail(exc)

    async def read_media(self):
        try:
            async for wire in self.media:
                header, payload = decode_packet(wire, self.source, self.spec)
                if (
                    header.get("compositionId") != self.composition["compositionId"]
                    or header.get("uiComposited") is not True
                    or header["mediaGeneration"] != 1
                ):
                    raise ValueError("Invalid composed frame identity")
                self.output.put_nowait((header, payload))
            if not self.finishing:
                raise ValueError("The composition video stream ended early")
        except Exception as exc:
            self.fail(exc)

    async def consume_ui(self, message):
        # Source control is consumed directly by the public SDK observer.
        # CLIView independently retains the same opaque messages for recordings.
        pass

    def encode(self, rgb, index):
        with self.codec_lock:
            if self.closed:
                raise ValueError("The compositor is closed")
            return self._encode(rgb, index)

    def _encode(self, rgb, index):
        import av
        import numpy as np

        if self.encoder is None:
            self.encoder = av.CodecContext.create("libvpx", "w")
            c = self.encoder
            c.width, c.height = self.spec.width, self.spec.height
            c.pix_fmt, c.time_base = "yuv420p", Fraction(1, self.spec.fps)
            c.framerate, c.bit_rate, c.gop_size, c.thread_count = (
                Fraction(self.spec.fps),
                8000000,
                1,
                2,
            )
            c.colorspace = c.color_primaries = c.color_trc = 1  # BT.709
            c.color_range = 1  # MPEG limited range
            c.options = {
                "deadline": "realtime",
                "cpu-used": "4",
                "lag-in-frames": "0",
                "auto-alt-ref": "0",
            }
            c.open()
        frame = av.VideoFrame.from_ndarray(
            np.frombuffer(rgb, np.uint8).reshape(self.spec.height, self.spec.width, 3),
            format="rgb24",
        )
        frame = frame.reformat(format="yuv420p", dst_colorspace="ITU709")
        frame.pts, frame.time_base = index, Fraction(1, self.spec.fps)
        packets = self.encoder.encode(frame)
        if len(packets) != 1 or packets[0].pts != index or not packets[0].is_keyframe:
            raise ValueError("The model bridge buffered or reordered a frame")
        return bytes(packets[0])

    def decode(self, payload):
        with self.codec_lock:
            if self.closed:
                raise ValueError("The compositor is closed")
            return self._decode(payload)

    def _decode(self, payload):
        import av

        if self.decoder is None:
            self.decoder = av.CodecContext.create("vp8", "r")
        frames = self.decoder.decode(av.Packet(payload))
        if len(frames) != 1:
            raise ValueError("The composition could not be decoded")
        return frames[0].to_ndarray(format="rgb24").tobytes()

    async def render(self, rgb, frame, spec):
        async with self.lock:
            if (
                self.closed
                or self.finishing
                or frame.index != self.count
                or spec != self.spec
                or len(rgb) != spec.frame_bytes
            ):
                raise ValueError("Invalid composition frame order")
            source = frame.provenance["cli_frame_header"]
            with stage(
                "cli_composition", self.source["sessionId"], service="room_runtime"
            ):
                payload = await asyncio.to_thread(self.encode, rgb, frame.index)
                header = {
                    **source,
                    "mediaGeneration": 1,
                    "outputFrameId": self.count + 1,
                    "outputPtsUs": frame.pts_us,
                    "type": "key",
                    "payloadBytes": len(payload),
                }
                data = json.dumps(header).encode()
                await self.wait(
                    self.upstream["video"].send(
                        len(data).to_bytes(4, "big") + data + payload
                    )
                )
                actual, encoded = await self.wait(self.output.get())
                await self.wait(self.configured.wait())
                if (
                    actual["outputFrameId"] != self.count + 1
                    or actual["outputPtsUs"] != frame.pts_us
                    or any(
                        actual[key] != source[key]
                        for key in ("source", "uiRevision", "uiCompleteThroughUs")
                    )
                ):
                    raise ValueError(
                        "The composed frame does not match the model source"
                    )
                output = await asyncio.to_thread(self.decode, encoded)
            self.count += 1
            return output

    async def finish(self):
        async with self.lock:
            if self.ended.is_set() or self.count == 0:
                return
            self.finishing = True
            await self.wait(
                self.upstream["control"].send(
                    json.dumps(
                        {
                            "type": "media.ended",
                            "streamId": self.stream_id,
                            "epoch": self.source["epoch"],
                            "mediaGeneration": 1,
                            "lastOutputFrameId": self.count,
                        }
                    )
                )
            )
            await self.wait(self.ended.wait(), timeout=8)

    async def close(self):
        if self.closed:
            return
        self.closed = True
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await asyncio.gather(
            *(c.close() for c in (self.control, self.media) if c),
            return_exceptions=True,
        )
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        if self.process and self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.process.wait(), 10)
            except TimeoutError:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await self.process.wait()
        # Cancelling to_thread does not stop native codec work. Wait before
        # dropping contexts; a cancelled session must not race codec teardown.
        await asyncio.to_thread(self.release_codecs)

    def release_codecs(self):
        with self.codec_lock:
            self.encoder = self.decoder = None


async def factory(*, descriptor, spec):
    renderer = StreamingCompositor(descriptor, spec)
    try:
        await renderer.start()
        return renderer
    except BaseException:
        await renderer.close()
        raise
