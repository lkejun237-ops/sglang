# SPDX-License-Identifier: Apache-2.0
"""One bounded play startup owns the early model lease until segment handoff."""

from __future__ import annotations

import asyncio
import copy
import re
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import msgspec

from ..source_prepare import (
    PROTOCOL,
    STARTUP_SECONDS,
    V2_PROTOCOL,
    condition_fingerprint,
    rgb_digest,
    snapshot_rgb,
    validate_snapshot_dimensions,
)
from .contracts import VideoSpec
from .diagnostics import model_preparation_error
from .model_geometry import ModelGeometry
from .render_contract import (
    MAX_REFERENCE_BYTES,
    MAX_REFERENCE_IMAGES,
    RenderError,
    canonical,
    prompt_text,
)


def startup_fingerprint(request):
    def normalize(value):
        if isinstance(value, bytes):
            return {"bytes": len(value), "sha256": rgb_digest(value)}
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return rgb_digest(canonical(normalize(request)))


async def open_startup_engine(factory, kwargs, timeout):
    """Cancel an in-flight join, or close a result that won a cancellation race."""
    task = asyncio.ensure_future(factory(**kwargs))
    try:
        async with asyncio.timeout(timeout):
            return await asyncio.shield(task)
    except BaseException:
        task.cancel()
        (result,) = await asyncio.gather(task, return_exceptions=True)
        if isinstance(result, Exception):
            raise result
        if not isinstance(result, BaseException):
            try:
                await asyncio.wait_for(result.close(), STARTUP_SECONDS)
            finally:
                renderer = getattr(result, "renderer", None)
                if renderer is not None:
                    await asyncio.wait_for(renderer.close(), 10)
        raise


class StartupAssets:
    """Fixed manifest, one in-progress image, ordered idempotent chunks."""

    def __init__(self, request, spec):
        self.spec = spec
        self.request = copy.deepcopy(request)
        self.items, self.data = {}, {}
        self.partial_id, self.partial = None, bytearray()
        self.committed = asyncio.Event()
        render = request.get("initial_render")
        if not isinstance(render, dict) or set(render) - {
            "prompt",
            "seed",
            "references",
        }:
            raise RenderError("CONDITION_INVALID")
        self.prompt = prompt_text(render.get("prompt"))
        self.seed = render.get("seed", 42)
        refs = render.get("references", [])
        if (
            type(self.seed) is not int
            or not 0 <= self.seed < 2**32
            or not isinstance(refs, list)
            or len(refs) > MAX_REFERENCE_IMAGES
        ):
            raise RenderError("CONDITION_INVALID")
        self.reference_ids = []
        for ref in refs:
            self._add(ref, reference=True)
        self.protocol = request["protocol"]
        self.frame = request.get("initial_frame")
        if self.frame is not None:
            if not isinstance(self.frame, dict):
                raise RenderError("CONDITION_INVALID")
            if self.protocol == V2_PROTOCOL:
                try:
                    validate_snapshot_dimensions(
                        self.frame.get("width"), self.frame.get("height")
                    )
                except ValueError as error:
                    raise RenderError("CONDITION_INVALID") from error
            self._add({**self.frame, "id": "__source_s0__"}, reference=False)
        if len(self.data) == len(self.items):
            self.commit()

    def _add(self, item, *, reference):
        if not isinstance(item, dict):
            raise RenderError("CONDITION_INVALID")
        allowed = {"id", "format", "encoded_sha256", "encoded_size", "payload"}
        if not reference:
            allowed |= {"width", "height", "rgb_sha256"}
        if set(item) - allowed:
            raise RenderError("CONDITION_INVALID")
        name, raw = item.get("id"), item.get("payload")
        if (
            not isinstance(name, str)
            or not 0 < len(name) <= 128
            or name in self.items
            or (reference and name in {"<Video 1>", "__source_s0__"})
        ):
            raise RenderError("CONDITION_INVALID")
        if item.get("format") not in (
            {"png", "jpeg", "webp"} if reference else {"png"}
        ):
            raise RenderError("CONDITION_INVALID")
        digest, size = item.get("encoded_sha256"), item.get("encoded_size")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or type(size) is not int
            or not 0 < size <= (8 << 20 if reference else 2 << 20)
        ):
            raise RenderError("LIMIT_EXCEEDED")
        if reference and (
            sum(x["encoded_size"] for x in self.items.values()) + size
            > MAX_REFERENCE_BYTES
        ):
            raise RenderError("LIMIT_EXCEEDED")
        self.items[name] = {k: v for k, v in item.items() if k != "payload"}
        if reference:
            self.reference_ids.append(name)
        if raw is not None:
            self._complete(name, raw)

    def _complete(self, name, raw):
        descriptor = self.items[name]
        if (
            not isinstance(raw, bytes)
            or len(raw) != descriptor["encoded_size"]
            or rgb_digest(raw) != descriptor["encoded_sha256"]
        ):
            raise RenderError("CONDITION_INVALID")
        self.data[name] = raw

    def chunk(self, name, offset, data):
        if (
            name not in self.items
            or type(offset) is not int
            or offset < 0
            or not isinstance(data, bytes)
            or not 0 < len(data) <= 1 << 20
        ):
            raise RenderError("INVALID_COMMAND")
        if name in self.data:
            if self.data[name][offset : offset + len(data)] != data:
                raise RenderError("REQUEST_ID_CONFLICT")
            return offset + len(data)
        if self.committed.is_set():
            raise RenderError("INVALID_COMMAND")
        if self.partial_id not in {None, name}:
            raise RenderError("LIMIT_EXCEEDED")
        if offset < len(self.partial):
            if offset < 0 or self.partial[offset : offset + len(data)] != data:
                raise RenderError("REQUEST_ID_CONFLICT")
            return offset + len(data)
        if (
            offset != len(self.partial)
            or offset + len(data) > self.items[name]["encoded_size"]
        ):
            raise RenderError("INVALID_COMMAND")
        self.partial_id = name
        self.partial.extend(data)
        end = len(self.partial)
        if end == self.items[name]["encoded_size"]:
            self._complete(name, bytes(self.partial))
            self.partial_id, self.partial = None, bytearray()
        return end

    def commit(self):
        if self.committed.is_set():
            return
        if len(self.items) != len(self.data) or self.partial_id is not None:
            raise RenderError("CONDITION_INVALID")
        if self.frame is not None:
            frame = {
                k: v
                for k, v in self.frame.items()
                if k not in {"encoded_size", "payload"}
            }
            try:
                self.rgb = snapshot_rgb(
                    {**frame, "payload": self.data["__source_s0__"]},
                    self.spec.width,
                    self.spec.height,
                    protocol=self.protocol,
                )
            except (ValueError, OSError) as error:
                raise RenderError("CONDITION_INVALID") from error
        else:
            self.rgb = None
        self.references = [
            dict(id=i, format=self.items[i]["format"], payload=self.data[i])
            for i in self.reference_ids
        ]
        self.committed.set()

    def clear(self):
        self.data.clear()
        self.partial.clear()
        self.references = []
        self.rgb = None


def validate_startup(value, spec):
    if (
        not isinstance(value, dict)
        or value.get("protocol") not in {PROTOCOL, V2_PROTOCOL}
        or value.get("intent") != "model"
    ):
        raise RenderError("CAPABILITY_UNAVAILABLE")
    if set(value) - {
        "protocol",
        "request_id",
        "intent",
        "world_revision",
        "initial_state_id",
        "initial_frame",
        "initial_render",
        "fallback_policy",
    }:
        raise RenderError("INVALID_COMMAND")
    if value.get("initial_state_id") is not None:
        # Existing CLI frames do not attest a world-initial-state manifest.
        raise RenderError("CAPABILITY_UNAVAILABLE")
    from .render_contract import identity

    identity(value.get("request_id"))
    if (value["protocol"] == V2_PROTOCOL and "fallback_policy" in value) or value.get(
        "fallback_policy", "reject"
    ) not in {"reject", "cli_initial_once"}:
        raise RenderError("INVALID_COMMAND")
    return StartupAssets(value, spec)


class PlayStartup:
    """The background task keeps exactly one ModelLeases context alive.

    A segment borrows it once, then owns closing it. Cancellation before handoff
    is owned by SourceSession; after handoff its normal segment close is awaited.
    """

    is_startup = True

    def __init__(self, owner, request):
        self.owner, self.request = owner, copy.deepcopy(request)
        self.assets = validate_startup(request, owner.config.spec)
        self.protocol = request["protocol"]
        self.request_fingerprint = startup_fingerprint(request)
        self.created_at = time.monotonic()
        self.deadline = self.created_at + STARTUP_SECONDS
        self.events = []
        self.started = self.transferred = self.closed = False
        self.attempt = 0
        self.admitted, self.prepared, self.stop = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )
        self.cli_ready = asyncio.Event()
        self.rtc_ready = asyncio.Event()
        self.cli_rgb = self.cli_proof = self.pinned_source = None
        self.cli_rgb_sha256 = None
        self.model_cli_rgb_sha256 = None
        self.ids = self._ids()
        self.ws = self.assignment = self.ready = None
        self.identity = None
        self.task = None
        self.lease = None
        self.event_sequence = 0

    def _ids(self):
        return {
            k: uuid4().hex
            for k in (
                "segment_id",
                "segment_epoch",
                "model_session_id",
                "generation_id",
                "preparation_id",
            )
        }

    async def progress(self, phase, **fields):
        self.note(phase)
        await self.owner.outgoing.put(
            msgspec.msgpack.encode(
                dict(
                    type="startup.progress",
                    protocol=self.protocol,
                    request_id=self.request["request_id"],
                    phase=phase,
                    attempt=self.attempt,
                    **fields,
                )
            )
        )

    def note(self, phase):
        self.events.append(
            dict(
                phase=phase,
                elapsed_ms=(time.monotonic() - self.created_at) * 1000,
                attempt=self.attempt,
            )
        )

    def launch(self, init):
        self.init = copy.deepcopy(init)
        source = init["source_stream"]
        self.geometry = ModelGeometry(
            VideoSpec(source["width"], source["height"], source["fps"]),
            getattr(self.owner.config, "model_target", None),
        )
        self.task = asyncio.create_task(self._run())
        self.task.add_done_callback(
            lambda t: t.exception() if not t.cancelled() else None
        )

    async def wait(self, event):
        waiter = asyncio.create_task(event.wait())
        try:
            async with asyncio.timeout(max(0, self.deadline - time.monotonic())):
                done, _ = await asyncio.wait(
                    (waiter, self.task), return_when=asyncio.FIRST_COMPLETED
                )
                if self.task in done:
                    await self.task
                    raise RenderError("INITIALIZATION_FAILED")
                return waiter.result()
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    async def _model_send(self, message, operation):
        try:
            await self.ws.send(msgspec.msgpack.encode(message))
        except Exception as error:
            raise model_preparation_error(
                {"code": type(error).__name__, "content": str(error)},
                trace_id=getattr(self.owner, "trace_id", None),
                operation=operation,
            ) from error

    async def _model_reply(self, operation):
        try:
            return msgspec.msgpack.decode(await self.ws.recv())
        except Exception as error:
            raise model_preparation_error(
                {"code": type(error).__name__, "content": str(error)},
                trace_id=getattr(self.owner, "trace_id", None),
                operation=operation,
            ) from error

    async def _event(self, kind, payload, expected):
        self.event_sequence += 1
        await self._model_send(
            dict(
                type="event", kind=kind, event_id=self.event_sequence, payload=payload
            ),
            kind,
        )
        while True:
            message = await self._model_reply(kind)
            if not isinstance(message, dict):
                raise model_preparation_error(
                    message,
                    trace_id=getattr(self.owner, "trace_id", None),
                    operation=kind,
                )
            if message.get("type") == expected:
                if message.get("identity") != self.identity:
                    raise RenderError("SOURCE_EPOCH_MISMATCH")
                return message
            if message.get("type") not in {"source.prepare.accepted", "heartbeat"}:
                raise model_preparation_error(
                    message,
                    trace_id=getattr(self.owner, "trace_id", None),
                    operation=kind,
                )

    async def _run(self):
        try:
            # Admission completes before the caller starts the expensive CLI.
            # Verified release clears the previous lease before the worker's
            # next heartbeat clears its reported load. Only the single fallback
            # may wait for that refresh; initial admission remains fail-fast.
            capacity_wait_ms = (
                min(10000, max(1, int((self.deadline - time.monotonic()) * 1000)))
                if self.attempt
                else 0
            )
            segment = {
                **self.ids,
                "capacity_wait_ms": capacity_wait_ms,
                "references": self.assets.reference_ids,
            }
            if getattr(self.owner, "model_binding", None) is not None:
                await self.owner.model_binding.wait_ready()
                self.lease = self.owner.segment_leases(segment)
            else:
                self.lease = self.owner.host.make_segment_leases(
                    self.owner.user_id, segment
                )
            async with self.lease.open() as (self.ws, self.assignment):
                self.note("admitted")
                self.admitted.set()
                async with asyncio.timeout(max(0, self.deadline - time.monotonic())):
                    await self.assets.committed.wait()
                    init = {
                        **self.init,
                        "prompt": self.assets.prompt,
                        "seed": self.assets.seed,
                        "references": self.assets.references,
                        "source_prepare_protocol": self.protocol,
                        "source_prompt_protocol": "ordered-boundary-v2",
                        "source_condition_switch_protocol": "clip-barrier-v1",
                        "source_demand_protocol": "source-demand-v1",
                    }
                    init = self.geometry.model_init(init)
                    init["record_model_inputs"] = True
                    self.model_input_init = init
                    await self._model_send(init, "init")
                    self.ready = await self._model_reply("init")
                    if (
                        not isinstance(self.ready, dict)
                        or self.ready.get("type") == "error"
                    ):
                        raise model_preparation_error(
                            self.ready,
                            trace_id=getattr(self.owner, "trace_id", None),
                            operation="init",
                        )
                    if (
                        self.ready.get("type") != "session.ready"
                        or self.ready.get("source_prepare_protocol") != self.protocol
                        or self.ready.get("source_stream") != init["source_stream"]
                        or not re.fullmatch(
                            r"[a-f0-9]{64}",
                            self.ready.get("source_prepare_profile_sha256", ""),
                        )
                    ):
                        raise RenderError("CAPABILITY_UNAVAILABLE")
                    rgb = self.assets.rgb
                    frame = self.assets.frame
                    if rgb is None:
                        await self.progress("waiting_source")
                        await self.cli_ready.wait()
                        rgb = self.cli_rgb
                    self.preview_width = (
                        self.assets.frame["width"]
                        if self.assets.frame is not None
                        else self.owner.config.spec.width
                    )
                    self.preview_height = (
                        self.assets.frame["height"]
                        if self.assets.frame is not None
                        else self.owner.config.spec.height
                    )
                    self.rgb = rgb
                    # V1 S0 equals private video zero; V2 previews remain native.
                    self.model_rgb = (
                        self.geometry.pad_rgb(rgb) if self.protocol == PROTOCOL else rgb
                    )
                    self.identity = dict(
                        preparation_id=self.ids["preparation_id"],
                        source_session_id=self.owner.session_id,
                        source_epoch=self.owner.source_epoch,
                        worker_epoch=self.assignment.denoiser.worker_epoch,
                        condition_fingerprint=condition_fingerprint(
                            rgb=self.model_rgb,
                            prompt=self.assets.prompt,
                            references=self.assets.references,
                            seed=self.assets.seed,
                            source_stream=init["source_stream"],
                            profile_sha256=self.ready["source_prepare_profile_sha256"],
                            protocol=self.protocol,
                            preview_width=self.preview_width,
                            preview_height=self.preview_height,
                        ),
                    )
                    await self.progress("model_preparing")
                    await self._event(
                        "source.prepare",
                        dict(
                            identity=self.identity,
                            rgb=self.model_rgb,
                            **(
                                {
                                    "width": self.preview_width,
                                    "height": self.preview_height,
                                }
                                if self.protocol == V2_PROTOCOL
                                else {}
                            ),
                        ),
                        "source.prepared",
                    )
                    self.prepared.set()
                    self.note("model_prepared")
                    await self.owner.outgoing.put(
                        msgspec.msgpack.encode(
                            dict(
                                type="model.prepared",
                                protocol=self.protocol,
                                request_id=self.request["request_id"],
                                identity=self.identity,
                                level="condition",
                                attempt=self.attempt,
                            )
                        )
                    )
                await self.stop.wait()
        finally:
            self.ws = None

    def observe_source(self, version, frame, captured):
        """Pin one real view frame, while the ordinary source pipeline advances."""
        if self.closed or self.started or self.pinned_source is not None:
            return
        source = (frame.provenance or {}).get("cli_frame_header", {}).get("source", {})
        if (
            len(frame.rgb) != self.owner.config.spec.frame_bytes
            or not source.get("presentationId")
            or type(source.get("sdkEpoch")) is not int
            or type(source.get("simulationTick")) is not int
            or source["simulationTick"] < 0
            or type(source.get("sourceTimeUs")) is not int
            or "worldRevision" not in source
        ):
            raise RenderError("SOURCE_EPOCH_MISMATCH")
        self.pinned_source = (version, (frame, captured))
        self.cli_rgb = frame.rgb
        self.cli_proof = dict(
            protocol="room-source-frame-v1",
            world_revision=getattr(self.owner.engine, "metadata", {}).get(
                "world_build_hash"
            ),
            source_frame_index=frame.index,
            source=copy.deepcopy(source),
            **self.owner.source_identity(),
        )
        self.note("cli_frame_selected")
        self.cli_ready.set()

    def source_started(self):
        self.started = True
        self.pinned_source = None
        self.cli_rgb = None
        self.note("source_started")

    async def bind_cli(self, rgb, proof):
        if (
            len(rgb) != self.owner.config.spec.frame_bytes
            or proof.get("protocol") != "room-source-frame-v1"
        ):
            raise RenderError("CAPABILITY_UNAVAILABLE")
        if self.request.get("initial_state_id") is not None:
            raise RenderError("CAPABILITY_UNAVAILABLE")
        if self.request.get("world_revision") is not None and self.request[
            "world_revision"
        ] != proof.get("world_revision"):
            raise RenderError("SOURCE_EPOCH_MISMATCH")
        self.cli_rgb, self.cli_proof = rgb, proof
        self.cli_rgb_sha256 = rgb_digest(rgb)
        self.model_cli_rgb_sha256 = rgb_digest(self.geometry.pad_rgb(rgb))
        self.note("cli_ready")
        self.cli_ready.set()
        await self.assets.committed.wait()
        # V2 S0 is an independent condition preview. Its different pixels or
        # size never replace the pinned live frame or trigger a second lease.
        if (
            self.protocol == PROTOCOL
            and self.assets.rgb is not None
            and rgb != self.assets.rgb
        ):
            if (
                self.request.get("fallback_policy") != "cli_initial_once"
                or self.attempt
                or self.started
            ):
                raise RenderError("SOURCE_EPOCH_MISMATCH")
            # Fully release the first worker attempt before reserving another.
            await self.close_attempt()
            if self.resource_state != "released":
                raise RenderError("RELEASE_UNKNOWN")
            self.attempt = 1
            self.ids = self._ids()
            self.assets.rgb = rgb
            self.admitted.clear()
            self.prepared.clear()
            self.stop.clear()
            await self.progress(
                "fallback",
                requested_rgb_sha256=self.request["initial_frame"]["rgb_sha256"],
                effective_rgb_sha256=rgb_digest(rgb),
            )
            self.launch(self.init)
        await self.wait(self.prepared)

    def receipt(self):
        result = dict(
            protocol=self.protocol,
            request_id=self.request["request_id"],
            request_fingerprint=self.request_fingerprint,
            attempt=self.attempt,
            identity=self.identity,
            profile_sha256=self.ready["source_prepare_profile_sha256"],
            cli_proof=self.cli_proof,
            source_clock="unmodified",
            start_policy="pinned_view_frame_then_latest",
            requested_rgb_sha256=(self.request.get("initial_frame") or {}).get(
                "rgb_sha256"
            ),
            effective_rgb_sha256=rgb_digest(self.rgb),
            condition_preview=dict(
                origin="request" if self.assets.frame is not None else "cli",
                width=self.preview_width,
                height=self.preview_height,
                rgb_sha256=rgb_digest(self.rgb),
            ),
            cli_frame_rgb_sha256=self.cli_rgb_sha256,
            model_cli_frame_rgb_sha256=self.model_cli_rgb_sha256,
            model_condition_rgb_sha256=rgb_digest(self.model_rgb),
            model_geometry=self.geometry.receipt(),
            events=list(self.events),
            level="condition",
        )
        if self.protocol == V2_PROTOCOL:
            result.update(
                condition_preview=dict(
                    origin="request" if self.assets.frame is not None else "cli",
                    width=self.preview_width,
                    height=self.preview_height,
                    rgb_sha256=rgb_digest(self.rgb),
                ),
            )
        return result

    @property
    def resource_state(self):
        return self.lease.resource_state if self.lease else "none"

    @property
    def cleanup_code(self):
        return getattr(self.lease, "cleanup_code", None)

    @asynccontextmanager
    async def open(self):
        await self.wait(self.prepared)
        if self.transferred or self.closed:
            raise RenderError("REQUEST_ID_CONFLICT")
        self.transferred = True
        try:
            yield self.ws, self.assignment
        finally:
            await self.close_attempt()

    async def attach(self):
        async with asyncio.timeout(max(0, self.deadline - time.monotonic())):
            await self.rtc_ready.wait()
            await self._event(
                "source.attach", dict(identity=self.identity), "source.attached"
            )
            self.note("attached")

    async def close_attempt(self):
        self.stop.set()
        if self.task and not self.task.done():
            self.task.cancel()
        if self.task:
            await asyncio.gather(self.task, return_exceptions=True)

    async def close(self):
        if self.closed:
            return
        self.closed = True
        await self.close_attempt()
        self.pinned_source = self.cli_rgb = None
        self.assets.clear()
