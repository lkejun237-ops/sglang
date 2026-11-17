# SPDX-License-Identifier: Apache-2.0
"""Independent world capture and one continuous, exclusive H3 generation.

Opt-in protocol. No model process/GPU lifetime changes. A never claims to be B.
Unsubmitted source frames may be superseded; submitted frames remain ordered.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import io
import logging
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import replace
from uuid import uuid4

import msgspec

from ..gateway import build_denoiser_url
from .contracts import capture_pts, presentation_pts
from .diagnostics import model_error, model_preparation_error
from .metrics import observe_duration, stage
from .playback import H264PlaybackEncoder
from .project_audio import public_ui_metadata
from .recording import reference_descriptors
from .session import WhiteboxUpstream
from .telemetry import (
    CHUNK_FIELDS,
    numeric_durations,
    sanitize_frame_stage_breakdown,
)

log = logging.getLogger(__name__)

# Two complete clip17 batches can overlap only if the four RGB frames held by
# the source Lynn decoder do not consume the next clip's input credit. This is
# a residency ceiling, not a batch size or a requirement to fill the window.
MODEL_SOURCE_WINDOW_FRAMES = 2 * 17 + 4
MODEL_HEARTBEAT_SECONDS = 5.0


def validate_configuration(value):
    if not isinstance(value, dict) or value.get("mode") not in ("whitebox", "model"):
        raise ValueError("Choose whitebox or model output.")
    prompt = value.get("prompt")
    refs = value.get("references")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16384:
        raise ValueError("Enter a prompt of at most 16384 characters.")
    if not isinstance(refs, list) or len(refs) > 9:
        raise ValueError("Choose up to nine reference images.")
    ids, total = set(), 0
    for ref in refs:
        if (
            not isinstance(ref, dict)
            or set(ref) - {"id", "format", "payload"}
            or not isinstance(ref.get("id"), str)
            or not 1 <= len(ref["id"]) <= 128
            or ref["id"] in ids
            or ref["id"] == "<Video 1>"
            or ref.get("format") not in ("png", "jpeg", "webp")
            or not isinstance(ref.get("payload"), bytes)
            or not ref["payload"]
        ):
            raise ValueError("Invalid reference image.")
        # Validate the immutable initialization parameters before starting H3.
        from PIL import Image

        try:
            with Image.open(io.BytesIO(ref["payload"])) as image:
                if (
                    image.format.lower() != ref["format"]
                    or image.width * image.height > 16_777_216
                ):
                    raise ValueError("Invalid reference image dimensions or format.")
                image.verify()
        except Exception as exc:
            raise ValueError("Invalid reference image.") from exc
        ids.add(ref["id"])
        total += len(ref["payload"])
    if total > 8 * 1024 * 1024:
        raise ValueError("Reference images exceed 8 MiB.")
    return dict(mode=value["mode"], prompt=prompt, references=copy.deepcopy(refs))


class LatestFrame:
    """One unpublished frame, independent cursors for source preview and model."""

    def __init__(self):
        self.value = None
        self.version = 0
        self.closed = False
        self.condition = asyncio.Condition()

    async def put(self, value):
        async with self.condition:
            self.value = value
            self.version += 1
            self.condition.notify_all()

    async def get(self, after):
        async with self.condition:
            await self.condition.wait_for(lambda: self.version > after or self.closed)
            if self.version <= after:
                return None
            return self.version, self.value

    async def close(self):
        async with self.condition:
            self.closed = True
            self.condition.notify_all()


class ModelLeases:
    def __init__(
        self,
        coordinator,
        connect,
        *,
        user_id,
        model_revision,
        vae_fingerprint,
        trace_id,
        max_bytes,
        renew_seconds=10,
        strict_release=False,
        session_id=None,
        generation_id=None,
        capacity_wait_ms=0,
        cleanup_timeout_s=30,
        operation_path=None,
    ):
        self.coordinator, self.connect = coordinator, connect
        self.request = dict(
            user_id=user_id,
            model_revision=model_revision,
            vae_fingerprint=vae_fingerprint,
            trace_id=trace_id,
            wait_for_capacity=False,
        )
        self.max_bytes, self.renew_seconds = max_bytes, renew_seconds
        self.strict_release = strict_release
        self.session_id, self.generation_id = session_id, generation_id
        self.cleanup_timeout_s = cleanup_timeout_s
        self.resource_state = "none"
        if operation_path not in {
            None,
            "/v1/realtime_video/prefill",
            "/v1/realtime_video/bound",
        }:
            raise ValueError("Invalid internal model operation")
        self.operation_path = operation_path
        self._compute_released = False
        self._active_assignment = self._active_renewal = None
        self.cleanup_code = None
        self.sdk_cleanup_provider = None
        self._cleanup_assignment = None
        if strict_release and capacity_wait_ms:
            self.request.update(
                wait_for_capacity=True, capacity_wait_ms=capacity_wait_ms
            )

    async def _release_verified(self, assignment):
        from .render_contract import RenderError

        self._cleanup_assignment = assignment
        self.resource_state = "releasing"
        deadline = time.monotonic() + self.cleanup_timeout_s
        options = {}
        if getattr(assignment, "player_expires_at", None) is not None:
            try:
                if self.sdk_cleanup_provider is None:
                    raise RenderError("RELEASE_UNKNOWN")
                receipt = await asyncio.wait_for(
                    self.sdk_cleanup_provider(assignment), self.cleanup_timeout_s
                )
                options["sdk_cleanup"] = receipt
            except BaseException:
                self.resource_state = "release_unknown"
                try:
                    await asyncio.wait_for(self.coordinator.quarantine(assignment), 5)
                except Exception:
                    logging.getLogger(__name__).warning(
                        "Player SDK cleanup quarantine unconfirmed", exc_info=True
                    )
                raise
        while time.monotonic() < deadline:
            try:
                await asyncio.wait_for(
                    self.coordinator.release_verified(assignment, **options),
                    min(5, max(0.1, deadline - time.monotonic())),
                )
                self.resource_state = "released"
                return
            except Exception:
                await asyncio.sleep(0.2)
        self.resource_state = "release_unknown"
        try:
            await asyncio.wait_for(self.coordinator.quarantine(assignment), 5)
            self.resource_state = "quarantined"
            # An earlier successful response may have been lost before the
            # quarantine request. Read its idempotent release receipt once more.
            await asyncio.wait_for(
                self.coordinator.release_verified(assignment, **options), 5
            )
            self.resource_state = "released"
            return
        except Exception:
            raise RenderError("RELEASE_UNKNOWN")

    async def retry_release(self):
        """Reconcile the same stopped lease; never repeat model admission."""
        from .render_contract import RenderError

        if self.resource_state in {"none", "released"}:
            return
        if (
            not self.strict_release
            or self._cleanup_assignment is None
            or self._active_renewal is not None
            and not self._active_renewal.done()
        ):
            raise RenderError("RELEASE_UNKNOWN")
        await self._release_verified(self._cleanup_assignment)
        self.cleanup_code = None

    async def release_compute(self):
        """Prefill acknowledged cleanup; disk observation can keep its socket."""
        if (
            self.operation_path != "/v1/realtime_video/prefill"
            or not self.strict_release
            or self._active_assignment is None
        ):
            raise ValueError("Early release requires an active prefill lease")
        if self._compute_released:
            return
        if self._active_renewal is not None:
            self._active_renewal.cancel()
            await asyncio.gather(self._active_renewal, return_exceptions=True)
        await self._release_verified(self._active_assignment)
        self._compute_released = True

    @asynccontextmanager
    async def open(self):
        assignment = None
        ws = renew = None
        primary_error = None
        # Finish an in-flight admit on cancellation so an acquired lease cannot
        # disappear between the HTTP response and assigning its cleanup owner.
        admit = asyncio.create_task(
            self.coordinator.admit(
                **self.request,
                **(
                    {"player_session": True}
                    if self.operation_path == "/v1/realtime_video/bound"
                    else {}
                ),
                session_id=self.session_id or uuid4().hex,
                generation_id=self.generation_id or uuid4().hex,
            )
        )
        try:
            try:
                assignment = await asyncio.shield(admit)
            except asyncio.CancelledError:
                assignment = await admit
                raise
            except Exception as exc:
                if self.strict_release and getattr(exc, "reason", None) not in {
                    "CAPACITY_EXHAUSTED",
                    "USER_SESSION_LIMIT",
                    "USER_ALREADY_ACTIVE",
                }:
                    self.resource_state = "release_unknown"
                raise
            self.resource_state = "reserved"
            self._active_assignment = assignment
            if assignment.vae is not None:
                raise ValueError("Interactive H3 requires local decoding.")
            url = build_denoiser_url(
                assignment.denoiser.endpoint,
                session_id=assignment.session_id,
                generation_id=assignment.generation_id,
                coordinator_token=assignment.token,
                worker_epoch=assignment.denoiser.worker_epoch,
                vae_url="",
                output_url="",
                output_token="",
                trace_id=self.request["trace_id"],
            )
            if self.operation_path is not None:
                from urllib.parse import urlsplit, urlunsplit

                parsed = urlsplit(url)
                url = urlunsplit(
                    (
                        parsed.scheme,
                        parsed.netloc,
                        self.operation_path,
                        parsed.query,
                        "",
                    )
                )
            ws = await self.connect(
                url,
                max_size=self.max_bytes,
                max_queue=1,
                compression=None,
                open_timeout=10,
                close_timeout=2,
                ping_interval=20,
                ping_timeout=60,
            )

            async def keep_lease():
                nonlocal assignment
                try:
                    while True:
                        await asyncio.sleep(self.renew_seconds)
                        assignment = await self.coordinator.renew(assignment)
                        self._active_assignment = assignment
                except Exception:
                    await ws.close()
                    raise

            renew = asyncio.create_task(keep_lease())
            self._active_assignment, self._active_renewal = assignment, renew
            yield ws, assignment
            if renew.done() and not self._compute_released:
                renew.result()
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if renew and not self.strict_release:
                renew.cancel()
                await asyncio.gather(renew, return_exceptions=True)
            try:
                try:
                    if ws:
                        await ws.close()
                finally:
                    if assignment and not self._compute_released:
                        if self.strict_release:
                            cleanup = asyncio.create_task(
                                self._release_verified(assignment)
                            )
                            try:
                                while not cleanup.done():
                                    try:
                                        await asyncio.shield(cleanup)
                                    except asyncio.CancelledError:
                                        continue
                                await cleanup
                            finally:
                                if renew:
                                    renew.cancel()
                                    await asyncio.gather(renew, return_exceptions=True)
                        else:
                            await self.coordinator.release(assignment)
            except Exception:
                # Cleanup uncertainty remains actionable, but cannot replace
                # the original failed operation (or turn cancellation into failure).
                if primary_error is None:
                    raise
                if self.resource_state != "released":
                    self.cleanup_code = "RELEASE_UNKNOWN"
                    primary_error.cleanup_code = self.cleanup_code
                log.exception(
                    "Model cleanup failed after primary error trace_id=%s",
                    self.request["trace_id"],
                )


def create_model_leases(*args, profile_hint=None, **kwargs):
    import os

    if os.environ.get("WHITEBOX_MODEL_BACKEND") == "role_pool":
        if kwargs.get("operation_path") == "/v1/realtime_video/prefill":
            from .disaggregated_prefill import DisaggregatedPrefillLeases

            return DisaggregatedPrefillLeases(
                *args, profile_hint=profile_hint, **kwargs
            )
        from .disaggregated_binding import DisaggregatedModelLeases

        return DisaggregatedModelLeases(*args, profile_hint=profile_hint, **kwargs)
    return ModelLeases(*args, **kwargs)


class _InitOnly:
    async def send(self, wire):
        self.init = msgspec.msgpack.decode(wire)

    async def close(self):
        pass


class InteractiveSession(WhiteboxUpstream):
    supports_model_padding = True

    def __init__(self, *, leases, **kwargs):
        self.leases = leases
        self.latest = LatestFrame()
        self.model_latest = LatestFrame()
        self.model_preview_done = asyncio.Event()
        self.model_preview_count = 0
        self.raw_model_frames = 0
        self.model_preview_dropped = 0
        self.revision = 0
        self.selection = None
        self.display_mode = "whitebox"
        self.commands = OrderedDict()
        self.receipts = OrderedDict()
        self.preview_count = 0
        self.preview_dropped = 0
        self.dropped_before_model = 0
        self.preview_done = asyncio.Event()
        self.model_task = None
        self.record_source = None
        self.video_transport = None
        self.epoch = None
        self.pairs = {}
        self.pair_condition = asyncio.Condition()
        self.pair_high_water = 0
        super().__init__(_InitOnly(), **kwargs)

    async def _start(self, message):
        self.display_mode = message.get("render_mode", "whitebox")
        if self.display_mode not in ("whitebox", "model"):
            raise ValueError("Choose whitebox or model display.")
        self.selection = validate_configuration(
            dict(
                mode="model",
                prompt=message.get("prompt"),
                references=message.get("references", []),
            )
        )
        await super()._start(message)

    async def _status(self, state, **fields):
        await self.outgoing.put(
            msgspec.msgpack.encode(
                dict(
                    version=2,
                    type="render.status",
                    revision=self.revision,
                    mode=self.display_mode,
                    state=state,
                    model_epoch=self.epoch,
                    **fields,
                )
            )
        )

    async def send(self, wire):
        if not isinstance(wire, bytes) or len(wire) > 8 * 1024 * 1024 + 4096:
            raise ValueError("Invalid interactive message.")
        message = msgspec.msgpack.decode(wire)
        if not isinstance(message, dict):
            raise ValueError("Invalid interactive message.")
        if (
            message.get("type") == "event"
            and message.get("kind") == "display.configure"
        ):
            await self.configure(message.get("payload"))
        elif message.get("type") == "event" and message.get("kind") == "prompt":
            # Prompt and references are immutable for this model session.
            await self._status("rejected", code="PARAMETERS_LOCKED")
        elif message.get("type") == "event" and message.get("kind") == "heartbeat":
            self.room_touch_requested.set()
        elif self.video_transport and message.get("kind") == "playback_ack":
            return  # Unreliable video has no cumulative, blocking frame credit.
        else:
            await super().send(wire)

    async def configure(self, payload):
        request_id = payload.get("request_id") if isinstance(payload, dict) else None
        try:
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
                raise ValueError("Invalid request ID.")
            if self.ready is None or not self.ready.is_set() or self.ended:
                raise ValueError("Wait for the world to become ready.")
            digest = hashlib.sha256(msgspec.msgpack.encode(payload)).hexdigest()
            if request_id in self.commands:
                previous, revision = self.commands[request_id]
                if previous != digest:
                    raise ValueError("Request ID was reused with different content.")
                await self._status(
                    "accepted",
                    request_id=request_id,
                    accepted_revision=revision,
                    duplicate=True,
                )
                return
            if (
                type(payload.get("expected_revision")) is not int
                or payload["expected_revision"] != self.revision
            ):
                await self._status(
                    "rejected", request_id=request_id, code="REVISION_CONFLICT"
                )
                return
            if set(payload) - {
                "request_id",
                "expected_revision",
                "mode",
            } or payload.get("mode") not in ("whitebox", "model"):
                raise ValueError("Display changes only accept a mode.")
            self.revision += 1
            self.display_mode = payload["mode"]
            self.commands[request_id] = digest, self.revision
            while len(self.commands) > 128:
                self.commands.popitem(last=False)
            await self._status(
                "accepted", request_id=request_id, accepted_revision=self.revision
            )
        except ValueError:
            await self._status(
                "rejected", request_id=request_id, code="INVALID_CONFIGURATION"
            )

    async def _read_h3(self):
        # Parent setup provides engine, archive and codecs, but no model lease.
        if not self.browser_ui:
            raise ValueError(
                "Interactive output requires browser-owned UI configuration."
            )
        if self.config.room_owned and not getattr(self.recording, "raw_archive", False):
            # Live preview and model selection have independent frame clocks.
            # A subset of inter-predicted preview AUs is not a valid archive:
            # keep the existing paired codec separate from the live codec.
            self.record_source = H264PlaybackEncoder(
                self.config.spec,
                max_bitrate=self.config.source_max_bitrate,
                gop=self.config.playback_gop,
                capture_clock=self.config.capture_clock,
                capture_fps=self.capture_rate_hint(),
            )
            self.record_source.independent_frames = self.config.playback_gop == 1
        elif not hasattr(self.recording, "pair"):
            self.record_source = H264PlaybackEncoder(
                self.config.spec, max_bitrate=800_000
            )
        # Identical encoding for the two experimental transports; otherwise an
        # all-IDR vs inter-frame comparison would confound the network result.
        self.playback.independent_frames = (
            not self.config.room_owned or self.config.playback_gop == 1
        )
        from .session_encoding import source_gop

        self.source_playback.independent_frames = (
            not self.config.room_owned or source_gop(self.config) == 1
        )
        self.recording.manifest.update(
            dataset_status="excluded_unvalidated",
            source_states={
                "path": "source-states.jsonl",
                "format": "opaque_source_envelope_v1",
            },
            rendering={
                "protocol": "whitebox-display-v1",
                "model_video": "continuous_model",
                "archive_source": "paired_with_model_output",
                "live_source": "independent_latest",
                "encoding_implementation": (
                    "room-independent-preview-paired-au-v1"
                    if self.config.room_owned
                    else "legacy"
                ),
                "superseded_source_frames": "not_recorded",
                "display_selection_field": "source-states.observed_input.display",
            },
        )
        if self.config.room_owned and not getattr(self.recording, "raw_archive", False):
            self.recording.manifest["video_streams"] = {
                "whitebox": self.record_source.stream,
                "model": self.playback.stream,
            }
            self.recording.manifest["live_source_video_stream"] = dict(
                self.source_playback.stream
            )
        await self.recording.configure()
        await self.recording.add_references(self.selection["references"])
        await self.outgoing.put(
            msgspec.msgpack.encode(
                dict(
                    version=2,
                    type="session.ready",
                    session_id=self.session_id,
                    source_stream=self.upstream.init["source_stream"],
                    play_seconds=self.config.play_seconds,
                    format="h264",
                    video_stream=self.playback.stream,
                    source_video_stream=self.source_playback.stream,
                    playback_flow_control="whitebox-dual-v2",
                    timeline="source_capture_vfr",
                    control_protocol="whitebox-display-v1",
                    continuous_model=True,
                    model_parameters_mutable=False,
                    independent_streams=True,
                    lossy_media=bool(self.video_transport),
                    media_transport=(
                        "webrtc-datachannel" if self.video_transport else "websocket"
                    ),
                    recording_id=self.recording.directory.name,
                    report_token=self.report.token if self.report else None,
                    **(
                        {"project_audio": self.engine.metadata["project_audio"]}
                        if getattr(self.engine, "metadata", {}).get("project_audio")
                        else {}
                    ),
                    engine=getattr(self.engine, "metadata", {}),
                    ui=public_ui_metadata(self.renderer.metadata),
                    multiplayer=self.recording.manifest.get("multiplayer"),
                    object_prefix=self.recording.manifest["object_prefix"],
                )
            )
        )
        self.ready.set()
        self._spawn(self._preview(), "latest-preview")
        if not self.config.room_owned:
            self._spawn(self._present_model(), "latest-model-preview")
        self.epoch = uuid4().hex
        self.model_task = asyncio.create_task(
            self._generation(0, copy.deepcopy(self.selection), self.epoch)
        )
        try:
            await self.model_task
            await self.render_queue.put(None)
        finally:
            self.model_task.cancel()
            await asyncio.gather(self.model_task, return_exceptions=True)

    async def _produce(self):
        await self.ready.wait()
        spec = self.config.spec
        iterator = self.engine.frames().__aiter__()
        while not self.ended:
            try:
                with stage("engine_frame_wait", self.trace_id):
                    frame = await anext(iterator)
            except StopAsyncIteration:
                break
            captured = time.perf_counter()
            if self.ended:
                break
            if (
                frame.index != self.submitted
                or frame.pts_us != spec.pts(frame.index)
                or len(frame.rgb) != spec.frame_bytes
            ):
                raise ValueError("Invalid source timeline.")
            self.submitted += 1
            observed = dict(
                frame.observed_input or {},
                latest=copy.deepcopy(self.observed_input),
                events=copy.deepcopy(self.input_events),
            )
            self.input_events.clear()
            # Receipt evidence may arrive on a superseded frame. Carry bounded
            # causal receipts forward so input latency still maps to displayed ticks.
            for receipt in observed.get("receipts", []):
                # UI actions have actionId, not a keyboard clientSequence.
                # Indexing all actions by (epoch, None) loses every receipt but
                # the last before it reaches reliable states or the archive.
                key = (
                    receipt.get("type", "input.receipt"),
                    receipt.get("epoch"),
                    (
                        receipt.get("actionId")
                        if receipt.get("type") == "action.receipt"
                        else receipt.get("clientSequence", receipt.get("sequence"))
                    ),
                )
                self.receipts[key] = copy.deepcopy(receipt)
            while len(self.receipts) > 256:
                self.receipts.popitem(last=False)
            observed["receipts"] = list(self.receipts.values())
            observed["display"] = dict(mode=self.display_mode, revision=self.revision)
            frame = replace(frame, observed_input=observed)
            if getattr(self, "source_only", False) and getattr(
                self.recording, "raw_archive", False
            ):
                self.recording.source_rgb(frame)
            state = dict(
                version=2,
                type="whitebox.states",
                source_frame_index=frame.index,
                pts_us=capture_pts(frame),
                source=frame.provenance,
                states=frame.states,
                observed_input=observed,
            )
            await self.source_state(state)
            await self.latest.put((frame, captured))
        await self.latest.close()
        self.ended = True
        self.finish_requested.set()

    async def source_state(self, state):
        await self.recording.write_source_state(state)
        # State histories remain reliable even when preview pixels are replaced.
        await self.outgoing.put(msgspec.msgpack.encode(state))

    async def _preview(self):
        cursor = 0
        retained = {}

        async def emit(outputs):
            for encoded_index, encoded in outputs:
                old_frame, old_captured = retained.pop(encoded_index)
                await self._media(
                    "whitebox", encoded_index, old_frame, encoded, old_captured
                )

        while (item := await self.latest.get(cursor)) is not None:
            index = self.preview_count
            # WS credits before encoding; subsequent capture only replaces latest.
            if self.video_transport is None and not self.config.room_owned:
                await self.source_credit.reserve(index)
            # Fetch again after credit; capture may have advanced while waiting.
            item = await self.latest.get(cursor)
            version, (frame, captured) = item
            self.preview_dropped += max(0, version - cursor - 1)
            cursor = version
            retained[index] = (frame, captured)
            with stage("source_preview_encode", self.trace_id, frame=index):
                outputs = await asyncio.to_thread(
                    self.source_playback.push, index, frame.rgb, capture_pts(frame)
                )
            self.preview_count += 1
            await emit(outputs)
        await emit(await asyncio.to_thread(self.source_playback.drain))
        if retained:
            raise ValueError("Source encoder did not return all preview frames")
        self.preview_done.set()

    async def _generation(self, revision, selection, epoch):
        cursor = self.latest.version  # Start from a newly captured source frame.
        self.model_wire_sequence = 0
        self.model_wire_lock = asyncio.Lock()
        self.source_send_lock = asyncio.Lock()
        refs = reference_descriptors(selection["references"])
        rendering = dict(mode=selection["mode"], revision=revision, model_epoch=epoch)
        await self._status("starting", effective_after_source_frame=self.submitted - 1)
        async with self.leases.open() as (ws, assignment):
            init = dict(
                self.upstream.init,
                prompt=selection["prompt"],
                references=selection["references"],
                source_demand_protocol="source-demand-v1",
            )
            geometry = self.model_geometry
            model_spec = geometry.model
            input_spec = geometry.ingress
            output_spec = geometry.output
            init = geometry.model_init(init)
            startup = self.leases if getattr(self.leases, "is_startup", False) else None
            from .model_inputs import ModelInputCapture

            recording = getattr(self, "recording", None)
            identity = (
                getattr(assignment, "session_id", None),
                getattr(assignment, "generation_id", None),
            )
            capture = self.model_inputs = (
                ModelInputCapture(recording, *identity)
                if recording is not None
                and all(isinstance(v, str) and v for v in identity)
                else None
            )
            if recording is not None and capture is None:
                summary = recording.manifest.setdefault("model_inputs", {})
                summary.update(
                    status="incomplete",
                    last_error="MODEL_INPUT_IDENTITY_UNAVAILABLE",
                    failed=summary.get("failed", 0) + 1,
                )
            init["record_model_inputs"] = capture is not None
            if capture is not None:
                capture.sent(getattr(startup, "model_input_init", None) or init)
            if startup is not None:
                ready = startup.ready
            else:
                try:
                    # A lost init write/result is uncertain even if this task
                    # is cancelled before any output proves condition use.
                    self.model_init_dispatched = True
                    await ws.send(msgspec.msgpack.encode(init))
                    ready = msgspec.msgpack.decode(
                        await asyncio.wait_for(ws.recv(), self.config.startup_seconds)
                    )
                except Exception as error:
                    raise model_preparation_error(
                        {"code": type(error).__name__, "content": str(error)},
                        trace_id=self.trace_id,
                        operation="init",
                    ) from error
                if not isinstance(ready, dict) or ready.get("type") == "error":
                    raise model_preparation_error(
                        ready, trace_id=self.trace_id, operation="init"
                    )
            if capture is not None:
                capture.received(ready)
            spec = self.config.spec
            if (
                ready.get("type") != "session.ready"
                or ready.get("source_stream") != init["source_stream"]
                or ready.get("max_source_message_bytes", 0)
                < input_spec.frame_bytes + 4096
            ):
                raise ValueError("Model source negotiation failed.")
            await self.model_protocol_ready(ws, assignment, ready)
            if getattr(self, "render_segment", False):
                cursor = self.latest.version
            # Respect negotiated and configured limits even on older workers.
            # Admission and playback have separate budgets. Preserve the default
            # two-clip overlap, but allow measured smaller admission windows.
            # Only older peers need four additional retained-tail slots.
            model_window = self.config.model_input_window(
                ready.get("source_output_release_protocol")
            )
            capacity = min(
                model_window,
                self.config.pending_frames,
                ready["max_pending_source_frames"],
                ready["max_pending_source_bytes"] // input_spec.frame_bytes,
            )
            # The opt-in 9-frame floor is only for a deployment verified to
            # produce its first AR block before the full 17-frame codec clip.
            # Output reservations separately cover both retained VFR AUs.
            if (
                self.config.model_subclip_admission
                and ready.get("source_output_release_protocol")
                != "valid-source-frames-v1"
            ):
                raise ValueError(
                    "Subclip admission requires proven source-frame release"
                )
            minimum_capacity = self.config.negotiated_minimum_model_input_frames(ready)
            if capacity < minimum_capacity:
                raise ValueError("Model source budget is too small.")
            # One reservation survives source encoding, H3, output queuing and
            # pairing. Conservative per-identity byte envelope includes raw RGB,
            # serialization/batch copies, both AUs and bounded opaque metadata.
            pair_bytes = 6 * model_spec.frame_bytes + 2 * 65536 + 1024 * 1024
            self.recording.manifest["model_geometry"] = geometry.receipt()
            overlap = self.config.room_owned and self.config.pipeline_overlap
            output_reserve = self.config.model_output_reserve_frames if overlap else 0
            pair_capacity = capacity
            if self.config.room_owned:
                capacity = min(
                    capacity,
                    self.config.pairing_max_bytes // pair_bytes - output_reserve,
                )
                if capacity < minimum_capacity:
                    raise ValueError("Pairing byte budget cannot hold a model clip.")
                pair_capacity = capacity + output_reserve
                self.recording.manifest["pairing"] = dict(
                    capacity=pair_capacity,
                    model_input_capacity=capacity,
                    minimum_input_capacity=minimum_capacity,
                    model_output_reserve_frames=output_reserve,
                    pipeline_overlap=overlap,
                    source_output_release_protocol=ready.get(
                        "source_output_release_protocol", "conservative-tail"
                    ),
                    **(
                        {"source_flow": ready["source_flow"]}
                        if "source_flow" in ready
                        else {}
                    ),
                    reserved_bytes_per_frame=pair_bytes,
                    max_bytes=self.config.pairing_max_bytes,
                    max_age_seconds=self.pairing_age_budget(),
                )
            pending = self.pairs if self.config.room_owned else {}
            condition = (
                self.pair_condition if self.config.room_owned else asyncio.Condition()
            )
            self.model_admission_condition = condition
            self.model_input_capacity = capacity
            self.model_pair_capacity = pair_capacity
            if getattr(self, "clip_switch", False) and (
                capacity < 17 or pair_capacity < capacity + output_reserve
            ):
                raise ValueError(
                    "The model input window cannot resume a condition switch"
                )
            if hasattr(self, "source_admission_ready"):
                self.source_admission_ready.set()
            if startup is not None:
                await startup.attach()
                for report in startup.timing_reports:
                    await self.model_control(report)
                startup.timing_reports.clear()
                self.model_wire_sequence = startup.event_sequence
            sent = accepted = consumed = received = demand_version = 0
            sampler = getattr(self.latest, "sampler", None)
            if sampler is not None:
                if ready.get("source_demand_protocol") == "source-demand-v1" and any(
                    ready.get(key) != getattr(assignment, key, None)
                    for key in ("session_id", "generation_id")
                ):
                    raise ValueError(
                        "Source demand identity does not match the model assignment"
                    )
                sampler.configure_native(
                    ready,
                    session_id=ready.get("session_id"),
                    generation_id=ready.get("generation_id"),
                    capacity=capacity,
                )
            submitted_timing = {}
            telemetry_received = 0
            ended = False
            # Two queued source RGBs plus the active encode. Their reservations
            # are already charged to pairs; no unbounded to_thread fan-out.
            raw_archive = getattr(self.recording, "raw_archive", False)
            source_encodes = (
                asyncio.Queue(maxsize=2) if overlap and not raw_archive else None
            )
            rendering.update(
                model_session_id=assignment.session_id,
                generation_id=assignment.generation_id,
                worker_id=assignment.denoiser.worker_id,
                model_revision=assignment.denoiser.model_revision,
            )
            await self._status(
                "active", effective_after_source_frame=self.submitted - 1
            )

            async def encode_sources():
                while True:
                    local, rgb, pts = await source_encodes.get()
                    try:
                        with stage(
                            "source_playback_encode", self.trace_id, frame=local
                        ):
                            outputs = await asyncio.to_thread(
                                self.record_source.push, local, rgb, pts
                            )
                        await self._store_archive_source_outputs(
                            outputs, pending, condition
                        )
                    finally:
                        source_encodes.task_done()

            async def source():
                nonlocal sent, cursor, ended
                deferred = None
                while True:
                    if deferred is not None:
                        from .source_sampling import wait_for_capture_or_progress

                        version, progress = deferred
                        started = time.perf_counter()
                        await wait_for_capture_or_progress(
                            self.latest,
                            version,
                            condition,
                            lambda: (received, demand_version),
                            progress,
                        )
                        observe_duration(
                            "source_cadence_wait",
                            (time.perf_counter() - started) * 1000,
                            self.trace_id,
                        )
                        deferred = None
                    blocked_ms = dict(
                        pair_capacity=0.0,
                        encoder_credit=0.0,
                        output_window=0.0,
                    )
                    if getattr(self, "clip_switch", False):
                        blocked_ms["condition_switch"] = 0.0
                    checked_at = time.perf_counter()
                    previously_blocked = ()

                    def capacity_available():
                        nonlocal checked_at, previously_blocked
                        stopped = getattr(self.latest, "stopped", None)
                        if stopped is not None and stopped.is_set():
                            return True  # EOS needs no new input credit or condition-ready receipt.
                        now = time.perf_counter()
                        for name in previously_blocked:
                            blocked_ms[name] = (
                                blocked_ms.get(name, 0.0) + (now - checked_at) * 1000
                            )
                        checked_at = now
                        previously_blocked = tuple(
                            name
                            for name, blocked in (
                                (
                                    "condition_switch",
                                    not self.model_source_allowed(sent),
                                ),
                                ("pair_capacity", len(pending) >= pair_capacity),
                                ("encoder_credit", sent - consumed >= capacity),
                                (
                                    "output_window",
                                    overlap and sent - received >= capacity,
                                ),
                            )
                            if blocked
                        )
                        return not previously_blocked

                    admission_wait_started = time.perf_counter()
                    async with condition:
                        capacity_ready = condition.wait_for(capacity_available)
                        if self.config.room_owned:
                            # Keep cancellation in the owner task. A ready child
                            # waiter can swallow concurrent cancel on Python 3.11.
                            async with asyncio.timeout(self.pairing_age_budget()):
                                await capacity_ready
                        else:
                            await capacity_ready
                    admission_wait_ms = (
                        time.perf_counter() - admission_wait_started
                    ) * 1000
                    observe_duration(
                        "input_admission_wait", admission_wait_ms, self.trace_id
                    )
                    for name, value in blocked_ms.items():
                        observe_duration("input_blocked_" + name, value, self.trace_id)
                    resume_after = max(cursor, getattr(self, "source_resume_after", 0))
                    self.dropped_before_model += resume_after - cursor
                    cursor = resume_after
                    sampler = getattr(self.latest, "sampler", None)
                    if sampler is not None:
                        now = time.perf_counter()
                        oldest = (
                            pending[next(iter(submitted_timing))][1]
                            if submitted_timing
                            else None
                        )
                        sampler.observe_queue(
                            now,
                            sent=sent,
                            buffered=sent - received,
                            reserve=minimum_capacity,
                            capacity=capacity,
                            oldest_age=now - oldest if oldest is not None else None,
                            period=sampler.period_us / 1_000_000
                            if sampler.period_us is not None
                            else None,
                            enabled=received > 0
                            and getattr(self, "pending_prompt", None) is None
                            and getattr(self, "condition_switch", None) is None,
                        )
                        sampler.supply(
                            sent - received,
                            minimum_capacity,
                            available=self.latest.has_capture(cursor),
                        )
                    item = await self.latest.get(cursor)
                    if item is None:
                        break
                    async with self.source_send_lock:
                        stopped = getattr(self.latest, "stopped", None)
                        if stopped is not None and stopped.is_set():
                            break
                        if not self.model_source_allowed(sent) or item[0] <= getattr(
                            self, "source_resume_after", 0
                        ):
                            continue
                        selector = getattr(self.latest, "select_model_source", None)
                        if selector is not None:
                            selected = selector(
                                item,
                                after=cursor,
                                now=time.perf_counter(),
                                buffered=sent - received,
                                reserve=minimum_capacity,
                                startup=received == 0,
                                urgent=getattr(self, "pending_prompt", None) is not None
                                or getattr(self, "condition_switch", None) is not None
                                or (sampler is not None and sampler.needs_input(sent)),
                            )
                            if sampler is not None:
                                self.recording.manifest["source_sampling"] = (
                                    sampler.snapshot()
                                )
                            if selected is None:
                                deferred = (item[0], (received, demand_version))
                                # The next wait occurs outside source_send_lock;
                                # control dispatch remains independent of captures.
                                continue
                            item = selected
                        version, (frame, captured) = item
                        self.dropped_before_model += max(0, version - cursor - 1)
                        cursor = version
                        self.last_selected_capture_version = version
                        observe_duration(
                            "source_age_on_submit",
                            (time.perf_counter() - captured) * 1000,
                            self.trace_id,
                        )
                        local = sent
                        selected_at = time.perf_counter()
                        await self.before_model_source(ws, frame, local)
                        frame = self.model_presentation_frame(frame, local)
                        if self.config.room_owned:
                            frame = replace(
                                frame,
                                provenance=dict(
                                    frame.provenance or {},
                                    model_source_frame=local,
                                    model_epoch=epoch,
                                ),
                            )
                            metadata = replace(frame, rgb=b"")
                            if len(msgspec.msgpack.encode(metadata)) > 1024 * 1024:
                                raise ValueError(
                                    "Selected source metadata exceeds its byte budget."
                                )
                            # This codec owns only selected archive sources. Live
                            # preview never waits for this reservation or model credit.
                            async with condition:
                                pending[local] = metadata, captured, None
                                condition.notify_all()
                            self.pair_high_water = max(
                                self.pair_high_water, len(pending)
                            )
                            if raw_archive:
                                self.recording.selected_source_rgb(local, frame)
                            elif source_encodes is not None:
                                await source_encodes.put(
                                    (local, frame.rgb, presentation_pts(frame))
                                )
                            else:
                                with stage(
                                    "source_playback_encode", self.trace_id, frame=local
                                ):
                                    outputs = await asyncio.to_thread(
                                        self.record_source.push,
                                        local,
                                        frame.rgb,
                                        presentation_pts(frame),
                                    )
                                await self._store_archive_source_outputs(
                                    outputs, pending, condition
                                )
                        else:
                            pending[local] = frame, captured
                        sent += 1
                        submitted_timing[local] = (
                            selected_at,
                            time.perf_counter(),
                            admission_wait_ms,
                            blocked_ms,
                        )
                        self.model_source_dispatch(local, time.perf_counter())
                        await self.send_model_event(
                            ws,
                            "source.append",
                            dict(
                                client_seq=local,
                                source_frame_start=local,
                                num_frames=1,
                                pts_us=spec.pts(local),
                                data=geometry.pad_rgb(frame.rgb),
                            ),
                        )
                        if startup is not None and local == 0:
                            # Only this model's pinned input is released. CLI
                            # capture and the shared world clock never stopped.
                            startup.source_started()
                if self.config.room_owned and not raw_archive:
                    if source_encodes is not None:
                        await source_encodes.join()
                    await self._store_archive_source_outputs(
                        await asyncio.to_thread(self.record_source.drain),
                        pending,
                        condition,
                    )
                ended = True
                self.model_source_ended = True
                await self.send_model_event(
                    ws, "source.end", dict(source_frame_end=sent)
                )

            async def outputs():
                nonlocal \
                    accepted, \
                    consumed, \
                    received, \
                    telemetry_received, \
                    demand_version
                while True:
                    try:
                        message = msgspec.msgpack.decode(await ws.recv())
                    except Exception as error:
                        raise model_error(
                            {"code": type(error).__name__, "content": str(error)},
                            trace_id=self.trace_id,
                            operation="render.receive",
                        ) from error
                    if capture is not None and isinstance(message, dict):
                        capture.received(message)
                    if not isinstance(message, dict) or message.get("type") == "error":
                        raise model_error(
                            message, trace_id=self.trace_id, operation="render.receive"
                        )
                    kind = message.get("type")
                    if await self.model_control(message):
                        continue
                    if kind == "source.accepted":
                        if (
                            message["source_frame_start"] != accepted
                            or not accepted < message["source_frame_end"] <= sent
                        ):
                            raise ValueError("Invalid source acknowledgement.")
                        accepted = message["source_frame_end"]
                    elif kind == "source.demand":
                        if ready.get(
                            "source_demand_protocol"
                        ) != "source-demand-v1" or any(
                            message.get(k) != ready.get(k)
                            for k in ("session_id", "generation_id")
                        ):
                            raise ValueError("Unexpected source demand identity")
                        sampler = getattr(self.latest, "sampler", None)
                        if sampler is not None:
                            sampler.model_demand(
                                message, sent=sent, now=time.perf_counter()
                            )
                        async with condition:
                            demand_version += 1
                            condition.notify_all()
                    elif kind == "source.consumed":
                        end = message["source_frame_end"]
                        if (
                            not consumed <= end <= accepted
                            or message["source_bytes_end"]
                            != end * input_spec.frame_bytes
                        ):
                            raise ValueError("Invalid consumption acknowledgement.")
                        async with condition:
                            consumed = end
                            condition.notify_all()
                    elif kind in ("frame_batch", "frame_batch_header"):
                        if kind == "frame_batch_header":
                            message["payload"] = await ws.recv()
                        start, count = (
                            message["output_frame_start"],
                            message["num_frames"],
                        )
                        if (
                            start != received
                            or count < 1
                            or start + count > accepted
                            or message.get("format") != "rgb24"
                            or message.get("encoding") not in (None, "raw")
                            or len(message["payload"]) != count * model_spec.frame_bytes
                            or (message.get("width"), message.get("height"))
                            != (model_spec.width, model_spec.height)
                            or message.get("pts_us") != spec.pts(start)
                        ):
                            raise ValueError("Invalid model output timeline.")
                        end = start
                        for span in message["source_alignment"]:
                            if (
                                span["source_frame_start"] != end
                                or span["output_frame_start"] != end
                                or span["num_frames"] < 1
                            ):
                                raise ValueError("Invalid model source alignment.")
                            end += span["num_frames"]
                        if end != start + count:
                            raise ValueError("Incomplete model source alignment.")
                        # Source metadata/UI and all downstream encoders stay native.
                        message["payload"] = geometry.crop_rgb(
                            message["payload"], num_frames=count
                        )
                        output_received = time.perf_counter()
                        input_at_completion = sent - received
                        (
                            output_prompt,
                            output_rendering,
                        ) = await self.model_output_condition(
                            message, pending, selection, rendering
                        )
                        output_refs = self.model_output_references(message, refs)
                        if capture is not None:
                            capture.applied(
                                message,
                                output_prompt,
                                output_refs,
                                output_rendering.get(
                                    "prompt_version",
                                    1 if message.get("event_id") is None else None,
                                ),
                            )
                        self.raw_model_frames += count
                        for local in range(start, start + count):
                            if self.config.room_owned:
                                frame, captured, _ = pending[local]
                            else:
                                frame, captured = pending.pop(local)
                            source_to_output_ms = (output_received - captured) * 1000
                            (
                                selected_at,
                                dispatched_at,
                                admission_wait_ms,
                                blocked_ms,
                            ) = submitted_timing.pop(local)
                            observe_duration(
                                "h3_source_to_output",
                                source_to_output_ms,
                                self.trace_id,
                            )
                            frame = replace(
                                frame,
                                provenance=dict(
                                    frame.provenance or {},
                                    model_source_frame=local,
                                    model_epoch=epoch,
                                    model_timing={
                                        "input_admission_wait_ms": admission_wait_ms,
                                        # Gates can overlap. These values must not be summed.
                                        "input_capacity_blocked_ms": blocked_ms,
                                        # Preserve the upstream WS packet boundary;
                                        # Room may later emit one preview per frame.
                                        "output_batch_start": start,
                                        "output_batch_frames": count,
                                        "source_to_raw_output_ms": source_to_output_ms,
                                        "source_to_selection_ms": (
                                            selected_at - captured
                                        )
                                        * 1000,
                                        "selection_to_dispatch_ms": (
                                            dispatched_at - selected_at
                                        )
                                        * 1000,
                                        "dispatch_to_raw_output_ms": (
                                            output_received - dispatched_at
                                        )
                                        * 1000,
                                    },
                                ),
                            )
                            offset = (local - start) * output_spec.frame_bytes
                            await self.render_queue.put(
                                (
                                    revision,
                                    frame,
                                    message["payload"][
                                        offset : offset + output_spec.frame_bytes
                                    ],
                                    captured,
                                    output_prompt,
                                    output_refs,
                                    output_rendering,
                                )
                            )
                            # RGB is now owned by the bounded output path. Model
                            # admission no longer waits for this frame's H.264 AU.
                            async with condition:
                                received = local + 1
                                condition.notify_all()
                        sampler = getattr(self.latest, "sampler", None)
                        if sampler is not None:
                            sampler.completed(
                                received,
                                output_received,
                                buffered=input_at_completion,
                                reserve=minimum_capacity,
                                learning_allowed=not ended
                                and getattr(self, "pending_prompt", None) is None
                                and getattr(self, "condition_switch", None) is None,
                            )
                            self.recording.manifest["source_sampling"] = (
                                sampler.snapshot()
                            )
                    elif kind == "chunk_telemetry":
                        durations = numeric_durations(message, CHUNK_FIELDS)
                        frames = message.get("num_frames")
                        if (
                            type(frames) is int
                            and 0 < frames <= 128
                            and received - frames == telemetry_received
                        ):
                            frame_breakdown = sanitize_frame_stage_breakdown(
                                message.get("frame_stage_breakdown"),
                                start_frame=telemetry_received,
                                num_frames=frames,
                            )
                            # Diagnostic-only sideband: never delay already decoded
                            # pixels while waiting for chunk timings or a subscriber.
                            timing = dict(
                                type="whitebox.model_timing",
                                model_epoch=epoch,
                                source_frame_start=telemetry_received,
                                source_frame_end=received,
                                chunk_index=message.get("chunk_index"),
                                durations=durations,
                                **(
                                    {"frame_stage_breakdown": frame_breakdown}
                                    if frame_breakdown
                                    else {}
                                ),
                            )
                            if frame_breakdown:
                                retained_breakdown = self.recording.manifest.setdefault(
                                    "frame_stage_breakdown", []
                                )
                                if (
                                    len(retained_breakdown) + len(frame_breakdown)
                                    <= 4096
                                ):
                                    retained_breakdown.extend(frame_breakdown)
                            try:
                                self.outgoing.put_nowait(msgspec.msgpack.encode(timing))
                            except asyncio.QueueFull:
                                pass
                        telemetry_received = received
                        for name, value in durations.items():
                            observe_duration(
                                "h3_telemetry_" + name,
                                value,
                                self.trace_id,
                                service="denoiser",
                            )
                    elif kind == "session.completed":
                        if not ended or received != sent:
                            raise ValueError("Model completed with missing frames.")
                        return

            writer, reader = (
                asyncio.create_task(source()),
                asyncio.create_task(outputs()),
            )
            keepalive = asyncio.create_task(self._model_heartbeat(ws))
            guard = (
                asyncio.create_task(self._guard_pairs())
                if self.config.room_owned
                else None
            )
            source_encoder = (
                asyncio.create_task(encode_sources())
                if source_encodes is not None
                else None
            )
            guards = (
                [keepalive]
                + ([guard] if guard else [])
                + ([source_encoder] if source_encoder else [])
            )
            tasks = [writer, reader, *guards]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                if reader in done:
                    # Valid EOS wins over a simultaneous heartbeat send seeing
                    # the peer's normal close. Reader validates all frame counts.
                    await reader
                    return
                for task in done:
                    task.result()
                if not reader.done():
                    done, _ = await asyncio.wait(
                        (reader, *guards), return_when=asyncio.FIRST_COMPLETED
                    )
                    if reader in done:
                        await reader
                        return
                    for task in done:
                        task.result()
                await reader
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if capture is not None:
                    capture.close()

    def model_source_allowed(self, sent):
        return True

    def model_output_references(self, message, references):
        return references

    async def _store_archive_source_outputs(self, outputs, pending, condition):
        async with condition:
            # Archive fail-open lets live model delivery retire its pairs before
            # a queued source encode completes. Late AUs have no archive owner;
            # never recreate that reservation or fail the still-live session.
            if not self.recording.accepting:
                condition.notify_all()
                return
            for encoded_index, source_au in outputs:
                if len(source_au["payload"]) > self.config.spec.frame_bytes + 65536:
                    raise ValueError("Source access unit exceeds its byte budget.")
                metadata, captured, _ = pending[encoded_index]
                pending[encoded_index] = (
                    metadata,
                    captured,
                    source_au if self.recording.accepting else None,
                )
            condition.notify_all()

    async def _model_heartbeat(self, ws):
        # A full source window is normal while the first GPU batch warms up.
        # Keep transport liveness separate from source/output flow control;
        # existing startup, pairing and drain deadlines still bound the work.
        while True:
            await asyncio.sleep(MODEL_HEARTBEAT_SECONDS)
            await self.send_model_event(ws, "heartbeat", {})

    async def send_model_event(self, ws, kind, payload, *, before_send=None):
        async with self.model_wire_lock:
            self.model_wire_sequence += 1
            event_id = self.model_wire_sequence
            if before_send is not None:
                receipt = before_send(event_id)
                if inspect.isawaitable(receipt):
                    await receipt
            message = dict(type="event", kind=kind, event_id=event_id, payload=payload)
            capture = getattr(self, "model_inputs", None)
            if capture is not None:
                capture.sent(message)
            wire = msgspec.msgpack.encode(message)
            try:
                await ws.send(wire)
            except Exception as error:
                raise model_error(
                    {"code": type(error).__name__, "content": str(error)},
                    trace_id=self.trace_id,
                    operation=kind,
                ) from error
            return event_id

    def model_presentation_frame(self, frame, local):
        return frame

    async def model_protocol_ready(self, ws, assignment, ready):
        pass

    async def before_model_source(self, ws, frame, local):
        pass

    def model_source_dispatch(self, local, dispatched_at):
        pass

    async def model_control(self, message):
        return False

    async def model_output_condition(self, message, pending, selection, rendering):
        return selection["prompt"], rendering

    def pairing_age_budget(self, captured=None):
        return self.config.pairing_max_age_seconds

    async def _guard_pairs(self):
        # Covers silence from either producer as well as a full input window.
        # Archive failure releases only archive AUs; H3 identities remain intact.
        while True:
            async with self.pair_condition:
                if not self.recording.accepting:
                    for index, (frame, captured, _) in self.pairs.items():
                        self.pairs[index] = frame, captured, None
                oldest = next(iter(self.pairs.values()))[1] if self.pairs else None
                remaining = self.pairing_age_budget(oldest)
                if self.pairs:
                    remaining -= time.perf_counter() - oldest
                    if remaining <= 0:
                        raise TimeoutError(
                            "Selected source exceeded the pairing age budget."
                        )
                try:
                    # Keep notification and cancellation in this task. On Python
                    # 3.11 wait_for can consume shutdown cancellation when its
                    # separate condition waiter completes at the same instant.
                    async with asyncio.timeout(min(remaining, 0.25)):
                        await self.pair_condition.wait()
                except asyncio.TimeoutError:
                    pass

    async def _media(
        self, stream, index, frame, encoded, captured, recording_index=None
    ):
        media_spec = (
            self.model_geometry.output if stream == "model" else self.config.spec
        )
        message = dict(
            version=2,
            type="whitebox.frame" if stream == "whitebox" else "frame_batch",
            stream=stream,
            output_frame_start=index,
            num_frames=1,
            pts_us=presentation_pts(frame) if stream == "model" else capture_pts(frame),
            source_frame_index=frame.index,
            source_capture_mono_ms=captured * 1000,
            recording_frame_index=recording_index,
            model_pts_us=(
                self.config.spec.pts(recording_index)
                if self.config.room_owned and recording_index is not None
                else frame.pts_us
            ),
            source=frame.provenance,
            observed_input=frame.observed_input,
            states=frame.states,
            rendering=frame.rendering,
            width=media_spec.width,
            height=media_spec.height,
            **encoded,
            server_epoch_ms=time.time() * 1000,
            pipeline_ms=(time.perf_counter() - captured) * 1000,
        )
        if self.video_transport:
            sent = await self.video_transport.send(message)
            if sent:
                self.media_delivered()
        else:
            await self.outgoing.put(msgspec.msgpack.encode(message))

    async def _present_model(self):
        cursor = 0
        while await self.model_latest.get(cursor) is not None:
            index = self.model_preview_count
            if self.video_transport is None:
                with stage("playback_delivery_wait", self.trace_id, frame=index):
                    await self.playback_credit.reserve(index)
            (
                version,
                (recording_index, frame, encoded, captured),
            ) = await self.model_latest.get(cursor)
            self.model_preview_dropped += max(0, version - cursor - 1)
            cursor = version
            await self._media("model", index, frame, encoded, captured, recording_index)
            observe_duration(
                "model_source_to_delivery",
                (time.perf_counter() - captured) * 1000,
                self.trace_id,
            )
            self.model_preview_count += 1
        self.model_preview_done.set()

    async def _deliver_model_frame(self, index, frame, encoded, captured, prompt):
        if index != self.received:
            raise ValueError("Model encoder output skipped a selected source frame")
        archived = replace(
            frame,
            index=index,
            pts_us=(
                self.config.spec.pts(index) if self.config.room_owned else frame.pts_us
            ),
            provenance=dict(
                frame.provenance or {},
                capture_pts_us=capture_pts(frame),
                source_frame_index=frame.index,
            ),
        )
        if getattr(self.recording, "raw_archive", False):
            pass  # Archive RGB was dispatched independently before live encoding.
        elif hasattr(self.recording, "pair"):
            if len(encoded["payload"]) > self.model_geometry.output.frame_bytes + 65536:
                raise ValueError("Model access unit exceeds its byte budget.")
            # One atomic offer contains both existing AUs, never source RGB.
            async with self.pair_condition:
                # Shutdown must cancel the render owner even when the source AU
                # arrives in the same loop turn. A separate wait_for task can
                # consume that cancellation on Python 3.11 and strand cleanup.
                async with asyncio.timeout(self.pairing_age_budget()):
                    await self.pair_condition.wait_for(
                        lambda: (
                            self.pairs[index][2] is not None
                            or not self.recording.accepting
                        )
                    )
            self.recording.pair(archived, self.pairs[index][2], encoded, prompt)
        else:
            with stage("archive_source_encode", self.trace_id, frame=index):
                source = await asyncio.to_thread(
                    self.record_source.encode, index, frame.rgb
                )
            await self.recording.encoded_video(
                index, source, name="whitebox", pts_us=presentation_pts(frame)
            )
            await self.recording.encoded_video(
                index, encoded, name="model", pts_us=presentation_pts(frame)
            )
            await self.recording.write_metadata(archived, prompt)
        # Archive every model output, while live delivery keeps only the
        # latest independently decodable frame. Browser ACKs never gate H3.
        if self.config.room_owned:
            # The Room pump immediately offers encoded frames to bounded
            # subscribers; preserve codec dependencies up to that boundary.
            await self._media(
                "model", self.model_preview_count, frame, encoded, captured, index
            )
            self.model_preview_count += 1
        else:
            await self.model_latest.put((index, frame, encoded, captured))
        self.received += 1
        if self.config.room_owned:
            async with self.pair_condition:
                del self.pairs[index]
                self.pair_condition.notify_all()

    async def _render(self):
        await self.ready.wait()
        last_pts = -1
        inputs = 0
        retained = {}

        async def emit(outputs):
            for index, encoded in outputs:
                frame, captured, prompt = retained.pop(index)
                await self._deliver_model_frame(index, frame, encoded, captured, prompt)

        while (item := await self.render_queue.get()) is not None:
            revision, frame, rgb, captured, prompt, refs, rendering = item
            if capture_pts(frame) <= last_pts:
                if self.config.room_owned:
                    raise ValueError("Selected source capture time moved backwards.")
                continue
            index = inputs
            if (
                self.config.room_owned
                and frame.provenance["model_source_frame"] != index
            ):
                raise ValueError("Model output skipped a selected source frame.")
            frame = replace(frame, rendering=copy.deepcopy(rendering), references=refs)
            if getattr(self.recording, "raw_archive", False):
                self.recording.model_rgb(index, frame, rgb, prompt)
            retained[index] = (frame, captured, prompt)
            hint = None
            if self.config.immediate_capture and self.config.room_owned:
                # Next selected SOURCE timing normally predates the current RGB.
                # This never waits for the next model RGB or a future input.
                following = self.pairs.get(index + 1)
                if following is not None:
                    hint = presentation_pts(following[0]) - presentation_pts(frame)
            with stage("playback_encode", self.trace_id, frame=index):
                outputs = await asyncio.to_thread(
                    self.playback.push,
                    index,
                    rgb,
                    presentation_pts(frame),
                    **(
                        {"duration_hint_us": hint}
                        if self.config.immediate_capture
                        else {}
                    ),
                )
            inputs += 1
            last_pts = capture_pts(frame)
            await emit(outputs)
        await emit(await asyncio.to_thread(self.playback.drain))
        if retained or inputs != self.received:
            raise ValueError("Model encoder did not return all selected frames")
        if self.config.room_owned:
            self.model_preview_done.set()
        await self.model_latest.close()
        await asyncio.wait_for(
            asyncio.gather(self.preview_done.wait(), self.model_preview_done.wait()),
            self.config.drain_seconds,
        )
        self.recording.manifest["rendering"]["source_frames_captured"] = self.submitted
        self.recording.manifest["rendering"]["live_source_frames"] = self.preview_count
        self.recording.manifest["rendering"]["live_source_superseded_frames"] = (
            self.preview_dropped
        )
        self.recording.manifest["rendering"][
            "source_frames_superseded_before_model"
        ] = self.dropped_before_model
        self.recording.manifest["rendering"]["transport_dropped_frames"] = (
            self.video_transport.dropped if self.video_transport else 0
        )
        self.recording.manifest["rendering"]["live_model_superseded_frames"] = (
            self.model_preview_dropped
        )
        await self.recording.configure()
        if self.config.room_owned:
            self.recording.manifest["pairing"]["high_water_frames"] = (
                self.pair_high_water
            )
            await self.recording.configure()
        self.outcome = "completed"
        self.finished.set()
        await self.outgoing.put(
            msgspec.msgpack.encode(
                dict(
                    version=2,
                    type="session.completed",
                    frames=self.model_preview_count,
                    recorded_frames=(
                        self.recording.last_frame + 1
                        if hasattr(self.recording, "pair")
                        else self.received
                    ),
                    recording_pending=hasattr(self.recording, "pair"),
                    source_frames=self.preview_count,
                )
            )
        )

    async def close(self):
        try:
            await super().close()
        finally:
            self.pairs.clear()
            if self.record_source:
                await asyncio.to_thread(self.record_source.close)
