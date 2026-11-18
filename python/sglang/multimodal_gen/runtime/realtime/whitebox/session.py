# SPDX-License-Identifier: Apache-2.0
"""A gateway upstream adapter: engine -> H3 source -> UI -> browser.

The existing gateway owns admission, leases, transport and authorization. This
adapter owns only one world's media pipeline. H3 remains the main-branch API.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import inspect
import json
import logging
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import msgspec

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import PROTOCOL_VERSION

from .contracts import EngineFactory, EngineFrame, VideoSpec, capture_pts
from .metrics import observe_duration, stage
from .model_geometry import ModelGeometry, classic_source_spec, resolution_spec
from .playback import (
    H264PlaybackEncoder,
    capture_rate_hint,
    create_source_playback_encoder,
    level31_supports,
)
from .playback_credit import PlaybackCredit
from .project_audio import public_ui_metadata
from .recording import ArchiveManager, S3ArchiveStore
from .telemetry import CHUNK_FIELDS, numeric_durations

logger = logging.getLogger(__name__)


def load_plugin(name: str):
    module, separator, attribute = name.partition(":")
    if not separator:
        raise ValueError("Plugin must be configured as module:attribute")
    return getattr(importlib.import_module(module), attribute)


@dataclass(frozen=True)
class WhiteboxConfig:
    engine_factory: EngineFactory
    ui_factory: Any
    spool: Path
    spec: VideoSpec = VideoSpec()
    model_target: VideoSpec | None = None
    startup_seconds: float = 90
    play_seconds: float = 180
    drain_seconds: float = 30
    pending_frames: int = 96
    playback_max_bitrate: int = 1_500_000
    source_max_bitrate: int = 800_000
    independent_archive: bool = True
    archive_source_bitrate: int = 4_000_000
    archive_paired_source_bitrate: int = 4_000_000
    archive_model_bitrate: int = 4_000_000
    archive_vbv_ms: int = 2000
    archive_keyframe_ms: int = 1000
    pairing_max_bytes: int = 384 * 1024 * 1024
    pairing_max_age_seconds: float = 90
    max_message_bytes: int = 64 * 1024 * 1024
    num_inference_steps: int = 4
    object_prefix: str = "whitebox"
    model_metadata: dict[str, Any] = field(default_factory=dict)
    room_owned: bool = False
    playback_gop: int = 1
    capture_clock: bool = False
    immediate_capture: bool = False
    source_intra_refresh: bool = False
    source_encoding: dict | None = None
    model_input_frames: int = 34
    pipeline_overlap: bool = False
    independent_prompt_control: bool = True
    model_output_reserve_frames: int = 4
    model_subclip_admission: bool = False
    rtc_model_cadence: bool = False
    model_source_sampling: str = "latest"

    @property
    def archive_encoding(self):
        return dict(
            source_bitrate=self.archive_source_bitrate,
            paired_source_bitrate=self.archive_paired_source_bitrate,
            model_bitrate=self.archive_model_bitrate,
            vbv_ms=self.archive_vbv_ms,
            keyframe_ms=self.archive_keyframe_ms,
            # Negotiation can admit four extra source frames for legacy peers.
            # The independent archive codec may also retain one model frame
            # after live playback has returned its reservation. This is a hard
            # limit, not a preallocation; the archive byte/age limits still apply.
            pair_frames=self.model_input_window()
            + self.model_output_reserve_frames
            + 1,
            pair_age=max(120, self.pairing_max_age_seconds),
        )

    def model_input_window(self, source_release_protocol=None):
        return self.model_input_frames + (
            0 if source_release_protocol == "valid-source-frames-v1" else 4
        )

    @property
    def minimum_model_input_frames(self):
        return (
            9
            if self.model_subclip_admission
            else 17 + int(self.room_owned and self.capture_clock)
        )

    def negotiated_minimum_model_input_frames(self, ready):
        flow = ready.get("source_flow")
        if flow is None:
            return self.minimum_model_input_frames
        if (
            not isinstance(flow, dict)
            or flow.get("protocol") != "clip17-source-flow-v1"
            or flow.get("input_groups") != [1, 8, 8]
            or type(flow.get("first_output_input_frames")) is not int
            or not 1 <= flow["first_output_input_frames"] <= 120
            or type(flow.get("progressive_output")) is not bool
            or ready.get("source_output_release_protocol") != "valid-source-frames-v1"
        ):
            raise ValueError("Invalid negotiated source flow")
        return flow["first_output_input_frames"]

    def __post_init__(self):
        if self.model_source_sampling not in ("latest", "adaptive"):
            raise ValueError("Unknown model source sampling policy")
        if self.model_target is not None:
            ModelGeometry(self.spec, self.model_target)
        if type(self.independent_archive) is not bool or any(
            type(rate) is not int or not 100_000 <= rate <= 50_000_000
            for rate in (
                self.archive_source_bitrate,
                self.archive_paired_source_bitrate,
                self.archive_model_bitrate,
            )
        ):
            raise ValueError("Invalid independent archive configuration")
        if any(
            type(value) is not int or not 250 <= value <= 5000
            for value in (self.archive_vbv_ms, self.archive_keyframe_ms)
        ):
            raise ValueError("Invalid archive encoding interval")
        if min(self.startup_seconds, self.play_seconds, self.drain_seconds) <= 0:
            raise ValueError("Whitebox deadlines must be positive")
        if (
            type(self.playback_max_bitrate) is not int
            or not 100_000 <= self.playback_max_bitrate <= 50_000_000
        ):
            raise ValueError("Invalid playback bitrate budget")
        if (
            type(self.source_max_bitrate) is not int
            or not 100_000 <= self.source_max_bitrate <= 50_000_000
        ):
            raise ValueError("Invalid source bitrate budget")
        if self.pairing_max_bytes <= 0 or not 0 < self.pairing_max_age_seconds <= 300:
            raise ValueError("Invalid pairing budget")
        if self.spec != VideoSpec(1280, 720) and (
            self.spec.fps != 24 or self.spec.width % 32 or self.spec.height % 32
        ):
            raise ValueError("H3 requires 24 FPS and dimensions divisible by 32")
        if min(self.spec.width, self.spec.height, self.pending_frames) <= 0:
            raise ValueError("Whitebox dimensions and pending frames must be positive")
        if type(self.playback_gop) is not int or not 1 <= self.playback_gop <= 24:
            raise ValueError("Playback GOP must be between 1 and 24 frames")
        if type(self.capture_clock) is not bool:
            raise ValueError("Capture clock must be boolean")
        if type(self.immediate_capture) is not bool or (
            self.immediate_capture and not (self.room_owned and self.capture_clock)
        ):
            raise ValueError("Immediate H.264 encoding requires a capture-clock Room")
        if self.source_encoding is not None:
            from .source_encoding import validate_source_encoding

            validate_source_encoding(self.source_encoding)
            if not (self.room_owned and self.capture_clock):
                raise ValueError("Custom source encoding requires a capture-clock Room")
        if type(self.source_intra_refresh) is not bool or (
            self.source_intra_refresh and not (self.room_owned and self.capture_clock)
        ):
            raise ValueError("Source intra refresh requires a capture-clock Room")
        if type(self.pipeline_overlap) is not bool:
            raise ValueError("Pipeline overlap must be boolean")
        if type(self.independent_prompt_control) is not bool:
            raise ValueError("Independent prompt control must be boolean")
        if type(self.model_subclip_admission) is not bool or (
            self.model_subclip_admission
            and not (self.room_owned and self.capture_clock and self.pipeline_overlap)
        ):
            raise ValueError(
                "Subclip admission requires an overlapping capture-clock Room pipeline"
            )
        if (
            type(self.model_output_reserve_frames) is not int
            or not 4 <= self.model_output_reserve_frames <= 17
        ):
            raise ValueError("Model output reserve must be between 4 and 17 frames")
        minimum_input_frames = self.minimum_model_input_frames
        if (
            type(self.model_input_frames) is not int
            or not minimum_input_frames <= self.model_input_frames <= 120
        ):
            raise ValueError(
                f"Model input window must be between {minimum_input_frames} and 120 frames"
            )
        if not isinstance(self.model_metadata, dict):
            raise ValueError("Whitebox model metadata must be an object")
        if type(
            self.num_inference_steps
        ) is not int or self.num_inference_steps not in (1, 2, 4):
            raise ValueError(
                "Whitebox supports one-step, two-step or four-step deployments"
            )


def from_environment(
    *, room=False
) -> tuple[WhiteboxConfig | None, ArchiveManager | None]:
    if not room and os.environ.get("WHITEBOX_ENABLED") != "1":
        return None, None
    engine = os.environ.get("WHITEBOX_ENGINE_FACTORY")
    ui = os.environ.get("WHITEBOX_UI_FACTORY")
    if room:
        engine = (
            engine
            or "sglang.multimodal_gen.runtime.realtime.whitebox.room_client:room_engine_factory"
        )
        ui = (
            ui
            or "sglang.multimodal_gen.runtime.realtime.whitebox.room_client:bound_room_renderer"
        )
    if not engine or not ui:
        raise ValueError("WHITEBOX_ENGINE_FACTORY and WHITEBOX_UI_FACTORY are required")
    sampling = os.environ.get("WHITEBOX_MODEL_SOURCE_SAMPLING", "latest")
    if sampling == "adaptive" and (
        not room or os.environ.get("WHITEBOX_RENDER_SEGMENTS") != "1"
    ):
        raise ValueError("Adaptive source sampling requires Room render segments")
    archive_encoding = os.environ.get("WHITEBOX_ARCHIVE_ENCODING", "independent-rgb-v2")
    if archive_encoding not in ("independent-rgb-v2", "playback-v1"):
        raise ValueError("Invalid archive encoding mode")
    if os.environ.get("WHITEBOX_ARCHIVE_RATE_CONTROL", "abr") != "abr":
        raise ValueError("Archive rate control must be ABR")
    source_intra_refresh = os.environ.get("WHITEBOX_SOURCE_INTRA_REFRESH", "0")
    if source_intra_refresh not in ("0", "1"):
        raise ValueError("Source intra refresh must be 0 or 1")
    target_width, target_height = (
        os.environ.get("WHITEBOX_MODEL_WIDTH"),
        os.environ.get("WHITEBOX_MODEL_HEIGHT"),
    )
    if (target_width is None) != (target_height is None):
        raise ValueError("Both model target dimensions must be configured")
    config = WhiteboxConfig(
        model_source_sampling=sampling,
        model_target=(
            VideoSpec(int(target_width), int(target_height))
            if target_width is not None
            else None
        ),
        room_owned=room,
        playback_gop=int(os.environ.get("WHITEBOX_PLAYBACK_GOP", "1")),
        capture_clock=os.environ.get("WHITEBOX_CAPTURE_CLOCK", "0") == "1",
        immediate_capture=os.environ.get("WHITEBOX_H264_IMMEDIATE", "0") == "1",
        source_intra_refresh=source_intra_refresh == "1",
        source_encoding=(
            json.loads(os.environ["WHITEBOX_SOURCE_ENCODING"])
            if os.environ.get("WHITEBOX_LOCAL_TEST") == "1"
            and os.environ.get("WHITEBOX_SOURCE_ENCODING")
            else None
        ),
        model_input_frames=int(os.environ.get("WHITEBOX_MODEL_INPUT_FRAMES", "34")),
        pipeline_overlap=os.environ.get("WHITEBOX_PIPELINE_OVERLAP", "0") == "1",
        model_subclip_admission=os.environ.get("WHITEBOX_MODEL_SUBCLIP_ADMISSION", "0")
        == "1",
        rtc_model_cadence=os.environ.get("WHITEBOX_RTC_MODEL_CADENCE", "0") == "1",
        model_output_reserve_frames=int(
            os.environ.get("WHITEBOX_MODEL_OUTPUT_RESERVE_FRAMES", "4")
        ),
        engine_factory=load_plugin(engine),
        ui_factory=load_plugin(ui),
        spool=Path(os.environ.get("WHITEBOX_SPOOL", "/var/lib/sglang/whitebox")),
        spec=VideoSpec(
            int(os.environ.get("WHITEBOX_WIDTH", "832")),
            int(os.environ.get("WHITEBOX_HEIGHT", "480")),
        ),
        startup_seconds=float(os.environ.get("WHITEBOX_STARTUP_SECONDS", "90")),
        play_seconds=float(os.environ.get("WHITEBOX_PLAY_SECONDS", "180")),
        drain_seconds=float(os.environ.get("WHITEBOX_DRAIN_SECONDS", "30")),
        object_prefix=os.environ.get("WHITEBOX_OSS_PREFIX", "whitebox"),
        pending_frames=int(os.environ.get("WHITEBOX_PENDING_FRAMES", "96")),
        playback_max_bitrate=int(
            os.environ.get("WHITEBOX_PLAYBACK_MAX_BITRATE", "1500000")
        ),
        source_max_bitrate=int(os.environ.get("WHITEBOX_SOURCE_MAX_BITRATE", "800000")),
        independent_archive=archive_encoding == "independent-rgb-v2",
        archive_source_bitrate=int(
            os.environ.get("WHITEBOX_ARCHIVE_SOURCE_BITRATE", "4000000")
        ),
        archive_paired_source_bitrate=int(
            os.environ.get("WHITEBOX_ARCHIVE_PAIRED_SOURCE_BITRATE", "4000000")
        ),
        archive_model_bitrate=int(
            os.environ.get("WHITEBOX_ARCHIVE_MODEL_BITRATE", "4000000")
        ),
        archive_vbv_ms=int(os.environ.get("WHITEBOX_ARCHIVE_VBV_BUFFER_MS", "2000")),
        archive_keyframe_ms=int(
            os.environ.get("WHITEBOX_ARCHIVE_KEYFRAME_INTERVAL_MS", "1000")
        ),
        pairing_max_bytes=int(
            os.environ.get("WHITEBOX_PAIRING_MAX_BYTES", str(384 * 1024 * 1024))
        ),
        pairing_max_age_seconds=float(
            os.environ.get("WHITEBOX_PAIRING_MAX_AGE_SECONDS", "90")
        ),
        max_message_bytes=int(
            os.environ.get("WHITEBOX_MAX_MESSAGE_BYTES", str(64 * 1024 * 1024))
        ),
        model_metadata=json.loads(os.environ.get("WHITEBOX_MODEL_METADATA", "{}")),
        num_inference_steps=int(os.environ.get("WHITEBOX_INFERENCE_STEPS", "4")),
    )
    if config.immediate_capture:
        # An incorrectly assembled image must fail readiness, not a user's first
        # video session after the world and model have already been allocated.
        from .native_capture import library

        library()
    bucket = os.environ.get("WHITEBOX_OSS_BUCKET")
    store = (
        S3ArchiveStore(
            bucket,
            os.environ["WHITEBOX_OSS_ENDPOINT"],
            os.environ.get("WHITEBOX_OSS_REGION", "cn-beijing"),
        )
        if bucket
        else None
    )
    manager = ArchiveManager
    options = {"processes": True}
    if room:
        from .stream_archive import StreamingArchiveManager

        manager, options = (
            StreamingArchiveManager,
            {
                "index_prefix": config.object_prefix,
                "rgb_budget_bytes": int(
                    os.environ.get("WHITEBOX_STREAM_ARCHIVE_RGB_BYTES", str(96 << 20))
                ),
            },
        )
    return config, manager(
        store,
        **options,
        recorder_slots=int(
            os.environ.get(
                "WHITEBOX_RECORDER_SLOTS",
                (
                    "3"
                    if room and os.environ.get("WHITEBOX_RENDER_SEGMENTS") == "1"
                    else "2"
                ),
            )
        ),
        queue_depth=int(
            os.environ.get(
                (
                    "WHITEBOX_STREAM_ARCHIVE_QUEUE_ITEMS"
                    if room
                    else "WHITEBOX_RECORDING_QUEUE_DEPTH"
                ),
                "256" if room else "24",
            )
        ),
        queue_bytes=int(
            os.environ.get(
                (
                    "WHITEBOX_STREAM_ARCHIVE_QUEUE_BYTES"
                    if room
                    else "WHITEBOX_RECORDING_QUEUE_BYTES"
                ),
                str((32 if room else 64) * 1024 * 1024),
            )
        ),
    )


class WhiteboxUpstream:
    supports_model_padding = False

    @property
    def model_geometry(self):
        return ModelGeometry(
            self.config.spec, getattr(self.config, "model_target", None)
        )

    def configure_resolution(self, message):
        """Freeze geometry before startup assets, model admission or CLI work."""
        world = message.get("whitebox", {})
        if not isinstance(world, dict):
            return  # The existing world validation owns this malformed init.
        resolver = (
            resolution_spec if self.supports_model_padding else classic_source_spec
        )
        if not self.supports_model_padding and self.config.model_target is not None:
            from .diagnostics import WhiteboxDiagnostic

            raise WhiteboxDiagnostic("RESOLUTION_UNSUPPORTED", "validation")
        selected = resolver(world.get("resolution"), self.config.spec)
        if selected != self.config.spec:
            self.config = replace(self.config, spec=selected)

    def capture_rate_hint(self, codec="h264"):
        """Separate source capture cadence from H.264's legacy SPS envelope."""
        spec = self.config.spec
        if codec == "h264" and level31_supports(spec, 60):
            return 60
        metadata = getattr(self.engine, "metadata", {})
        fps = metadata.get("capture_fps")
        if fps is None:
            from .cli_room import capture_fps

            fps = capture_fps(
                spec, os.environ, getattr(self, "requested_capture_fps", None)
            )
        if type(fps) is not int or not 1 <= fps <= 60:
            raise ValueError("CLI capture FPS must be between 1 and 60")
        hint = capture_rate_hint(spec, fps) if codec == "h264" else fps
        return hint

    def __init__(
        self,
        upstream,
        *,
        config: WhiteboxConfig,
        archives: ArchiveManager,
        trace_id: str,
        session_id: str,
        user_id: str = "anonymous",
        worker_metadata: dict | None = None,
        reports=None,
        report=None,
    ):
        self.upstream, self.config, self.archives = upstream, config, archives
        self.trace_id, self.session_id = trace_id, session_id
        self.user_id = user_id
        self.worker_metadata = worker_metadata or {}
        self.reports, self.report = reports, report
        self.engine = self.renderer = self.recording = None
        self.playback = self.source_playback = None
        self.browser_ui = False
        self.playback_credit = PlaybackCredit()
        self.source_credit = PlaybackCredit()
        self.ready = None
        self.finished, self.first_media = asyncio.Event(), asyncio.Event()
        self.finish_requested = asyncio.Event()
        self.room_touch_requested = asyncio.Event()
        self.changed = asyncio.Event()
        self.credit = asyncio.Condition()
        self.writer = asyncio.Lock()
        self.outgoing = asyncio.Queue(maxsize=2)
        self.render_queue = asyncio.Queue(maxsize=2)
        self.frames: dict[int, EngineFrame] = {}
        self.capture_times: dict[int, float] = {}
        self.tasks: set[asyncio.Task] = set()
        self.prompts = {0: ""}
        self.event_id = self.submitted = self.consumed = self.accepted = (
            self.received
        ) = 0
        self.input_sequence = -1
        from .source_sampling import InputChangeTracker

        self.sampling_inputs = InputChangeTracker()
        self.observed_input = {}
        self.input_events = []
        self.ended = self.closed = False
        self.drain_deadline_monotonic = None
        self.outcome = "aborted"
        self._spawn(self._deadline(), "deadline")
        self._spawn(self._room_heartbeat(), "room-heartbeat")

    def _spawn(self, coroutine, name):
        task = asyncio.create_task(coroutine, name="whitebox-" + name)
        self.tasks.add(task)
        self.changed.set()

        def done(task):
            self.changed.set()
            # recv() observes task errors as well as queued media. Keep finished
            # tasks until close so errors cannot disappear between receive calls.
            if not task.cancelled() and task.exception():
                self.outcome = "failed"
                logger.error(
                    "Whitebox stage failed: %s trace_id=%s",
                    name,
                    self.trace_id,
                    exc_info=task.exception(),
                )

        task.add_done_callback(done)

    async def _room_heartbeat(self):
        # A slow Room RPC must not hold up browser playback credits or inputs.
        # Coalesce heartbeats in one session-owned task; close() cancels it.
        while True:
            await self.room_touch_requested.wait()
            self.room_touch_requested.clear()
            touch = getattr(self.engine, "touch", None)
            if touch is not None and not self.finished.is_set():
                await asyncio.wait_for(touch(), 10.0)

    async def _event(self, kind, payload):
        async with self.writer:
            self.event_id += 1
            if kind == "prompt":
                if len(self.prompts) >= 256:
                    raise ValueError("Too many pending prompt updates")
                self.prompts[self.event_id] = payload
            await self.upstream.send(
                msgspec.msgpack.encode(
                    dict(
                        type="event", kind=kind, event_id=self.event_id, payload=payload
                    )
                )
            )

    def observe_client_metric(self, message):
        """Consume bounded diagnostics without entering the gameplay queue."""
        if (
            self.report
            and message.get("type") == "client_metric"
            and message.get("stage")
            in {"client_video_decode", "client_render_wait", "client_receive_queue"}
        ):
            self.report.observe(
                dict(
                    event="client." + message["stage"],
                    duration_ms=message.get("duration_ms"),
                )
            )

    async def send(self, wire):
        from .diagnostics import WhiteboxDiagnostic

        if not isinstance(wire, bytes) or len(wire) > 8 * 1024 * 1024 + 4096:
            raise WhiteboxDiagnostic("INVALID_CONTROL_MESSAGE", "validation")
        try:
            message = msgspec.msgpack.decode(wire)
        except msgspec.DecodeError as error:
            raise WhiteboxDiagnostic("INVALID_CONTROL_MESSAGE", "validation") from error
        if not isinstance(message, dict):
            raise WhiteboxDiagnostic("INVALID_CONTROL_MESSAGE", "validation")
        if message.get("type") in {"client_metric", "client_metric_batch"}:
            # The gateway observes these before forwarding. They are neither
            # engine controls nor H3 conditioning and must not alter the timeline.
            self.observe_client_metric(message)
            return
        if message.get("type") == "init":
            if self.ready is not None:
                raise WhiteboxDiagnostic("INVALID_CONTROL_MESSAGE", "validation")
            await self._start(message)
            return
        if self.ready is None:
            raise WhiteboxDiagnostic("INVALID_CONTROL_MESSAGE", "validation")
        kind, payload = message.get("kind"), message.get("payload")
        if message.get("type") == "event" and kind == "heartbeat":
            touch = getattr(self.engine, "touch", None)
            if touch is not None:
                self.room_touch_requested.set()
        if kind == "network.ping" and isinstance(payload, dict):
            nonce = payload.get("nonce")
            if type(nonce) is not int or not 0 <= nonce < 2**53:
                raise WhiteboxDiagnostic("INVALID_CONTROL_MESSAGE", "validation")
            await self.outgoing.put(
                msgspec.msgpack.encode(
                    dict(
                        version=PROTOCOL_VERSION,
                        type="whitebox.pong",
                        nonce=nonce,
                        server_epoch_ms=time.time() * 1000,
                    )
                )
            )
            return
        if not self.ready.is_set():
            if message.get("type") == "event" and kind == "heartbeat":
                # Keep draining the bounded browser receive queue during a cold
                # CLI download. H3 must still receive init as its first message.
                return
            raise WhiteboxDiagnostic("INVALID_CONTROL_MESSAGE", "validation")
        if kind == "input":
            sequence = message.get("sequence")
            if (
                type(sequence) is not int
                or sequence != self.input_sequence + 1
                or not isinstance(payload, dict)
                or len(wire) > 64 * 1024
            ):
                raise WhiteboxDiagnostic("INVALID_CONTROL_MESSAGE", "validation")
            self.input_sequence = sequence
            if not self.ended:
                self.sampling_inputs.observe(payload, sequence)
                if len(self.input_events) >= 1024:
                    raise WhiteboxDiagnostic("CONTROL_QUEUE_FULL", "control")
                self.input_events.append(
                    dict(sequence=sequence, payload=copy.deepcopy(payload))
                )
                self.observed_input = copy.deepcopy(payload)
                if self.browser_ui:
                    await self.outgoing.put(
                        msgspec.msgpack.encode(
                            dict(
                                version=PROTOCOL_VERSION,
                                type="whitebox.input_received",
                                sequence=sequence,
                                server_epoch_ms=time.time() * 1000,
                            )
                        )
                    )
                with stage("engine_input", self.trace_id):
                    await self.engine.send_input(sequence, payload)
                self.input_sequence = sequence
        elif kind == "mock.frame" and hasattr(self.engine, "receive_frame"):
            with stage("mock_capture_receive", self.trace_id):
                await self.engine.receive_frame(payload)
        elif kind == "prompt":
            if (
                not isinstance(payload, str)
                or not payload.strip()
                or len(payload) > 16384
            ):
                raise WhiteboxDiagnostic("INVALID_CONTROL_MESSAGE", "validation")
            if not self.ended:
                if hasattr(self.engine, "note"):
                    await self.engine.note("prompt", {"prompt": payload})
                await self._event("prompt", payload)
        elif kind in {"finish", "server_finish"}:
            await self.finish()
        elif (
            kind == "playback_ack"
            and isinstance(payload, dict)
            and payload.get("protocol") in {"whitebox-frame-v1", "whitebox-dual-v2"}
        ):
            if payload.get("protocol") == "whitebox-dual-v2" and payload.get(
                "stream"
            ) not in {"whitebox", "model"}:
                raise WhiteboxDiagnostic("INVALID_CONTROL_MESSAGE", "validation")
            credit = (
                self.source_credit
                if payload.get("stream") == "whitebox"
                else self.playback_credit
            )
            await credit.acknowledge(payload)
        elif kind in {"heartbeat", "playback_ack"}:
            # Heartbeats carry no source/prompt event ID and never alter input data.
            # H3 may close after source.end while the UI still drains its frames.
            # Room liveness was refreshed above; forwarding into that closed H3
            # socket would prematurely terminate the browser's video tail.
            if self.ended:
                return
            async with self.writer:
                await self.upstream.send(msgspec.msgpack.encode(message))
        else:
            raise WhiteboxDiagnostic("INVALID_CONTROL_MESSAGE", "validation")

    async def _start(self, message):
        from .diagnostics import cli_fps_override

        self.configure_resolution(message)
        world = message.get("whitebox")
        if isinstance(world, dict) and "source_encoding" in world:
            from .session_encoding import canonical_source_encoding

            if (
                not self.config.room_owned
                or canonical_source_encoding(world["source_encoding"])
                != self.config.source_encoding
            ):
                raise ValueError("Source encoding requires an authorized RTC source")
        if not isinstance(world, dict) or not isinstance(world.get("world_id"), str):
            raise ValueError("World ID is required")
        from .area_contract import area_source_options

        selected_source = area_source_options(world)
        if selected_source and not getattr(self, "source_only", False):
            raise ValueError("AREA_SOURCE_UNSUPPORTED")
        world_id = world["world_id"]
        cache_world_id = world_id
        cli_fps_override(world.get("cli_fps"))
        self.requested_capture_fps = world.get("cli_fps")
        if world.get("cli_fps") is not None and not hasattr(
            self.config.engine_factory, "open_session"
        ):
            from .diagnostics import WhiteboxDiagnostic

            raise WhiteboxDiagnostic("CLI_FPS_UNSUPPORTED", "validation")
        if not world_id or len(world_id) > 256:
            raise ValueError("Invalid world ID")
        if world.get("source_id") is not None:
            source_id = world["source_id"]
            if (
                not isinstance(source_id, str)
                or not source_id
                or len(source_id) > 128
                or "/" in source_id
                or "/" in world_id
            ):
                raise ValueError("Invalid world source")
            world_id = source_id + "/" + world_id
        if world.get("game_id") and not hasattr(
            self.config.engine_factory, "open_session"
        ):
            raise ValueError("Multiplayer requires a room runtime")
        prompt = message.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16384:
            raise ValueError("Prompt is required")
        references = message.get("references", [])
        if not isinstance(references, list) or len(references) > 9:
            raise ValueError("Invalid references")
        ids = set()
        for item in references:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("id"), str)
                or item["id"] in ids
                or item.get("format") not in {"png", "jpeg", "webp"}
                or not isinstance(item.get("payload"), bytes)
            ):
                raise ValueError("Invalid reference image")
            ids.add(item["id"])
        seed = message.get("seed", 42)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Invalid seed")
        self.ready = asyncio.Event()
        self.prompts[0] = prompt
        spec = self.config.spec
        init = dict(
            type="init",
            world_id=cache_world_id,
            game_id=world_id,
            cache_write=True,
            prompt=prompt,
            seed=seed,
            references=references,
            zing_realtime_api_version=1,
            size=f"{spec.width}x{spec.height}",
            fps=spec.fps,
            num_inference_steps=self.config.num_inference_steps,
            zing_realtime_decode_mode="stable",
            realtime_output_format="raw",
            source_message_budget_bytes=self.config.max_message_bytes,
            source_stream=dict(
                format="rgb24", width=spec.width, height=spec.height, fps=spec.fps
            ),
        )
        # A cold delivery can take minutes. Awaiting it in send() blocks browser
        # heartbeats and eventually even WebSocket ping/pong behind max_queue=1.
        # Track startup with the other owned tasks so failure and disconnect
        # still cancel it and release the admitted session.
        if (
            getattr(self, "startup", None) is not None
            and getattr(self, "expected_publication", None) is None
            and not getattr(self, "defer_startup_until_source", False)
        ):
            self.startup.launch(init)
        self._spawn(self._open_engine(world_id, world, init), "startup")

    async def _open_engine(self, world_id, world, init):
        from .area_contract import area_source_options

        spec = self.config.spec
        prompt, seed, references = init["prompt"], init["seed"], init["references"]
        if (
            getattr(self, "startup", None) is not None
            and getattr(self, "expected_publication", None) is None
            and not getattr(self, "defer_startup_until_source", False)
        ):
            await self.startup.wait(self.startup.admitted)
        # Startup includes engine launch, H3 conditioning, rendering and delivery.
        with stage("engine_start", self.trace_id):
            factory = self.config.engine_factory
            kwargs = dict(world_id=world_id, spec=spec, trace_id=self.trace_id)
            if hasattr(factory, "open_session"):
                kwargs["context"] = dict(
                    **(
                        {
                            "startup_prepare": True,
                            "startup_timeout_seconds": max(
                                0, self.startup.deadline - time.monotonic()
                            ),
                        }
                        if getattr(self, "startup", None) is not None
                        else {}
                    ),
                    **(
                        {"start_suspended": True}
                        if not getattr(self, "inference_enabled", lambda: True)()
                        else {}
                    ),
                    game_id=world.get("game_id"),
                    session_id=self.session_id,
                    user_id=self.user_id,
                    **(
                        {"expected_publication": self.expected_publication}
                        if getattr(self, "expected_publication", None) is not None
                        else {}
                    ),
                    **(
                        {"source_options": area_source_options(world)}
                        if area_source_options(world)
                        else {}
                    ),
                    **(
                        {"cli_fps": world["cli_fps"]}
                        if world.get("cli_fps") is not None
                        else {}
                    ),
                )
                factory = factory.open_session
            if getattr(self, "startup", None) is not None:
                from .startup import open_startup_engine

                self.engine = await open_startup_engine(
                    factory, kwargs, self.config.startup_seconds
                )
            else:
                self.engine = await asyncio.wait_for(
                    factory(**kwargs),
                    self.config.startup_seconds,
                )
        if getattr(self, "defer_startup_until_source", False):
            await self.source_model_ready(init)
        if (
            getattr(self, "startup", None) is not None
            and getattr(self, "expected_publication", None) is not None
            and not getattr(self, "defer_startup_until_source", False)
        ):
            # The original CLI descriptor must match before managed model work.
            self.startup.launch(init)
            await self.startup.wait(self.startup.admitted)
        self.renderer = getattr(
            self.engine, "renderer", None
        ) or self.config.ui_factory(world_id=world_id, trace_id=self.trace_id)
        if inspect.isawaitable(self.renderer):
            self.renderer = await self.renderer
        self.browser_ui = self.renderer.metadata.get("mode") == "browser"
        identity = getattr(self.engine, "identity", {})
        recording_id = identity.get("stream_id") or uuid4().hex
        inference = {key: value for key, value in init.items() if key != "references"}
        inference["deployment"] = {**self.config.model_metadata, **self.worker_metadata}
        self.recording = await self.archives.create_recording(
            directory=self.config.spool / recording_id,
            spec=spec,
            **(
                {"model_spec": self.model_geometry.output}
                if self.config.model_target is not None
                and not getattr(self, "source_only", False)
                else {}
            ),
            world_id=world_id,
            trace_id=self.trace_id,
            inference=inference,
            ui=self.renderer.metadata,
            references=references,
            object_prefix=identity.get(
                "object_prefix", f"{self.config.object_prefix}/{recording_id}"
            ),
            **({"source_only": True} if getattr(self, "source_only", False) else {}),
            **(
                {"archive_encoding": self.config.archive_encoding}
                if self.browser_ui
                and self.config.room_owned
                and getattr(self, "source_only", False)
                and self.config.independent_archive
                and getattr(self.archives, "supports_rgb", False)
                else {}
            ),
        )
        if getattr(self.recording, "raw_archive", False):
            await self.recording.prepare_raw()
        if self.browser_ui:
            if hasattr(self.recording, "bind_ui"):
                self.recording.bind_ui(self.engine, self.renderer.metadata)
            if hasattr(self.recording, "capture_assets"):
                # The archive owns asset copying too. It must not delay H3 init.
                task = asyncio.create_task(
                    self.recording.capture_assets(self.engine, self.renderer.metadata)
                )
                self.archives.tasks.add(task)
                task.add_done_callback(self.archives.tasks.discard)
                task.add_done_callback(
                    lambda t: t.exception() if not t.cancelled() else None
                )
            else:
                await self._copy_ui_assets()
        if identity:
            self.recording.manifest["multiplayer"] = {
                key: identity[key]
                for key in (
                    "game_id",
                    "player_id",
                    "stream_id",
                    "session_id",
                    "user_id",
                    "owner_id",
                    "owner_epoch",
                    "joined_room_time_us",
                    "game_object_prefix",
                )
            }
            self.recording.manifest["multiplayer"]["identity_scope"] = (
                "outer_gateway_identity"
            )
            await self.engine.note(
                "inference.init",
                dict(
                    prompt=prompt,
                    seed=seed,
                    references=self.recording.references,
                    stream_object_prefix=identity["object_prefix"],
                ),
            )
        metadata = getattr(self.engine, "metadata", {})
        self.recording.manifest["engine"] = metadata
        excluded = [
            value
            for value in (metadata, self.config.model_metadata, self.renderer.metadata)
            if value.get("training_eligible") is False
        ]
        if excluded:
            self.recording.manifest["dataset_status"] = (
                "excluded_mock"
                if any("mock" in str(value.get("mode", "")) for value in excluded)
                else "excluded_unvalidated"
            )
        if self.reports and self.report:
            self.reports.attach(self.report, self.recording)
        self.playback = H264PlaybackEncoder(
            (
                self.model_geometry.output
                if not getattr(self, "source_only", False)
                else spec
            ),
            max_bitrate=self.config.playback_max_bitrate,
            gop=self.config.playback_gop if self.config.room_owned else None,
            capture_clock=self.config.room_owned and self.config.capture_clock,
            immediate_capture=self.config.immediate_capture,
            capture_fps=(
                self.model_geometry.output.fps
                if self.config.model_target is not None
                and not getattr(self, "source_only", False)
                else self.capture_rate_hint()
            ),
        )
        if self.browser_ui:
            self.source_playback = create_source_playback_encoder(
                spec,
                max_bitrate=self.config.source_max_bitrate,
                gop=self.config.playback_gop if self.config.room_owned else None,
                capture_clock=self.config.room_owned and self.config.capture_clock,
                capture_fps=self.capture_rate_hint(
                    (self.config.source_encoding or {}).get("codec", "h264")
                ),
                **({"intra_refresh": True} if self.config.source_intra_refresh else {}),
                settings=self.config.source_encoding,
            )
        self.recording.manifest["playback"] = dict(self.playback.stream)
        from ..cache_observer import issue_recording

        cache_recording = issue_recording(
            init["world_id"], self.recording.manifest["object_prefix"]
        )
        if cache_recording is not None:
            init["cache_recording"] = cache_recording
            self.recording.manifest["cache_operations"] = dict(
                prefix="cache-operations/",
                publication="asynchronous_append_only",
                coverage="check_actual_receipts",
                included_in_video_object_digest=False,
            )
        await self.recording.configure()
        async with self.writer:
            await self.upstream.send(msgspec.msgpack.encode(init))
        self._spawn(self._read_h3(), "h3-receive")
        self._spawn(self._produce(), "engine-source")
        self._spawn(self._render(), "video-output")
        if hasattr(self.engine, "client_messages"):
            self._spawn(self._engine_messages(), "engine-client")

    async def _copy_ui_assets(self):
        total = 0
        for path, digest in self.renderer.metadata.get("files", {}).items():
            value = await self.engine.ui_asset(path)
            total += len(value)
            if total > 32 * 1024 * 1024:
                raise ValueError("The browser UI bundle exceeds the delivery budget")
            import hashlib

            if hashlib.sha256(value).hexdigest() != digest:
                raise ValueError("The transferred UI asset changed")
            target = self.recording.directory / "ui_bundle" / path
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("Invalid UI asset path")
            target.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(target.write_bytes, value)

    async def _engine_messages(self):
        await self.ready.wait()
        async for message in self.engine.client_messages():
            await self.outgoing.put(
                msgspec.msgpack.encode(dict(version=PROTOCOL_VERSION, **message))
            )

    async def _produce(self):
        await self.ready.wait()
        spec, limits = self.config.spec, self.limits
        iterator = self.engine.frames().__aiter__()
        while True:
            async with self.credit:
                with stage("source_credit_wait", self.trace_id):
                    await self.credit.wait_for(
                        lambda: (
                            self.submitted - self.consumed
                            < limits["max_pending_source_frames"]
                            and (self.submitted - self.consumed + 1) * spec.frame_bytes
                            <= limits["max_pending_source_bytes"]
                            and len(self.frames) < self.config.pending_frames
                        )
                    )
            try:
                with stage("engine_frame_wait", self.trace_id):
                    frame = await anext(iterator)
            except StopAsyncIteration:
                break
            if (
                frame.index != self.submitted
                or frame.pts_us != spec.pts(frame.index)
                or len(frame.rgb) != spec.frame_bytes
            ):
                raise ValueError("Engine returned an invalid frame timeline")
            frame = copy.deepcopy(frame)
            if frame.effective_input is None:
                frame = replace(
                    frame,
                    observed_input={
                        **(frame.observed_input or {}),
                        "latest": copy.deepcopy(self.observed_input),
                        "events": copy.deepcopy(self.input_events),
                        "mapping": (
                            "receipt_evidence_only"
                            if (frame.observed_input or {}).get("receipts")
                            else "unverified_received_before_capture"
                        ),
                    },
                )
            self.input_events = []
            self.frames[frame.index] = frame
            self.capture_times[frame.index] = time.perf_counter()
            if self.browser_ui:
                with stage("source_playback_encode", self.trace_id, frame=frame.index):
                    encoded = await asyncio.to_thread(
                        self.source_playback.encode, frame.index, frame.rgb
                    )
                await self.recording.encoded_video(
                    frame.index, encoded, name="whitebox", pts_us=capture_pts(frame)
                )
                await self.source_credit.reserve(frame.index)
                await self.outgoing.put(
                    msgspec.msgpack.encode(
                        dict(
                            version=PROTOCOL_VERSION,
                            type="whitebox.frame",
                            stream="whitebox",
                            output_frame_start=frame.index,
                            num_frames=1,
                            pts_us=capture_pts(frame),
                            model_pts_us=frame.pts_us,
                            width=spec.width,
                            height=spec.height,
                            source=frame.provenance,
                            observed_input=frame.observed_input,
                            **encoded,
                        )
                    )
                )
            else:
                await self.recording.video("whitebox", frame.index, frame.rgb)
            payload = dict(
                client_seq=self.submitted,
                source_frame_start=frame.index,
                num_frames=1,
                pts_us=frame.pts_us,
                data=frame.rgb,
            )
            # Reserve the watermark before yielding to the writer: a local/mock
            # worker can acknowledge in the same event-loop turn as send().
            self.submitted += 1
            with stage("source_send", self.trace_id, frame=frame.index):
                await self._event("source.append", payload)
        if self.drain_deadline_monotonic is None:
            self.drain_deadline_monotonic = (
                asyncio.get_running_loop().time() + self.config.drain_seconds
            )
        self.ended = True
        self.finish_requested.set()
        await self._event("source.end", dict(source_frame_end=self.submitted))

    async def _read_h3(self):
        spec = self.config.spec
        while True:
            wire = await self.upstream.recv()
            message = msgspec.msgpack.decode(wire)
            kind = message.get("type")
            if kind == "session.ready":
                if self.ready.is_set() or message.get("source_stream") != dict(
                    format="rgb24", width=spec.width, height=spec.height, fps=spec.fps
                ):
                    raise ValueError("H3 source negotiation failed")
                if (
                    message.get("max_message_bytes") != self.config.max_message_bytes
                    or message.get("max_pending_source_frames", 0) < 1
                    or message.get("max_pending_source_bytes", 0) < spec.frame_bytes
                    or message.get("max_source_message_bytes", 0)
                    < spec.frame_bytes + 4096
                ):
                    raise ValueError("H3 source budget is insufficient")
                self.limits = message
                self.ready.set()
                await self.outgoing.put(
                    msgspec.msgpack.encode(
                        dict(
                            version=PROTOCOL_VERSION,
                            type="session.ready",
                            session_id=self.session_id,
                            source_stream=message["source_stream"],
                            play_seconds=self.config.play_seconds,
                            format="h264",
                            video_stream=self.playback.stream,
                            playback_flow_control=(
                                "whitebox-dual-v2"
                                if self.browser_ui
                                else "whitebox-frame-v1"
                            ),
                            timeline=(
                                "source_capture_vfr"
                                if self.browser_ui
                                else "frame_clock"
                            ),
                            source_video_stream=(
                                self.source_playback.stream
                                if self.source_playback
                                else None
                            ),
                            ui=(
                                public_ui_metadata(self.renderer.metadata)
                                if self.browser_ui
                                else None
                            ),
                            recording_id=self.recording.directory.name,
                            report_token=self.report.token if self.report else None,
                            **(
                                {"project_audio": self.engine.metadata["project_audio"]}
                                if getattr(self.engine, "metadata", {}).get(
                                    "project_audio"
                                )
                                else {}
                            ),
                            engine=getattr(self.engine, "metadata", {}),
                            multiplayer=self.recording.manifest.get("multiplayer"),
                            object_prefix=self.recording.manifest["object_prefix"],
                        )
                    )
                )
            elif kind == "source.accepted":
                if (
                    message["source_frame_start"] != self.accepted
                    or not self.accepted < message["source_frame_end"] <= self.submitted
                ):
                    raise ValueError("Invalid H3 acceptance watermark")
                self.accepted = message["source_frame_end"]
            elif kind == "source.consumed":
                end = message["source_frame_end"]
                if (
                    not self.consumed <= end <= self.accepted
                    or message["source_bytes_end"] != end * spec.frame_bytes
                ):
                    raise ValueError("Invalid H3 consumption watermark")
                async with self.credit:
                    self.consumed = end
                    self.credit.notify_all()
            elif kind in {"frame_batch", "frame_batch_header"}:
                if kind == "frame_batch_header":
                    message["payload"] = await self.upstream.recv()
                start, count = message["output_frame_start"], message["num_frames"]
                if (
                    start != self.received
                    or count < 1
                    or start + count > self.accepted
                    or len(message["payload"]) != count * spec.frame_bytes
                    or message.get("format") != "rgb24"
                    or message.get("encoding") not in {None, "raw"}
                    or (message.get("width"), message.get("height"))
                    != (spec.width, spec.height)
                    or message.get("pts_us") != spec.pts(start)
                ):
                    raise ValueError("Invalid H3 output timeline")
                cursor = start
                for span in message["source_alignment"]:
                    if (
                        span["source_frame_start"] != cursor
                        or span["output_frame_start"] != cursor
                        or span["num_frames"] < 1
                    ):
                        raise ValueError("H3 source alignment is not one-to-one")
                    cursor += span["num_frames"]
                if cursor != start + count:
                    raise ValueError("Incomplete H3 source alignment")
                for index in range(start, start + count):
                    observe_duration(
                        "h3_source_to_output",
                        (time.perf_counter() - self.capture_times[index]) * 1000,
                        self.trace_id,
                    )
                self.received += count
                await self.render_queue.put(message)
            elif kind == "chunk_telemetry":
                for name, value in numeric_durations(message, CHUNK_FIELDS).items():
                    observe_duration(
                        "h3_telemetry_" + name, value, self.trace_id, service="denoiser"
                    )
            elif kind == "session.completed":
                if not self.ended or self.received != self.submitted:
                    raise ValueError("H3 ended before all frames were returned")
                await self.render_queue.put(None)
                return
            elif kind == "error":
                from .diagnostics import model_error

                raise model_error(
                    message, trace_id=self.trace_id, operation="render.receive"
                )
            # Source acknowledgements remain internal. Source
            # credit is released only by consumed, never accepted or rendered.

    async def _render(self):
        spec = self.config.spec
        while (message := await self.render_queue.get()) is not None:
            prompt_id = message.get("event_id") or 0
            if prompt_id not in self.prompts:
                raise ValueError("Unknown H3 prompt ownership")
            prompt = self.prompts[prompt_id]
            for index in range(
                message["output_frame_start"],
                message["output_frame_start"] + message["num_frames"],
            ):
                offset = (index - message["output_frame_start"]) * spec.frame_bytes
                rgb = message["payload"][offset : offset + spec.frame_bytes]
                frame = self.frames[index]
                if self.browser_ui:
                    composed = rgb
                else:
                    await self.recording.video("model", index, rgb)
                    with stage("ui_render", self.trace_id, frame=index):
                        composed = await self.renderer.render(rgb, frame, spec)
                if len(composed) != spec.frame_bytes:
                    raise ValueError("UI renderer changed video dimensions")
                with stage("playback_encode", self.trace_id, frame=index):
                    encoded = await asyncio.to_thread(
                        self.playback.encode, index, composed
                    )
                if self.browser_ui:
                    await self.recording.encoded_video(
                        index, encoded, name="model", pts_us=capture_pts(frame)
                    )
                else:
                    await self.recording.encoded_video(index, encoded)
                with stage("recording_metadata", self.trace_id, frame=index):
                    await self.recording.write_metadata(frame, prompt)
                with stage("playback_delivery_wait", self.trace_id, frame=index):
                    await self.playback_credit.reserve(index)
                with stage("playback_queue", self.trace_id, frame=index):
                    await self.outgoing.put(
                        msgspec.msgpack.encode(
                            dict(
                                version=PROTOCOL_VERSION,
                                type="frame_batch",
                                output_frame_start=index,
                                num_frames=1,
                                pts_us=(
                                    capture_pts(frame)
                                    if self.browser_ui
                                    else frame.pts_us
                                ),
                                model_pts_us=frame.pts_us,
                                stream="model",
                                source=frame.provenance,
                                observed_input=frame.observed_input,
                                width=spec.width,
                                height=spec.height,
                                **encoded,
                                event_id=prompt_id,
                                server_epoch_ms=time.time() * 1000,
                                states=frame.states,
                                pipeline_ms=(
                                    time.perf_counter() - self.capture_times[index]
                                )
                                * 1000,
                            )
                        )
                    )
                async with self.credit:
                    del self.frames[index], self.capture_times[index]
                    self.credit.notify_all()
            # Prompt events older than the actual applied event cannot reappear.
            self.prompts = {
                key: value for key, value in self.prompts.items() if key >= prompt_id
            }
        finish_renderer = (
            None if self.browser_ui else getattr(self.renderer, "finish", None)
        )
        if finish_renderer:
            with stage("ui_drain", self.trace_id):
                await finish_renderer()
        await asyncio.to_thread(self.playback.finish)
        if self.source_playback:
            await asyncio.to_thread(self.source_playback.finish)
        self.outcome = "completed"
        self.finished.set()
        await self.outgoing.put(
            msgspec.msgpack.encode(
                dict(
                    version=PROTOCOL_VERSION,
                    type="session.completed",
                    frames=self.received,
                )
            )
        )

    async def _deadline(self):
        await asyncio.wait_for(self.first_media.wait(), self.config.startup_seconds)
        completed = asyncio.create_task(self.finished.wait())
        requested = asyncio.create_task(self.finish_requested.wait())
        try:
            done, _ = await asyncio.wait(
                (completed, requested),
                timeout=self.config.play_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if completed in done:
                return
            if not done:
                await self.finish()
        finally:
            for task in (completed, requested):
                if not task.done():
                    task.cancel()
            await asyncio.gather(completed, requested, return_exceptions=True)
        await asyncio.wait_for(self.finished.wait(), self.config.drain_seconds)

    async def finish(self):
        if not self.ended and self.engine is not None:
            self.drain_deadline_monotonic = (
                asyncio.get_running_loop().time() + self.config.drain_seconds
            )
            self.ended = True
            self.finish_requested.set()
            await asyncio.wait_for(self.engine.finish(), self.config.drain_seconds)

    def media_delivered(self):
        if self.first_media is not None:
            self.first_media.set()

    async def recv(self):
        get = asyncio.create_task(self.outgoing.get())
        changed = None
        try:
            while True:
                self.changed.clear()
                for task in self.tasks:
                    if task.cancelled() and not self.closed:
                        self.outcome = "failed"
                        raise RuntimeError(
                            "Whitebox stage stopped unexpectedly: " + task.get_name()
                        )
                    if task.done() and not task.cancelled() and task.exception():
                        task.result()
                changed = asyncio.create_task(self.changed.wait())
                done, _ = await asyncio.wait(
                    (get, changed), return_when=asyncio.FIRST_COMPLETED
                )
                if get in done:
                    return get.result()
                await changed
        finally:
            for task in (get, changed):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (get, changed) if task is not None),
                return_exceptions=True,
            )

    async def close(self):
        if self.closed:
            return
        self.closed = True
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        try:
            if self.playback:
                await asyncio.to_thread(self.playback.close)
            if self.source_playback:
                await asyncio.to_thread(self.source_playback.close)
            if self.engine:
                await asyncio.wait_for(self.engine.close(), self.config.drain_seconds)
        finally:
            try:
                if self.renderer:
                    await asyncio.wait_for(
                        self.renderer.close(), self.config.drain_seconds
                    )
            finally:
                try:
                    await self.upstream.close()
                finally:
                    if self.recording:
                        self.archives.finalize(
                            self.recording, self.outcome, report=self.report
                        )
