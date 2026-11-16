# SPDX-License-Identifier: Apache-2.0
"""Adapter for worldkit stream v1.

CLIRoom retains one controller/view. The Room driver owns independent private
NativeCLIView instances for multiplayer members.
The default cli_compositor factory uses the SDK's public streaming compositor;
WHITEBOX_CLI_UI_FACTORY may replace it. It consumes the
original control messages via consume_ui(message), renders with the untouched
FrameHeader in frame.provenance, and owns bounded revision history.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import math
import os
import re
import signal
import tempfile
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

from .area_contract import (
    AREA_PROTOCOL,
    area_source_options,
    read_area_frame,
    read_area_receipt,
    read_area_result,
    read_area_transition,
    validate_area_descriptor,
)
from .cli_config import (
    INPUT_SESSION_RENEW_SECONDS,
    game_cache_directory,
    resource_transport,
    startup_seconds,
    stream_policy_arguments,
)
from .cli_logging import cli_log_sink, forward_blocked_log
from .contracts import EngineFrame
from .diagnostics import WhiteboxDiagnostic, cli_fps_override
from .metrics import observe_duration, stage
from .native_pixels import (
    MAX_HEADER,
    MAX_PIXELS,
    decode_native_packet,
    read_frame_identity,
    safe_int,
)
from .render_contract import canonical, identity
from .rooms import new_id

UI_EVENT_BATCH_LIMIT = 512

AREA_FRAME_HISTORY_BYTES = 8 << 20
AREA_FRAME_HISTORY_COUNT = 128
AREA_PENDING_LIMIT = 32
MODEL_CONTEXT_RECOVERY_SECONDS = 15
MODEL_CONTEXT_SUBSCRIPTION_SECONDS = 30

CLI_CONTRACT_REVISION = "dcf702e90092f90dbd230c058740f54925ee095f"


def _recovery_error(code):
    error = ConnectionError(code)
    error.code = code
    error.stage = "source"
    return error


def capture_fps(spec, env, requested=None):
    requested = cli_fps_override(requested)
    value = (
        requested
        if requested is not None
        else int(env.get("WHITEBOX_CLI_FPS", str(spec.fps)))
    )
    if not 1 <= value <= 60:
        raise ValueError("CLI capture FPS must be between 1 and 60")
    return value


PUBLICATION_IDENTITY_FIELDS = (
    "sourceId",
    "worldId",
    "publicationId",
    "sourceManifestSha256",
    "renditionInputSha256",
    "manifestSha256",
)


def publication_identity(value):
    if (
        not isinstance(value, dict)
        or not set(PUBLICATION_IDENTITY_FIELDS) <= value.keys()
        or value.keys() - set(PUBLICATION_IDENTITY_FIELDS) - {"attempt"}
        or any(
            not isinstance(value[k], str) or not 1 <= len(value[k]) <= 256
            for k in PUBLICATION_IDENTITY_FIELDS[:3]
        )
        or any(
            not isinstance(value[k], str) or not re.fullmatch(r"[a-f0-9]{64}", value[k])
            for k in PUBLICATION_IDENTITY_FIELDS[3:]
        )
        or (
            "attempt" in value
            and (
                type(value["attempt"]) is not int
                or not 0 <= value["attempt"] <= 9007199254740991
            )
        )
    ):
        raise ValueError("Invalid frozen publication identity")
    return {key: value[key] for key in PUBLICATION_IDENTITY_FIELDS}


def validate_frozen_descriptor(descriptor, expected):
    expected = publication_identity(expected)
    actual = descriptor.get("artifact") if isinstance(descriptor, dict) else None
    if (
        not isinstance(actual, dict)
        or any(actual.get(key) != value for key, value in expected.items())
        or actual.get("worldBuildHash") != expected["manifestSha256"]
        or descriptor.get("worldBuildHash") != expected["manifestSha256"]
    ):
        raise WhiteboxDiagnostic("WTP_PLAYABLE_PUBLICATION_CHANGED")


def cli_arguments(
    entry, world_id, spec, env, *, delivery=None, cli_fps=None, source_options=None
):
    """Keep deployment policy outside the CLI; never send platform credentials in argv."""
    source_options = area_source_options(source_options or {})
    if source_options and (
        delivery
        or resource_transport(env) == "archive"
        or env.get("WHITEBOX_CLI_CACHE_UPDATE") == "async"
    ):
        raise ValueError("AREA_SOURCE_OPTIONS_INVALID")
    args = [
        env.get("WHITEBOX_NODE", "node"),
        str(entry),
        "stream",
        "--delivery" if delivery else "--wtp",
        str(delivery) if delivery else world_id,
        "--width",
        str(spec.width),
        "--height",
        str(spec.height),
        "--fps",
        str(capture_fps(spec, env, cli_fps)),
        "--rendering",
        env.get("WHITEBOX_CLI_RENDERING", "auto"),
        "--timeout",
        str(startup_seconds(env)),
        "--json",
    ]
    args.extend(stream_policy_arguments(env))
    transport = resource_transport(env)
    cache = game_cache_directory(env)
    if not delivery:
        if transport == "archive":
            raise ValueError("Archive mode requires a prepared delivery")
        args.extend(["--resource-transport", transport])
        if cache:
            args.extend(["--cache-dir", cache])
        if env.get("WHITEBOX_CLI_CACHE_UPDATE"):
            if env["WHITEBOX_CLI_CACHE_UPDATE"] not in {"sync", "async"}:
                raise ValueError("Invalid CLI cache update policy")
            args.extend(["--cache-update", env["WHITEBOX_CLI_CACHE_UPDATE"]])
    if not delivery and env.get("WHITEBOX_CLI_PLATFORM_URL"):
        args.extend(["--platform-url", env["WHITEBOX_CLI_PLATFORM_URL"]])
    if env.get("WHITEBOX_CLI_BROWSER_EXECUTABLE"):
        args.extend(["--browser-executable", env["WHITEBOX_CLI_BROWSER_EXECUTABLE"]])
    if source_options:
        args += ["--loading", "areas"]
        for group in source_options["initial_groups"]:
            args += ["--area-group", group]
    return args


async def prepare_delivery(
    entry, world_id, directory, trace_id, *, create_context=None, verified_caller=None
):
    timeout = float(os.environ.get("WHITEBOX_CLI_DELIVERY_TIMEOUT_SECONDS", "300"))
    if not math.isfinite(timeout) or not 0 < timeout <= 900:
        raise ValueError("Invalid CLI delivery deadline")
    process = await asyncio.create_subprocess_exec(
        os.environ.get("WHITEBOX_NODE", "node"),
        "--import",
        "tsx",
        str(Path(__file__).with_name("worldkit_delivery.mjs")),
        cwd=entry.parents[3],
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        with stage("cli_delivery_prepare", trace_id, service="room_runtime"):
            stdout, stderr = await asyncio.wait_for(
                process.communicate(
                    json.dumps(
                        {
                            "worldId": world_id,
                            "directory": directory,
                            "timeoutSeconds": timeout,
                            **(
                                {
                                    "createContext": create_context,
                                    "verifiedCaller": verified_caller,
                                }
                                if create_context is not None
                                else {}
                            ),
                        }
                    ).encode()
                ),
                timeout + 5,
            )
            if process.returncode:
                code = stderr.decode(errors="replace").strip()
                if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,100}", code):
                    code = "DELIVERY_PREPARATION_FAILED"
                raise ValueError("CLI delivery preparation failed: " + code)
            return json.loads(stdout)
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()


def validate_prepared_delivery(descriptor, prepared):
    if (
        any(
            descriptor.get("artifact", {}).get(k) != v
            for k, v in prepared["identity"].items()
        )
        or descriptor.get("worldBuildHash") != prepared["identity"]["worldBuildHash"]
        or (
            "uiBundleHash" in prepared
            and descriptor.get("uiBundleHash") != prepared["uiBundleHash"]
        )
    ):
        raise ValueError("The CLI started a different prepared delivery")


def observe_service_cleanup(value, receipt):
    """Bind the native cancellation receipt to this one CLI invocation."""
    kind = value.get("kind")
    if kind not in {"worldkit-service-request", "worldkit-service-cleanup"}:
        return False
    request_id = value.get("requestId")
    if (
        type(value.get("schemaVersion")) is not int
        or value.get("schemaVersion") != 1
        or value.get("protocol") != "worldkit-service-cleanup-v1"
        or not isinstance(request_id, str)
        or re.fullmatch(r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}", request_id)
        is None
    ):
        receipt["invalid"] = True
        return True
    if kind == "worldkit-service-request":
        if receipt.get("request_id") not in {None, request_id}:
            receipt["invalid"] = True
        receipt["request_id"] = request_id
    else:
        terminal = {"state": value.get("state"), "origin": value.get("origin")}
        if (
            receipt.get("request_id") != request_id
            or terminal["state"] not in {"not_created", "retired"}
            or terminal["origin"] not in {"client", "service"}
            or terminal == {"state": "retired", "origin": "client"}
            or receipt.get("terminal", terminal) != terminal
        ):
            receipt["invalid"] = True
        else:
            receipt["terminal"] = terminal
            receipt["confirmed"] = True
    return True


async def cli_diagnostics(
    process, trace_id, startup_metrics=None, service_cleanup=None, service_ready=None
):
    """Drain stderr without forwarding untrusted SDK text or credentials."""
    diagnostic = None
    async for line in process.stderr:
        try:
            text = line.decode(errors="replace")
            key = value = None
            if text.startswith("worldkit: Preparation: "):
                key = "preparationMs"
                value = json.loads(text.split(": ", 2)[2]).get("elapsedMs")
            elif text.startswith("worldkit: Startup timing: "):
                timing = json.loads(text.split(": ", 2)[2])
                key = {
                    "browser-launch": "browserLaunchMs",
                    "runtime-prepare": "runtimePrepareMs",
                    "browser-context": "browserContextMs",
                    "world-load": "worldLoadMs",
                    "environment-inspection": "environmentInspectionMs",
                    "producer-start": "producerStartMs",
                }.get(timing.get("phase"))
                value = timing.get("elapsedMs")
            if (
                key
                and startup_metrics is not None
                and key not in startup_metrics
                and type(value) in (int, float)
                and math.isfinite(value)
                and 0 <= value <= 3600000
            ):
                startup_metrics[key] = value
                observe_duration("cli_" + key, value, trace_id, service="room_runtime")
            if key:
                continue
            value = json.loads(line)
            if service_cleanup is not None and observe_service_cleanup(
                value, service_cleanup
            ):
                if (
                    service_ready is not None
                    and service_cleanup.get("request_id")
                    and not service_cleanup.get("invalid")
                ):
                    service_ready.set()
                continue
            if forward_blocked_log(value, trace_id):
                continue
            code = value.get("error", {}).get("code", "")
            if value.get("kind") == "worldkit-cli-error" and re.fullmatch(
                r"[A-Z][A-Z0-9_]{1,100}", code
            ):
                logging.getLogger(__name__).error(
                    "CLI startup failed trace_id=%s code=%s", trace_id, code
                )
                diagnostic = WhiteboxDiagnostic(code, value.get("stage", "startup"))
                # These SDK failures precede World/view execution. In the pinned
                # service manager, SERVICE_QUEUE_TIMEOUT rejects an unassigned
                # request or removes it from the queue; dispatch clears that
                # timer before invoking the worker. STREAM_STARTUP_TIMEOUT and
                # generic service failures can occur after dispatch and remain
                # unconfirmed, even when the client process has exited.
                diagnostic.sdk_not_created = value.get(
                    "schemaVersion"
                ) == 1 and code in {
                    "WTP_DELIVERY_NOT_FOUND",
                    "STREAM_DELIVERY_UI_REQUIRED",
                    "SERVICE_QUEUE_TIMEOUT",
                }
        except (ValueError, TypeError, AttributeError):
            pass
    return diagnostic


def validate_descriptor(value, spec, *, fps=None):
    base = urlsplit(value["baseUrl"])
    native = value.get("media", {}).get("kind") == "pixels"
    if (
        value.get("kind") != "worldkit-stream-session"
        or value.get("schemaVersion") != 1
        or value.get("protocolVersion") != 1
        or value.get("status") != "ready"
        or base.scheme != "http"
        or base.hostname != "127.0.0.1"
        or not base.port
        or base.username
        or base.password
        or base.path not in {"", "/"}
        or base.query
        or base.fragment
        or (value.get("width"), value.get("height"), value.get("fps"))
        != (spec.width, spec.height, spec.fps if fps is None else fps)
        or (
            value.get("media", {}).get("pixelFormat") != "rgb24"
            if native
            else value["streams"]["video"].get("codec") != "vp8"
        )
        or not isinstance(value.get("accessToken"), str)
        or not value["accessToken"]
    ):
        raise ValueError("The CLI returned an unsupported session")
    for channel, path in (
        (("pixels", "pixels") if native else ("video", "media")),
        ("ui", "control"),
    ):
        url = urlsplit(value["streams"][channel]["url"])
        if (
            url.scheme != "ws"
            or url.netloc != base.netloc
            or url.query
            or url.fragment
            or url.path != f"/v1/sessions/{value['sessionId']}/{path}"
        ):
            raise ValueError("The CLI returned an invalid local endpoint")
    if value.get("modelContext") is not None:
        from .source_conditions import validate_capability

        validate_capability(value["modelContext"])
    if "sceneTransitions" in value:
        from .scene_transition import read_scene_transitions

        if not native or value.get("accessMode") == "capability":
            raise ValueError("Unsupported scene transition source")
        read_scene_transitions(value["sceneTransitions"])
    return value


def decode_packet(wire, descriptor, spec):
    if not isinstance(wire, bytes) or not 5 <= len(wire) <= 4259844:
        raise ValueError("Invalid CLI media packet")
    length = int.from_bytes(wire[:4], "big")
    if not 1 <= length <= 65536 or 4 + length >= len(wire):
        raise ValueError("Invalid CLI media header")
    header = json.loads(wire[4 : 4 + length])
    payload = wire[4 + length :]
    if (
        header.get("protocolVersion") != 1
        or header.get("sessionId") != descriptor["sessionId"]
        or header.get("epoch") != descriptor["epoch"]
        or header.get("payloadBytes") != len(payload)
        or len(payload) > 4194304
        or header.get("type") not in {"key", "delta"}
        or (header.get("widthPixels"), header.get("heightPixels"))
        != (spec.width, spec.height)
    ):
        raise ValueError("The CLI media identity is invalid")
    for key in (
        "outputFrameId",
        "outputPtsUs",
        "mediaGeneration",
        "uiRevision",
        "uiCompleteThroughUs",
    ):
        if type(header.get(key)) is not int or not 0 <= header[key] <= 2**53 - 1:
            raise ValueError("The CLI frame timeline is invalid")
    for key in (
        "sourceFrameId",
        "sourceTimeUs",
        "simulationTick",
        "sdkEpoch",
        "worldRevision",
    ):
        if (
            type(header.get("source", {}).get(key)) is not int
            or not 0 <= header["source"][key] <= 2**53 - 1
        ):
            raise ValueError("The CLI source timeline is invalid")
    if not isinstance(header["source"].get("presentationId"), str):
        raise ValueError("The CLI presentation is missing")
    # The VP8 bitstream key-frame bit must agree with the envelope.
    if (payload[0] & 1 == 0) != (header["type"] == "key"):
        raise ValueError("The CLI keyframe header disagrees with the video")
    return header, payload


CLI_MOUSE_BUTTON_CODES = {0: "MouseLeft", 2: "MouseRight"}


def cli_input(sequence, payload):
    keys = payload.get("held_keys", [])
    if (
        not isinstance(keys, list)
        or len(keys) > 64
        or any(not isinstance(key, str) or len(key) > 32 for key in keys)
    ):
        raise ValueError("Invalid held keys")
    buttons = payload.get("held_buttons", [])
    if (
        not isinstance(buttons, list)
        or len(buttons) > 64
        or any(type(button) is not int or button not in (0, 1, 2) for button in buttons)
    ):
        raise ValueError("Invalid held buttons")
    keys = list(
        dict.fromkeys(
            keys
            + [
                CLI_MOUSE_BUTTON_CODES[b]
                for b in buttons
                if b in CLI_MOUSE_BUTTON_CODES
            ]
        )
    )
    if len(keys) > 64:
        raise ValueError("Invalid held keys and buttons")
    events = payload.get("events", [])
    if not isinstance(events, list) or len(events) > 128:
        raise ValueError("Invalid input events")
    edges, yaw, pitch = [], 0.0, 0.0
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Invalid input event")
        if event.get("type") in {"keydown", "keyup"}:
            edges.append(
                {
                    "code": event["code"],
                    "kind": "down" if event["type"] == "keydown" else "up",
                }
            )
        elif event.get("type") in {"pointerdown", "pointerup"}:
            button = event.get("button")
            if type(button) is not int or button not in (0, 1, 2):
                raise ValueError("Invalid pointer buttons")
            if button in CLI_MOUSE_BUTTON_CODES:
                edges.append(
                    {
                        "code": CLI_MOUSE_BUTTON_CODES[button],
                        "kind": "down" if event["type"] == "pointerdown" else "up",
                    }
                )
        elif event.get("type") == "pointermove":
            # Match the SDK browser camera: rightward drag decreases yaw.
            yaw -= float(event.get("dx", 0)) * 0.002
            pitch += float(event.get("dy", 0)) * 0.002
    if len(edges) > 128 or not all(
        math.isfinite(x) and abs(x) <= 100 for x in (yaw, pitch)
    ):
        raise ValueError("Invalid camera input")
    return dict(
        sequence=sequence,
        heldKeys=keys,
        keyEdges=edges,
        yawDeltaRadians=yaw,
        pitchDeltaRadians=pitch,
        distanceDeltaMeters=0,
    )


class CLIRoom:
    max_players = 1  # Change through a new multi-view driver, not by raising this.

    def __init__(self, process, descriptor, spec, renderer_factory, trace_id):
        self.process, self.descriptor, self.spec = process, descriptor, spec
        self.renderer_factory, self.trace_id = renderer_factory, trace_id
        self.view = None
        self.prepared_delivery = self.temporary = self.diagnostics = None
        self.service_cleanup = {}
        self.service_ready = asyncio.Event()
        self.resident_service = False
        self.http_delete_confirmed = False
        self.player_id = None
        self.input_renew_task = None

    @property
    def project_input_source(self):
        return self.descriptor is not None and self.descriptor.get("input") is not None

    @property
    def external_input(self):
        return self.project_input_source

    async def public_session(self, player_id, public_origin):
        if (
            not self.project_input_source
            or self.view is None
            or self.view.closed
            or player_id != self.player_id
        ):
            raise ValueError("The player input session is unavailable")
        import httpx

        from .player_transport import publish_input_session

        descriptor = self.descriptor
        async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
            response = await client.post(
                descriptor["baseUrl"].rstrip("/")
                + f"/v1/sessions/{descriptor['sessionId']}/input-session",
                headers={"Authorization": "Bearer " + descriptor["accessToken"]},
            )
            response.raise_for_status()
        session = response.json()
        if (
            self.view.closed
            or any(
                session.get(key) != descriptor.get(key)
                for key in (
                    "sessionId",
                    "epoch",
                    "worldBuildHash",
                    "uiBundleHash",
                    "input",
                    "media",
                    "width",
                    "height",
                    "fps",
                )
            )
            or session.get("accessMode") != "capability"
            or not re.fullmatch(r"[a-f0-9]{48}", session.get("accessToken", ""))
        ):
            raise ValueError("The player input session has changed")
        self.view.player_descriptor = session
        published = publish_input_session(session, public_origin)
        if self.input_renew_task is None:
            self.input_renew_task = asyncio.create_task(
                self._renew_input_session(public_origin)
            )
        return published

    async def _renew_input_session(self, public_origin):
        try:
            while self.view is not None and not self.view.closed:
                await asyncio.sleep(INPUT_SESSION_RENEW_SECONDS)
                if self.view.closed:
                    return
                await self.public_session(self.player_id, public_origin)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.getLogger(__name__).exception("CLI input grant renewal failed")
            if self.view is not None:
                self.view.end_reason = "PLAYER_INPUT_SESSION_EXPIRED"
                await self.view.close()

    async def join(self, *, player_id, spec):
        if self.view is not None:
            raise ValueError("This CLI version supports one player view")
        view = (
            NativeCLIView(self)
            if self.descriptor.get("media", {}).get("kind") == "pixels"
            else CLIView(self)
        )
        try:
            await view.start()
        except BaseException:
            await view.close()
            raise
        self.view = view
        self.player_id = player_id
        return view

    async def close(self):
        try:
            try:
                if self.input_renew_task is not None:
                    self.input_renew_task.cancel()
                    await asyncio.gather(self.input_renew_task, return_exceptions=True)
                    self.input_renew_task = None
            finally:
                if self.view:
                    await self.view.close()
        finally:
            try:
                if self.process.returncode is None:
                    await self._stop_cli()
            finally:
                try:
                    if self.diagnostics:
                        await asyncio.gather(self.diagnostics, return_exceptions=True)
                    if self.resident_service and self.descriptor is not None:
                        confirmed = self.service_cleanup.get(
                            "confirmed"
                        ) is True and not self.service_cleanup.get("invalid")
                        legacy_confirmed = (
                            not self.service_cleanup
                            and self.http_delete_confirmed
                            and self.process.returncode == 0
                        )
                        if not (confirmed or legacy_confirmed):
                            raise ValueError("SDK_SERVICE_CLEANUP_UNKNOWN")
                finally:
                    if self.temporary:
                        self.temporary.cleanup()

    async def _stop_cli(self):
        try:
            if self.descriptor:
                # Normal CLI completion acknowledges the resident request only
                # after source/temp cleanup. Socket loss cancels asynchronously.
                async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                    response = await client.delete(
                        self.descriptor["baseUrl"]
                        + "/v1/sessions/"
                        + self.descriptor["sessionId"],
                        headers={
                            "Authorization": "Bearer " + self.descriptor["accessToken"]
                        },
                    )
                if response.status_code == 200:
                    self.http_delete_confirmed = True
                    await asyncio.wait_for(self.process.wait(), 5)
        except (httpx.HTTPError, asyncio.TimeoutError):
            pass
        finally:
            if self.process.returncode is None:
                await self._terminate_cli()

    async def _terminate_cli(self):
        if (
            self.resident_service
            and self.descriptor is None
            and not self.service_ready.is_set()
            and self.process.returncode is None
        ):
            # Wait briefly for the original CLI's signal handler, not for World
            # readiness. The marker alone never proves resource cleanup.
            ready = asyncio.create_task(self.service_ready.wait())
            exited = asyncio.create_task(self.process.wait())
            try:
                await asyncio.wait(
                    (ready, exited), timeout=5, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                ready.cancel()
                exited.cancel()
                await asyncio.gather(ready, exited, return_exceptions=True)
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(self.process.wait(), 35)
        finally:
            if self.process.returncode is None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await self.process.wait()


class CLIView:
    def __init__(self, room):
        from .capture_wait import CaptureWait

        self.capture_wait = CaptureWait()
        self.room, self.descriptor, self.spec = room, room.descriptor, room.spec
        self.external_input = getattr(room, "external_input", False)
        if getattr(room, "project_input_source", False):
            grant = self.descriptor.get("inputSession")
            if (
                not isinstance(grant, dict)
                or grant.get("accessMode") != "capability"
                or grant.get("sessionId") != self.descriptor.get("sessionId")
            ):
                raise ValueError("The CLI input session is unavailable")
            self.player_descriptor = grant
        self.control = self.media = self.control_task = self.renderer = None
        self.transport_generation = 0
        self.transport_lock = asyncio.Lock()
        self.source_debug_supported = False
        self.source_debug_condition = []
        self.scene_occurrences = []
        self.model_context = None
        if self.descriptor.get("modelContext") is not None:
            if self.descriptor.get("media", {}).get("kind") != "pixels":
                raise ValueError("MODEL_CONTEXT_CAPABILITY_UNSUPPORTED")
            from .source_condition_channel import CLIConditionChannel

            self.model_context = CLIConditionChannel(self)
        self.area_transition = None
        self.area_runtime = self.area_publication = None
        self.area_frames = OrderedDict()
        self.area_pending = {}
        self._close_task = None
        self.closed = False
        self.end_reason = None
        self.finish_requested = asyncio.Event()
        self.configured, self.ready = asyncio.Event(), asyncio.Event()
        self.controller_ready = False
        self.binding_sequence = None
        self.config, self.receipts, self.ui_events = None, [], []
        self.audio_snapshot = None
        self.audio_history = OrderedDict()
        self.audio_history_bytes = 0
        self.ui_bytes = 0
        self.stopped, self.last_id, self.last_pts, self.sequence = False, None, None, -1
        self.ui_revision, self.ui_through = -1, -1
        self.ui_bootstrap_time = None
        self.ui_changed = asyncio.Condition()
        self.metadata = {
            **(
                {"area_protocol": AREA_PROTOCOL}
                if self.descriptor.get("area") == {"protocol": AREA_PROTOCOL}
                and self.descriptor.get("loading", {}).get("kind") == "areas"
                else {}
            ),
            "mode": "worldkit_cli",
            "cli_contract_revision": CLI_CONTRACT_REVISION,
            "artifact": self.descriptor.get("artifact"),
            "wtp_delivery": room.prepared_delivery,
            "runtime": self.descriptor.get("runtime"),
            "startup_metrics": getattr(room, "startup_metrics", None),
            "cache": self.descriptor.get("cache"),
            "service": self.descriptor.get("service"),
            "world_build_hash": self.descriptor.get("worldBuildHash"),
            "ui_bundle_hash": self.descriptor.get("uiBundleHash"),
            "max_players": 1,
            **(
                {"scene_transitions": self.descriptor["sceneTransitions"]}
                if "sceneTransitions" in self.descriptor
                else {}
            ),
            "source_codec": (
                "rgb24"
                if self.descriptor.get("media", {}).get("kind") == "pixels"
                else "vp8"
            ),
            "model_input": "rgb24",
            "capture_fps": self.descriptor.get("fps", self.spec.fps),
            "model_fps": self.spec.fps,
            "input_mapping": "receipt_evidence_only",
            **(
                {"project_input": True}
                if getattr(room, "project_input_source", False)
                else {}
            ),
        }

    def _accept_area_transition(self, value):
        if self.descriptor.get("area") != {"protocol": AREA_PROTOCOL}:
            raise ValueError("AREA_PRESENTATION_UNSUPPORTED")
        transition = read_area_transition(value)
        if transition["epoch"] != self.descriptor["epoch"]:
            raise ValueError("AREA_PRESENTATION_STALE")
        if self.area_runtime is not None and canonical(self.area_runtime) != canonical(
            transition["runtime"]
        ):
            raise ValueError("AREA_PRESENTATION_RUNTIME")
        operation = transition["status"].get("transition") or transition["status"].get(
            "pending"
        )
        publication = operation.get("publication") if operation else None
        expected = (
            self.descriptor.get("loading", {})
            .get("initialPreparation", {})
            .get("identity")
        )
        if publication is not None and (
            expected is not None
            and canonical(publication) != canonical(expected)
            or self.area_publication is not None
            and canonical(publication) != canonical(self.area_publication)
            or self.descriptor.get("worldBuildHash") is not None
            and publication["manifestSha256"] != self.descriptor["worldBuildHash"]
        ):
            raise ValueError("AREA_PRESENTATION_PUBLICATION")
        self.area_runtime = copy.deepcopy(transition["runtime"])
        if publication is not None:
            self.area_publication = copy.deepcopy(publication)
        if (
            self.area_transition
            and transition["revision"] <= self.area_transition["revision"]
        ):
            return
        current = transition["status"].get("transition")
        if (
            not self.area_transition
            or canonical(current)
            != canonical(self.area_transition["status"].get("transition"))
            or transition["status"]["phase"] != "awaiting-presented"
        ):
            self.area_frames.clear()
        for operation_id, future in self.area_pending.values():
            if not future.done() and (
                current is None
                or current["operationId"] != operation_id
                or transition["status"]["phase"] in {"failed", "cancelled", "disposed"}
            ):
                future.set_exception(ValueError("AREA_PRESENTATION_STALE"))
        self.area_transition = transition

    async def area_status(self):
        if self.descriptor.get("area") != {"protocol": AREA_PROTOCOL}:
            raise ValueError("AREA_PRESENTATION_UNSUPPORTED")
        if self.closed or self.stopped:
            raise ConnectionError("AREA_SOURCE_CLOSED")
        return (
            {
                key: copy.deepcopy(value)
                for key, value in self.area_transition.items()
                if key != "ack"
            }
            if self.area_transition
            else None
        )

    async def area_frame(self, header):
        await self.area_status()
        frame = read_frame_identity(header)
        return self._area_provenance(frame).get("area_frame")

    async def area_presented(self, body, *, before_send=None):
        if not isinstance(body, dict) or set(body) != {"request_id", "receipt"}:
            raise ValueError("AREA_RECEIPT_INVALID")
        request_id = identity(body.get("request_id"))
        receipt = read_area_receipt(body.get("receipt"))
        status = await self.area_status()
        if (
            not status
            or status["status"]["phase"] != "awaiting-presented"
            or canonical(status["status"].get("transition"))
            != canonical(receipt["identity"])
            or not self.area_transition.get("ack")
            or receipt["frame"]["sessionId"] != self.descriptor["sessionId"]
            or receipt["frame"]["epoch"] != self.descriptor["epoch"]
            or not self._area_provenance(receipt["frame"])
        ):
            raise ValueError("AREA_PRESENTATION_STALE")
        if (
            request_id in self.area_pending
            or len(self.area_pending) >= AREA_PENDING_LIMIT
        ):
            raise ValueError("AREA_PRESENTATION_PENDING")
        # This synchronous owner check follows every await before native dispatch.
        # A rejection here has no effect and must not become an unknown send.
        if before_send is not None:
            before_send()
        future = asyncio.get_running_loop().create_future()
        self.area_pending[request_id] = (receipt["identity"]["operationId"], future)
        try:
            await self.control.send(
                json.dumps(
                    dict(
                        type="area.presented",
                        epoch=self.descriptor["epoch"],
                        requestId=request_id,
                        nonce=self.area_transition["ack"]["nonce"],
                        **receipt,
                    )
                )
            )

            async def confirmed():
                return await future

            result = await self._wait_or_error(confirmed())
            if result is None:
                raise ConnectionError("AREA_PRESENTATION_UNKNOWN")
            return result
        except (
            asyncio.CancelledError,
            ConnectionError,
            TimeoutError,
            ValueError,
        ) as error:
            # The original journal request may already have had an effect. No retry here.
            raise ConnectionError("AREA_PRESENTATION_UNKNOWN") from error
        finally:
            self.area_pending.pop(request_id, None)
            if not future.done():
                future.cancel()

    def _area_provenance(self, header):
        evidence = self.area_frames.get(header["outputFrameId"])
        if not evidence:
            return {}
        frame = {key: header.get(key) for key in evidence["frame"]}
        if (
            canonical(frame) != canonical(evidence["frame"])
            or not self.area_transition
            or canonical(evidence["identity"])
            != canonical(self.area_transition["status"].get("transition"))
        ):
            return {}
        return {"area_frame": copy.deepcopy(evidence)}

    async def start(self):
        self.client_id = new_id()
        renderer = self.room.renderer_factory(
            descriptor=self.descriptor,
            spec=self.spec,
            **(
                {"package": self.room.ui_package}
                if getattr(self.room, "ui_package", None)
                else {}
            ),
        )
        self.renderer = await renderer if inspect.isawaitable(renderer) else renderer
        if not callable(getattr(self.renderer, "consume_ui", None)):
            raise ValueError(
                "The CLI requires a compositor that consumes source UI revisions"
            )
        from .project_audio import install_audio_package

        await install_audio_package(self)
        await self._open_transport()
        self.control_task = asyncio.create_task(self._control_transport())
        await self._wait_or_error(self.ready.wait())
        if self.end_reason is not None:
            raise ValueError("The CLI session ended before player readiness")

    async def _open_transport(self):
        from websockets.asyncio.client import connect

        private = self.descriptor.get("accessMode") == "capability"
        protocols = None
        if private:
            token = self.descriptor["accessToken"]
            if not re.fullmatch(r"[a-f0-9]{48}", token):
                raise ValueError("The CLI viewer capability is invalid")
            protocols = ["worldkit.stream.v1", "worldkit.cap." + token]

        def url(channel):
            value = urlsplit(self.descriptor["streams"][channel]["url"])
            query = dict(clientId=self.client_id)
            if self.external_input:
                query["role"] = "spectator"
            if not private:
                query["token"] = self.descriptor["accessToken"]
            if channel == "ui":
                query.update(ui="1", clock="1")
                if not private and "sceneTransitions" in self.descriptor:
                    query["sceneOccurrences"] = "1"
            return urlunsplit(value._replace(query=urlencode(query)))

        self.source_debug_supported = False
        self.source_debug_condition = []
        self.control = await connect(
            url("ui"),
            subprotocols=protocols,
            max_size=1048576,
            max_queue=8,
            compression=None,
            close_timeout=1,
        )
        self.media = await connect(
            url(
                "pixels"
                if self.descriptor.get("media", {}).get("kind") == "pixels"
                else "video"
            ),
            subprotocols=protocols,
            max_size=MAX_PIXELS + MAX_HEADER + 4,
            max_queue=2,
            compression=None,
            close_timeout=1,
        )
        self.transport_generation += 1

    async def recover_model_context_transport(self, generation):
        """Reopen transport only; retain the source, consumer, adapter and journal.

        Never replay input/actions or a model commit here. The channel resumes
        the contiguous control prefix and queries its original pending request.
        """
        if self.model_context is None or self.closed or self.stopped:
            raise _recovery_error("MODEL_CONTEXT_RECOVERY_UNAVAILABLE")
        async with self.transport_lock:
            if generation != self.transport_generation:
                return
            try:
                async with asyncio.timeout(MODEL_CONTEXT_RECOVERY_SECONDS):
                    delay = 0.1
                    while not self.closed and not self.stopped:
                        await asyncio.gather(
                            *(
                                socket.close()
                                for socket in (self.control, self.media)
                                if socket is not None
                            ),
                            return_exceptions=True,
                        )
                        self.ready.clear()
                        self.configured.clear()
                        self.controller_ready = False
                        self.binding_sequence = None
                        self.model_context.bound.clear()
                        try:
                            await self._open_transport()
                            if self.model_context.segment is not None:
                                await self.model_context.reconnect()
                            return
                        except (OSError, ConnectionClosed, TimeoutError):
                            await asyncio.sleep(delay)
                            delay = min(1, delay * 2)
            except TimeoutError as error:
                raise _recovery_error("MODEL_CONTEXT_RECOVERY_EXPIRED") from error
            raise _recovery_error("MODEL_CONTEXT_RECOVERY_UNAVAILABLE")

    async def _control_transport(self):
        while not self.closed and not self.stopped:
            generation = self.transport_generation
            try:
                await self._control()
            except (OSError, ConnectionClosed):
                if self.model_context is None or self.closed or self.stopped:
                    raise
            if self.end_reason is not None or self.closed or self.stopped:
                return
            if self.model_context is None:
                return
            await self.recover_model_context_transport(generation)

    async def _wait_or_error(self, operation, *, timeout=15):
        task = asyncio.create_task(operation)
        finishing = asyncio.create_task(self.finish_requested.wait())
        try:
            done, _ = await asyncio.wait(
                (task, self.control_task, finishing),
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Finish may wake the reader in the same turn as a real reset or
            # producer/transport error. Preserve an already completed fault.
            if self.control_task in done and not self.control_task.cancelled():
                self.control_task.result()
            if self.finish_requested.is_set():
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                if not task.cancelled():
                    error = task.exception()
                    # Context retirement may wake a stale sample wait. A real
                    # media transport failure is independent of that teardown.
                    if isinstance(
                        error, (OSError, ConnectionClosed)
                    ) and not isinstance(error, ConnectionClosedOK):
                        raise error
                return None
            if self.control_task in done:
                if self.control_task.cancelled():
                    # The Room owner can close the CLI while a frame read is
                    # pending. That ends the stream; it must not silently
                    # cancel the caller's independent source producer.
                    raise ConnectionError("The CLI control stream was closed")
                if getattr(self, "end_reason", None) is not None:
                    return None
                raise ValueError("The CLI control stream ended")
            if task not in done:
                raise ValueError("The CLI stream did not become ready")
            try:
                return task.result()
            except ConnectionClosedOK:
                # Media and control use independent sockets. A normal media
                # close may arrive before its terminal control message. Use the
                # same bounded close grace as the SDK socket connections.
                done, _ = await asyncio.wait((self.control_task,), timeout=1)
                if self.control_task in done and not self.control_task.cancelled():
                    self.control_task.result()
                    if getattr(self, "end_reason", None) is not None:
                        return None
                raise
        finally:
            for pending in (task, finishing):
                if not pending.done():
                    pending.cancel()
            await asyncio.gather(task, finishing, return_exceptions=True)

    async def refresh_source_debug(self):
        if not self.source_debug_supported or self.closed or self.stopped:
            return None
        control = self.control
        if control is None:
            return None
        await control.send(
            json.dumps(
                dict(
                    type="model.context.source-debug.get",
                    epoch=self.descriptor["epoch"],
                )
            )
        )
        return self.source_debug_condition if control is self.control else None

    async def _control(self):
        async for wire in self.control:
            value = json.loads(wire)
            kind = value.get("type")
            if kind == "model.context.source-debug.result":
                if value.get("epoch") == self.descriptor["epoch"]:
                    self.source_debug_condition = value.get("conditions", [])
                continue
            if isinstance(kind, str) and (
                kind.startswith("model.context.") or kind.startswith("model.condition")
            ):
                if self.model_context is None:
                    raise ValueError("MODEL_CONTEXT_CAPABILITY_UNSUPPORTED")
                self.model_context.offer(value)
                continue

            if kind in {"error", "producer.error"}:
                raw_code = str(
                    value.get("code") or value.get("message") or "CLI_SOURCE_ERROR"
                )
                matched = re.search(r"\b[A-Z][A-Z0-9_]{2,100}\b", raw_code)
                safe_code = matched.group() if matched else "CLI_SOURCE_ERROR"
                logging.getLogger(__name__).error(
                    "CLI source error trace_id=%s code=%s",
                    getattr(self.room, "trace_id", None),
                    safe_code,
                )
                raise ValueError("CLI source error: " + safe_code)
            # Terminal SDK events may omit epoch; data messages may not. An
            # explicit epoch still fences stale terminal events from another run.
            if kind == "session.ended":
                if "epoch" in value and value["epoch"] != self.descriptor["epoch"]:
                    raise ValueError("The CLI session was reset")
                if value.get("reason") not in {"stopped", "idle-timeout"}:
                    raise ValueError("The CLI source failed")
                self.end_reason = value["reason"]
                self.stopped = True
                return
            if value.get("epoch") != self.descriptor["epoch"]:
                raise ValueError("The CLI session was reset")
            if kind == "session.ready":
                self.source_debug_supported = value.get("sourceDebug") is True
                if self.external_input and value.get("role") == "spectator":
                    # The browser owns its SDK input lease. Model/source capture
                    # observes media/UI without competing for that controller.
                    self.ready.set()
                    continue
                if value.get("role") != "controller":
                    # A resumed private viewer may await its predecessor's
                    # asynchronous release before the SDK promotes it.
                    if (
                        self.descriptor.get("accessMode") == "capability"
                        and value.get("role") == "spectator"
                    ):
                        continue
                    raise ValueError("The CLI did not grant player control")
                self.controller_ready = True
                if self.descriptor.get("accessMode") != "capability":
                    self.ready.set()
            elif kind == "input.binding":
                if (
                    self.descriptor.get("accessMode") != "capability"
                    or not self.controller_ready
                    or not safe_int(value.get("bindingSequence"), 1)
                ):
                    raise ValueError("The CLI input binding is invalid")
                if (
                    self.binding_sequence is not None
                    and self.binding_sequence != value["bindingSequence"]
                ):
                    raise ValueError("The CLI input binding changed")
                self.binding_sequence = value["bindingSequence"]
                await self.control.send(
                    json.dumps(
                        {
                            "type": "input.binding.applied",
                            "epoch": self.descriptor["epoch"],
                            "bindingSequence": self.binding_sequence,
                        }
                    )
                )
                self.ready.set()
            elif kind == "source.reset":
                self.scene_occurrences.clear()
                await self._source_reset(value)
            elif kind == "scene.occurrence":
                from .scene_transition import read_scene_occurrence

                if len(self.scene_occurrences) < 128:
                    self.scene_occurrences.append(
                        read_scene_occurrence(value.get("occurrence"))
                    )
            elif kind == "media.config":
                config = value.get("config", {})
                if self.descriptor.get("media", {}).get("kind") == "pixels":
                    if not safe_int(value.get("mediaGeneration"), 1) or config != dict(
                        kind="pixels",
                        pixelFormat="rgb24",
                        widthPixels=self.spec.width,
                        heightPixels=self.spec.height,
                        rowStrideBytes=self.spec.width * 3,
                        colorSpace="srgb",
                        fps=self.descriptor.get("fps", self.spec.fps),
                    ):
                        raise ValueError("Unsupported CLI pixel configuration")
                elif config.get("codec") != "vp8" or (
                    config.get("codedWidth"),
                    config.get("codedHeight"),
                ) != (self.spec.width, self.spec.height):
                    raise ValueError("Unsupported CLI video configuration")
                if self.config and (
                    self.config != value
                    or getattr(self, "pending_reset", None) is not None
                ):
                    await self._reconfigure(value)
                self.config = value
                self.configured.set()
            elif kind == "area.transition":
                self._accept_area_transition(value)
            elif kind == "area.frame":
                evidence = read_area_frame(value)
                if not self.area_transition or canonical(
                    evidence["identity"]
                ) != canonical(self.area_transition["status"].get("transition")):
                    continue
                self.area_frames[evidence["frame"]["outputFrameId"]] = evidence
                while (
                    len(self.area_frames) > AREA_FRAME_HISTORY_COUNT
                    or sum(len(canonical(item)) for item in self.area_frames.values())
                    > AREA_FRAME_HISTORY_BYTES
                ):
                    self.area_frames.popitem(last=False)
            elif kind == "area.presentation-result":
                result = read_area_result(value)
                pending = self.area_pending.get(result["requestId"])
                if (
                    pending
                    and pending[0] == result["operationId"]
                    and not pending[1].done()
                ):
                    pending[1].set_result(result)
            elif kind == "audio.snapshot":
                from .project_audio import read_audio_snapshot

                snapshot = read_audio_snapshot(value)
                if (
                    snapshot["sessionId"] != self.descriptor["sessionId"]
                    or snapshot["epoch"] != self.descriptor["epoch"]
                ):
                    raise ValueError("The CLI audio source identity changed")
                previous = self.audio_snapshot
                if previous and (
                    snapshot["source"]["presentationId"]
                    != previous["source"]["presentationId"]
                    or snapshot["source"]["sdkEpoch"] != previous["source"]["sdkEpoch"]
                    or snapshot["completeThroughUs"] < previous["completeThroughUs"]
                ):
                    raise ValueError("The CLI audio source clock changed")
                # Snapshots contain the complete bounded event history and current
                # voices; transport reconnect never rebases their source clock.
                self.audio_snapshot = snapshot
                key = self._audio_key(snapshot)
                old = self.audio_history.pop(key, None)
                if old is not None:
                    self.audio_history_bytes -= len(canonical(old))
                self.audio_history[key] = snapshot
                self.audio_history_bytes += len(canonical(snapshot))
                while (
                    len(self.audio_history) > 128 or self.audio_history_bytes > 8 << 20
                ):
                    _, retired = self.audio_history.popitem(last=False)
                    self.audio_history_bytes -= len(canonical(retired))
            elif kind in {"ui.snapshot", "ui.commit", "ui.clock"}:
                if (
                    kind == "ui.snapshot"
                    and self.ui_bootstrap_time is None
                    and self.descriptor.get("media", {}).get("kind") == "pixels"
                ):
                    bootstrap = value["snapshot"].get("effectiveSourceTimeUs")
                    if type(bootstrap) is not int or not 0 <= bootstrap <= 2**53 - 1:
                        raise ValueError("The CLI UI bootstrap clock is invalid")
                    self.ui_bootstrap_time = bootstrap
                await self.renderer.consume_ui(value)
                self.ui_bytes += len(canonical(value))
                if len(self.ui_events) >= 4096 or self.ui_bytes > 8 * 1024 * 1024:
                    raise ValueError("The UI history exceeded its limit")
                self.ui_events.append(value)
                content = value.get(
                    "snapshot", value.get("commit", value.get("clock", value))
                )
                async with self.ui_changed:
                    self.ui_revision = max(
                        self.ui_revision,
                        content.get("revision", content.get("uiRevision", -1)),
                    )
                    self.ui_through = max(
                        self.ui_through, content.get("completeThroughUs", -1)
                    )
                    self.ui_changed.notify_all()
            elif kind in {"input.receipt", "action.receipt"}:
                if len(self.receipts) >= 4096:
                    raise ValueError("The input receipt queue exceeded its limit")
                self.receipts.append(value)
            elif kind in {"session.reset", "session.ended", "error"}:
                raise ValueError("The CLI session ended or reset")

    async def _source_reset(self, value):
        self.source_debug_condition = []
        raise ValueError("The CLI source was reset")

    async def _reconfigure(self, value):
        raise ValueError("The CLI video generation changed")

    @staticmethod
    def _audio_key(value):
        source = value["source"]
        return (
            value["sessionId"],
            value["epoch"],
            source["presentationId"],
            source["sdkEpoch"],
            source["sourceFrameId"],
            source["sourceTimeUs"],
        )

    def _frame_audio(self, header):
        # Audio is auxiliary: independent control transport may lag pixels.
        # Preserve exact available proof, otherwise let the Player remain silent.
        if self.descriptor.get("audio") is None:
            return {}
        snapshot = self.audio_history.get(self._audio_key(header))
        return (
            {"audio_snapshot": copy.deepcopy(snapshot)} if snapshot is not None else {}
        )

    def _take_ui_events(self):
        # BrowserUiTimeline admits at most 512 events per ingest. A slow first
        # frame must not turn the larger source history into an invalid batch.
        events = self.ui_events[:UI_EVENT_BATCH_LIMIT]
        self.ui_events = self.ui_events[UI_EVENT_BATCH_LIMIT:]
        self.ui_bytes = sum(len(canonical(value)) for value in self.ui_events)
        return events

    async def frames(self):
        import av

        decoder = av.CodecContext.create("vp8", "r")
        index = 0
        while not self.stopped:
            wire = await self._wait_or_error(
                self.capture_wait.receive(
                    self.media.recv(), started=self.ready.is_set()
                ),
                timeout=None,
            )
            if self.finish_requested.is_set() or self.end_reason is not None:
                return
            header, payload = decode_packet(wire, self.descriptor, self.spec)
            await self._wait_or_error(self.configured.wait())
            if self.finish_requested.is_set() or self.end_reason is not None:
                return
            if header["mediaGeneration"] != self.config["mediaGeneration"]:
                raise ValueError("The CLI video generation changed")
            if self.last_id is None and header["type"] != "key":
                # Request a fresh snapshot/keyframe using the documented API.
                await self.control.send(
                    json.dumps({"type": "ui.resync", "epoch": self.descriptor["epoch"]})
                )
                continue
            if self.last_id is not None and (
                header["outputFrameId"] != self.last_id + 1
                or header["outputPtsUs"] <= self.last_pts
            ):
                raise ValueError("The CLI video lost a frame")

            async def ui_ready():
                async with self.ui_changed:
                    await self.ui_changed.wait_for(
                        lambda: (
                            self.ui_revision >= header["uiRevision"]
                            and self.ui_through >= header["source"]["sourceTimeUs"]
                        )
                    )

            await self._wait_or_error(ui_ready())
            if self.finish_requested.is_set() or self.end_reason is not None:
                return
            with stage("cli_vp8_decode", self.room.trace_id, service="room_runtime"):
                decoded = await asyncio.to_thread(decoder.decode, av.Packet(payload))
                if len(decoded) != 1:
                    raise ValueError("The CLI video frame could not be decoded")
                rgb = await asyncio.to_thread(
                    lambda: decoded[0].to_ndarray(format="rgb24").tobytes()
                )
            # The compositor enforces exact revision/time coverage at render().
            states = self._take_ui_events()
            receipts, self.receipts = self.receipts, []
            yield EngineFrame(
                index,
                self.spec.pts(index),
                rgb,
                {"ui_events": states, **self._frame_audio(header)},
                observed_input={
                    "receipts": receipts,
                    "mapping": "receipt_evidence_only",
                },
                provenance={
                    "cli_frame_header": header,
                    **self._area_provenance(header),
                },
            )
            self.last_id, self.last_pts = header["outputFrameId"], header["outputPtsUs"]
            index += 1

    async def capture_control(self, action, request_id):
        if action not in {"hold", "resume"}:
            raise ValueError("Invalid capture action")
        import httpx

        if action == "hold":
            self.capture_wait.pending(request_id)
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                self.descriptor["baseUrl"]
                + "/v1/sessions/"
                + self.descriptor["sessionId"]
                + "/capture",
                headers={"Authorization": "Bearer " + self.descriptor["accessToken"]},
                json=dict(action=action, requestId=request_id),
            )
            response.raise_for_status()
            result = response.json()
        if result.get("requestId") != request_id or result.get("state") != (
            "held" if action == "hold" else "running"
        ):
            raise ValueError("Invalid capture receipt")
        if action == "hold":
            frame = result.get("frame")
            if not isinstance(frame, dict) or any(
                frame.get(key) != self.descriptor.get(key)
                for key in ("sessionId", "epoch")
            ):
                raise ValueError("Invalid capture frame identity")
            self.capture_wait.held(request_id)
        else:
            self.capture_wait.resumed(request_id)
        return result

    async def send_input(self, sequence, payload):
        if self.external_input:
            raise ValueError("The browser owns player input")
        # Socket readiness alone does not authorize private input. The SDK must
        # offer the binding and receive our acknowledgement on this connection.
        if self.descriptor.get("accessMode") == "capability":
            await self._wait_or_error(self.ready.wait())
            if self.finish_requested.is_set() or self.end_reason is not None:
                raise ValueError("The CLI session ended before player input")
        binding = (
            {"bindingSequence": self.binding_sequence}
            if self.descriptor.get("accessMode") == "capability"
            else {}
        )
        if sequence != self.sequence + 1:
            raise ValueError("Invalid CLI input sequence")
        actions = payload.get("actions", [])
        if not isinstance(actions, list) or len(actions) > 32:
            raise ValueError("Invalid UI actions")
        messages = []
        catalog = self.descriptor.get("ui", {}).get("catalog", {}).get("actions", {})
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError("Invalid UI action")
            name, params = action.get("name"), action.get("params", {})
            action_id = action.get("id") or new_id()
            if not isinstance(name, str) or name not in catalog:
                raise ValueError("The UI action is not declared by this world")
            if (
                not isinstance(params, dict)
                or not isinstance(action_id, str)
                or len(action_id) > 128
            ):
                raise ValueError("Invalid UI action parameters")
            messages.append(
                {
                    "type": "input.action",
                    "epoch": self.descriptor["epoch"],
                    "actionId": action_id,
                    **binding,
                    "name": name,
                    "params": params,
                }
            )
        await self.control.send(
            json.dumps(
                {
                    "type": "input.state",
                    "epoch": self.descriptor["epoch"],
                    "input": cli_input(sequence, payload),
                    **binding,
                }
            )
        )
        self.sequence = sequence
        # Semantic UI actions are declared by the world, alongside keyboard and
        # camera controls. Preserve their IDs so receipts can be archived.
        for message in messages:
            await self.control.send(json.dumps(message))

    async def bind_model_context(self, **kwargs):
        if self.model_context is not None:
            await self.model_context.bind(**kwargs)
        return self.model_context

    async def read_model_context_resource(self, handle):
        # Construct the only allowed resource route from the final authenticated
        # descriptor; never send its credential to an author-supplied URL.
        from urllib.parse import quote

        from .source_conditions import HANDLE, keys

        keys(handle, HANDLE)
        descriptor = self.descriptor
        capability = descriptor.get("modelContext", {})
        if any(
            handle.get(k) != expected
            for k, expected in dict(
                sessionId=descriptor["sessionId"],
                epoch=descriptor["epoch"],
                gameBuildHash=capability.get("gameBuildHash"),
                configHash=capability.get("configHash"),
                compilerVersion=1,
            ).items()
        ):
            raise ValueError("MODEL_CONTEXT_RESOURCE_IDENTITY")
        identity(handle["resource"])
        query = urlencode(
            {k: handle[k] for k in ("epoch", "gameBuildHash", "configHash", "sha256")}
        )
        url = (
            descriptor["baseUrl"].rstrip("/")
            + "/v1/sessions/"
            + quote(descriptor["sessionId"], safe="")
            + "/model-context/resources/"
            + quote(handle["resource"], safe="")
            + "?"
            + query
        )
        async with httpx.AsyncClient(
            timeout=30, follow_redirects=False, trust_env=False
        ) as client:
            async with client.stream(
                "GET",
                url,
                headers={"Authorization": "Bearer " + descriptor["accessToken"]},
            ) as response:
                if response.status_code != 200:
                    raise ValueError("MODEL_CONTEXT_RESOURCE_FAILED")
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > 8 << 20:
                        raise ValueError("MODEL_CONTEXT_RESOURCE_SIZE")
                    chunks.append(chunk)
                return b"".join(chunks)

    async def finish(self):
        # Wake pending pixel/context reads before retiring their owner. A user
        # finish ends capture normally; transport/reset failures still raise.
        self.finish_requested.set()
        if self.model_context is not None:
            await self.model_context.close()
        self.stopped = True
        if self.control and self.end_reason is None:
            await self.control.send(
                json.dumps({"type": "input.release", "epoch": self.descriptor["epoch"]})
            )

    async def close(self):
        # Renewal and Room retirement may request close concurrently. Cancelling
        # either caller must not interrupt cleanup or make a later caller skip it.
        if self._close_task is None:
            self.closed = True
            self.stopped = True
            self._close_task = asyncio.create_task(self._close_resources())
        await asyncio.shield(self._close_task)

    async def _close_resources(self):
        try:
            if self.model_context is not None:
                await self.model_context.close()
        finally:
            self.area_frames.clear()
            for _, future in self.area_pending.values():
                if not future.done():
                    future.set_exception(ConnectionError("AREA_PRESENTATION_UNKNOWN"))
            if self.control_task:
                self.control_task.cancel()
                await asyncio.gather(self.control_task, return_exceptions=True)
            try:
                # Drain both sockets even if model-context cleanup failed.
                await asyncio.gather(
                    *(
                        connection.close()
                        for connection in (self.media, self.control)
                        if connection is not None
                    ),
                    return_exceptions=True,
                )
            finally:
                if self.renderer:
                    await self.renderer.close()


class NativeCLIView(CLIView):
    """Two-frame window, including the frame held by the downstream consumer."""

    def __init__(self, room):
        super().__init__(room)
        self.on_source_reset = None
        self.source_identity = None
        self.pending_reset = None
        self.credited_generation = None
        self.credited_transport = None
        self.first_generation = self.retired_generation = None
        self.last_frame_identity = self.last_frame_transport = None
        self.scene_entry_lock = asyncio.Lock()
        self.scene_entry = None
        self.scene_entry_target = None
        self.scene_entry_commit = None
        self.scene_entry_generation = None
        self.scene_entry_media_ready = asyncio.Event()
        self.scene_entry_active = None

    async def scene_entry_control(self, command):
        """Drive the dedicated source with its CLI owner credential."""
        action, preparation_id = command.get("action"), command.get("preparationId")
        states = {
            "hold": "held",
            "prepare": "prepared",
            "snapshot": "snapshot",
            "publish": "publishing",
            "frame": "publishing",
            "commit": "committed",
            "activate": "active",
        }
        if (
            action not in states
            or not isinstance(preparation_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", preparation_id)
            or self.descriptor.get("accessMode") == "capability"
        ):
            raise ValueError("SCENE_ENTRY_SOURCE_UNAVAILABLE")
        async with self.scene_entry_lock:
            new_hold = action == "hold" and self.scene_entry is None
            if action == "hold":
                if self.scene_entry is None:
                    if self.pending_reset is not None or self.config is None:
                        raise ValueError("SCENE_ENTRY_SOURCE_UNAVAILABLE")
                    self.capture_wait.pending(preparation_id)
                    self.scene_entry = preparation_id
                    self.scene_entry_target = copy.deepcopy(command["target"])
                    self.scene_entry_media_ready.clear()
                    self.scene_entry_active = None
                elif (
                    self.scene_entry != preparation_id
                    or command.get("target") != self.scene_entry_target
                ):
                    raise ValueError("SCENE_ENTRY_SOURCE_BUSY")
            elif (
                action == "activate"
                and self.scene_entry is None
                and self.scene_entry_active
            ):
                old_id, presented, receipt = self.scene_entry_active
                if preparation_id == old_id and command.get("presented") == presented:
                    return copy.deepcopy(receipt)
                raise ValueError("SCENE_ENTRY_SOURCE_STALE")
            elif self.scene_entry != preparation_id:
                raise ValueError("SCENE_ENTRY_SOURCE_STALE")

            if action == "publish" and self.scene_entry_generation is None:
                # Config can arrive over the original WS before HTTP settles.
                self.scene_entry_generation = self.config["mediaGeneration"] + 1

            async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
                response = await client.post(
                    f"{self.descriptor['baseUrl']}/v1/sessions/{self.descriptor['sessionId']}/scene-entry",
                    headers={
                        "Authorization": "Bearer " + self.descriptor["accessToken"]
                    },
                    json=command,
                )
                if new_hold and response.status_code in (403, 404, 405):
                    self.capture_wait.resumed(preparation_id)
                    self.scene_entry = None
                    self.scene_entry_target = None
                response.raise_for_status()
                receipt = response.json()
            if not isinstance(receipt, dict) or receipt.get("state") != states[action]:
                raise ValueError("SCENE_ENTRY_SOURCE_RECEIPT_INVALID")
            if action == "hold" and not self.capture_wait.confirmed:
                self.capture_wait.held(preparation_id)
            elif action == "snapshot":
                from .scene_transition import read_scene_condition, read_scene_snapshot

                snapshot = dict(
                    state="snapshot",
                    **read_scene_snapshot(receipt, self.scene_entry_target),
                )
                if "condition" in receipt:
                    snapshot["condition"] = read_scene_condition(
                        receipt["condition"], self.descriptor
                    )
                return snapshot
            elif action == "frame" and receipt.get("frame") is not None:
                frame = read_frame_identity(receipt["frame"])
                if frame["mediaGeneration"] != self.scene_entry_generation or any(
                    frame[key] != self.descriptor[key] for key in ("sessionId", "epoch")
                ):
                    raise ValueError("SCENE_ENTRY_SOURCE_RECEIPT_INVALID")
            elif action == "commit":
                tick = receipt.get("commitTick")
                if not safe_int(tick, 1) or self.scene_entry_commit not in (None, tick):
                    raise ValueError("SCENE_ENTRY_SOURCE_RECEIPT_INVALID")
                self.scene_entry_commit = tick
            elif action == "activate":
                tick = receipt.get("commitTick")
                # Condition commit and gameplay activation are separate fixed
                # ticks. Gameplay starts only after the presented target frame.
                minimum = command["presented"]["source"]["simulationTick"] + 1
                if self.scene_entry_commit is not None:
                    minimum = max(minimum, self.scene_entry_commit + 1)
                if not safe_int(tick, minimum):
                    raise ValueError("SCENE_ENTRY_SOURCE_RECEIPT_INVALID")
                self.scene_entry_active = (
                    preparation_id,
                    copy.deepcopy(command["presented"]),
                    copy.deepcopy(receipt),
                )
                self.scene_entry = self.scene_entry_generation = None
                self.scene_entry_target = None
                self.scene_entry_commit = None
            return receipt

    def credit_window(self):
        return min(2, MAX_PIXELS // self.spec.frame_bytes)

    async def _source_reset(self, value):
        if (
            set(value) != {"type", "epoch", "presentationId", "sdkEpoch"}
            or not safe_int(value.get("epoch"))
            or not safe_int(value.get("sdkEpoch"))
            or not isinstance(value.get("presentationId"), str)
            or not 1 <= len(value["presentationId"]) <= 128
        ):
            raise ValueError("Invalid CLI source reset")
        if self.scene_entry is not None:
            raise ValueError("SCENE_ENTRY_SOURCE_CHANGED")
        current = (value["presentationId"], value["sdkEpoch"])
        if current == self.source_identity:
            return
        if self.pending_reset is not None or (
            self.source_identity is not None and current[1] <= self.source_identity[1]
        ):
            raise ValueError("The CLI source reset is stale")
        self.source_identity = current
        self.source_debug_condition = []
        self.pending_reset = value
        if self.model_context is not None:
            self.model_context.retire_source(value)
        elif self.on_source_reset is not None:
            # The native control reader validates reset identity. Its source
            # owner fences synchronously and retires model work independently.
            self.on_source_reset(value)
        self.configured.clear()
        # Native reset revokes old UI/input/AREA and model-context evidence,
        # while retaining the Room session, source clock and world resources.
        self.receipts.clear()
        self.audio_snapshot = None
        self.audio_history.clear()
        self.audio_history_bytes = 0
        self.ui_events = [value]
        self.ui_bytes = len(canonical(value))
        self.ui_revision = self.ui_through = -1
        self.ui_bootstrap_time = None
        self.area_transition = None
        self.area_frames.clear()
        for _, future in self.area_pending.values():
            if not future.done():
                future.set_exception(ValueError("AREA_PRESENTATION_STALE"))
        await self.renderer.consume_ui(value)
        await self.control.send(
            json.dumps({"type": "ui.resync", "epoch": self.descriptor["epoch"]})
        )

    def bind_source_reset(self, callback):
        self.on_source_reset = callback

    async def _reconfigure(self, value):
        generation = value.get("mediaGeneration")
        scene_publish = self.scene_entry_generation == generation
        if (
            (self.pending_reset is None and not scene_publish)
            or not safe_int(generation, 1)
            or generation != self.config["mediaGeneration"] + 1
            or value["config"] != self.config["config"]
        ):
            raise ValueError("The CLI video generation changed")
        if scene_publish:
            if (
                self.pending_reset is not None
                or self.capture_wait.request_id != self.scene_entry
            ):
                raise ValueError("SCENE_ENTRY_SOURCE_CHANGED")
            self.capture_wait.resumed(self.scene_entry)
        self.pending_reset = None
        self.retired_generation = self.config["mediaGeneration"]
        self.config = value
        if self.credited_generation is not None:
            await self.grant(self.credit_window())
        if scene_publish:
            self.scene_entry_media_ready.set()

    async def grant(self, frames=1, *, generation=None, transport=None):
        current = self.config["mediaGeneration"]
        # A held frame belongs to its original media generation and socket.
        # Reset/reconnect grants a separate window to the replacement owner.
        if (generation is not None and generation != current) or (
            transport is not None and transport != self.transport_generation
        ):
            return
        self.credited_generation = current
        if self.first_generation is None:
            self.first_generation = current
        transport = self.transport_generation
        try:
            await self.control.send(
                json.dumps(
                    dict(
                        type="media.credit",
                        epoch=self.descriptor["epoch"],
                        mediaGeneration=current,
                        frames=frames,
                        bytes=frames * self.spec.frame_bytes,
                    )
                )
            )
            self.credited_transport = transport
        except (OSError, ConnectionClosed):
            if self.model_context is None:
                raise
            await self.recover_model_context_transport(transport)

    async def _wait_model_context_subscription(self):
        channel = self.model_context
        if channel is None or channel.suspended:
            return
        deadline = asyncio.timeout(MODEL_CONTEXT_SUBSCRIPTION_SECONDS)
        try:
            async with deadline:
                # Observe source/control termination while the model handshake
                # waits; only this deadline may degrade to preview mode.
                await self._wait_or_error(channel.bound.wait(), timeout=None)
        except TimeoutError:
            if not deadline.expired():
                raise
            channel.check()
            await channel.suspend("MODEL_CONTEXT_SUBSCRIPTION_TIMEOUT")
            return
        if not self.finish_requested.is_set() and self.end_reason is None:
            channel.check()

    async def frames(self):
        await self._wait_or_error(self.configured.wait())
        if self.finish_requested.is_set() or self.end_reason is not None:
            return
        # Match the public Node reader's bounded window. The next capture can
        # overlap this frame's downstream work; replenish only when the
        # consumer advances past its yield.
        await self._wait_model_context_subscription()
        if self.finish_requested.is_set() or self.end_reason is not None:
            return
        window = self.credit_window()
        if window < 1:
            raise ValueError("The CLI frame exceeds its pixel credit budget")
        await self.grant(window)
        index = 0
        origin = None
        missing_sample_recovered = False
        received_transport = None
        while not self.stopped:
            transport = self.transport_generation
            if self.credited_transport != transport:
                await self._wait_or_error(self.configured.wait())
                await self._wait_model_context_subscription()
                if self.finish_requested.is_set() or self.end_reason is not None:
                    return
                await self.grant(window, transport=transport)
                if transport != self.transport_generation:
                    continue
            try:
                wire = await self._wait_or_error(
                    self.capture_wait.receive(
                        self.media.recv(), started=self.ready.is_set()
                    ),
                    timeout=None,
                )
            except (OSError, ConnectionClosed):
                if self.model_context is None:
                    raise
                await self.recover_model_context_transport(transport)
                continue
            if self.finish_requested.is_set() or self.end_reason is not None:
                return
            if transport != self.transport_generation:
                continue
            header, rgb = decode_native_packet(wire, self.descriptor, self.spec)
            generation = header["mediaGeneration"]
            if self.scene_entry is not None:
                if generation != self.scene_entry_generation:
                    continue
                await self._wait_or_error(self.scene_entry_media_ready.wait())
                if self.stopped or transport != self.transport_generation:
                    continue
            source = header["source"]
            bootstrap = received_transport != transport
            received_transport = transport
            if (
                bootstrap
                and self.last_frame_transport is not None
                and transport != self.last_frame_transport
                and header["mediaGeneration"] == self.config["mediaGeneration"]
                and (source["presentationId"], source["sdkEpoch"])
                == self.source_identity
                and canonical(header) == self.last_frame_identity
            ):
                # The broker replays one retained packet to a fresh socket and
                # charges its credit. It is already delivered, never a new UI or
                # model sample; only this exact first-packet identity may replay.
                await self.grant(
                    generation=header["mediaGeneration"], transport=transport
                )
                continue
            try:
                context_sample = (
                    # A sample may settle while control resets or ends the
                    # source. The terminal owner must win before pixels escape.
                    await self._wait_or_error(
                        self.model_context.frame(header), timeout=None
                    )
                    if self.model_context is not None
                    and not self.model_context.suspended
                    else None
                )
            except TimeoutError:
                if missing_sample_recovered:
                    await self.model_context.suspend(
                        "MODEL_CONTEXT_SAMPLE_RECOVERY_EXPIRED"
                    )
                    await self.grant(
                        generation=header["mediaGeneration"], transport=transport
                    )
                    continue
                missing_sample_recovered = True
                await self.recover_model_context_transport(transport)
                continue
            if self.finish_requested.is_set():
                return
            if transport != self.transport_generation:
                continue
            if (
                self.model_context is not None
                and not self.model_context.suspended
                and context_sample is None
            ):
                # Context recovery retired this exact sample while its pixel
                # was in flight. Replenish only the same media generation.
                await self.grant(
                    generation=header["mediaGeneration"], transport=transport
                )
                continue
            if not self.configured.is_set():
                await self._wait_or_error(self.configured.wait())
            if self.finish_requested.is_set() or self.end_reason is not None:
                return
            if (
                self.scene_entry is not None
                and generation != self.scene_entry_generation
            ):
                continue
            if (
                self.retired_generation is not None
                and self.first_generation is not None
                and self.first_generation <= generation <= self.retired_generation
            ):
                # At most the previous credit window can cross the independent
                # control/media sockets. Retired pixels carry no current UI.
                continue
            current = (source["presentationId"], source["sdkEpoch"])
            if generation != self.config["mediaGeneration"] or (
                self.source_identity is not None and current != self.source_identity
            ):
                raise ValueError("The CLI source generation changed")
            if self.last_id is not None and (
                header["outputFrameId"] <= self.last_id
                or source["sourceTimeUs"] <= self.last_pts
            ):
                raise ValueError("The native source clock moved backwards")
            self.source_identity = current
            missing_sample_recovered = False
            audio_states = self._frame_audio(header)

            # UI evidence arrives on the independent control socket. Forward
            # valid pixels immediately, including bootstrap pixels predating a
            # UI snapshot. Late events accompany a later reliable state message;
            # consumers must select exact historical UI or display pixels alone.
            states = self._take_ui_events()
            # Action receipts acknowledge handler settlement and have no
            # simulationTick. Forward them once without inventing a frame
            # mapping; only sampled input receipts wait for their source tick.
            receipts = [
                r
                for r in self.receipts
                if r.get("type") == "action.receipt"
                or r["simulationTick"] <= source["simulationTick"]
            ]
            self.receipts = [
                r
                for r in self.receipts
                if r.get("type") != "action.receipt"
                and r["simulationTick"] > source["simulationTick"]
            ]
            origin = source["sourceTimeUs"] if origin is None else origin
            self.last_frame_identity = canonical(header)
            self.last_frame_transport = transport
            frame_id, source_pts = header["outputFrameId"], source["sourceTimeUs"]
            occurrences = [
                row
                for row in self.scene_occurrences
                if row["simulationTick"] <= source["simulationTick"]
            ]
            self.scene_occurrences = [
                row
                for row in self.scene_occurrences
                if row["simulationTick"] > source["simulationTick"]
            ]
            yield EngineFrame(
                index,
                self.spec.pts(index),
                rgb,
                {"ui_events": states, **audio_states},
                observed_input={
                    "receipts": receipts,
                    "mapping": "receipt_evidence_only",
                },
                provenance={
                    "cli_frame_header": header,
                    **(
                        {"model_context_sample": context_sample}
                        if context_sample is not None
                        else {}
                    ),
                    "capture_pts_us": source["sourceTimeUs"] - origin,
                    "capture_origin_us": origin,
                    **({"scene_occurrences": occurrences} if occurrences else {}),
                    **self._area_provenance(header),
                },
            )
            self.last_id, self.last_pts = frame_id, source_pts
            index += 1
            if not self.stopped:
                await self.grant(generation=generation, transport=transport)


def load_local_ui_package(descriptor):
    """Consume only the verified same-host CLI handoff for this live session."""
    value = descriptor.get("localResources", {}).get("uiRoot")
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("The CLI must provide verified local UI resources")
    root = Path(value)
    manifest = root / "manifest.json"
    if (
        root.is_symlink()
        or manifest.is_symlink()
        or not manifest.is_file()
        or manifest.stat().st_size > 1024 * 1024
    ):
        raise ValueError("Invalid local CLI UI manifest")
    raw = manifest.read_bytes()
    if len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != descriptor.get(
        "uiBundleHash"
    ):
        raise ValueError("The local CLI UI bundle identity changed")
    return {"root": str(root), "manifest": json.loads(raw)}


async def prepare_publication(entry, world_id, timeout, *, expected_publication=None):
    process = await asyncio.create_subprocess_exec(
        os.environ.get("WHITEBOX_NODE", "node"),
        "--import",
        "tsx",
        str(Path(__file__).with_name("worldkit_publication_prepare.mjs")),
        cwd=entry.parents[3],
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=16384,
        start_new_session=True,
    )
    try:
        out, errors = await asyncio.wait_for(
            process.communicate(
                json.dumps(
                    {
                        "worldId": world_id,
                        "timeoutSeconds": min(120, timeout),
                        **(
                            {
                                "expectedPublication": publication_identity(
                                    expected_publication
                                ),
                                "transport": (
                                    "cdn"
                                    if resource_transport(os.environ) == "auto"
                                    else resource_transport(os.environ)
                                ),
                            }
                            if expected_publication is not None
                            else {}
                        ),
                    }
                ).encode()
            ),
            min(120, timeout) + 1,
        )
        if process.returncode:
            code = errors.decode(errors="replace").strip()
            if code not in {
                "WTP_PLAYABLE_PREPARATION_FAILED",
                "WTP_PLAYABLE_PUBLICATION_CHANGED",
                "WTP_PLAYABLE_AUTHENTICATION_REQUIRED",
                "WTP_REFERENCE_ORIGIN_MISMATCH",
            }:
                code = "WTP_PLAYABLE_PREPARATION_FAILED"
            raise WhiteboxDiagnostic(code)
        value = json.loads(out)
        if value.get("ready") is not True:
            raise WhiteboxDiagnostic("WTP_PLAYABLE_PREPARATION_FAILED")
        if expected_publication is not None and value.get(
            "publicationIdentity"
        ) != publication_identity(expected_publication):
            raise WhiteboxDiagnostic("WTP_PLAYABLE_PUBLICATION_CHANGED")
        return value["elapsedMs"]
    except asyncio.TimeoutError as error:
        raise WhiteboxDiagnostic("STREAM_STARTUP_TIMEOUT") from error
    finally:
        if process.returncode is None:
            try:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                await asyncio.wait_for(process.wait(), 5)
            finally:
                if process.returncode is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await process.wait()


async def cli_room(
    *,
    world_id,
    spec,
    trace_id,
    cli_fps=None,
    source_options=None,
    expected_publication=None,
):
    source_options = area_source_options(source_options or {})
    if expected_publication is not None:
        expected_publication = publication_identity(expected_publication)
    from .session import load_plugin

    entry = Path(os.environ["WHITEBOX_CLI_ENTRYPOINT"])
    compositor = os.environ.get(
        "WHITEBOX_CLI_UI_FACTORY",
        "sglang.multimodal_gen.runtime.realtime.whitebox.browser_ui:factory",
    )
    if not entry.is_absolute() or not entry.is_file() or not compositor:
        raise ValueError(
            "Configure the CLI entrypoint and a source-aware UI compositor"
        )
    renderer_factory = load_plugin(compositor)
    transport, timeout = resource_transport(os.environ), startup_seconds(os.environ)
    if source_options and (
        transport == "archive" or os.environ.get("WHITEBOX_CLI_CACHE_UPDATE") == "async"
    ):
        raise ValueError("AREA_SOURCE_OPTIONS_INVALID")
    fps = capture_fps(spec, os.environ, cli_fps)
    deadline = asyncio.get_running_loop().time() + timeout
    temporary = (
        tempfile.TemporaryDirectory(prefix="whitebox-delivery-")
        if transport == "archive"
        else None
    )
    prepared = process = room = spawn = None
    # A foreground CLI owns its browser and views in this private process group.
    # A resident-service client does not own the service's execution resources.
    owns_process_group = stream_policy_arguments(os.environ)[1] == "off"
    metrics = {}
    try:
        if transport == "archive":
            prepared = await prepare_delivery(entry, world_id, temporary.name, trace_id)
        for attempt in range(2):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise WhiteboxDiagnostic("STREAM_STARTUP_TIMEOUT")
            env = {
                **os.environ,
                "WHITEBOX_CLI_STARTUP_TIMEOUT_SECONDS": str(math.ceil(remaining)),
                # A controlled source cannot refresh to another publication after ready.
                **(
                    {"WHITEBOX_CLI_CACHE_UPDATE": "sync"}
                    if expected_publication is not None
                    else {}
                ),
            }
            launched = asyncio.get_running_loop().time()

            async def launch():
                # The spawn task also owns inherited descriptors until fork/exec
                # has completed, including cancellation before the handle returns.
                with cli_log_sink(env) as (child_environment, collector_descriptors):
                    return await asyncio.create_subprocess_exec(
                        *cli_arguments(
                            entry,
                            world_id,
                            spec,
                            env,
                            delivery=prepared["archive"] if prepared else None,
                            cli_fps=cli_fps,
                            **(
                                {"source_options": source_options}
                                if source_options
                                else {}
                            ),
                        ),
                        env=child_environment,
                        pass_fds=collector_descriptors,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        limit=1048576,
                        start_new_session=True,
                    )

            spawn = asyncio.create_task(launch())
            process = await asyncio.shield(spawn)
            room = CLIRoom(process, None, spec, renderer_factory, trace_id)
            room.resident_service = not owns_process_group
            room.temporary = temporary
            room.startup_metrics = metrics
            room.diagnostics = asyncio.create_task(
                cli_diagnostics(
                    process, trace_id, metrics, room.service_cleanup, room.service_ready
                )
            )
            line = await asyncio.wait_for(process.stdout.readline(), remaining + 5)
            if not line.endswith(b"\n"):
                diagnostic = await asyncio.wait_for(asyncio.shield(room.diagnostics), 2)
                # Only a known failed explicit-CDN startup can request derived
                # publication. Warm local snapshots take no extra metadata trip.
                if (
                    expected_publication is None
                    and attempt == 0
                    and transport == "cdn"
                    and getattr(diagnostic, "code", None) == "WTP_PLAYABLE_NOT_READY"
                ):
                    await room.close()
                    room = None
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise WhiteboxDiagnostic("STREAM_STARTUP_TIMEOUT")
                    metrics["publicationPreparationMs"] = await prepare_publication(
                        entry, world_id, remaining
                    )
                    continue
                raise diagnostic or WhiteboxDiagnostic("CLI_STARTUP_FAILED")
            room.descriptor = validate_descriptor(json.loads(line), spec, fps=fps)
            validate_area_descriptor(room.descriptor, source_options)
            if expected_publication is not None:
                validate_frozen_descriptor(room.descriptor, expected_publication)
            if prepared:
                validate_prepared_delivery(room.descriptor, prepared)
            room.ui_package = await asyncio.to_thread(
                load_local_ui_package, room.descriptor
            )
            room.prepared_delivery = (
                {key: prepared[key] for key in ("reference", "receipt", "identity")}
                if prepared
                else {
                    "reference": {"input": world_id},
                    "receipt": None,
                    "identity": room.descriptor.get("artifact"),
                }
            )
            metrics["cliReadyMs"] = (
                asyncio.get_running_loop().time() - launched
            ) * 1000
            observe_duration(
                "cli_ready", metrics["cliReadyMs"], trace_id, service="room_runtime"
            )
            return room
        raise WhiteboxDiagnostic("CLI_STARTUP_FAILED")
    except BaseException as original_error:
        from .room_startup import (
            FactoryCleanupReceipt,
            drain_startup_cleanup,
            startup_cleanup_error,
        )

        async def cleanup():
            nonlocal process, room
            if process is None and spawn is not None:
                process = await spawn
            if process is not None and room is None:
                # Cancellation may win the shielded spawn reply. Keep the same
                # child and drain its native receipt before attempting release.
                room = CLIRoom(process, None, spec, renderer_factory, trace_id)
                room.resident_service = not owns_process_group
                room.temporary = temporary
                room.diagnostics = asyncio.create_task(
                    cli_diagnostics(
                        process,
                        trace_id,
                        metrics,
                        room.service_cleanup,
                        room.service_ready,
                    )
                )
            if process is not None and owns_process_group:
                from .worldkit_room import stop_process

                # A leader can exit before its browser. Reap the complete private
                # group, even when another cancellation arrives during cleanup.
                await stop_process(process)
            if room:
                await room.close()
            if temporary:
                temporary.cleanup()

        _, interrupted = await drain_startup_cleanup(cleanup())
        error = asyncio.CancelledError() if interrupted else original_error
        if isinstance(error, (Exception, asyncio.CancelledError)) and (
            spawn is None  # This invocation never initiated CLI execution.
            or (
                process is not None
                and process.returncode is not None
                and (
                    owns_process_group
                    or (
                        room is not None
                        and room.process is process
                        and process.returncode >= 0
                        and room.service_cleanup.get("confirmed") is True
                        and not room.service_cleanup.get("invalid")
                    )
                    or (
                        process.returncode > 0
                        and getattr(error, "sdk_not_created", False)
                    )
                )
            )
        ):
            receipt = FactoryCleanupReceipt(trace_id)
            logging.getLogger(__name__).info(
                "CLI startup cleanup confirmed trace_id=%s foreground=%s spawned=%s cancelled=%s",
                trace_id,
                owns_process_group,
                spawn is not None,
                isinstance(error, asyncio.CancelledError),
            )
            raise startup_cleanup_error(error, receipt) from error
        if interrupted:
            raise error
        raise
