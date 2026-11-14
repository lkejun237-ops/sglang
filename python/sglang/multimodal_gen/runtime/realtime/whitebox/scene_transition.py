# SPDX-License-Identifier: Apache-2.0
"""Room-owned cutscene state for the original native source session."""

import asyncio
import base64
import copy
import logging
import math
import re
import uuid
from collections import OrderedDict

from ..source_prepare import MAX_SNAPSHOT_BYTES, decode_snapshot_png, rgb_digest
from .native_pixels import read_frame_identity
from .render_contract import (
    MAX_REFERENCE_IMAGES,
    RenderError,
    canonical,
    identity,
    prompt_text,
)

PROTOCOL = "worldkit-scene-transition-v1"
MODEL_PROTOCOL = "worldkit-scene-model-transition-v1"
COMMANDS = {
    "scene.transition.request",
    "scene.transition.media",
    "scene.transition.presented",
    "scene.transition.retry",
    "scene.transition.retry_prefill",
}
BASE_FIELDS = {
    "type",
    "version",
    "op",
    "request_id",
    "source_session_id",
    "source_epoch",
    "connection_generation",
}


def entry(value):
    if not isinstance(value, dict) or set(value) != {"sceneId", "entryId"}:
        raise RenderError("INVALID_COMMAND")
    return {key: identity(item) for key, item in value.items()}


def read_scene_snapshot(receipt, target):
    """Validate the CLI-owned condition image independently from the live frames."""
    image = receipt.get("image")
    digest = receipt.get("entryDefinitionHash")
    encoded = receipt.get("png")
    if (
        receipt.get("state") != "snapshot"
        or entry(receipt.get("target")) != target
        or not isinstance(digest, str)
        or not re.fullmatch(r"[a-f0-9]{64}", digest)
        or not isinstance(image, dict)
        or not isinstance(encoded, str)
        or type(image.get("byteLength")) is not int
        or not 0 < image["byteLength"] <= MAX_SNAPSHOT_BYTES
        or len(encoded) != 4 * ((image["byteLength"] + 2) // 3)
    ):
        raise ValueError("SCENE_ENTRY_SNAPSHOT_INVALID")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != image["byteLength"] or rgb_digest(raw) != image.get("sha256"):
        raise ValueError("SCENE_ENTRY_SNAPSHOT_MISMATCH")
    width, height = image.get("width"), image.get("height")
    rgb = decode_snapshot_png(raw, width, height)
    return dict(
        target=target,
        entry_definition_sha256=digest,
        encoded_sha256=image["sha256"],
        png=raw,
        rgb=rgb,
        rgb_sha256=rgb_digest(rgb),
        width=width,
        height=height,
    )


def read_scene_condition(value, descriptor):
    """Keep the target's compiled condition scoped to the same CLI source."""
    capability = descriptor.get("modelContext") or {}
    if (
        not isinstance(value, dict)
        or set(value)
        != {"eventId", "configHash", "contentHash", "prompt", "references"}
        or not isinstance(value["configHash"], str)
        or value["configHash"] != capability.get("configHash")
        or not isinstance(value["contentHash"], str)
        or not re.fullmatch(r"[a-f0-9]{64}", value["contentHash"])
        or not isinstance(value["references"], list)
        or len(value["references"]) > MAX_REFERENCE_IMAGES
    ):
        raise ValueError("SCENE_ENTRY_CONDITION_INVALID")
    identity(value["eventId"])
    prompt_text(value["prompt"])
    aliases = set()
    for picture, ref in enumerate(value["references"], start=1):
        if not isinstance(ref, dict) or set(ref) != {
            "alias",
            "picture",
            "format",
            "sha256",
            "resource",
        }:
            raise ValueError("SCENE_ENTRY_CONDITION_INVALID")
        alias = identity(ref["alias"])
        digest = ref["sha256"]
        if (
            alias in aliases
            or type(ref["picture"]) is not int
            or ref["picture"] != picture
            or ref["format"] not in ("png", "jpeg", "webp")
            or not isinstance(digest, str)
            or not re.fullmatch(r"[a-f0-9]{64}", digest)
            or ref["resource"]
            != dict(
                sessionId=descriptor["sessionId"],
                epoch=descriptor["epoch"],
                gameBuildHash=capability.get("gameBuildHash"),
                configHash=value["configHash"],
                compilerVersion=1,
                resource=alias,
                sha256=digest,
            )
        ):
            raise ValueError("SCENE_ENTRY_CONDITION_INVALID")
        aliases.add(alias)
    return value


def read_scene_occurrence(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"eventId", "occurrenceId", "simulationTick"}
        or type(value["simulationTick"]) is not int
        or not 0 <= value["simulationTick"] < 2**53
    ):
        raise RenderError("INVALID_COMMAND")
    return dict(
        eventId=identity(value["eventId"]),
        occurrenceId=identity(value["occurrenceId"]),
        simulationTick=value["simulationTick"],
    )


def read_scene_transitions(value):
    """Validate the private CLI projection of the verified playable inventory."""
    if (
        not isinstance(value, dict)
        or set(value)
        != {"schemaVersion", "initial", "entries", "transitions", "videos"}
        or type(value["schemaVersion"]) is not int
        or value["schemaVersion"] != 1
        or not isinstance(value["entries"], list)
        or not 1 <= len(value["entries"]) <= 128 * 128
        or not isinstance(value["transitions"], list)
        or not isinstance(value["videos"], list)
        or not 1 <= len(value["transitions"]) <= 128
        or len(value["videos"]) > 128
    ):
        raise ValueError("Invalid scene transition inventory")
    videos = {}
    for row in value["videos"]:
        if not isinstance(row, dict) or set(row) != {
            "path",
            "sha256",
            "byteLength",
            "durationSeconds",
        }:
            raise ValueError("Invalid scene transition video")
        path, size, duration, digest = (
            row["path"],
            row["byteLength"],
            row["durationSeconds"],
            row["sha256"],
        )
        if (
            not isinstance(path, str)
            or not re.fullmatch(r"[^\\:%?#\x00-\x1f]{1,512}\.(?:mp4|webm)", path, re.I)
            or any(part in ("", ".", "..") for part in path.split("/"))
            or path in videos
            or type(size) is not int
            or not 0 < size <= 256 * 1024 * 1024
            or type(duration) not in (int, float)
            or not math.isfinite(duration)
            or duration <= 0
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError("Invalid scene transition video")
        videos[path] = row
    transitions, triggers = {}, {}
    for row in value["transitions"]:
        if not isinstance(row, dict) or set(row) - {"trigger", "video"} != {
            "id",
            "target",
        }:
            raise ValueError("Invalid scene transition declaration")
        transition_id = identity(row["id"])
        path = row.get("video")
        if transition_id in transitions or (
            "video" in row and (not isinstance(path, str) or path not in videos)
        ):
            raise ValueError("Invalid scene transition declaration")
        transitions[transition_id] = (entry(row["target"]), videos.get(path))
        if "trigger" in row:
            trigger = row["trigger"]
            if not isinstance(trigger, dict) or set(trigger) != {"eventId"}:
                raise ValueError("Invalid scene transition trigger")
            event_id = identity(trigger["eventId"])
            if event_id in triggers:
                raise ValueError("Duplicate scene transition trigger")
            triggers[event_id] = transition_id
    if {video["path"] for _, video in transitions.values() if video is not None} != set(
        videos
    ):
        raise ValueError("Invalid scene transition video coverage")
    entries = {canonical(entry(row)) for row in value["entries"]}
    initial = entry(value["initial"])
    if (
        len(entries) != len(value["entries"])
        or canonical(initial) not in entries
        or any(canonical(target) not in entries for target, _ in transitions.values())
    ):
        raise ValueError("Invalid scene transition target coverage")
    return initial, entries, transitions, triggers


class SceneTransition:
    def __init__(
        self,
        inventory,
        source,
        emit,
        remaining,
        *,
        model_lifecycle=None,
        verify_model_frame=None,
    ):
        self.active_entry, self.entries, self.transitions, self.triggers = (
            read_scene_transitions(inventory)
        )
        self.source, self.emit, self.remaining = source, emit, remaining
        self.model_lifecycle = model_lifecycle
        self.verify_model_frame = verify_model_frame
        self.revision, self.generation = 0, None
        self.current = None
        self.last = None
        self.requests = OrderedDict()
        self.occurrences = set()
        self.task = None
        self.closed = False
        self.deadline = None
        self.expiration = None
        self.model_input = None
        self.prefill_attempt = None

    @property
    def active(self):
        return self.current is not None

    def snapshot(self):
        row = self.current or self.last
        active = self.current or {"phase": "running"}
        receipt = {}
        if row is not None:
            receipt = {
                key: row[key]
                for key in (
                    "transition_id",
                    "target_entry",
                    "occurrence",
                    "request_id",
                    "media_state",
                    "scene_prepared",
                    "model_phase",
                    "condition_rgb_sha256",
                    "prefill",
                    "prefill_retryable",
                    "generation_cache",
                    "presented",
                    "error",
                )
                if key in row
            }
            receipt["has_video"] = row["video"] is not None
        return dict(
            protocol=PROTOCOL,
            revision=self.revision,
            active_entry=copy.deepcopy(self.active_entry),
            **copy.deepcopy(
                {
                    key: value
                    for key, value in active.items()
                    if key
                    not in {
                        "request_id",
                        "condition_rgb_sha256",
                        "prefill",
                        "prefill_retryable",
                        "generation_cache",
                        "presented",
                    }
                }
            ),
            **({"receipt": copy.deepcopy(receipt)} if receipt else {}),
        )

    async def changed(self):
        self.revision += 1
        await self.emit(self.snapshot())

    async def attach(self, generation):
        if generation is None or (
            self.generation is not None and generation <= self.generation
        ):
            return
        self.generation = generation
        row = self.current
        if row and row["phase"] in {"entering", "playing"} and not row["media_ended"]:
            row.update(
                playback_attempt=row["playback_attempt"] + 1, media_state="waiting"
            )
            await self.changed()

    def schedule(self, operation, row):
        async def run():
            try:
                async with asyncio.timeout_at(self.deadline):
                    await operation(row)
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                self.model_input = None
                row.update(phase="failed", error="SCENE_TRANSITION_TIMEOUT")
                await self.changed()
            except Exception:
                logging.getLogger(__name__).exception(
                    "Scene transition source operation failed"
                )
                self.model_input = None
                row.update(phase="failed", error="SCENE_TRANSITION_SOURCE_FAILED")
                await self.changed()

        self.task = asyncio.create_task(run())

    async def expire(self, row):
        await asyncio.sleep(max(0, self.deadline - asyncio.get_running_loop().time()))
        if self.current is not row or row["phase"] == "failed":
            return
        if self.task is not None and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if self.current is row and row["phase"] != "failed":
            self.model_input = None
            row.update(phase="failed", error="SCENE_TRANSITION_TIMEOUT")
            await self.changed()

    def begin(self, target, video, occurrence=None, *, model=False):
        if self.current is not None or self.closed:
            raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
        if model and self.model_lifecycle is None:
            raise RenderError("CAPABILITY_UNAVAILABLE")
        budget = min(240, self.remaining())
        if budget <= (video["durationSeconds"] if video else 0) + 15:
            raise RenderError("LIMIT_EXCEEDED")
        self.deadline = asyncio.get_running_loop().time() + budget
        self.model_input = None
        self.current = row = dict(
            transition_id=uuid.uuid4().hex,
            target_entry=target,
            video=video,
            phase="entering",
            playback_attempt=1,
            media_state="waiting" if video else "ended",
            media_ended=video is None,
            scene_prepared=False,
            first_frame=None,
        )
        if model:
            row["model_phase"] = "retiring"
        if occurrence is not None:
            row["occurrence"] = occurrence
        self.expiration = asyncio.create_task(self.expire(row))
        return row

    async def trigger(self, occurrence, *, model=False):
        occurrence = read_scene_occurrence(occurrence)
        event_id, occurrence_id = occurrence["eventId"], occurrence["occurrenceId"]
        transition = self.triggers.get(event_id)
        if transition is None or occurrence_id in self.occurrences:
            return
        if len(self.occurrences) >= 256:
            raise RenderError("LIMIT_EXCEEDED")
        self.occurrences.add(occurrence_id)
        row = self.begin(*self.transitions[transition], occurrence, model=model)
        await self.changed()
        if not self.closed:
            self.schedule(self.prepare, row)

    async def command(self, command, *, model=False):
        op = command["op"]
        fields = {
            "scene.transition.request": {"expected_revision", "target"}
            if "target" in command
            else {"expected_revision", "transition"},
            "scene.transition.media": {
                "transition_id",
                "playback_attempt",
                "media_state",
            },
            "scene.transition.retry": {"transition_id", "playback_attempt"},
            "scene.transition.retry_prefill": {"transition_id"},
            "scene.transition.presented": {"transition_id", "frame"},
        }[op]
        generation = command.get("connection_generation")
        if (
            set(command) != BASE_FIELDS | fields
            or command["type"] != "command"
            or type(command["version"]) is not int
            or command["version"] != 2
            or type(generation) is not int
            or generation != self.generation
        ):
            raise RenderError("INVALID_COMMAND")

        request = identity(command["request_id"])
        payload = canonical(
            {
                key: value
                for key, value in command.items()
                if key != "connection_generation"
            }
        )
        if request in self.requests:
            if self.requests[request] != payload:
                raise RenderError("REQUEST_ID_CONFLICT")
            return
        if len(self.requests) >= 256:
            raise RenderError("LIMIT_EXCEEDED")

        row = self.current
        operation = None
        if op == "scene.transition.request":
            if (
                row
                or type(command["expected_revision"]) is not int
                or command["expected_revision"] != self.revision
            ):
                raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
            video = None
            if "target" in command:
                target = entry(command["target"])
                if canonical(target) not in self.entries:
                    raise RenderError("INVALID_COMMAND")
            else:
                selected = self.transitions.get(identity(command["transition"]))
                if selected is None:
                    raise RenderError("INVALID_COMMAND")
                target, video = selected
            row = self.begin(target, video, model=model)
            row["request_id"] = request
            operation = self.prepare
        else:
            if row is None or command["transition_id"] != row["transition_id"]:
                raise RenderError("INVALID_COMMAND")
            if (
                asyncio.get_running_loop().time() >= self.deadline
                or row["phase"] == "failed"
            ):
                raise RenderError("RENDER_TRANSITION_IN_PROGRESS")
            if op in {"scene.transition.media", "scene.transition.retry"}:
                attempt = command["playback_attempt"]
                if (
                    type(attempt) is not int
                    or attempt != row["playback_attempt"]
                    or not row["video"]
                ):
                    raise RenderError("INVALID_COMMAND")
            if op == "scene.transition.media":
                event = command["media_state"]
                if (
                    row["phase"] != "playing"
                    or row["media_ended"]
                    or event not in ("playing", "ended", "failed")
                ):
                    raise RenderError("INVALID_COMMAND")
                if event == "ended" and row["media_state"] != "playing":
                    raise RenderError("INVALID_COMMAND")
                if row["media_state"] == "failed":
                    raise RenderError("INVALID_COMMAND")
                row.update(media_state=event, media_ended=event == "ended")
                if row["media_ended"] and row["scene_prepared"]:
                    row["phase"] = "starting"
                    operation = self.publish
            elif op == "scene.transition.retry":
                if row["phase"] != "playing" or row["media_state"] != "failed":
                    raise RenderError("INVALID_COMMAND")
                row.update(playback_attempt=attempt + 1, media_state="waiting")
            elif op == "scene.transition.retry_prefill":
                if row["phase"] != "playing" or not row.get("prefill_retryable"):
                    raise RenderError("INVALID_COMMAND")
                row.pop("prefill_retryable")
                row.pop("prefill", None)
                self.prefill_attempt = None
                operation = self.prefill
            else:
                if row["phase"] != "revealing":
                    raise RenderError("INVALID_COMMAND")
                try:
                    frame = read_frame_identity(command["frame"])
                except ValueError as error:
                    raise RenderError("INVALID_COMMAND") from error
                first = row["first_frame"]
                if (
                    any(
                        frame[key] != first[key]
                        for key in ("sessionId", "epoch", "mediaGeneration")
                    )
                    or frame["outputFrameId"] < first["outputFrameId"]
                    or any(
                        frame["source"][key] != first["source"][key]
                        for key in ("presentationId", "sdkEpoch")
                    )
                ):
                    raise RenderError("INVALID_COMMAND")
                if row.get("model_stream_id"):
                    if self.verify_model_frame is None:
                        raise RenderError("CAPABILITY_UNAVAILABLE")
                    self.verify_model_frame(row["model_stream_id"], frame)
                row["phase"] = "activating"

                async def operation(current):
                    await self.activate(current, frame)

        self.requests[request] = payload
        await self.changed()
        if operation and not self.closed:
            self.schedule(operation, row)

    async def prepare(self, row):
        model = row.get("model_phase") is not None
        if model:
            await self.model_lifecycle.fence()
        await self.source(
            dict(
                action="hold",
                preparationId=row["transition_id"],
                target=row["target_entry"],
            )
        )
        row["phase"] = "playing"
        await self.changed()
        await self.source(dict(action="prepare", preparationId=row["transition_id"]))
        if model:
            row["model_phase"] = "prefilling"
            await self.changed()
            snapshot = await self.source(
                dict(action="snapshot", preparationId=row["transition_id"])
            )
            if snapshot.get("rgb_sha256"):
                row["condition_rgb_sha256"] = snapshot["rgb_sha256"]

            self.model_input = await self.model_lifecycle.prepare(snapshot)
            await self.prefill(row)
            return
        await self.prepared(row)

    async def prefill(self, row):
        attempt = self.prefill_attempt = object()

        async def observe(message):
            if (
                self.prefill_attempt is not attempt
                or (self.current or self.last) is not row
            ):
                return
            kind = message.get("type")
            prefill = dict(row.get("prefill", {}), request_id=message["request_id"])
            cache_key = message.get("cache_key")
            if isinstance(cache_key, str) and re.fullmatch(r"[a-f0-9]{64}", cache_key):
                prefill["cache_key"] = cache_key
            if kind == "prefill.persisted":
                prefill.update(memory_ready=True, persistence="persisted")
            elif kind == "prefill.memory_ready":
                prefill["memory_ready"] = True
            elif kind == "prefill.failed" and message.get("memory_ready") is True:
                prefill.update(memory_ready=True, persistence="failed")
            elif kind == "prefill.failed":
                prefill["compute_failed"] = True
            if prefill != row.get("prefill"):
                row["prefill"] = prefill
                await self.changed()

        try:
            self.model_input["cache"] = await self.model_lifecycle.prefill(
                self.model_input, self.deadline, observe=observe
            )
        except RenderError as error:
            if error.code != "PREFILL_FAILED":
                raise
            row["prefill_retryable"] = True
            await self.changed()
            return
        row["model_phase"] = "prepared"
        await self.prepared(row)

    async def prepared(self, row):
        row["scene_prepared"] = True
        if row["media_ended"]:
            row["phase"] = "starting"
            await self.changed()
            await self.publish(row)
        else:
            await self.changed()

    async def publish(self, row):
        await self.source(dict(action="publish", preparationId=row["transition_id"]))
        while True:
            receipt = await self.source(
                dict(action="frame", preparationId=row["transition_id"])
            )
            if receipt["frame"] is not None:
                if self.model_input is not None:
                    row["model_phase"] = "starting"
                    await self.changed()
                    candidate = await self.model_lifecycle.generate(
                        self.model_input,
                        receipt["frame"],
                        self.deadline,
                        transition_id=row["transition_id"],
                    )
                    try:
                        await self.source(
                            dict(action="commit", preparationId=row["transition_id"])
                        )
                        row["model_stream_id"] = await self.model_lifecycle.start(
                            candidate
                        )
                    except BaseException:
                        await self.model_lifecycle.discard()
                        raise
                    row["model_phase"] = "ready"
                row.update(phase="revealing", first_frame=receipt["frame"])
                await self.changed()
                return
            await asyncio.sleep(0.05)

    async def activate(self, row, frame):
        await self.source(
            dict(action="activate", preparationId=row["transition_id"], presented=frame)
        )
        row["presented"] = frame
        self.active_entry, self.current, self.last = row["target_entry"], None, row
        self.model_input = None
        self.expiration.cancel()
        await self.changed()

    async def record_generation_cache(self, transition_id, cache):
        row = self.current or self.last
        if (
            row is None
            or row["transition_id"] != transition_id
            or "generation_cache" in row
        ):
            return
        row["generation_cache"] = {
            key: cache[key] for key in ("cache_key", "cache_result", "miss_reason")
        }
        await self.changed()

    async def close(self):
        self.closed = True
        tasks = [task for task in (self.task, self.expiration) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.model_input = None
