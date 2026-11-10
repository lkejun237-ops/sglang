# SPDX-License-Identifier: Apache-2.0
"""Bounded parallel encoders and a recoverable, asynchronous archive publisher."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .contracts import EngineFrame, VideoSpec, capture_pts, presentation_pts
from .metrics import stage

logger = logging.getLogger(__name__)


def room_event_max_bytes(event):
    # SDK batches allow 64 KiB and binding declarations 128 KiB, before their
    # transport and Room envelopes. Keep ordinary event and total queue budgets.
    payload = event.get("payload")
    message = payload.get("message") if isinstance(payload, dict) else None
    if (
        event.get("kind") in ("viewer_input_observed", "viewer_input_receipt")
        and isinstance(message, dict)
        and message.get("type") in ("input.semantic.batch", "input.semantic.binding")
    ):
        return 256 * 1024
    return 64 * 1024


def json_line(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n"
    )


def write_manifest(directory: Path, manifest: dict) -> None:
    temporary = directory / "manifest.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2)
    )
    temporary.replace(directory / "manifest.json")


class InputIntervals:
    """Six-field full snapshots on the output's actual prompt timeline."""

    def __init__(self, path: Path | None, spec: VideoSpec, *, file=None):
        self.file = file if file is not None else path.open("w")
        self.spec = spec
        self.current = None
        self.start = self.end = self.rows = 0
        self.start_us = self.end_us = 0
        self.last_pts_us = -1

    def append(self, frame: int, inputs: dict, prompt: str, pts_us=None) -> None:
        if frame != self.end:
            raise ValueError("Input intervals must be contiguous")
        pts_us = self.spec.pts(frame) if pts_us is None else pts_us
        if type(pts_us) is not int or pts_us <= self.last_pts_us:
            raise ValueError("Input timestamps must increase")
        self.end_us = pts_us
        # Freeze mutable plugin data and compare object values, not key order.
        snapshot = json.loads(json_line([inputs, prompt]))
        if self.current is not None and self.current != snapshot:
            self._flush(frame)
        if self.current != snapshot:
            self.current, self.start = snapshot, frame
            self.start_us = pts_us
        self.end = frame + 1
        self.last_pts_us = pts_us
        self.end_us = pts_us + self.spec.pts(1)

    def _flush(self, end: int) -> None:
        if self.current is None or end == self.start:
            return
        inputs, prompt = self.current
        self.file.write(
            json_line(
                dict(
                    input=inputs,
                    prompt=prompt,
                    start_frame=self.start,
                    end_frame=end,
                    start_time_us=self.start_us,
                    end_time_us=self.end_us,
                )
            )
        )
        self.file.flush()
        self.rows += 1

    def close(self) -> None:
        if not self.file.closed:
            self._flush(self.end)
            self.file.close()


class H264File:
    """One encoder owner per video; calls are serialized by EncoderQueue."""

    def __init__(self, path: Path, spec: VideoSpec):
        import av

        self.spec = spec
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=spec.fps)
        self.stream.width, self.stream.height = spec.width, spec.height
        self.stream.pix_fmt = "yuv420p"
        from fractions import Fraction

        self.stream.time_base = self.stream.codec_context.time_base = Fraction(
            1, 24_000_000
        )
        self.stream.codec_context.thread_count = 1
        self.stream.options = {"preset": "veryfast", "crf": "18", "bf": "0"}
        self.count = 0

    def append(self, index: int, rgb: bytes, pts_us=None) -> None:
        import av
        import numpy as np

        if index != self.count or len(rgb) != self.spec.frame_bytes:
            raise ValueError("Video frames must be contiguous RGB24")
        pixels = np.frombuffer(rgb, dtype=np.uint8).reshape(
            self.spec.height, self.spec.width, 3
        )
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        from fractions import Fraction

        frame.pts = (
            index * 24_000_000 // self.spec.fps if pts_us is None else pts_us * 24
        )
        frame.time_base = Fraction(1, 24_000_000)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.count += 1

    def close(self) -> None:
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()


class H264MuxFile:
    """Remux the exact playback access units; never open a second encoder."""

    def __init__(self, path, spec, *, streaming=False):
        self.path, self.spec = path, spec
        self.streaming = streaming
        self.container = self.stream = None
        self.time_base = None
        self.count = 0
        self.keyframes = 0
        self.payload_bytes = 0
        self.first_pts_us = None
        self.pending_packet = None
        self.duration_provisional = None
        self.closed = False

    def append(self, index, encoded, pts_us=None):
        import io
        from fractions import Fraction

        import av

        if self.closed or index != self.count or not encoded.get("payload"):
            raise ValueError("Recorded H.264 frames must be contiguous")
        provisional = encoded.get("duration_provisional", False)
        if provisional and pts_us is None:
            raise ValueError("Provisional video duration requires source timestamps")
        if self.duration_provisional is None:
            self.duration_provisional = provisional
        elif provisional != self.duration_provisional:
            raise ValueError("Recorded H.264 duration policy changed")
        if self.container is None:
            if not encoded.get("keyframe"):
                raise ValueError("Recorded H.264 must start with a keyframe")
            with av.open(io.BytesIO(encoded["payload"]), format="h264") as source:
                codec = source.streams.video[0].codec_context
                if (codec.width, codec.height) != (self.spec.width, self.spec.height):
                    raise ValueError("Recorded H.264 changed video dimensions")
                # The movie clock also controls edit-list offsets. Its default
                # millisecond scale rounds a segment's nonzero source start,
                # even when the video track itself uses microseconds.
                options = {"movie_timescale": "1000000"} if pts_us is not None else {}
                if self.streaming:
                    # Delay the initial movie header until the first fragment so
                    # its edit list preserves a nonzero source-clock start.
                    options.update(
                        movflags="delay_moov+default_base_moof+frag_keyframe",
                        use_editlist="1",
                        avoid_negative_ts="disabled",
                    )
                self.container = av.open(
                    self.path if self.streaming else str(self.path),
                    mode="w",
                    **({"format": "mp4"} if self.streaming else {}),
                    options=options,
                )
                self.stream = self.container.add_stream_from_template(
                    source.streams.video[0]
                )
                # VFR input is already integer microseconds. A 24 MHz MP4 clock
                # overflows signed sample deltas after an 89-second source gap.
                # Keep that clock only for the exact rational legacy CFR path.
                self.time_base = Fraction(
                    1, 1_000_000 if pts_us is not None else 24_000_000
                )
                self.stream.time_base = self.time_base
        packet = av.Packet(encoded["payload"])
        packet.stream = self.stream
        scale = self.time_base.denominator
        packet.pts = packet.dts = (
            index * scale // self.spec.fps
            if pts_us is None
            else pts_us * scale // 1_000_000
        )
        packet.duration = (
            scale // self.spec.fps
            if pts_us is None
            else encoded.get("duration_us", self.spec.pts(1)) * scale // 1_000_000
        )
        packet.time_base = self.time_base
        if self.first_pts_us is None:
            self.first_pts_us = int(packet.pts * self.time_base * 1_000_000)
        packet.is_keyframe = encoded["keyframe"]
        if self.pending_packet is not None:
            duration = packet.pts - self.pending_packet.pts
            if duration <= 0:
                raise ValueError("Recorded source timestamps must increase")
            self.pending_packet.duration = duration
            self.container.mux(self.pending_packet)
            self.pending_packet = None
        if provisional:
            # Only the independent archive retains one compressed AU to learn
            # its exact duration. Live playback already received this AU.
            self.pending_packet = packet
        else:
            self.container.mux(packet)
        self.count += 1
        self.keyframes += int(encoded["keyframe"])
        self.payload_bytes += len(encoded["payload"])

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.container:
            if self.pending_packet is not None:
                self.container.mux(self.pending_packet)
                self.pending_packet = None
            self.container.close()


class EncoderQueue:
    def __init__(self, path, spec, trace_id, depth, encoder_factory=H264File):
        self.path, self.spec, self.trace_id = path, spec, trace_id
        self.factory = encoder_factory
        self.queue = asyncio.Queue(maxsize=depth)
        self.count = 0
        self.task = asyncio.create_task(
            self._run(), name="whitebox-encode-" + path.stem
        )

    async def _run(self):
        encoder = await asyncio.to_thread(self.factory, self.path, self.spec)
        try:
            while (item := await self.queue.get()) is not None:
                with stage("encode_" + self.path.stem, self.trace_id):
                    await asyncio.to_thread(encoder.append, *item)
                self.count += 1
        finally:
            await asyncio.to_thread(encoder.close)

    async def put(self, item):
        # Observe encoder failures while waiting on a full queue; never deadlock.
        put = asyncio.create_task(self.queue.put(item))
        try:
            done, _ = await asyncio.wait(
                (put, self.task), return_when=asyncio.FIRST_COMPLETED
            )
            if self.task in done:
                self.task.result()
                if item is None:
                    return
                raise RuntimeError("Encoder stopped before input completed")
            await put
        finally:
            if not put.done():
                put.cancel()
                await asyncio.gather(put, return_exceptions=True)

    async def close(self):
        if not self.task.done():
            await self.put(None)
        await self.task


class ArchiveStore(Protocol):
    async def upload(self, path: Path, key: str) -> None: ...


class S3ArchiveStore:
    """OSS S3-compatible endpoint. Credentials use boto3's credential chain."""

    def __init__(self, bucket: str, endpoint: str, region: str):
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config

        self.bucket = bucket
        self.worker_config = dict(bucket=bucket, endpoint=endpoint, region=region)
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            config=Config(
                s3={"addressing_style": "virtual"},
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
                connect_timeout=3,
                read_timeout=10,
                retries={"mode": "standard", "total_max_attempts": 2},
            ),
        )
        public_endpoint = endpoint.replace("-internal.aliyuncs.com", ".aliyuncs.com")
        self.playback_client = (
            self.client
            if public_endpoint == endpoint
            else boto3.client(
                "s3",
                endpoint_url=public_endpoint,
                region_name=region,
                config=Config(
                    s3={"addressing_style": "virtual"}, signature_version="s3v4"
                ),
            )
        )
        # Concurrency is owned by publish_archive, not multiplied by a second
        # SDK thread pool for every object. This matches main_ali OSS settings.
        self.transfer = TransferConfig(use_threads=False)

    def playback_url(self, key: str) -> str:
        return self.playback_client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=900
        )

    async def upload(self, path: Path, key: str) -> None:
        await asyncio.to_thread(
            self.client.upload_file,
            str(path),
            self.bucket,
            key,
            Config=self.transfer,
            ExtraArgs={
                "ContentType": (
                    "video/mp4"
                    if path.suffix == ".mp4"
                    else (
                        "application/json"
                        if path.suffix == ".json"
                        else "application/octet-stream"
                    )
                )
            },
        )


async def publish_archive(
    directory: Path,
    store: ArchiveStore,
    *,
    concurrency=3,
    retries=3,
    trace_snapshot=None,
):
    """Idempotent retry entry point; immutable recording prefixes are never shared.

    Only the final manifest is published after all listed objects succeeded. The
    spool survives failures and process restarts; run the publisher again to retry.
    """
    manifest = json.loads((directory / "manifest.json").read_text())
    if concurrency < 1 or retries < 1 or not manifest.get("objects"):
        raise ValueError("A finalized archive and positive upload limits are required")
    if manifest.get("recording_status") not in {"complete", "partial"}:
        raise ValueError("The recording is still being finalized")

    def validate_objects():
        for relative, expected in manifest["objects"].items():
            path = directory / relative
            if (
                Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or path.resolve().parent != directory.resolve()
                and directory.resolve() not in path.resolve().parents
                or path.is_symlink()
                or relative in {"manifest.json", "manifest.tmp"}
            ):
                raise ValueError("Invalid archive object path")
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            if (
                path.stat().st_size != expected["bytes"]
                or digest.hexdigest() != expected["sha256"]
            ):
                raise ValueError("Archive object changed after finalization")

    await asyncio.to_thread(validate_objects)
    gate = asyncio.Semaphore(concurrency)
    statistics = dict(
        object_attempts=0,
        object_retries=0,
        objects_uploaded=0,
        concurrency=concurrency,
        attempts_per_object=retries,
    )
    manifest["upload_statistics"] = statistics

    async def upload(relative: str):
        async with gate:
            for attempt in range(retries):
                if relative != "manifest.json":
                    statistics["object_attempts"] += 1
                    statistics["object_retries"] += int(attempt > 0)
                try:
                    with stage("oss_upload", manifest["trace_id"], object=relative):
                        await store.upload(
                            directory / relative, manifest["object_prefix"] + relative
                        )
                    if relative != "manifest.json":
                        statistics["objects_uploaded"] += 1
                    return
                except Exception:
                    if attempt + 1 == retries:
                        raise
                    await asyncio.sleep(min(2**attempt, 10))

    try:
        results = await asyncio.gather(
            *(upload(path) for path in manifest["objects"]), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        if trace_snapshot:
            manifest["trace"] = trace_snapshot()
        manifest["upload_status"] = "complete"
        await asyncio.to_thread(write_manifest, directory, manifest)
        await upload("manifest.json")
    except Exception:
        manifest["upload_status"] = "failed"
        await asyncio.to_thread(write_manifest, directory, manifest)
        raise


def reference_descriptors(references):
    return [
        dict(
            id=ref["id"],
            asset_id="sha256:" + (digest := hashlib.sha256(ref["payload"]).hexdigest()),
            sha256=digest,
            path=f"references/{digest}.{ref['format']}",
        )
        for ref in references
    ]


class Recording:
    def __init__(
        self,
        *,
        directory: Path,
        spec: VideoSpec,
        world_id: str,
        trace_id: str,
        inference: dict,
        ui: dict,
        references: list,
        object_prefix: str,
        queue_depth=12,
        encoder_factory=H264File,
        synchronous=False,
        reuse_playback=False,
        source_only=False,
        model_spec: VideoSpec | None = None,
    ):
        self.directory, self.spec, self.trace_id = directory, spec, trace_id
        self.model_spec = model_spec or spec
        self.trace_snapshot = None
        directory.mkdir(parents=True, exist_ok=False)
        self.inputs = InputIntervals(directory / "inputs.jsonl", spec)
        self.states = (directory / "states.jsonl").open("w")
        self.report_frames = (directory / "report-frames.jsonl").open("w")
        self.output_contexts = (directory / "output-contexts.jsonl").open("w")
        self.references = []
        for index, reference in enumerate(references):
            path = f"references/{index:02d}.{reference['format']}"
            target = directory / path
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(reference["payload"])
            self.references.append(
                dict(
                    id=reference["id"],
                    asset_id="sha256:"
                    + hashlib.sha256(reference["payload"]).hexdigest(),
                    path=path,
                    sha256=hashlib.sha256(reference["payload"]).hexdigest(),
                )
            )
        self.synchronous, self.reuse_playback = synchronous, reuse_playback
        self.source_only = source_only
        self.browser_ui = ui.get("mode") == "browser"
        self.encoders = {}
        for name in (
            ("whitebox",)
            if source_only
            else (
                ("whitebox", "model")
                if self.browser_ui
                else ("whitebox", "model", "ui_rendered")
            )
        ):
            factory = (
                H264MuxFile
                if source_only
                or self.browser_ui
                or (name == "ui_rendered" and reuse_playback)
                else encoder_factory
            )
            path = directory / (name + ".mp4")
            encoder_spec = spec if name == "whitebox" else self.model_spec
            self.encoders[name] = (
                factory(path, encoder_spec)
                if synchronous
                else EncoderQueue(path, encoder_spec, trace_id, queue_depth, factory)
            )
        self.manifest = dict(
            schema_version=1,
            recording_id=directory.name,
            world_id=world_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            trace_id=trace_id,
            object_prefix=object_prefix.rstrip("/") + "/",
            recording_status="finalizing",
            upload_status="pending",
            dataset_status="candidate",
            inference=inference,
            ui=ui,
            quality={"unknown_input_frames": 0},
            timeline=dict(
                fps={"num": spec.fps, "den": 1},
                mode="source_capture_vfr" if self.browser_ui else "frame_clock",
                model_clock_fps=spec.fps,
                pts_unit="us",
                first_frame=0,
                interval="[start,end)",
                alignment="unverified",
            ),
            inputs={"path": "inputs.jsonl", "format": "effective_intervals_v1"},
            states={"path": "states.jsonl", "format": "opaque_frame_envelope_v1"},
            report_frames={
                "path": "report-frames.jsonl",
                "format": "report_frame_index_v1",
            },
            output_contexts={
                "path": "output-contexts.jsonl",
                "format": "model_output_context_v1",
            },
        )
        if self.browser_ui:
            self.manifest["client_ui"] = {
                "owner": "browser",
                "receipt_path": "ui-recording.json",
                "status": "awaiting_upload",
            }
        if source_only:
            self.manifest["recording_kind"] = "source"
        write_manifest(directory, self.manifest)

    async def video(self, name: str, index: int, rgb: bytes, pts_us=None):
        with stage("recording_queue", self.trace_id, video=name, frame=index):
            if self.synchronous:
                self.encoders[name].append(index, rgb, pts_us)
            else:
                await self.encoders[name].put(
                    (index, rgb) if pts_us is None else (index, rgb, pts_us)
                )

    async def encoded_video(self, index, encoded, *, name="ui_rendered", pts_us=None):
        await self.video(name, index, encoded, pts_us)

    async def configure(self):
        await asyncio.to_thread(write_manifest, self.directory, self.manifest)

    def store_references(self, references):
        descriptors = reference_descriptors(references)
        for reference, descriptor in zip(references, descriptors):
            target = self.directory / descriptor["path"]
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                target.write_bytes(reference["payload"])
        return descriptors

    async def add_references(self, references, *, wait=False):
        pending = asyncio.create_task(
            asyncio.to_thread(self.store_references, references)
        )
        try:
            return await asyncio.shield(pending)
        finally:
            await pending

    def source_state(self, message):
        # Opaque CLI deltas must survive video frame dropping. This stream is
        # ordered by capture identity, independently of the paired video archive.
        with (self.directory / "source-states.jsonl").open("a") as target:
            target.write(json_line(message))

    async def write_source_state(self, message):
        task = asyncio.create_task(asyncio.to_thread(self.source_state, message))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    def aligned_metadata(self, frame: EngineFrame, prompt: str, *, archive_pts_us=None):
        inputs = dict(frame.effective_input or {})
        if frame.effective_input is None:
            self.manifest["quality"]["unknown_input_frames"] += 1
            inputs["observed"] = frame.observed_input
        inputs["references"] = (
            self.references if frame.references is None else frame.references
        )
        if frame.rendering is not None:
            inputs["rendering"] = frame.rendering
        if self.manifest.get("multiplayer"):
            inputs["identity"] = {
                key: self.manifest["multiplayer"][key]
                for key in ("game_id", "player_id", "stream_id", "user_id")
            }
        self.inputs.append(frame.index, inputs, prompt, capture_pts(frame))
        self.states.write(
            json_line(
                dict(
                    frame=frame.index,
                    pts_us=capture_pts(frame),
                    model_pts_us=frame.pts_us,
                    presentation_pts_us=presentation_pts(frame),
                    **(
                        {"archive_pts_us": archive_pts_us}
                        if archive_pts_us is not None
                        else {}
                    ),
                    states=frame.states,
                    **(
                        {"rendering": frame.rendering}
                        if frame.rendering is not None
                        else {}
                    ),
                    **({"source": frame.provenance} if frame.provenance else {}),
                )
            )
        )
        self.states.flush()
        self.report_frames.write(
            json_line(
                dict(
                    schema_version=1,
                    frame=frame.index,
                    source_frame_index=(frame.provenance or {}).get(
                        "source_frame_index"
                    ),
                    archive_pts_us=archive_pts_us or {},
                )
            )
        )
        self.report_frames.flush()
        if archive_pts_us is not None and "model" in archive_pts_us:
            references = [
                {key: ref[key] for key in ("id", "sha256") if key in ref}
                for ref in inputs["references"]
            ]
            context = dict(
                prompt=dict(
                    text=prompt, sha256=hashlib.sha256(prompt.encode()).hexdigest()
                ),
                references=references,
            )
            if type((frame.rendering or {}).get("prompt_version")) is int:
                context["prompt"]["version"] = frame.rendering["prompt_version"]
            # This is called after model_output_condition/event_id selected the
            # actual output prompt, never when a request was merely accepted.
            if context != getattr(self, "_last_output_context", None):
                self.output_contexts.write(
                    json_line(
                        dict(
                            schema_version=1,
                            frame=frame.index,
                            source_frame_index=(frame.provenance or {}).get(
                                "source_frame_index"
                            ),
                            archive_pts_us=archive_pts_us,
                            **context,
                        )
                    )
                )
                self.output_contexts.flush()
                self._last_output_context = context
            self._output_context_frames = getattr(self, "_output_context_frames", 0) + 1

    async def write_metadata(self, frame: EngineFrame, prompt: str):
        # A cancelled coroutine must not leave a thread writing into files that
        # the archive finalizer is about to close and hash.
        pending = asyncio.create_task(
            asyncio.to_thread(self.aligned_metadata, frame, prompt)
        )
        try:
            await asyncio.shield(pending)
        finally:
            await pending

    async def finish(self, outcome: str):
        if self.synchronous:
            results = []
            for encoder in self.encoders.values():
                try:
                    encoder.close()
                    results.append(None)
                except Exception as exc:
                    results.append(exc)
        else:
            results = await asyncio.gather(
                *(encoder.close() for encoder in self.encoders.values()),
                return_exceptions=True,
            )
        self.inputs.close()
        self.states.close()
        self.report_frames.close()
        self.output_contexts.close()
        self.manifest["output_contexts"]["frames"] = getattr(
            self, "_output_context_frames", 0
        )
        counts = {name: encoder.count for name, encoder in self.encoders.items()}
        primary = "whitebox" if self.source_only else "model"
        complete = (
            outcome == "completed"
            and len(set(counts.values())) == 1
            and counts[primary] > 0
            and counts[primary] == self.inputs.end
            and not any(isinstance(result, BaseException) for result in results)
        )
        self.manifest.update(
            execution_outcome=outcome,
            finished_at=datetime.now(timezone.utc).isoformat(),
            recording_status="complete" if complete else "partial",
        )
        # Every encoder was given the same authoritative frame-derived PTS. Tests
        # independently decode the MP4s and verify these rational timestamps.
        self.manifest["timeline"].update(
            alignment=(
                ("source_timeline" if self.source_only else "one_to_one")
                if complete
                else "partial"
            ),
            end_frame=min(counts.values()),
            end_time_us=self.inputs.end_us,
        )
        self.manifest["inputs"].update(rows=self.inputs.rows, end_frame=self.inputs.end)
        self.manifest["videos"] = {
            name: dict(
                path=name + ".mp4",
                frames=count,
                width=(self.spec if name == "whitebox" else self.model_spec).width,
                height=(self.spec if name == "whitebox" else self.model_spec).height,
                codec="h264",
                container="mp4",
                pixel_format="yuv420p",
                **(
                    {
                        "first_pts_us": self.encoders[name].first_pts_us,
                        "playback_timeline": "mp4_preserved_pts_v1",
                        "playback_time_origin_us": 0,
                    }
                    if isinstance(self.encoders[name], H264MuxFile)
                    else {}
                ),
                encoder={
                    "name": "libx264",
                    "preset": "veryfast",
                    "crf": 18,
                    "b_frames": 0,
                    "reused_playback_bitstream": self.browser_ui
                    or (name == "ui_rendered" and self.reuse_playback),
                    **(
                        {
                            "profile": "baseline",
                            "tune": "zerolatency",
                            **{
                                key: self.manifest["playback"][key]
                                for key in ("max_bitrate", "vbv_buffer_ms")
                                if key in self.manifest.get("playback", {})
                            },
                        }
                        if name == "ui_rendered" and self.reuse_playback
                        else {}
                    ),
                },
            )
            for name, count in counts.items()
        }

        for name, settings in self.manifest.get("video_streams", {}).items():
            if name in self.manifest["videos"]:
                self.manifest["videos"][name]["encoder"].update(settings)
                self.manifest["videos"][name]["encoder"].update(
                    reused_playback_bitstream=True,
                    keyframes=getattr(self.encoders[name], "keyframes", None),
                    payload_bytes=getattr(self.encoders[name], "payload_bytes", None),
                )

        def summarize():
            objects = {}
            for path in self.directory.rglob("*"):
                if not path.is_file() or path.name in {"manifest.json", "manifest.tmp"}:
                    continue
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                objects[str(path.relative_to(self.directory))] = dict(
                    bytes=path.stat().st_size, sha256=digest.hexdigest()
                )
            if self.trace_snapshot:
                self.manifest["trace"] = self.trace_snapshot()
            self.manifest["objects"] = objects
            write_manifest(self.directory, self.manifest)

        await asyncio.to_thread(summarize)


class ArchiveManager:
    def __init__(
        self,
        store: ArchiveStore | None = None,
        max_pending=16,
        *,
        processes=False,
        queue_depth=24,
        queue_bytes=64 * 1024 * 1024,
        recorder_slots=2,
    ):
        self.store, self.max_pending = store, max_pending
        self.tasks: set[asyncio.Task] = set()
        self.active = set()
        self.recovery = None
        self.processes = processes
        self.queue_depth, self.queue_bytes = queue_depth, queue_bytes
        if type(recorder_slots) is not int or not 1 <= recorder_slots <= 9:
            raise ValueError("Invalid recorder process budget")
        # v2 reserves one persistent source recorder plus a current segment and
        # one finishing segment per player (up to three players). Uploads keep
        # their separate single process.
        self.recorder_slots = recorder_slots
        self.recorders = asyncio.Semaphore(recorder_slots)
        self.upload_gate = asyncio.Semaphore(1)
        self.upload_pool = None
        self.upload_pending = 0
        cpus = (
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        )
        self.recorder_cpu = cpus[-1] if len(cpus) >= 3 else None
        self.upload_cpu = cpus[-2] if len(cpus) >= 3 else None

    async def create_recording(self, **kwargs):
        if not self.processes:
            return Recording(**kwargs, reuse_playback=True)
        from .recording_process import ProcessRecording

        await self.recorders.acquire()
        try:
            return await ProcessRecording.create(
                **kwargs,
                queue_depth=self.queue_depth,
                queue_bytes=self.queue_bytes,
                cpu=self.recorder_cpu,
                release=self.recorders.release,
            )
        except BaseException:
            self.recorders.release()
            raise

    async def _publish(self, directory, *, report=None, trace_snapshot=None):
        if not self.processes or not isinstance(self.store, S3ArchiveStore):
            return await publish_archive(
                directory, self.store, trace_snapshot=trace_snapshot
            )
        from concurrent.futures.process import BrokenProcessPool

        from .metrics import observe_duration
        from .recording_process import _publish, process_pool

        started = time.monotonic()
        self.upload_pending += 1
        try:
            async with self.upload_gate:
                if self.upload_pool is None:
                    self.upload_pool = process_pool(self.upload_cpu)
                if report:
                    observe_duration(
                        "upload_worker_queue",
                        (time.monotonic() - started) * 1000,
                        report.trace_id,
                    )
                try:
                    spans, failed = await asyncio.get_running_loop().run_in_executor(
                        self.upload_pool,
                        _publish,
                        directory,
                        self.store.worker_config,
                        trace_snapshot() if trace_snapshot else None,
                    )
                except BrokenProcessPool:
                    # A killed uploader must not poison every later retry.
                    await asyncio.to_thread(self.upload_pool.shutdown, wait=True)
                    self.upload_pool = None
                    raise
                if report:
                    for span in spans:
                        observe_duration(
                            span["stage"],
                            span["duration_ms"],
                            report.trace_id,
                            result=span["result"],
                            service="uploader",
                        )
                if failed:
                    raise RuntimeError("Archive upload failed; retained for retry")
        finally:
            self.upload_pending -= 1

    def start(self, spool):
        if self.store is None or self.recovery is not None:
            return

        async def recover():
            while True:
                for path in sorted(Path(spool).glob("*/manifest.json")):
                    if path.parent in self.active:
                        continue
                    try:
                        manifest = json.loads(path.read_text())
                        if (
                            manifest.get("recording_status") in {"complete", "partial"}
                            and manifest.get("upload_status") != "complete"
                        ):
                            self.active.add(path.parent)
                            try:
                                await self._publish(path.parent)
                            finally:
                                self.active.discard(path.parent)
                    except Exception:
                        logger.exception("Archive retry failed: %s", path.parent.name)
                await asyncio.sleep(30)

        self.recovery = asyncio.create_task(recover(), name="whitebox-archive-recovery")

    def finalize(self, recording: Recording, outcome: str, *, report=None):
        # Encoders must always be closed, even if upload capacity is exhausted.
        upload = len(self.tasks) < self.max_pending
        self.active.add(recording.directory)

        async def finish():
            try:
                with stage("archive_finalize", recording.trace_id):
                    await recording.finish(outcome)
                if report:
                    try:
                        await asyncio.wait_for(report.gateway_done.wait(), 10)
                    except TimeoutError:
                        pass
                    recording.manifest["trace"] = report.snapshot()
                    # Process recorders persist their snapshot in the worker;
                    # the publisher adds the final gateway snapshot before upload.
                    if not self.processes:
                        await asyncio.to_thread(
                            write_manifest, recording.directory, recording.manifest
                        )
                if self.store is not None and upload:
                    await self._publish(
                        recording.directory,
                        report=report,
                        trace_snapshot=recording.trace_snapshot,
                    )
            finally:
                self.active.discard(recording.directory)
                if report:
                    report.close()

        task = asyncio.create_task(
            finish(), name="whitebox-archive-" + recording.directory.name
        )
        self.tasks.add(task)

        def done(task):
            self.tasks.discard(task)
            if not task.cancelled() and task.exception():
                logger.error(
                    "Whitebox archive retained for retry: %s",
                    recording.directory,
                    exc_info=task.exception(),
                )

        task.add_done_callback(done)

    async def close(self):
        if self.recovery:
            self.recovery.cancel()
            await asyncio.gather(self.recovery, return_exceptions=True)
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.upload_pool:
            await asyncio.to_thread(
                self.upload_pool.shutdown, wait=True, cancel_futures=True
            )
