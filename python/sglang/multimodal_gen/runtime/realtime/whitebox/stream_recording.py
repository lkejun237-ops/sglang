# SPDX-License-Identifier: Apache-2.0
"""Diskless archive child: remux encoded video and upload bounded byte streams."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import PurePosixPath

from .contracts import presentation_pts
from .recording import (
    H264MuxFile,
    InputIntervals,
    Recording,
    json_line,
    reference_descriptors,
)
from .stream_upload import (
    MultipartObject,
    TextObject,
    UploadBudgetExceeded,
    UploadPool,
    retry_same_bytes,
    s3_client,
)

STORAGE = "oss_streaming_v1"


def relative_name(name):
    path = PurePosixPath(name)
    if (
        not isinstance(name, str)
        or not name
        or name.startswith("/")
        or "\\" in name
        or ".." in path.parts
        or str(path) != name
    ):
        raise ValueError("Invalid archive object name")
    return name


def put_json(client, bucket, key, value):
    data = json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()
    if len(data) > 2 << 20:
        raise ValueError("Archive manifest exceeds its byte budget")
    put_bytes(client, bucket, key, data, "application/json")


def put_bytes(client, bucket, key, data, content_type):
    sha = hashlib.sha256(data).hexdigest()
    retry_same_bytes(
        lambda: client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            ContentMD5=base64.b64encode(hashlib.md5(data).digest()).decode(),
            Metadata={"sha256": sha},
        )
    )
    head = client.head_object(Bucket=bucket, Key=key)
    if (
        head.get("ContentLength") != len(data)
        or head.get("Metadata", {}).get("sha256") != sha
    ):
        raise RuntimeError("Archive manifest confirmation failed")


class StreamingData:
    """Worker-only state. Its directory is a logical recording ID, never a file."""

    def __init__(self, client, config, kwargs, manifest, settings):
        self.client, self.config, self.settings = client, config, settings
        self.spec = kwargs["spec"]
        self.model_spec = kwargs.get("model_spec", self.spec)
        self.manifest = copy.deepcopy(manifest)
        self.pool = UploadPool(
            max_bytes=settings["upload_bytes"],
            max_jobs=16,
            workers=2,
            max_archive_bytes=settings["archive_bytes"],
        )
        self.objects = {}
        self.token_hash = None
        self.references = []
        self.last_frame = -1
        self.inputs = InputIntervals(
            None, self.spec, file=TextObject(self.object("inputs.jsonl"))
        )
        self.states = TextObject(self.object("states.jsonl"))
        self.source_states = TextObject(self.object("source-states.jsonl"))
        names = ("whitebox",) if kwargs.get("source_only") else ("whitebox", "model")
        self.encoders = {
            name: H264MuxFile(
                self.object(name + ".mp4"),
                self.model_spec if name == "model" else self.spec,
                streaming=True,
            )
            for name in names
        }
        self.independent = None
        self.reference_executor = None
        self.reference_pending = {}
        self.reference_ready = []
        self.confirmed_references = {}
        if kwargs.get("archive_encoding"):
            from .archive_video import IndependentArchiveVideos

            self.independent = IndependentArchiveVideos(
                self, kwargs["archive_encoding"], kwargs.get("source_only", False)
            )
            self.reference_executor = ThreadPoolExecutor(max_workers=1)
        for i, ref in enumerate(kwargs["references"]):
            descriptor = reference_descriptors([ref])[0]
            descriptor["path"] = f"references/{i:02d}.{ref['format']}"
            self.blob(descriptor["path"], ref["payload"])
            self.references.append(descriptor)

    def object(self, name):
        relative_name(name)
        if name not in self.objects:
            if len(self.objects) >= 192:
                raise ValueError("Too many archive objects")
            self.objects[name] = MultipartObject(
                self.client,
                self.config["bucket"],
                self.manifest["object_prefix"] + name,
                self.pool,
                part_bytes=self.settings["part_bytes"],
                content_type=(
                    "video/mp4" if name.endswith(".mp4") else "application/octet-stream"
                ),
            )
        return self.objects[name]

    def blob(self, name, data):
        relative_name(name)
        old = self.objects.get(name)
        if old is not None:
            if (
                old.length != len(data)
                or old.digest.hexdigest() != hashlib.sha256(data).hexdigest()
            ):
                raise ValueError("Immutable archive asset changed")
            return
        target = self.object(name)
        target.write(data)
        # Static assets stay bounded by the source's UI/reference budgets.
        # Large video and JSONL objects start transferring before finalization.

    def operation(self, kind, value):
        self.pool.check()
        self.last_encoding_ms = {}
        if kind in ("source_rgb", "selected_source_rgb", "model_rgb"):
            if self.independent is None:
                raise ValueError("Independent archive encoding is not configured")
            self.independent.operation(kind, value)
            name = "model" if kind == "model_rgb" else "whitebox"
            self.last_encoding_ms[kind] = self.independent.codecs[name].last_encode_ms
        elif kind == "reference_prepare":
            if self.reference_executor is None or len(self.reference_pending) >= 2:
                raise ValueError("Reference confirmation capacity exceeded")
            request_id, reference = value
            descriptor = reference_descriptors([reference])[0]
            path = relative_name(descriptor["path"])
            payload = reference["payload"]
            digest = hashlib.sha256(payload).hexdigest()
            if request_id in self.reference_pending or len(payload) > 8 << 20:
                raise ValueError("Invalid reference confirmation request")
            known = (
                set(self.objects)
                | set(self.confirmed_references)
                | {item[1] for item in self.reference_pending.values()}
            )
            if path not in known and len(known) >= 192:
                raise ValueError("Too many archive objects")
            existing = self.confirmed_references.get(path)
            obj = self.objects.get(path)
            if existing or (obj is not None and obj.committed):
                existing = existing or {
                    "bytes": obj.length,
                    "sha256": obj.digest.hexdigest(),
                }
                if existing != {"bytes": len(payload), "sha256": digest}:
                    raise ValueError("Immutable archive reference changed")
                self.reference_ready.append(request_id)
                return
            # All accounting remains on this worker's command thread. The
            # auxiliary thread only confirms an immutable object, never mutates
            # the muxers/UploadPool and never blocks RGB command processing.
            self.pool.account(len(payload))
            future = self.reference_executor.submit(
                put_bytes,
                self.client,
                self.config["bucket"],
                self.manifest["object_prefix"] + path,
                payload,
                "application/octet-stream",
            )
            self.reference_pending[request_id] = future, path, digest, len(payload)
        elif kind == "archive_status":
            pass
        elif kind == "configure":
            quality = dict(self.manifest["quality"])
            self.manifest.update(copy.deepcopy(value))
            self.manifest["quality"].update(quality)
        elif kind == "report":
            self.token_hash, _ = value
        elif kind == "asset":
            self.blob("ui_bundle/" + relative_name(value[0]), value[1])
        elif kind == "references":
            for ref, descriptor in zip(value, reference_descriptors(value)):
                self.blob(descriptor["path"], ref["payload"])
                # A condition switch may use this image after the worker ACK.
                # Confirm its immutable object before returning that receipt.
                self.objects[descriptor["path"]].finish()
        elif kind == "state":
            self.source_states.write(json_line(value))
        elif kind == "pair":
            frame, source, encoded, prompt = value
            pts = presentation_pts(frame)
            self.encoders["whitebox"].append(frame.index, source, pts)
            self.encoders["model"].append(frame.index, encoded, pts)
            Recording.aligned_metadata(self, frame, prompt)
            self.last_frame = frame.index
        elif kind == "source_frame":
            frame, encoded = value
            self.encoders["whitebox"].append(
                frame.index, encoded, frame.provenance["capture_pts_us"]
            )
            Recording.aligned_metadata(self, frame, "")
            self.last_frame = frame.index
        elif kind == "video":
            name, index, encoded, pts = value
            self.encoders[name].append(index, encoded, pts)
        elif kind == "metadata":
            frame, prompt = value
            Recording.aligned_metadata(self, frame, prompt)
            self.last_frame = frame.index
        else:
            raise ValueError("Unsupported streaming archive operation")

    def record_metadata(self, frame, prompt):
        Recording.aligned_metadata(self, frame, prompt)

    def reference_receipts(self):
        complete, self.reference_ready = self.reference_ready, []
        for request_id, (future, path, digest, size) in list(
            self.reference_pending.items()
        ):
            if future.done():
                future.result()
                self.confirmed_references[path] = {"bytes": size, "sha256": digest}
                self.pool.confirmed_bytes += size
                del self.reference_pending[request_id]
                complete.append(request_id)
        return complete

    def index(self):
        if self.token_hash is None:
            return
        put_json(
            self.client,
            self.config["bucket"],
            self.settings["index_key"],
            {
                "schema_version": 1,
                "recording_id": self.manifest["recording_id"],
                "token_hash": self.token_hash,
                "object_prefix": self.manifest["object_prefix"],
                "archive_storage": STORAGE,
            },
        )

    def finish(self, outcome, manifest, rgb_callback=None):
        quality = dict(self.manifest["quality"])
        self.manifest.update(manifest)
        self.manifest["quality"].update(quality)
        if self.independent:
            # Every RGB command has been ACKed. The codec's retained frame owns
            # its pixels; release shared mappings before network finalization.
            self.independent.close_readers()
            if rgb_callback:
                rgb_callback()
        self.pool.begin_finish(self.settings["finish_seconds"])
        for future, *_ in list(self.reference_pending.values()):
            future.result(timeout=max(0, self.pool.deadline - time.monotonic()))
        self.reference_receipts()
        if self.independent:
            self.independent.finish(outcome)
        for mux in self.encoders.values():
            mux.close()
        self.inputs.close()
        self.states.close()
        self.source_states.close()
        counts = {name: mux.count for name, mux in self.encoders.items()}
        complete = (
            outcome == "completed"
            and len(set(counts.values())) == 1
            and min(counts.values()) > 0
            and min(counts.values()) == self.inputs.end
        )
        if self.independent:
            rendering = self.manifest.get("rendering", {})
            expected = rendering.get(
                (
                    "source_frames_captured"
                    if self.independent.source_only
                    else "model_frames"
                ),
                self.independent.next_index,
            )
            complete = complete and min(counts.values()) == expected
        if self.manifest.get("recording_kind") == "source":
            files = self.manifest.get("ui", {}).get("files", {})
            assets_complete = all(
                "ui_bundle/" + name in self.objects
                and self.objects["ui_bundle/" + name].digest.hexdigest() == digest
                for name, digest in files.items()
            )
            self.manifest["quality"]["ui_assets_complete"] = assets_complete
            complete = complete and assets_complete
        self.manifest.update(
            finished_at=datetime.now(timezone.utc).isoformat(),
            execution_outcome=outcome,
            recording_status="complete" if complete else "partial",
            archive_storage=STORAGE,
        )
        self.manifest["timeline"].update(
            alignment=(
                ("source_timeline" if len(counts) == 1 else "one_to_one")
                if complete
                else "partial"
            ),
            end_frame=min(counts.values()),
            end_time_us=self.inputs.end_us,
        )
        self.manifest["inputs"].update(rows=self.inputs.rows, end_frame=self.inputs.end)
        self.manifest["videos"] = {
            name: {
                "path": name + ".mp4",
                "frames": mux.count,
                "width": mux.spec.width,
                "height": mux.spec.height,
                "codec": "h264",
                "container": "mp4",
                "fragmented": True,
                "pixel_format": "yuv420p",
                "encoder": {
                    "name": "libx264",
                    "preset": "veryfast",
                    "crf": 18,
                    "b_frames": 0,
                    **self.manifest.get("video_streams", {}).get(name, {}),
                    "reused_playback_bitstream": self.independent is None,
                    "keyframes": mux.keyframes,
                    "payload_bytes": mux.payload_bytes,
                },
            }
            for name, mux in self.encoders.items()
            if mux.count
        }
        if self.independent:
            for video in self.manifest["videos"].values():
                video["encoder"].pop("crf", None)
                video["first_pts_us"] = self.independent.first_pts_us
                video["duration_us"] = (
                    self.independent.end_time_us - self.independent.first_pts_us
                )
                video["encoder"]["measured_payload_bitrate"] = round(
                    video["encoder"]["payload_bytes"] * 8_000_000 / video["duration_us"]
                )
        for name, mux in self.encoders.items():
            if not mux.count:
                self.objects.pop(name + ".mp4")
        self.manifest["objects"] = {
            name: obj.finish() for name, obj in self.objects.items()
        }
        self.manifest["objects"].update(self.confirmed_references)
        self.manifest["upload_statistics"] = {
            "objects_uploaded": len(self.objects),
            "produced_bytes": self.pool.total_bytes,
            "confirmed_upload_bytes": self.pool.confirmed_bytes,
            "upload_high_water_bytes": self.pool.high_water_bytes,
            "upload_workers": 2,
            "reference_upload_workers": int(self.reference_executor is not None),
        }
        # The capability index contains only a hash and lives outside the data
        # prefix. Never mark the archive complete before both are confirmed.
        self.index()
        self.manifest["upload_status"] = "complete"
        put_json(
            self.client,
            self.config["bucket"],
            self.manifest["object_prefix"] + "manifest.json",
            self.manifest,
        )
        return self.manifest

    def abort(self):
        for obj in self.objects.values():
            obj.abort()

    def close(self):
        if self.reference_executor:
            self.reference_executor.shutdown(wait=False, cancel_futures=True)
        if self.independent:
            self.independent.close()
        for mux in self.encoders.values():
            try:
                mux.close()
            except Exception:
                pass
        self.pool.close()
        self.abort()

    def unfinished_uploads(self):
        rows = []
        for obj in self.objects.values():
            if not obj.committed and obj.initiated is not None and obj.initiated.done():
                try:
                    rows.append({"key": obj.key, "upload_id": obj.initiated.result()})
                except Exception:
                    pass
        return rows


def streaming_worker(
    commands, replies, config, kwargs, manifest, settings, cpu, memory_bytes
):
    from .async_recording import limit_child

    # RLIMIT_FSIZE=0 is a second fence against accidental archive file writes.
    limit_child(cpu, memory_bytes, 0)
    data = None
    try:
        client = s3_client(config)
        data = StreamingData(client, config, kwargs, manifest, settings)
        replies.put(("boot", {}))
        while True:
            kind, value = commands.get()
            started = time.monotonic()
            if kind == "finish":
                result = data.finish(
                    *value,
                    rgb_callback=(
                        (lambda: replies.put(("rgb_released", {})))
                        if data.independent
                        else None
                    ),
                )
                replies.put(("finished", result))
                return
            data.operation(kind, value)
            replies.put(
                (
                    "ack",
                    {
                        "last_frame": data.last_frame,
                        "duration_ms": (time.monotonic() - started) * 1000,
                        "upload_pending_bytes": data.pool.pending_bytes,
                        "object_buffer_bytes": data.pool.buffer_bytes,
                        "produced_bytes": data.pool.total_bytes,
                        "confirmed_upload_bytes": data.pool.confirmed_bytes,
                        "reference_receipts": data.reference_receipts(),
                        "encoding_ms": data.last_encoding_ms,
                    },
                )
            )
    except BaseException as error:
        try:
            reason = (
                "UPLOAD_MEMORY_BUDGET_EXCEEDED"
                if isinstance(error, UploadBudgetExceeded)
                else (
                    "UPLOAD_TIMEOUT"
                    if isinstance(error, TimeoutError)
                    else "STREAM_UPLOAD_FAILED"
                )
            )
            replies.put(
                (
                    "failed",
                    {
                        "last_frame": data.last_frame if data else -1,
                        "reason": reason,
                        # Only the parent server logger consumes this field. It never
                        # enters a manifest or browser response. The child may not write
                        # even to an inherited regular-file stderr under RLIMIT_FSIZE=0.
                        "diagnostic": type(error).__name__ + ": " + str(error)[:2000],
                        "unfinished_uploads": data.unfinished_uploads() if data else [],
                    },
                )
            )
        except Exception:
            pass
    finally:
        if data:
            try:
                data.abort()
            finally:
                data.close()
