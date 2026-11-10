"""Memory-only OSS doubles and real H.264 exercise the new archive boundary."""

import asyncio
import base64
import copy
import hashlib
import io
import json
import threading
import time
from types import SimpleNamespace

import av
import numpy as np
import pytest
from botocore.exceptions import ClientError

from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import (
    EngineFrame,
    VideoSpec,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.playback import H264PlaybackEncoder
from sglang.multimodal_gen.runtime.realtime.whitebox.stream_archive import (
    StreamingArchiveManager,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.stream_recording import (
    STORAGE,
    StreamingData,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.stream_upload import (
    MultipartObject,
    UploadBudgetExceeded,
    UploadPool,
)


class MemoryS3:
    def __init__(self):
        self.objects, self.uploads = {}, {}
        self.lock = threading.RLock()
        self.part_arrived = threading.Event()
        self.release = None
        self.lose_completion_reply = False
        self.part_calls = 0
        self.aborts = 0
        self.next_upload_id = 0
        self.reject_parts = False

    def create_multipart_upload(self, **kw):
        with self.lock:
            self.next_upload_id += 1
            uid = str(self.next_upload_id)
            self.uploads[uid] = {
                "key": (kw["Bucket"], kw["Key"]),
                "metadata": kw.get("Metadata", {}),
                "parts": {},
            }
        return {"UploadId": uid}

    def upload_part(self, **kw):
        if self.release is not None:
            self.release.wait(3)
        data = bytes(kw["Body"])
        assert kw["ContentMD5"] == base64.b64encode(hashlib.md5(data).digest()).decode()
        with self.lock:
            self.uploads[kw["UploadId"]]["parts"][kw["PartNumber"]] = data
            self.part_calls += 1
        self.part_arrived.set()
        return {"ETag": hashlib.md5(data).hexdigest()}

    def complete_multipart_upload(self, **kw):
        with self.lock:
            v = self.uploads[kw["UploadId"]]
            parts = kw["MultipartUpload"]["Parts"]
            assert [p["PartNumber"] for p in parts] == list(range(1, len(parts) + 1))
            for p in parts:
                assert p["ETag"] == hashlib.md5(v["parts"][p["PartNumber"]]).hexdigest()
            self.objects[v["key"]] = (
                b"".join(v["parts"][p["PartNumber"]] for p in parts),
                v["metadata"],
            )
            del self.uploads[kw["UploadId"]]
        if self.lose_completion_reply:
            raise TimeoutError("lost reply after commit")
        return {}

    def put_object(self, **kw):
        data = bytes(kw["Body"])
        assert kw["ContentMD5"] == base64.b64encode(hashlib.md5(data).digest()).decode()
        with self.lock:
            self.objects[(kw["Bucket"], kw["Key"])] = (data, kw.get("Metadata", {}))
        return {}

    def head_object(self, **kw):
        with self.lock:
            try:
                data, meta = self.objects[(kw["Bucket"], kw["Key"])]
            except KeyError:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "HeadObject")
            return {"ContentLength": len(data), "Metadata": meta}

    def get_object(self, **kw):
        with self.lock:
            try:
                data, meta = self.objects[(kw["Bucket"], kw["Key"])]
            except KeyError:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            return {
                "ContentLength": len(data),
                "Metadata": meta,
                "Body": io.BytesIO(data),
            }

    def abort_multipart_upload(self, **kw):
        with self.lock:
            self.uploads.pop(kw["UploadId"], None)
            self.aborts += 1


async def prototype(tmp_path, *, source=False):
    m = StreamingArchiveManager(index_prefix="test")
    spec = VideoSpec(64, 64)
    kwargs = dict(
        directory=tmp_path / ("a" * 32),
        spec=spec,
        world_id="world",
        trace_id="stream-test",
        inference={},
        ui={"mode": "browser", "files": {}},
        references=[],
        object_prefix="test/" + ("a" * 32) + "/",
        source_only=source,
    )
    r = await m.create_recording(**kwargs)
    manifest = copy.deepcopy(r.manifest)
    if source:
        manifest["recording_kind"] = "source"
    await m.close()
    return kwargs, manifest


def test_multipart_starts_before_close_and_reconciles_lost_completion():
    client = MemoryS3()
    pool = UploadPool(max_bytes=1 << 20)
    target = MultipartObject(
        client, "bucket", "video.mp4", pool, content_type="video/mp4", part_bytes=65536
    )
    data = b"abcdefgh" * 20000
    try:
        target.write(data)
        assert client.part_arrived.wait(1)
        assert ("bucket", "video.mp4") not in client.objects
        assert len(target.buffer) < 65536 and pool.pending_bytes <= pool.max_bytes
        client.lose_completion_reply = True
        pool.begin_finish()
        result = target.finish()
        assert result == {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        assert client.objects[("bucket", "video.mp4")][0] == data
        assert not client.uploads
    finally:
        pool.close()


def test_stalled_network_has_a_hard_upload_memory_bound():
    client = MemoryS3()
    client.release = threading.Event()
    pool = UploadPool(max_bytes=131072, max_jobs=16)
    target = MultipartObject(
        client, "b", "video.mp4", pool, content_type="video/mp4", part_bytes=65536
    )
    try:
        target.write(b"x" * 131072)
        with pytest.raises(UploadBudgetExceeded):
            target.write(b"y" * 65536)
        assert pool.pending_bytes <= 131072
        assert pool.buffer_bytes <= pool.max_buffer_bytes
    finally:
        client.release.set()
        pool.close()
        target.abort()
    assert not client.uploads


def test_shared_mux_buffers_are_bounded_even_before_a_full_upload_part():
    pool = UploadPool(max_buffer_bytes=100)
    try:
        a = MultipartObject(
            MemoryS3(), "b", "a", pool, content_type="x", part_bytes=1024
        )
        b = MultipartObject(
            MemoryS3(), "b", "b", pool, content_type="x", part_bytes=1024
        )
        a.write(b"a" * 60)
        with pytest.raises(UploadBudgetExceeded):
            b.write(b"b" * 60)
        assert pool.buffer_bytes == 60
    finally:
        pool.close()


def test_real_streamed_video_preserves_pixels_absolute_pts_and_inputs_without_disk(
    tmp_path, monkeypatch
):
    kwargs, manifest = asyncio.run(prototype(tmp_path))
    client = MemoryS3()
    settings = dict(
        upload_bytes=32 << 20,
        archive_bytes=2 << 30,
        part_bytes=65536,
        finish_seconds=10,
        index_key="test/.archive-index/v1/" + ("a" * 32) + ".json",
    )
    data = StreamingData(client, {"bucket": "b"}, kwargs, manifest, settings)
    spec = kwargs["spec"]
    encoder = H264PlaybackEncoder(
        spec, max_bitrate=4_000_000, gop=6, capture_clock=True
    )
    reference = av.CodecContext.create("h264", "r")
    hashes = []
    expected_pts = []
    rng = np.random.default_rng(22)

    def emit(items):
        for i, encoded in items:
            pts = 2_000_000 + round(i * 1e6 / 30)
            frame = EngineFrame(
                i,
                spec.pts(i),
                b"",
                {"frame": i},
                effective_input={"held_keys": ["W"]},
                provenance={"capture_pts_us": pts},
            )
            data.operation("pair", (frame, encoded, encoded, "test prompt"))
            pixels = reference.decode(av.Packet(encoded["payload"]))[0].to_ndarray(
                format="rgb24"
            )
            hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
            expected_pts.append(pts)

    try:
        # Any accidental archive file/directory write is a test failure.
        from pathlib import Path

        def forbidden(*a, **kw):
            raise AssertionError("Archive attempted a filesystem write")

        monkeypatch.setattr(Path, "mkdir", forbidden)
        monkeypatch.setattr(Path, "write_bytes", forbidden)
        monkeypatch.setattr(Path, "write_text", forbidden)
        for i in range(60):
            emit(
                encoder.push(
                    i,
                    rng.integers(0, 256, (64, 64, 3), dtype=np.uint8).tobytes(),
                    2_000_000 + round(i * 1e6 / 30),
                )
            )
        emit(encoder.drain())
        assert client.part_arrived.wait(1)
        assert not kwargs["directory"].exists()
        data.operation(
            "report", (hashlib.sha256(b"capability").hexdigest(), time.time())
        )
        result = data.finish("completed", manifest)
        assert result["recording_status"] == result["upload_status"] == "complete"
        assert result["archive_storage"] == STORAGE
        report_index = [
            json.loads(line)
            for line in client.objects[
                ("b", manifest["object_prefix"] + "report-frames.jsonl")
            ][0].splitlines()
        ]
        assert [row["archive_pts_us"]["model"] for row in report_index] == expected_pts
        assert [
            row["archive_pts_us"]["whitebox"] for row in report_index
        ] == expected_pts
        for name in ["whitebox", "model"]:
            assert result["videos"][name]["first_pts_us"] == expected_pts[0]
            assert result["videos"][name]["playback_time_origin_us"] == 0
            assert result["videos"][name]["playback_timeline"] == "mp4_preserved_pts_v1"
            raw = client.objects[("b", manifest["object_prefix"] + name + ".mp4")][0]
            with av.open(io.BytesIO(raw)) as movie:
                frames = list(movie.decode(video=0))
            assert [
                hashlib.sha256(f.to_ndarray(format="rgb24").tobytes()).hexdigest()
                for f in frames
            ] == hashes
            assert [
                round(float(f.pts * f.time_base) * 1e6) for f in frames
            ] == expected_pts
        assert result["timeline"]["end_frame"] == 60
        assert (
            result["objects"]["model.mp4"]["sha256"] == hashlib.sha256(raw).hexdigest()
        )
    finally:
        encoder.close()
        data.abort()
        data.close()


def test_missing_store_and_slow_boot_never_wait_in_admission_or_live_offers(tmp_path):
    async def run():
        manager = StreamingArchiveManager(index_prefix="test")
        before = time.monotonic()
        r = await manager.create_recording(
            directory=tmp_path / ("b" * 32),
            spec=VideoSpec(32, 32),
            world_id="w",
            trace_id="missing-store",
            inference={},
            ui={"mode": "browser"},
            references=[],
            object_prefix="test/" + ("b" * 32) + "/",
        )
        assert time.monotonic() - before < 0.1
        assert not r.worker_ready
        for i in range(300):
            r.source_frame(
                EngineFrame(
                    i, i * 41666, b"", {}, provenance={"capture_pts_us": i * 41666}
                ),
                {"payload": b"x", "keyframe": True, "duration_us": 41666},
            )
        assert (
            r.pending_bytes <= manager.queue_bytes
            and len(r.items) <= manager.queue_depth
        )
        await manager.close()
        assert r.failure and not r.directory.exists()

    asyncio.run(run())


def test_ui_assets_remain_available_when_archive_is_disabled(tmp_path):
    async def run():
        blob = b"export const ui = true;"
        sha = hashlib.sha256(blob).hexdigest()
        manager = StreamingArchiveManager(index_prefix="test")
        r = await manager.create_recording(
            directory=tmp_path / ("c" * 32),
            spec=VideoSpec(32, 32),
            world_id="w",
            trace_id="ui-without-archive",
            inference={},
            ui={"mode": "browser", "files": {"main.js": sha}},
            references=[],
            object_prefix="test/" + ("c" * 32) + "/",
        )

        class Engine:
            async def ui_asset(self, name):
                return blob

        r.bind_ui(Engine(), r.manifest["ui"])
        await r.task
        assert r.failure == "ARCHIVE_UNAVAILABLE"
        assert await r.get_ui_asset("main.js") == blob
        assert not r.directory.exists()
        await manager.close()
        assert not r.ui_assets

    asyncio.run(run())


@pytest.fixture
def local_s3(monkeypatch):
    """Exercise boto3 and spawned archive workers against loopback only."""
    import xml.etree.ElementTree as ET
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, unquote, urlsplit

    client = MemoryS3()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def respond(self, status=200, body=b"", headers=None):
            self.send_response(status)
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD" and body:
                self.wfile.write(body)

        def route(self):
            path = urlsplit(self.path)
            bucket, key = unquote(path.path).lstrip("/").split("/", 1)
            return bucket, key, parse_qs(path.query, keep_blank_values=True)

        def do_PUT(self):
            b, k, q = self.route()
            data = self.rfile.read(int(self.headers["Content-Length"]))
            kw = dict(
                Bucket=b, Key=k, Body=data, ContentMD5=self.headers["Content-MD5"]
            )
            if "uploadId" in q:
                if client.reject_parts:
                    self.respond(503, b"<Error><Code>ServiceUnavailable</Code></Error>")
                    return
                kw.update(UploadId=q["uploadId"][0], PartNumber=int(q["partNumber"][0]))
                v = client.upload_part(**kw)
            else:
                kw["Metadata"] = {
                    key.lower()[11:]: value
                    for key, value in self.headers.items()
                    if key.lower().startswith("x-amz-meta-")
                }
                v = client.put_object(**kw)
            self.respond(headers={"ETag": v.get("ETag", "put-etag")})

        def do_POST(self):
            b, k, q = self.route()
            if "uploads" in q:
                metadata = {
                    key.lower()[11:]: value
                    for key, value in self.headers.items()
                    if key.lower().startswith("x-amz-meta-")
                }
                uid = client.create_multipart_upload(
                    Bucket=b, Key=k, Metadata=metadata
                )["UploadId"]
                self.respond(
                    body=f"<InitiateMultipartUploadResult><Bucket>{b}</Bucket><Key>{k}</Key><UploadId>{uid}</UploadId></InitiateMultipartUploadResult>".encode()
                )
                return
            root = ET.fromstring(self.rfile.read(int(self.headers["Content-Length"])))
            parts = []
            for part in root:
                fields = {x.tag.split("}")[-1]: x.text for x in part}
                parts.append(
                    {"PartNumber": int(fields["PartNumber"]), "ETag": fields["ETag"]}
                )
            client.complete_multipart_upload(
                Bucket=b,
                Key=k,
                UploadId=q["uploadId"][0],
                MultipartUpload={"Parts": parts},
            )
            self.respond(
                body=b"<CompleteMultipartUploadResult><ETag>complete-etag</ETag></CompleteMultipartUploadResult>"
            )

        def do_DELETE(self):
            b, k, q = self.route()
            client.abort_multipart_upload(Bucket=b, Key=k, UploadId=q["uploadId"][0])
            self.respond(204)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            b, k, q = self.route()
            try:
                v = client.get_object(Bucket=b, Key=k)
            except ClientError:
                self.respond(404, b"<Error><Code>NoSuchKey</Code></Error>")
                return
            body = v["Body"].read()
            headers = {
                f"x-amz-meta-{key}": value for key, value in v["Metadata"].items()
            }
            self.respond(body=body, headers=headers)

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "local-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "local-test-only")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("SGLANG_LIGHTWEIGHT_RUNTIME", "1")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield client, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_reference_wait_confirms_streamed_object_before_session_finish(
    tmp_path, local_s3
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.recording import S3ArchiveStore

    fake, endpoint = local_s3

    async def run():
        manager = StreamingArchiveManager(
            S3ArchiveStore("bucket", endpoint, "ap-southeast-7"),
            index_prefix="test",
        )
        rid = "e" * 32
        prefix = "test/" + rid + "/"
        record = await manager.create_recording(
            directory=tmp_path / rid,
            spec=VideoSpec(64, 64),
            world_id="w",
            trace_id="reference-confirmation",
            inference={},
            ui={"mode": "browser", "files": {}},
            references=[],
            object_prefix=prefix,
        )
        try:
            await asyncio.wait_for(record.startup_complete.wait(), 10)
            assert record.worker_ready, record.failure
            refs = [{"id": "reference-0", "format": "png", "payload": b"image-bytes"}]
            descriptors = await asyncio.wait_for(
                record.add_references(refs, wait=True), 3
            )
            assert (
                fake.objects[("bucket", prefix + descriptors[0]["path"])][0]
                == b"image-bytes"
            )
            assert not record.closed and not record.failure
            assert ("bucket", prefix + "manifest.json") not in fake.objects
            # Reusing a committed object and clearing references also finish
            # while the source is live, without waiting for archive finalization.
            assert (
                await asyncio.wait_for(record.add_references(refs, wait=True), 3)
                == descriptors
            )
            assert await asyncio.wait_for(record.add_references([], wait=True), 3) == []
            assert not list(tmp_path.rglob("*"))
        finally:
            await manager.close()

    asyncio.run(run())


def test_spawned_worker_real_sdk_reports_replay_and_optional_ui_never_create_archive_files(
    tmp_path, local_s3
):
    import httpx
    from fastapi import FastAPI

    from sglang.multimodal_gen.runtime.realtime.whitebox.recording import S3ArchiveStore
    from sglang.multimodal_gen.runtime.realtime.whitebox.reports import (
        Reports,
        SessionReport,
    )

    fake, endpoint = local_s3

    async def run():
        store = S3ArchiveStore("bucket", endpoint, "ap-southeast-7")
        manager = StreamingArchiveManager(store, index_prefix="test", finish_timeout=30)
        reports = Reports(tmp_path, manager)
        rid = "d" * 32
        prefix = "test/" + rid + "/"
        spec = VideoSpec(64, 64)
        encoder = H264PlaybackEncoder(spec, max_bitrate=4_000_000, gop=6)
        before = time.monotonic()
        r = await manager.create_recording(
            directory=tmp_path / rid,
            spec=spec,
            world_id="w",
            trace_id="diskless-sdk",
            inference={},
            ui={"mode": "browser", "files": {}},
            references=[],
            object_prefix=prefix,
        )
        assert time.monotonic() - before < 0.1
        report = SessionReport("diskless-sdk")
        reports.attach(report, r)
        token = report.token
        try:
            await asyncio.wait_for(r.startup_complete.wait(), 10)
            assert r.worker_ready, r.failure
            # Several large metadata records force real multipart transfer while
            # the session is still open. This is not a live-frame workload.
            for i in range(4):
                await r.write_source_state({"n": i, "state": "x" * (3 << 20)})
                async with asyncio.timeout(5):
                    while r.pending_bytes and not r.failure:
                        await asyncio.sleep(0.01)
                assert not r.failure, r.failure
            assert await asyncio.to_thread(fake.part_arrived.wait, 3)
            assert ("bucket", prefix + "manifest.json") not in fake.objects
            rng = np.random.default_rng(17)
            for i in range(20):
                encoded = encoder.encode(
                    i, rng.integers(0, 256, (64, 64, 3), dtype=np.uint8).tobytes()
                )
                f = EngineFrame(
                    i,
                    spec.pts(i),
                    b"",
                    {"i": i},
                    effective_input={"held_keys": []},
                    provenance={"capture_pts_us": 2_000_000 + spec.pts(i)},
                )
                assert r.pair(f, encoded, encoded, "p")
                await asyncio.sleep(0.005)
            manager.finalize(r, "completed", report=report)
            await asyncio.wait_for(r.task, 35)
            await asyncio.gather(*list(manager.tasks))
            assert r.manifest["upload_status"] == "complete", r.manifest
            assert r.manifest["videos"]["model"]["frames"] == 20
            assert (
                not (tmp_path / rid).exists() and not (tmp_path / ".reports").exists()
            )
            app = FastAPI()
            reports.install(app)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as http:
                headers = {"Authorization": "Bearer " + token}
                response = await http.get(
                    "/v1/whitebox/recordings/" + rid, headers=headers
                )
                assert response.status_code == 200, response.text
                assert len(response.json()["videos"]) == 2
                # Clearing memory simulates a new Room reading only its OSS index.
                reports.live.clear()
                manager.live_recordings.clear()
                response = await http.get(
                    "/v1/whitebox/recordings/" + rid, headers=headers
                )
                assert response.status_code == 200, response.text
                assert response.json()["recording_status"] == "complete"
                response = await http.get(
                    "/v1/whitebox/recordings/" + rid + "/inputs", headers=headers
                )
                assert response.status_code == 200, response.text
                assert response.json()["intervals"][0]["start_time_us"] == 2_000_000
                assert (
                    await http.get(
                        "/v1/whitebox/recordings/" + rid,
                        headers={"Authorization": "Bearer wrong"},
                    )
                ).status_code == 404
                candidate = fake.objects[("bucket", prefix + "model.mp4")][0]
                body = dict(
                    bytes=len(candidate),
                    sha256=hashlib.sha256(candidate).hexdigest(),
                    content_md5=base64.b64encode(
                        hashlib.md5(candidate).digest()
                    ).decode(),
                )
                response = await http.post(
                    "/v1/whitebox/recordings/" + rid + "/ui-recording/intent",
                    json=body,
                    headers=headers,
                )
                assert response.status_code == 200, response.text
                upload_id = response.json()["upload_id"]
                # Only the staged object is client-writable; the verifier uploads
                # the validated in-memory bytes to its own published key.
                store.client.put_object(
                    Bucket="bucket",
                    Key=prefix + f"client-ui/{upload_id}/incoming.mp4",
                    Body=candidate,
                    ContentMD5=body["content_md5"],
                )
                response = await http.post(
                    "/v1/whitebox/recordings/"
                    + rid
                    + "/ui-recording/"
                    + upload_id
                    + "/complete",
                    headers=headers,
                )
                assert response.status_code == 200, response.text
                async with asyncio.timeout(30):
                    while reports.client_recordings.streaming.jobs:
                        await asyncio.sleep(0.05)
                response = await http.get(
                    "/v1/whitebox/recordings/" + rid, headers=headers
                )
                assert response.json()["client_ui"]["status"] == "complete", (
                    response.text
                )
                assert any(
                    x["name"] == "ui_rendered" for x in response.json()["videos"]
                )
            assert not list(tmp_path.iterdir())
        finally:
            encoder.close()
            await manager.close()
            report.close()

    asyncio.run(run())


def test_four_model_segments_and_prompt_updates_keep_playing_without_any_archive_store(
    tmp_path, monkeypatch
):
    import test_whitebox_render_lifecycle as lifecycle

    monkeypatch.setattr(
        lifecycle,
        "AsyncArchiveManager",
        lambda **kw: StreamingArchiveManager(index_prefix="test", **kw),
    )
    lifecycle.test_source_survives_four_segments_and_ordered_prompt_updates(
        tmp_path, "valid-source-frames-v1", True
    )
    assert not list(tmp_path.rglob("*.mp4"))
    assert not list(tmp_path.rglob("*.jsonl"))
    assert not list(tmp_path.rglob("render-assets"))


def test_reference_assets_are_memory_only_and_still_validate_identity(tmp_path):
    from PIL import Image

    from sglang.multimodal_gen.runtime.realtime.whitebox.render_assets import (
        SourceAssets,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.render_contract import (
        RenderError,
    )

    async def run():
        output = io.BytesIO()
        Image.new("RGB", (32, 32), "blue").save(output, format="PNG")
        blob = output.getvalue()
        assets = SourceAssets(tmp_path / "never-created", memory=True)
        descriptor = await assets.upload("png", blob)
        duplicate = await assets.upload("png", blob)
        assert duplicate["duplicate"]
        refs = await assets.resolve([{"id": "ref", "asset_id": descriptor["asset_id"]}])
        assert refs[0]["payload"] == blob
        assert not assets.directory.exists()
        with pytest.raises(RenderError):
            await assets.upload("png", b"invalid")
        assets.release_memory()
        assert not assets.payloads and assets.bytes == 0
        with pytest.raises(RenderError):
            await assets.resolve([{"id": "ref", "asset_id": descriptor["asset_id"]}])

    asyncio.run(run())


def slow_boot_worker(*args):
    time.sleep(2)


def test_slow_child_boot_does_not_block_live_event_loop_and_is_reaped(tmp_path):
    async def run():
        store = SimpleNamespace(worker_config={"unused": True})
        manager = StreamingArchiveManager(
            store, index_prefix="test", boot_timeout=0.1, max_age=0.1
        )
        manager.worker_target = slow_boot_worker
        t = time.monotonic()
        r = await manager.create_recording(
            directory=tmp_path / ("e" * 32),
            spec=VideoSpec(32, 32),
            world_id="w",
            trace_id="slow-boot",
            inference={},
            ui={"mode": "browser"},
            references=[],
            object_prefix="test/" + ("e" * 32) + "/",
        )
        assert time.monotonic() - t < 0.1
        ticks = 0
        async with asyncio.timeout(5):
            while not r.closed:
                ticks += 1
                await asyncio.sleep(0.005)
        assert ticks >= 10
        assert r.failure and r.process and not r.process.is_alive()
        assert not r.directory.exists()
        await manager.close()

    asyncio.run(run())


def test_failure_notice_reconciles_a_committed_manifest_before_overwriting(tmp_path):
    async def run():
        fake = MemoryS3()
        store = SimpleNamespace(client=fake, bucket="b")
        manager = StreamingArchiveManager(store, index_prefix="test")
        rid = "f" * 32
        prefix = "test/" + rid + "/"
        confirmed = {
            "recording_id": rid,
            "object_prefix": prefix,
            "recording_status": "complete",
            "upload_status": "complete",
            "archive_storage": STORAGE,
        }
        fake.objects[("b", prefix + "manifest.json")] = (
            json.dumps(confirmed).encode(),
            {},
        )
        r = SimpleNamespace(
            directory=tmp_path / rid,
            manifest={
                **confirmed,
                "recording_status": "failed",
                "upload_status": "failed",
            },
            token_hash=None,
            failure="lost reply",
        )
        manager.publish_failure(r)
        await asyncio.gather(*list(manager.tasks))
        assert r.failure is None and r.manifest == confirmed
        assert (
            json.loads(fake.objects[("b", prefix + "manifest.json")][0])[
                "upload_status"
            ]
            == "complete"
        )
        await manager.close()

    asyncio.run(run())


def test_real_upload_failure_releases_capacity_without_blocking_next_session(
    tmp_path, local_s3
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.recording import S3ArchiveStore

    fake, endpoint = local_s3
    fake.reject_parts = True

    async def run():
        store = S3ArchiveStore("bucket", endpoint, "ap-southeast-7")
        manager = StreamingArchiveManager(
            store,
            index_prefix="test",
            recorder_slots=1,
            queue_bytes=64 << 20,
            finish_timeout=20,
        )
        rid = "a1" * 16
        kwargs = dict(
            directory=tmp_path / rid,
            spec=VideoSpec(32, 32),
            world_id="w",
            trace_id="upload-failed",
            inference={},
            ui={"mode": "browser"},
            references=[],
            object_prefix="test/" + rid + "/",
        )
        recording = await manager.create_recording(**kwargs)
        await asyncio.wait_for(recording.startup_complete.wait(), 10)
        # Real boto3 UploadPart gets two 503 responses. Live offers still return
        # immediately, and the whole archive fails rather than losing alignment.
        for i in range(4):
            await recording.write_source_state({"index": i, "payload": "x" * (3 << 20)})
        manager.finalize(recording, "completed")
        ticks = 0
        async with asyncio.timeout(15):
            while not recording.closed:
                ticks += 1
                await asyncio.sleep(0.005)
        assert ticks >= 10
        await asyncio.gather(*list(manager.tasks))
        assert recording.failure
        assert recording.failure not in {"QUEUE_FULL", "BYTE_BUDGET_EXCEEDED"}
        assert not recording.process.is_alive()
        assert not fake.uploads
        assert fake.aborts
        manifest = json.loads(
            fake.objects[("bucket", kwargs["object_prefix"] + "manifest.json")][0]
        )
        assert manifest["upload_status"] == "failed"
        assert not any(key.endswith(".mp4") for _, key in fake.objects)

        fake.reject_parts = False
        rid = "a2" * 16
        kwargs.update(
            directory=tmp_path / rid,
            object_prefix="test/" + rid + "/",
            trace_id="next-session",
        )
        next_recording = await manager.create_recording(**kwargs)
        await asyncio.wait_for(next_recording.startup_complete.wait(), 10)
        assert next_recording.worker_ready
        manager.finalize(next_recording, "completed")
        await asyncio.gather(*list(manager.tasks))
        assert next_recording.manifest["upload_status"] == "complete"
        # No frames is a partial recording, with no invalid empty video exposed.
        assert next_recording.manifest["recording_status"] == "partial"
        assert next_recording.manifest["videos"] == {}
        assert not any(key.endswith(".mp4") for _, key in fake.objects)
        await manager.close()
        assert not list(tmp_path.iterdir())

    asyncio.run(run())


def test_full_room_runtime_game_journal_is_diskless_and_slow_upload_does_not_hold_admission(
    tmp_path,
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_mock import (
        shared_mock_room,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.room_runtime import RoomRuntime
    from sglang.multimodal_gen.runtime.realtime.whitebox.rooms import (
        MemoryRoomStore,
        RoomDirectory,
        new_id,
    )

    waiting, release = threading.Event(), threading.Event()

    class Store(MemoryS3):
        def put_object(self, **kw):
            if kw["Key"].endswith("inputs.jsonl"):
                waiting.set()
                assert release.wait(5)
            return super().put_object(**kw)

    async def run():
        fake = Store()
        manager = StreamingArchiveManager(
            SimpleNamespace(client=fake, bucket="b"), index_prefix="test"
        )
        directory = RoomDirectory(MemoryRoomStore(), {"a": "http://a"})
        runtime = RoomRuntime(
            directory=directory,
            owner_id="a",
            driver_factory=shared_mock_room,
            spool=tmp_path,
            archives=manager,
            prefix="test",
            spec=VideoSpec(32, 32),
        )
        logs = []
        try:
            for attempt in range(2):
                game = new_id()
                await directory.resolve(game, "world", create=True)
                request = dict(
                    game_id=game,
                    world_id="world",
                    player_id=new_id(),
                    stream_id=new_id(),
                    session_id=new_id(),
                    user_id="test",
                    trace_id="game-journal",
                    spec={"width": 32, "height": 32, "fps": 24},
                )
                started = time.monotonic()
                joined = await runtime.join(request)
                assert time.monotonic() - started < 0.1
                log = runtime.rooms[game]["log"]
                logs.append(log)
                assert log.diskless and log.journal is None
                payload = {"keys": ["KeyW"]}
                await runtime.event(
                    joined["stream_id"], "input", {"sequence": 0, "input": payload}
                )
                payload["keys"].clear()
                if attempt:
                    # Oversized archival metadata must not prevent source frames.
                    await runtime.event(
                        joined["stream_id"], "diagnostic", {"text": "x" * 65536}
                    )
                    assert log.failure and not log.buffer
                assert (await runtime.next_frame(joined["stream_id"]))["index"] == 0
                started = time.monotonic()
                await runtime.leave(joined["stream_id"])
                assert time.monotonic() - started < 0.2
                assert not runtime.rooms and not runtime.players
                if not attempt:
                    assert await asyncio.to_thread(waiting.wait, 2)
                    assert manager.active
            release.set()
            await asyncio.gather(*list(manager.tasks))
            assert logs[0].manifest["upload_status"] == "complete"
            raw = fake.objects[
                ("b", logs[0].manifest["object_prefix"] + "inputs.jsonl")
            ][0]
            events = [json.loads(line) for line in raw.splitlines()]
            assert [e["room_seq"] for e in events] == [1, 2, 3]
            assert events[1]["payload"]["input"]["keys"] == ["KeyW"]
            assert logs[1].manifest["upload_status"] == "failed"
            assert all(not log.buffer and log.archive_payload is None for log in logs)
            assert not list(tmp_path.iterdir())
        finally:
            release.set()
            await runtime.close()

    asyncio.run(run())
