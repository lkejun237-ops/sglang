# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
import threading
import types

import pytest

from sglang.multimodal_gen.runtime.realtime import world_recording
from sglang.multimodal_gen.runtime.realtime.world_recording import RecordingManager


class _S3:
    def __init__(self, *, block_create=False, fail_upload=False):
        self.block_create = block_create
        self.fail_upload = fail_upload
        self.create_started = threading.Event()
        self.release_create = threading.Event()
        if not block_create:
            self.release_create.set()
        self.created = []
        self.uploaded = []
        self.completed = []
        self.aborted = []

    def create_multipart_upload(self, **kwargs):
        self.created.append(kwargs)
        self.create_started.set()
        if not self.release_create.wait(timeout=2):
            raise TimeoutError("test did not release CreateMultipartUpload")
        return {"UploadId": "upload-1"}

    def upload_part(self, **kwargs):
        if self.fail_upload:
            raise OSError("injected upload failure")
        self.uploaded.append(kwargs)
        return {"ETag": f"etag-{kwargs['PartNumber']}"}

    def complete_multipart_upload(self, **kwargs):
        self.completed.append(kwargs)
        return {}

    def abort_multipart_upload(self, **kwargs):
        self.aborted.append(kwargs)
        return {}


async def _eventually(predicate, *, timeout=2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was not met before timeout")
        await asyncio.sleep(0.001)


def _init():
    return {"type": "media_init", "codec": "h264", "container": "fmp4"}


def _payload(sequence, payload):
    return {"type": "media_payload", "sequence": sequence, "payload": payload}


def _final():
    return {"type": "media_chunk_complete", "is_final_chunk": True}


def _manager(s3, callbacks, *, uri_scheme="s3"):
    async def recorded(run_id, attempt_id, video_uri):
        callbacks.append((run_id, attempt_id, video_uri))
        return True

    return RecordingManager(
        bucket="run-videos",
        s3_client=s3,
        on_recorded=recorded,
        uri_scheme=uri_scheme,
    )


def test_recording_streams_ordered_parts_and_callbacks_only_after_complete(monkeypatch):
    async def run():
        monkeypatch.setattr(world_recording, "PART_SIZE", 8)
        monkeypatch.setattr(world_recording, "QUEUE_MAX_BYTES", 8)
        s3 = _S3()
        callbacks = []
        manager = _manager(s3, callbacks)
        recorder = manager.start(
            run_id="run-1",
            attempt_id="attempt-1",
            session_id="session-1",
            generation_id="generation-1",
            trace_id="trace-1",
        )

        recorder.offer(_init())
        recorder.offer(_payload(0, b"12345678"))
        await _eventually(lambda: recorder.queued_bytes == 0)
        recorder.offer(_payload(1, b"90"))
        recorder.offer(_final())
        recorder.session_closed()  # A valid final marker detaches finalization.

        assert await recorder.wait() == "ready"
        await _eventually(lambda: bool(callbacks))
        await manager.close()

        assert [part["Body"] for part in s3.uploaded] == [b"12345678", b"90"]
        assert [part["PartNumber"] for part in s3.uploaded] == [1, 2]
        assert s3.created == [
            {
                "Bucket": "run-videos",
                "Key": "runs/run-1/video.mp4",
                "ContentType": "video/mp4",
                "CacheControl": "public, max-age=31536000, immutable",
            }
        ]
        assert len(s3.completed) == 1
        assert not s3.aborted
        assert callbacks == [
            (
                "run-1",
                "attempt-1",
                "s3://run-videos/runs/run-1/video.mp4",
            )
        ]
        assert s3.completed[0]["MultipartUpload"] == {
            "Parts": [
                {"ETag": "etag-1", "PartNumber": 1},
                {"ETag": "etag-2", "PartNumber": 2},
            ]
        }

    asyncio.run(run())


def test_recording_can_report_an_oss_uri(monkeypatch):
    async def run():
        monkeypatch.setattr(world_recording, "PART_SIZE", 8)
        monkeypatch.setattr(world_recording, "QUEUE_MAX_BYTES", 8)
        s3 = _S3()
        callbacks = []
        manager = _manager(s3, callbacks, uri_scheme="oss")
        recorder = manager.start(
            run_id="run-cn-1",
            attempt_id="attempt-cn-1",
            session_id="session-cn-1",
            generation_id="generation-cn-1",
            trace_id="trace-cn-1",
        )

        recorder.offer(_init())
        recorder.offer(_payload(0, b"video"))
        recorder.offer(_final())
        recorder.session_closed()

        assert await recorder.wait() == "ready"
        await _eventually(lambda: bool(callbacks))
        await manager.close()
        assert callbacks == [
            (
                "run-cn-1",
                "attempt-cn-1",
                "oss://run-videos/runs/run-cn-1/video.mp4",
            )
        ]

    asyncio.run(run())


def test_recording_factory_scopes_explicit_credentials_to_one_profile(monkeypatch):
    captured = {}

    def client(service, **kwargs):
        captured.update(service=service, **kwargs)
        return _S3()

    class Config:
        def __init__(self, **kwargs):
            self.values = kwargs

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=client))
    monkeypatch.setitem(sys.modules, "botocore", types.ModuleType("botocore"))
    monkeypatch.setitem(
        sys.modules, "botocore.config", types.SimpleNamespace(Config=Config)
    )
    manager = world_recording.create_s3_recording_manager(
        bucket="cn-runs",
        region="cn-beijing",
        endpoint_url="https://s3.oss-cn-beijing.aliyuncs.com",
        uri_scheme="oss",
        access_key_id="profile-key",
        secret_access_key="profile-secret",
        on_recorded=lambda *_args: None,
    )

    assert captured["service"] == "s3"
    assert captured["endpoint_url"].startswith("https://s3.oss-")
    assert captured["aws_access_key_id"] == "profile-key"
    assert captured["aws_secret_access_key"] == "profile-secret"
    assert captured["config"].values["s3"] == {"addressing_style": "virtual"}
    assert captured["config"].values["request_checksum_calculation"] == "when_required"
    assert captured["config"].values["response_checksum_validation"] == "when_required"
    asyncio.run(manager.close())


def test_recording_factory_rejects_partial_credentials():
    with pytest.raises(ValueError, match="both key fields"):
        world_recording.create_s3_recording_manager(
            bucket="cn-runs",
            region="cn-beijing",
            access_key_id="profile-key",
            on_recorded=lambda *_args: None,
        )


def test_recording_buffer_overflow_aborts_without_waiting_for_s3(monkeypatch):
    async def run():
        monkeypatch.setattr(world_recording, "PART_SIZE", 8)
        monkeypatch.setattr(world_recording, "QUEUE_MAX_BYTES", 8)
        s3 = _S3(block_create=True)
        callbacks = []
        manager = _manager(s3, callbacks)
        recorder = manager.start(
            run_id="run-2",
            attempt_id="attempt-2",
            session_id="session-2",
            generation_id="generation-2",
            trace_id="trace-2",
        )

        recorder.offer(_init())
        await _eventually(s3.create_started.is_set)
        recorder.offer(_payload(0, b"12345678"))
        recorder.offer(_payload(1, b"9"))

        assert recorder.state == "failed"
        assert recorder.failure_reason == "buffer_full"
        s3.release_create.set()
        assert await recorder.wait() == "buffer_full"
        await manager.close()

        assert not s3.uploaded
        assert not s3.completed
        assert len(s3.aborted) == 1
        assert not callbacks

    asyncio.run(run())


def test_recording_without_final_marker_is_incomplete_and_aborted(monkeypatch):
    async def run():
        monkeypatch.setattr(world_recording, "PART_SIZE", 8)
        monkeypatch.setattr(world_recording, "QUEUE_MAX_BYTES", 8)
        s3 = _S3()
        callbacks = []
        manager = _manager(s3, callbacks)
        recorder = manager.start(
            run_id="run-3",
            attempt_id="attempt-3",
            session_id="session-3",
            generation_id="generation-3",
            trace_id="trace-3",
        )

        recorder.offer(_init())
        await _eventually(s3.create_started.is_set)
        recorder.offer(_payload(0, b"partial"))
        manager.session_closed("session-3", "generation-3")

        assert await recorder.wait() == "incomplete"
        await manager.close()

        assert not s3.completed
        assert len(s3.aborted) == 1
        assert not callbacks

    asyncio.run(run())


def test_recording_upload_failure_is_isolated_and_aborted(monkeypatch):
    async def run():
        monkeypatch.setattr(world_recording, "PART_SIZE", 4)
        monkeypatch.setattr(world_recording, "QUEUE_MAX_BYTES", 8)
        s3 = _S3(fail_upload=True)
        callbacks = []
        manager = _manager(s3, callbacks)
        recorder = manager.start(
            run_id="run-4",
            attempt_id="attempt-4",
            session_id="session-4",
            generation_id="generation-4",
            trace_id="trace-4",
        )

        recorder.offer(_init())
        recorder.offer(_payload(0, b"data"))
        recorder.offer(_final())

        assert await recorder.wait() == "upload_failed"
        await manager.close()

        assert not s3.completed
        assert len(s3.aborted) == 1
        assert not callbacks

    asyncio.run(run())


@pytest.mark.parametrize(
    "messages",
    [
        [_payload(0, b"without-init")],
        [_init(), _payload(1, b"sequence-gap")],
        [_init(), _final()],
        [{"type": "media_init", "codec": "h264", "container": "mpegts"}],
    ],
)
def test_recording_rejects_incomplete_media_without_publishing(messages):
    async def run():
        s3 = _S3()
        callbacks = []
        manager = _manager(s3, callbacks)
        recorder = manager.start(
            run_id="run-invalid",
            attempt_id="attempt-invalid",
            session_id="session-invalid",
            generation_id="generation-invalid",
            trace_id="trace-invalid",
        )

        for message in messages:
            recorder.offer(message)

        assert await recorder.wait() == "incomplete"
        await manager.close()

        assert not s3.completed
        assert not callbacks

    asyncio.run(run())


def test_production_recording_uses_configurable_queue_larger_than_s3_part_size(
    monkeypatch,
):
    assert world_recording.PART_SIZE == 5 * 1024 * 1024
    assert world_recording.QUEUE_MAX_BYTES == 16 * 1024 * 1024
    monkeypatch.delenv(world_recording.QUEUE_MAX_BYTES_ENV, raising=False)
    assert world_recording._load_recording_queue_max_bytes() == 16 * 1024 * 1024

    monkeypatch.setenv(world_recording.QUEUE_MAX_BYTES_ENV, str(24 * 1024 * 1024))
    assert world_recording._load_recording_queue_max_bytes() == 24 * 1024 * 1024


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not-bytes", "must be an integer number of bytes"),
        (str(5 * 1024 * 1024 - 1), "must be at least"),
    ],
)
def test_production_recording_rejects_invalid_queue_size(monkeypatch, value, message):
    monkeypatch.setenv(world_recording.QUEUE_MAX_BYTES_ENV, value)
    with pytest.raises(ValueError, match=message):
        world_recording._load_recording_queue_max_bytes()
