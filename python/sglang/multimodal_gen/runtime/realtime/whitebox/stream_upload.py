# SPDX-License-Identifier: Apache-2.0
"""Forward-only archive objects with a bounded, worker-owned upload pool.

No filesystem, temp file, or unbounded executor queue is used. This module runs
in an archive child process, never in the source/inference event loop.
"""

from __future__ import annotations

import base64
import hashlib
import io
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from uuid import uuid4


class UploadBudgetExceeded(RuntimeError):
    pass


def s3_client(config):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=config["endpoint"],
        region_name=config["region"],
        config=Config(
            s3={"addressing_style": "virtual"},
            connect_timeout=3,
            read_timeout=10,
            # Initiation/completion have unknown-outcome semantics. Part/PUT
            # retries below reuse the exact object identity and bytes.
            retries={"total_max_attempts": 1},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            max_pool_connections=4,
        ),
    )


def retry_same_bytes(call):
    for attempt in range(2):
        try:
            return call()
        except Exception:
            if attempt:
                raise
            time.sleep(0.1)


class UploadPool:
    def __init__(
        self,
        *,
        max_bytes=32 << 20,
        max_jobs=16,
        workers=2,
        max_archive_bytes=2 << 30,
        max_buffer_bytes=96 << 20,
    ):
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.max_bytes, self.max_jobs = max_bytes, max_jobs
        self.max_archive_bytes = max_archive_bytes
        self.max_buffer_bytes, self.buffer_bytes = max_buffer_bytes, 0
        self.pending = {}
        self.pending_bytes = self.total_bytes = self.high_water_bytes = 0
        self.confirmed_bytes = 0
        self.finishing = False
        self.deadline = None

    def check(self):
        for future, size in list(self.pending.items()):
            if future.done():
                del self.pending[future]
                self.pending_bytes -= size
                future.result()
                self.confirmed_bytes += size

    def account(self, size):
        if size < 0 or self.total_bytes + size > self.max_archive_bytes:
            raise UploadBudgetExceeded("Archive byte budget exceeded")
        self.total_bytes += size

    def submit(self, function, *, size):
        self.check()
        if size > self.max_bytes:
            raise UploadBudgetExceeded("Upload item exceeds memory budget")
        while (
            len(self.pending) >= self.max_jobs
            or self.pending_bytes + size > self.max_bytes
        ):
            if not self.finishing:
                raise UploadBudgetExceeded("Upload backlog exceeded")
            remaining = self.deadline - time.monotonic()
            if remaining <= 0 or not self.pending:
                raise TimeoutError("Archive completion timed out")
            wait(self.pending, timeout=min(0.1, remaining), return_when=FIRST_COMPLETED)
            self.check()
        future = self.executor.submit(function)
        self.pending[future] = size
        self.pending_bytes += size
        self.high_water_bytes = max(self.high_water_bytes, self.pending_bytes)
        return future

    def begin_finish(self, seconds=90):
        self.finishing = True
        self.deadline = time.monotonic() + seconds

    def result(self, future):
        remaining = self.deadline - time.monotonic() if self.deadline else 30
        if remaining <= 0:
            raise TimeoutError("Archive completion timed out")
        value = future.result(timeout=remaining)
        self.check()
        return value

    def close(self):
        # Parent process watchdog bounds the lifetime of any stuck SDK thread.
        self.executor.shutdown(wait=True, cancel_futures=True)


class MultipartObject(io.RawIOBase):
    """A bounded non-seekable MP4/JSONL sink, preserving bytes exactly."""

    def __init__(self, client, bucket, key, pool, *, content_type, part_bytes=8 << 20):
        super().__init__()
        self.client, self.bucket, self.key, self.pool = client, bucket, key, pool
        self.content_type, self.part_bytes = content_type, part_bytes
        self.buffer = bytearray()
        self.digest = hashlib.sha256()
        self.length = 0
        self.nonce = uuid4().hex
        self.initiated = None
        self.parts = []
        self.committed = False

    def writable(self):
        return True

    def seekable(self):
        return False

    def tell(self):
        return self.length

    def write(self, data):
        if self.closed or self.committed:
            raise ValueError("Archive object is closed")
        self.pool.check()
        view = memoryview(data)
        self.pool.account(len(view))
        self.digest.update(view)
        self.length += len(view)
        while view:
            size = min(len(view), self.part_bytes - len(self.buffer))
            if self.pool.buffer_bytes + size > self.pool.max_buffer_bytes:
                raise UploadBudgetExceeded("Archive mux buffers exceeded")
            self.buffer.extend(view[:size])
            self.pool.buffer_bytes += size
            view = view[size:]
            if len(self.buffer) == self.part_bytes:
                self._part()
        return len(data)

    def _part(self):
        if not self.buffer:
            return
        if len(self.parts) >= 10000:
            raise UploadBudgetExceeded("Too many archive parts")
        if self.initiated is None:
            self.initiated = self.pool.submit(
                lambda: self.client.create_multipart_upload(
                    Bucket=self.bucket,
                    Key=self.key,
                    ContentType=self.content_type,
                    Metadata={"archive-nonce": self.nonce},
                )["UploadId"],
                size=0,
            )
        data = bytes(self.buffer)
        self.pool.buffer_bytes -= len(data)
        self.buffer.clear()
        number = len(self.parts) + 1

        def upload():
            upload_id = self.initiated.result(timeout=15)
            result = retry_same_bytes(
                lambda: self.client.upload_part(
                    Bucket=self.bucket,
                    Key=self.key,
                    UploadId=upload_id,
                    PartNumber=number,
                    Body=data,
                    ContentLength=len(data),
                    ContentMD5=base64.b64encode(hashlib.md5(data).digest()).decode(),
                )
            )
            return {"PartNumber": number, "ETag": result["ETag"]}

        self.parts.append(self.pool.submit(upload, size=len(data)))

    def _confirmed(self):
        head = self.client.head_object(Bucket=self.bucket, Key=self.key)
        return (
            head.get("ContentLength") == self.length
            and head.get("Metadata", {}).get("archive-nonce") == self.nonce
        )

    def finish(self):
        if self.committed:
            return {"bytes": self.length, "sha256": self.digest.hexdigest()}
        if self.initiated is not None:
            self._part()
            parts = [self.pool.result(future) for future in self.parts]
            upload_id = self.pool.result(self.initiated)
            try:
                self.client.complete_multipart_upload(
                    Bucket=self.bucket,
                    Key=self.key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts},
                )
            except Exception:
                # A lost completion reply must not create a second upload or
                # publish success without checking this exact generation.
                if not self._confirmed():
                    raise
        else:
            data = bytes(self.buffer)
            self.pool.buffer_bytes -= len(data)
            self.buffer.clear()
            future = self.pool.submit(
                lambda: retry_same_bytes(
                    lambda: self.client.put_object(
                        Bucket=self.bucket,
                        Key=self.key,
                        Body=data,
                        ContentType=self.content_type,
                        ContentMD5=base64.b64encode(
                            hashlib.md5(data).digest()
                        ).decode(),
                        Metadata={"archive-nonce": self.nonce},
                    )
                ),
                size=len(data),
            )
            self.pool.result(future)
        if not self._confirmed():
            raise RuntimeError("Archive object confirmation failed")
        self.committed = True
        return {"bytes": self.length, "sha256": self.digest.hexdigest()}

    def abort(self):
        if self.committed or self.initiated is None or not self.initiated.done():
            return
        try:
            self.client.abort_multipart_upload(
                Bucket=self.bucket,
                Key=self.key,
                UploadId=self.initiated.result(),
            )
        except Exception:
            pass  # A prefix-scoped OSS lifecycle is the crash cleanup backstop.


class TextObject:
    def __init__(self, target):
        self.target, self.closed = target, False

    def write(self, value):
        if self.closed:
            raise ValueError("Archive text object is closed")
        self.target.write(value.encode("utf-8"))
        return len(value)

    def flush(self):
        pass

    def close(self):
        self.closed = True
