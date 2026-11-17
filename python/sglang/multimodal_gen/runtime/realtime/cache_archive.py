# SPDX-License-Identifier: Apache-2.0
"""Bounded OSS snapshots of immutable Block0 artifacts.

Only CPU background workers use this module. It never owns a model Slot or
runs inference. A small complete marker is committed after every payload.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from uuid import uuid4

from .cache_tiers import CacheTiers, StageTimes

HEX = re.compile(r"[0-9a-f]{64}\Z")
ARTIFACT = re.compile(r"[0-9a-f]{32}\Z")
SCHEMA = "zing-block0-archive-v1"
RESTORE_GET_WORKERS = 4


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def world_token(world):
    if not isinstance(world, str) or not world.strip() or len(world.encode()) > 256:
        raise ValueError("Invalid cache world identity")
    return hashlib.sha256(world.encode()).hexdigest()


class ArchiveUnavailable(RuntimeError):
    pass


class ArchiveProbeTimeout(ArchiveUnavailable):
    """The archive did not confirm a hit within the metadata budget."""


class S3CacheObjects:
    """Same boto3 credential chain as the existing recording uploader."""

    def __init__(self, policy, *, client=None):
        self.bucket = policy.bucket
        self.deadline_s = policy.deadline_s
        if client is None:
            from botocore.config import Config

            from .cache_credentials import storage_session

            client = storage_session().client(
                "s3",
                endpoint_url=policy.endpoint,
                region_name=policy.region,
                config=Config(
                    s3={"addressing_style": "virtual"},
                    connect_timeout=3,
                    read_timeout=10,
                    retries={"mode": "standard", "total_max_attempts": 2},
                    request_checksum_calculation="when_required",
                    response_checksum_validation="when_required",
                ),
            )
        self.client = client
        # OSS multi-delete requires Content-MD5; new botocore defaults to an
        # AWS checksum header instead. Sign the actual serialized XML body.
        self.client.meta.events.register(
            "before-call.s3.DeleteObjects",
            self._delete_md5,
            unique_id="sglang-cache-oss-delete-md5",
        )

    @staticmethod
    def _delete_md5(params, **_):
        import base64

        body = params.get("body")
        if isinstance(body, (bytes, bytearray)):
            params.setdefault("headers", {})["Content-MD5"] = base64.b64encode(
                hashlib.md5(body, usedforsecurity=False).digest()
            ).decode("ascii")

    def get(self, key, limit):
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if str(code) in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        body = response["Body"]
        try:
            length = response["ContentLength"]
            if not 0 <= length <= limit:
                raise ArchiveUnavailable("Cache object exceeds its transfer budget")
            data = body.read(limit + 1)
            if len(data) != length:
                raise ArchiveUnavailable("Incomplete cache object read")
            return data
        finally:
            body.close()

    def put(self, key, data):
        from boto3.s3.transfer import TransferConfig

        self.client.upload_fileobj(
            io.BytesIO(data),
            self.bucket,
            key,
            Config=TransferConfig(
                use_threads=False,
                multipart_threshold=64 << 20,
                multipart_chunksize=16 << 20,
            ),
        )

    def size(self, key):
        try:
            return self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"]
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if str(code) in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise

    def keys(self, prefix):
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                yield item["Key"]

    def delete(self, key):
        """Retire one completed upload intent without administrative list rights."""
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def delete_prefix(self, prefix):
        """Versions are removed too; access denial is never treated as success."""
        deleted = 0
        deadline = time.monotonic() + self.deadline_s

        def check():
            if time.monotonic() >= deadline:
                raise ArchiveUnavailable("Cache deletion deadline exceeded")

        # A non-versioned bucket also supports listing versions (null versions).
        pages = self.client.get_paginator("list_object_versions")
        for page in pages.paginate(Bucket=self.bucket, Prefix=prefix):
            check()
            objects = [
                {"Key": row["Key"], "VersionId": row["VersionId"]}
                for kind in ("Versions", "DeleteMarkers")
                for row in page.get(kind, [])
            ]
            for start in range(0, len(objects), 1000):
                check()
                result = self.client.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": objects[start : start + 1000], "Quiet": False},
                )
                if result.get("Errors"):
                    raise ArchiveUnavailable("Some cache versions could not be deleted")
                deleted += len(result.get("Deleted", []))
        # Some S3-compatible endpoints omit unversioned objects above.
        batch = []
        for key in self.keys(prefix):
            check()
            batch.append({"Key": key})
            if len(batch) == 1000:
                result = self.client.delete_objects(
                    Bucket=self.bucket, Delete={"Objects": batch}
                )
                if result.get("Errors"):
                    raise ArchiveUnavailable("Some cache objects could not be deleted")
                deleted += len(batch)
                batch = []
        if batch:
            result = self.client.delete_objects(
                Bucket=self.bucket, Delete={"Objects": batch}
            )
            if result.get("Errors"):
                raise ArchiveUnavailable("Some cache objects could not be deleted")
            deleted += len(batch)
        pages = self.client.get_paginator("list_multipart_uploads")
        for page in pages.paginate(Bucket=self.bucket, Prefix=prefix):
            check()
            for row in page.get("Uploads", []):
                self.client.abort_multipart_upload(
                    Bucket=self.bucket, Key=row["Key"], UploadId=row["UploadId"]
                )
        return deleted


class CacheArchive:
    def __init__(self, policy: CacheTiers, namespace, *, objects=None, clock=time.time):
        if not policy.l3:
            raise ValueError("OSS cache tier is disabled")
        self.policy, self.namespace, self.clock = policy, namespace, clock
        self.objects = objects if objects is not None else S3CacheObjects(policy)
        self.prefix = policy.prefix
        self.namespace_token = hashlib.sha256(namespace.encode()).hexdigest()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="block0-archive"
        )
        # A slow SDK call can outlive the caller. Keep these read-only probes
        # bounded and independent of payload uploads and restores.
        self._probes = ThreadPoolExecutor(
            max_workers=policy.max_pending, thread_name_prefix="block0-oss-probe"
        )
        self._probe_slots = threading.BoundedSemaphore(policy.max_pending)
        self._slots = threading.BoundedSemaphore(policy.max_pending)
        self._lock = threading.RLock()
        self._active = {}
        self._observed_fences = OrderedDict()
        self._deleting = set()
        self._closed = False
        self.owner_id = uuid4().hex

    def _world(self, world, kind="data"):
        return self.prefix + kind + "/" + world_token(world) + "/"

    def _data_root(self, world, epoch, key):
        # A single world fence covers all model/layout namespaces in this region.
        # Management can delete a world without loading a model or scanning peers.
        return self._world(world) + epoch + "/" + self.namespace_token + "/" + key + "/"

    def _fence(self, world):
        raw = self.objects.get(self._world(world, "control") + "fence.json", 4096)
        value = (
            {"epoch": "initial", "deleting": False} if raw is None else json.loads(raw)
        )
        epoch = value.get("epoch")
        if epoch != "initial" and not (
            isinstance(epoch, str) and ARTIFACT.fullmatch(epoch)
        ):
            raise ArchiveUnavailable("Invalid cache deletion fence")
        if type(value.get("deleting")) is not bool:
            raise ArchiveUnavailable("Invalid cache deletion fence")
        with self._lock:
            self._observed_fences[world] = value
            self._observed_fences.move_to_end(world)
            if len(self._observed_fences) > 4096:
                self._observed_fences.popitem(last=False)
        return value

    def submission_fence(self, world):
        """Pin outer Mooncake CPU-queue work without an OSS read on the GPU thread.

        A producer that has never seen this world assumes the initial generation.
        If a deletion already happened, the background check learns its fence and
        rejects this attempt. A later submission can use the observed generation.
        """
        with self._lock:
            return self._observed_fences.get(
                world, {"epoch": "initial", "deleting": False}
            ).copy()

    def check_submission(self, world, expected):
        fence = self._fence(world)
        if expected is not None and expected != fence:
            raise ArchiveUnavailable("Cache upload was fenced before archive admission")
        return fence

    def submit(self, world, key, manifest, read_object, *, expected_fence=None):
        """Admit a background write against the current world generation."""
        world_token(world)
        if not HEX.fullmatch(key):
            raise ValueError("Invalid cache key")
        with self._lock:
            if self._closed or world in self._deleting:
                return None
        # Mooncake calls submit from its persistence worker. Read the fence
        # outside the local lock and before queueing behind other worlds.
        fence = self.check_submission(world, expected_fence)
        if fence["deleting"]:
            return None
        with self._lock:
            if self._closed or world in self._deleting:
                return None
            identity = (world, key, fence["epoch"])
            old = self._active.get(identity)
            if old is not None:
                return old
            if not self._slots.acquire(blocking=False):
                return None
            cancelled = threading.Event()
            try:
                future = self._executor.submit(
                    self._publish, world, key, manifest, read_object, cancelled, fence
                )
            except BaseException:
                self._slots.release()
                raise
            self._active[identity] = future
            future.cancel_archive = cancelled.set

            def done(_):
                with self._lock:
                    self._active.pop(identity, None)
                self._slots.release()

            future.add_done_callback(done)
            return future

    @contextmanager
    def _publication_turn(self, world, key, fence, deadline, current, cancelled):
        # Use unique intents and strongly consistent prefix listings. A later
        # writer cannot enter while a publisher's intent is still visible.
        prefix = (
            self._world(world, "uploads")
            + f"{fence['epoch']}/{self.namespace_token}/{key}/"
        )
        lease = prefix + uuid4().hex + ".json"
        intent = canonical(
            dict(
                epoch=fence["epoch"],
                owner_id=self.owner_id,
                created_at=self.clock(),
                expires_at=self.clock() + self.policy.deadline_s + 60,
            )
        )
        delay = 0.1
        while True:
            current()
            existing = self.find_durable(world, key, deadline=deadline)
            if existing is not None:
                current()
                yield existing
                return
            if next(iter(self.objects.keys(prefix)), None) is None:
                try:
                    self.objects.put(lease, intent)
                    if list(self.objects.keys(prefix)) == [lease]:
                        # Another writer may have published and retired between
                        # our first lookup and admission. Check again under the turn.
                        existing = self.find_durable(world, key, deadline=deadline)
                        current()
                        yield existing
                        return
                finally:
                    self.objects.delete(lease)
            # Simultaneous arrivals withdraw only their own intent and back off.
            # A timed-out waiter never takes over an unconfirmed publisher.
            cancelled.wait(
                min(
                    random.uniform(delay / 2, delay),
                    max(0, deadline - time.monotonic()),
                )
            )
            delay = min(2.0, delay * 2)

    def _publish(self, world, key, native, read_object, cancelled, fence):
        timing = StageTimes()
        began = self.clock()
        deadline = time.monotonic() + self.policy.deadline_s
        artifact = native["artifact_id"]
        if not ARTIFACT.fullmatch(artifact):
            raise ValueError("Invalid cache artifact identity")
        root = self._data_root(world, fence["epoch"], key)
        folder = root + artifact + "/"

        def current():
            with self._lock:
                stopped = self._closed or world in self._deleting
            if (
                stopped
                or cancelled.is_set()
                or time.monotonic() >= deadline
                or fence["deleting"]
                or self._fence(world) != fence
            ):
                raise ArchiveUnavailable(
                    "Cache upload was cancelled, expired or fenced"
                )

        rows = []
        with self._publication_turn(
            world, key, fence, deadline, current, cancelled
        ) as existing:
            if existing is not None:
                return dict(
                    state="persisted",
                    tier="l3",
                    artifact_id=existing["artifact_id"],
                    deduplicated=True,
                    bytes=0,
                    stages=timing.stages,
                    elapsed_host_ms=(self.clock() - began) * 1000,
                    completed_epoch_ms=self.clock() * 1000,
                )
            for name, size in native["objects"].items():
                current()
                if (
                    type(size) is not int
                    or not 0 < size <= self.policy.max_object_bytes
                ):
                    raise ArchiveUnavailable("Cache object exceeds the archive budget")
                with timing.measure("l3_native_export"):
                    payload = read_object(name)
                current()
                if len(payload) != size:
                    raise ArchiveUnavailable("Cache snapshot was evicted or truncated")
                leaf = hashlib.sha256(name.encode()).hexdigest()
                with timing.measure("l3_upload"):
                    self.objects.put(folder + leaf, payload)
                rows.append(
                    {
                        "key": name,
                        "file": leaf,
                        "bytes": size,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
                del payload
            current()
            header = dict(
                schema=SCHEMA,
                world_id=world,
                cache_key=key,
                artifact_id=artifact,
                native=native,
                objects=rows,
                created_at=began,
                expires_at=began + self.policy.retention_days * 86400,
                epoch=fence["epoch"],
            )
            with timing.measure("l3_publish"):
                self.objects.put(folder + "manifest.json", canonical(header))
                current()
                self.objects.put(folder + "complete", b"")
                current()
                self.objects.put(
                    root + "current.json", canonical({"artifact_id": artifact})
                )
                current()
            return {
                "state": "persisted",
                "tier": "l3",
                "artifact_id": artifact,
                "deduplicated": False,
                "bytes": sum(r["bytes"] for r in rows),
                "elapsed_host_ms": (self.clock() - began) * 1000,
                "stages": timing.stages,
                "completed_epoch_ms": self.clock() * 1000,
            }

    def probe(self, world, key, *, queued_s=0.0):
        """Bound the whole existence decision, including waiting for a probe slot."""
        timeout = self.policy.oss_probe_timeout_s - queued_s
        deadline = time.monotonic() + timeout
        if timeout <= 0 or not self._probe_slots.acquire(timeout=timeout):
            raise ArchiveProbeTimeout("OSS cache metadata probe timed out")
        try:
            future = self._probes.submit(self.find, world, key, deadline=deadline)
        except BaseException:
            self._probe_slots.release()
            raise
        future.add_done_callback(lambda _: self._probe_slots.release())
        try:
            return future.result(timeout=max(0, deadline - time.monotonic()))
        except TimeoutError as error:
            future.cancel()
            raise ArchiveProbeTimeout("OSS cache metadata probe timed out") from error

    def find(self, world, key, *, deadline=None):
        if not HEX.fullmatch(key):
            raise ValueError("Invalid cache key")

        def check_deadline():
            if deadline is not None and time.monotonic() >= deadline:
                raise ArchiveProbeTimeout("OSS cache metadata probe timed out")

        def get_metadata(path, limit):
            check_deadline()
            data = self.objects.get(path, limit)
            check_deadline()
            return data

        check_deadline()
        fence = self._fence(world)
        check_deadline()
        if fence["deleting"]:
            return None
        root = self._data_root(world, fence["epoch"], key)
        index = get_metadata(root + "current.json", 4096)
        if not index:
            return None
        artifact = json.loads(index).get("artifact_id")
        if not isinstance(artifact, str) or not ARTIFACT.fullmatch(artifact):
            return None
        folder = root + artifact + "/"
        if get_metadata(folder + "complete", 1) != b"":
            return None
        raw = get_metadata(folder + "manifest.json", 2 << 20)
        if not raw:
            return None
        value = json.loads(raw)
        if (
            value.get("schema") != SCHEMA
            or value.get("cache_key") != key
            or value.get("world_id") != world
            or value.get("artifact_id") != artifact
            or value.get("epoch") != fence["epoch"]
            or type(value.get("expires_at")) not in (int, float)
            or self.clock() >= value["expires_at"]
        ):
            return None
        rows = value.get("objects")
        native = value.get("native")
        if not isinstance(rows, list) or not rows or not isinstance(native, dict):
            return None
        if len(rows) != len({r.get("key") for r in rows if isinstance(r, dict)}):
            return None
        if {r["key"]: r["bytes"] for r in rows} != native.get("objects"):
            return None
        expected = self.namespace + ":" + key + ":a" + artifact + ":"
        if any(
            not isinstance(r.get("key"), str) or not r["key"].startswith(expected)
            for r in rows
        ):
            return None
        check_deadline()
        value["folder"] = folder
        return value

    def find_durable(self, world, key, *, deadline):
        """Verify a published archive without exporting or downloading any KV."""
        found = self.find(world, key, deadline=deadline)
        if found is None or found["native"].get("artifact_id") != found["artifact_id"]:
            return None
        for row in found["objects"]:
            if time.monotonic() >= deadline:
                raise ArchiveUnavailable("Cache durability verification timed out")
            leaf = hashlib.sha256(row["key"].encode()).hexdigest()
            size = row["bytes"]
            if (
                row.get("file") != leaf
                or type(size) is not int
                or not 0 < size <= self.policy.max_object_bytes
                or self.objects.size(found["folder"] + leaf) != size
            ):
                return None
        if self._fence(world) != {"epoch": found["epoch"], "deleting": False}:
            raise ArchiveUnavailable("Cache durability check crossed a world deletion")
        return found

    def restore(
        self, world, key, put_object, *, deadline=None, found=None, timing=None
    ):
        """One payload read, no full hash pre-pass; the consumer checks sizes."""
        timing = timing if timing is not None else StageTimes()
        with timing.measure("l3_metadata"):
            value = self.probe(world, key) if found is None else found
        if value is None:
            return None
        deadline = (
            deadline
            if deadline is not None
            else time.monotonic() + self.policy.deadline_s
        )

        def download(row):
            if time.monotonic() >= deadline:
                raise ArchiveUnavailable("Cache restore deadline exceeded")
            with timing.measure("l3_download"):
                data = self.objects.get(value["folder"] + row["file"], row["bytes"])
            if data is None or len(data) != row["bytes"]:
                raise ArchiveUnavailable("Cache archive member is missing or truncated")
            return data

        pending = deque()

        def import_next():
            row, future = pending.popleft()
            data = future.result(timeout=max(0, deadline - time.monotonic()))
            if time.monotonic() >= deadline:
                raise ArchiveUnavailable("Cache restore deadline exceeded")
            with timing.measure("l3_native_import"):
                put_object(row["key"], data)
            return len(data)

        # Preserve the previous two-max-sized-payload memory bound per restore.
        # Completed GETs still reserve their bytes until the ordered import ends.
        byte_budget = 2 * self.policy.max_object_bytes
        reserved = 0
        with ThreadPoolExecutor(
            max_workers=RESTORE_GET_WORKERS, thread_name_prefix="block0-download"
        ) as pool:
            try:
                for row in value["objects"]:
                    size = row["bytes"]
                    if (
                        type(size) is not int
                        or not 0 < size <= self.policy.max_object_bytes
                        or row["file"]
                        != hashlib.sha256(row["key"].encode()).hexdigest()
                    ):
                        raise ArchiveUnavailable("Invalid cache archive member")
                    while (
                        len(pending) >= RESTORE_GET_WORKERS
                        or reserved + size > byte_budget
                    ):
                        reserved -= import_next()
                    pending.append((row, pool.submit(download, row)))
                    reserved += size
                while pending:
                    import_next()
            finally:
                for _, future in pending:
                    future.cancel()
        if self._fence(world) != {"epoch": value["epoch"], "deleting": False}:
            raise ArchiveUnavailable("Cache restore crossed a world deletion")
        return value["native"]

    @contextmanager
    def _delete_turn(self, world):
        # All namespaces share the world fence. As with publication turns,
        # strongly consistent LIST decides admission after unique intents
        # are visible. A competing instance can only withdraw its own intent.
        prefix = self._world(world, "control") + "deletions/"
        if next(iter(self.objects.keys(prefix)), None) is not None:
            raise ArchiveUnavailable("A world deletion is already running")
        intent = prefix + uuid4().hex + ".json"
        try:
            self.objects.put(
                intent,
                canonical({"owner_id": self.owner_id, "created_at": self.clock()}),
            )
            if list(self.objects.keys(prefix)) != [intent]:
                raise ArchiveUnavailable("A world deletion is already running")
            yield
        finally:
            self.objects.delete(intent)

    def delete_world(self, world):
        """Fence discovery immediately. Unconfirmed writers remain explicit."""
        with self._lock:
            if world in self._deleting:
                raise ArchiveUnavailable("A world deletion is already running")
            self._deleting.add(world)
            owned = [f for (w, _, _), f in self._active.items() if w == world]
            for future in owned:
                future.cancel_archive()
        try:
            with self._delete_turn(world):
                return self._delete_fenced_world(world, owned)
        finally:
            with self._lock:
                self._deleting.discard(world)

    def _delete_fenced_world(self, world, owned):
        epoch = uuid4().hex
        root = self._world(world)
        control = self._world(world, "control") + "fence.json"
        uploads = self._world(world, "uploads")
        self.objects.put(control, canonical({"epoch": epoch, "deleting": True}))
        for future in owned:
            try:
                future.result(timeout=self.policy.deadline_s + 30)
            except TimeoutError:
                return {
                    "state": "pending",
                    "logical_deleted": True,
                    "reason": "writer_not_drained",
                }
            except Exception:
                pass  # A cancelled/failed writer has drained its ownership.
        deleted = self.objects.delete_prefix(root)
        live = []
        for name in self.objects.keys(uploads):
            payload = self.objects.get(name, 4096)
            # Elapsed time is not proof that a remote native read/multipart
            # upload has stopped. Only retirement of its intent confirms it.
            if payload:
                live.append(name)
        if live:
            return {
                "state": "pending",
                "logical_deleted": True,
                "deleted": deleted,
                "reason": "remote_writers_not_drained",
            }
        # Reap leftovers after publishers have retired; never remove videos.
        deleted += self.objects.delete_prefix(root)
        # Writers retire their own unique intents; a late, fenced arrival
        # may still be between its registration and generation check.
        self.objects.put(control, canonical({"epoch": epoch, "deleting": False}))
        return {"state": "completed", "logical_deleted": True, "deleted": deleted}

    def close(self):
        with self._lock:
            self._closed = True
            for future in self._active.values():
                future.cancel_archive()
        self._executor.shutdown(wait=True)
        self._probes.shutdown(wait=True, cancel_futures=True)
