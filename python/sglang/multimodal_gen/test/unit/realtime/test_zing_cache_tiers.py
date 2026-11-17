# SPDX-License-Identifier: Apache-2.0
"""Real archive scheduling/bytes and native descriptor contracts, no cloud writes."""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from sglang.multimodal_gen.runtime.realtime.cache_archive import (
    ArchiveUnavailable,
    CacheArchive,
)
from sglang.multimodal_gen.runtime.realtime.cache_tiers import (
    CacheTiers,
    NativeReplicas,
)

KEY = "a" * 64
ART = "b" * 32
NSPACE = "test:g3:model"


class Objects:
    def __init__(self):
        self.data = {}
        self.writes = []
        self.fail = None
        self.lock = threading.RLock()

    def get(self, key, limit):
        with self.lock:
            result = self.data.get(key)
            if result is not None and len(result) > limit:
                raise ArchiveUnavailable("limit")
            return result

    def put(self, key, data):
        with self.lock:
            if self.fail and key.endswith(self.fail):
                raise OSError("injected upload error")
            self.data[key] = bytes(data)
            self.writes.append(key)

    def keys(self, prefix):
        with self.lock:
            return [key for key in self.data if key.startswith(prefix)]

    def size(self, key):
        with self.lock:
            return len(self.data[key]) if key in self.data else None

    def delete_prefix(self, prefix):
        with self.lock:
            keys = self.keys(prefix)
            for key in keys:
                del self.data[key]
            return len(keys)

    def delete(self, key):
        with self.lock:
            self.data.pop(key, None)


def policy(**kwargs):
    return replace(
        CacheTiers.parse(
            dict(
                l3=True,
                l1_to_l3=True,
                bucket="test",
                endpoint="https://oss.example",
                region="test",
            )
        ),
        **kwargs,
    )


def manifest(artifact=ART):
    parts = {
        f"{NSPACE}:{KEY}:a{artifact}:r0:kv:l00": b"real kv bytes",
        f"{NSPACE}:{KEY}:a{artifact}:r0:cond": b"conditions",
    }
    return dict(
        artifact_id=artifact, objects={k: len(v) for k, v in parts.items()}
    ), parts


def test_same_key_reuses_published_artifact_across_restarts_and_refills():
    objects = Objects()
    now = [1000.0]
    header, parts = manifest()
    writer = CacheArchive(policy(), NSPACE, objects=objects, clock=lambda: now[0])
    try:
        first = writer.submit("world", KEY, header, parts.__getitem__).result(3)
        assert not first["deduplicated"] and first["bytes"] == sum(
            map(len, parts.values())
        )
    finally:
        writer.close()
    writes = objects.writes.copy()
    reader = CacheArchive(policy(), NSPACE, objects=objects, clock=lambda: now[0])
    try:
        now[0] += 100
        other, fresh = manifest("c" * 32)
        for candidate in [header, other]:
            receipt = reader.submit(
                "world", KEY, candidate, lambda _: pytest.fail("duplicate export")
            ).result(3)
            assert (
                receipt["artifact_id"],
                receipt["deduplicated"],
                receipt["bytes"],
            ) == (ART, True, 0)
        assert objects.writes == writes
        assert reader.find("world", KEY)["created_at"] == 1000.0
        now[0] += 7 * 86400
        again = reader.submit("world", KEY, other, fresh.__getitem__).result(3)
        assert (
            not again["deduplicated"] and again["artifact_id"] == other["artifact_id"]
        )
    finally:
        reader.close()


def test_two_publishers_racing_on_an_empty_key_upload_once():
    # Force both processes to see an empty prefix before either registers.
    class RaceObjects(Objects):
        def __init__(self):
            super().__init__()
            self.start = threading.Barrier(2)
            self.first_list = threading.local()

        def keys(self, prefix):
            names = super().keys(prefix)
            if not getattr(self.first_list, "done", False):
                self.first_list.done = True
                self.start.wait(4)
            return names

    objects = RaceObjects()
    writers = [CacheArchive(policy(), NSPACE, objects=objects) for _ in range(2)]
    futures = []
    exports = []
    try:
        for writer, artifact in zip(writers, [ART, "c" * 32]):
            header, parts = manifest(artifact)

            def read(key, parts=parts):
                exports.append(key)
                return parts[key]

            futures.append(writer.submit("world", KEY, header, read))
        receipts = [f.result(8) for f in futures]
        assert sorted(r["deduplicated"] for r in receipts) == [False, True]
        assert len({r["artifact_id"] for r in receipts}) == 1
        assert len(exports) == len(parts)
        assert sum(k.endswith("/current.json") for k in objects.writes) == 1
    finally:
        for writer in writers:
            writer.close()


def test_missing_published_payload_is_repaired_and_uncertain_head_fails_closed(
    monkeypatch,
):
    objects = Objects()
    writer = CacheArchive(policy(), NSPACE, objects=objects)
    header, parts = manifest()
    try:
        writer.submit("world", KEY, header, parts.__getitem__).result(3)
        first = writer.find("world", KEY)
        writes = objects.writes.copy()
        other, fresh = manifest("c" * 32)
        with monkeypatch.context() as scoped:

            def forbidden(_):
                raise PermissionError("403 is not absence")

            scoped.setattr(objects, "size", forbidden)
            with pytest.raises(PermissionError):
                writer.submit("world", KEY, other, fresh.__getitem__).result(3)
        assert objects.writes == writes
        objects.delete(first["folder"] + first["objects"][0]["file"])
        receipt = writer.submit("world", KEY, other, fresh.__getitem__).result(3)
        assert (
            receipt["artifact_id"] == other["artifact_id"]
            and not receipt["deduplicated"]
        )
        restored = {}
        assert writer.restore("world", KEY, restored.__setitem__) == other
        assert restored == fresh
    finally:
        writer.close()


def test_uncertain_foreign_writer_is_never_stolen_and_retry_after_retirement_succeeds():
    objects = Objects()
    archive = CacheArchive(policy(deadline_s=0.2), NSPACE, objects=objects)
    header, parts = manifest()
    prefix = (
        archive._world("world", "uploads") + f"initial/{archive.namespace_token}/{KEY}/"
    )
    foreign_intent = prefix + "remote.json"
    objects.put(foreign_intent, b'{"expires_at":0}')
    try:
        with pytest.raises(ArchiveUnavailable):
            archive.submit(
                "world", KEY, header, lambda _: pytest.fail("overlapping export")
            ).result(2)
        assert objects.writes == [foreign_intent]
        assert objects.keys(prefix) == [foreign_intent]
        objects.delete(foreign_intent)
        receipt = archive.submit("world", KEY, header, parts.__getitem__).result(2)
        assert not receipt["deduplicated"] and receipt["artifact_id"] == ART
    finally:
        archive.close()


def test_normal_archive_write_does_not_require_administrative_listing():
    class WriterObjects(Objects):
        def delete_prefix(self, prefix):
            raise PermissionError("Administrative listing is unavailable")

    objects = WriterObjects()
    archive = CacheArchive(policy(), NSPACE, objects=objects)
    native, parts = manifest()
    try:
        assert (
            archive.submit("world", KEY, native, parts.__getitem__).result(5)["state"]
            == "persisted"
        )
        assert archive.find("world", KEY)["artifact_id"] == ART
        assert not objects.keys(archive.prefix + "uploads/")
    finally:
        archive.close()


def test_tier_configuration_enforces_independent_edges_and_scoped_oss():
    assert policy().l3 and not policy().l2
    with pytest.raises(ValueError, match="both tiers"):
        CacheTiers.parse(dict(l2_to_l3=True))
    with pytest.raises(ValueError, match="cache-only"):
        CacheTiers.parse(
            dict(
                l3=True,
                endpoint="https://oss.example",
                bucket="test",
                region="test",
                prefix="white_box_game_record/",
            )
        )
    with pytest.raises(ValueError, match="booleans"):
        CacheTiers.parse(dict(l1="false"))
    assert CacheTiers.parse(dict(l1=False)).durable_tier is None


def test_marker_last_and_returning_from_enqueue_does_not_wait_for_storage():
    objects = Objects()
    archive = CacheArchive(policy(), NSPACE, objects=objects)
    header, parts = manifest()
    entered, release = threading.Event(), threading.Event()

    def read(key):
        entered.set()
        assert release.wait(5)
        return parts[key]

    try:
        future = archive.submit("world", KEY, header, read)
        assert entered.wait(2)
        assert not future.done()
        assert archive.find("world", KEY) is None
        release.set()
        assert future.result(3)["state"] == "persisted"
        commits = [k.rsplit("/", 1)[-1] for k in objects.writes]
        assert (
            commits.index("manifest.json")
            < commits.index("complete")
            < commits.index("current.json")
        )
        restored = {}
        assert archive.restore("world", KEY, restored.__setitem__) == header
        assert restored == parts
    finally:
        release.set()
        archive.close()


@pytest.mark.parametrize(("size", "window"), [(16, 4), (96, 2)])
def test_restore_bounds_prefetched_members_and_imports_in_manifest_order(size, window):
    objects = Objects()
    archive = CacheArchive(policy(max_object_bytes=100), NSPACE, objects=objects)
    parts = {
        f"{NSPACE}:{KEY}:a{ART}:r0:kv:l{i:02}": bytes([i]) * size for i in range(9)
    }
    native = dict(artifact_id=ART, objects={k: len(v) for k, v in parts.items()})
    owners = list(parts)
    imported = []
    outstanding = set()
    first_window = threading.Barrier(window)
    lock = threading.Lock()
    try:
        archive.submit("world", KEY, native, parts.__getitem__).result(5)
        found = archive.find("world", KEY)
        member = {found["folder"] + row["file"]: row["key"] for row in found["objects"]}
        original_get = objects.get

        def get(path, limit):
            if path in member:
                name = member[path]
                assert limit == size
                with lock:
                    outstanding.add(name)
                    assert len(outstanding) <= window
                if name in owners[:window]:
                    first_window.wait(timeout=5)
            return original_get(path, limit)

        def put(key, payload):
            assert payload == parts[key]
            with lock:
                outstanding.remove(key)
            imported.append(key)

        objects.get = get
        assert archive.restore("world", KEY, put, found=found) == native
        assert imported == owners and not outstanding
    finally:
        archive.close()


@pytest.mark.parametrize("fail_in", ["get", "import"])
def test_failed_restore_does_not_import_past_failed_member(fail_in):
    objects = Objects()
    archive = CacheArchive(policy(), NSPACE, objects=objects)
    header, parts = manifest()
    try:
        archive.submit("world", KEY, header, parts.__getitem__).result(5)
        found = archive.find("world", KEY)
        first = found["objects"][0]
        if fail_in == "get":
            del objects.data[found["folder"] + first["file"]]
        attempted = []

        def put(key, data):
            attempted.append(key)
            raise OSError("injected import failure")

        with pytest.raises(ArchiveUnavailable if fail_in == "get" else OSError):
            archive.restore("world", KEY, put, found=found)
        assert attempted == ([] if fail_in == "get" else [first["key"]])
        restored = {}
        if fail_in == "get":
            objects.put(found["folder"] + first["file"], parts[first["key"]])
        assert archive.restore("world", KEY, restored.__setitem__) == header
        assert restored == parts
    finally:
        archive.close()


def test_partial_or_expired_archives_never_restore_as_complete():
    objects = Objects()
    now = [1000.0]
    archive = CacheArchive(policy(), NSPACE, objects=objects, clock=lambda: now[0])
    header, parts = manifest()
    try:
        objects.fail = "complete"
        with pytest.raises(OSError):
            archive.submit("world", KEY, header, parts.__getitem__).result(3)
        assert archive.find("world", KEY) is None
        objects.fail = None
        archive.submit("world", KEY, header, parts.__getitem__).result(3)
        value = archive.find("world", KEY)
        leaf = value["folder"] + value["objects"][0]["file"]
        saved = objects.data[leaf]
        objects.data[leaf] = saved[:-1]
        with pytest.raises(ArchiveUnavailable, match="truncated"):
            archive.restore("world", KEY, lambda *_: None)
        objects.data[leaf] = saved
        now[0] += 7 * 86400
        assert archive.find("world", KEY) is None
    finally:
        archive.close()


def test_world_delete_fences_remote_upload_and_preserves_other_worlds():
    objects = Objects()
    writer = CacheArchive(policy(), NSPACE, objects=objects)
    deleter = CacheArchive(policy(), NSPACE, objects=objects)
    header, parts = manifest()
    entered, release = threading.Event(), threading.Event()

    def read(key):
        entered.set()
        assert release.wait(5)
        return parts[key]

    try:
        writer.submit("other", KEY, header, parts.__getitem__).result(3)
        objects.put("white_box_game_record/video.mp4", b"keep video")
        old = writer.submit("world", KEY, header, read)
        assert entered.wait(2)
        result = deleter.delete_world("world")
        assert result["logical_deleted"] and result["state"] == "pending"
        release.set()
        with pytest.raises(ArchiveUnavailable, match="fenced"):
            old.result(3)
        assert deleter.delete_world("world")["state"] == "completed"
        assert writer.find("world", KEY) is None
        assert writer.find("other", KEY) is not None
        assert objects.data["white_box_game_record/video.mp4"] == b"keep video"
        # Deletion does not permanently ban new computation/upload.
        writer.submit("world", KEY, header, parts.__getitem__).result(3)
        assert writer.find("world", KEY) is not None
    finally:
        release.set()
        writer.close()
        deleter.close()


def test_archive_admission_is_bounded_and_duplicate_fill_is_joined():
    archive = CacheArchive(policy(max_pending=1), NSPACE, objects=Objects())
    header, parts = manifest()
    entered, release = threading.Event(), threading.Event()

    def read(key):
        entered.set()
        assert release.wait(5)
        return parts[key]

    try:
        first = archive.submit("world", KEY, header, read)
        assert entered.wait(2)
        assert archive.submit("world", KEY, header, read) is first
        other_artifact, other_parts = manifest("c" * 32)
        assert (
            archive.submit("world", KEY, other_artifact, other_parts.__getitem__)
            is first
        )
        assert archive.submit("other", KEY, header, read) is None
    finally:
        release.set()
        archive.close()


def replica(kind):
    return NS(
        status="ReplicaStatus.COMPLETE",
        is_memory_replica=lambda: kind == "l1",
        is_local_disk_replica=lambda: kind == "l2",
    )


def test_native_l2_promotion_checks_real_memory_residency():
    states = {"kv": [replica("l2")]}
    client = NS(
        batch_get_replica_desc=lambda keys: {k: states.get(k, []) for k in keys},
        get_size=lambda key: 3,
    )

    def get(key):
        states[key] = [replica("l1"), replica("l2")]
        return b"abc"

    client.get = get
    tiers = NativeReplicas(client, replace(CacheTiers(), l2=True))
    assert not tiers.memory_ready(["kv"])
    assert tiers.promote(["kv"], float("inf"))
    assert tiers.memory_ready(["kv"])
    states["kv"] = [replica("l2")]
    disabled = NativeReplicas(client, CacheTiers())
    assert disabled.layers(["kv"])["kv"] == set()


def test_oss_probe_budget_config():
    assert CacheTiers.parse({}).oss_probe_timeout_s == 1.0
    assert CacheTiers.parse({"oss_probe_timeout_s": 0.25}).oss_probe_timeout_s == 0.25
    for invalid in (True, "1", 0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="deadline"):
            CacheTiers.parse({"oss_probe_timeout_s": invalid})


def test_timed_out_probe_does_not_queue_or_continue_metadata_reads():
    from concurrent.futures import ThreadPoolExecutor

    from sglang.multimodal_gen.runtime.realtime.cache_archive import ArchiveProbeTimeout

    entered, release = threading.Event(), threading.Event()
    calls = []

    class SlowFence(Objects):
        def get(self, key, limit):
            calls.append(key)
            if len(calls) == 1:
                entered.set()
                release.wait(5)
            return super().get(key, limit)

    archive = CacheArchive(
        policy(max_pending=1, oss_probe_timeout_s=0.12), NSPACE, objects=SlowFence()
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(archive.probe, "world", KEY)
            assert entered.wait(2)
            second = callers.submit(archive.probe, "other", KEY)
            for future in (first, second):
                with pytest.raises(ArchiveProbeTimeout):
                    future.result(timeout=2)
            assert len(calls) == 1
            release.set()
        # The first timed-out fence must not lead to reading the old pointer.
        assert archive.probe("other", KEY) is None
        assert len(calls) == 3 and calls[1] != calls[0]
    finally:
        release.set()
        archive.close()


def test_oss_probe_uses_one_budget_across_all_metadata_reads(monkeypatch):
    from sglang.multimodal_gen.runtime.realtime import cache_archive

    objects = Objects()
    archive = CacheArchive(policy(), NSPACE, objects=objects)
    header, parts = manifest()
    try:
        archive.submit("world", KEY, header, parts.__getitem__).result(5)
        elapsed = [0.0]
        paths = []
        read = objects.get

        def get(key, limit):
            paths.append(key.rsplit("/", 1)[-1])
            elapsed[0] += 0.3
            return read(key, limit)

        monkeypatch.setattr(objects, "get", get)
        monkeypatch.setattr(cache_archive, "time", NS(monotonic=lambda: elapsed[0]))
        with pytest.raises(cache_archive.ArchiveProbeTimeout):
            archive.probe("world", KEY)
        assert paths == ["fence.json", "current.json", "complete", "manifest.json"]
        # A later request may still use the same complete artifact.
        monkeypatch.setattr(objects, "get", read)
        assert archive.probe("world", KEY)["artifact_id"] == ART
    finally:
        archive.close()


def test_world_delete_invalidates_remote_upload_queued_behind_another_world():
    objects = Objects()
    writer = CacheArchive(policy(max_pending=2), NSPACE, objects=objects)
    manager = CacheArchive(policy(), NSPACE, objects=objects)
    entered, release = threading.Event(), threading.Event()
    native, parts = manifest()

    def blocked(key):
        entered.set()
        assert release.wait(5)
        return parts[key]

    try:
        first = writer.submit("other", KEY, native, blocked)
        assert entered.wait(2)
        queued = writer.submit("world", KEY, native, parts.__getitem__)
        assert manager.delete_world("world")["state"] == "completed"
        release.set()
        first.result(3)
        with pytest.raises(ArchiveUnavailable, match="fenced"):
            queued.result(3)
        assert writer.find("world", KEY) is None
        assert not objects.keys(writer._world("world"))
        writer.submit("world", KEY, native, parts.__getitem__).result(3)
        assert writer.find("world", KEY)["artifact_id"] == ART
    finally:
        release.set()
        writer.close()
        manager.close()


def test_world_deletion_turn_excludes_other_managers_across_namespaces():
    entered, release = threading.Event(), threading.Event()

    class PausedSweep(Objects):
        sweeps = 0

        def delete_prefix(self, prefix):
            self.sweeps += 1
            if self.sweeps == 2:
                entered.set()
                assert release.wait(5)
            return super().delete_prefix(prefix)

    objects = PausedSweep()
    first = CacheArchive(policy(), NSPACE, objects=objects)
    second = CacheArchive(policy(), NSPACE + "-other", objects=objects)
    native, parts = manifest()
    try:
        with ThreadPoolExecutor(1) as executor:
            deletion = executor.submit(first.delete_world, "world")
            try:
                assert entered.wait(2)
                with pytest.raises(ArchiveUnavailable, match="already running"):
                    second.delete_world("world")
                assert first.submit("world", KEY, native, parts.__getitem__) is None
            finally:
                release.set()
            assert deletion.result(3)["state"] == "completed"
        first.submit("world", KEY, native, parts.__getitem__).result(3)
        assert first.find("world", KEY)["artifact_id"] == ART
        assert second.delete_world("world")["state"] == "completed"
        assert not objects.keys(first._world("world"))
    finally:
        release.set()
        first.close()
        second.close()


def test_outer_mooncake_queue_keeps_the_epoch_seen_before_oss_submission():
    objects = Objects()
    writer = CacheArchive(policy(), NSPACE, objects=objects)
    admin = CacheArchive(policy(), NSPACE, objects=objects)
    native, parts = manifest()
    try:
        ticket = writer.submission_fence("world")
        admin.delete_world("world")
        with pytest.raises(ArchiveUnavailable, match="fenced"):
            writer.submit(
                "world", KEY, native, parts.__getitem__, expected_fence=ticket
            )
        assert writer.find("world", KEY) is None
        fresh = writer.submission_fence("world")
        writer.submit(
            "world", KEY, native, parts.__getitem__, expected_fence=fresh
        ).result(3)
        assert writer.find("world", KEY)["artifact_id"] == ART
    finally:
        writer.close()
        admin.close()
