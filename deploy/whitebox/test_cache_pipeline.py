"""Exercise real tensor serialization through the composed L1/L3 provider."""

import threading
from dataclasses import asdict

from sglang.multimodal_gen.runtime.realtime import cache_archive
from sglang.multimodal_gen.test.unit.realtime.test_zing_cache_tiers import (
    Objects,
    policy,
)
from sglang.multimodal_gen.test.unit.realtime.test_zing_mooncake_cache import (
    KEY,
    MemoryStore,
    assert_restored,
    provider,
    write,
)


def test_l3_recall_reuses_prepared_bytes_and_hot_hit_does_not_repair(monkeypatch):
    import time

    class SlowFirstPayload(Objects):
        delayed = False

        def get(self, key, limit):
            if not self.delayed and len(key.rsplit("/", 1)[-1]) == 64:
                self.delayed = True
                time.sleep(0.25)
            return super().get(key, limit)

    objects = SlowFirstPayload()
    monkeypatch.setattr(cache_archive, "S3CacheObjects", lambda _: objects)
    calls = []

    class Store(MemoryStore):
        def get(self, key):
            calls.append((key, threading.current_thread().name))
            return super().get(key)

    backend, cache, store = provider(
        Store(),
        config={
            "tiers": asdict(policy(snapshot_bytes=1 << 20, oss_probe_timeout_s=0.15))
        },
    )
    try:
        backend.bind_world(KEY, "world", s0_sha256="1" * 64)
        write(backend, cache)
        backend._cloud_jobs[KEY].result(timeout=5)
        published = dict(objects.data)
        # Simulate loss of this host's RAM; the immutable cloud artifact survives.
        store.objects.clear()
        calls.clear()
        backend.bind_world(KEY, "world", s0_sha256="2" * 64)
        entry = backend.lookup(KEY, allow_disk=True)
        assert entry is not None and objects.delayed
        assert backend.last_lookup["cache_result"] == "l3_hit"
        assert backend.last_lookup["baseline_s0_sha256"] == "1" * 64
        assert backend.last_lookup["online_s0_sha256"] == "2" * 64
        assert "l3_download" in backend.last_lookup["stages"]
        conditions = [(key, thread) for key, thread in calls if key.endswith(":cond")]
        assert len(conditions) == 1 and conditions[0][1].startswith("block0-recall")
        backend.restore(cache, entry)
        assert_restored(cache)

        def forbidden(*args):
            raise AssertionError("A hot hit must not probe or repair OSS")

        objects.get = forbidden
        assert backend.lookup(KEY) is not None
        assert backend.last_lookup["cache_result"] == "l1_hit"
        assert objects.data == published
    finally:
        backend.close()


def test_legacy_local_requests_remain_pixel_scoped_with_world_metadata():
    from types import SimpleNamespace as NS

    import torch

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.preparation_cache import (
        LocalPagedBackend,
        preparation_key,
    )

    local = object.__new__(LocalPagedBackend)
    local.identity = {"model": "same-model"}
    codecs = NS()
    request = {"world_id": "world", "prompt": "same prompt", "media": []}
    first = {"frames": torch.zeros(3, 1, 2, 2, dtype=torch.uint8)}
    second = {"frames": torch.ones(3, 1, 2, 2, dtype=torch.uint8)}
    config = {"width": 2, "height": 2}
    assert preparation_key(
        codecs, request, first, config, local.key_identity
    ) != preparation_key(codecs, request, second, config, local.key_identity)
    assert preparation_key(
        codecs, request, first, config, local.identity
    ) == preparation_key(codecs, request, second, config, local.identity)
    del request["world_id"]
    assert preparation_key(
        codecs, request, first, config, local.key_identity
    ) == preparation_key(codecs, request, first, config, local.identity)
    local.bind_world("key", "world")


def test_optional_snapshot_owns_cpu_reference_pixels_until_write_finishes():
    import torch

    from sglang.multimodal_gen.test.unit.realtime.test_zing_preparation_cache import (
        inputs,
    )

    store = MemoryStore()
    entered, release = threading.Event(), threading.Event()

    def delay():
        entered.set()
        assert release.wait(3)

    store.before_write = delay
    backend, cache, _ = provider(store)
    source = inputs()
    original = source.condition_media[0]["frames"].clone()
    try:
        assert backend.fill(KEY, source, "condition", None, cache, required=False)
        assert entered.wait(2)
        source.condition_media[0]["frames"].zero_()
        source.positive["context"].zero_()
        release.set()
        entry = backend.finish_fill(wait=True)
        assert entry is not None
        assert torch.equal(entry.inputs.condition_media[0]["frames"], original)
    finally:
        release.set()
        backend.close()


def test_condition_payload_larger_than_sdk_scratch_uses_registered_buffers():
    store = MemoryStore()
    backend, cache, _ = provider(
        store,
        config={"metadata_buffer_bytes": 1024, "tiers": {"snapshot_bytes": 1 << 20}},
    )
    reads = []
    original = store.batch_get_into_multi_buffers

    def record(keys, pointers, sizes):
        reads.extend(keys)
        return original(keys, pointers, sizes)

    store.batch_get_into_multi_buffers = record
    try:
        write(backend, cache)
        entry = backend.lookup(KEY)
        assert entry is not None
        assert backend._cond_key(KEY) in reads
        assert set(store.registrations) == set(backend._registered)
        backend.restore(cache, entry)
        assert_restored(cache)
    finally:
        backend.close()


def test_prefill_can_recompute_after_one_second_even_if_old_oss_hit_arrives(
    monkeypatch,
):
    import time

    entered, release = threading.Event(), threading.Event()

    class DelayedManifest(Objects):
        def get(self, key, limit):
            if key.endswith("/manifest.json"):
                entered.set()
                assert release.wait(5)
            return super().get(key, limit)

    objects = DelayedManifest()
    monkeypatch.setattr(cache_archive, "S3CacheObjects", lambda _: objects)
    backend, cache, store = provider(
        MemoryStore(), config={"tiers": asdict(policy(snapshot_bytes=1 << 20))}
    )
    try:
        backend.bind_world(KEY, "world")
        old = write(backend, cache)
        backend._cloud_jobs[KEY].result(timeout=5)
        store.objects.clear()
        start = time.monotonic()
        assert backend.lookup(KEY, allow_disk=True) is None
        assert entered.is_set() and 0.85 <= time.monotonic() - start < 2.5
        assert backend.last_lookup["miss_reason"] == "l3_probe_timeout"
        assert backend.lookup(KEY, resident_only=True) is None
        new = write(backend, cache)
        assert new.data.artifact_id != old.data.artifact_id
        release.set()
        backend._cloud_jobs[KEY].result(timeout=5)
        assert backend.lookup(KEY).data.artifact_id == new.data.artifact_id
        assert all(name not in store.objects for name in old.data.keys)
    finally:
        release.set()
        backend.close()


def test_oss_miss_does_not_wait_behind_another_keys_payload(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    objects = Objects()
    monkeypatch.setattr(cache_archive, "S3CacheObjects", lambda _: objects)
    started, release = threading.Event(), threading.Event()
    backend, _, _ = provider(config={"tiers": asdict(policy(oss_probe_timeout_s=0.12))})
    try:

        def hold_recall_lane():
            started.set()
            release.wait(5)

        occupied = backend._recall_io.submit(hold_recall_lane)
        assert started.wait(2)
        backend.bind_world(KEY, "world")
        with ThreadPoolExecutor(max_workers=1) as callers:
            assert (
                callers.submit(backend.lookup, KEY, allow_disk=True).result(2) is None
            )
        assert backend.last_lookup["miss_reason"] == "l3_probe_timeout"
        assert KEY not in backend._loads
        release.set()
        occupied.result(2)
    finally:
        release.set()
        backend.close()


def test_encoder_does_not_retry_an_unconfirmed_prefill_lookup():
    from types import SimpleNamespace as NS

    from sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server import (
        SourceEncoderExecutor,
    )
    from sglang.multimodal_gen.test.unit.realtime.test_zing_preparation_cache import (
        inputs,
    )

    first = dict(
        backend="mooncake", cache_result="miss", miss_reason="l3_probe_timeout"
    )
    conditioning = NS(
        cache_key=KEY,
        cache_hit=False,
        inputs=inputs(),
        cache_request={},
        cache_lookup={"lookup": first, "rank_lookups": [first]},
        conditioner=NS(identity_state=NS(identity="condition")),
        forward=lambda batch: None,
        dispose=lambda: None,
    )

    def control(operation, **arguments):
        if operation == "identity":
            return {"identity": "test"}
        assert operation == "prefill" and arguments["resident_only"] is True
        assert arguments["force"] is False and arguments["inputs"] is not None
        return {
            "hit": False,
            "timings": {"miss_reason": "not_resident"},
            "rank_timings": [{}],
        }

    fake = NS(
        owner=None,
        dit_link=object(),
        _cache_query=control,
        conditioning=conditioning,
        codecs=NS(spatial_ratio=8),
    )
    batch = NS(
        extra={
            "zing_source_command": {"config": {"height": 32, "width": 32}},
            "zing_prefill": {"source_frame": b""},
        }
    )
    receipt = SourceEncoderExecutor._prefill(fake, batch).usage["zing_prefill"]
    assert receipt["timings"]["lookup"] == first
    assert receipt["timings"]["miss_reason"] == "l3_probe_timeout"
    assert receipt["rank_timings"][0]["miss_reason"] == "l3_probe_timeout"
