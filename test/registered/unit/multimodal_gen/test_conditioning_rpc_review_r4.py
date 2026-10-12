# SPDX-License-Identifier: Apache-2.0
"""R4 deadline boundary after real CPU cache lookup and RPC eviction."""

if __name__ == "__main__":
    import runpy
    import sys
    from pathlib import Path

    test_file = Path(__file__).resolve()
    runner = (
        test_file.parents[4] / "benchmark/zing_h3_causal/run_conditioning_cpu_tests.py"
    )
    arguments = ["-x" if value == "-f" else value for value in sys.argv[1:]]
    sys.argv = [str(runner), str(test_file), *arguments]
    runpy.run_path(str(runner), run_name="__main__")

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from sglang.multimodal_gen.runtime.conditioning import service as service_module
from sglang.multimodal_gen.runtime.conditioning.cache import tensor_bytes
from sglang.test.ci.ci_register import register_cpu_ci
from test_conditioning_rpc_core import (
    Encoder,
    assert_idle,
    assert_ok,
    components,
    encode,
    limits,
    reference_digest,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


async def wait_until(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < deadline, "test boundary was not reached"
        await asyncio.sleep(0.001)


def single_entry_components():
    capacity = tensor_bytes(Encoder()(SimpleNamespace(prompt="p", media=[])))
    return components(
        configured_limits=limits(cache_bytes=capacity, cache_bytes_per_tenant=capacity)
    )


@pytest.mark.parametrize("expired", [True, False])
def test_lookup_evicted_by_real_rpc_only_encodes_with_live_deadline(
    monkeypatch, expired
):
    async def run():
        profile, app, client, encoder = single_entry_components()
        entered, release_get = threading.Event(), threading.Event()
        key = ("tenant", reference_digest("p", [], profile.profile_id))
        original_get = app.cache.get
        observations = []
        target = None

        def controlled_get(lookup_key):
            if lookup_key == key:
                entered.set()
                assert release_get.wait(2), "test cache lookup was not released"
            result = original_get(lookup_key)
            if lookup_key == key:
                observations.append(result is None)
            return result

        try:
            assert_ok(await encode(client, profile, prompt="p"))
            assert app.cache.entry_size(key) == app.cache.max_bytes
            monkeypatch.setattr(app.cache, "get", controlled_get)
            target = asyncio.create_task(
                encode(client, profile, prompt="p", deadline_ms=60 if expired else 1000)
            )
            await wait_until(entered.is_set)
            assert_ok(await encode(client, profile, prompt="q"))
            assert app.cache.entry_size(key) is None, "a real RPC must evict p"
            assert encoder.calls == ["p", "q"]
            encoder.started.clear()
            if expired:
                await asyncio.sleep(0.12)
                encoder.release.clear()
            release_get.set()
            response = await asyncio.wait_for(target, 1)
            await asyncio.sleep(0.02)
            assert observations == [True], "lookup must return the real cache miss"
            if expired:
                assert response.status_code == 504
                assert response.json()["code"] == "CONDITION_TIMEOUT"
                assert encoder.calls == ["p", "q"], "expired miss must not encode"
                assert not encoder.started.is_set(), "native must not start"
                assert app.cache.entry_size(key) is None
            else:
                assert assert_ok(response)[0]["cache"] == "miss"
                assert encoder.calls == ["p", "q", "p"]
                assert encoder.started.is_set()
                assert app.cache.entry_size(key) == app.cache.max_bytes
                assert (
                    assert_ok(await encode(client, profile, prompt="p"))[0]["cache"]
                    == "hit"
                )
                assert encoder.calls == ["p", "q", "p"]
            assert app.flights == {}
            assert app.queue.depth == 0
            assert_idle(app)
        finally:
            release_get.set()
            encoder.release.set()
            if target is not None:
                await asyncio.gather(target, return_exceptions=True)
            await client.aclose()
            await app.close()
            assert_idle(app)

    asyncio.run(run())


@pytest.mark.parametrize("expired", [True, False])
def test_delayed_real_cache_hit_checks_deadline_before_success_pack(
    monkeypatch, expired
):
    async def run():
        profile, app, client, encoder = single_entry_components()
        entered, release_get = threading.Event(), threading.Event()
        original_get, original_pack = app.cache.get, service_module.pack_message
        lookups, packs = [], []
        target = None

        def controlled_get(key):
            entered.set()
            assert release_get.wait(2), "test cache lookup was not released"
            result = original_get(key)
            lookups.append(result is not None)
            return result

        def observed_pack(*args):
            packs.append(args[0]["cache"])
            return original_pack(*args)

        try:
            assert_ok(await encode(client, profile, prompt="p"))
            monkeypatch.setattr(app.cache, "get", controlled_get)
            monkeypatch.setattr(service_module, "pack_message", observed_pack)
            encoder.started.clear()
            target = asyncio.create_task(
                encode(client, profile, prompt="p", deadline_ms=60 if expired else 1000)
            )
            await wait_until(entered.is_set)
            await asyncio.sleep(0.12 if expired else 0.01)
            release_get.set()
            response = await asyncio.wait_for(target, 1)
            assert lookups == [True], "the real cached result must still exist"
            assert encoder.calls == ["p"]
            assert not encoder.started.is_set()
            if expired:
                assert response.status_code == 504
                assert response.json()["code"] == "CONDITION_TIMEOUT"
                assert packs == [], "expired cache hit must not serialize success"
                assert app._stats["tenant"]["hit"] == 0
            else:
                assert assert_ok(response)[0]["cache"] == "hit"
                assert packs == ["hit"]
                assert app._stats["tenant"]["hit"] == 1
            assert app.flights == {}
            assert app.queue.depth == 0
            assert_idle(app)
        finally:
            release_get.set()
            encoder.release.set()
            if target is not None:
                await asyncio.gather(target, return_exceptions=True)
            await client.aclose()
            await app.close()
            assert_idle(app)

    asyncio.run(run())


@pytest.mark.parametrize("expired", [True, False])
def test_delayed_lookup_joins_existing_flight_only_with_live_deadline(
    monkeypatch, expired
):
    async def run():
        profile, app, client, encoder = single_entry_components()
        entered, release_get = threading.Event(), threading.Event()
        key = ("tenant", reference_digest("p", [], profile.profile_id))
        original_get, original_leave = app.cache.get, app.queue.leave
        lookups, leaves = [], []
        target = live = None

        def controlled_get(lookup_key):
            if lookup_key == key:
                entered.set()
                assert release_get.wait(2), "test cache lookup was not released"
            result = original_get(lookup_key)
            if lookup_key == key:
                lookups.append(result is None)
            return result

        def observed_leave(flight):
            leaves.append(flight.waiters)
            return original_leave(flight)

        try:
            assert_ok(await encode(client, profile, prompt="p"))
            monkeypatch.setattr(app.cache, "get", controlled_get)
            target = asyncio.create_task(
                encode(client, profile, prompt="p", deadline_ms=60 if expired else 1000)
            )
            await wait_until(entered.is_set)
            assert_ok(await encode(client, profile, prompt="q"))
            assert app.cache.entry_size(key) is None
            encoder.started.clear()
            encoder.release.clear()
            live = asyncio.create_task(
                encode(client, profile, prompt="p", deadline_ms=1000)
            )
            await wait_until(encoder.started.is_set)
            flight = app.flights[key]
            assert flight.state == "running" and flight.waiters == 1
            assert encoder.calls == ["p", "q", "p"]
            monkeypatch.setattr(app.queue, "leave", observed_leave)
            if expired:
                await asyncio.sleep(0.12)
            release_get.set()
            if not expired:
                await wait_until(lambda: app._stats["tenant"]["joined"] == 1)
                assert flight.waiters == 2
                encoder.release.set()
            response = await asyncio.wait_for(target, 1)
            assert lookups == [True]
            assert encoder.calls == ["p", "q", "p"]
            if expired:
                assert response.status_code == 504
                assert response.json()["code"] == "CONDITION_TIMEOUT"
                assert app._stats["tenant"]["joined"] == 0
                assert leaves == [], "expired lookup must not add/remove a waiter"
                assert flight.state == "running" and flight.waiters == 1
                assert not live.done()
                assert app.ledger.snapshot()["global"]["inflight"] > 0
                encoder.release.set()
            else:
                assert assert_ok(response)[0]["cache"] == "joined"
            assert assert_ok(await asyncio.wait_for(live, 1))[0]["cache"] == "miss"
            assert app.flights == {}
            assert app.queue.depth == 0
            assert_idle(app)
        finally:
            release_get.set()
            encoder.release.set()
            await asyncio.gather(
                *(task for task in (target, live) if task is not None),
                return_exceptions=True,
            )
            await client.aclose()
            await app.close()
            assert_idle(app)

    asyncio.run(run())
