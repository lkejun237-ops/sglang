"""Fixed RTC endpoints keep independently admitted sources isolated."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.realtime.whitebox import native_rtc as rtc

ENDPOINTS = [
    {"listen": "127.0.0.1:41081", "public_port": 41081},
    {"listen": "127.0.0.1:41082", "public_port": 41082},
]


@pytest.mark.parametrize(
    "rows",
    [
        [],
        {},
        "invalid",
        [{"listen": "host:80", "public_port": 443}],
        [{"listen": "0.0.0.0:0", "public_port": 443}],
        [{"listen": "0.0.0.0:40000", "public_port": True}],
        [{"listen": "0.0.0.0:40000", "public_port": 65536}],
        [ENDPOINTS[0], ENDPOINTS[0]],
        [ENDPOINTS[0], {"listen": "127.0.0.1:41083", "public_port": 41081}],
    ],
)
def test_invalid_endpoint_pools_fail_at_startup(monkeypatch, rows):
    monkeypatch.setenv("WHITEBOX_NATIVE_RTC_ENDPOINTS", json.dumps(rows))
    with pytest.raises(ValueError, match="endpoint configuration"):
        rtc.validate_rtc_source_capacity(2)


def test_capacity_requires_enough_distinct_endpoints_and_preserves_default(monkeypatch):
    monkeypatch.delenv("WHITEBOX_NATIVE_RTC_ENDPOINTS", raising=False)
    rtc.validate_rtc_source_capacity(1)
    with pytest.raises(ValueError):
        rtc.validate_rtc_source_capacity(2)
    monkeypatch.setenv("WHITEBOX_NATIVE_RTC_ENDPOINTS", json.dumps(ENDPOINTS))
    rtc.validate_rtc_source_capacity(2)
    with pytest.raises(ValueError):
        rtc.validate_rtc_source_capacity(3)


def test_ports_stay_reserved_until_child_exit_even_when_close_caller_is_cancelled(
    monkeypatch,
):
    async def run():
        exited = asyncio.Event()
        launched = []

        async def spawn(*args, **kwargs):
            launched.append(args)

            async def wait():
                await exited.wait()
                return 0

            return SimpleNamespace(stdin=SimpleNamespace(close=lambda: None), wait=wait)

        async def ready(peer):
            peer.ready.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        monkeypatch.setattr(rtc.NativeRTC, "_read", ready)
        monkeypatch.setattr(rtc.NativeRTC, "_stderr", ready)
        peers = [rtc.NativeRTC(lambda _: None) for _ in range(4)]
        try:
            await asyncio.gather(peers[0].start(), peers[1].start())
            assert peers[0].listen != peers[1].listen
            with pytest.raises(RuntimeError, match="available"):
                await peers[2].start()
            closing = asyncio.create_task(peers[0].close())
            await asyncio.sleep(0)
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            with pytest.raises(RuntimeError, match="available"):
                await peers[3].start()
            assert not peers[1].closed
            exited.set()
            await peers[0].close()
            replacement = rtc.NativeRTC(lambda _: None)
            peers.append(replacement)
            await replacement.start()
            assert replacement.listen == peers[0].listen
            assert len(launched) == 3
        finally:
            exited.set()
            await asyncio.gather(*(peer.close() for peer in peers))
        assert not rtc._endpoint_leases

    monkeypatch.setenv("WHITEBOX_NATIVE_RTC_ENDPOINTS", json.dumps(ENDPOINTS))
    asyncio.run(run())


def test_spawn_failure_releases_only_its_own_endpoint(monkeypatch):
    async def fail(*args, **kwargs):
        raise FileNotFoundError("test binary missing")

    async def run():
        peer = rtc.NativeRTC(lambda _: None)
        with pytest.raises(FileNotFoundError):
            await peer.start()
        assert peer.closed and not rtc._endpoint_leases

    monkeypatch.setenv("WHITEBOX_NATIVE_RTC_ENDPOINTS", json.dumps(ENDPOINTS))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail)
    asyncio.run(run())


def test_real_native_processes_bind_separate_udp_ports_and_reuse_after_exit(
    monkeypatch,
):
    binary = os.environ.get("WHITEBOX_NATIVE_RTC_BINARY", "/app/whitebox-rtc")
    if not Path(binary).is_file():
        pytest.skip("Requires the native RTC binary from the deployment image")

    async def run():
        peers = [rtc.NativeRTC(lambda _: None, public_ip="127.0.0.1") for _ in range(3)]
        try:
            await asyncio.gather(peers[0].start(), peers[1].start())
            assert len({peer.process.pid for peer in peers[:2]}) == 2
            assert peers[0].public_port != peers[1].public_port
            await peers[0].close()
            assert peers[0].process.returncode is not None
            assert peers[1].process.returncode is None
            await peers[2].start()
            assert peers[2].listen == peers[0].listen
        finally:
            await asyncio.gather(*(peer.close() for peer in peers))
        assert not rtc._endpoint_leases

    monkeypatch.setenv("WHITEBOX_NATIVE_RTC_ENDPOINTS", json.dumps(ENDPOINTS))
    asyncio.run(run())


def test_dual_source_archive_budget_preserves_defaults_and_bounds(monkeypatch):
    from sglang.multimodal_gen.runtime.realtime.whitebox.archive_rgb import (
        RGBBudget,
        RGBSlots,
    )
    from sglang.multimodal_gen.runtime.realtime.whitebox.session import from_environment

    monkeypatch.setenv("WHITEBOX_RENDER_SEGMENTS", "1")
    monkeypatch.delenv("WHITEBOX_RECORDER_SLOTS", raising=False)
    _, legacy = from_environment(room=True)
    assert legacy.recorder_slots == 3
    monkeypatch.setenv("WHITEBOX_RECORDER_SLOTS", "6")
    _, dual = from_environment(room=True)
    assert dual.recorder_slots == 6
    assert len(dual.recorder_cpu_slots) == 6
    # Two sources plus two old/new model pairs fit the unchanged RGB byte budget.
    budget = RGBBudget()
    pools = []
    try:
        for _ in range(10):
            pools.append(RGBSlots(832 * 480 * 3, budget))
        assert budget.used <= 96 << 20
    finally:
        for pool in pools:
            pool.close()
    assert budget.used == 0
    monkeypatch.setenv("WHITEBOX_RECORDER_SLOTS", "10")
    with pytest.raises(ValueError, match="recorder process budget"):
        from_environment(room=True)


def test_three_players_keep_source_and_overlapping_segment_archive_capacity(
    monkeypatch,
):
    from sglang.multimodal_gen.runtime.realtime.whitebox.archive_rgb import RGBSlots
    from sglang.multimodal_gen.runtime.realtime.whitebox.session import from_environment

    monkeypatch.setenv("WHITEBOX_RENDER_SEGMENTS", "1")
    monkeypatch.setenv("WHITEBOX_RECORDER_SLOTS", "9")
    monkeypatch.setenv("WHITEBOX_STREAM_ARCHIVE_RGB_BYTES", str(192 << 20))
    _, archives = from_environment(room=True)
    assert archives.recorder_slots == 9
    assert len(archives.recorder_cpu_slots) == 9
    # Each player holds one source lane and two paired-source/model segments
    # while the previous segment finishes. They share one bounded Room budget.
    pools = []
    try:
        for _ in range(3 * (1 + 2 * 2)):
            pools.append(RGBSlots(832 * 480 * 3, archives.rgb_budget))
        assert archives.rgb_budget.used > 96 << 20
        assert archives.rgb_budget.used <= 192 << 20
        # Exhaustion remains an explicit rejection, never an unbounded pool.
        while True:
            try:
                pools.append(RGBSlots(832 * 480 * 3, archives.rgb_budget))
            except BufferError:
                break
        assert archives.rgb_budget.used <= 192 << 20
    finally:
        for pool in pools:
            pool.close()
    assert archives.rgb_budget.used == 0


@pytest.mark.parametrize("budget", [0, (32 << 20) - 1, (512 << 20) + 1])
def test_stream_archive_rgb_budget_rejects_invalid_configuration(monkeypatch, budget):
    from sglang.multimodal_gen.runtime.realtime.whitebox.session import from_environment

    monkeypatch.setenv("WHITEBOX_STREAM_ARCHIVE_RGB_BYTES", str(budget))
    with pytest.raises(ValueError, match="streaming archive RGB budget"):
        from_environment(room=True)
