"""Signed archive scope and nonblocking, post-player cache receipts."""

import json
import threading
import time
from types import SimpleNamespace as NS

from sglang.multimodal_gen.runtime.realtime.cache_observer import (
    CacheObserver,
    issue_recording,
    verify_recording,
)


def configured(monkeypatch):
    monkeypatch.setenv("WHITEBOX_INTERNAL_TOKEN", "test-internal-" + "x" * 40)
    monkeypatch.setenv("WHITEBOX_CACHE_RECORDING_ENABLED", "1")


def test_recording_capability_cannot_change_world_or_escape_its_archive(monkeypatch):
    configured(monkeypatch)
    token = issue_recording("world-a", "white_box_game_record/test/player-a/")
    target = verify_recording(token, "world-a")
    assert target["prefix"] == "white_box_game_record/test/player-a/cache-operations/"
    assert verify_recording(token, "world-b") is None
    assert (
        verify_recording(token[:-1] + ("0" if token[-1] != "0" else "1"), "world-a")
        is None
    )
    assert verify_recording(token, "world-a", now=target["expires_at"]) is None
    assert issue_recording("world-a", "white_box_game_record/../other/") is None
    assert issue_recording("world-a", "models/") is None


def test_late_recording_owns_immutable_scope_and_does_not_wait_for_storage(monkeypatch):
    configured(monkeypatch)
    target = verify_recording(
        issue_recording("world-a", "white_box_game_record/test/player-a/"), "world-a"
    )
    entered, release = threading.Event(), threading.Event()
    written = {}

    class Storage:
        def put(self, key, data):
            entered.set()
            assert release.wait(2)
            written[key] = json.loads(data)

    observer = CacheObserver(NS(), objects=Storage())
    try:
        started = time.monotonic()
        assert observer.offer(
            target,
            {
                "event": "l3_writeback",
                "state": "persisted",
                "session_id": "original-player",
            },
        )
        assert time.monotonic() - started < 0.1
        assert entered.wait(1)
        # Player teardown / a subsequent player must not retarget the queued IO.
        target["prefix"] = "white_box_game_record/test/player-b/cache-operations/"
        target["world_id"] = "world-b"
        release.set()
        observer.close()
        assert observer.written == 1 and observer.failed == 0
        key, row = next(iter(written.items()))
        assert key.startswith("white_box_game_record/test/player-a/cache-operations/")
        assert row["world_id"] == "world-a" and row["session_id"] == "original-player"
    finally:
        release.set()
        observer.close()
