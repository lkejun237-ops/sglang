import asyncio
from collections import deque
from types import SimpleNamespace

import pytest
from docker_idle_probe import archive_snapshot

PREFIX = "sglang.multimodal_gen.runtime.realtime.whitebox."


def fake(kind, module, **values):
    cls = type(kind, (SimpleNamespace,), {"__module__": PREFIX + module})
    return cls(**values)


def fixture():
    loop = asyncio.new_event_loop()
    done = loop.create_future()
    done.set_result(None)
    manager = fake(
        "StreamingArchiveManager",
        "stream_archive",
        active=set(),
        tasks=set(),
        status_pending=set(),
        io_active=0,
        upload_pending=0,
        rgb_budget=SimpleNamespace(used=0),
        recovery=None,
        upload_pool=None,
        live_recordings={},
    )
    runtime = fake(
        "RoomRuntime",
        "room_runtime",
        owner_id="owner",
        archives=manager,
        **{
            k: {}
            for k in (
                "rooms",
                "players",
                "pending_startups",
                "join_attempts",
                "retiring",
                "retiring_players",
                "player_closures",
            )
        },
    )
    recording = fake(
        "StreamingRecording",
        "stream_archive",
        closed=True,
        task=done,
        pending=None,
        items=deque(),
        rgb_slots={},
        ui_tasks={},
        pending_bytes=0,
        upload_buffer_bytes=0,
        object_buffer_bytes=0,
        ui_bytes=0,
        process=SimpleNamespace(_popen=SimpleNamespace(returncode=0)),
    )
    manager.live_recordings["old"] = recording
    return loop, manager, runtime, recording


def test_snapshot_allows_closed_history_without_mutation():
    loop, manager, runtime, recording = fixture()
    try:
        runtime.member_retirements = {"old": {"state": "retired"}}
        result = archive_snapshot([manager, runtime, recording], "owner")
        assert result["retained_recordings"] == 1
        assert set(result["counters"].values()) == {0}
        assert recording.closed is True and manager.live_recordings["old"] is recording
    finally:
        loop.close()


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "unfinished",
        "admission",
        "epoch",
        "publication",
        "receipt",
        "retirement",
        "room",
    ],
)
def test_failed_join_history_requires_original_owner_cleanup(mutation):
    loop, manager, runtime, recording = fixture()
    try:
        runtime.epoch = "epoch"
        attempt = {"sdk_started": True, "failed": True}
        runtime.join_attempts["stream"] = attempt
        admission = dict(
            stream_id="stream",
            status="retired",
            binding=dict(
                gameId="game", ownerEpoch="epoch", publicationId="publication"
            ),
        )
        receipt = dict(
            game_id="game",
            room_cleanup_complete=True,
            retire_status="succeeded",
            invocation={
                "createContext": {
                    "frozenPublication": {"world": {"publicationId": "publication"}}
                }
            },
        )
        runtime.control = SimpleNamespace(
            admissions={"a": admission}, creates={"c": receipt}
        )
        if mutation == "unfinished":
            attempt["failed"] = False
        elif mutation == "admission":
            admission["status"] = "blocked"
        elif mutation == "epoch":
            admission["binding"]["ownerEpoch"] = "previous-owner"
        elif mutation == "publication":
            admission["binding"]["publicationId"] = "other-publication"
        elif mutation == "receipt":
            receipt["room_cleanup_complete"] = False
        elif mutation == "retirement":
            receipt["retire_status"] = "unknown"
        elif mutation == "room":
            receipt["game_id"] = "other-room"
        result = archive_snapshot([manager, runtime, recording], "owner")
        assert result["counters"]["runtime_join_attempts"] == int(mutation is not None)
        assert runtime.join_attempts["stream"] is attempt
        assert len(runtime.join_attempts) == 1
        # An actual remaining owner still blocks cutover even with a receipt.
        runtime.rooms["other-room"] = {}
        assert (
            archive_snapshot([manager, runtime, recording], "owner")["counters"][
                "runtime_rooms"
            ]
            == 1
        )
    finally:
        loop.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("active", {"game-log"}),
        ("tasks", None),
        ("io_active", 1),
        ("status_pending", {"metadata"}),
        ("rgb_budget", SimpleNamespace(used=1)),
    ],
)
def test_snapshot_observes_tail_work_missing_from_report(field, value):
    loop, manager, runtime, recording = fixture()
    try:
        if field == "tasks":
            value = {loop.create_future()}
        setattr(manager, field, value)
        result = archive_snapshot([manager, runtime, recording], "owner")
        assert any(value != 0 for value in result["counters"].values())
    finally:
        loop.close()


@pytest.mark.parametrize(
    "mutation",
    ["missing", "wrong_type", "second_manager", "owner", "alive", "recording_buffer"],
)
def test_snapshot_fails_closed_or_exposes_nonidle(mutation):
    loop, manager, runtime, recording = fixture()
    objects = [manager, runtime, recording]
    try:
        if mutation == "missing":
            del manager.io_active
        if mutation == "wrong_type":
            manager.io_active = False
        if mutation == "second_manager":
            objects.append(fake("StreamingArchiveManager", "stream_archive"))
        if mutation == "owner":
            runtime.owner_id = "other"
        if mutation == "alive":
            recording.process._popen.returncode = None
        if mutation == "recording_buffer":
            recording.object_buffer_bytes = 1
        if mutation in ["alive", "recording_buffer"]:
            assert any(
                v != 0 for v in archive_snapshot(objects, "owner")["counters"].values()
            )
        else:
            with pytest.raises((ValueError, KeyError)):
                archive_snapshot(objects, "owner")
    finally:
        loop.close()
