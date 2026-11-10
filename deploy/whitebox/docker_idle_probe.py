"""Read-only CPython 3.14 archive snapshot for the immutable Docker operator.

Exact source is copied into an exclusive private directory. Remote execution only
reads the running interpreter's owner objects and writes numeric evidence.
"""


def archive_snapshot(objects, owner_id):
    from collections import deque

    def fields(obj):
        return vars(obj)

    def length(data, key, kinds):
        value = data[key]
        if type(value) not in kinds:
            raise ValueError("Unknown archive collection")
        return len(value)

    def integer(data, key):
        value = data[key]
        if type(value) is not int or value < 0:
            raise ValueError("Unknown archive counter")
        return value

    def unfinished(task):
        if type(task).__module__ != "_asyncio" or type(task).__name__ not in {
            "Task",
            "Future",
        }:
            raise ValueError("Unknown archive task")
        state = task._state
        if state not in {"PENDING", "FINISHED", "CANCELLED"}:
            raise ValueError("Unknown archive task state")
        return int(state == "PENDING")

    def named(obj, module, name):
        return type(obj).__module__ == module and type(obj).__name__ == name

    module = "sglang.multimodal_gen.runtime.realtime.whitebox."
    managers = [
        o
        for o in objects
        if named(o, module + "stream_archive", "StreamingArchiveManager")
    ]
    runtimes = [o for o in objects if named(o, module + "room_runtime", "RoomRuntime")]
    if len(managers) != 1 or len(runtimes) != 1:
        raise ValueError("Archive owner is not unique")
    manager, runtime = fields(managers[0]), fields(runtimes[0])
    if runtime["owner_id"] != owner_id or runtime["archives"] is not managers[0]:
        raise ValueError("Archive owner mismatch")
    counters = {}
    for key in ("active", "tasks", "status_pending"):
        counters[key] = length(manager, key, (set,))
    for key in ("io_active", "upload_pending"):
        counters[key] = integer(manager, key)
    counters["rgb_used"] = integer(fields(manager["rgb_budget"]), "used")
    counters["unfinished_manager_tasks"] = sum(unfinished(t) for t in manager["tasks"])
    counters["recovery"] = (
        0 if manager["recovery"] is None else unfinished(manager["recovery"])
    )
    counters["upload_pool"] = int(manager["upload_pool"] is not None)
    for key in (
        "rooms",
        "players",
        "pending_startups",
        "join_attempts",
        "retiring",
        "retiring_players",
    ):
        counters["runtime_" + key] = length(runtime, key, (dict,))
    counters["runtime_player_closures"] = sum(
        unfinished(t) for t in runtime["player_closures"].values()
    )
    recordings = manager["live_recordings"]
    if type(recordings) is not dict:
        raise ValueError("Unknown live recording registry")
    totals = dict(
        open=0,
        unfinished=0,
        pending=0,
        items=0,
        pending_bytes=0,
        upload_buffer_bytes=0,
        object_buffer_bytes=0,
        ui_bytes=0,
        ui_tasks=0,
        rgb_slots=0,
        alive_processes=0,
    )
    for item in recordings.values():
        if not named(item, module + "stream_archive", "StreamingRecording"):
            raise ValueError("Unknown live recording")
        data = fields(item)
        if type(data["closed"]) is not bool:
            raise ValueError("Unknown recording lifecycle")
        totals["open"] += int(not data["closed"])
        totals["unfinished"] += unfinished(data["task"])
        totals["pending"] += int(data["pending"] is not None)
        totals["items"] += length(data, "items", (deque,))
        totals["rgb_slots"] += length(data, "rgb_slots", (dict,))
        totals["ui_tasks"] += sum(unfinished(t) for t in data["ui_tasks"].values())
        for key in (
            "pending_bytes",
            "upload_buffer_bytes",
            "object_buffer_bytes",
            "ui_bytes",
        ):
            totals[key] += integer(data, key)
        process = data["process"]
        if process is not None:
            # Read cached lifecycle state; never call is_alive/poll/join, which
            # can reap or mutate the running process owner's child registry.
            popen = fields(process)["_popen"]
            if popen is not None:
                code = fields(popen)["returncode"]
                if code is not None and type(code) is not int:
                    raise ValueError("Unknown archive child state")
                totals["alive_processes"] += int(code is None)
    counters.update({"recording_" + k: v for k, v in totals.items()})
    return {
        "manager_count": 1,
        "runtime_count": 1,
        "retained_recordings": len(recordings),
        "counters": counters,
    }


def __whitebox_cutover_idle_probe_v1():
    import gc
    import hashlib
    import json
    import os
    from pathlib import Path
    import sys
    import threading
    import time

    script = Path("/tmp/whitebox-cutover-probe-v1/probe.py")
    request_file = script.with_name("request.json")
    if not request_file.exists():
        return False
    request = json.loads(request_file.read_text())
    result = dict(
        schema_version=1,
        nonce=request["nonce"],
        pid=os.getpid(),
        starttime=Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
        probe_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        python=list(sys.version_info[:3]),
        main_thread=threading.current_thread() is threading.main_thread(),
        gil=sys._is_gil_enabled(),
        observed_at=time.time(),
    )
    try:
        result.update(archive_snapshot(gc.get_objects(), request["owner_id"]))
    except Exception as error:  # noqa: BLE001 - never expose business data in diagnostics
        result["error"] = type(error).__name__
    output = script.with_name("result.json")
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(result, stream)
    return True


if __whitebox_cutover_idle_probe_v1():
    del __whitebox_cutover_idle_probe_v1, archive_snapshot
