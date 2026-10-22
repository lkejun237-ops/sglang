# SPDX-License-Identifier: Apache-2.0
"""Runtime allocator history plus bounded per-process sampling, benchmark only."""

import json
import os
import threading
import time
import traceback
from pathlib import Path

STARTED = None
MAX_ENTRIES = int(os.environ.get("ZING_BENCHMARK_MEMORY_ENTRIES", "8000000"))
INTERVAL_PEAKS = {}
ENABLED = True


def emit(kind, **fields):
    import torch

    if not ENABLED:
        return
    directory = Path(os.environ["ZING_BENCHMARK_MEMORY_DIR"]) / "memory"
    directory.mkdir(exist_ok=True, parents=True)
    device = torch.cuda.current_device()
    row = dict(
        kind=kind,
        role=os.environ["ZING_BENCHMARK_ROLE"],
        pid=os.getpid(),
        device=device,
        local_rank=os.environ.get("LOCAL_RANK"),
        monotonic=time.monotonic(),
        unix_ns=time.time_ns(),
        allocated=torch.cuda.memory_allocated(device),
        reserved=torch.cuda.memory_reserved(device),
        peak_allocated=torch.cuda.max_memory_allocated(device),
        peak_reserved=torch.cuda.max_memory_reserved(device),
        **fields,
    )
    with (directory / ("samples-%s-%d.jsonl" % (row["role"], row["pid"]))).open(
        "a"
    ) as f:
        f.write(json.dumps(row) + "\n")


def history(clear=False):
    import torch

    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=MAX_ENTRIES,
        clear_history=clear,
    )


def start():
    global STARTED
    import torch

    if STARTED == os.getpid():
        return
    STARTED = os.getpid()
    device = torch.cuda.current_device()
    history()
    root = Path(os.environ["ZING_BENCHMARK_MEMORY_DIR"]) / "memory"
    root.mkdir(exist_ok=True, parents=True)
    identity = dict(
        clock_monotonic=time.monotonic(),
        clock_unix_ns=time.time_ns(),
        role=os.environ["ZING_BENCHMARK_ROLE"],
        pid=os.getpid(),
        device=device,
        name=torch.cuda.get_device_name(device),
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        max_entries=MAX_ENTRIES,
        torch_version=torch.__version__,
        record_stacks="python",
    )
    (root / ("process-%d.json" % os.getpid())).write_text(
        json.dumps(identity, indent=2)
    )
    emit("history_started")

    def loop():
        global ENABLED
        torch.cuda.set_device(device)
        last = 0
        while True:
            try:
                emit("sample")
                path = (
                    Path(os.environ["ZING_BENCHMARK_MEMORY_DIR"])
                    / "memory-control.json"
                )
                command = json.loads(path.read_text()) if path.exists() else None
                if command and command["sequence"] > last:
                    last = command["sequence"]
                    label = command["label"]
                    assert label.replace("-", "").isalnum()
                    torch.cuda.synchronize(device)
                    before = dict(
                        allocated=torch.cuda.memory_allocated(device),
                        reserved=torch.cuda.memory_reserved(device),
                        peak_allocated=torch.cuda.max_memory_allocated(device),
                        peak_reserved=torch.cuda.max_memory_reserved(device),
                    )
                    before["peak_allocated"] = max(
                        before["peak_allocated"],
                        INTERVAL_PEAKS.get(device, {}).get("allocated", 0),
                    )
                    before["peak_reserved"] = max(
                        before["peak_reserved"],
                        INTERVAL_PEAKS.get(device, {}).get("reserved", 0),
                    )
                    snapshot = None
                    if command["operation"] == "snapshot":
                        snapshot = root / (
                            "%s-%s-d%d-p%d-s%d.pickle"
                            % (label, identity["role"], device, os.getpid(), last)
                        )
                        torch.cuda.memory._dump_snapshot(str(snapshot))
                    elif command["operation"] not in ("reset", "disable"):
                        raise ValueError("Unknown memory command")
                    if command["operation"] == "disable":
                        torch.cuda.memory._record_memory_history(
                            enabled=None, clear_history=True
                        )
                        ENABLED = False
                    else:
                        history(clear=True)
                    torch.cuda.reset_peak_memory_stats(device)
                    INTERVAL_PEAKS[device] = {
                        "allocated": torch.cuda.memory_allocated(device),
                        "reserved": torch.cuda.memory_reserved(device),
                    }
                    emit(
                        "memory_command",
                        sequence=last,
                        label=label,
                        snapshot=None if snapshot is None else snapshot.name,
                    )
                    ack = dict(
                        identity,
                        sequence=last,
                        label=label,
                        snapshot=None if snapshot is None else snapshot.name,
                        passed=True,
                        counters=before,
                        max_entries=MAX_ENTRIES,
                    )
                    target = root / ("ack-%d-%d.json" % (last, os.getpid()))
                    temporary = target.with_suffix(".next")
                    temporary.write_text(json.dumps(ack, indent=2))
                    os.replace(temporary, target)
                    if command["operation"] == "disable":
                        return
            except Exception:
                (root / ("ERROR-%d.txt" % os.getpid())).write_text(
                    traceback.format_exc()
                )
                return
            time.sleep(0.2)

    threading.Thread(target=loop, name="allocator-observer", daemon=True).start()


def install():
    import torch

    original = torch.cuda.set_device

    def set_device(device):
        value = original(device)
        if os.environ.get("ZING_BENCHMARK_ROLE"):
            start()
        return value

    torch.cuda.set_device = set_device
    reset = torch.cuda.reset_peak_memory_stats

    def observed_reset(device=None):
        if not ENABLED:
            return reset(device)
        device = torch.cuda.current_device() if device is None else device
        if not isinstance(device, int):
            device = torch.device(device).index
        if device is None:
            device = torch.cuda.current_device()
        peak = INTERVAL_PEAKS.setdefault(device, {"allocated": 0, "reserved": 0})
        peak["allocated"] = max(
            peak["allocated"], torch.cuda.max_memory_allocated(device)
        )
        peak["reserved"] = max(peak["reserved"], torch.cuda.max_memory_reserved(device))
        emit("peak_reset_observed", reset_device=device)
        return reset(device)

    torch.cuda.reset_peak_memory_stats = observed_reset
