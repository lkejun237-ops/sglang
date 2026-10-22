# SPDX-License-Identifier: Apache-2.0
"""Replay recorded allocator events, anchored to the final allocator snapshot."""

ACTIONS = {"alloc", "free_requested", "free_completed"}


def final_live(snapshot, device):
    return {
        b["address"]: [
            b.get("requested_size", b["size"]),
            b["state"] in ("active_awaiting_free", "active_pending_free"),
            None,
            False,
            b.get("frames", []),
        ]
        for s in snapshot["segments"]
        if s["device"] == device
        for b in s["blocks"]
        if b["state"]
        in ("active_allocated", "active_awaiting_free", "active_pending_free")
    }


def initial_live(trace, ending):
    live = {a: list(v) for a, v in ending.items()}
    for event in reversed(trace):
        action = event["action"]
        addr = event.get("addr")
        size = event.get("size")
        if action == "alloc":
            old = live.pop(addr, None)
            if old is None or old[0] != size or old[1]:
                raise ValueError("Reverse alloc mismatch")
        elif action == "free_completed":
            if addr in live:
                raise ValueError("Reverse free_completed overlaps live address")
            live[addr] = [size, True, None, False, []]
        elif action == "free_requested":
            old = live.get(addr)
            if old is None or old[0] != size or not old[1]:
                raise ValueError(
                    "Reverse free_requested mismatch: "
                    + str(
                        (addr, size, None if old is None else old[:2], event["time_us"])
                    )
                )
            old[1] = False
    return live


def replay(trace, ending, start_us, partial=False, buckets=500):
    live = initial_live(trace, ending)
    first = min(e["time_us"] for e in trace)
    origin = max(first, start_us) if partial else start_us
    baseline = None
    new = 0
    pending = 0
    peak = -1
    peak_pending = -1
    peak_state = None
    pending_state = None
    points = []
    allocated = sum(v[0] for v in live.values() if not v[1])
    first_time = origin
    last_time = max(e["time_us"] for e in trace)
    width = max(1, (last_time - first_time) / buckets)
    bucket = None
    bucket_points = []

    def flush():
        if not bucket_points:
            return
        keep = {0, len(bucket_points) - 1}
        for col in (1, 2, 3):
            keep.add(
                min(range(len(bucket_points)), key=lambda i: bucket_points[i][col])
            )
            keep.add(
                max(range(len(bucket_points)), key=lambda i: bucket_points[i][col])
            )
        keep.add(
            max(
                range(len(bucket_points)),
                key=lambda i: bucket_points[i][1] + bucket_points[i][2],
            )
        )
        keep.add(
            min(
                range(len(bucket_points)),
                key=lambda i: bucket_points[i][1] + bucket_points[i][2],
            )
        )
        points.extend(bucket_points[i] for i in sorted(keep))
        bucket_points.clear()

    def observe(t):
        nonlocal bucket, peak, peak_pending, peak_state, pending_state
        point = [(t - start_us) / 1e6, new, pending, allocated - baseline, allocated]
        key = int((t - first_time) / width)
        if bucket is not None and key != bucket:
            flush()
        bucket = key
        bucket_points.append(point)
        if new > peak:
            peak = new
            peak_state = (
                t,
                [(a, list(v)) for a, v in live.items() if v[3] and not v[1]],
            )
        if new + pending > peak_pending:
            peak_pending = new + pending
            pending_state = t

    for i, e in enumerate(trace):
        t = e["time_us"]
        action = e["action"]
        addr = e.get("addr")
        size = e.get("size")
        if baseline is None and t >= origin:
            baseline = allocated
            for v in live.values():
                v[3] = False
            observe(origin)
        if action == "alloc":
            if addr in live:
                raise ValueError("Allocation address is still live")
            born = baseline is not None
            live[addr] = [size, False, i, born, e.get("frames", [])]
            allocated += size
            if born:
                new += size
        elif action == "free_requested":
            v = live.get(addr)
            if v is None or v[0] != size or v[1]:
                raise ValueError("free_requested mismatch")
            allocated -= size
            v[1] = True
            if v[3]:
                new -= size
                pending += size
        elif action == "free_completed":
            v = live.pop(addr, None)
            if v is None or v[0] != size or not v[1]:
                raise ValueError("free_completed mismatch")
            if v[3]:
                pending -= size
        if baseline is not None and action in ACTIONS:
            observe(t)
    flush()
    expected = {a: (v[0], v[1]) for a, v in ending.items()}
    actual = {a: (v[0], v[1]) for a, v in live.items()}
    if actual != expected:
        raise ValueError("Forward replay does not match final snapshot")
    if baseline is None:
        raise ValueError("No events in selected interval")
    return dict(
        origin_us=origin,
        baseline_bytes=baseline,
        new_peak_bytes=peak,
        new_peak_us=peak_state[0],
        new_active_peak_bytes=peak_pending,
        new_active_peak_us=pending_state,
        net_peak_bytes=max(p[3] for p in points),
        end_allocated_bytes=allocated,
        peak_live=peak_state[1],
        points=points,
        closure=True,
    )


def stack_groups(live):
    groups = {}
    for addr, v in live:
        frames = v[4]
        filtered = [
            f
            for f in frames
            if not any(
                x in f.get("filename", "")
                for x in ["memory_probe.py", "full_chain_observer.py"]
            )
        ]
        chosen = (filtered or frames)[:10]
        key = tuple((f.get("filename"), f.get("line"), f.get("name")) for f in chosen)
        row = groups.setdefault(key, dict(bytes=0, count=0, frames=chosen))
        row["bytes"] += v[0]
        row["count"] += 1
    return sorted(groups.values(), key=lambda v: v["bytes"], reverse=True)
