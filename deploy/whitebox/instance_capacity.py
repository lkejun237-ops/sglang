"""Admit a measured instance footprint before starting its GPU worker command.

This is a local launcher, not a scheduler. All instances must use the same
machine policy/directory. GPU costs include weights, KV, workspaces and peak
headroom and must come from a measured hardware profile. Driver free memory is
checked too; software declarations cannot enforce memory used by bypassing jobs.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from sglang.multimodal_gen.runtime.utils.shared_capacity import (
    CapacityUnavailable,
    SharedCapacity,
)


def gpu_inventory():
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {
        row[0].strip(): {"total": int(row[1]) * 2**20, "free": int(row[2]) * 2**20}
        for line in raw.splitlines()
        if line.strip()
        for row in [line.split(",")]
    }


def admit(policy, instance, inventory):
    if not Path(policy["capacity_dir"]).is_absolute():
        raise ValueError("Machine capacity directory must be absolute")
    item = policy["instances"][instance]
    total = policy["host_cache_bytes"]
    if type(total) is not int or total <= 0:
        raise ValueError("Machine host cache budget must be positive")
    for profile in policy["instances"].values():
        if (
            type(profile["host_cache_bytes"]) is not int
            or not 0 < profile["host_cache_bytes"] <= total
        ):
            raise ValueError("Instance host cache budget is invalid")
    if sum(p["host_cache_bytes"] for p in policy["instances"].values()) > total:
        raise ValueError("Configured instance L1 budgets exceed the machine total")
    capacity = SharedCapacity(policy["capacity_dir"])
    leases = []
    try:
        with capacity.locked():
            capacity.policy(
                "instance-plan",
                {
                    key: policy[key]
                    for key in ("host_cache_bytes", "gpu_bytes", "instances")
                },
            )
            rows = capacity.owners()
            owners = {r.get("owner") for r in rows if r["resource"].startswith("gpu:")}
            records = []
            for marker in capacity.root.glob("*.instance"):
                record = json.loads(marker.read_text())
                records.append(record)
                if record["instance"] not in owners and set(record["gpus"]) & set(
                    item["gpu_bytes"]
                ):
                    raise CapacityUnavailable(
                        "Unconfirmed GPU owner exit; quarantine requires operator recovery"
                    )
            for gpu, amount in sorted(item["gpu_bytes"].items()):
                limit = policy["gpu_bytes"][gpu]
                if (
                    type(limit) is not int
                    or limit <= 0
                    or limit > inventory[gpu]["total"]
                ):
                    raise ValueError("GPU budget exceeds actual hardware memory")
                if (
                    type(amount) is not int
                    or not 0 < amount <= limit
                    or amount > inventory[gpu]["free"]
                ):
                    raise ValueError(
                        "GPU instance footprint does not fit available memory"
                    )
                resource = "gpu:" + gpu
                capacity.policy(resource, {"limit": limit})
                marked = {r["instance"] for r in records}
                used = sum(r["gpus"].get(gpu, 0) for r in records) + sum(
                    r["bytes"]
                    for r in rows
                    if r["resource"] == resource and r.get("owner") not in marked
                )
                if used + amount > limit:
                    raise CapacityUnavailable("Shared GPU capacity budget exhausted")
                leases.append(
                    capacity.create(
                        dict(resource=resource, bytes=amount, owner=instance)
                    )
                )
            if not leases:
                raise ValueError("Instance must declare its measured GPU footprint")
            leases.append(capacity.reserve_locked("instance:" + instance, 1, 1))
        return leases
    except BaseException:
        for lease in leases:
            lease.close()
        raise


def supervise(policy, instance, command, leases):
    """Keep capacity until the entire foreground command group is gone.

    A killed supervisor leaves a durable quarantine marker. It must never be
    mistaken for proof that its GPU children released memory. Worker commands
    must not daemonize or escape their process group.
    """
    capacity = SharedCapacity(policy["capacity_dir"])
    marker = capacity.root / (
        hashlib.sha256(instance.encode()).hexdigest() + ".instance"
    )
    with capacity.locked():
        with marker.open("x") as handle:
            json.dump(
                dict(
                    instance=instance, gpus=policy["instances"][instance]["gpu_bytes"]
                ),
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
    process = None
    handlers = {}
    try:
        env = dict(
            os.environ,
            SGLANG_CACHE_CAPACITY_DIR=str(capacity.root),
            SGLANG_CACHE_HOST_TOTAL_BYTES=str(policy["host_cache_bytes"]),
            SGLANG_CACHE_INSTANCE_HOST_BYTES=str(
                policy["instances"][instance]["host_cache_bytes"]
            ),
        )
        process = subprocess.Popen(command, env=env, start_new_session=True)

        def forward(sig, _frame):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass

        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, forward)
        code = process.wait()
        while True:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            # Orphaned children retain capacity. Never kill them or claim free
            # slots solely because their launcher exited.
            time.sleep(0.1)
        with capacity.locked():
            marker.unlink()
        return code
    except BaseException:
        if process is None:
            with capacity.locked():
                marker.unlink(missing_ok=True)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        for lease in leases:
            lease.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A worker command is required")
    policy = json.loads(args.policy.read_text())
    leases = admit(policy, args.instance, gpu_inventory())
    raise SystemExit(supervise(policy, args.instance, command, leases))


if __name__ == "__main__":
    main()
