"""Bounded, explicit unused-image maintenance on the local Docker host.

Never removes containers, volumes, writable layers or database files. Planning
is read-only. Applying requires an exact plan hash and rechecks each image's
current references and immutable registry recovery before non-forced deletion.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

GIB = 1024**3
IMAGE_ID = re.compile(r"sha256:[a-f0-9]{64}")
DOCKER = ["docker", "--host", "unix:///var/run/docker.sock"]


def command(*args):
    return subprocess.check_output(
        DOCKER + list(args), universal_newlines=True, timeout=60
    )


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def fingerprint(plan):
    return hashlib.sha256(canonical(plan)).hexdigest()


def snapshot():
    info = json.loads(command("info", "--format", "{{json .}}"))
    root = Path(info["DockerRootDir"])
    if not root.is_dir():
        raise ValueError("Run disk maintenance on the actual Docker host")
    ids = sorted(set(command("image", "ls", "-aq", "--no-trunc").split()))
    images = []
    for start in range(0, len(ids), 50):
        for image in json.loads(command("image", "inspect", *ids[start : start + 50])):
            images.append(
                {
                    k: image.get(k)
                    for k in (
                        "Id",
                        "Created",
                        "Size",
                        "RepoTags",
                        "RepoDigests",
                        "RootFS",
                    )
                }
            )
    ids = command("ps", "-aq", "--no-trunc").split()
    containers = []
    for start in range(0, len(ids), 50):
        for c in json.loads(command("container", "inspect", *ids[start : start + 50])):
            containers.append({"id": c["Id"], "image": c["Image"]})
    return {
        "daemon_id": info["ID"],
        "root": str(root.resolve()),
        "free_bytes": shutil.disk_usage(root).free,
        "images": images,
        "containers": containers,
    }


def protected_images(state, protected):
    direct = set(protected) | {c["image"] for c in state["containers"]}
    # Preserve ancestor image records too; deleting their names rarely frees
    # payload while a live/rollback descendant still references those layers.
    chains = [
        i.get("RootFS", {}).get("Layers", [])
        for i in state["images"]
        if i["Id"] in direct
    ]
    result = set(direct)
    for i in state["images"]:
        layers = i.get("RootFS", {}).get("Layers", [])
        if layers and any(c[: len(layers)] == layers for c in chains):
            result.add(i["Id"])
    return result


def created_epoch(value):
    # Docker RFC3339Nano, including offsets; the operator host may use Python3.6.
    m = re.fullmatch(
        r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.\d+)?(Z|[+-]\d\d:\d\d)", value
    )
    if not m:
        raise ValueError("Invalid image creation time")
    t = datetime.datetime.strptime(m[1], "%Y-%m-%dT%H:%M:%S")
    offset = 0
    if m[2] != "Z":
        hours, minutes = int(m[2][1:3]), int(m[2][4:6])
        if hours > 23 or minutes > 59:
            raise ValueError("Invalid image timezone")
        offset = (hours * 60 + minutes) * (1 if m[2][0] == "+" else -1)
    return t.replace(
        tzinfo=datetime.timezone(datetime.timedelta(minutes=offset))
    ).timestamp()


def candidates(state, repositories, protected, *, now, age_hours):
    protected = protected_images(state, protected)
    allowed = set(repositories)
    result = []
    for i in state["images"]:
        refs = (i.get("RepoTags") or []) + (i.get("RepoDigests") or [])
        if i["Id"] in protected or not IMAGE_ID.fullmatch(i["Id"]):
            continue
        # Untagged/local-only build artifacts have no proven registry restore.
        if not refs or not i.get("RepoDigests"):
            continue
        if any(
            ref.split("@")[0].rsplit(":", 1)[0] not in allowed
            for ref in i.get("RepoTags") or []
        ):
            continue
        if any(ref.split("@")[0] not in allowed for ref in i["RepoDigests"]):
            continue
        if any(
            ref.rsplit(":", 1)[-1] in {"latest", "current", "rollback", "prod"}
            for ref in i.get("RepoTags") or []
        ):
            continue
        try:
            created = created_epoch(i["Created"])
        except (TypeError, ValueError):
            continue
        if now - created < age_hours * 3600:
            continue
        result.append({k: i[k] for k in ("Id", "Created", "RepoTags", "RepoDigests")})
    return sorted(result, key=lambda x: x["Created"])


def make_plan(state, repositories, protected, *, age_hours, target_free_bytes, now):
    if (
        type(age_hours) is not int
        or age_hours < 72
        or type(target_free_bytes) is not int
        or target_free_bytes < 0
    ):
        raise ValueError("Require at least 72 hours of age and a nonnegative target")
    if not repositories or any(not x or "://" in x or "@" in x for x in repositories):
        raise ValueError("Provide exact scoped registry repository names")
    if any(not IMAGE_ID.fullmatch(x) for x in protected):
        raise ValueError("Protected images must be immutable local image IDs")
    return {
        "schema": 1,
        "created_at": now,
        "daemon_id": state["daemon_id"],
        "root": state["root"],
        "repositories": repositories,
        "protected": protected,
        "age_hours": age_hours,
        "target_free_bytes": target_free_bytes,
        "images": candidates(
            state, repositories, protected, now=now, age_hours=age_hours
        ),
    }


def registry_exists(ref):
    # --verbose supplies the actual immutable platform descriptor. Refuse
    # ambiguous/multi-platform responses rather than treating an auth/error
    # page or a differently tagged image as a recoverable artifact.
    if not re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", ref):
        return False
    try:
        value = json.loads(command("manifest", "inspect", "--verbose", ref))
        return (
            isinstance(value, dict)
            and value.get("Descriptor", {}).get("digest") == ref.rsplit("@", 1)[1]
        )
    except (subprocess.SubprocessError, ValueError):
        return False


def check_headroom(incoming_gib, reserve_gib=15, *, read=snapshot):
    import math

    if (
        not all(math.isfinite(x) for x in (incoming_gib, reserve_gib))
        or incoming_gib <= 0
        or reserve_gib < 0
    ):
        raise ValueError("Incoming size must be positive and reserve nonnegative")
    state = read()
    required = int((reserve_gib + 2 * incoming_gib) * GIB)
    result = {
        "free_bytes": state["free_bytes"],
        "required_bytes": required,
        "allowed": state["free_bytes"] >= required,
    }
    if not result["allowed"]:
        raise ValueError("Insufficient Docker disk headroom; do not pull/build")
    return result


def apply(
    plan,
    expected,
    receipt,
    *,
    now=time.time,
    read=snapshot,
    recover=registry_exists,
    remove=None,
):
    if fingerprint(plan) != expected:
        raise ValueError("Plan hash changed")
    if plan.get("schema") != 1 or not 0 <= now() - plan["created_at"] <= 3600:
        raise ValueError("Plan is stale; inspect the current host again")
    # Validate policy again; an apply file cannot bypass the planning bounds.
    initial = read()
    make_plan(
        initial,
        plan["repositories"],
        plan["protected"],
        age_hours=plan["age_hours"],
        target_free_bytes=plan["target_free_bytes"],
        now=now(),
    )
    if initial["daemon_id"] != plan["daemon_id"] or initial["root"] != plan["root"]:
        raise ValueError("Docker host identity changed")
    # A previous/unknown mutation must be inspected; never blindly replay it.
    with Path(receipt).open("x") as log:

        def record(value):
            log.write(json.dumps(value) + "\n")
            log.flush()
            os.fsync(log.fileno())

        record({"event": "begin", "plan_sha256": expected, "at": now()})
        for image in plan["images"]:
            state = read()
            if state["daemon_id"] != plan["daemon_id"] or state["root"] != plan["root"]:
                raise ValueError("Docker host identity changed")
            if state["free_bytes"] >= plan["target_free_bytes"]:
                break
            eligible = candidates(
                state,
                plan["repositories"],
                plan["protected"],
                now=now(),
                age_hours=plan["age_hours"],
            )
            if image not in eligible:
                record({"event": "skipped_changed", "image": image["Id"]})
                continue
            if not any(recover(ref) for ref in image["RepoDigests"]):
                record({"event": "skipped_no_registry_proof", "image": image["Id"]})
                continue
            # Recheck after the network lookup. Docker's own non-force removal
            # remains the last fence against a container starting concurrently.
            fresh = read()
            if fresh["daemon_id"] != plan["daemon_id"] or fresh["root"] != plan["root"]:
                raise ValueError("Docker host identity changed")
            if image not in candidates(
                fresh,
                plan["repositories"],
                plan["protected"],
                now=now(),
                age_hours=plan["age_hours"],
            ):
                record({"event": "skipped_changed", "image": image["Id"]})
                continue
            record({"event": "remove_submitted", "image": image["Id"], "at": now()})
            if remove is None:
                result = subprocess.run(
                    DOCKER + ["image", "rm", "--no-prune", image["Id"]],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    timeout=60,
                )
                record(
                    {
                        "event": "remove_returned",
                        "image": image["Id"],
                        "exit_code": result.returncode,
                        "output": (result.stdout + result.stderr)[-2000:],
                    }
                )
            else:
                remove(image["Id"])
                record(
                    {"event": "remove_returned", "image": image["Id"], "exit_code": 0}
                )
        final = read()
        if final["daemon_id"] != plan["daemon_id"] or final["root"] != plan["root"]:
            raise ValueError("Docker host identity changed")
        record(
            {
                "event": "complete",
                "free_bytes": final["free_bytes"],
                "target_met": final["free_bytes"] >= plan["target_free_bytes"],
            }
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="action")
    commands.required = True
    inventory = commands.add_parser("plan")
    inventory.add_argument("--repository", action="append", required=True)
    inventory.add_argument("--protect", action="append", default=[])
    inventory.add_argument("--age-hours", type=int, default=72)
    inventory.add_argument("--target-free-gib", type=int, default=35)
    inventory.add_argument("--output", type=Path, required=True)
    commit = commands.add_parser("apply")
    commit.add_argument("--plan", type=Path, required=True)
    commit.add_argument("--plan-sha256", required=True)
    commit.add_argument("--receipt", type=Path, required=True)
    check = commands.add_parser("check")
    check.add_argument("--incoming-gib", type=float, required=True)
    check.add_argument("--reserve-gib", type=float, default=15)
    args = p.parse_args()
    if args.action == "apply":
        apply(json.loads(args.plan.read_text()), args.plan_sha256, args.receipt)
    elif args.action == "plan":
        state = snapshot()
        plan = make_plan(
            state,
            args.repository,
            args.protect,
            age_hours=args.age_hours,
            target_free_bytes=args.target_free_gib * GIB,
            now=time.time(),
        )
        with args.output.open("x") as f:
            json.dump(plan, f, indent=2)
        print(
            json.dumps(
                {
                    "candidates": len(plan["images"]),
                    "free_bytes": state["free_bytes"],
                    "plan_sha256": fingerprint(plan),
                    "applied": False,
                }
            )
        )
    else:
        # Conservative allowance for retained unpacked layers plus temporary
        # compressed data. Shared-layer savings must not justify a full disk.
        print(json.dumps(check_headroom(args.incoming_gib, args.reserve_gib)))


if __name__ == "__main__":
    main()
