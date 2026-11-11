"""Prepare a new standalone Room container without modifying existing services.

Writes a non-secret environment file and a reviewable docker argv. Execution,
port forwarding/GA wiring, and enrollment remain explicit operator actions.
"""

import argparse
import copy
import ipaddress
import json
import math
import os
import re
from pathlib import Path
from urllib.parse import urlsplit


def plan(
    bundle,
    *,
    name,
    image,
    room_url,
    public_ip,
    public_port,
    listen_port,
    state_dir,
    output_dir,
    secret_file,
    cpus,
    memory,
    config_mounts=(),
):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name):
        raise ValueError("Use a unique safe Room owner/container name")
    if not re.fullmatch(r"[^\s@]+@sha256:[a-f0-9]{64}", image):
        raise ValueError("An immutable complete Room image digest is required")
    address = urlsplit(room_url)
    if (
        address.scheme not in {"http", "https"}
        or not address.hostname
        or address.username
        or address.password
        or address.path not in {"", "/"}
        or address.query
        or address.fragment
        or not address.port
    ):
        raise ValueError("The new Room requires a direct origin with explicit port")
    ipaddress.ip_address(public_ip)
    if any(
        type(p) is not int or not 1 <= p <= 65535 for p in (public_port, listen_port)
    ):
        raise ValueError("Invalid media port")
    if listen_port == address.port:
        raise ValueError("HTTP and RTC listeners require distinct ports")
    if Path(state_dir).name != name or state_dir == bundle["bindings"].get(
        "state_root"
    ):
        raise ValueError(
            "The private state directory must end with the new owner name and differ from the original state root"
        )
    for path in (state_dir, output_dir, secret_file):
        if not Path(path).is_absolute() or any(
            c in str(path) for c in ("\n", "\r", ",", "\0")
        ):
            raise ValueError("Use absolute private paths without separators")
    if (
        not math.isfinite(cpus)
        or cpus <= 0
        or not re.fullmatch(r"[1-9][0-9]*[mg]", memory)
    ):
        raise ValueError("Explicit CPU and memory budgets are required")
    env = dict(bundle["environments"]["room"])
    if env.get("WHITEBOX_DYNAMIC_ROOMS") != "1" or not env.get(
        "WHITEBOX_ROOM_POOL_COMPATIBILITY"
    ):
        raise ValueError("Enable room_pool_compatibility in the existing pool bundle")
    if int(env.get("WHITEBOX_ROOM_PIPELINE_CAPACITY", "1")) != 1:
        raise ValueError(
            "This single-endpoint template requires one source; use a reviewed multi-endpoint configuration for larger Rooms"
        )
    from config import render

    policy, bindings = copy.deepcopy(bundle["policy"]), dict(bundle["bindings"])
    bindings.update(
        owner_id=name,
        room_url=room_url.rstrip("/"),
        rtc_public_ip=public_ip,
        rtc_public_port=str(public_port),
        state_root=state_dir,
    )
    policy["environment"]["room"].update(
        WHITEBOX_ROOM_PORT=str(address.port),
        WHITEBOX_NATIVE_RTC_LISTEN=f"0.0.0.0:{listen_port}",
        WHITEBOX_ROOM_SPOOL="/var/lib/sglang/whitebox/rooms",
        WHITEBOX_RELEASE_BUNDLE="/etc/whitebox/room-bundle.json",
    )
    policy["environment"]["room"].pop("WHITEBOX_NATIVE_RTC_ENDPOINTS", None)
    generated_bundle = render(policy, bindings)
    env = dict(generated_bundle["environments"]["room"])
    env["WHITEBOX_RELEASE_BUNDLE"] = "/etc/whitebox/room-bundle.json"
    env_path = str(Path(output_dir) / "room.env")
    mounts = []
    targets = set()
    for item in config_mounts:
        if not isinstance(item, str) or item.count("=") != 1:
            raise ValueError("Use HOST_FILE=CONTAINER_FILE for each readonly config")
        source, target = item.split("=", 1)
        for path in (source, target):
            if (
                not Path(path).is_absolute()
                or len(Path(path).parts) < 2
                or ".." in Path(path).parts
                or any(c in path for c in (",", "\n", "\r", "\0"))
            ):
                raise ValueError("Invalid config mount path")
        if (
            Path(target).parts[1] in {"app", "usr", "bin", "lib", "lib64"}
            or target == "/opt/worldkit"
            or target.startswith("/opt/worldkit/")
            or target == "/etc/whitebox/room-bundle.json"
            or Path(target).suffix not in {".json", ".yaml", ".yml", ".toml"}
            or target in targets
        ):
            raise ValueError(
                "Config mounts cannot replace runtime code or release identity"
            )
        targets.add(target)
        mounts.extend(["--mount", f"type=bind,src={source},dst={target},readonly"])
    return {
        "schema_version": 1,
        "owner_id": name,
        "room_url": room_url.rstrip("/"),
        "image": image,
        "environment": env,
        "bundle": generated_bundle,
        "required_secrets": bundle["required_secret_keys"]["room"],
        "state_dir": state_dir,
        "docker_argv": [
            "docker",
            "run",
            "--detach",
            "--name",
            name,
            "--network",
            "host",
            "--cpus",
            str(cpus),
            "--memory",
            memory,
            "--log-driver",
            "local",
            "--log-opt",
            "max-size=32m",
            "--log-opt",
            "max-file=4",
            "--mount",
            f"type=bind,src={state_dir},dst=/var/lib/sglang/whitebox",
            "--mount",
            f"type=bind,src={Path(output_dir) / 'bundle.json'},dst=/etc/whitebox/room-bundle.json,readonly",
            *mounts,
            "--env-file",
            secret_file,
            "--env-file",
            env_path,
            image,
        ],
        "activation": "manual_start_then_explicit_enrollment",
        "existing_services_modified": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, type=Path)
    for name in (
        "name",
        "image",
        "room-url",
        "public-ip",
        "state-dir",
        "output-dir",
        "secret-file",
        "memory",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument(
        "--config-mount",
        action="append",
        default=[],
        dest="config_mounts",
        help="Readonly HOST_FILE=CONTAINER_FILE; repeat for existing role-pool/runtime config",
    )
    p.add_argument("--cpus", required=True, type=float)
    p.add_argument("--public-port", required=True, type=int)
    p.add_argument("--listen-port", required=True, type=int)
    a = p.parse_args()
    try:
        values = vars(a).copy()
        bundle = json.loads(values.pop("bundle").read_text())
        output = Path(a.output_dir)
        if output.exists():
            raise ValueError("The output directory already exists")
        result = plan(bundle, **values)
        output.mkdir(mode=0o700, parents=True)
        for name, value in [
            ("bundle.json", json.dumps(result["bundle"], indent=2) + "\n"),
            (
                "room.env",
                "".join(f"{k}={v}\n" for k, v in sorted(result["environment"].items())),
            ),
            (
                "plan.json",
                json.dumps(
                    {
                        k: v
                        for k, v in result.items()
                        if k not in {"environment", "bundle"}
                    },
                    indent=2,
                )
                + "\n",
            ),
        ]:
            with (output / name).open(
                "x", opener=lambda path, flags: os.open(path, flags, 0o600)
            ) as f:
                f.write(value)
        print(output / "plan.json")
    except (ValueError, KeyError, OSError) as error:
        p.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
