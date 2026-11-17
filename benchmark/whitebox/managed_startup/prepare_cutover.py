"""Bind a reviewed Shanghai-copy Room/Gateway cutover to fresh exact identities."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operator_directory", type=Path)
    p.add_argument("snapshot", type=Path)
    p.add_argument("images", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--room-only", action="store_true")
    p.add_argument("--name-suffix", default="")
    p.add_argument("--room-selection", type=Path)
    a = p.parse_args()
    prefix = "/var/lib/whitebox-shanghai-copy/managed-parallel-startup-20261113/"
    if not all(
        str(path).startswith(prefix) for path in [a.snapshot, a.images, a.output]
    ):
        raise ValueError("Expected private test stage")
    sys.path.insert(0, str(a.operator_directory))
    from docker_cutover import Docker, environment, http, validate_config, write_private

    engine = Docker("/run/worldkit-audio-docker/docker.sock")
    images = json.loads(a.images.read_text())
    specs = []
    current = {}
    for role, component, port in [("room", "room", 8092), ("gateway", "cpu", 18080)]:
        if a.room_only and role != "room":
            continue
        before = json.loads((a.snapshot / (role + "-inspect.private.json")).read_text())
        old = engine.inspect(before["Id"])
        if not old["State"]["Running"] or old["Name"] != before["Name"]:
            raise ValueError("Test target changed; take a fresh snapshot")
        release = json.loads(engine.file(old["Id"], "/app/release.json"))
        image = images[component]
        if not image["branch"].startswith("codex/managed-parallel-startup-"):
            raise ValueError("Unexpected candidate branch")
        env = environment(old)
        current[role] = (old, env)
        remove = [
            {"source": m["Source"], "target": m["Destination"]}
            for m in old["Mounts"]
            if m["Destination"] in ["/app/python", "/app/release.json"]
        ]
        specs.append(
            {
                "role": role,
                "old_id": old["Id"],
                "old_name": old["Name"].lstrip("/"),
                "candidate_name": "whitebox-sh-copy-"
                + role
                + "-managed-"
                + image["source"][:12]
                + a.name_suffix,
                "image": image["image"],
                "source": image["source"],
                "sdk": image["sdk"],
                "branch": image["branch"],
                "old_source": release["source_sha"],
                "old_sdk": release["sdk_sha"],
                "sdk_files": image.get("sdk_files", {}),
                "remove_mounts": remove,
                "startup_prepare": {
                    "expected": env["WHITEBOX_MODEL_STARTUP_PREPARE"],
                    "value": "1",
                },
                "health_url": "http://127.0.0.1:" + str(port) + "/healthz",
                "version_url": "http://127.0.0.1:" + str(port) + "/v1/whitebox/version",
            }
        )
        if role == "room" and a.room_selection:
            selection = json.loads(a.room_selection.read_text())
            specs[-1]["service_selection"] = {
                key: {"expected": env.get(key), "value": json.dumps(value)}
                for key, value in selection.items()
            }
    room, env = current["room"]
    cfg = json.loads(engine.file(room["Id"], env["WHITEBOX_ROLE_POOL_CONFIG"]))
    endpoints = []
    seen = set()
    for route in cfg.get("routes", {"default": {}}).values():
        selected = {**cfg, **route}
        url = selected["coordinator_url"].rstrip("/")
        if url != "http://10.0.1.37:18181":
            raise ValueError("Unexpected model pool")
        body = {
            key: selected[key] for key in ["profile", "network_domain", "compatibility"]
        }
        key = json.dumps(body, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        endpoints.append(
            {
                "url": url + "/v1/role-pool/capacity",
                "body": body,
                "token_role": "room",
                "token_env": selected["room_token_env"],
            }
        )
    health = http("http://127.0.0.1:8092/healthz")
    nginx = Path("/etc/nginx/conf.d/whitebox-copy.conf")
    text = nginx.read_text()
    if "shcopy.world-test-platform.loopit.com.cn" not in text:
        raise ValueError("Wrong nginx route")
    paths = []
    for base in ["/v1", "/backends/minwm/v1"]:
        for path in [
            "/webrtc/play",
            "/realtime_video/generate",
            "/realtime_video/prefill",
            "/whitebox/single-player-control",
            "/whitebox/room-control",
            "/whitebox/member-player-control",
        ]:
            paths += [base + path, "/whitebox" + base + path]
    value = {
        "docker_socket": "/run/worldkit-audio-docker/docker.sock",
        "authorized_roles": [item["role"] for item in specs],
        "containers": specs,
        "idle": {
            "room_status": {
                "url": "http://127.0.0.1:8092/internal/worlds/runtime",
                "body": {"operation": "status"},
                "token_role": "room",
                "token_env": "WHITEBOX_INTERNAL_TOKEN",
            },
            "role_pools": endpoints,
            "manager_probe": {
                "owner_id": health["owner_id"],
                "probe_sha256": hashlib.sha256(
                    (a.operator_directory / "docker_idle_probe.py").read_bytes()
                ).hexdigest(),
            },
        },
        "nginx": {
            "path": str(nginx),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "pid_file": "/run/nginx.pid",
            "marker": " location /whitebox/ {",
            "paths": paths,
            "drain_timeout_s": 90,
        },
    }
    validate_config(value)
    write_private(a.output, value)
    print(
        json.dumps(
            {
                "configured": True,
                "roles": value["authorized_roles"],
                "source": images["room"]["source"],
                "old_ids": [i["old_id"] for i in specs],
                "pool_routes": len(endpoints),
            }
        )
    )


if __name__ == "__main__":
    os.umask(0o077)
    main()
