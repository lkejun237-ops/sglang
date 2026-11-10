"""Prepare and switch exactly authorized Docker Room/Gateway images (Python 3.12).

Private configuration and receipts stay outside Git. No image build, pull,
prune, model mutation or container deletion is performed by this operator.
"""

import argparse
import copy
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import signal
import subprocess
import time
from urllib.parse import quote
from urllib.request import Request, HTTPRedirectHandler, build_opener

IDLE_METRICS = (
    "world_recording_active_sessions",
    "world_recording_buffer_bytes",
    "world_model_whitebox_recording_pending_operations",
    "world_model_whitebox_recording_pending_bytes",
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def write_private(path, value):
    with path.open("x", encoding="utf8") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


class Docker:
    def __init__(self, path):
        require(Path(path).is_absolute(), "Docker socket must be absolute")
        self.socket = path

    def api(self, method, path, body=None):
        connection = http.client.HTTPConnection("localhost", timeout=60)
        connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.sock.settimeout(60)
        connection.sock.connect(self.socket)
        try:
            connection.request(
                method,
                "/v1.41" + path,
                body=None if body is None else json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            data = response.read()
            require(
                response.status < 300, "Docker request failed: " + str(response.status)
            )
            return json.loads(data) if data else None
        finally:
            connection.close()

    def inspect(self, value, image=False):
        return self.api(
            "GET",
            ("/images/" if image else "/containers/") + quote(value, safe="") + "/json",
        )

    def cli(self, *args):
        # Never print Docker output on failure: inspect/probes can contain secrets.
        result = subprocess.run(
            ["docker", "--host", "unix://" + self.socket, *args],
            capture_output=True,
            check=False,
            timeout=120,
        )
        require(result.returncode == 0, "Docker command failed")
        return result.stdout

    def file(self, container, path):
        return self.cli(
            "exec",
            container,
            "python",
            "-c",
            "import pathlib,sys;sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())",
            path,
        )

    def stop(self, container):
        self.api("POST", "/containers/" + container + "/stop?t=30")

    def start(self, container):
        self.api("POST", "/containers/" + container + "/start")

    def probe(self, item, user):
        script = """import json,hashlib,sys,os
from pathlib import Path
r=json.loads(Path('/app/release.json').read_text())
f=json.loads(sys.argv[1]);r['sdk_files']={p:hashlib.sha256((Path('/opt/worldkit')/p).read_bytes()).hexdigest() for p in f}
r['owner']=[os.getuid(),os.getgid()]
print(json.dumps(r))"""
        user_args = ("--user", user) if user else ()
        return json.loads(
            self.cli(
                "run",
                "--rm",
                "--network",
                "none",
                *user_args,
                "--entrypoint",
                "python",
                item["image"],
                "-c",
                script,
                json.dumps(item.get("sdk_files", {})),
            )
        )

    def archive_probe(self, container, spec):
        probe = Path(__file__).with_name("docker_idle_probe.py")
        require(sha(probe.read_bytes()) == spec["probe_sha256"], "Probe source changed")
        nonce = os.urandom(24).hex()
        base = "/tmp/whitebox-cutover-probe-v1"
        setup = """import json,os,sys
from pathlib import Path
assert sys.version_info[:2] == (3,14)
module=b'sglang.multimodal_gen.runtime.entrypoints.whitebox_room_server'
rows=[p for p in Path('/proc').iterdir() if p.name.isdigit() and module in (p/'cmdline').read_bytes().split(b'\\0')]
assert len(rows)==1
p=rows[0]; uid=p.stat().st_uid;gid=p.stat().st_gid
base=Path(sys.argv[1]);base.mkdir(mode=0o700);os.chown(base,uid,gid)
r=json.loads(sys.argv[2]);r.update(pid=int(p.name),starttime=(p/'stat').read_text().rsplit(')',1)[1].split()[19])
f=base/'request.json';f.write_text(json.dumps(r));f.chmod(0o600);os.chown(f,uid,gid)
print(json.dumps(r))"""
        request = {"nonce": nonce, "owner_id": spec["owner_id"]}
        binding = json.loads(
            self.cli(
                "exec",
                "--user",
                "0",
                container,
                "python",
                "-c",
                setup,
                base,
                json.dumps(request),
            )
        )
        self.cli("cp", str(probe), container + ":" + base + "/probe.py")
        self.cli(
            "exec",
            "--user",
            "0",
            "--privileged",
            container,
            "python",
            "-c",
            "import sys;sys.remote_exec(int(sys.argv[1]),sys.argv[2])",
            str(binding["pid"]),
            base + "/probe.py",
        )
        result = None
        try:
            for attempt in range(100):
                result = json.loads(
                    self.cli(
                        "exec",
                        "--user",
                        "0",
                        container,
                        "python",
                        "-c",
                        "from pathlib import Path;import sys; p=Path(sys.argv[1]);print(p.read_text() if p.exists() else 'null')",
                        base + "/result.json",
                    )
                )
                if result is not None:
                    break
                time.sleep(0.1)
            require(result is not None, "Interpreter probe did not complete")
            current = (
                self.cli(
                    "exec",
                    "--user",
                    "0",
                    container,
                    "python",
                    "-c",
                    "from pathlib import Path;import sys;print(Path('/proc/'+sys.argv[1]+'/stat').read_text().rsplit(')',1)[1].split()[19])",
                    str(binding["pid"]),
                )
                .decode()
                .strip()
            )
            require(current == binding["starttime"], "Interpreter process changed")
            validate_archive_probe(result, binding, spec)
            return result
        finally:
            # A delayed remote script must still find its exact input. Keep the
            # exclusive directory on timeout for explicit reconciliation.
            if isinstance(result, dict) and result.get("nonce") == nonce:
                self.cli("exec", "--user", "0", container, "rm", "-r", base)


def validate_archive_probe(value, binding, spec):
    require(not value.get("error"), "Archive probe could not inspect the owner")
    require(
        all(value.get(k) == binding[k] for k in ("nonce", "pid", "starttime")),
        "Archive probe process identity mismatch",
    )
    require(
        value.get("schema_version") == 1
        and value.get("main_thread") is True
        and value.get("gil") is True
        and value.get("python", [])[:2] == [3, 14]
        and value.get("probe_sha256") == spec["probe_sha256"]
        and value.get("manager_count") == value.get("runtime_count") == 1,
        "Archive probe identity or interpreter is unsupported",
    )
    require(
        abs(time.time() - value["observed_at"]) < 30, "Archive probe evidence is stale"
    )
    expected = {
        "active",
        "tasks",
        "status_pending",
        "io_active",
        "upload_pending",
        "rgb_used",
        "unfinished_manager_tasks",
        "recovery",
        "upload_pool",
        *(
            "runtime_" + k
            for k in (
                "rooms",
                "players",
                "pending_startups",
                "join_attempts",
                "retiring",
                "retiring_players",
                "player_closures",
            )
        ),
        *(
            "recording_" + k
            for k in (
                "open",
                "unfinished",
                "pending",
                "items",
                "pending_bytes",
                "upload_buffer_bytes",
                "object_buffer_bytes",
                "ui_bytes",
                "ui_tasks",
                "rgb_slots",
                "alive_processes",
            )
        ),
    }
    counters = value.get("counters", {})
    require(set(counters) == expected, "Archive probe fields are incomplete")
    require(
        all(type(v) is int and v >= 0 for v in counters.values()),
        "Unknown archive counter",
    )
    require(
        all(v == 0 for k, v in counters.items() if k != "rgb_used"),
        "Archives are still busy",
    )
    require(
        counters["rgb_used"] == 0 or spec.get("allow_orphaned_rgb_budget") is True,
        "Residual archive RGB budget needs explicit review",
    )


def environment(old):
    return dict(value.split("=", 1) for value in old["Config"]["Env"])


def candidate_body(old, item, bundle_path, digest):
    require(
        old["Id"] == item["old_id"] and old["Name"] == "/" + item["old_name"],
        "Old container identity changed",
    )
    require(environment(old)["WHITEBOX_ROLE"] == item["role"], "Role mismatch")
    require(
        old["HostConfig"]["NetworkMode"] == "host",
        "Only existing host networking is supported",
    )
    require(
        not old["Config"].get("Volumes")
        and all(mount["Type"] == "bind" for mount in old.get("Mounts", [])),
        "Only explicit bind mounts are supported",
    )
    require(
        not old["HostConfig"].get("VolumesFrom"),
        "Inherited container volumes are not supported",
    )
    body = copy.deepcopy(old["Config"])
    host = copy.deepcopy(old["HostConfig"])
    expected = {(x["source"], x["target"]) for x in item.get("remove_mounts", [])}
    require(
        len(expected) == len(item.get("remove_mounts", [])), "Duplicate mount removal"
    )
    require(
        all(target in {"/app/python", "/app/release.json"} for _, target in expected),
        "Only runtime source overlays may be removed",
    )
    found = set()
    bundle_target = environment(old)["WHITEBOX_RELEASE_BUNDLE"]
    bundle_count = 0
    binds = []
    for bind in host.get("Binds") or []:
        parts = bind.split(":")
        require(len(parts) in {2, 3}, "Unsupported bind syntax")
        source, target = parts[:2]
        if (source, target) in expected:
            require(
                len(parts) == 3 and parts[2] == "ro", "Source overlay must be read-only"
            )
            found.add((source, target))
            continue
        require(
            target != "/app"
            and not any(
                target == blocked or target.startswith(blocked + "/")
                for blocked in (
                    "/app/python",
                    "/app/release.json",
                    "/opt/worldkit",
                    "/opt/whitebox-native",
                )
            ),
            "Unexpected runtime overlay",
        )
        if target == bundle_target:
            require(len(parts) == 3 and parts[2] == "ro", "Bundle must be read-only")
            parts[0] = bundle_path
            bundle_count += 1
        binds.append(":".join(parts))
    host["Binds"] = binds
    mounts = []
    for mount in host.get("Mounts") or []:
        source, target = mount.get("Source"), mount["Target"]
        if (source, target) in expected:
            require(
                mount.get("ReadOnly") and mount["Type"] == "bind",
                "Invalid source overlay",
            )
            found.add((source, target))
            continue
        require(
            target != "/app"
            and not any(
                target == blocked or target.startswith(blocked + "/")
                for blocked in (
                    "/app/python",
                    "/app/release.json",
                    "/opt/worldkit",
                    "/opt/whitebox-native",
                )
            ),
            "Unexpected runtime overlay",
        )
        if target == bundle_target:
            require(
                mount.get("ReadOnly") and mount["Type"] == "bind",
                "Bundle must be read-only",
            )
            mount["Source"] = bundle_path
            bundle_count += 1
        mounts.append(mount)
    if "Mounts" in host:
        host["Mounts"] = mounts
    require(found == expected and bundle_count == 1, "Mount inventory changed")
    env = environment(old)
    env["WHITEBOX_RELEASE_CONFIG_SHA256"] = digest
    body.update(
        Image=item["image"], Env=[k + "=" + v for k, v in env.items()], HostConfig=host
    )
    # Keep service ownership labels while updating the immutable image identity.
    body.setdefault("Labels", {})
    body["Labels"] = dict(
        body["Labels"] or {},
        **{
            "org.opencontainers.image.revision": item["source"],
            "com.seedleap.worldkit.revision": item["sdk"],
            "com.seedleap.source.branch": item["branch"],
        },
    )
    return body


def validate_pool(value):
    groups = value["roles"]
    require("model_group" in groups, "Model lease evidence missing")
    for group in groups.values():
        for key in ("active_sessions", "queued_sessions", "waiting_sessions"):
            require(group.get(key) == 0, "Role pool is busy or incomplete")
    for group in value.get("role_capacity", {}).values():
        require(group.get("active_sessions") == 0, "Role lease is active")


def validate_metrics(text):
    for metric in IDLE_METRICS:
        values = [
            float(line.split()[-1])
            for line in text.splitlines()
            if line.startswith(metric + " ") or line.startswith(metric + "{")
        ]
        require(
            values and all(value == 0 for value in values),
            "Recording evidence is busy or incomplete",
        )


def http(url, *, body=None, token=None, raw=False):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = Request(
        url, data=None if body is None else json.dumps(body).encode(), headers=headers
    )

    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    with build_opener(NoRedirect).open(request, timeout=15) as response:
        data = response.read(4 * 1024 * 1024 + 1)
    require(len(data) <= 4 * 1024 * 1024, "HTTP evidence is too large")
    return data.decode() if raw else json.loads(data)


def idle(config, engine, snapshots):
    def read(endpoint, raw=False):
        token = None
        if endpoint.get("token_env"):
            token = environment(snapshots[endpoint["token_role"]])[
                endpoint["token_env"]
            ]
        return http(endpoint["url"], body=endpoint.get("body"), token=token, raw=raw)

    spec = config["idle"]
    value = read(spec["room_status"])
    service = value["service"]
    require(service.get("active") == service.get("queued") == 0, "Room is busy")
    require(
        isinstance(service.get("workers"), list)
        and all(worker.get("busy") is False for worker in service["workers"]),
        "CLI worker evidence is busy or incomplete",
    )
    require(
        value.get("maintenance", {}).get("operation") is None,
        "Room maintenance is active",
    )
    require(spec["role_pools"], "Role pool evidence missing")
    for endpoint in spec["role_pools"]:
        validate_pool(read(endpoint))
    require(
        ("metrics" in spec) != ("manager_probe" in spec),
        "Exactly one recording evidence owner is required",
    )
    if "manager_probe" in spec:
        engine.archive_probe(snapshots["room"]["Id"], spec["manager_probe"])
    else:
        validate_metrics(read(spec["metrics"], raw=True))


def gated_text(config, original):
    require(sha(original.encode()) == config["sha256"], "Nginx configuration changed")
    require(original.count(config["marker"]) == 1, "Nginx route marker is not unique")
    paths = config["paths"]
    require(paths and len(paths) == len(set(paths)), "Ingress paths are required")
    require(
        all(
            re.fullmatch(r"/[A-Za-z0-9_./-]+", path) and ".." not in path
            for path in paths
        ),
        "Invalid ingress path",
    )
    gate = "\n# immutable-room-gateway-cutover\n" + "".join(
        " location = " + path + ' { add_header Retry-After "5" always; return 503; }\n'
        for path in paths
    )
    return original.replace(config["marker"], gate + config["marker"], 1)


def nginx_reload():
    for args in (["nginx", "-t"], ["nginx", "-s", "reload"]):
        result = subprocess.run(args, capture_output=True, timeout=30)
        require(result.returncode == 0, "Nginx validation/reload failed")


def process_start(pid, proc=Path("/proc")):
    return (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]


def nginx_master(master, proc=Path("/proc")):
    require(
        b"nginx: master process" in (proc / str(master) / "cmdline").read_bytes(),
        "Nginx master process identity mismatch",
    )
    return process_start(master, proc)


def nginx_workers(master, proc=Path("/proc")):
    children = (
        (proc / str(master) / "task" / str(master) / "children").read_text().split()
    )
    workers = {}
    for child in children:
        try:
            command = (proc / child / "cmdline").read_bytes().replace(b"\x00", b" ")
            if b"nginx: worker process" in command:
                workers[child] = process_start(child, proc)
        except FileNotFoundError:
            continue
    return workers


def nginx_drain(config):
    """Wait for pre-gate workers, including accepted WebSockets, without signals."""
    master = int(Path(config["pid_file"]).read_text().strip())
    master_start = nginx_master(master)
    before = nginx_workers(master)
    require(before, "Nginx worker evidence missing")
    evidence = {
        "master_pid": master,
        "master_start": master_start,
        "workers": before,
        "drained": False,
    }
    path = Path(config["evidence_path"]) if config.get("evidence_path") else None
    if path:
        write_private(path, evidence)

    def record(current, drained):
        if path:
            evidence.update(
                remaining={
                    pid: started
                    for pid, started in before.items()
                    if current.get(pid) == started
                },
                drained=drained,
            )
            # The intent already exists; keep this private evidence up to date.
            with path.open("w") as stream:
                json.dump(evidence, stream)
                stream.flush()
                os.fsync(stream.fileno())

    def wait():
        deadline = time.monotonic() + config.get("drain_timeout_s", 30)
        while True:
            require(
                int(Path(config["pid_file"]).read_text().strip()) == master
                and nginx_master(master) == master_start,
                "Nginx master changed",
            )
            current = nginx_workers(master)
            require(current, "Nginx has no live worker evidence")
            drained = all(
                current.get(pid) != started for pid, started in before.items()
            )
            record(current, drained)
            if drained:
                return
            require(
                time.monotonic() < deadline,
                "Old Nginx workers still have connections; no container was stopped",
            )
            time.sleep(0.2)

    return wait


def validate_config(config):
    if "nginx" in config:
        require(
            Path(config["nginx"]["path"]).is_absolute()
            and re.fullmatch(r"[a-f0-9]{64}", config["nginx"]["sha256"]),
            "Exact absolute Nginx file/hash required",
        )
        require(
            0 < config["nginx"].get("drain_timeout_s", 30) <= 300,
            "Invalid drain timeout",
        )
        require(
            config["nginx"].get("pid_file") in {"/run/nginx.pid", "/var/run/nginx.pid"},
            "Nginx PID file must be an approved absolute path",
        )
    items = config["containers"]
    roles = [item["role"] for item in items]
    require(
        set(roles) <= {"room", "gateway"}
        and len(roles) == len(set(roles))
        and "room" in roles,
        "Only one Room and optional Gateway are allowed",
    )
    require(
        sorted(config["authorized_roles"]) == sorted(roles),
        "Authorized role list must match exactly",
    )
    for item in items:
        require(
            all(
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", item[name])
                for name in ("old_name", "candidate_name")
            )
            and item["old_name"] != item["candidate_name"],
            "Distinct exact container names required",
        )
        require(
            re.fullmatch(r"[a-f0-9]{64}", item["old_id"]),
            "Full old container ID required",
        )
        require(
            re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", item["image"]),
            "Immutable image digest required",
        )
        for key in ("source", "old_source", "sdk", "old_sdk"):
            require(
                re.fullmatch(r"[a-f0-9]{40}", item[key]), "Full source/SDK SHA required"
            )
        require(
            item["branch"] == "main" or item["branch"].startswith("codex/"),
            "Untrusted branch",
        )
        if item["role"] == "room":
            require(
                item.get("sdk_files"), "Independent frozen SDK file hashes required"
            )
        for name, digest in item.get("sdk_files", {}).items():
            require(
                not name.startswith("/")
                and ".." not in name.split("/")
                and re.fullmatch(r"[a-f0-9]{64}", digest),
                "Invalid SDK proof",
            )
    require(
        len({item["old_id"] for item in items}) == len(items), "Duplicate old container"
    )
    require(
        len({item["candidate_name"] for item in items}) == len(items),
        "Duplicate candidate name",
    )


def prepare(config, root, engine):
    validate_config(config)
    require(
        not (root / "prepare-intent.json").exists(),
        "Reconcile existing preparation before retrying",
    )
    write_private(
        root / "prepare-intent.json",
        {"config_sha256": fingerprint(config), "at": time.time()},
    )
    snapshots, candidates = {}, []
    for item in config["containers"]:
        old = engine.inspect(item["old_id"])
        require(old["State"]["Running"], "Old container is not running")
        release = json.loads(engine.file(old["Id"], "/app/release.json"))
        require(
            release["source_sha"] == item["old_source"]
            and release["sdk_sha"] == item["old_sdk"],
            "Old release changed",
        )
        image = engine.inspect(item["image"], image=True)
        require(
            item["image"] in image.get("RepoDigests", []),
            "Image digest is not local and authenticated",
        )
        observed = engine.probe(item, old["Config"].get("User", ""))
        require(
            all(
                observed[key] == item[target]
                for key, target in [
                    ("source_sha", "source"),
                    ("sdk_sha", "sdk"),
                    ("source_branch", "branch"),
                ]
            ),
            "Candidate release does not match",
        )
        require(
            observed["sdk_files"] == item.get("sdk_files", {}),
            "Candidate SDK proof does not match",
        )
        env = environment(old)
        raw_bundle = engine.file(old["Id"], env["WHITEBOX_RELEASE_BUNDLE"])
        bundle = json.loads(raw_bundle)
        bundle["policy"].update(
            source_sha=item["source"],
            sdk_sha=item["sdk"],
            release_branch=item["branch"],
        )
        digest = fingerprint(
            {"policy": bundle["policy"], "bindings": bundle["bindings"]}
        )
        bundle["config_sha256"] = digest
        bundle["environments"][item["role"]]["WHITEBOX_RELEASE_CONFIG_SHA256"] = digest
        path = root / (item["role"] + "-release.private.json")
        write_private(path, bundle)
        owner = observed["owner"]
        require(
            len(owner) == 2
            and all(type(value) is int and value >= 0 for value in owner),
            "Candidate owner proof missing",
        )
        os.chown(path, *owner)
        body = candidate_body(old, item, str(path), digest)
        created = engine.api(
            "POST",
            "/containers/create?name=" + quote(item["candidate_name"], safe=""),
            body,
        )
        write_private(
            root / (item["role"] + "-created.json"),
            {"id": created["Id"], "name": item["candidate_name"]},
        )
        candidate = engine.inspect(created["Id"])
        require(
            candidate["State"]["Status"] == "created"
            and candidate["Image"] == image["Id"],
            "Candidate creation mismatch",
        )
        snapshots[item["role"]] = old
        candidates.append(
            {
                "id": created["Id"],
                "role": item["role"],
                "image_id": image["Id"],
                "body": body,
                "config_sha256": digest,
                "old_bundle_sha256": sha(raw_bundle),
                "snapshot": candidate,
                "bundle_path": str(path),
                "bundle_sha256": sha(path.read_bytes()),
            }
        )
    write_private(
        root / "prepared.json",
        {
            "config_sha256": fingerprint(config),
            "snapshots": snapshots,
            "candidates": candidates,
        },
    )


def wait_ready(item, candidate, reader=http):
    for attempt in range(30):
        try:
            value = reader(item["version_url"])
            require(
                value["source_sha"] == item["source"]
                and value["sdk_sha"] == item["sdk"]
                and value["config_sha256"] == candidate["config_sha256"],
                "Candidate runtime identity mismatch",
            )
            require(reader(item["health_url"])["status"] == "ok", "Candidate unhealthy")
            return
        except (OSError, ValueError, KeyError):
            if attempt == 29:
                raise
            time.sleep(1)


def switch(
    config,
    root,
    engine,
    checker=idle,
    reload=nginx_reload,
    ready=wait_ready,
    drain=nginx_drain,
):
    validate_config(config)
    proof = json.loads((root / "prepared.json").read_text())
    require(
        proof["config_sha256"] == fingerprint(config), "Private configuration changed"
    )
    require(
        not (root / "switch-intent.json").exists(),
        "Reconcile prior switch before retrying",
    )
    snapshots = proof["snapshots"]
    for item, candidate in zip(config["containers"], proof["candidates"], strict=True):
        old = engine.inspect(item["old_id"])
        before = snapshots[item["role"]]
        require(
            all(old[key] == before[key] for key in ("Config", "HostConfig", "Image"))
            and old["State"]["Running"]
            and old["State"]["StartedAt"] == before["State"]["StartedAt"],
            "Old container changed since preparation",
        )
        require(
            sha(engine.file(old["Id"], environment(old)["WHITEBOX_RELEASE_BUNDLE"]))
            == candidate["old_bundle_sha256"],
            "Old mounted bundle changed",
        )
        current = engine.inspect(candidate["id"])
        require(
            current["State"]["Status"] == "created"
            and current["Image"] == candidate["image_id"]
            and all(
                current[key] == candidate["snapshot"][key]
                for key in ("Config", "HostConfig")
            ),
            "Candidate changed since preparation",
        )
        require(
            sha(Path(candidate["bundle_path"]).read_bytes())
            == candidate["bundle_sha256"],
            "Candidate bundle changed since preparation",
        )
    nginx = Path(config["nginx"]["path"])
    original = nginx.read_text()
    gated = gated_text(config["nginx"], original)
    checker(config, engine, snapshots)
    write_private(
        root / "switch-intent.json",
        {"config_sha256": fingerprint(config), "at": time.time()},
    )
    drained = drain(
        {**config["nginx"], "evidence_path": str(root / "nginx-workers.private.json")}
    )
    stopped = []
    gate_written = False
    rollback_failed = False
    try:
        gate_written = True
        nginx.write_text(gated)
        reload()
        drained()
        checker(config, engine, snapshots)
        for item, candidate in zip(
            config["containers"], proof["candidates"], strict=True
        ):
            require(
                engine.inspect(item["old_id"])["State"]["StartedAt"]
                == snapshots[item["role"]]["State"]["StartedAt"],
                "Old container restarted",
            )
            stopped.append((item, candidate))
            engine.stop(item["old_id"])
            engine.start(candidate["id"])
            ready(item, candidate)
        write_private(
            root / "deployed.json",
            {
                "state": "deployed_health_only",
                "roles": config["authorized_roles"],
                "at": time.time(),
            },
        )
    except BaseException:
        rollback_errors = []
        for item, candidate in reversed(stopped):
            try:
                if engine.inspect(candidate["id"])["State"]["Running"]:
                    engine.stop(candidate["id"])
                if not engine.inspect(item["old_id"])["State"]["Running"]:
                    engine.start(item["old_id"])
                old_item = {
                    **item,
                    "source": item["old_source"],
                    "sdk": item["old_sdk"],
                }
                ready(
                    old_item,
                    {
                        "config_sha256": environment(snapshots[item["role"]])[
                            "WHITEBOX_RELEASE_CONFIG_SHA256"
                        ]
                    },
                )
            except BaseException as error:
                rollback_errors.append(type(error).__name__)
        rollback_failed = bool(rollback_errors)
        write_private(
            root / "rollback.json", {"errors": rollback_errors, "at": time.time()}
        )
        require(
            not rollback_errors, "Rollback readiness unconfirmed; ingress remains gated"
        )
        raise
    finally:
        if gate_written and not rollback_failed:
            require(
                nginx.read_text() == gated,
                "Nginx changed during switch; reconcile gate manually",
            )
            nginx.write_text(original)
            reload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "switch"])
    parser.add_argument("config", type=Path)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    os.umask(0o077)

    def interrupted(signum, frame):
        raise KeyboardInterrupt("Cutover interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    root = args.stage.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    engine = Docker(config["docker_socket"])
    lock = Path(
        "/var/lock/whitebox-cutover-"
        + sha(config["docker_socket"].encode())[:16]
        + ".lock"
    )
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        globals()[args.mode](config, root, engine)
    print(json.dumps({"stage": str(root), "mode": args.mode, "complete": True}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit(
            "Docker cutover failed; inspect private stage and reconcile before retrying"
        )
