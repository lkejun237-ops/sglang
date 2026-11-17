"""Prepare and switch exactly authorized Docker Room/Gateway images (host Python 3.6+).

Private configuration and receipts stay outside Git. No image build, pull,
prune, model mutation or container deletion is performed by this operator.
"""

import argparse
import copy
import fcntl
import hashlib
import http.client as http_client
import json
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

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
        connection = http_client.HTTPConnection("localhost", timeout=60)
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
        self.cli("exec", "--user", "0", container, "chmod", "0444", base + "/probe.py")
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


def startup_prepare_setting(env, item):
    change = item.get("startup_prepare")
    if change is None:
        return
    key = "WHITEBOX_MODEL_STARTUP_PREPARE"
    require(
        isinstance(change, dict)
        and set(change) == {"expected", "value"}
        and change["expected"] in {"0", "1"}
        and change["value"] in {"0", "1"},
        "Invalid explicit startup prepare setting",
    )
    require(env.get(key) == change["expected"], "Startup prepare setting drifted")
    require(
        change["value"] != "1" or env.get("WHITEBOX_BOUND_COMPUTE") == "1",
        "Startup preparation requires the existing player binding",
    )
    env[key] = change["value"]


def room_cache_environment(env, item):
    """Explicit cache rollout settings; no general environment override."""
    require(
        not (item.get("room_cache") and item.get("service_selection")),
        "Room cache routing and explicit service selection both set role hints",
    )
    change = item.get("room_cache")
    if change is None:
        return {}
    require(item["role"] == "room", "Cache settings belong to Room only")
    require(
        env.get("WHITEBOX_OSS_PREFIX") == change["expected_recording_prefix"],
        "Recording prefix changed",
    )
    prefix = change["recording_prefix"]
    require(
        re.fullmatch(r"white_box_game_record/[A-Za-z0-9_/-]+", prefix)
        and ".." not in prefix,
        "Invalid recording prefix",
    )
    previous = env.get("WHITEBOX_CACHE_RECORDING_SECRET")
    require(
        (sha(previous.encode()) if previous is not None else None)
        == change.get("expected_recording_secret_sha256"),
        "Recording credential changed",
    )
    path = Path(change["recording_secret_file"])
    require(path.is_absolute() and path.stat().st_size <= 4096, "Invalid secret file")
    raw = path.read_bytes()
    require(sha(raw) == change["recording_secret_sha256"], "Secret file changed")
    value = raw.decode("ascii").strip()
    require(
        len(value) >= 32 and all(33 <= ord(char) <= 126 for char in value),
        "Invalid recording credential",
    )
    require(
        env.get("WHITEBOX_CACHE_RECORDING_ENABLED") in {None, "0", "1"},
        "Invalid current cache recording setting",
    )
    return {
        "WHITEBOX_OSS_PREFIX": prefix,
        "WHITEBOX_CACHE_RECORDING_SECRET": value,
        "WHITEBOX_CACHE_RECORDING_ENABLED": "1",
        # The new compatibility selects this complete role group. Old global
        # preferences can name a service outside it and block every admission.
        "WHITEBOX_PREFERRED_SERVICE_IDS": "",
        "WHITEBOX_AVOID_SERVICE_IDS": "",
    }


GATEWAY_PREFILL_POOL = "/run/whitebox/role-pool-prefill.json"


def gateway_prefill_environment(env, item):
    """Enable batch only with a frozen pool, credential and Redis binding."""
    change = item.get("gateway_prefill")
    if change is None:
        return {}
    require(item["role"] == "gateway", "Prefill settings belong to Gateway only")
    documents = {}
    for name in ("environment", "pool"):
        path = Path(change[name + "_file"])
        require(
            path.is_absolute() and path.stat().st_size <= 1024**2,
            "Invalid private binding",
        )
        raw = path.read_bytes()
        require(sha(raw) == change[name + "_sha256"], "Private prefill binding changed")
        documents[name] = json.loads(raw)
    values, pool = documents["environment"], documents["pool"]
    token = pool.get("room_token_env", "")
    require(
        re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}_TOKEN", token),
        "Invalid room credential name",
    )
    keys = {
        "WHITEBOX_PREFILL_ENABLED",
        "WHITEBOX_PREFILL_REDIS_URL",
        "WHITEBOX_MODEL_BACKEND",
        "WHITEBOX_ROLE_POOL_CONFIG",
        token,
    }
    require(
        set(values) == keys and set(change["expected_sha256"]) == keys,
        "Unreviewed prefill environment",
    )
    for key in keys:
        previous = env.get(key)
        require(
            (sha(previous.encode()) if previous is not None else None)
            == change["expected_sha256"][key],
            "Gateway prefill environment changed",
        )
        require(
            isinstance(values[key], str) and "\n" not in values[key],
            "Invalid environment value",
        )
    require(
        values["WHITEBOX_PREFILL_ENABLED"] == "1"
        and values["WHITEBOX_MODEL_BACKEND"] == "role_pool"
        and values["WHITEBOX_ROLE_POOL_CONFIG"] == GATEWAY_PREFILL_POOL
        and len(values[token]) >= 32
        and len(env.get("WHITEBOX_RTC_AUTH_SECRET", "")) >= 32,
        "Prefill authorization or backend is incomplete",
    )
    redis = urlsplit(values["WHITEBOX_PREFILL_REDIS_URL"])
    coordinator = urlsplit(pool.get("coordinator_url", ""))
    require(
        redis.scheme in {"redis", "rediss"}
        and redis.hostname
        and not redis.query
        and not redis.fragment
        and re.fullmatch(r"/\d+", redis.path)
        and coordinator.scheme in {"http", "https"}
        and coordinator.hostname
        and coordinator.username is None
        and coordinator.password is None
        and isinstance(pool.get("routes"), dict)
        and pool["routes"],
        "Prefill result store or pool route is incomplete",
    )
    return values


def prepare_room_cache_route(old, item, root, engine, owner):
    if item.get("gateway_prefill") is not None:
        require(
            item.get("room_condition_switch") is None,
            "Condition switch route belongs to Room only",
        )
        env = environment(old)
        gateway_prefill_environment(env, item)
        require(
            env.get("WHITEBOX_ROLE_POOL_CONFIG") in {None, GATEWAY_PREFILL_POOL},
            "Existing pool path changed",
        )
        change = item["gateway_prefill"]
        raw = Path(change["pool_file"]).read_bytes()
        path = root / "gateway-role-pool.private.json"
        write_private(path, json.loads(raw))
        os.chown(path, *owner)
        previous = (
            engine.file(old["Id"], GATEWAY_PREFILL_POOL)
            if env.get("WHITEBOX_ROLE_POOL_CONFIG")
            else None
        )
        return dict(
            target=GATEWAY_PREFILL_POOL,
            old_sha256=sha(previous) if previous is not None else None,
            path=str(path),
            sha256=sha(path.read_bytes()),
            new=previous is None,
        )
    change = item.get("room_cache")
    protocol = item.get("room_condition_switch")
    if change is None and protocol is None:
        return None
    require(item["role"] == "room", "Role pool route belongs to Room only")
    target = environment(old)["WHITEBOX_ROLE_POOL_CONFIG"]
    require(Path(target).is_absolute(), "Role pool config path must be absolute")
    raw = engine.file(old["Id"], target)
    require(len(raw) <= 1024**2, "Role pool config is too large")
    document = json.loads(raw)
    if change is not None:
        route = document["routes"][change["route"]]
        require(
            route["compatibility"] == change["expected_compatibility"],
            "Role pool route changed",
        )
        require(
            re.fullmatch(r"[A-Za-z0-9_.-]{1,256}", change["compatibility"]),
            "Invalid candidate compatibility",
        )
        route["compatibility"] = change["compatibility"]
    if protocol is not None:
        require(
            isinstance(protocol, dict)
            and set(protocol) == {"route", "expected_pool_sha256", "value"}
            and protocol["expected_pool_sha256"] == sha(raw)
            and protocol["value"] in ("clip-barrier-v1", "clip-barrier-v2")
            and isinstance(protocol["route"], str)
            and protocol["route"] in document.get("routes", {}),
            "Invalid or changed Room condition switch route",
        )
        document["routes"][protocol["route"]]["condition_switch_protocol"] = protocol[
            "value"
        ]
    path = root / "room-role-pool.private.json"
    write_private(path, document)
    os.chown(path, *owner)
    return dict(
        target=target,
        old_sha256=sha(raw),
        path=str(path),
        sha256=sha(path.read_bytes()),
    )


def service_selection_settings(env, item):
    """Apply only reviewed Room selection hints, with an exact drift guard."""
    changes = item.get("service_selection", {})
    require(isinstance(changes, dict), "Invalid service selection settings")
    for key, change in changes.items():
        require(
            item["role"] == "room"
            and key in {"WHITEBOX_PREFERRED_SERVICE_IDS", "WHITEBOX_AVOID_SERVICE_IDS"}
            and isinstance(change, dict)
            and set(change) == {"expected", "value"},
            "Invalid service selection setting",
        )
        require(env.get(key) == change["expected"], "Service selection setting drifted")
        require(isinstance(change["value"], str), "Service selection must be JSON")
        value = json.loads(change["value"])
        require(isinstance(value, dict), "Service selection must be an object")
        for role, services in value.items():
            services = [services] if isinstance(services, str) else services
            require(
                role in {"encoder", "dit", "decoder"}
                and isinstance(services, list)
                and bool(services)
                and all(isinstance(s, str) and bool(s) for s in services),
                "Invalid selected service IDs",
            )
            if key == "WHITEBOX_AVOID_SERVICE_IDS":
                require(
                    isinstance(value[role], list), "Avoid service IDs must be lists"
                )
        env[key] = change["value"]


def candidate_body(old, item, bundle_path, digest, config_binding=None):
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
    config_count = 0
    if config_binding is not None:
        require(
            (
                (
                    item.get("room_cache") is not None
                    or item.get("room_condition_switch") is not None
                )
                and item["role"] == "room"
                and config_binding["target"]
                == environment(old)["WHITEBOX_ROLE_POOL_CONFIG"]
            )
            or (
                item.get("gateway_prefill") is not None
                and item["role"] == "gateway"
                and config_binding["target"] == GATEWAY_PREFILL_POOL
            ),
            "Unreviewed config binding",
        )
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
        if config_binding is not None and target == config_binding["target"]:
            require(
                len(parts) == 3 and parts[2] == "ro", "Role config must be read-only"
            )
            parts[0] = config_binding["path"]
            config_count += 1
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
        if config_binding is not None and target == config_binding["target"]:
            require(
                mount.get("ReadOnly") and mount["Type"] == "bind",
                "Role config must be read-only",
            )
            mount["Source"] = config_binding["path"]
            config_count += 1
        mounts.append(mount)
    if "Mounts" in host:
        host["Mounts"] = mounts
    if config_binding is not None and config_binding.get("new"):
        require(config_count == 0, "Unexpected pre-existing pool mount")
        host["Binds"].append(
            config_binding["path"] + ":" + config_binding["target"] + ":ro"
        )
        config_count = 1
    require(found == expected and bundle_count == 1, "Mount inventory changed")
    require(
        config_count == (1 if config_binding else 0), "Config mount inventory changed"
    )
    require(
        bool(config_binding)
        == bool(
            item.get("room_cache")
            or item.get("room_condition_switch")
            or item.get("gateway_prefill")
        ),
        "Cache route binding missing",
    )
    env = environment(old)
    env.update(room_cache_environment(env, item))
    env.update(gateway_prefill_environment(env, item))
    startup_prepare_setting(env, item)
    service_selection_settings(env, item)
    if item["branch"].startswith("codex/"):
        env["WHITEBOX_ALLOW_DEVELOPMENT_RELEASE"] = "1"
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


def http(url, *, body=None, token=None, raw=False, direct=False):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = Request(
        url, data=None if body is None else json.dumps(body).encode(), headers=headers
    )

    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = (
        build_opener(ProxyHandler({}), NoRedirect)
        if direct
        else build_opener(NoRedirect)
    )
    with opener.open(request, timeout=15) as response:
        data = response.read(4 * 1024 * 1024 + 1)
    require(len(data) <= 4 * 1024 * 1024, "HTTP evidence is too large")
    return data.decode() if raw else json.loads(data)


def idle(config, engine, snapshots):
    spec = config["idle"]
    source = spec.get("credential_source", "container")
    if "WHITEBOX_CONTROL_CREDENTIAL_REFS" in os.environ:
        from control_bindings import verify

        require(
            source == "publisher",
            "Managed control bindings require publisher credentials",
        )
        verify("route_publisher")
    require(source in ("container", "publisher"), "Unknown idle credential source")

    def read(endpoint, raw=False, publisher_env=None):
        token = None
        name = endpoint.get("token_env")
        if source == "publisher":
            require(name == publisher_env, "Unexpected publisher credential reference")
            if name:
                require(
                    os.environ.get(name + "_AUDIENCE") == endpoint["url"],
                    "Publisher credential does not cover this endpoint",
                )
                token = os.environ.get(name)
                require(token and token.strip(), "Publisher credential is not bound")
        elif name:
            token = environment(snapshots[endpoint["token_role"]])[name]
        return http(
            endpoint["url"],
            body=endpoint.get("body"),
            token=token,
            raw=raw,
            direct=source == "publisher",
        )

    if source == "publisher":
        require(
            spec["room_status"].get("body") == {"operation": "status"},
            "Publisher Room check requires the status operation",
        )
    value = read(spec["room_status"], publisher_env="WHITEBOX_INTERNAL_TOKEN")
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
        validate_pool(read(endpoint, publisher_env="ADAPTIVE_ROLE_POOL_ROOM_TOKEN"))
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
        result = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
        )
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
    if (
        "control_credential_refs" in config
        or "WHITEBOX_CONTROL_CREDENTIAL_REFS" in os.environ
    ):
        from control_bindings import REFS_ENV, parse_manifest, verify

        require(
            config["idle"].get("credential_source") == "publisher",
            "Managed control bindings require the independent publisher",
        )
        require(
            REFS_ENV in os.environ
            and "control_credential_refs" in config
            and parse_manifest(os.environ[REFS_ENV])
            == config["idle"].get(
                "control_credential_refs", config["control_credential_refs"]
            ),
            "Publisher and deployment control binding manifests differ",
        )
        parse_manifest(json.dumps(config["control_credential_refs"]))
        verify("route_publisher")
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
        if (
            "control_credential_refs" in bundle.get("bindings", {})
            or "WHITEBOX_CONTROL_CREDENTIAL_REFS" in env
        ):
            require(
                "control_credential_refs" in config,
                "Existing managed bindings require a reviewed target manifest",
            )
        bundle["policy"].update(
            source_sha=item["source"],
            sdk_sha=item["sdk"],
            release_branch=item["branch"],
        )
        if item.get("startup_prepare") is not None:
            policy_env = (
                bundle["policy"]
                .setdefault("environment", {})
                .setdefault(item["role"], {})
            )
            # Historical bundles have only compiled environments. Seed these
            # two controls from the observed runtime, then apply the same guard.
            for key in ("WHITEBOX_MODEL_STARTUP_PREPARE", "WHITEBOX_BOUND_COMPUTE"):
                policy_env.setdefault(key, env[key])
            startup_prepare_setting(policy_env, item)
            startup_prepare_setting(bundle["environments"][item["role"]], item)
        if item.get("service_selection"):
            policy_env = (
                bundle["policy"]
                .setdefault("environment", {})
                .setdefault(item["role"], {})
            )
            for key in item["service_selection"]:
                if key in env:
                    policy_env.setdefault(key, env[key])
            service_selection_settings(policy_env, item)
            service_selection_settings(bundle["environments"][item["role"]], item)
        cache_env = room_cache_environment(env, item)
        if cache_env:
            policy_env = (
                bundle["policy"]
                .setdefault("environment", {})
                .setdefault(item["role"], {})
            )
            for key in (
                "WHITEBOX_OSS_PREFIX",
                "WHITEBOX_CACHE_RECORDING_ENABLED",
                "WHITEBOX_PREFERRED_SERVICE_IDS",
                "WHITEBOX_AVOID_SERVICE_IDS",
            ):
                policy_env[key] = cache_env[key]
            # Secrets stay in private runtime bindings, not public policy metadata.
            bundle["environments"][item["role"]].update(cache_env)
        prefill_env = gateway_prefill_environment(env, item)
        if prefill_env:
            public = (
                bundle["policy"].setdefault("environment", {}).setdefault("gateway", {})
            )
            for key in (
                "WHITEBOX_PREFILL_ENABLED",
                "WHITEBOX_MODEL_BACKEND",
                "WHITEBOX_ROLE_POOL_CONFIG",
            ):
                public[key] = prefill_env[key]
            bundle["environments"]["gateway"].update(prefill_env)
        if item["branch"].startswith("codex/"):
            bundle["policy"].setdefault("environment", {}).setdefault(item["role"], {})[
                "WHITEBOX_ALLOW_DEVELOPMENT_RELEASE"
            ] = "1"
            bundle["environments"][item["role"]][
                "WHITEBOX_ALLOW_DEVELOPMENT_RELEASE"
            ] = "1"
        if "control_credential_refs" in config:
            from control_bindings import REFS_ENV

            serialized = json.dumps(
                config["control_credential_refs"], sort_keys=True, separators=(",", ":")
            )
            bundle["bindings"]["control_credential_refs"] = serialized
            bundle["environments"][item["role"]][REFS_ENV] = serialized
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
        config_binding = prepare_room_cache_route(old, item, root, engine, owner)
        body = candidate_body(old, item, str(path), digest, config_binding)
        if "control_credential_refs" in config:
            body["Env"] = [v for v in body["Env"] if not v.startswith(REFS_ENV + "=")]
            body["Env"].append(REFS_ENV + "=" + serialized)
            from control_bindings import verify

            verify(
                item["role"],
                environ=dict(v.split("=", 1) for v in body["Env"]),
                preparing=True,
            )
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
                **(
                    {
                        "control_binding_sha256": fingerprint(
                            config["control_credential_refs"]
                        )
                    }
                    if "control_credential_refs" in config
                    else {}
                ),
                "snapshot": candidate,
                "bundle_path": str(path),
                "bundle_sha256": sha(path.read_bytes()),
                "config_binding": config_binding,
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
            if "control_binding_sha256" in candidate:
                require(
                    value.get("control_binding_sha256")
                    == candidate["control_binding_sha256"],
                    "Candidate control binding versions differ",
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
    require(
        len(config["containers"]) == len(proof["candidates"]),
        "Candidate count changed",
    )
    for item, candidate in zip(config["containers"], proof["candidates"]):
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
        room_cache_environment(environment(old), item)
        gateway_prefill_environment(environment(old), item)
        binding = candidate.get("config_binding")
        if binding is not None:
            require(
                (
                    binding.get("new")
                    or sha(engine.file(old["Id"], binding["target"]))
                    == binding["old_sha256"]
                )
                and sha(Path(binding["path"]).read_bytes()) == binding["sha256"],
                "Role pool config changed since preparation",
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
        for item, candidate in zip(config["containers"], proof["candidates"]):
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
