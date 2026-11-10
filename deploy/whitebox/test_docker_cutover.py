import copy
import json
from pathlib import Path

import pytest

import docker_cutover as cutover


def fixture():
    env = [
        "WHITEBOX_ROLE=room",
        "WHITEBOX_RELEASE_CONFIG_SHA256=old",
        "WHITEBOX_CLI_FPS=36",
        "PRIVATE_TOKEN=not-real",
        "WHITEBOX_RELEASE_BUNDLE=/run/whitebox/release.json",
    ]
    old = {
        "Id": "a" * 64,
        "Name": "/old-room",
        "Image": "old-image",
        "State": {"Running": True, "StartedAt": "same"},
        "Config": {
            "Image": "old-image",
            "Env": env,
            "Cmd": ["dual-cli"],
            "Labels": {"user": "keep"},
        },
        "HostConfig": {
            "NetworkMode": "host",
            "Memory": 123,
            "Binds": [
                "/private/release:/run/whitebox/release.json:ro",
                "/private/python:/app/python:ro",
                "/data:/data:rw",
            ],
        },
    }
    item = {
        "role": "room",
        "old_id": old["Id"],
        "old_name": "old-room",
        "candidate_name": "new-room",
        "image": "registry/room@sha256:" + "b" * 64,
        "old_source": "c" * 40,
        "source": "d" * 40,
        "old_sdk": "e" * 40,
        "sdk": "f" * 40,
        "branch": "main",
        "remove_mounts": [{"source": "/private/python", "target": "/app/python"}],
        "health_url": "http://127.0.0.1:8092/healthz",
        "version_url": "http://127.0.0.1:8092/v1/whitebox/version",
        "sdk_files": {"packages/cli/src/main.ts": "1" * 64},
    }
    return old, item


def test_candidate_preserves_runtime_and_removes_only_exact_overlay(tmp_path):
    old, item = fixture()
    before = copy.deepcopy(old)
    body = cutover.candidate_body(old, item, str(tmp_path / "release.json"), "newhash")
    assert old == before
    assert body["Cmd"] == ["dual-cli"]
    assert body["HostConfig"]["Memory"] == 123
    assert body["HostConfig"]["Binds"] == [
        str(tmp_path / "release.json") + ":/run/whitebox/release.json:ro",
        "/data:/data:rw",
    ]
    env = dict(x.split("=", 1) for x in body["Env"])
    assert env["WHITEBOX_CLI_FPS"] == "36" and env["PRIVATE_TOKEN"] == "not-real"
    assert env["WHITEBOX_RELEASE_CONFIG_SHA256"] == "newhash"


@pytest.mark.parametrize("mutation", ["unexpected", "wrongsource", "sdkoverlay"])
def test_candidate_refuses_unreviewed_mount_changes(tmp_path, mutation):
    old, item = fixture()
    if mutation == "unexpected":
        old["HostConfig"]["Binds"].append("/other:/app/release.json:ro")
    elif mutation == "wrongsource":
        item["remove_mounts"][0]["source"] = "/wrong"
    else:
        old["HostConfig"]["Binds"].append("/other:/opt/worldkit:ro")
    with pytest.raises(ValueError):
        cutover.candidate_body(old, item, str(tmp_path / "release.json"), "newhash")


def test_idle_requires_each_metric_and_all_role_leases():
    metrics = "\n".join(x + " 0" for x in cutover.IDLE_METRICS)
    cutover.validate_metrics(metrics)
    with pytest.raises(ValueError):
        cutover.validate_metrics(metrics.replace(cutover.IDLE_METRICS[0] + " 0", ""))
    cutover.validate_pool(
        {
            "roles": {
                "model_group": {
                    "active_sessions": 0,
                    "queued_sessions": 0,
                    "waiting_sessions": 0,
                }
            }
        }
    )
    with pytest.raises(ValueError):
        cutover.validate_pool(
            {
                "roles": {
                    "model_group": {
                        "active_sessions": 1,
                        "queued_sessions": 0,
                        "waiting_sessions": 0,
                    }
                }
            }
        )


def test_gate_requires_exact_snapshot_and_unique_marker(tmp_path):
    path = tmp_path / "nginx.conf"
    original = (
        "server {\n location /whitebox/ { proxy_pass http://127.0.0.1:18080/; }\n}"
    )
    path.write_text(original)
    cfg = {
        "path": str(path),
        "sha256": cutover.sha(original.encode()),
        "marker": " location /whitebox/",
        "paths": ["/v1/webrtc/play", "/whitebox/v1/webrtc/play"],
    }
    gate = cutover.gated_text(cfg, original)
    assert gate.count("return 503") == 2
    assert cutover.gated_text(cfg, original) == gate
    with pytest.raises(ValueError):
        cutover.gated_text(cfg, original + "changed")


class FakeDocker:
    def __init__(self):
        self.old, self.item = fixture()
        self.old["Config"]["User"] = ""
        self.containers = {self.old["Id"]: self.old}
        self.events = []
        self.bundle = {
            "policy": {
                "source_sha": self.item["old_source"],
                "sdk_sha": self.item["old_sdk"],
                "release_branch": "main",
                "sampler": "adaptive",
            },
            "bindings": {"model": "existing"},
            "environments": {"room": {"WHITEBOX_RELEASE_CONFIG_SHA256": "old"}},
            "config_sha256": "old",
        }
        self.old_release = {
            "source_sha": self.item["old_source"],
            "sdk_sha": self.item["old_sdk"],
        }

    def inspect(self, value, image=False):
        if image:
            return {"Id": "new-image", "RepoDigests": [value]}
        return copy.deepcopy(self.containers[value])

    def file(self, container, path):
        return json.dumps(
            self.old_release if path == "/app/release.json" else self.bundle
        ).encode()

    def probe(self, item, user):
        return {
            "source_sha": item["source"],
            "sdk_sha": item["sdk"],
            "source_branch": item["branch"],
            "sdk_files": item["sdk_files"],
            "owner": [10001, 10001],
        }

    def api(self, method, path, body):
        assert method == "POST" and path.startswith("/containers/create?name=")
        self.containers["candidate"] = {
            "Id": "candidate",
            "Image": "new-image",
            "Config": {k: v for k, v in body.items() if k != "HostConfig"},
            "HostConfig": body["HostConfig"],
            "State": {"Status": "created", "Running": False, "StartedAt": ""},
        }
        return {"Id": "candidate"}

    def stop(self, value):
        self.events.append(("stop", value))
        self.containers[value]["State"]["Running"] = False

    def start(self, value):
        self.events.append(("start", value))
        self.containers[value]["State"]["Running"] = True


def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(cutover.os, "chown", lambda *args: None)
    fake = FakeDocker()
    path = tmp_path / "nginx.conf"
    path.write_text(
        "server {\n location /whitebox/ { proxy_pass http://127.0.0.1:18080/; }\n}"
    )
    config = {
        "docker_socket": "/run/example.sock",
        "authorized_roles": ["room"],
        "containers": [fake.item],
        "nginx": {
            "pid_file": "/run/nginx.pid",
            "path": str(path),
            "sha256": cutover.sha(path.read_bytes()),
            "marker": " location /whitebox/",
            "paths": ["/v1/webrtc/play"],
        },
    }
    cutover.prepare(config, tmp_path, fake)
    return config, fake, path


def test_prepare_only_creates_and_does_not_mutate_old_bundle(tmp_path, monkeypatch):
    config, fake, path = prepared(tmp_path, monkeypatch)
    assert fake.events == []
    assert fake.bundle["policy"]["source_sha"] == fake.item["old_source"]
    bundle = json.loads((tmp_path / "room-release.private.json").read_text())
    assert bundle["policy"]["sampler"] == "adaptive"
    assert bundle["policy"]["sdk_sha"] == fake.item["sdk"]
    assert (tmp_path / "prepared.json").exists()
    assert "return 503" not in path.read_text()


def test_switch_checks_idle_twice_and_stops_only_authorized_id(tmp_path, monkeypatch):
    config, fake, path = prepared(tmp_path, monkeypatch)
    original = path.read_text()
    checks = []

    def idle(*args):
        checks.append("return 503" in path.read_text())

    cutover.switch(
        config,
        tmp_path,
        fake,
        checker=idle,
        reload=lambda: None,
        drain=lambda spec: lambda: None,
        ready=lambda *args: None,
    )
    assert checks == [False, True]
    assert fake.events == [("stop", fake.item["old_id"]), ("start", "candidate")]
    assert not fake.containers[fake.item["old_id"]]["State"]["Running"]
    assert path.read_text() == original


def test_failed_candidate_restores_old_and_exact_ingress(tmp_path, monkeypatch):
    config, fake, path = prepared(tmp_path, monkeypatch)
    original = path.read_text()

    def ready(item, candidate):
        if item["source"] == fake.item["source"]:
            raise ValueError("unhealthy")

    with pytest.raises(ValueError, match="unhealthy"):
        cutover.switch(
            config,
            tmp_path,
            fake,
            checker=lambda *args: None,
            reload=lambda: None,
            drain=lambda spec: lambda: None,
            ready=ready,
        )
    assert fake.events == [
        ("stop", fake.item["old_id"]),
        ("start", "candidate"),
        ("stop", "candidate"),
        ("start", fake.item["old_id"]),
    ]
    assert path.read_text() == original
    assert json.loads((tmp_path / "rollback.json").read_text())["errors"] == []


def test_second_idle_failure_never_stops_old(tmp_path, monkeypatch):
    config, fake, path = prepared(tmp_path, monkeypatch)
    original = path.read_text()

    def idle(*args):
        if "return 503" in path.read_text():
            raise ValueError("busy")

    with pytest.raises(ValueError, match="busy"):
        cutover.switch(
            config,
            tmp_path,
            fake,
            checker=idle,
            reload=lambda: None,
            drain=lambda spec: lambda: None,
        )
    assert fake.events == [] and path.read_text() == original


@pytest.mark.parametrize("change", ["candidate-env", "candidate-bundle", "old-restart"])
def test_prepared_drift_rejected_before_gate(tmp_path, monkeypatch, change):
    config, fake, path = prepared(tmp_path, monkeypatch)
    if change == "candidate-env":
        fake.containers["candidate"]["Config"]["Env"].append("CHANGED=1")
    elif change == "candidate-bundle":
        (tmp_path / "room-release.private.json").write_text("{}")
    else:
        fake.containers[fake.item["old_id"]]["State"]["StartedAt"] = "restarted"
    with pytest.raises(ValueError):
        cutover.switch(
            config,
            tmp_path,
            fake,
            checker=lambda *args: None,
            reload=lambda: None,
            drain=lambda spec: lambda: None,
            ready=lambda *args: None,
        )
    assert fake.events == [] and "return 503" not in path.read_text()


def test_rollback_failure_keeps_ingress_closed(tmp_path, monkeypatch):
    config, fake, path = prepared(tmp_path, monkeypatch)

    def ready(*args):
        raise ValueError("health unavailable")

    with pytest.raises(ValueError, match="Rollback readiness unconfirmed"):
        cutover.switch(
            config,
            tmp_path,
            fake,
            checker=lambda *args: None,
            reload=lambda: None,
            drain=lambda spec: lambda: None,
            ready=ready,
        )
    assert "return 503" in path.read_text()
    assert json.loads((tmp_path / "rollback.json").read_text())["errors"]


def test_prepare_wrong_sdk_never_creates_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(cutover.os, "chown", lambda *args: None)
    fake = FakeDocker()
    config = {"authorized_roles": ["room"], "containers": [fake.item]}
    fake.probe = lambda item, user: {
        "source_sha": item["source"],
        "sdk_sha": "0" * 40,
        "source_branch": "main",
        "sdk_files": item["sdk_files"],
        "owner": [10001, 10001],
    }
    with pytest.raises(ValueError, match="Candidate release"):
        cutover.prepare(config, tmp_path, fake)
    assert list(fake.containers) == [fake.item["old_id"]]


@pytest.mark.parametrize(
    "roles", [["room", "coordinator"], ["gateway"], ["room", "room"]]
)
def test_authorized_role_set_must_match_exactly(roles):
    _, item = fixture()
    with pytest.raises(ValueError, match="Authorized role"):
        cutover.validate_config({"authorized_roles": roles, "containers": [item]})


def test_bundle_hash_covers_only_identity_edits_and_retains_other_root_keys(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cutover.os, "chown", lambda *args: None)
    fake = FakeDocker()
    fake.bundle.update(
        activation={"keep": True},
        configuration_scope="existing",
        restart_scope=["room"],
    )
    fake.bundle["policy"].update(
        gpu_source_sha="other", media={"model": 736}, role_pool={"same": True}
    )
    before = copy.deepcopy(fake.bundle)
    cutover.prepare(
        {"authorized_roles": ["room"], "containers": [fake.item]}, tmp_path, fake
    )
    result = json.loads((tmp_path / "room-release.private.json").read_text())
    for key in ["activation", "configuration_scope", "restart_scope", "bindings"]:
        assert result[key] == before[key]
    for key in ["gpu_source_sha", "media", "role_pool"]:
        assert result["policy"][key] == before["policy"][key]


def test_two_role_gateway_failure_restores_both_originals(tmp_path, monkeypatch):
    monkeypatch.setattr(cutover.os, "chown", lambda *args: None)
    fake = FakeDocker()
    gateway = copy.deepcopy(fake.old)
    gateway.update(Id="9" * 64, Name="/old-gateway")
    gateway["Config"]["Env"] = [
        value.replace("WHITEBOX_ROLE=room", "WHITEBOX_ROLE=gateway")
        for value in gateway["Config"]["Env"]
    ]
    fake.containers[gateway["Id"]] = gateway
    fake.bundle["environments"]["gateway"] = copy.deepcopy(
        fake.bundle["environments"]["room"]
    )
    gw = {
        **fake.item,
        "role": "gateway",
        "old_id": gateway["Id"],
        "old_name": "old-gateway",
        "candidate_name": "new-gateway",
    }
    original_api = fake.api

    def create(method, path, body):
        result = original_api(method, path, body)
        if "new-gateway" in path:
            fake.containers["candidate-gateway"] = fake.containers.pop("candidate")
            fake.containers["candidate-gateway"]["Id"] = "candidate-gateway"
            return {"Id": "candidate-gateway"}
        return result

    fake.api = create
    path = tmp_path / "nginx.conf"
    path.write_text("server {\n marker\n}")
    config = {
        "authorized_roles": ["room", "gateway"],
        "containers": [fake.item, gw],
        "nginx": {
            "pid_file": "/run/nginx.pid",
            "path": str(path),
            "sha256": cutover.sha(path.read_bytes()),
            "marker": " marker",
            "paths": ["/v1/webrtc/play"],
        },
    }
    cutover.prepare(config, tmp_path, fake)
    # The fake API's temporary key must not overwrite the already created Room.
    proof = json.loads((tmp_path / "prepared.json").read_text())
    fake.containers["candidate"] = copy.deepcopy(proof["candidates"][0]["snapshot"])

    def ready(item, candidate):
        if item["role"] == "gateway" and item["source"] == gw["source"]:
            raise ValueError("gateway unhealthy")

    with pytest.raises(ValueError, match="gateway unhealthy"):
        cutover.switch(
            config,
            tmp_path,
            fake,
            checker=lambda *args: None,
            reload=lambda: None,
            drain=lambda spec: lambda: None,
            ready=ready,
        )
    assert fake.containers[fake.old["Id"]]["State"]["Running"]
    assert fake.containers[gateway["Id"]]["State"]["Running"]
    assert fake.events[-4:] == [
        ("stop", "candidate-gateway"),
        ("start", gateway["Id"]),
        ("stop", "candidate"),
        ("start", fake.old["Id"]),
    ]
    assert "return 503" not in path.read_text()


@pytest.mark.parametrize("kind", ["bridge", "anonymous-volume", "named-volume"])
def test_candidate_rejects_unsupported_network_or_volume(tmp_path, kind):
    old, item = fixture()
    if kind == "bridge":
        old["HostConfig"]["NetworkMode"] = "bridge"
    elif kind == "anonymous-volume":
        old["Config"]["Volumes"] = {"/data": {}}
    else:
        old["Mounts"] = [{"Type": "volume", "Name": "state", "Destination": "/data"}]
    with pytest.raises(ValueError):
        cutover.candidate_body(old, item, str(tmp_path / "release.json"), "newhash")


def test_pending_old_nginx_websocket_prevents_any_container_stop(tmp_path, monkeypatch):
    config, fake, path = prepared(tmp_path, monkeypatch)
    original = path.read_text()
    checks = []

    def drain(spec):
        def wait():
            assert "return 503" in path.read_text()
            raise ValueError("old WebSocket is pending")

        return wait

    with pytest.raises(ValueError, match="old WebSocket"):
        cutover.switch(
            config,
            tmp_path,
            fake,
            checker=lambda *args: checks.append("idle"),
            reload=lambda: None,
            drain=drain,
        )
    assert fake.events == [] and checks == ["idle"]
    assert path.read_text() == original


def test_nginx_drain_waits_old_worker_and_rejects_master_change(tmp_path, monkeypatch):
    pid = tmp_path / "nginx.pid"
    pid.write_text("11")
    monkeypatch.setattr(cutover, "nginx_master", lambda master: "master-start")
    observed = iter(
        [{"12": "original"}, {"12": "original", "13": "new"}, {"13": "new"}]
    )
    monkeypatch.setattr(cutover, "nginx_workers", lambda master: next(observed))
    monkeypatch.setattr(cutover.time, "sleep", lambda *args: None)
    cutover.nginx_drain({"pid_file": str(pid)})()
    monkeypatch.setattr(cutover, "nginx_workers", lambda master: {"12": "original"})
    wait = cutover.nginx_drain({"pid_file": str(pid)})
    pid.write_text("99")
    with pytest.raises(ValueError, match="master changed"):
        wait()


def test_rollback_receipt_write_failure_cannot_reopen_ingress(tmp_path, monkeypatch):
    config, fake, path = prepared(tmp_path, monkeypatch)
    original_write = cutover.write_private

    def write(destination, value):
        if destination.name == "rollback.json":
            raise OSError("disk full")
        original_write(destination, value)

    monkeypatch.setattr(cutover, "write_private", write)

    def ready(*args):
        raise ValueError("old and new unavailable")

    with pytest.raises(OSError):
        cutover.switch(
            config,
            tmp_path,
            fake,
            checker=lambda *args: None,
            reload=lambda: None,
            ready=ready,
            drain=lambda spec: lambda: None,
        )
    assert "return 503" in path.read_text()


def test_proc_worker_identity_uses_pid_and_start_time(tmp_path):
    proc = tmp_path / "proc"
    master = proc / "10"
    (master / "task" / "10").mkdir(parents=True)
    (master / "task" / "10" / "children").write_text("11 12")
    for pid, command in [("11", b"nginx: worker process\0"), ("12", b"unrelated\0")]:
        child = proc / pid
        child.mkdir()
        (child / "cmdline").write_bytes(command)
        (child / "stat").write_text(
            pid + " (worker name) S " + " ".join(["0"] * 18 + ["123456"] + ["0"] * 5)
        )
    assert cutover.nginx_workers(10, proc) == {"11": "123456"}
    (proc / "11" / "stat").write_text(
        "11 (worker name) S " + " ".join(["0"] * 18 + ["654321"] + ["0"] * 5)
    )
    assert cutover.nginx_workers(10, proc) == {"11": "654321"}


@pytest.mark.parametrize(
    "path", ["nginx.pid", "/tmp/not-nginx.pid", "/run/../tmp/nginx.pid"]
)
def test_pid_file_whitelist_rejects_unreviewed_paths(path):
    _, item = fixture()
    with pytest.raises(ValueError, match="approved absolute"):
        cutover.validate_config(
            {
                "containers": [item],
                "authorized_roles": ["room"],
                "nginx": {
                    "pid_file": path,
                    "path": "/etc/nginx/conf.d/site.conf",
                    "sha256": "1" * 64,
                },
            }
        )


def test_real_docker_api_uses_http_client_after_http_function_definition(monkeypatch):
    import http.client
    assert cutover.http_client is http.client
    monkeypatch.setattr(cutover.http_client, 'HTTPConnection', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('reached-client')))
    with pytest.raises(RuntimeError, match='reached-client'):
        cutover.Docker('/var/run/docker.sock').api('GET','/containers/x/json')
