import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT, load, render
from room_instance import plan


def inputs():
    b = json.loads((ROOT / "bindings.example.json").read_text())
    b["room_pool_compatibility"] = "test-v1"
    return render(load("shanghai"), b), dict(
        name="room-b",
        image="registry.test/room@sha256:" + "a" * 64,
        room_url="http://10.0.1.34:28092",
        public_ip="192.0.2.1",
        public_port=4444,
        listen_port=40001,
        state_dir="/var/lib/room-b",
        output_dir="/private/room-b",
        secret_file="/private/room-secrets.env",
        cpus=4,
        memory="8g",
    )


def test_new_room_plan_only_starts_new_container_with_private_data():
    bundle, args = inputs()
    before = json.dumps(bundle, sort_keys=True)
    value = plan(bundle, **args)
    assert json.dumps(bundle, sort_keys=True) == before
    command = value["docker_argv"]
    assert command[:3] == ["docker", "run", "--detach"]
    assert not {"stop", "restart", "rm", "kill"} & set(command)
    assert value["environment"]["WHITEBOX_ROOM_OWNER_ID"] == "room-b"
    assert value["environment"]["WHITEBOX_ROOM_PORT"] == "28092"
    assert value["environment"]["WHITEBOX_NATIVE_RTC_LISTEN"] == "0.0.0.0:40001"
    assert value["environment"]["WHITEBOX_NATIVE_RTC_PUBLIC_PORT"] == "4444"
    assert "--cpus" in command and "--memory" in command
    assert not any(
        "TOKEN" in k or "SECRET" in k or "PASSWORD" in k for k in value["environment"]
    )
    assert not value["existing_services_modified"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("image", "registry.test/room:latest"),
        ("room_url", "http://user:secret@room:8092"),
        ("listen_port", 28092),
        ("state_dir", "/var/lib/room,other"),
        ("name", "room; restart"),
        ("memory", "0g"),
    ],
)
def test_invalid_launch_identity_or_resources_rejected(field, value):
    bundle, args = inputs()
    args[field] = value
    with pytest.raises(ValueError):
        plan(bundle, **args)


def test_new_instance_bundle_passes_runtime_identity_validation(monkeypatch, tmp_path):
    from sglang.multimodal_gen.runtime.realtime.release_identity import release_identity

    bundle, args = inputs()
    value = plan(bundle, **args)
    manifest = tmp_path / "release.json"
    manifest.write_text(
        json.dumps(dict(source_sha="a" * 40, sdk_sha="b" * 40, source_branch="main"))
    )
    generated = tmp_path / "bundle.json"
    generated.write_text(json.dumps(value["bundle"]))
    for key, item in value["environment"].items():
        monkeypatch.setenv(key, item)
    monkeypatch.setenv("WHITEBOX_RELEASE_MANIFEST", str(manifest))
    import importlib

    module = importlib.import_module(
        "sglang.multimodal_gen.runtime.realtime.release_identity"
    )
    container_path = value["environment"]["WHITEBOX_RELEASE_BUNDLE"]
    monkeypatch.setattr(
        module, "Path", lambda path: generated if path == container_path else Path(path)
    )
    identity = release_identity("room")
    assert identity["identity_status"] == "configured"
    assert identity["config_sha256"] == value["bundle"]["config_sha256"]


def test_existing_role_pool_configuration_can_be_mounted_readonly():
    bundle, args = inputs()
    value = plan(
        bundle, **args, config_mounts=["/private/pool.json=/etc/whitebox/pool.json"]
    )
    assert (
        "type=bind,src=/private/pool.json,dst=/etc/whitebox/pool.json,readonly"
        in value["docker_argv"]
    )


@pytest.mark.parametrize(
    "target",
    [
        "/app/settings.json",
        "/opt/worldkit/registry.json",
        "/etc/whitebox/room-bundle.json",
        "/etc/whitebox/plugin.py",
        "/etc/../app/config.json",
    ],
)
def test_config_mounts_cannot_overlay_service_source_or_identity(target):
    bundle, args = inputs()
    with pytest.raises(ValueError):
        plan(bundle, **args, config_mounts=["/private/pool.json=" + target])
