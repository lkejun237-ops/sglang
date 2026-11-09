"""Source-stage contracts and opt-in checks against separately built images.

Stage inspection is deliberately not a substitute for running the native/SDK
image gates or launcher lifecycle fixtures. Supply exact candidate image refs
through WHITEBOX_RUNTIME_IMAGE / WHITEBOX_ROOM_IMAGE for payload/import checks.
"""

import os
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker/whitebox.Dockerfile"
TESTS = "python/sglang/multimodal_gen/test"


def stages():
    result = {}
    current = None
    for line in DOCKERFILE.read_text().replace("\\\n", " ").splitlines():
        words = shlex.split(line, comments=True)
        if not words:
            continue
        if words[0] == "FROM":
            current = words[3]
            result[current] = {"base": words[1], "instructions": []}
        elif current:
            result[current]["instructions"].append((words[0], words[1:]))
    return result


def copies(stage):
    return [
        args for instruction, args in stage["instructions"] if instruction == "COPY"
    ]


def test_development_permission_is_opt_in_and_never_inherited_by_runtime():
    graph = stages()
    for name in ("checks", "room-checks"):
        assert ("ARG", ["WHITEBOX_TEST_ALLOW_DEVELOPMENT_RELEASE=0"]) in graph[name][
            "instructions"
        ]
        assert (
            "ENV",
            [
                "WHITEBOX_ALLOW_DEVELOPMENT_RELEASE=$WHITEBOX_TEST_ALLOW_DEVELOPMENT_RELEASE"
            ],
        ) in graph[name]["instructions"]
    for target in ("room", "release"):
        name = target
        while name in graph:
            assert not any(
                op == "ENV"
                and any("WHITEBOX_ALLOW_DEVELOPMENT_RELEASE" in arg for arg in args)
                for op, args in graph[name]["instructions"]
            ), name
            name = graph[name]["base"]


def test_both_runtime_consumers_use_readonly_projected_source():
    graph = stages()
    projection = graph.get("runtime-source")
    assert projection is not None, (
        "Both broad runtime copies need a projection producer"
    )
    commands = [args for op, args in projection["instructions"] if op == "RUN"]
    assert any(
        "source=python/sglang" in " ".join(args) and ",readonly" in " ".join(args)
        for args in commands
    )
    for name in ("runtime", "room-runtime"):
        runtime_copies = [
            args for args in copies(graph[name]) if args[-1] == "/app/python/sglang"
        ]
        assert runtime_copies == [
            ["--from=runtime-source", "/runtime-source/sglang", "/app/python/sglang"]
        ]


def test_projection_stage_command_runs_on_actual_source_tree(tmp_path):
    graph = stages()
    commands = [
        args for op, args in graph["runtime-source"]["instructions"] if op == "RUN"
    ]
    assert len(commands) == 1
    mounts = {}
    command = []
    for argument in commands[0]:
        if argument.startswith("--mount="):
            options = dict(
                item.split("=", 1) for item in argument[8:].split(",") if "=" in item
            )
            mounts[options["target"]] = str(ROOT / options["source"])
        else:
            command.append(argument)
    assert command[:4] == ["test", "-n", "$WHITEBOX_SOURCE_SHA", "&&"]
    command = command[4:]
    assert command[0] == "python"
    command[0] = sys.executable
    command = [mounts.get(argument, argument) for argument in command]
    output = tmp_path / "projected"
    command[command.index("--destination") + 1] = str(output)
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    source = Path(command[command.index("--source") + 1])
    assert not (output / "multimodal_gen/test").exists()
    expected = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file() and not path.is_relative_to(source / "multimodal_gen/test")
    }
    actual = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert actual == expected
    for consumer in ("runtime", "room-runtime"):
        runtime_copy = next(
            args for args in copies(graph[consumer]) if args[-1] == "/app/python/sglang"
        )
        assert Path(runtime_copy[1]).name == "sglang"
        assert (
            output / "multimodal_gen/runtime/realtime/whitebox/cli_room.py"
        ).is_file()
        assert (
            output / "multimodal_gen/runtime/realtime/whitebox/worldkit_room.py"
        ).is_file()


@pytest.mark.parametrize("name", ["checks", "room-checks"])
def test_check_targets_restore_complete_test_tree_with_node_fixtures(name, tmp_path):
    graph = stages()
    # Execute the source restoration edge on real repository content, including
    # fixtures a Python-only test-file allowlist would silently omit.
    restoration = [args for args in copies(graph[name]) if args[-1] == f"/app/{TESTS}"]
    assert restoration == [[TESTS, f"/app/{TESTS}"]]
    source = ROOT / restoration[0][0]
    destination = tmp_path / "checks-tests"
    shutil.copytree(source, destination)
    expected = {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    actual = {
        p.relative_to(destination): p.read_bytes()
        for p in destination.rglob("*")
        if p.is_file()
    }
    assert actual == expected
    assert (
        destination / "unit/realtime/fixtures/capture-member-media-commands.mjs"
    ).is_file()


def test_release_edges_keep_existing_checks_native_tools_and_room_only_sdk():
    graph = stages()

    def dependencies(name):
        stage = graph[name]
        direct = {stage["base"]} | {
            arg[7:]
            for args in copies(stage)
            for arg in args
            if arg.startswith("--from=")
        }
        return direct | {
            nested for item in direct if item in graph for nested in dependencies(item)
        }

    assert "checks" in dependencies("release")
    assert "sdk" not in dependencies("release")
    assert {
        "checks",
        "room-checks",
        "sdk",
        "rtc-builder",
        "immediate-encoder-build",
        "room-process-init",
    } <= dependencies("room")
    assert [
        "--from=checks",
        "/tmp/whitebox-contracts-passed",
        "/app/whitebox-contracts-passed",
    ] in copies(graph["release"])
    assert [
        "--from=checks",
        "/tmp/whitebox-contracts-passed",
        "/app/whitebox-contracts-passed",
    ] in copies(graph["room-runtime"])
    for marker in ("whitebox-cli-passed", "whitebox-immediate-codec-passed"):
        assert ["--from=room-checks", f"/tmp/{marker}", f"/app/{marker}"] in copies(
            graph["room"]
        )
    commands = " ".join(
        " ".join(args) for op, args in graph["checks"]["instructions"] if op == "RUN"
    )
    assert "test_runtime_source_projection.py" in commands
    assert "test_runtime_image_contract.py" in commands
    assert "--confcutdir=deploy/whitebox" in commands
    # Existing test gates and their fixture environment must not be exchanged
    # for the new source-stage checks.
    required = (
        "test_whitebox.py",
        "test_whitebox_native.py",
        "test_realtime_coordinator_server.py",
        "test_realtime_gateway_server.py",
        "test_prepare_assets.py",
        "test_gpu_guard.py",
        "test_encoder_peer.py",
        "test_debian_main_mirror.py",
    )
    assert all(name in commands for name in required)
    room_commands = " ".join(
        " ".join(args)
        for op, args in graph["room-checks"]["instructions"]
        if op == "RUN"
    )
    for value in (
        "WHITEBOX_TEST_CLI_SDK=/opt/worldkit",
        "WHITEBOX_REQUIRE_NATIVE_X264=1",
        "test_whitebox_cli_integration.py",
        "test_whitebox_immediate_capture.py",
        "test_whitebox_capture_clock.py",
        "test_whitebox_stream_archive.py",
        "test_whitebox_independent_archive.py",
    ):
        assert value in room_commands


@pytest.mark.parametrize(
    "variable,room", [("WHITEBOX_RUNTIME_IMAGE", False), ("WHITEBOX_ROOM_IMAGE", True)]
)
def test_candidate_payload_and_dynamic_factories(variable, room):
    candidate = os.environ.get(variable)
    if not candidate:
        pytest.skip(f"{variable} is unset; actual image payload/import gate is pending")
    # No model/GPU or remote endpoints. This inspects the actual final image;
    # full local launcher/native-codec acceptance remains a separate gate.
    script = """
import importlib, pathlib, shutil
root = pathlib.Path('/app/python/sglang')
assert root.is_dir()
assert not (root / 'multimodal_gen/test').exists()
from sglang.multimodal_gen.runtime.realtime.whitebox.session import load_plugin
for name in (
    'sglang.multimodal_gen.runtime.realtime.whitebox.engine:WebSocketEngineFactory',
    'sglang.multimodal_gen.runtime.realtime.whitebox.browser_engine:factory',
    'sglang.multimodal_gen.runtime.realtime.whitebox.browser_engine:ui_factory',
    'sglang.multimodal_gen.runtime.realtime.whitebox.cli_room:cli_room',
    'sglang.multimodal_gen.runtime.realtime.whitebox.worldkit_room:worldkit_room',
    'sglang.multimodal_gen.runtime.realtime.whitebox.cli_ui:factory',
    'sglang.multimodal_gen.runtime.realtime.whitebox.browser_ui:factory',
    'sglang.multimodal_gen.runtime.realtime.whitebox.ui:RgbaOverlayRenderer',
):
    assert callable(load_plugin(name)), name
for module in ('realtime_coordinator_server', 'realtime_gateway_server', 'whitebox_room_server'):
    importlib.import_module('sglang.multimodal_gen.runtime.entrypoints.' + module)
importlib.import_module('sglang.multimodal_gen.runtime.realtime.whitebox.mock_worker')
assert pathlib.Path('/app/whitebox-contracts-passed').is_file()

# Exercise the configured factory, not only module imports: regional Gateway
# deployments enable shared Redis prefill even when Coordinator has a fallback.
import asyncio, os
from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
    HTTPCoordinatorClient, create_app,
)
from sglang.multimodal_gen.runtime.realtime.whitebox.prefill_gateway import RedisPrefillStore
os.environ['WHITEBOX_PREFILL_ENABLED'] = '1'
os.environ['WHITEBOX_PREFILL_REDIS_URL'] = 'redis://127.0.0.1:9/0'
os.environ['WHITEBOX_RTC_AUTH_SECRET'] = 'image-prefill-fixture-only-' * 3
async def configured_prefill():
    coordinator = HTTPCoordinatorClient('http://127.0.0.1:9')
    app = create_app(coordinator, model_revision='image-prefill-fixture',
                     vae_fingerprint='local', internal_output_url='ws://127.0.0.1:9')
    async with app.router.lifespan_context(app):
        manager = app.state.prefill_manager
        assert isinstance(manager.store, RedisPrefillStore)
        assert not manager.tasks and manager.uploads == 0
asyncio.run(configured_prefill())
"""
    if room:
        script += """
for name in ('ffmpeg', 'ffprobe', 'node', 'tini'):
    assert shutil.which(name), name
for name in ('whitebox-cli-passed', 'whitebox-immediate-codec-passed', 'whitebox-rtc', 'WHITEBOX_CODEC_VERSION'):
    assert pathlib.Path('/app', name).is_file(), name
assert pathlib.Path('/opt/worldkit/packages/cli/bin/worldkit.mjs').is_file()
assert (root / 'multimodal_gen/runtime/realtime/whitebox/native/libwhitebox_x264.so').is_file()
assert any(pathlib.Path('/opt/playwright').glob('chromium-*'))
"""
    container = "whitebox-projection-check-" + uuid.uuid4().hex
    try:
        completed = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--name",
                container,
                "--network=none",
                "--entrypoint",
                "python",
                candidate,
                "-c",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container], capture_output=True, timeout=30
        )
