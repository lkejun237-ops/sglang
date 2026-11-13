"""Source-stage contracts and opt-in checks against separately built images.

Stage inspection is deliberately not a substitute for running the native/SDK
image gates or launcher lifecycle fixtures. Supply exact candidate image refs
through WHITEBOX_RUNTIME_IMAGE / WHITEBOX_ROOM_IMAGE / WHITEBOX_MODEL_IMAGE
for payload/import checks.
"""

import base64
import contextlib
import hashlib
import http.server
import importlib.metadata
import json
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import threading
import urllib.request
import uuid
import venv
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker/whitebox.Dockerfile"
MODEL_DOCKERFILE = ROOT / "docker/whitebox-model.Dockerfile"
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


def test_model_dockerfile_enforces_prebuilt_kernels_without_compilation():
    if (
        not MODEL_DOCKERFILE.is_file()
        and DOCKERFILE.is_file()
        and ROOT == Path("/app")
        and not (ROOT / ".git").exists()
    ):
        pytest.skip(
            "whitebox-model.Dockerfile is excluded from CPU control-plane "
            "Docker build context"
        )
    assert MODEL_DOCKERFILE.is_file()
    lockfile_on_disk = ROOT / "deploy/whitebox/kernels-wheel.lock.txt"
    assert lockfile_on_disk.is_file()
    lock_text = lockfile_on_disk.read_text(encoding="utf-8")
    assert "seedleap-kernels==" in lock_text
    assert "--hash=sha256:" in lock_text

    instructions = []
    for line in MODEL_DOCKERFILE.read_text().replace("\\\n", " ").splitlines():
        words = shlex.split(line, comments=True)
        if words:
            instructions.append((words[0], words[1:]))

    arg_defs = [arg for op, args in instructions if op == "ARG" for arg in args]
    assert (
        "KERNELS_WHEEL_FIND_LINKS=https://zing-cdn.loopit.com.cn/releases/seedleap-kernels/simple/seedleap-kernels/index.html"
        in arg_defs
    )

    lock_copy_idx = next(
        (
            idx
            for idx, (op, args) in enumerate(instructions)
            if op == "COPY"
            and args
            == [
                "deploy/whitebox/kernels-wheel.lock.txt",
                "/opt/whitebox/deploy/whitebox/kernels-wheel.lock.txt",
            ]
        ),
        None,
    )
    assert lock_copy_idx is not None

    sglang_copy_idx = next(
        (
            idx
            for idx, (op, args) in enumerate(instructions)
            if op == "COPY" and args == ["python/sglang", "/opt/whitebox/python/sglang"]
        ),
        None,
    )
    assert sglang_copy_idx is not None
    assert lock_copy_idx < sglang_copy_idx

    kernels_pip_idx = next(
        (
            idx
            for idx, (op, args) in enumerate(instructions)
            if op == "RUN"
            and "pip install" in " ".join(args)
            and "/opt/whitebox/deploy/whitebox/kernels-wheel.lock.txt" in " ".join(args)
        ),
        None,
    )
    assert kernels_pip_idx is not None
    assert lock_copy_idx < kernels_pip_idx < sglang_copy_idx

    env_args = [arg for op, args in instructions if op == "ENV" for arg in args]
    assert "SEEDLEAP_KERNELS_DISABLE_JIT=1" in env_args
    assert "PYTHONPATH=/opt/whitebox/python" in env_args
    assert not any(".seedleap-kernels" in arg for arg in env_args)

    expected_gate = (
        "python /opt/whitebox/deploy/whitebox/verify_model_kernels.py "
        "--manifest /opt/whitebox/release.json "
        "--lockfile /opt/whitebox/deploy/whitebox/kernels-wheel.lock.txt"
    )
    run_commands = [" ".join(args) for op, args in instructions if op == "RUN"]
    assert any(expected_gate in cmd for cmd in run_commands)

    flattened = [" ".join([op, *args]) for op, args in instructions]
    for line in flattened:
        assert ".seedleap-kernels" not in line
        assert "WHITEBOX_KERNELS_SHA" not in line
        assert "nvcc" not in line.lower()

    pip_commands = [cmd for cmd in run_commands if "pip install" in cmd]
    assert len(pip_commands) == 2
    assert "/opt/whitebox" not in pip_commands[0]
    for required_token in (
        "--no-cache-dir",
        "--no-index",
        "--no-deps",
        "--force-reinstall",
        "--only-binary=:all:",
        "--require-hashes",
        "--find-links $KERNELS_WHEEL_FIND_LINKS",
        "-r /opt/whitebox/deploy/whitebox/kernels-wheel.lock.txt",
    ):
        assert required_token in pip_commands[1]


REQUIRED_KERNEL_OPS = (
    "fused_rmsnorm_indexed_scale_shift_quant_fp8",
    "fused_gated_residual_rmsnorm_indexed_scale_shift_quant_fp8",
    "fused_indexed_gated_residual_add",
    "fused_swiglu_quant_fp8",
    "block_sparse_attention",
    "fa4_attention",
    "fp8_scaled_mm",
)


def _install_mock_model_kernel_env(
    monkeypatch,
    tmp_path,
    *,
    disable_jit="1",
    dist_installed=True,
    wheel_purelib=False,
    pkg_under_whitebox=False,
    missing_op=None,
    prebuilt_ext=True,
    wheel_torch="2.13.0+cu128",
    runtime_torch="2.13.1+cu129",
    aot_cuda=True,
    git_sha="8bbc712caa54695778a36abcf2fc033c9c270845",
    version="0.2.0",
    missing_ext_symbol=None,
):
    if disable_jit is None:
        monkeypatch.delenv("SEEDLEAP_KERNELS_DISABLE_JIT", raising=False)
    else:
        monkeypatch.setenv("SEEDLEAP_KERNELS_DISABLE_JIT", disable_jit)

    wheel_meta = (
        "Wheel-Version: 1.0\n"
        "Generator: hatchling\n"
        f"Root-Is-Purelib: {'true' if wheel_purelib else 'false'}\n"
        f"Tag: {'py3-none-any' if wheel_purelib else 'cp311-cp311-linux_x86_64'}\n"
    )

    class FakeDist:
        def __init__(self):
            self.version = version

        def read_text(self, filename):
            if filename == "WHEEL":
                return wheel_meta
            return None

    real_distribution = importlib.metadata.distribution

    def fake_distribution(name):
        if name in ("seedleap-kernels", "seedleap_kernels"):
            if not dist_installed:
                raise importlib.metadata.PackageNotFoundError(name)
            return FakeDist()
        return real_distribution(name)

    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)

    if pkg_under_whitebox:
        pkg_init = Path(
            "/opt/whitebox/.seedleap-kernels/src/seedleap_kernels/__init__.py"
        )
    else:
        site_dir = tmp_path / "site-packages" / "seedleap_kernels"
        site_dir.mkdir(parents=True, exist_ok=True)
        pkg_init = site_dir / "__init__.py"
        pkg_init.write_text("", encoding="utf-8")

    sl_pkg = ModuleType("seedleap_kernels")
    sl_pkg.__file__ = str(pkg_init)

    sl_ops = ModuleType("seedleap_kernels.ops")
    for op_name in REQUIRED_KERNEL_OPS:
        if op_name != missing_op:
            setattr(sl_ops, op_name, lambda *a, **kw: None)

    sl_build_info = ModuleType("seedleap_kernels._build_info")
    sl_build_info.BUILD_INFO = {
        "aot_cuda": aot_cuda,
        "compiled_extensions": ["seedleap_kernels.backends.cuda._fp8_gemm_sm90_ext"],
        "cuda_archs": ["90a", "100a"],
        "cuda_version": "12.8",
        "git_sha": git_sha,
        "torch_version": wheel_torch,
    }

    ext_obj = SimpleNamespace(
        fp8_scaled_mm_sm90=lambda *a, **kw: None,
        fp8_scaled_mm_sm90_by_config=lambda *a, **kw: None,
    )
    if missing_ext_symbol is not None:
        delattr(ext_obj, missing_ext_symbol)

    sl_backends = ModuleType("seedleap_kernels.backends")
    sl_cuda = ModuleType("seedleap_kernels.backends.cuda")
    sl_sm90 = ModuleType("seedleap_kernels.backends.cuda.fp8_gemm_sm90")
    sl_sm90.has_prebuilt_cutlass_ext = lambda: prebuilt_ext
    sl_sm90._load_cutlass_ext = lambda: ext_obj

    sl_pkg.ops = sl_ops
    sl_pkg._build_info = sl_build_info
    sl_pkg.backends = sl_backends
    sl_backends.cuda = sl_cuda
    sl_cuda.fp8_gemm_sm90 = sl_sm90

    mock_torch = ModuleType("torch")
    mock_torch.__version__ = runtime_torch
    mock_torch.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: False)
    )

    monkeypatch.setitem(sys.modules, "seedleap_kernels", sl_pkg)
    monkeypatch.setitem(sys.modules, "seedleap_kernels.ops", sl_ops)
    monkeypatch.setitem(sys.modules, "seedleap_kernels._build_info", sl_build_info)
    monkeypatch.setitem(sys.modules, "seedleap_kernels.backends", sl_backends)
    monkeypatch.setitem(sys.modules, "seedleap_kernels.backends.cuda", sl_cuda)
    monkeypatch.setitem(
        sys.modules, "seedleap_kernels.backends.cuda.fp8_gemm_sm90", sl_sm90
    )
    monkeypatch.setitem(sys.modules, "torch", mock_torch)


def _import_verify_model_kernels():
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import verify_model_kernels

    return verify_model_kernels


def test_verify_model_kernels_accepts_prebuilt_wheel_and_preserves_release_identity(
    monkeypatch, tmp_path
):
    py_root = str(ROOT / "python")
    if py_root not in sys.path:
        monkeypatch.syspath_prepend(py_root)
    if "sglang" not in sys.modules:
        monkeypatch.setenv("SGLANG_LIGHTWEIGHT_RUNTIME", "1")
    from sglang.multimodal_gen.runtime.realtime.release_identity import release_identity

    vmk = _import_verify_model_kernels()
    expected_sha = "8bbc712caa54695778a36abcf2fc033c9c270845"
    _install_mock_model_kernel_env(
        monkeypatch,
        tmp_path,
        version="0.2.0",
        git_sha=expected_sha,
        wheel_torch="2.13.0+cu128",
        runtime_torch="2.13.2+cu129",
    )

    source_sha = "a" * 40
    sdk_sha = "b" * 40
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_branch": "main",
                "source_sha": source_sha,
                "sdk_sha": sdk_sha,
            }
        ),
        encoding="utf-8",
    )

    rc = vmk.main(["--manifest", str(manifest_path)])
    assert rc == 0

    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated == {
        "schema_version": 1,
        "source_branch": "main",
        "source_sha": source_sha,
        "sdk_sha": sdk_sha,
        "kernels_version": "0.2.0",
        "kernels_git_sha": expected_sha,
    }

    monkeypatch.setenv("WHITEBOX_RELEASE_MANIFEST", str(manifest_path))
    monkeypatch.delenv("WHITEBOX_RELEASE_CONFIG_SHA256", raising=False)
    monkeypatch.delenv("WHITEBOX_RELEASE_REGION", raising=False)
    monkeypatch.delenv("WHITEBOX_CHECKPOINT_SHA256", raising=False)
    monkeypatch.delenv("WHITEBOX_MODEL_CONFIG_SHA256", raising=False)
    identity = release_identity("model")
    assert identity == {
        "schema_version": 1,
        "role": "model",
        "source_branch": "main",
        "source_sha": source_sha,
        "sdk_sha": sdk_sha,
        "identity_status": "image_only",
    }


@pytest.mark.parametrize(
    "kwargs,err_pattern",
    [
        ({"disable_jit": None}, "SEEDLEAP_KERNELS_DISABLE_JIT"),
        ({"disable_jit": "0"}, "SEEDLEAP_KERNELS_DISABLE_JIT"),
        ({"dist_installed": False}, "not installed in MODEL_BASE_IMAGE"),
        ({"wheel_purelib": True}, "Root-Is-Purelib: false"),
        ({"pkg_under_whitebox": True}, "/opt/whitebox"),
        ({"missing_op": "fp8_scaled_mm"}, "fp8_scaled_mm"),
        ({"missing_op": "fused_swiglu_quant_fp8"}, "fused_swiglu_quant_fp8"),
        ({"prebuilt_ext": False}, "MODEL_BASE_IMAGE"),
        ({"aot_cuda": False}, "aot_cuda"),
        (
            {"wheel_torch": "2.12.1+cu128", "runtime_torch": "2.13.0+cu128"},
            "ABI mismatch",
        ),
        (
            {"missing_ext_symbol": "fp8_scaled_mm_sm90_by_config"},
            "fp8_scaled_mm_sm90_by_config",
        ),
        (
            {"manifest_missing": True},
            "Release manifest file does not exist",
        ),
    ],
)
def test_verify_model_kernels_fails_closed_on_contract_violations(
    monkeypatch, tmp_path, kwargs, err_pattern
):
    vmk = _import_verify_model_kernels()
    env_kwargs = dict(kwargs)
    manifest_missing = env_kwargs.pop("manifest_missing", False)
    _install_mock_model_kernel_env(monkeypatch, tmp_path, **env_kwargs)
    manifest_arg = tmp_path / "nonexistent_release.json" if manifest_missing else None
    with pytest.raises(RuntimeError, match=err_pattern):
        vmk.verify_model_kernels(manifest_arg)


def _build_wheel_archive(
    wheel_dir: Path,
    *,
    dist_name: str,
    version: str,
    tag: str,
    purelib: bool,
    files: dict[str, bytes],
) -> Path:
    norm_name = dist_name.replace("-", "_")
    dist_info = f"{norm_name}-{version}.dist-info"
    wheel_path = wheel_dir / f"{norm_name}-{version}-{tag}.whl"

    metadata_bytes = (
        f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: {version}\n"
    ).encode("utf-8")
    wheel_bytes = (
        "Wheel-Version: 1.0\n"
        "Generator: whitebox-contract-test\n"
        f"Root-Is-Purelib: {'true' if purelib else 'false'}\n"
        f"Tag: {tag}\n"
    ).encode("utf-8")

    all_entries = dict(files)
    all_entries[f"{dist_info}/METADATA"] = metadata_bytes
    all_entries[f"{dist_info}/WHEEL"] = wheel_bytes

    record_lines: list[str] = []
    for arcname in sorted(all_entries):
        payload = all_entries[arcname]
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        record_lines.append(f"{arcname},sha256={digest},{len(payload)}")
    record_lines.append(f"{dist_info}/RECORD,,")
    all_entries[f"{dist_info}/RECORD"] = ("\n".join(record_lines) + "\n").encode(
        "utf-8"
    )

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, payload in sorted(all_entries.items()):
            zf.writestr(arcname, payload)
    return wheel_path


def test_verify_model_kernels_loads_real_compiled_wheel_in_isolated_venv(tmp_path):
    compiler = next(
        (shutil.which(cmd) for cmd in ("cc", "clang", "gcc") if shutil.which(cmd)),
        None,
    )
    if compiler is None:
        pytest.skip("No host C compiler (cc/clang/gcc) available")

    py_include = sysconfig.get_paths().get("include")
    if not py_include or not (Path(py_include) / "Python.h").is_file():
        pytest.skip("Python.h development headers are not available")

    venv_dir = tmp_path / "isolated_venv"
    try:
        venv.EnvBuilder(with_pip=True, clear=True, symlinks=(os.name != "nt")).create(
            venv_dir
        )
    except Exception as exc:
        pytest.skip(f"Isolated venv with pip is unavailable: {exc}")

    bin_dir = venv_dir / ("Scripts" if sys.platform == "win32" else "bin")
    venv_python = bin_dir / ("python.exe" if sys.platform == "win32" else "python")
    if not venv_python.is_file():
        pytest.skip("Isolated venv python binary is missing")

    probe_script = (
        "import json, sysconfig\n"
        "try:\n"
        "    from pip._vendor.packaging.tags import sys_tags\n"
        "except ImportError:\n"
        "    from packaging.tags import sys_tags\n"
        "t = next(iter(sys_tags()))\n"
        "print(json.dumps({\n"
        "    'tag': f'{t.interpreter}-{t.abi}-{t.platform}',\n"
        "    'ext_suffix': sysconfig.get_config_var('EXT_SUFFIX') or '.so',\n"
        "    'include': sysconfig.get_paths().get('include', ''),\n"
        "}))\n"
    )
    probe_proc = subprocess.run(
        [str(venv_python), "-I", "-c", probe_script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe_proc.returncode != 0:
        pytest.skip(
            f"Unable to query wheel tag from isolated venv: {probe_proc.stderr}"
        )
    venv_meta = json.loads(probe_proc.stdout.strip())
    wheel_tag = venv_meta["tag"]
    ext_suffix = venv_meta["ext_suffix"]
    inc_dir = (
        venv_meta["include"]
        if venv_meta["include"] and (Path(venv_meta["include"]) / "Python.h").is_file()
        else py_include
    )

    c_src = tmp_path / "_fp8_gemm_sm90_ext.c"
    c_src.write_text(
        "#define PY_SSIZE_T_CLEAN\n"
        "#include <Python.h>\n\n"
        "static PyObject* fp8_scaled_mm_sm90(\n"
        "    PyObject* self, PyObject* args, PyObject* kwargs\n"
        ") {\n"
        "    (void)self; (void)args; (void)kwargs;\n"
        "    Py_RETURN_NONE;\n"
        "}\n\n"
        "static PyObject* fp8_scaled_mm_sm90_by_config(\n"
        "    PyObject* self, PyObject* args, PyObject* kwargs\n"
        ") {\n"
        "    (void)self; (void)args; (void)kwargs;\n"
        "    Py_RETURN_NONE;\n"
        "}\n\n"
        "static PyMethodDef ExtMethods[] = {\n"
        '    {"fp8_scaled_mm_sm90", (PyCFunction)fp8_scaled_mm_sm90,\n'
        '     METH_VARARGS | METH_KEYWORDS, "SM90 FP8 GEMM native entry"},\n'
        '    {"fp8_scaled_mm_sm90_by_config",\n'
        "     (PyCFunction)fp8_scaled_mm_sm90_by_config,\n"
        '     METH_VARARGS | METH_KEYWORDS, "SM90 FP8 GEMM config native entry"},\n'
        "    {NULL, NULL, 0, NULL}\n"
        "};\n\n"
        "static struct PyModuleDef ext_module = {\n"
        "    PyModuleDef_HEAD_INIT,\n"
        '    "_fp8_gemm_sm90_ext",\n'
        '    "Compiled native SM90 FP8 extension",\n'
        "    -1,\n"
        "    ExtMethods\n"
        "};\n\n"
        "PyMODINIT_FUNC PyInit__fp8_gemm_sm90_ext(void) {\n"
        "    return PyModule_Create(&ext_module);\n"
        "}\n",
        encoding="utf-8",
    )

    so_file = tmp_path / f"_fp8_gemm_sm90_ext{ext_suffix}"
    if sys.platform == "darwin":
        cc_cmd = [
            compiler,
            "-O2",
            f"-I{inc_dir}",
            "-bundle",
            "-undefined",
            "dynamic_lookup",
            str(c_src),
            "-o",
            str(so_file),
        ]
    else:
        cc_cmd = [
            compiler,
            "-O2",
            "-fPIC",
            "-shared",
            f"-I{inc_dir}",
            str(c_src),
            "-o",
            str(so_file),
        ]
    subprocess.run(cc_cmd, check=True, timeout=30)
    assert so_file.is_file() and so_file.stat().st_size > 0

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(parents=True, exist_ok=True)

    torch_whl = _build_wheel_archive(
        wheelhouse,
        dist_name="torch",
        version="2.13.1",
        tag="py3-none-any",
        purelib=True,
        files={
            "torch/__init__.py": b'__version__ = "2.13.1+cu129"\n',
        },
    )

    expected_sha = "8bbc712caa54695778a36abcf2fc033c9c270845"
    ops_lines = [
        f"def {op}(*args, **kwargs):\n    return None\n" for op in REQUIRED_KERNEL_OPS
    ]
    sm90_py = (
        "import importlib\n"
        "import importlib.util\n"
        "import os\n\n"
        'PREBUILT_EXT_MODULE = "seedleap_kernels.backends.cuda._fp8_gemm_sm90_ext"\n\n'
        "def has_prebuilt_cutlass_ext() -> bool:\n"
        "    return importlib.util.find_spec(PREBUILT_EXT_MODULE) is not None\n\n"
        "def _load_cutlass_ext():\n"
        "    if has_prebuilt_cutlass_ext():\n"
        "        return importlib.import_module(PREBUILT_EXT_MODULE)\n"
        "    if os.environ.get('SEEDLEAP_KERNELS_DISABLE_JIT') == '1':\n"
        "        raise RuntimeError('JIT disabled')\n"
        "    raise RuntimeError('Missing prebuilt extension')\n"
    )
    build_info_py = (
        "BUILD_INFO = {\n"
        "    'aot_cuda': True,\n"
        "    'compiled_extensions': [\n"
        "        'seedleap_kernels.backends.cuda._fp8_gemm_sm90_ext'\n"
        "    ],\n"
        "    'cuda_archs': ['90a', '100a'],\n"
        "    'cuda_version': '12.8',\n"
        f"    'git_sha': {expected_sha!r},\n"
        "    'torch_version': '2.13.0+cu128',\n"
        "}\n"
    )

    kernels_whl = _build_wheel_archive(
        wheelhouse,
        dist_name="seedleap-kernels",
        version="0.2.0",
        tag=wheel_tag,
        purelib=False,
        files={
            "seedleap_kernels/__init__.py": b'__version__ = "0.2.0"\n',
            "seedleap_kernels/_build_info.py": build_info_py.encode("utf-8"),
            "seedleap_kernels/ops/__init__.py": ("\n".join(ops_lines)).encode("utf-8"),
            "seedleap_kernels/backends/__init__.py": b"",
            "seedleap_kernels/backends/cuda/__init__.py": b"",
            "seedleap_kernels/backends/cuda/fp8_gemm_sm90.py": sm90_py.encode("utf-8"),
            f"seedleap_kernels/backends/cuda/{so_file.name}": so_file.read_bytes(),
        },
    )

    clean_env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "PYTHONPATH",
            "PYTHONHOME",
            "PIP_REQUIRE_VIRTUALENV",
            "PIP_CONFIG_FILE",
            "CUTLASS_DIR",
            "SEEDLEAP_CUTLASS_INCLUDE",
            "SGLANG_CSRC_DIR",
        }
    }
    clean_env["SEEDLEAP_KERNELS_DISABLE_JIT"] = "1"

    stale_sha = "0" * 40
    stale_build_info_py = build_info_py.replace(expected_sha, stale_sha)
    old_wheelhouse = tmp_path / "old_wheelhouse"
    old_wheelhouse.mkdir(parents=True, exist_ok=True)
    stale_kernels_whl = _build_wheel_archive(
        old_wheelhouse,
        dist_name="seedleap-kernels",
        version="0.2.0",
        tag=wheel_tag,
        purelib=False,
        files={
            "seedleap_kernels/__init__.py": b'__version__ = "0.2.0"\n',
            "seedleap_kernels/_build_info.py": stale_build_info_py.encode("utf-8"),
            "seedleap_kernels/ops/__init__.py": ("\n".join(ops_lines)).encode("utf-8"),
            "seedleap_kernels/backends/__init__.py": b"",
            "seedleap_kernels/backends/cuda/__init__.py": b"",
            "seedleap_kernels/backends/cuda/fp8_gemm_sm90.py": sm90_py.encode("utf-8"),
            f"seedleap_kernels/backends/cuda/{so_file.name}": so_file.read_bytes(),
        },
    )

    subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--only-binary=:all:",
            str(torch_whl),
            str(stale_kernels_whl),
        ],
        env=clean_env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    bump_script = ROOT / "deploy/whitebox/bump_kernels_wheel.py"
    lockfile_path = tmp_path / "kernels-wheel.lock.txt"
    bump_proc = subprocess.run(
        [
            str(venv_python),
            "-I",
            str(bump_script),
            "--wheel-file",
            str(kernels_whl),
            "--lockfile",
            str(lockfile_path),
        ],
        env=clean_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert bump_proc.returncode == 0, bump_proc.stdout + bump_proc.stderr

    # Simulate OSS REST static object layout where trailing-slash directories
    # return 404 NoSuchKey and only explicit index.html / wheel objects exist.
    oss_root = tmp_path / "oss_static"
    oss_wheels_dir = oss_root / "releases" / "seedleap-kernels" / "wheels"
    oss_simple_dir = (
        oss_root / "releases" / "seedleap-kernels" / "simple" / "seedleap-kernels"
    )
    oss_wheels_dir.mkdir(parents=True, exist_ok=True)
    oss_simple_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(kernels_whl, oss_wheels_dir / kernels_whl.name)
    kernels_whl_sha = hashlib.sha256(kernels_whl.read_bytes()).hexdigest()
    (oss_simple_dir / "index.html").write_text(
        "<!DOCTYPE html>\n<html><body>\n"
        f'<a href="../../wheels/{kernels_whl.name}#sha256={kernels_whl_sha}">'
        f"{kernels_whl.name}</a>\n"
        "</body></html>\n",
        encoding="utf-8",
    )

    class StrictOssObjectHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            rel = self.path.split("?", 1)[0].split("#", 1)[0]
            if rel.endswith("/") or ".." in rel:
                self.send_error(404, "NoSuchKey")
                return
            target = (oss_root / rel.lstrip("/")).resolve()
            if not target.is_relative_to(oss_root.resolve()) or not target.is_file():
                self.send_error(404, "NoSuchKey")
                return
            data = target.read_bytes()
            ctype = (
                "text/html; charset=utf-8"
                if target.suffix == ".html"
                else "application/octet-stream"
            )
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):  # noqa: A002
            pass

    oss_server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), StrictOssObjectHandler
    )
    oss_thread = threading.Thread(target=oss_server.serve_forever, daemon=True)
    oss_thread.start()
    find_links_url = (
        f"http://127.0.0.1:{oss_server.server_port}"
        "/releases/seedleap-kernels/simple/seedleap-kernels/index.html"
    )
    try:
        bad_hash_lock = tmp_path / "bad-hash.lock.txt"
        bad_hash_lock.write_text(
            "seedleap-kernels==0.2.0 \\\n    --hash=sha256:" + ("f" * 64) + "\n",
            encoding="utf-8",
        )
        bad_hash_proc = subprocess.run(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-index",
                "--no-deps",
                "--force-reinstall",
                "--only-binary=:all:",
                "--require-hashes",
                "--trusted-host",
                "127.0.0.1",
                "--find-links",
                find_links_url,
                "-r",
                str(bad_hash_lock),
            ],
            env=clean_env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert bad_hash_proc.returncode != 0

        subprocess.run(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-index",
                "--no-deps",
                "--force-reinstall",
                "--only-binary=:all:",
                "--require-hashes",
                "--trusted-host",
                "127.0.0.1",
                "--find-links",
                find_links_url,
                "-r",
                str(lockfile_path),
            ],
            env=clean_env,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        oss_server.shutdown()
        oss_server.server_close()

    run_cwd = tmp_path / "empty_cwd"
    run_cwd.mkdir(parents=True, exist_ok=True)

    source_sha = "c" * 40
    sdk_sha = "d" * 40
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_branch": "main",
                "source_sha": source_sha,
                "sdk_sha": sdk_sha,
            }
        ),
        encoding="utf-8",
    )

    verify_script = ROOT / "deploy/whitebox/verify_model_kernels.py"
    completed = subprocess.run(
        [
            str(venv_python),
            "-I",
            str(verify_script),
            "--manifest",
            str(manifest_path),
            "--lockfile",
            str(lockfile_path),
        ],
        cwd=str(run_cwd),
        env=clean_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    summary = json.loads(completed.stdout.strip())
    assert summary["aot_cuda"] is True
    assert summary["ext_loaded"] is True
    assert summary["kernels_version"] == "0.2.0"
    assert summary["kernels_git_sha"] == expected_sha

    updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated_manifest == {
        "schema_version": 1,
        "source_branch": "main",
        "source_sha": source_sha,
        "sdk_sha": sdk_sha,
        "kernels_version": "0.2.0",
        "kernels_git_sha": expected_sha,
    }

    wrong_lockfile = tmp_path / "wrong-kernels-wheel.lock.txt"
    wrong_lockfile.write_text(
        "seedleap-kernels==0.1.9 \\\n    --hash=sha256:" + ("a" * 64) + "\n",
        encoding="utf-8",
    )
    mismatch_proc = subprocess.run(
        [
            str(venv_python),
            "-I",
            str(verify_script),
            "--lockfile",
            str(wrong_lockfile),
        ],
        cwd=str(run_cwd),
        env=clean_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert mismatch_proc.returncode != 0
    assert "does not match locked version" in mismatch_proc.stderr

    missing_manifest = tmp_path / "does_not_exist_release.json"
    missing_proc = subprocess.run(
        [
            str(venv_python),
            "-I",
            str(verify_script),
            "--manifest",
            str(missing_manifest),
        ],
        cwd=str(run_cwd),
        env=clean_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert missing_proc.returncode != 0
    assert "Release manifest file does not exist" in missing_proc.stderr


def test_verify_model_kernels_lockfile_version_check(monkeypatch, tmp_path):
    vmk = _import_verify_model_kernels()
    _install_mock_model_kernel_env(monkeypatch, tmp_path, version="0.1.0.dev3")

    # Backward compatibility when --lockfile is omitted.
    assert vmk.main([]) == 0

    lockfile = tmp_path / "kernels-wheel.lock.txt"
    lockfile.write_text(
        "# Managed by deploy/whitebox/bump_kernels_wheel.py (prebuilt binary CUDA wheel only).\n"
        "seedleap-kernels==0.1.0.dev3 \\\n"
        "    --hash=sha256:" + ("b" * 64) + "\n",
        encoding="utf-8",
    )
    assert vmk.main(["--lockfile", str(lockfile)]) == 0

    mismatched_lock = tmp_path / "mismatch.lock.txt"
    mismatched_lock.write_text(
        "seedleap-kernels==0.1.0.dev0 \\\n    --hash=sha256:" + ("b" * 64) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not match locked version"):
        vmk.main(["--lockfile", str(mismatched_lock)])

    missing_lock = tmp_path / "missing.lock.txt"
    with pytest.raises(RuntimeError, match="lockfile does not exist"):
        vmk.main(["--lockfile", str(missing_lock)])

    unhashed_lock = tmp_path / "unhashed.lock.txt"
    unhashed_lock.write_text("seedleap-kernels==0.1.0.dev3\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid kernels wheel lockfile"):
        vmk.main(["--lockfile", str(unhashed_lock)])


def _import_bump_kernels_wheel():
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import bump_kernels_wheel

    return bump_kernels_wheel


def _make_sample_kernels_wheel(
    wheel_dir: Path,
    *,
    version: str = "0.1.0.dev3",
    tag: str = "cp312-cp312-linux_x86_64",
    purelib: bool = False,
    include_so: bool = True,
    git_sha: str = "c9bb3c89a08cd37f0649035e0e14f89ad4137eba",
    aot_cuda: bool = True,
) -> Path:
    build_info_py = (
        "BUILD_INFO = {\n"
        f"    'aot_cuda': {aot_cuda!r},\n"
        "    'compiled_extensions': [\n"
        "        'seedleap_kernels.backends.cuda._fp8_gemm_sm90_ext'\n"
        "    ],\n"
        "    'cuda_archs': ['90a'],\n"
        "    'cuda_version': '13.0',\n"
        f"    'git_sha': {git_sha!r},\n"
        "    'torch_version': '2.13.0+cu130',\n"
        "}\n"
    ).encode("utf-8")
    files: dict[str, bytes] = {
        "seedleap_kernels/__init__.py": f'__version__ = "{version}"\n'.encode("utf-8"),
        "seedleap_kernels/_build_info.py": build_info_py,
    }
    if include_so:
        files[
            "seedleap_kernels/backends/cuda/_fp8_gemm_sm90_ext.cpython-312-x86_64-linux-gnu.so"
        ] = b"\x7fELF-mock-sm90-cutlass-binary"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    return _build_wheel_archive(
        wheel_dir,
        dist_name="seedleap-kernels",
        version=version,
        tag=tag,
        purelib=purelib,
        files=files,
    )


@contextlib.contextmanager
def _serve_directory_over_mock_https(monkeypatch, bkw, root_dir: Path, host: str):
    class QuietDirectoryHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root_dir), **kwargs)

        def log_message(self, format, *args):  # noqa: A002
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), QuietDirectoryHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def loopback_open(url: str, *, allowed_hosts: set[str], timeout: float):
        bkw.validate_https_url(url)
        from urllib.parse import urlsplit

        parsed = urlsplit(url)
        assert parsed.hostname is not None
        if parsed.hostname.lower() not in allowed_hosts:
            raise ValueError(f"Host {parsed.hostname!r} is not in {allowed_hosts}")
        if parsed.hostname.lower() != host.lower():
            raise urllib.error.URLError(f"Unroutable host {parsed.hostname}")
        local_url = f"http://127.0.0.1:{server.server_port}{parsed.path}"
        return urllib.request.urlopen(local_url, timeout=timeout)

    monkeypatch.setattr(bkw, "_open_https_stream", loopback_open)
    try:
        yield f"https://{host}"
    finally:
        server.shutdown()
        server.server_close()


def test_bump_kernels_wheel_remote_upgrade_and_idempotency(
    monkeypatch, tmp_path, capsys
):
    bkw = _import_bump_kernels_wheel()
    cdn_root = tmp_path / "cdn" / "releases" / "seedleap-kernels"
    wheels_dir = cdn_root / "wheels"
    wheels_dir.mkdir(parents=True, exist_ok=True)

    git_sha = "11223344556677889900aabbccddeeff00112233"
    whl_path = _make_sample_kernels_wheel(
        wheels_dir, version="0.1.0.dev3", git_sha=git_sha
    )
    whl_sha256 = hashlib.sha256(whl_path.read_bytes()).hexdigest()

    latest_payload = {
        "version": "0.1.0.dev3",
        "filename": whl_path.name,
        "url": f"https://zing-cdn.loopit.com.cn/releases/seedleap-kernels/wheels/{whl_path.name}",
        "sha256": whl_sha256,
        "git_sha": git_sha,
    }
    (cdn_root / "latest.json").write_text(
        json.dumps(latest_payload, indent=2), encoding="utf-8"
    )

    lockfile = tmp_path / "kernels-wheel.lock.txt"
    initial_sha = "bbd9a326f2ec928457e770ddf189b29f5f8c233e98bce042d74b65c5401db97b"
    lockfile.write_text(
        "# Managed by deploy/whitebox/bump_kernels_wheel.py (prebuilt binary CUDA wheel only).\n"
        f"seedleap-kernels==0.1.0.dev0 \\\n    --hash=sha256:{initial_sha}\n",
        encoding="utf-8",
    )

    with _serve_directory_over_mock_https(
        monkeypatch, bkw, tmp_path / "cdn", "zing-cdn.loopit.com.cn"
    ) as base_url:
        latest_url = f"{base_url}/releases/seedleap-kernels/latest.json"
        rc = bkw.main(["--latest-url", latest_url, "--lockfile", str(lockfile)])
        assert rc == 0
        summary = json.loads(capsys.readouterr().out.strip())
        assert summary == {
            "changed": True,
            "git_sha": git_sha,
            "new_version": "0.1.0.dev3",
            "old_version": "0.1.0.dev0",
            "sha256": whl_sha256,
        }
        expected_lock = (
            "# Managed by deploy/whitebox/bump_kernels_wheel.py (prebuilt binary CUDA wheel only).\n"
            f"seedleap-kernels==0.1.0.dev3 \\\n    --hash=sha256:{whl_sha256}\n"
        )
        assert lockfile.read_text(encoding="utf-8") == expected_lock

        # Idempotent re-run must report changed=False and leave lockfile untouched.
        rc2 = bkw.main(["--latest-url", latest_url, "--lockfile", str(lockfile)])
        assert rc2 == 0
        summary2 = json.loads(capsys.readouterr().out.strip())
        assert summary2 == {
            "changed": False,
            "git_sha": git_sha,
            "new_version": "0.1.0.dev3",
            "old_version": "0.1.0.dev3",
            "sha256": whl_sha256,
        }
        assert lockfile.read_text(encoding="utf-8") == expected_lock


def test_bump_kernels_wheel_local_file_mode(tmp_path, capsys):
    bkw = _import_bump_kernels_wheel()
    git_sha = "deadbeefcafebabe1234567890abcdef12345678"
    whl_path = _make_sample_kernels_wheel(
        tmp_path, version="0.1.0.dev5", git_sha=git_sha
    )
    whl_sha256 = hashlib.sha256(whl_path.read_bytes()).hexdigest()
    lockfile = tmp_path / "new-lock.txt"

    rc = bkw.main(["--wheel-file", str(whl_path), "--lockfile", str(lockfile)])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out.strip())
    assert summary == {
        "changed": True,
        "git_sha": git_sha,
        "new_version": "0.1.0.dev5",
        "old_version": None,
        "sha256": whl_sha256,
    }
    assert (
        f"seedleap-kernels==0.1.0.dev5 \\\n    --hash=sha256:{whl_sha256}\n"
        in lockfile.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://zing-cdn.loopit.com.cn/releases/seedleap-kernels/latest.json",
        "https://user:secret@zing-cdn.loopit.com.cn/releases/seedleap-kernels/latest.json",
        "https://zing-cdn.loopit.com.cn/releases/seedleap-kernels/latest.json?token=abc",
        "https://zing-cdn.loopit.com.cn/releases/seedleap-kernels/latest.json#frag",
        "https://zing-cdn.loopit.com.cn/releases/seedleap-kernels/\nlatest.json",
        "https://localhost/releases/seedleap-kernels/latest.json",
        "https://127.0.0.1/releases/seedleap-kernels/latest.json",
        "https://169.254.169.254/latest/meta-data/",
        "https://[::1]/releases/seedleap-kernels/latest.json",
        "https://zing-cdn.loopit.com.cn:8443/releases/seedleap-kernels/latest.json",
    ],
)
def test_bump_kernels_wheel_rejects_unsafe_latest_urls(tmp_path, bad_url):
    bkw = _import_bump_kernels_wheel()
    lockfile = tmp_path / "kernels-wheel.lock.txt"
    original = "seedleap-kernels==0.1.0.dev0 \\\n    --hash=sha256:" + ("a" * 64) + "\n"
    lockfile.write_text(original, encoding="utf-8")
    with pytest.raises((SystemExit, ValueError)):
        bkw.main(["--latest-url", bad_url, "--lockfile", str(lockfile)])
    assert lockfile.read_text(encoding="utf-8") == original


def test_bump_kernels_wheel_forbids_no_verify_download_and_invalid_allow_hosts(
    tmp_path,
):
    bkw = _import_bump_kernels_wheel()
    lockfile = tmp_path / "kernels-wheel.lock.txt"
    with pytest.raises(SystemExit) as exc:
        bkw.main(["--no-verify-download", "--lockfile", str(lockfile)])
    assert exc.value.code == 2

    for bad_host in ("localhost", "127.0.0.1", "169.254.169.254", "[::1]", "intranet"):
        with pytest.raises((SystemExit, ValueError)):
            bkw.main(
                [
                    "--allow-download-host",
                    bad_host,
                    "--lockfile",
                    str(lockfile),
                ]
            )


def test_bump_kernels_wheel_archive_and_filename_hardening(monkeypatch, tmp_path):
    bkw = _import_bump_kernels_wheel()

    # 1. Reject extra '-' components (> 6) or compound 'any' platform tags
    for bad_name in (
        "seedleap_kernels-0.1.0.dev3-1-cp312-cp312-linux_x86_64-extra.whl",
        "seedleap_kernels-0.1.0.dev3-cp312-cp312-linux_x86_64.any.whl",
        "seedleap_kernels-0.1.0.dev3-cp312-cp312-any.linux_x86_64.whl",
    ):
        with pytest.raises(ValueError):
            bkw.validate_wheel_filename(bad_name, expected_version="0.1.0.dev3")

    base_whl = _make_sample_kernels_wheel(tmp_path / "good", version="0.1.0.dev3")

    # 2. Reject oversized local --wheel-file
    monkeypatch.setattr(bkw, "MAX_WHEEL_BYTES", 64)
    with pytest.raises(ValueError, match="maximum size"):
        bkw.inspect_binary_wheel_file(base_whl)
    monkeypatch.undo()

    # 3. Reject archive path traversal, absolute path, backslash, or symlink entry
    for bad_arcname, is_symlink in (
        ("../escape.py", False),
        ("/etc/passwd", False),
        ("seedleap_kernels\\bad.py", False),
        ("seedleap_kernels/link.so", True),
    ):
        forged_whl = (
            tmp_path
            / "forged"
            / "seedleap_kernels-0.1.0.dev3-cp312-cp312-linux_x86_64.whl"
        )
        forged_whl.parent.mkdir(parents=True, exist_ok=True)
        with (
            zipfile.ZipFile(base_whl, "r") as src,
            zipfile.ZipFile(forged_whl, "w") as dst,
        ):
            for info in src.infolist():
                dst.writestr(info, src.read(info.filename))
            bad_info = zipfile.ZipInfo(bad_arcname)
            if is_symlink:
                bad_info.external_attr = 0o120777 << 16
            dst.writestr(bad_info, b"payload")
        with pytest.raises(ValueError):
            bkw.inspect_binary_wheel_file(forged_whl)

    # 4. Reject mismatched dist-info directory name
    wrong_dist_whl = (
        tmp_path
        / "wrong_dist"
        / "seedleap_kernels-0.1.0.dev3-cp312-cp312-linux_x86_64.whl"
    )
    wrong_dist_whl.parent.mkdir(parents=True, exist_ok=True)
    with (
        zipfile.ZipFile(base_whl, "r") as src,
        zipfile.ZipFile(wrong_dist_whl, "w") as dst,
    ):
        for info in src.infolist():
            renamed = info.filename.replace(
                "seedleap_kernels-0.1.0.dev3.dist-info/",
                "other_pkg-0.1.0.dev3.dist-info/",
            )
            dst.writestr(renamed, src.read(info.filename))
    with pytest.raises(ValueError, match="dist-info"):
        bkw.inspect_binary_wheel_file(wrong_dist_whl)

    # 5. Reject cumulative uncompressed size exceeding MAX_UNCOMPRESSED_WHEEL_BYTES
    monkeypatch.setattr(bkw, "MAX_UNCOMPRESSED_WHEEL_BYTES", 32)
    with pytest.raises(ValueError, match="uncompressed"):
        bkw.inspect_binary_wheel_file(base_whl)


def test_bump_kernels_wheel_rejects_unsafe_redirects():
    bkw = _import_bump_kernels_wheel()
    handler = bkw._SafeHTTPSRedirectHandler({"zing-cdn.loopit.com.cn"})
    req = urllib.request.Request(
        "https://zing-cdn.loopit.com.cn/releases/seedleap-kernels/latest.json"
    )
    with pytest.raises(ValueError, match="HTTPS"):
        handler.redirect_request(
            req,
            None,
            302,
            "Found",
            {},
            "http://zing-cdn.loopit.com.cn/releases/seedleap-kernels/latest.json",
        )
    with pytest.raises(ValueError, match="host"):
        handler.redirect_request(
            req,
            None,
            302,
            "Found",
            {},
            "https://evil.example/releases/seedleap-kernels/latest.json",
        )


def test_bump_kernels_wheel_rejects_forged_or_corrupted_wheels(monkeypatch, tmp_path):
    bkw = _import_bump_kernels_wheel()
    cdn_root = tmp_path / "cdn" / "releases" / "seedleap-kernels"
    wheels_dir = cdn_root / "wheels"
    wheels_dir.mkdir(parents=True, exist_ok=True)

    lockfile = tmp_path / "kernels-wheel.lock.txt"
    original = (
        "# Managed by deploy/whitebox/bump_kernels_wheel.py (prebuilt binary CUDA wheel only).\n"
        "seedleap-kernels==0.1.0.dev0 \\\n"
        "    --hash=sha256:bbd9a326f2ec928457e770ddf189b29f5f8c233e98bce042d74b65c5401db97b\n"
    )
    lockfile.write_text(original, encoding="utf-8")

    valid_whl = _make_sample_kernels_wheel(wheels_dir, version="0.1.0.dev4")
    valid_sha = hashlib.sha256(valid_whl.read_bytes()).hexdigest()

    pure_whl = _make_sample_kernels_wheel(
        wheels_dir, version="0.1.0.dev4", tag="py3-none-any", purelib=True
    )
    pure_sha = hashlib.sha256(pure_whl.read_bytes()).hexdigest()

    fake_pure_platform_whl = _make_sample_kernels_wheel(
        tmp_path / "fake_pure",
        version="0.1.0.dev4",
        tag="cp312-cp312-linux_x86_64",
        purelib=True,
    )
    shutil.copy2(fake_pure_platform_whl, wheels_dir / "fake_pure.whl")

    missing_so_dir = tmp_path / "no_so"
    missing_so_dir.mkdir()
    missing_so_whl = _make_sample_kernels_wheel(
        missing_so_dir, version="0.1.0.dev6", include_so=False
    )
    shutil.copy2(missing_so_whl, wheels_dir / missing_so_whl.name)
    missing_so_sha = hashlib.sha256(missing_so_whl.read_bytes()).hexdigest()

    with _serve_directory_over_mock_https(
        monkeypatch, bkw, tmp_path / "cdn", "zing-cdn.loopit.com.cn"
    ) as base_url:
        latest_file = cdn_root / "latest.json"
        latest_url = f"{base_url}/releases/seedleap-kernels/latest.json"

        # 1. Reject py3-none-any.whl (both local --wheel-file and remote latest.json)
        with pytest.raises(ValueError, match="binary"):
            bkw.main(["--wheel-file", str(pure_whl), "--lockfile", str(lockfile)])
        latest_file.write_text(
            json.dumps(
                {
                    "version": "0.1.0.dev4",
                    "filename": pure_whl.name,
                    "url": f"{base_url}/releases/seedleap-kernels/wheels/{pure_whl.name}",
                    "sha256": pure_sha,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="binary"):
            bkw.main(["--latest-url", latest_url, "--lockfile", str(lockfile)])

        # 2. Reject Root-Is-Purelib: true disguised under platform tag
        with pytest.raises(ValueError, match="Root-Is-Purelib: false"):
            bkw.main(
                [
                    "--wheel-file",
                    str(fake_pure_platform_whl),
                    "--lockfile",
                    str(lockfile),
                ]
            )

        # 3. Reject SHA256 mismatch on remote download
        latest_file.write_text(
            json.dumps(
                {
                    "version": "0.1.0.dev4",
                    "filename": valid_whl.name,
                    "url": f"{base_url}/releases/seedleap-kernels/wheels/{valid_whl.name}",
                    "sha256": "0" * 64,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            bkw.main(["--latest-url", latest_url, "--lockfile", str(lockfile)])

        # 4. Reject forged wheel missing _fp8_gemm_sm90_ext*.so
        latest_file.write_text(
            json.dumps(
                {
                    "version": "0.1.0.dev6",
                    "filename": missing_so_whl.name,
                    "url": f"{base_url}/releases/seedleap-kernels/wheels/{missing_so_whl.name}",
                    "sha256": missing_so_sha,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="_fp8_gemm_sm90_ext"):
            bkw.main(["--latest-url", latest_url, "--lockfile", str(lockfile)])

        # 5. Reject cross-domain wheel URL or http:// wheel URL in latest.json
        latest_file.write_text(
            json.dumps(
                {
                    "version": "0.1.0.dev4",
                    "filename": valid_whl.name,
                    "url": f"https://other-cdn.example/wheels/{valid_whl.name}",
                    "sha256": valid_sha,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="host"):
            bkw.main(["--latest-url", latest_url, "--lockfile", str(lockfile)])

        latest_file.write_text(
            json.dumps(
                {
                    "version": "0.1.0.dev4",
                    "filename": valid_whl.name,
                    "url": f"http://zing-cdn.loopit.com.cn/releases/seedleap-kernels/wheels/{valid_whl.name}",
                    "sha256": valid_sha,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="HTTPS"):
            bkw.main(["--latest-url", latest_url, "--lockfile", str(lockfile)])

        # 6. Reject +local or newline/injected version strings
        for bad_ver in ("0.1.0+cu130", "0.1.0\n--extra-index-url=https://evil.example"):
            latest_file.write_text(
                json.dumps(
                    {
                        "version": bad_ver,
                        "filename": f"seedleap_kernels-{bad_ver}-cp312-cp312-linux_x86_64.whl",
                        "url": f"{base_url}/releases/seedleap-kernels/wheels/{valid_whl.name}",
                        "sha256": valid_sha,
                    }
                ),
                encoding="utf-8",
            )
            with pytest.raises(ValueError, match="version"):
                bkw.main(["--latest-url", latest_url, "--lockfile", str(lockfile)])

    assert lockfile.read_text(encoding="utf-8") == original


def test_candidate_model_payload():
    candidate = os.environ.get("WHITEBOX_MODEL_IMAGE")
    if not candidate:
        pytest.skip(
            "WHITEBOX_MODEL_IMAGE is unset; actual model image payload gate is pending"
        )
    script = """
import json, pathlib, subprocess, sys
subprocess.run(
    [
        sys.executable,
        '/opt/whitebox/deploy/whitebox/verify_model_kernels.py',
        '--lockfile',
        '/opt/whitebox/deploy/whitebox/kernels-wheel.lock.txt',
    ],
    check=True,
)
manifest = json.loads(pathlib.Path('/opt/whitebox/release.json').read_text())
assert manifest.get('schema_version') == 1
assert manifest.get('kernels_version')
assert manifest.get('kernels_git_sha')
import diffusers
assert diffusers.__version__ == '0.41.0'
import sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.loader
"""
    container = "whitebox-model-payload-check-" + uuid.uuid4().hex
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
