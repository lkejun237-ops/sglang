"""CPU contract tests; these never contact a platform, GPU or production service."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT, load, overlay, render
from model_service import commands, stop_children
from release import export, verify

from sglang.multimodal_gen.runtime.realtime.release_identity import (
    install_release_route,
)


def bundle(region="thailand"):
    return render(
        load(region), json.loads((ROOT / "bindings.example.json").read_text())
    )


def test_regional_policies_keep_independent_media_archives_and_capacity():
    sh, th = bundle("shanghai"), bundle()
    for p in (sh, th):
        e = p["environments"]["room"]
        assert e["WHITEBOX_PLAYBACK_MAX_BITRATE"] == "4000000"
        assert e["WHITEBOX_SOURCE_MAX_BITRATE"] == "1000000"
        assert e["WHITEBOX_CLI_FPS"] == "30"
        assert e["WHITEBOX_MODEL_INPUT_FRAMES"] == "25"
        assert e["WHITEBOX_PLAYBACK_GOP"] == "6"
        assert all(
            e[key] == "4000000"
            for key in (
                "WHITEBOX_ARCHIVE_SOURCE_BITRATE",
                "WHITEBOX_ARCHIVE_PAIRED_SOURCE_BITRATE",
                "WHITEBOX_ARCHIVE_MODEL_BITRATE",
            )
        )
        assert p["active_session_capacity"] == 1
        assert p["environments"]["coordinator"]["WHITEBOX_SLOTS"] == "1"
        assert all(
            "WHITEBOX_INTERNAL_TOKEN" not in env for env in p["environments"].values()
        )
        assert p["policy"]["model"]["cache_host_gb"] == 416
    assert sh["environments"]["room"]["WHITEBOX_MODEL_MIN_BITRATE"] == "2000000"
    assert sh["environments"]["room"]["WHITEBOX_RTC_MIN_SEND_BITRATE"] == "3000000"
    assert sh["environments"]["room"]["WHITEBOX_RTC_MAX_SEND_BITRATE"] == "6000000"
    assert th["environments"]["room"]["WHITEBOX_MODEL_MIN_BITRATE"] == "2000000"
    assert sh["environments"]["room"]["WHITEBOX_PLAY_SECONDS"] == "1800"
    assert th["environments"]["room"]["WHITEBOX_PLAY_SECONDS"] == "1800"
    assert sh["config_sha256"] != th["config_sha256"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("cache_host_gb", -1),
        ("width", 1280),
        ("denoising_steps", [500, 1000]),
        ("dit_gpus", [0, 1, 2]),
    ],
)
def test_invalid_model_policy_rejected(key, value):
    c = load("thailand")
    c["model"][key] = value
    with pytest.raises(ValueError):
        render(c, bundle()["bindings"])


def test_unknown_fields_and_secret_binding_rejected():
    with pytest.raises(ValueError, match="Unknown field"):
        overlay(load("shanghai"), {"modle": {}})  # codespell:ignore modle
    with pytest.raises(ValueError, match="Invalid type"):
        overlay(load("shanghai"), {"play_seconds": True})
    b = bundle()["bindings"]
    b["token"] = "never-persist"
    with pytest.raises(ValueError, match="unknown"):
        render(load("shanghai"), b)


def test_bitrate_bound_and_digest_change():
    c, b = load("thailand"), bundle()["bindings"]
    original = render(c, b)["config_sha256"]
    c["media"]["model_min_bps"] = 4500000
    with pytest.raises(ValueError, match="Minimum bitrate"):
        render(c, b)
    c["media"]["model_min_bps"] = 2100000
    assert render(c, b)["config_sha256"] != original


def test_adaptive_source_sampling_is_explicit_and_allows_cli_60():
    c = load("thailand")
    assert c["environment"]["room"]["WHITEBOX_MODEL_SOURCE_SAMPLING"] == "latest"
    selected = overlay(
        c,
        {
            "media": {"cli_fps": 60},
            "environment": {"room": {"WHITEBOX_MODEL_SOURCE_SAMPLING": "adaptive"}},
        },
    )
    compiled = render(selected, bundle()["bindings"])
    assert compiled["environments"]["room"]["WHITEBOX_CLI_FPS"] == "60"
    assert (
        compiled["environments"]["room"]["WHITEBOX_MODEL_SOURCE_SAMPLING"] == "adaptive"
    )
    assert "WHITEBOX_MODEL_SOURCE_SAMPLING" not in compiled["environments"]["model"]
    selected["environment"]["room"]["WHITEBOX_RENDER_SEGMENTS"] = "0"
    with pytest.raises(ValueError, match="sampling policy"):
        render(selected, bundle()["bindings"])


def test_model_736_preserves_native_480_source():
    policy = load("thailand")
    policy["model"].update(width=1280, height=736, quantization="fp8")
    candidate = render(policy, bundle()["bindings"])
    for role in ("room", "gateway"):
        env = candidate["environments"][role]
        assert (env["WHITEBOX_WIDTH"], env["WHITEBOX_HEIGHT"]) == ("832", "480")
        assert (env["WHITEBOX_MODEL_WIDTH"], env["WHITEBOX_MODEL_HEIGHT"]) == (
            "1280",
            "736",
        )
    policy["environment"]["shared"]["WHITEBOX_WIDTH"] = "1280"
    with pytest.raises(ValueError, match="source capture"):
        render(policy, bundle()["bindings"])


def test_role_plan_preserves_separate_encoder_conditioning_dit_decoder_and_cache():
    p = commands(bundle(), python="/venv/bin/python")
    assert list(p) == ["encoder", "conditioning", "decoder", "dit", "heartbeat"]
    # The integrated owner uses the legacy transport, not dynamic group RPCs.
    assert "--static-connection" in p["decoder"]["argv"]
    assert [p[k]["devices"] for k in p] == [[2], [2], [3], [0, 1], []]
    args = p["dit"]["argv"]
    assert args[args.index("--mem-fraction-static") + 1] == "0.8"
    assert args[args.index("--zing-block0-cache-storage") + 1] == "local"
    assert args[args.index("--zing-denoising-step-list") + 1] == "1000,500"
    assert args[args.index("--zing-block0-cache-host-gb") + 1] == "416"
    assert "--zing-source-defer-final" in args
    assert "--zing-block0-cache-gc" in args
    assert args[args.index("--zing-block0-cache-host-total-gb") + 1] == "416"
    assert args[args.index("--zing-block0-cache-min-free-gb") + 1] == "128"
    assert args[args.index("--zing-block0-cache-gc-high-watermark") + 1] == "0.9"
    assert args[args.index("--zing-block0-cache-gc-low-watermark") + 1] == "0.8"
    sh = commands(bundle("shanghai"))["dit"]["argv"]
    assert sh[sh.index("--zing-source-defer-final") + 1] == "true"
    assert p["heartbeat"]["argv"][p["heartbeat"]["argv"].index("--capacity") + 1] == "1"


def test_online_fp8_is_explicit_dit_only_and_keeps_four_gpu_roles():
    original = bundle()
    policy = load("thailand")
    policy["model"].update(quantization="fp8", quantization_scope="ffn-attention")
    candidate = render(policy, original["bindings"])
    plan = commands(candidate)
    args = plan["dit"]["argv"]
    assert args[args.index("--quantization") + 1] == "fp8"
    assert args[args.index("--direct-gpu-weight-loading") + 1] == "false"
    assert (
        candidate["environments"]["model"]["SGLANG_ZING_QUANT_SCOPE"] == "ffn-attention"
    )
    assert candidate["config_sha256"] != original["config_sha256"]
    for role in ("encoder", "conditioning", "decoder", "heartbeat"):
        assert plan[role] == commands(original)[role]
    assert [plan[k]["devices"] for k in plan] == [[2], [2], [3], [0, 1], []]


@pytest.mark.parametrize("region", ["shanghai", "thailand"])
def test_default_and_legacy_bundles_do_not_enable_quantization(region):
    candidate = bundle(region)
    assert "--quantization" not in commands(candidate)["dit"]["argv"]
    assert "SGLANG_ZING_QUANT_SCOPE" not in candidate["environments"]["model"]
    policy = load(region)
    policy["model"].pop("quantization")
    policy["model"].pop("quantization_scope")
    legacy = render(policy, candidate["bindings"])
    assert commands(legacy) == commands(candidate)


@pytest.mark.parametrize(
    "field,value",
    [("quantization", "mxfp8"), ("quantization_scope", "all")],
)
def test_unvalidated_quantization_policy_rejected(field, value):
    policy = load("thailand")
    policy["model"][field] = value
    with pytest.raises(ValueError):
        render(policy, bundle()["bindings"])


def test_plan_is_side_effect_free_and_does_not_overwrite(tmp_path):
    output = tmp_path / "plan.json"
    args = [
        sys.executable,
        str(ROOT / "release.py"),
        "plan",
        "--region",
        "shanghai",
        "--bindings",
        str(ROOT / "bindings.example.json"),
        "--output",
        str(output),
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["deployed"] is False
    before = output.read_bytes()
    assert subprocess.run(args, capture_output=True, check=False).returncode != 0
    assert output.read_bytes() == before
    result = subprocess.run(
        [sys.executable, str(ROOT / "model_service.py"), "--bundle", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert list(json.loads(result.stdout)) == [
        "encoder",
        "conditioning",
        "decoder",
        "dit",
        "heartbeat",
    ]


def test_git_archive_excludes_dirty_local_patches_and_secrets(tmp_path):
    repo, target = tmp_path / "repo", tmp_path / "export"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "runtime.py").write_text("verified source")
    for name in (
        "asset-library/subjects/old.glb",
        "asset-library/tests/fixture.glb",
        "asset-library/tests/fixtures/atlas-recording/recording.json",
        "asset-library/tests/fixtures/atlas-recording/inputs.zip",
        "asset-library/package.json",
        "packages/stream-host/tests/raw-session.test.ts",
    ):
        file = repo / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(name)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    (repo / "runtime.py").write_text("hidden patch")
    (repo / "secret.txt").write_text("private")
    export(repo, "HEAD", target)
    assert (target / "runtime.py").read_text() == "verified source"
    assert not (target / "secret.txt").exists()
    assert (target / "asset-library/subjects/old.glb").is_file()

    slim = tmp_path / "room-sdk"
    export(repo, "HEAD", slim, room_sdk=True)
    assert (slim / "runtime.py").read_text() == "verified source"
    assert (slim / "asset-library/package.json").is_file()
    assert (slim / "packages/stream-host/tests/raw-session.test.ts").is_file()
    assert not (slim / "asset-library/subjects").exists()
    assert not (slim / "asset-library/tests/fixture.glb").exists()
    for name in ("recording.json", "inputs.zip"):
        rel = Path("asset-library/tests/fixtures/atlas-recording") / name
        assert (slim / rel).read_bytes() == (repo / rel).read_bytes()


def test_platform_builder_binds_release_identity_to_actual_checkout():
    source = (ROOT.parents[1] / "benchmark/whitebox/Jenkinsfile").read_text()
    assert 'build-arg:WHITEBOX_SOURCE_SHA="$GIT_SHA"' in source
    assert 'build-arg:WHITEBOX_SDK_SHA="$sdk_sha"' in source
    sdk_stage = source.split("stage('Pinned CLI SDK')", 1)[1].split(
        "stage('Platform Dockerfile')", 1
    )[0]
    assert '"$WHITEBOX_SDK_REVISION" origin/dev' in sdk_stage
    assert "origin/white_box" not in sdk_stage
    assert 'build-arg:WHITEBOX_SOURCE_BRANCH="$WHITEBOX_SOURCE_BRANCH"' in source
    assert (
        'sh ../deploy/whitebox/room-sdk-archive.sh . "$WHITEBOX_SDK_REVISION"' in source
    )
    assert 'tar -xf "$archive" -C ../.whitebox-sdk' in source


def test_version_identity_is_frozen_and_never_returns_credentials(
    tmp_path, monkeypatch
):
    f = tmp_path / "release.json"
    m = {"source_sha": "a" * 40, "sdk_sha": "b" * 40, "source_branch": "main"}
    f.write_text(json.dumps(m))
    monkeypatch.setenv("WHITEBOX_RELEASE_MANIFEST", str(f))
    b = bundle("shanghai")
    for key, value in b["environments"]["room"].items():
        monkeypatch.setenv(key, value)
    # Mount location is deployment wiring; the fixture uses a private local path.
    mounted = tmp_path / "bundle.json"
    b["environments"]["room"]["WHITEBOX_RELEASE_BUNDLE"] = str(mounted)
    mounted.write_text(json.dumps(b))
    monkeypatch.setenv("WHITEBOX_RELEASE_BUNDLE", str(mounted))
    monkeypatch.setenv("WHITEBOX_INTERNAL_TOKEN", "must-not-be-returned")
    app = FastAPI()
    install_release_route(app, "room")
    f.write_text(json.dumps({**m, "source_sha": "d" * 40}))
    result = TestClient(app).get("/v1/whitebox/version").json()
    assert result["source_sha"] == "a" * 40
    assert result["source_branch"] == "main"
    assert result["identity_status"] == "configured"
    assert "must-not-be-returned" not in json.dumps(result)
    monkeypatch.setenv("WHITEBOX_MODEL_MIN_BITRATE", "100000")
    with pytest.raises(ValueError, match="WHITEBOX_MODEL_MIN_BITRATE"):
        install_release_route(FastAPI(), "room")


def test_unmanaged_version_and_bad_manifest(monkeypatch, tmp_path):
    monkeypatch.delenv("WHITEBOX_RELEASE_MANIFEST", raising=False)
    app = FastAPI()
    install_release_route(app, "gateway")
    assert (
        TestClient(app).get("/v1/whitebox/version").json()["identity_status"]
        == "unmanaged"
    )
    f = tmp_path / "bad.json"
    f.write_text("{}")
    monkeypatch.setenv("WHITEBOX_RELEASE_MANIFEST", str(f))
    with pytest.raises(ValueError):
        install_release_route(FastAPI(), "room")


def test_development_identity_requires_explicit_opt_in(monkeypatch, tmp_path):
    manifest = tmp_path / "development.json"
    manifest.write_text(
        json.dumps(
            dict(
                source_sha="a" * 40,
                sdk_sha="b" * 40,
                source_branch="codex/isolated-h100-test",
            )
        )
    )
    monkeypatch.setenv("WHITEBOX_RELEASE_MANIFEST", str(manifest))
    monkeypatch.delenv("WHITEBOX_ALLOW_DEVELOPMENT_RELEASE", raising=False)
    with pytest.raises(ValueError, match="main"):
        install_release_route(FastAPI(), "room")
    monkeypatch.setenv("WHITEBOX_ALLOW_DEVELOPMENT_RELEASE", "1")
    app = FastAPI()
    install_release_route(app, "room")
    identity = TestClient(app).get("/v1/whitebox/version").json()
    assert identity["source_branch"] == "codex/isolated-h100-test"
    assert identity["source_sha"] == "a" * 40
    assert identity["release_scope"] == "isolated_development_test"
    manifest.write_text(
        manifest.read_text().replace("codex/isolated-h100-test", "white_box")
    )
    with pytest.raises(ValueError, match="main"):
        install_release_route(FastAPI(), "room")


def test_verification_compares_platform_digest_and_running_release():
    p = bundle()
    observations = {
        role: {
            "platform_image": "registry/image@sha256:" + "e" * 64,
            "version": {
                "role": role,
                "source_branch": "main",
                "source_sha": "a" * 40,
                "sdk_sha": "b" * 40,
                "config_sha256": p["config_sha256"],
                "region": p["region"],
                "identity_status": "configured",
            },
        }
        for role in ("gateway", "coordinator", "room", "model")
    }
    assert verify(p, "a" * 40, "b" * 40, observations)["performance_verified"] is False
    observations["room"]["version"]["sdk_sha"] = "d" * 40
    with pytest.raises(ValueError, match="mismatch"):
        verify(p, "a" * 40, "b" * 40, observations)


def test_cleanup_targets_only_own_process_groups(tmp_path, monkeypatch):
    killed = []

    class Process:
        pid = 2345

        def wait(self, **kwargs):
            return 0

    log = (tmp_path / "log").open("w")
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killed.append(pid))
    stop_children([(Process(), log)])
    assert killed == [2345] and log.closed


def test_manual_builder_rejects_unmerged_source_before_any_docker_command(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import release

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    (repo / "source").write_text("production")
    git("add", ".")
    git("commit", "-qm", "production")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    (repo / "source").write_text("candidate")
    git("add", ".")
    git("commit", "-qm", "candidate")
    git("update-ref", "refs/remotes/origin/white_box", "HEAD")
    monkeypatch.setattr(release, "REPO", repo)
    with pytest.raises(subprocess.CalledProcessError):
        release.build(SimpleNamespace(source="HEAD"))


@pytest.mark.parametrize("sdk_in_dev", [True, False])
def test_manual_builder_requires_sdk_pin_promoted_to_dev(
    tmp_path, monkeypatch, sdk_in_dev
):
    from types import SimpleNamespace

    import release

    def git(repo, *args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    sdk = tmp_path / "sdk"
    source = tmp_path / "source"
    for repo in (sdk, source):
        repo.mkdir()
        git(repo, "init", "-q")
        git(repo, "config", "user.name", "Test")
        git(repo, "config", "user.email", "test@example.com")
    (sdk / "runtime").write_text("previous dev")
    git(sdk, "add", ".")
    git(sdk, "commit", "-qm", "dev")
    git(sdk, "update-ref", "refs/remotes/origin/dev", "HEAD")
    (sdk / "runtime").write_text("stream retirement fix")
    git(sdk, "commit", "-qam", "candidate")
    pin = git(sdk, "rev-parse", "HEAD")
    git(sdk, "update-ref", "refs/remotes/origin/white_box", pin)
    if sdk_in_dev:
        git(sdk, "update-ref", "refs/remotes/origin/dev", pin)
    pin_file = source / "benchmark/whitebox/sdk-revision.txt"
    pin_file.parent.mkdir(parents=True)
    pin_file.write_text(pin + "\n")
    git(source, "add", ".")
    git(source, "commit", "-qm", "release")
    git(source, "update-ref", "refs/remotes/origin/main", "HEAD")
    monkeypatch.setattr(release, "REPO", source)
    args = SimpleNamespace(source="HEAD", sdk_checkout=sdk)
    if sdk_in_dev:
        assert release.release_inputs(args) == (git(source, "rev-parse", "HEAD"), pin)
    else:
        with pytest.raises(subprocess.CalledProcessError):
            release.release_inputs(args)


@pytest.mark.parametrize("fail_readiness", [True, False])
def test_all_role_cleanup_never_automatically_restarts(
    tmp_path, monkeypatch, fail_readiness
):
    from types import SimpleNamespace

    import model_service

    p = bundle()
    root = tmp_path / "assets"
    (root / "model").mkdir(parents=True)
    state = tmp_path / "state"
    p["bindings"].update(asset_root=str(root), state_root=str(state))
    p = render(p["policy"], p["bindings"])
    (root / "pipeline.json").write_text(
        json.dumps(
            {
                "zing_adaln_mode": "resident",
                "zing_kv_pool_sink_chunks": 0,
                "zing_kv_pool_window_chunks": 6,
                "decoder_config": {"type": "lynn_streaming"},
            }
        )
    )
    (root / "model/zing_config.json").write_text(
        json.dumps(
            {
                "num_frame_per_block": [3, 2],
                "split_first_latent": False,
                "checkpoint_sha256": p["policy"]["model"]["checkpoint_sha256"],
            }
        )
    )
    (root / "model/zing_realtime_profile.json").write_text(
        json.dumps({"source_attention_window": {"window_chunks": 6, "sink_chunks": 0}})
    )
    (root / "conditioning-profile.json").write_text("{}")
    (root / "worker.secret.json").write_text('{"bearer_token":"fixture-only"}')
    monkeypatch.setattr(model_service, "hardware", lambda *_: [])
    # Asset integrity has its own real-file regressions; this case isolates
    # subprocess ownership, readiness ordering and cleanup.
    monkeypatch.setattr(model_service, "verify_prepared", lambda *_: {})
    monkeypatch.setitem(
        sys.modules,
        "sglang.multimodal_gen.runtime.disaggregation.role_launcher",
        SimpleNamespace(wait_ready=lambda *a, **k: None),
    )
    monkeypatch.setattr(model_service.signal, "signal", lambda *_: None)
    monkeypatch.setenv("WORKER_EPOCH", "stale-parent-epoch")
    monkeypatch.setenv("WORKER_EPOCH_FILE", "/old-release/worker-epoch")
    spawned = []
    terminated = []

    class Process:
        def __init__(self, argv, **kwargs):
            self.pid = 1000 + len(spawned)
            self.argv = argv
            self.env = kwargs["env"]
            self.stdout = __import__("io").BytesIO()
            spawned.append(self)

        def poll(self):
            return None

        def wait(self, **kwargs):
            return 0

    monkeypatch.setattr(model_service.subprocess, "Popen", Process)
    monkeypatch.setattr(
        model_service.os, "killpg", lambda pid, sig: terminated.append(pid)
    )

    def ready(proc, url, headers):
        if fail_readiness and url.endswith("/health"):
            raise TimeoutError("fixture readiness failed")

    monkeypatch.setattr(model_service, "wait_http", ready)

    def stop_loop(_):
        raise SystemExit(0)

    monkeypatch.setattr(model_service.time, "sleep", stop_loop)
    with pytest.raises(TimeoutError if fail_readiness else SystemExit):
        model_service.run(p, ["GPU-a", "GPU-b", "GPU-c", "GPU-d"], retries=0)
    # A failed role retains evidence and exits; it never starts another group.
    # A healthy manual shutdown also cleans up the heartbeat.
    assert len(spawned) == (4 if fail_readiness else 5)
    assert sorted(terminated) == [process.pid for process in spawned]
    epoch = (state / "worker-epoch").read_text()
    assert epoch != "stale-parent-epoch" and len(epoch) == 32
    assert {p.env["WORKER_EPOCH"] for p in spawned} == {epoch}
    assert {p.env["WORKER_EPOCH_FILE"] for p in spawned} == {
        str(state / "worker-epoch")
    }
    assert all("fixture-only" not in repr(process.argv) for process in spawned)
    assert [process.env["CUDA_VISIBLE_DEVICES"] for process in spawned[:4]] == [
        "GPU-c",
        "GPU-c",
        "GPU-d",
        "GPU-a,GPU-b",
    ]
    status = json.loads((state / "status.json").read_text())
    assert status["phase"] == ("failed" if fail_readiness else "stopped")
    assert status["attempt"] == 0


@pytest.fixture
def readable_archive_repo(tmp_path):
    repo = tmp_path / "archive-repo"
    package = repo / "python/sglang/nested"
    package.mkdir(parents=True)
    (package / "module.py").write_text("VALUE = 42\n")
    (package / "run.sh").write_text("#!/bin/sh\nexit 0\n")
    (package / "run.sh").chmod(0o755)
    (package / "alias.py").symlink_to("module.py")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "readable fixture",
        ],
        check=True,
    )
    return repo


@pytest.mark.parametrize("room_sdk", [False, True])
def test_private_builder_umask_does_not_hide_exported_runtime(
    readable_archive_repo, tmp_path, room_sdk
):
    destination = tmp_path / "context"
    previous = os.umask(0o077)
    try:
        export(readable_archive_repo, "HEAD", destination, room_sdk=room_sdk)
    finally:
        os.umask(previous)
    nested = destination / "python/sglang/nested"
    assert nested.stat().st_mode & 0o777 == 0o755
    assert (nested / "module.py").stat().st_mode & 0o777 == 0o644
    assert (nested / "run.sh").stat().st_mode & 0o777 == 0o755
    assert (nested / "alias.py").is_symlink()
    assert os.readlink(nested / "alias.py") == "module.py"


def test_archive_filter_still_rejects_escaping_symlink(readable_archive_repo, tmp_path):
    import tarfile

    (readable_archive_repo / "escape").symlink_to("../outside")
    subprocess.run(
        ["git", "-C", str(readable_archive_repo), "add", "escape"], check=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(readable_archive_repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "unsafe link fixture",
        ],
        check=True,
    )
    with pytest.raises(tarfile.LinkOutsideDestinationError):
        export(readable_archive_repo, "HEAD", tmp_path / "context")
    assert not (tmp_path / "outside").exists()


@pytest.mark.skipif(
    os.environ.get("WHITEBOX_TEST_DOCKER_EXPORT") != "1",
    reason="opt-in local Docker archive/projection/non-root gate",
)
def test_git_archive_projection_is_readable_by_actual_runtime_uid(
    readable_archive_repo, tmp_path
):
    import shutil
    import uuid

    context = tmp_path / "docker-context"
    previous = os.umask(0o077)
    try:
        export(readable_archive_repo, "HEAD", context)
    finally:
        os.umask(previous)
    shutil.copyfile(
        ROOT.parents[1] / "deploy/whitebox/project_runtime_source.py",
        context / "project.py",
    )
    (context / "Dockerfile").write_text("""FROM python:3.14-slim-bookworm AS projection
COPY project.py /project.py
RUN --mount=type=bind,source=python/sglang,target=/source/sglang,readonly python /project.py --source /source/sglang --destination /runtime/sglang
FROM python:3.14-slim-bookworm
COPY --from=projection /runtime/sglang /app/sglang
USER 10001:10001
RUN python -c 'import os,pathlib,subprocess; p=pathlib.Path("/app/sglang/nested"); assert os.getuid()==10001; assert (p/"module.py").read_text()=="VALUE = 42\\n"; assert (p/"alias.py").read_text()=="VALUE = 42\\n"; subprocess.run([str(p/"run.sh")],check=True)'
""")
    tag = "whitebox-export-regression:" + uuid.uuid4().hex
    try:
        subprocess.run(
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "--tag",
                tag,
                str(context),
            ],
            check=True,
        )
    finally:
        subprocess.run(
            ["docker", "image", "rm", "--no-prune", tag],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def test_dynamic_room_pool_is_explicit_and_only_adds_runtime_wiring():
    bindings = json.loads((ROOT / "bindings.example.json").read_text())
    static = render(load("shanghai"), bindings)
    dynamic = render(
        load("shanghai"), {**bindings, "room_pool_compatibility": "tested-room-pool-v1"}
    )
    assert "WHITEBOX_DYNAMIC_ROOMS" not in static["environments"]["room"]
    for role in ("gateway", "coordinator", "room"):
        assert dynamic["environments"][role]["WHITEBOX_DYNAMIC_ROOMS"] == "1"
    assert dynamic["environments"]["model"] == static["environments"]["model"] | {
        "WHITEBOX_RELEASE_CONFIG_SHA256": dynamic["config_sha256"]
    }
    assert (
        dynamic["environments"]["room"]["WHITEBOX_ROOM_ADVERTISE_URL"]
        == bindings["room_url"]
    )
    assert "WHITEBOX_ROOM_ADMIN_TOKEN" in dynamic["required_secret_keys"]["coordinator"]
    assert (
        "WHITEBOX_ROOM_ADMIN_TOKEN" not in static["required_secret_keys"]["coordinator"]
    )


def test_rendered_shanghai_control_roles_check_the_same_manifest_at_startup(
    tmp_path, monkeypatch
):
    from control_bindings import REFS_ENV, marker
    from test_control_bindings import manifest

    pins = manifest()
    serialized = json.dumps(pins, sort_keys=True, separators=(",", ":"))
    b = json.loads((ROOT / "bindings.example.json").read_text())
    b["gateway_url"] = "https://shcopy.world-test-platform.loopit.com.cn"
    b["control_credential_refs"] = serialized
    plan = render(load("shanghai"), b)
    with pytest.raises(ValueError, match="region"):
        render(load("thailand"), b)
    sf = tmp_path / "release.json"
    sf.write_text(
        json.dumps(
            {"source_sha": "a" * 40, "sdk_sha": "b" * 40, "source_branch": "main"}
        )
    )
    monkeypatch.setenv("WHITEBOX_RELEASE_MANIFEST", str(sf))
    bf = tmp_path / "bundle"
    for env in plan["environments"].values():
        env["WHITEBOX_RELEASE_BUNDLE"] = str(bf)
    bf.write_text(json.dumps(plan))
    monkeypatch.setenv("WHITEBOX_RELEASE_BUNDLE", str(bf))
    identities = set()
    for role in ("room", "gateway", "coordinator"):
        for k, v in plan["environments"][role].items():
            monkeypatch.setenv(k, v)
        assert plan["environments"][role][REFS_ENV] == serialized
        for env, purpose in (
            ("WHITEBOX_INTERNAL_TOKEN", "room_internal"),
            ("WHITEBOX_RTC_AUTH_SECRET", "rtc"),
            ("WHITEBOX_ROOM_ADMIN_TOKEN", "room_admin"),
        ):
            monkeypatch.setenv(env, "local-fixture-" + purpose)
            monkeypatch.setenv(env + "_OOS_REF", marker(pins["refs"][purpose]))
        app = FastAPI()
        install_release_route(app, role)
        identities.add(app.state.whitebox_release["control_binding_sha256"])
        monkeypatch.setenv("WHITEBOX_INTERNAL_TOKEN_OOS_REF", "wrong")
        with pytest.raises(ValueError, match="room_internal"):
            install_release_route(FastAPI(), role)
    assert len(identities) == 1
