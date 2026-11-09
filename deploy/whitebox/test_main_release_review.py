"""Release review regressions using real Git ancestry and production CLI parsing."""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release
from config import load, validate
from test_release import bundle


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def test_production_policy_requires_main():
    policy = load("shanghai")
    assert policy["release_branch"] == "main"
    policy["release_branch"] = "white_box"
    with pytest.raises(ValueError, match="release policy"):
        validate(policy)


def test_main_only_commit_builds_with_main_identity(tmp_path, monkeypatch, capsys):
    source, sdk = tmp_path / "source", tmp_path / "sdk"
    for repo in (source, sdk):
        repo.mkdir()
        git(repo, "init", "-q")
        git(repo, "config", "user.name", "Test")
        git(repo, "config", "user.email", "test@example.com")
    (sdk / "runtime").write_text("sdk")
    git(sdk, "add", "runtime")
    git(sdk, "commit", "-qm", "sdk")
    sdk_sha = git(sdk, "rev-parse", "HEAD")
    git(sdk, "update-ref", "refs/remotes/origin/dev", sdk_sha)
    pin = source / "benchmark/whitebox/sdk-revision.txt"
    pin.parent.mkdir(parents=True)
    pin.write_text(sdk_sha + "\n")
    (source / "runtime.py").write_text("old")
    git(source, "add", ".")
    git(source, "commit", "-qm", "old release")
    git(source, "update-ref", "refs/remotes/origin/white_box", "HEAD")
    (source / "runtime.py").write_text("main-only fix")
    git(source, "commit", "-qam", "main-only fix")
    sha = git(source, "rev-parse", "HEAD")
    git(source, "update-ref", "refs/remotes/origin/main", sha)
    monkeypatch.setattr(release, "REPO", source)
    monkeypatch.setattr(release, "check_headroom", lambda *a: None)
    monkeypatch.setattr(release, "inspect_local", lambda *a: {"verified": True})
    real_run = subprocess.run
    builds = []

    def run(argv, **kwargs):
        if argv[:2] == ["docker", "build"]:
            builds.append(argv)
            assert (Path(kwargs["cwd"]) / "runtime.py").read_text() == "main-only fix"
            return subprocess.CompletedProcess(argv, 0)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    release.build(
        SimpleNamespace(
            source=sha,
            sdk_checkout=sdk,
            component="cpu",
            tag="fixture",
            platform="linux/amd64",
        )
    )
    assert len(builds) == 1
    assert "WHITEBOX_SOURCE_BRANCH=main" in builds[0]
    result = json.loads(capsys.readouterr().out)
    assert result["source_branch"] == "main"
    assert result["source_sha"] == sha
    assert result["deployed"] is False


@pytest.mark.parametrize("region", ["shanghai", "thailand"])
@pytest.mark.parametrize("quantum", [0, 1])
def test_region_command_effective_cli_values_override_environment(
    monkeypatch, region, quantum
):
    from model_service import commands

    from sglang.multimodal_gen.runtime.realtime.zing_source_stream import (
        source_work_quantum,
    )
    from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs

    candidate = bundle(region)
    # Both deployed regions inherit quantum=1. Also exercise an explicit common
    # policy change to 0; a regional override of this field is rejected below.
    assert candidate["policy"]["model"]["source_work_quantum"] == 1
    candidate["policy"]["model"]["source_work_quantum"] = quantum
    # Opposing parent values must not change an explicitly compiled deployment.
    monkeypatch.setenv("SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM", str(1 - quantum))
    monkeypatch.setenv("SGLANG_ZING_SOURCE_PROMPT_JOINT", "1")
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    argv = commands(candidate)["dit"]["argv"]
    args = parser.parse_args(argv[argv.index("--model-path") :])
    assert source_work_quantum(args.zing_source_work_quantum) == quantum
    assert args.zing_source_prompt_joint is False
    assert args.zing_source_defer_final is bool(quantum)
    assert args.zing_source_decoder_overlap is bool(quantum)
    assert args.zing_source_encoder_prefetch is bool(quantum)
    assert args.zing_source_async_output is bool(quantum)
    assert args.zing_source_encoder_mode == ("overlap" if quantum else "serial")


def test_regions_have_identical_effective_behavior():
    from model_service import commands

    from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed = []
    policies = []
    for region in ("shanghai", "thailand"):
        candidate = bundle(region)
        argv = commands(candidate)["dit"]["argv"]
        args = vars(parser.parse_args(argv[argv.index("--model-path") :]))
        args.pop("zing_block0_cache_dir")
        parsed.append(args)
        policy = copy.deepcopy(candidate["policy"])
        for key in ("region", "deployment", "gpu"):
            policy.pop(key)
        policy["model"].pop("cache_dir")
        policies.append(policy)
    assert parsed[0] == parsed[1]
    assert policies[0] == policies[1]


@pytest.mark.parametrize(
    "patch",
    [
        {"model": {"source_work_quantum": 0}},
        {"model": {"adaln_mode": "table"}},
        {"environment": {"model": {"SGLANG_ZING_SOURCE_PROMPT_JOINT": "1"}}},
        {"play_seconds": 7200},
        {"media": {"model_min_bps": 2500000}},
    ],
)
def test_region_cannot_override_common_behavior(tmp_path, patch):
    from config import ROOT

    (tmp_path / "common.json").write_bytes((ROOT / "common.json").read_bytes())
    (tmp_path / "regions").mkdir()
    (tmp_path / "regions/shanghai.json").write_text(json.dumps(patch))
    with pytest.raises(ValueError, match="common.json"):
        load("shanghai", root=tmp_path)


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "old_source",
        "wrong_config",
        "wrong_sdk",
        "wrong_role",
        "missing_digest",
    ],
)
def test_coordinator_release_identity_is_required(failure):
    candidate = bundle()
    observations = {
        role: {
            "platform_image": "registry/image@sha256:" + "e" * 64,
            "version": {
                "role": role,
                "source_branch": candidate["policy"]["release_branch"],
                "source_sha": "a" * 40,
                "sdk_sha": "b" * 40,
                "config_sha256": candidate["config_sha256"],
                "region": candidate["region"],
                "identity_status": "configured",
            },
        }
        for role in ("gateway", "coordinator", "room", "model")
    }
    assert release.verify(candidate, "a" * 40, "b" * 40, observations)["identity_match"]
    broken = copy.deepcopy(observations)
    if failure == "missing":
        del broken["coordinator"]
    elif failure == "missing_digest":
        del broken["coordinator"]["platform_image"]
    else:
        field, value = {
            "old_source": ("source_sha", "d" * 40),
            "wrong_config": ("config_sha256", "d" * 64),
            "wrong_sdk": ("sdk_sha", "d" * 40),
            "wrong_role": ("role", "gateway"),
        }[failure]
        broken["coordinator"]["version"][field] = value
    with pytest.raises(ValueError, match="coordinator"):
        release.verify(candidate, "a" * 40, "b" * 40, broken)
