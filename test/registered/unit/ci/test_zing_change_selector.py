"""The selector must inspect real Git paths and preserve all six suites."""

import shutil

import pytest
from zing_ci_fixtures import SUITES, commit, git, make_repo, result_json, select, write

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, stage="base-a", runner_config="cpu")


@pytest.mark.parametrize(
    "path",
    [
        "python/sglang/multimodal_gen/runtime/pipelines/zing_h3_causal_pipeline.py",
        "python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/adapter.py",
        "python/sglang/multimodal_gen/runtime/server_args/server_args.py",
        "python/sglang/multimodal_gen/runtime/server_args.py",
        "python/sglang/multimodal_gen/runtime/loader/fsdp_load.py",
        "python/sglang/multimodal_gen/configs/pipeline_configs/zing.py",
        "python/sglang/kernels/ops/layernorm/norm.py",
        "python/sglang/kernels/ops/activation/activation.py",
        "python/sglang/kernels/ops/diffusion/ops.py",
        "python/sglang/__init__.py",
        "python/sglang/_platform_stubs.py",
        "python/sglang/srt/scheduler.py",
        "python/sglang/utils.py",
        "python/sglang/utils/helpers.py",
        "python/sglang/version.py",
        "python/sglang/test/ci/ci_register.py",
        "python/pyproject.toml",
        "test/pytest.ini",
        "test/conftest.py",
        "test/run_suite.py",
        "test/registered/unit/ci/test_future_zing.py",
        "scripts/ci/zing/requirements-tooling.txt",
        "benchmark/zing_h3_causal/ISSUE_256_CPU_ENVIRONMENT.json",
        ".github/workflows/zing-ci.yml",
        ".github/actions/test/action.yml",
        ".pre-commit-config.yaml",
        ".isort.cfg",
        "pytest.ini",
        "pyproject.toml",
        "setup.cfg",
        "tox.ini",
        "conftest.py",
        "python/sglang/multimodal_gen/test/unit/conditioning/test_future.py",
        "test/registered/unit/multimodal_gen/test_generic.py",
    ],
)
def test_relevant_path_selects_literal_six_suites(tmp_path, path):
    repo = make_repo(tmp_path)
    base = git(repo, "rev-parse", "HEAD").stdout.strip()
    write(repo, path, "changed\n")
    head = commit(repo)
    process, output = select(repo, base, head)
    assert process.returncode == 0, process.stderr
    receipt = result_json(output)
    assert receipt["suites"] == SUITES
    assert receipt["base_sha"] == base and receipt["head_sha"] == head
    assert receipt["checkout_sha"] == head
    assert path in receipt["changed_paths"]


@pytest.mark.parametrize("operation", ["delete", "rename-out", "rename-in"])
def test_deleted_and_both_rename_sides_participate(tmp_path, operation):
    repo = make_repo(tmp_path)
    relevant = "python/sglang/kernels/ops/layernorm/norm.py"
    old = "notes/old file\nname.py" if operation == "rename-in" else relevant
    write(repo, old, "same content\n" * 20)
    base = commit(repo)
    new = relevant if operation == "rename-in" else "notes/new file\nname.py"
    if operation == "delete":
        (repo / old).unlink()
    else:
        (repo / new).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(repo / old, repo / new)
    head = commit(repo)
    process, output = select(repo, base, head)
    assert process.returncode == 0, process.stderr
    receipt = result_json(output)
    assert receipt["suites"] == SUITES
    assert old in receipt["changed_paths"]
    if operation != "delete":
        assert new in receipt["changed_paths"]


def test_nul_paths_with_newlines_spaces_and_shell_characters(tmp_path):
    repo = make_repo(tmp_path)
    base = commit(repo)
    path = "python/sglang/kernels/odd space\n$(touch ESCAPED).py"
    write(repo, path, "fixture\n")
    head = commit(repo)
    process, output = select(repo, base, head)
    assert process.returncode == 0
    assert result_json(output)["changed_paths"] == [path]
    assert result_json(output)["suites"] == SUITES
    assert not (repo / "ESCAPED").exists()


@pytest.mark.parametrize("empty", [True, False])
def test_empty_or_unrelated_diff_is_explicit(tmp_path, empty):
    repo = make_repo(tmp_path)
    base = commit(repo)
    if not empty:
        write(repo, "notes/research.md", "unrelated\n")
    head = commit(repo)
    process, output = select(repo, base, head)
    assert process.returncode == 0
    assert result_json(output)["suites"] == []
    assert result_json(output)["event"] == "pull_request"


def test_dispatch_runs_all_even_without_diff(tmp_path):
    repo = make_repo(tmp_path)
    head = commit(repo)
    process, output = select(repo, head, head, "workflow_dispatch")
    assert process.returncode == 0
    assert result_json(output)["suites"] == SUITES


@pytest.mark.parametrize(
    "invalid", ["missing-base", "missing-head", "no-common-history"]
)
def test_missing_refs_and_missing_merge_base_fail_closed(tmp_path, invalid):
    repo = make_repo(tmp_path)
    base = head = commit(repo)
    if invalid == "missing-base":
        base = "f" * 40
    elif invalid == "missing-head":
        head = "f" * 40
    else:
        git(repo, "checkout", "--orphan", "unrelated")
        git(repo, "rm", "-rf", ".")
        write(repo, "orphan.txt", "orphan\n")
        head = commit(repo)
    process, output = select(repo, base, head)
    assert process.returncode != 0
    assert result_json(output)["status"] == "error"
    assert result_json(output)["suites"] == []


def test_merge_base_diff_includes_feature_changes_only(tmp_path):
    repo = make_repo(tmp_path)
    common = commit(repo)
    git(repo, "checkout", "-qb", "feature")
    write(repo, "notes/feature.md", "feature\n")
    head = commit(repo)
    git(repo, "checkout", "-qb", "base-tip", common)
    write(repo, "python/sglang/kernels/new_base.py", "base-only\n")
    base = commit(repo)
    process, output = select(repo, base, head)
    assert process.returncode == 0
    receipt = result_json(output)
    assert receipt["merge_base_sha"] == common
    assert receipt["changed_paths"] == ["notes/feature.md"]
    assert receipt["suites"] == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
