"""Workflow outcomes must obey the executed detector/tooling truth table."""

import json
import os
import re
import subprocess
import sys

import pytest
import yaml
from zing_ci_fixtures import (
    ROOT,
    SUITES,
    TOOLING_FILES,
    cli,
    commit,
    copy_tooling_entrypoint,
    json_file,
    make_repo,
    result_json,
    run_tooling,
    select,
    write,
)

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, stage="base-a", runner_config="cpu")


@pytest.mark.parametrize("producer_attempt,summary_attempt", [(1, 1), (1, 2), (2, 3)])
def test_summary_downloads_the_successful_detector_artifact_on_partial_rerun(
    producer_attempt, summary_attempt
):
    jobs = yaml.safe_load((ROOT / ".github/workflows/zing-ci.yml").read_text())["jobs"]
    detector = jobs["zing-changes"]
    upload = next(
        step for step in detector["steps"] if "upload-artifact@" in step.get("uses", "")
    )
    download = next(
        step
        for step in jobs["zing-ci"]["steps"]
        if "download-artifact@" in step.get("uses", "")
    )
    sha, run_id, artifact_id = "current-merge-sha", "42", "1001"

    def resolve(value, context):
        return re.sub(r"\$\{\{\s*(.*?)\s*\}\}", lambda match: context[match[1]], value)

    producer = {
        "github.run_id": run_id,
        "github.run_attempt": str(producer_attempt),
        "github.sha": sha,
    }
    artifact_name = resolve(upload["with"]["name"], producer)
    # The upload action returns an immutable ID. GitHub preserves outputs of
    # successful prerequisites when only failed downstream jobs are rerun.
    producer[f"steps.{upload.get('id')}.outputs.artifact-id"] = artifact_id
    consumer = {
        "github.run_id": run_id,
        "github.run_attempt": str(summary_attempt),
        "github.sha": sha,
    }
    consumer.update(
        {
            f"needs.zing-changes.outputs.{key}": resolve(value, producer)
            for key, value in detector["outputs"].items()
            if key != "run_cpu"
        }
    )
    artifacts = {artifact_name: {"id": artifact_id, "sha": sha}}
    # Include an unrelated artifact: matching a prefix or selecting the newest
    # artifact must never silently accept evidence for a different commit.
    artifacts["zing-selection-42-99-other-sha"] = {"id": "9999", "sha": "other-sha"}
    inputs = download["with"]
    if "artifact-ids" in inputs:
        requested = resolve(inputs["artifact-ids"], consumer)
        matches = [item for item in artifacts.values() if item["id"] == requested]
        assert inputs.get("merge-multiple") is True
    else:
        requested = resolve(inputs["name"], consumer)
        matches = [artifacts[requested]] if requested in artifacts else []
    assert matches == [{"id": artifact_id, "sha": sha}], requested
    assert "pattern" not in inputs and "run-id" not in inputs


@pytest.mark.parametrize(
    "filename", ["zing-ci.yml", "zing-cpu-suite.yml", "zing-conditioning-suite.yml"]
)
def test_workflow_cpu_environment_runs_full_tiles_without_hub_network(filename):
    workflow = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())
    required = {
        "ZING_FULL_TILE_TESTS": "1",
        "HF_HUB_OFFLINE": "1",
        "SGLANG_LIGHTWEIGHT_RUNTIME": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    environment = os.environ.copy()
    # The workflow must provide these values rather than accidentally passing
    # because the host already enabled a test matrix or an offline cache.
    for key in required:
        environment.pop(key, None)
    environment.update({key: str(value) for key, value in workflow["env"].items()})
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,os\n"
            f"required = {required!r}\n"
            "observed = {key: os.environ.get(key) for key in required}\n"
            "print(json.dumps(observed))\n"
            "assert observed == required, observed\n",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == required


def test_migrated_business_checks_have_one_owner_in_the_new_workflow():
    required = {"decoder-contracts", "adaln-export-contracts", "conditioning-contracts"}
    owners = {name: [] for name in required}
    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
        workflow = yaml.safe_load(path.read_text())
        if not isinstance(workflow, dict):
            continue
        for job_id, job in workflow.get("jobs", {}).items():
            # Check logical suite ownership. Reusable jobs have a nested
            # display name in GitHub, which is not a branch-protection alias.
            check_name = job.get("name", job_id)
            if check_name in required:
                owners[check_name].append(path.name)
    assert owners == {name: ["zing-ci.yml"] for name in required}, owners


def test_reusable_suites_are_all_gated_and_required_by_summary():
    workflows = ROOT / ".github/workflows"
    jobs = yaml.safe_load((workflows / "zing-ci.yml").read_text())["jobs"]
    infrastructure = {
        "zing-changes",
        "zing-lint",
        "ci-tooling-contracts",
        "area-contracts",
    }
    assert set(jobs) == infrastructure | set(SUITES) | {"zing-ci"}
    assert set(jobs["zing-ci"]["needs"]) == infrastructure | set(SUITES)
    assert jobs["zing-ci"]["if"] == "always()"
    for job_id in ("zing-lint", "ci-tooling-contracts", "area-contracts"):
        assert "needs" not in jobs[job_id] and "if" not in jobs[job_id]
    for suite in SUITES:
        caller = jobs[suite]
        assert caller["needs"] == ["zing-changes"]
        assert caller["if"] == "needs.zing-changes.outputs.run_cpu == 'true'"
        filename = (
            "zing-conditioning-suite.yml"
            if suite == "conditioning-contracts"
            else "zing-cpu-suite.yml"
        )
        assert caller["uses"] == f"./.github/workflows/{filename}"
        assert "steps" not in caller and "continue-on-error" not in caller
        callee = yaml.safe_load((workflows / filename).read_text())
        # BaseLoader preserves YAML's `on` key instead of parsing it as True.
        triggers = yaml.load((workflows / filename).read_text(), Loader=yaml.BaseLoader)
        assert set(triggers["on"]) == {"workflow_call"}
        assert "concurrency" not in callee
        if suite != "conditioning-contracts":
            assert caller["with"]["suite"] == suite
    dit = jobs["dit-pipeline-contracts"]
    assert dit["strategy"]["fail-fast"] is False
    assert dit["strategy"]["matrix"]["shard"] == list(range(dit["with"]["shard-count"]))
    assert dit["with"]["shard-index"] == "${{ matrix.shard }}"


def needs(selected):
    return {
        "zing-changes": {"result": "success"},
        "zing-lint": {"result": "success"},
        "ci-tooling-contracts": {"result": "success"},
        **{
            suite: {"result": "success" if suite in selected else "skipped"}
            for suite in SUITES
        },
    }


def summarize(repo, selection, outcomes, event="pull_request"):
    selection_path = json_file(repo.parent / "selection.json", selection)
    needs_path = json_file(repo.parent / "needs.json", outcomes)
    output = repo.parent / "summary.json"
    process = cli(
        "summarize_results.py",
        repo,
        "--selection",
        selection_path,
        "--needs",
        needs_path,
        "--event",
        event,
        "--output",
        output,
    )
    return process, output


@pytest.mark.parametrize("selected", [[], SUITES])
def test_complete_valid_results_pass_and_record_actual_business_coverage(
    tmp_path, selected
):
    repo = make_repo(tmp_path)
    process, output = summarize(
        repo, {"status": "success", "suites": selected}, needs(selected)
    )
    assert process.returncode == 0, process.stderr
    receipt = result_json(output)
    assert receipt["status"] == "success"
    assert receipt["business_ran"] is bool(selected)
    assert receipt["tooling_ran"] is True


@pytest.mark.parametrize(
    "job", ["zing-changes", "zing-lint", "ci-tooling-contracts", *SUITES]
)
@pytest.mark.parametrize("bad", ["failure", "cancelled", "skipped", "missing"])
def test_any_required_non_success_fails_including_conditioning(tmp_path, job, bad):
    repo = make_repo(tmp_path)
    outcomes = needs(SUITES)
    if bad == "missing":
        outcomes.pop(job)
    else:
        outcomes[job]["result"] = bad
    process, output = summarize(repo, {"status": "success", "suites": SUITES}, outcomes)
    assert process.returncode != 0
    assert result_json(output)["status"] == "error"
    assert job in result_json(output)["errors"]


@pytest.mark.parametrize("bad", ["failure", "cancelled", "skipped", "missing"])
def test_empty_selection_still_requires_tooling_success(tmp_path, bad):
    repo = make_repo(tmp_path)
    outcomes = needs([])
    if bad == "missing":
        outcomes.pop("ci-tooling-contracts")
    else:
        outcomes["ci-tooling-contracts"]["result"] = bad
    process, output = summarize(repo, {"status": "success", "suites": []}, outcomes)
    assert process.returncode != 0
    assert "ci-tooling-contracts" in result_json(output)["errors"]


@pytest.mark.parametrize(
    "selection",
    [
        {},
        {"status": "error", "suites": []},
        {"status": "success"},
        {"status": "success", "suites": ["invented"]},
    ],
)
def test_detector_missing_or_malformed_output_is_failure(tmp_path, selection):
    repo = make_repo(tmp_path)
    process, output = summarize(repo, selection, needs([]))
    assert process.returncode != 0
    assert result_json(output)["status"] == "error"


def test_dispatch_cannot_treat_empty_selection_as_legal_skip(tmp_path):
    repo = make_repo(tmp_path)
    process, output = summarize(
        repo, {"status": "success", "suites": []}, needs([]), "workflow_dispatch"
    )
    assert process.returncode != 0
    assert result_json(output)["status"] == "error"


def test_docs_only_harness_really_runs_tooling_then_passes_summary(tmp_path):
    repo = make_repo(tmp_path)
    base = commit(repo)
    write(repo, "notes/readme.md", "docs only\n")
    head = commit(repo)
    detector, output = select(repo, base, head)
    assert detector.returncode == 0
    selection = result_json(output)
    assert selection["suites"] == []
    tooling, artifacts = run_tooling(repo)
    assert tooling.returncode == 0
    assert (
        sum(r["passed"] for r in result_json(artifacts / "receipt.json")["results"])
        == 5
    )
    process, output = summarize(repo, selection, needs([]))
    assert process.returncode == 0
    assert result_json(output)["business_ran"] is False


def test_always_empty_selector_mutant_is_caught_by_actual_tooling_and_summary(tmp_path):
    repo = make_repo(tmp_path)
    copy_tooling_entrypoint(repo)
    # Replace only the external selector boundary with the concrete known bug.
    mutant = write(
        repo,
        "scripts/ci/zing/select_changes.py",
        "import json,sys\nfrom pathlib import Path\nPath(sys.argv[sys.argv.index('--output')+1]).write_text(json.dumps({'status':'success','suites':[]}))\n",
    )
    source = f"""import json,subprocess,sys
from pathlib import Path
def test_kernel_selector_contract(tmp_path):
    output = tmp_path / "selected.json"
    process = subprocess.run([sys.executable, {str(mutant)!r}, "--repo", ".", "--base", "base", "--head", "head", "--event", "pull_request", "--output", str(output)], capture_output=True, text=True)
    assert process.returncode == 0
    assert json.loads(output.read_text())["suites"] == {SUITES!r}
"""
    write(repo, TOOLING_FILES[1], source)
    process = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/ci/zing/run_tooling_tests.py"),
            "--repo",
            str(repo),
            "--output-dir",
            str(repo.parent / "mutant-results"),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert process.returncode != 0, (
        "The empty-selector mutant must fail a real contract assertion"
    )
    receipt = result_json(repo.parent / "mutant-results/receipt.json")
    failed = [r for r in receipt["results"] if r["path"] == TOOLING_FILES[1]]
    assert len(failed) == 1 and failed[0]["failed"] == 1
    outcomes = needs([])
    outcomes["ci-tooling-contracts"]["result"] = "failure"
    process, output = summarize(repo, {"status": "success", "suites": []}, outcomes)
    assert process.returncode != 0
    assert "ci-tooling-contracts" in result_json(output)["errors"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))


def media_preparer():
    import runpy

    return runpy.run_path(str(ROOT / "scripts/ci/zing/prepare_media_tooling.py"))


def test_media_mirror_rewrite_preserves_signed_source_identity(tmp_path):
    helper = media_preparer()
    path = write(
        tmp_path,
        "sources.list.d/ubuntu.sources",
        """Types: deb
URIs: http://azure.archive.ubuntu.com/ubuntu/
Suites: noble noble-updates noble-security
Components: main universe
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
""",
    )
    original = path.read_text()
    assert helper["use_official_mirror"](tmp_path) == 1
    assert path.read_text() == original.replace(
        "http://azure.archive.ubuntu.com/ubuntu/", "https://archive.ubuntu.com/ubuntu/"
    )
    assert helper["use_official_mirror"](tmp_path) == 0
    custom = write(
        tmp_path, "sources.list", "deb https://example.invalid/ubuntu noble main\n"
    )
    assert helper["use_official_mirror"](tmp_path) == 0
    assert custom.read_text() == "deb https://example.invalid/ubuntu noble main\n"


def test_verified_media_tools_need_no_package_install(tmp_path):
    from types import SimpleNamespace

    calls = []

    def run(args, **options):
        calls.append((args, options))
        return SimpleNamespace(stdout=" V....D libx264 H.264 encoder\n")

    media_preparer()["prepare"](tmp_path, run)
    assert all(args[0] != "apt-get" for args, _ in calls)
    assert calls[-1][0] == ["ffmpeg", "-version"]
    assert not list(tmp_path.iterdir())


def test_media_install_has_deadlines_and_propagates_package_failure(tmp_path):
    helper = media_preparer()
    calls = []

    def run(args, **options):
        calls.append((args, options))
        if args[0] == "ffprobe":
            raise FileNotFoundError("not installed")
        raise subprocess.CalledProcessError(100, args)

    with pytest.raises(subprocess.CalledProcessError):
        helper["prepare"](tmp_path, run)
    apt, options = calls[-1]
    assert apt[0] == "apt-get" and apt[-1] == "update"
    assert "Acquire::Retries=1" in apt and "Acquire::https::Timeout=20" in apt
    assert options["check"] is True and options["timeout"] == 180
    assert not any("allow-unauthenticated" in arg for arg in apt)


@pytest.mark.parametrize(
    "filename", ["zing-cpu-suite.yml", "zing-conditioning-suite.yml"]
)
@pytest.mark.parametrize("exit_code", [0, 17])
def test_workflow_prepares_pinned_media_and_propagates_failure(
    tmp_path, filename, exit_code
):
    workflow = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())
    job = next(iter(workflow["jobs"].values()))
    step = next(s for s in job["steps"] if s.get("name") == "Prepare media tooling")
    assert step["timeout-minutes"] == 8
    assert "continue-on-error" not in step
    shell = step.get(
        "shell",
        job.get("defaults", {})
        .get("run", {})
        .get("shell", workflow.get("defaults", {}).get("run", {}).get("shell")),
    )
    assert shell in (None, "bash")
    # Actions' implicit Linux shell is bash -e; explicit bash also enables
    # pipefail. Do not grant the test stronger failure handling than production.
    command = ["bash", "-e"]
    if shell == "bash":
        command.extend(["-o", "pipefail"])

    binary = write(
        tmp_path,
        "pinned media/ffmpeg",
        '#!/bin/sh\n[ "$1" = "-version" ] || exit 99\n'
        f"echo pinned-media-fixture\nexit {exit_code}\n",
    )
    binary.chmod(0o755)
    # Only the wheel's binary locator is a fixture. Execute the actual workflow
    # shell and its filesystem/PATH handoff, including nonzero encoder exits.
    provider = write(
        tmp_path,
        "provider/imageio_ffmpeg.py",
        f"def get_ffmpeg_exe():\n    return {str(binary)!r}\n",
    )
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "python").symlink_to(sys.executable)
    for name in ["apt-get", "sudo"]:
        guard = write(tools, name, "#!/bin/sh\nexit 98\n")
        guard.chmod(0o755)
    runner = tmp_path / "runner"
    github_path = tmp_path / "github-path"
    result = subprocess.run(
        [*command, "-c", step["run"]],
        env={
            **os.environ,
            "PATH": str(tools) + os.pathsep + os.environ["PATH"],
            "PYTHONPATH": str(provider.parent),
            "RUNNER_TEMP": str(runner),
            "GITHUB_PATH": str(github_path),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == exit_code, result.stderr
    assert result.stdout.strip() == "pinned-media-fixture"
    assert (runner / "bin/ffmpeg").resolve() == binary
    assert github_path.read_text().splitlines() == [str(runner / "bin")]


def test_media_package_install_is_verified_before_business_can_start(tmp_path):
    from types import SimpleNamespace

    installed = False
    calls = []

    def run(args, **options):
        nonlocal installed
        calls.append((args, options))
        if args[0] == "apt-get" and "install" in args:
            installed = True
        elif args[0] == "ffprobe" and not installed:
            raise FileNotFoundError("not installed")
        return SimpleNamespace(stdout=" V....D libx264 H.264 encoder\n")

    media_preparer()["prepare"](tmp_path, run)
    apt_calls = [(args, options) for args, options in calls if args[0] == "apt-get"]
    assert len(apt_calls) == 2 and apt_calls[0][0][-1] == "update"
    assert apt_calls[1][0][-3:] == ["install", "--yes", "ffmpeg"]
    assert all(
        options["check"] and options["timeout"] == 180 for _, options in apt_calls
    )
    assert [args[0] for args, _ in calls[-3:]] == ["ffprobe", "ffmpeg", "ffmpeg"]


@pytest.mark.parametrize("scheme", ["mirror+file:", "mirror+file://"])
def test_media_setup_resolves_hosted_runner_mirror_lists_without_touching_custom_uris(
    tmp_path, scheme
):
    helper = media_preparer()
    before = f"""Types: deb
URIs: {scheme}/etc/apt/apt-mirrors.txt
Suites: noble noble-updates noble-security
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
"""
    path = write(tmp_path, "sources.list.d/runner.sources", before)
    security = write(
        tmp_path,
        "sources.list.d/security.list",
        f"deb {scheme}/etc/apt/apt-mirrors-security.txt noble-security main\n",
    )
    custom = write(
        tmp_path,
        "sources.list.d/custom.list",
        "deb mirror+file:/etc/apt/custom.txt noble main\n",
    )
    assert helper["use_official_mirror"](tmp_path) == 2
    assert path.read_text() == before.replace(
        f"{scheme}/etc/apt/apt-mirrors.txt", "https://archive.ubuntu.com/ubuntu"
    )
    assert (
        security.read_text()
        == "deb https://security.ubuntu.com/ubuntu noble-security main\n"
    )
    assert custom.read_text() == "deb mirror+file:/etc/apt/custom.txt noble main\n"
