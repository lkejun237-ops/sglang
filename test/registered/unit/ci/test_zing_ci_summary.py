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
    infrastructure = {"zing-changes", "zing-lint", "ci-tooling-contracts"}
    assert set(jobs) == infrastructure | set(SUITES) | {"zing-ci"}
    assert set(jobs["zing-ci"]["needs"]) == infrastructure | set(SUITES)
    assert jobs["zing-ci"]["if"] == "always()"
    for job_id in ("zing-lint", "ci-tooling-contracts"):
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
