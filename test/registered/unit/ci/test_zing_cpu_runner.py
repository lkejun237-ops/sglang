"""Real pytest children, strict receipts, timeout cleanup and isolation."""

import ast
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml
from zing_ci_fixtures import (
    BUSINESS,
    ROOT,
    TOOLING_FILES,
    cli,
    conditioning_fixture,
    entry,
    git,
    make_repo,
    manifest,
    result_json,
    run_cpu,
    run_tooling,
    write,
)

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, stage="base-a", runner_config="cpu")


def make_shard_fixture(tmp_path, file_count):
    repo = make_repo(tmp_path)
    paths = [
        f"benchmark/zing_h3_causal/test_shard_{index}.py" for index in range(file_count)
    ]
    for index, path in enumerate(paths):
        write(
            repo,
            path,
            f"from pathlib import Path\ndef test_cpu():\n    Path('executed-{index}.marker').write_text('real CPU case')\n",
        )
    # The approved input order deliberately differs from the deterministic
    # lexicographic order used to assign whole files to CI shards.
    entries = [entry(path, "dit-pipeline-contracts") for path in reversed(paths)]
    return repo, paths, entries


def launch_shard(
    repo,
    entries,
    index=0,
    count=1,
    suite="dit-pipeline-contracts",
    triage=False,
    recipes=None,
    label="shard",
):
    inventory = manifest(repo, entries, recipes)
    output = repo.parent / f"{label}-{index}-{count}"
    options = [
        "--manifest",
        inventory,
        "--suite",
        suite,
        "--shard-index",
        index,
        "--shard-count",
        count,
        "--output-dir",
        output,
    ]
    if triage:
        options.append("--triage")
    return cli("run_cpu_tests.py", repo, *options), output


def require_shard_cli(repo, entries):
    process, output = launch_shard(repo, entries, label="shard-probe")
    assert process.returncode == 0, process.stderr
    assert result_json(output / "receipt.json")["status"] == "success"
    for marker in repo.glob("executed-*.marker"):
        marker.unlink()


@pytest.mark.parametrize("file_count,shard_count", [(3, 2), (4, 4)])
def test_shards_execute_disjoint_complete_file_groups(
    tmp_path, file_count, shard_count
):
    repo, paths, entries = make_shard_fixture(tmp_path, file_count)
    observed = []
    for index in range(shard_count):
        process, output = launch_shard(repo, entries, index, shard_count)
        assert process.returncode == 0, process.stderr
        receipt = result_json(output / "receipt.json")
        selected = paths[index::shard_count]
        assert receipt["shard"] == {
            "index": index,
            "count": shard_count,
            "candidate_files": paths,
            "selected_files": selected,
        }
        actual = [record["path"] for record in receipt["results"]]
        assert actual == selected
        assert all(
            record["passed"] == 1 and not record["deselected"]
            for record in receipt["results"]
        )
        observed.extend(actual)
    assert sorted(observed) == paths
    assert len(set(observed)) == len(observed)


def test_default_unsharded_ci_still_executes_the_complete_file_list(tmp_path):
    repo, paths, entries = make_shard_fixture(tmp_path, 4)
    require_shard_cli(repo, entries)
    inventory = manifest(repo, entries)
    output = repo.parent / "default-unsharded"
    process = cli(
        "run_cpu_tests.py",
        repo,
        "--manifest",
        inventory,
        "--suite",
        "dit-pipeline-contracts",
        "--output-dir",
        output,
    )
    assert process.returncode == 0, process.stderr
    receipt = result_json(output / "receipt.json")
    assert sorted(record["path"] for record in receipt["results"]) == paths
    assert sum(record["passed"] for record in receipt["results"]) == 4
    assert receipt["shard"] == {
        "index": 0,
        "count": 1,
        "candidate_files": paths,
        "selected_files": paths,
    }


@pytest.mark.parametrize(
    "index,count,suite,triage",
    [
        (-1, 2, "dit-pipeline-contracts", False),
        (2, 2, "dit-pipeline-contracts", False),
        (0, 0, "dit-pipeline-contracts", False),
        (0, -1, "dit-pipeline-contracts", False),
        (3, 4, "dit-pipeline-contracts", False),
        (0, 2, "dit-pipeline-contracts", True),
        (0, 2, "conditioning-contracts", False),
        (0, 2, "all", False),
    ],
)
def test_sharding_rejects_bad_bounds_empty_and_incomplete_execution_modes(
    tmp_path, index, count, suite, triage
):
    repo, _, entries = make_shard_fixture(tmp_path, 3)
    require_shard_cli(repo, entries)
    recipes = None
    if suite == "conditioning-contracts":
        condition_entries, recipes = conditioning_fixture(repo)
        entries += condition_entries
    process, output = launch_shard(repo, entries, index, count, suite, triage, recipes)
    assert process.returncode != 0
    receipt = result_json(output / "receipt.json")
    assert receipt["status"] == "error"
    assert receipt["results"] == []
    assert not list(repo.glob("executed-*.marker"))


@pytest.mark.parametrize("status", ["UNRUN", "FAILED"])
def test_sharding_checks_approval_for_files_hidden_in_other_shards(tmp_path, status):
    repo, paths, entries = make_shard_fixture(tmp_path, 3)
    require_shard_cli(repo, entries)
    for candidate in entries:
        if candidate["path"] == paths[1]:
            candidate.update(status=status, approved=False)
    process, output = launch_shard(repo, entries, 0, 2)
    assert process.returncode != 0
    receipt = result_json(output / "receipt.json")
    assert receipt["status"] == "error"
    assert paths[1] in receipt["error"]
    assert receipt["results"] == []
    assert not list(repo.glob("executed-*.marker"))


def test_workflow_dit_matrix_executes_all_four_unique_shards(tmp_path):
    workflow = yaml.safe_load((ROOT / ".github/workflows/zing-ci.yml").read_text())
    job = workflow["jobs"]["dit-pipeline-contracts"]
    strategy = job.get("strategy", {})
    assert strategy.get("matrix", {}).get("shard") == [0, 1, 2, 3]
    assert strategy.get("fail-fast") is False
    for name in (
        "decoder-contracts",
        "adaln-export-contracts",
        "conditioning-contracts",
    ):
        assert "matrix" not in workflow["jobs"][name].get("strategy", {})
    callee = yaml.safe_load((ROOT / job["uses"]).read_text())["jobs"]["cpu-suite"]
    run = next(
        step["run"]
        for step in callee["steps"]
        if "run_cpu_tests.py" in step.get("run", "")
    )
    upload = next(
        step
        for step in callee["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    observed, artifacts, output_paths = [], [], []
    for index in strategy["matrix"]["shard"]:
        # Each reusable invocation has an independent runner workspace.
        workspace = tmp_path / str(index)
        workspace.mkdir()
        repo, paths, entries = make_shard_fixture(workspace, 4)
        manifest(repo, entries)
        inputs = {
            key: str(value).replace("${{ matrix.shard }}", str(index))
            for key, value in job["with"].items()
        }
        command = run
        artifact = upload["with"]["name"]
        for key, value in inputs.items():
            expression = "${{ inputs." + key + " }}"
            artifact = artifact.replace(expression, value)
            for variable, expression_value in callee["env"].items():
                if expression_value == expression:
                    command = command.replace("$" + variable, value)
        arguments = shlex.split(command)
        assert arguments[:2] == ["python", "scripts/ci/zing/run_cpu_tests.py"]
        process = cli("run_cpu_tests.py", repo, *arguments[2:])
        assert process.returncode == 0, process.stderr
        output = repo / arguments[arguments.index("--output-dir") + 1]
        receipt = result_json(output / "receipt.json")
        assert receipt["shard"]["index"] == index and receipt["shard"]["count"] == 4
        assert receipt["shard"]["candidate_files"] == paths
        assert receipt["shard"]["selected_files"] == [paths[index]]
        assert all(
            record["passed"] == 1 and not record["deselected"]
            for record in receipt["results"]
        )
        observed.extend(receipt["shard"]["selected_files"])
        artifacts.append(artifact)
        output_paths.append(str(output))
    assert sorted(observed) == paths and len(set(observed)) == 4
    assert len(set(artifacts)) == 4 and len(set(output_paths)) == 4


def test_success_has_real_junit_environment_and_uncommitted_fingerprints(tmp_path):
    repo = make_repo(tmp_path)
    source = write(repo, BUSINESS, "def test_cpu():\n    assert 2 + 2 == 4\n")
    process, output = run_cpu(repo)
    assert process.returncode == 0, process.stderr
    receipt = result_json(output / "receipt.json")
    record = receipt["results"][0]
    assert record["path"] == BUSINESS
    assert record["passed"] == 1 and record["collected"] == 1
    assert record["skipped"] == 0 and record["exit_code"] == 0
    assert ET.parse(record["junit_path"]).findall(".//testcase")
    assert record["log_path"] and os.path.isfile(record["log_path"])
    assert receipt["environment"]["python"] == sys.version.split()[0]
    assert record["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(receipt["manifest_sha256"]) == 64
    assert len(receipt["candidate_fingerprint"]) == 64
    assert receipt["tooling_sha256"]
    source.write_text(source.read_text() + "\n# uncommitted candidate change\n")
    second_process, second_output = run_cpu(repo)
    assert second_process.returncode == 0
    second = result_json(second_output / "receipt.json")
    assert second["checkout_sha"] == receipt["checkout_sha"]
    assert second["candidate_fingerprint"] != receipt["candidate_fingerprint"]
    assert second["results"][0]["source_sha256"] != record["source_sha256"]


@pytest.mark.parametrize(
    "content,reason",
    [
        ("def test_failure():\n    assert False\n", "TEST_FAILED"),
        ("VALUE = 3\n", "ZERO_CASES"),
        (
            "import pytest\ndef test_skip():\n    pytest.skip('unreviewed')\n",
            "UNKNOWN_SKIP",
        ),
        ("import nonexistent_issue277_dependency\n", "COLLECTION_FAILED"),
    ],
)
def test_failures_never_become_cpu_pass(tmp_path, content, reason):
    repo = make_repo(tmp_path)
    write(repo, BUSINESS, content)
    process, output = run_cpu(repo)
    assert process.returncode != 0
    record = result_json(output / "receipt.json")["results"][0]
    assert record["status"] == "FAILED"
    assert record["reason_code"] == reason
    assert not record.get("approved", False)


@pytest.mark.parametrize(
    "conditioning", [False, True], ids=["ordinary", "conditioning"]
)
def test_early_success_exit_cannot_attest_complete_execution(tmp_path, conditioning):
    repo = make_repo(tmp_path)
    entries, recipes = conditioning_fixture(repo) if conditioning else (None, None)
    path = entries[0]["path"] if conditioning else BUSINESS
    write(
        repo,
        path,
        "import pytest\nfrom pathlib import Path\n"
        "def test_pass():\n    assert True\n"
        "def test_abort():\n    pytest.exit('early successful stop', returncode=0)\n"
        "def test_missing():\n    Path('MISSING_CASE_EXECUTED').write_text('bad')\n    assert False\n",
    )
    process, output = run_cpu(
        repo,
        entries,
        "conditioning-contracts" if conditioning else "benchmark-contracts",
        recipes,
    )
    receipt = result_json(output / "receipt.json")
    record = receipt["results"][0]
    assert record["passed"] == 1 and record["failed"] == 0
    assert record["exit_code"] == 0
    assert not (repo / "MISSING_CASE_EXECUTED").exists()
    assert ET.parse(record["junit_path"]).findall(".//testcase")
    assert process.returncode != 0, f"Early exit falsely passed: {output}"
    assert receipt["status"] == "error"
    assert record["status"] == "FAILED"
    assert record["reason_code"] == "INCOMPLETE_EXECUTION"
    assert record["completed_nodeids"] == [path + "::test_pass"]
    assert set(record["incomplete_nodeids"]) == set(record["nodeids"]) - set(
        record["completed_nodeids"]
    )
    assert {path + "::test_abort", path + "::test_missing"} <= set(
        record["incomplete_nodeids"]
    )


@pytest.mark.parametrize(
    "conditioning", [False, True], ids=["ordinary", "conditioning"]
)
def test_whole_module_collection_skip_is_not_a_cpu_pass(tmp_path, conditioning):
    repo = make_repo(tmp_path)
    entries, recipes = conditioning_fixture(repo) if conditioning else (None, None)
    path = entries[1]["path"] if conditioning else BUSINESS
    reason = "missing optional CPU dependency"
    write(
        repo,
        path,
        "import pytest\npytest.skip('missing optional CPU dependency', allow_module_level=True)\n"
        "def test_real_cpu_case():\n    assert False\n",
    )
    process, output = run_cpu(
        repo,
        entries,
        "conditioning-contracts" if conditioning else "benchmark-contracts",
        recipes,
    )
    receipt = result_json(output / "receipt.json")
    record = receipt["results"][0]
    junit_skips = ET.parse(record["junit_path"]).findall(".//testcase/skipped")
    assert len(junit_skips) == 1
    assert reason in "".join(junit_skips[0].itertext())
    assert record["passed"] == (2 if conditioning else 0)
    assert record["failed"] == 0
    assert process.returncode != 0, f"Collection skip falsely passed: {output}"
    assert receipt["status"] == "error"
    assert record["status"] == "FAILED"
    assert record["reason_code"] == "UNKNOWN_COLLECTION_SKIP"
    assert record["collection_skips"] == [{"nodeid": path, "reason": reason}]


@pytest.mark.parametrize("empty_index", [0, 2], ids=["default-root", "registered-root"])
def test_conditioning_empty_required_file_cannot_hide_behind_other_cpu_passes(
    tmp_path, empty_index
):
    repo = make_repo(tmp_path)
    entries, recipes = conditioning_fixture(repo)
    path = entries[empty_index]["path"]
    write(repo, path, "VALUE = 42\n")
    process, output = run_cpu(repo, entries, "conditioning-contracts", recipes)
    receipt = result_json(output / "receipt.json")
    record = receipt["results"][0]
    assert record["collected"] == record["passed"] == 2
    assert len(ET.parse(record["junit_path"]).findall(".//testcase")) == 2
    assert process.returncode != 0, f"Empty required file falsely passed: {output}"
    assert receipt["status"] == "error"
    assert record["status"] == "FAILED"
    assert record["reason_code"] == "INCOMPLETE_FILE_COVERAGE"
    assert record["uncollected_files"] == [path]


@pytest.mark.parametrize(
    "conditioning", [False, True], ids=["ordinary", "conditioning"]
)
def test_completed_stdlib_subtests_are_not_incomplete_execution(tmp_path, conditioning):
    repo = make_repo(tmp_path)
    entries, recipes = conditioning_fixture(repo) if conditioning else (None, None)
    path = entries[0]["path"] if conditioning else BUSINESS
    write(
        repo,
        path,
        "import unittest\nclass TestCPU(unittest.TestCase):\n"
        "    def test_subtests(self):\n"
        "        for value in (1, 2, 3):\n"
        "            with self.subTest(value=value):\n"
        "                self.assertGreater(value, 0)\n",
    )
    process, output = run_cpu(
        repo,
        entries,
        "conditioning-contracts" if conditioning else "benchmark-contracts",
        recipes,
    )
    assert process.returncode == 0, process.stderr
    record = result_json(output / "receipt.json")["results"][0]
    assert record["status"] == "CPU_PASS"
    assert record["collected"] == record["passed"] == (3 if conditioning else 1)
    assert record["failed"] == record["skipped"] == 0
    assert ET.parse(record["junit_path"]).findall(".//testcase")


def test_conftest_import_error_without_junit_is_collection_failure(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, BUSINESS, "def test_cpu():\n    assert True\n")
    write(
        repo,
        "benchmark/zing_h3_causal/conftest.py",
        "import nonexistent_issue277_conftest_dependency\n",
    )
    process, output = run_cpu(repo)
    assert process.returncode != 0
    record = result_json(output / "receipt.json")["results"][0]
    assert record["exit_code"] == 4
    assert record["reason_code"] == "COLLECTION_FAILED"
    assert record["collected"] == 0 and record["passed"] == 0
    log = open(record["log_path"]).read()
    assert "ImportError while loading conftest" in log
    assert "nonexistent_issue277_conftest_dependency" in log


@pytest.mark.parametrize("source", ["pytest-ini", "environment"])
def test_partial_deselection_cannot_silently_reduce_cpu_coverage(tmp_path, source):
    repo = make_repo(tmp_path)
    write(
        repo,
        BUSINESS,
        "def test_retained():\n    assert 2 + 2 == 4\n"
        "def test_filtered():\n    assert 6 * 7 == 42\n",
    )
    if source == "pytest-ini":
        config = repo / "test/pytest.ini"
        config.write_text(config.read_text() + "\naddopts = -k retained\n")
    inventory = manifest(repo)
    output = repo.parent / "partial-deselection-results"
    process = cli(
        "run_cpu_tests.py",
        repo,
        "--manifest",
        inventory,
        "--suite",
        "benchmark-contracts",
        "--output-dir",
        output,
        env={"PYTEST_ADDOPTS": "-k retained" if source == "environment" else ""},
    )
    record = result_json(output / "receipt.json")["results"][0]
    # Establish that the omitted case is the issue, not an empty suite or a
    # broken fixture: pytest genuinely ran and passed the retained CPU case.
    assert record["passed"] == 1
    assert record["deselected"] == [BUSINESS + "::test_filtered"]
    assert process.returncode != 0, "Partial deselection must not become CPU_PASS"
    assert record["status"] == "FAILED"
    assert record["reason_code"] == "UNREVIEWED_DESELECTION"
    assert os.path.isfile(record["log_path"])


def test_case_level_known_skip_requires_a_real_cpu_pass(tmp_path):
    repo = make_repo(tmp_path)
    write(
        repo,
        BUSINESS,
        "import pytest\ndef test_cpu():\n    assert True\ndef test_cuda():\n    pytest.skip('CUDA required')\n",
    )
    allowed = [{"nodeid": BUSINESS + "::test_cuda", "reason": "CUDA required"}]
    process, output = run_cpu(repo, [entry(allowed_skips=allowed)])
    assert process.returncode == 0, process.stderr
    record = result_json(output / "receipt.json")["results"][0]
    assert record["passed"] == 1 and record["skipped"] == 1
    assert record["skip_nodeids"] == [BUSINESS + "::test_cuda"]


def test_known_skip_with_wrong_reason_is_rejected(tmp_path):
    repo = make_repo(tmp_path)
    write(
        repo,
        BUSINESS,
        "import pytest\ndef test_cpu():\n    assert True\ndef test_cuda():\n    pytest.skip('missing dependency')\n",
    )
    process, output = run_cpu(
        repo,
        [
            entry(
                allowed_skips=[
                    {"nodeid": BUSINESS + "::test_cuda", "reason": "CUDA required"}
                ]
            )
        ],
    )
    assert process.returncode != 0
    assert (
        result_json(output / "receipt.json")["results"][0]["reason_code"]
        == "UNKNOWN_SKIP"
    )


def test_only_known_skips_cannot_claim_real_cpu_execution(tmp_path):
    repo = make_repo(tmp_path)
    write(
        repo,
        BUSINESS,
        "import pytest\ndef test_cuda():\n    pytest.skip('CUDA required')\n",
    )
    allowed = [{"nodeid": BUSINESS + "::test_cuda", "reason": "CUDA required"}]
    process, output = run_cpu(repo, [entry(allowed_skips=allowed)])
    assert process.returncode != 0
    record = result_json(output / "receipt.json")["results"][0]
    assert record["passed"] == 0 and record["skipped"] == 1
    assert record["status"] == "FAILED"


def test_timeout_preserves_current_case_and_stack_in_log(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, BUSINESS, "import time\ndef test_stalled_case():\n    time.sleep(30)\n")
    process, output = run_cpu(repo, timeout=3)
    assert process.returncode != 0
    record = result_json(output / "receipt.json")["results"][0]
    assert record["reason_code"] == "TIMEOUT"
    log = Path(record["log_path"]).read_text()
    assert "test_stalled_case" in log and "Timeout (" in log
    assert "in test_stalled_case" in log
    assert log.count("Timeout (") == 1


def test_timeout_kills_descendant_before_it_can_write(tmp_path):
    repo = make_repo(tmp_path)
    pid_file = repo / "grandchild.pid"
    marker = repo / "leaked-child"
    child = f"import os,time; from pathlib import Path; Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(3); Path({str(marker)!r}).write_text('leaked'); time.sleep(20)"
    write(
        repo,
        BUSINESS,
        f"import subprocess,sys,time\ndef test_hang():\n    subprocess.Popen([sys.executable, '-c', {child!r}])\n    time.sleep(30)\n",
    )
    process, output = run_cpu(repo, timeout=1)
    try:
        assert process.returncode != 0
        record = result_json(output / "receipt.json")["results"][0]
        assert record["reason_code"] == "TIMEOUT"
        assert record["process_tree_cleaned"] is True
        assert pid_file.exists(), "The hanging fixture must actually spawn a descendant"
        time.sleep(3.1)
        assert not marker.exists(), "A descendant survived the runner timeout"
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_ci_refuses_unapproved_but_triage_runs_twice_and_preserves_status(tmp_path):
    repo = make_repo(tmp_path)
    write(
        repo,
        BUSINESS,
        "import os\nfrom pathlib import Path\ndef test_pid():\n    with Path('trial-pids').open('a') as log:\n        log.write(str(os.getpid()) + '\\n')\n",
    )
    entries = [entry(status="UNRUN", approved=False)]
    process, _ = run_cpu(repo, entries)
    assert process.returncode != 0
    assert not (repo / "trial-pids").exists()
    inventory = manifest(repo, entries)
    before = inventory.read_bytes()
    output = repo.parent / "triage-results"
    process = cli(
        "run_cpu_tests.py",
        repo,
        "--manifest",
        inventory,
        "--suite",
        "benchmark-contracts",
        "--triage",
        "--trials",
        "2",
        "--output-dir",
        output,
    )
    assert process.returncode == 0, process.stderr
    pids = (repo / "trial-pids").read_text().splitlines()
    assert len(pids) == 2 and len(set(pids)) == 2
    receipt = result_json(output / "receipt.json")
    assert len(receipt["results"]) == 2
    assert all(r["passed"] == 1 for r in receipt["results"])
    assert inventory.read_bytes() == before, (
        "trial evidence must not silently approve entries"
    )


def test_triage_continues_after_failure_but_returns_nonzero(tmp_path):
    repo = make_repo(tmp_path)
    second = "benchmark/zing_h3_causal/test_second.py"
    write(repo, BUSINESS, "def test_bad():\n    assert False\n")
    write(
        repo,
        second,
        "from pathlib import Path\ndef test_after():\n    Path('continued').write_text('yes')\n",
    )
    inventory = manifest(
        repo,
        [
            entry(status="UNRUN", approved=False),
            entry(second, status="UNRUN", approved=False),
        ],
    )
    output = repo.parent / "triage-results"
    process = cli(
        "run_cpu_tests.py",
        repo,
        "--manifest",
        inventory,
        "--suite",
        "benchmark-contracts",
        "--triage",
        "--output-dir",
        output,
    )
    assert process.returncode != 0
    assert (repo / "continued").exists()
    assert len(result_json(output / "receipt.json")["results"]) == 2


@pytest.mark.parametrize("problem", ["python", "dependency", "missing-lock"])
def test_explicit_recipe_rejects_python_or_dependency_drift(tmp_path, problem):
    repo = make_repo(tmp_path)
    write(
        repo,
        BUSINESS,
        "from pathlib import Path\ndef test_cpu():\n    Path('RECIPE_TEST_EXECUTED').write_text('yes')\n",
    )
    lock = "scripts/ci/zing/requirements-fixture.txt"
    write(repo, lock, "pytest==" + pytest.__version__ + "\n")
    recipe = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "requirements": lock,
    }
    if problem == "python":
        recipe["python_version"] = "3.99"
    elif problem == "dependency":
        write(repo, lock, "pytest==0.0.1\n")
    else:
        (repo / lock).unlink()
    process, output = run_cpu(repo, recipes={"benchmark-contracts": recipe})
    assert process.returncode != 0
    assert result_json(output / "receipt.json")["status"] == "error"
    assert not (repo / "RECIPE_TEST_EXECUTED").exists()


def test_explicit_recipe_records_the_actual_dependency_lock_hash(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, BUSINESS, "def test_cpu():\n    assert True\n")
    lock = "scripts/ci/zing/requirements-fixture.txt"
    source = write(repo, lock, "pytest==" + pytest.__version__ + "\n")
    recipe = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "requirements": lock,
    }
    process, output = run_cpu(repo, recipes={"benchmark-contracts": recipe})
    assert process.returncode == 0, process.stderr
    receipt = result_json(output / "receipt.json")
    assert "environment_recipes" in receipt
    evidence = receipt["environment_recipes"]["benchmark-contracts"]
    assert (
        evidence["requirements_sha256"][lock]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert evidence["dependencies"]["pytest"] == pytest.__version__


@pytest.mark.parametrize("cuda_value", ["", "2,3"])
def test_verified_cpu_recipe_removes_only_empty_cuda_visibility(tmp_path, cuda_value):
    repo = make_repo(tmp_path)
    write(
        repo,
        BUSINESS,
        "import json,os\nfrom pathlib import Path\ndef test_cpu_environment():\n"
        "    device = os.environ.get('CUDA_VISIBLE_DEVICES', '0')[0]\n"
        "    Path('cpu-environment-observed.json').write_text(json.dumps({\n"
        "        'device': device, 'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'),\n"
        "        'nvidia': os.environ.get('NVIDIA_VISIBLE_DEVICES')}))\n",
    )
    # Tooling has no torch dependency. Mock its metadata boundary only; the
    # declared CPU recipe is validated and the child runs real stdlib cases.
    startup = tmp_path / "cpu metadata"
    startup.mkdir()
    write(
        startup,
        "sitecustomize.py",
        "import importlib.metadata\n"
        "original_version = importlib.metadata.version\n"
        "def fixture_version(name):\n"
        "    return '2.8.0+cpu' if name == 'torch' else original_version(name)\n"
        "importlib.metadata.version = fixture_version\n",
    )
    lock = "scripts/ci/zing/requirements-fixture.txt"
    write(repo, lock, "torch==2.8.0+cpu\n")
    recipe = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "requirements": lock,
    }
    inventory = manifest(repo, recipes={"benchmark-contracts": recipe})
    output = repo.parent / "cpu-environment-results"
    process = cli(
        "run_cpu_tests.py",
        repo,
        "--manifest",
        inventory,
        "--suite",
        "benchmark-contracts",
        "--output-dir",
        output,
        env={
            "PYTHONPATH": str(startup) + os.pathsep + str(repo / "python"),
            "CUDA_VISIBLE_DEVICES": cuda_value,
            "NVIDIA_VISIBLE_DEVICES": "none",
        },
    )
    assert process.returncode == 0, process.stderr
    assert result_json(output / "receipt.json")["results"][0]["passed"] == 1
    observed = json.loads((repo / "cpu-environment-observed.json").read_text())
    assert observed["device"] == (cuda_value[0] if cuda_value else "0")
    assert observed["cuda"] == (cuda_value if cuda_value else None)
    assert observed["nvidia"] == "none"


def test_conditioning_expands_default_roots_and_junit_in_child(tmp_path):
    repo = make_repo(tmp_path)
    entries, recipes = conditioning_fixture(repo)
    process, output = run_cpu(repo, entries, "conditioning-contracts", recipes)
    assert process.returncode == 0, process.stderr
    assert not (repo / "PARENT_IMPORT_FORBIDDEN").exists()
    receipt = result_json(output / "receipt.json")
    assert sum(r["passed"] for r in receipt["results"]) == 3
    command = receipt["results"][0]["command"]
    for root in recipes["conditioning-contracts"]["default_roots"]:
        assert any(
            str(repo / root) == argument or argument == root for argument in command
        )
    assert any(arg.startswith("--junitxml") for arg in command)
    assert (
        receipt["conditioning"]["history_base"]
        == recipes["conditioning-contracts"]["history_base"]
    )
    assert len(receipt["conditioning"]["environment_lock_sha256"]) == 64


@pytest.mark.parametrize("package", ["torch", "torchvision"])
def test_conditioning_rejects_accelerator_wheel_before_business_starts(
    tmp_path, package
):
    repo = make_repo(tmp_path)
    entries, recipes = conditioning_fixture(repo)
    for item in entries:
        write(
            repo,
            item["path"],
            "from pathlib import Path\ndef test_cpu():\n    Path('CUDA_WHEEL_BUSINESS_EXECUTED').write_text('bad')\n",
        )
    lock = repo / recipes["conditioning-contracts"]["environment_lock"]
    lock.write_text('{"dependencies": {"' + package + '": "2.14.1"}}\n')
    # Mock only the metadata lookup in the external CLI's Python startup.
    # The three actual pytest cases stay real, with no GPU module import and
    # no global sys.platform mutation. Existing code wrongly strips +cu128.
    startup = tmp_path / "metadata boundary"
    startup.mkdir()
    write(
        startup,
        "sitecustomize.py",
        f"import importlib.metadata\n"
        f"original_version = importlib.metadata.version\n"
        f"def fixture_version(name):\n"
        f"    return '2.14.1+cu128' if name == {package!r} else original_version(name)\n"
        f"importlib.metadata.version = fixture_version\n",
    )
    inventory = manifest(repo, entries, recipes)
    output = repo.parent / "accelerator-wheel-results"
    process = cli(
        "run_cpu_tests.py",
        repo,
        "--manifest",
        inventory,
        "--suite",
        "conditioning-contracts",
        "--output-dir",
        output,
        env={"PYTHONPATH": str(startup) + os.pathsep + str(repo / "python")},
    )
    assert process.returncode != 0, "An accelerator wheel cannot satisfy a CPU recipe"
    assert result_json(output / "receipt.json")["status"] == "error"
    assert not (repo / "CUDA_WHEEL_BUSINESS_EXECUTED").exists()


@pytest.mark.parametrize(
    "missing",
    ["runner", "environment_lock", "history_base", "default_root", "recipe_root"],
)
def test_conditioning_missing_dependencies_or_roots_fail_closed(tmp_path, missing):
    repo = make_repo(tmp_path)
    entries, recipes = conditioning_fixture(repo)
    recipe = recipes["conditioning-contracts"]
    if missing in {"runner", "environment_lock"}:
        (repo / recipe[missing]).unlink()
    elif missing == "history_base":
        recipe["history_base"] = "f" * 40
    elif missing == "default_root":
        (repo / recipe["default_roots"][0]).unlink()
    else:
        recipe["default_roots"].pop()
    process, output = run_cpu(repo, entries, "conditioning-contracts", recipes)
    assert process.returncode != 0
    assert result_json(output / "receipt.json")["status"] == "error"
    assert not (repo / "PARENT_IMPORT_FORBIDDEN").exists()


@pytest.mark.parametrize("baseline_present", [True, False])
def test_workflow_baseline_uses_checkout_objects_without_origin_credentials(
    tmp_path, baseline_present
):
    baseline = "b1899c0bd538a342056a39893e8ce048b17afa32"
    integration = "8b4bf2c3d9f5ceeb2b8bc9c20af7afbba6f24982"
    workflow = yaml.safe_load((ROOT / ".github/workflows/zing-ci.yml").read_text())
    caller = workflow["jobs"]["conditioning-contracts"]
    callee = yaml.safe_load((ROOT / caller["uses"]).read_text())
    steps = callee["jobs"]["conditioning-suite"]["steps"]
    baseline_steps = [step for step in steps if baseline in step.get("run", "")]
    assert len(baseline_steps) == 1
    repo = make_repo(tmp_path)
    git(repo, "remote", "add", "origin", str(tmp_path / "unreachable-private-origin"))
    if baseline_present:
        common = git(ROOT, "rev-parse", "--git-common-dir").stdout.strip()
        objects = (ROOT / common).resolve() / "objects"
        alternates = repo / ".git/objects/info/alternates"
        alternates.parent.mkdir(parents=True, exist_ok=True)
        alternates.write_text(str(objects) + "\n")
        git(repo, "cat-file", "-e", baseline + "^{commit}")
        git(repo, "update-ref", "HEAD", integration)
    else:
        assert (
            git(repo, "cat-file", "-e", baseline + "^{commit}", check=False).returncode
            != 0
        )
    process = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", baseline_steps[0]["run"]],
        cwd=repo,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    if baseline_present:
        assert process.returncode == 0, process.stderr
    else:
        assert process.returncode != 0, "A missing baseline must still fail closed"


def test_tooling_runs_all_fixed_files_when_detector_is_empty_or_failed(tmp_path):
    repo = make_repo(tmp_path)
    process, output = run_tooling(
        repo, env={"ZING_SELECTED_SUITES": "[]", "ZING_DETECTOR_RESULT": "failure"}
    )
    assert process.returncode == 0, process.stderr
    records = result_json(output / "receipt.json")["results"]
    assert [r["path"] for r in records] == TOOLING_FILES
    assert all(r["collected"] == 1 and r["passed"] == 1 for r in records)
    assert all(ET.parse(r["junit_path"]).findall(".//testcase") for r in records)


@pytest.mark.parametrize(
    "bad", ["failure", "skip", "empty", "missing", "no-junit", "invalid-junit"]
)
def test_tooling_rejects_false_green_and_retains_evidence(tmp_path, bad):
    repo = make_repo(tmp_path)
    path = TOOLING_FILES[0]
    content = {
        "failure": "def test_bad():\n    assert False\n",
        "skip": "import pytest\ndef test_skip():\n    pytest.skip('not allowed')\n",
        "empty": "VALUE = 42\n",
        "no-junit": "import os\nos._exit(0)\n",
        "invalid-junit": "import os,sys\nfrom pathlib import Path\nfor arg in sys.argv:\n    if arg.startswith('--junitxml='):\n        Path(arg.split('=',1)[1]).write_text('not xml')\nos._exit(0)\n",
    }
    if bad == "missing":
        (repo / path).unlink()
    else:
        write(repo, path, content[bad])
    process, output = run_tooling(repo)
    assert process.returncode != 0
    receipt = result_json(output / "receipt.json")
    assert receipt["status"] == "error"
    assert any(
        r["path"] == path and r["status"] == "FAILED" for r in receipt["results"]
    )


@pytest.mark.parametrize("name", ["api", "response_validation", "worker"])
@pytest.mark.parametrize("outcome,exit_code", [("pass", 0), ("fail", 1), ("empty", 5)])
def test_conditioning_direct_entrypoint_preserves_pytest_exit(
    tmp_path, name, outcome, exit_code
):
    relative = f"test/registered/unit/multimodal_gen/test_conditioning_rpc_{name}.py"
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    entrypoint = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test).replace("'", '"') == '__name__ == "__main__"'
    )
    # Execute the real entry block in a subprocess. Only the heavy environment
    # bootstrap is replaced; pytest collection, assertions and exit are real.
    write(
        tmp_path,
        "benchmark/zing_h3_causal/run_conditioning_cpu_tests.py",
        "import os\nos.environ['CONDITIONING_BOOTSTRAPPED'] = '1'\n"
        "if __name__ == '__main__':\n    raise AssertionError('nested pytest run')\n",
    )
    test_file = write(
        tmp_path,
        relative,
        ast.unparse(entrypoint) + "\n"
        "import os\n"
        "def test_probe():\n"
        "    assert os.environ['CONDITIONING_BOOTSTRAPPED'] == '1'\n"
        f"    assert {outcome != 'fail'}\n",
    )
    arguments = ["-q", "-f"]
    if outcome == "empty":
        arguments += ["-k", "no_such_test"]
    result = subprocess.run(
        [sys.executable, str(test_file), *arguments],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert {0: "1 passed", 1: "1 failed", 5: "1 deselected"}[exit_code] in result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
