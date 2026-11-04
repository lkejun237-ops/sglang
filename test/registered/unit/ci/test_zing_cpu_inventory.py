"""Inventory is a completeness gate, never an alternative test runner."""

import pytest
from zing_ci_fixtures import (
    BUSINESS,
    TOOLING_FILES,
    UNIT,
    cli,
    entry,
    make_repo,
    manifest,
    result_json,
    write,
)

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, stage="base-a", runner_config="cpu")


def check(repo, entries):
    inventory = manifest(repo, entries)
    output = repo.parent / "inventory-results"
    process = cli(
        "run_cpu_tests.py",
        repo,
        "--check-inventory",
        "--manifest",
        inventory,
        "--output-dir",
        output,
    )
    return process, output


def test_inventory_only_counts_candidates_without_starting_business(tmp_path):
    repo = make_repo(tmp_path)
    write(
        repo,
        BUSINESS,
        "from pathlib import Path\nPath('BUSINESS_EXECUTED').write_text('bad')\ndef test_real():\n    assert False\n",
    )
    process, output = check(repo, [entry()])
    assert process.returncode == 0, process.stderr
    receipt = result_json(output / "receipt.json")
    assert receipt["inventory"]["discovered"] == [BUSINESS]
    assert receipt["inventory"]["tooling_files"] == TOOLING_FILES
    assert receipt["results"] == []
    assert not (repo / "BUSINESS_EXECUTED").exists()


@pytest.mark.parametrize(
    "unknown",
    [
        UNIT + "/test_zing_new.py",
        UNIT + "/realtime/test_zing_new.py",
        UNIT + "/realtime/test_worker_connections.py",
        UNIT + "/realtime/test_cleanup_watchdog.py",
        UNIT + "/conditioning/test_generic_name.py",
        UNIT + "/test_zing_conditioning_new.py",
        "test/registered/unit/multimodal_gen/test_generic_name.py",
        "test/registered/unit/zing/test_zing_new.py",
        "benchmark/zing_h3_causal/test_new.py",
    ],
)
def test_unknown_new_candidate_is_not_silently_lost(tmp_path, unknown):
    repo = make_repo(tmp_path)
    write(repo, BUSINESS, "def test_ok():\n    assert True\n")
    write(repo, unknown, "def test_new():\n    assert True\n")
    process, output = check(repo, [entry()])
    assert process.returncode != 0
    receipt = result_json(output / "receipt.json")
    assert unknown in receipt["inventory"]["unknown"]


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate",
        "missing",
        "unrun-approved",
        "failed-approved",
        "unknown-status",
        "no-source",
    ],
)
def test_invalid_manifest_is_rejected(tmp_path, problem):
    repo = make_repo(tmp_path)
    write(repo, BUSINESS, "def test_ok():\n    assert True\n")
    entries = [entry()]
    if problem == "duplicate":
        entries.append(entry())
    elif problem == "missing":
        entries[0]["path"] = "benchmark/zing_h3_causal/test_disappeared.py"
    elif problem == "unrun-approved":
        entries[0]["status"] = "UNRUN"
    elif problem == "failed-approved":
        entries[0]["status"] = "FAILED"
    elif problem == "unknown-status":
        entries[0]["status"] = "LOOKS_OK"
    else:
        entries[0]["sources"] = []
    process, output = check(repo, entries)
    assert process.returncode != 0
    assert result_json(output / "receipt.json")["status"] == "error"


def test_unrun_candidates_are_accounted_for_without_claiming_approval(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, BUSINESS, "def test_ok():\n    assert True\n")
    process, output = check(repo, [entry(status="UNRUN", approved=False)])
    assert process.returncode == 0
    inventory = result_json(output / "receipt.json")["inventory"]
    assert inventory["approved"] == []
    assert inventory["unapproved"] == [BUSINESS]


def test_two_source_overlap_is_one_file_and_source_metadata_is_retained(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, BUSINESS, "def test_overlap():\n    assert True\n")
    process, output = check(repo, [entry(sources=["main", "conditioning"])])
    assert process.returncode == 0
    inventory = result_json(output / "receipt.json")["inventory"]
    assert inventory["discovered"] == [BUSINESS]
    assert inventory["sources"][BUSINESS] == ["main", "conditioning"]


def test_archive_results_are_excluded_from_discovery(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, BUSINESS, "def test_ok():\n    assert True\n")
    write(
        repo,
        "benchmark/zing_h3_causal/results/old/test_archived.py",
        "raise RuntimeError('archive')\n",
    )
    process, output = check(repo, [entry()])
    assert process.returncode == 0
    assert result_json(output / "receipt.json")["inventory"]["discovered"] == [BUSINESS]


def test_new_infra_test_requires_fixed_tooling_list_update(tmp_path):
    repo = make_repo(tmp_path)
    write(repo, BUSINESS, "def test_ok():\n    assert True\n")
    extra = "test/registered/unit/ci/test_zing_future_tooling.py"
    write(repo, extra, "def test_new_tool():\n    assert True\n")
    process, output = check(repo, [entry()])
    assert process.returncode != 0
    assert extra in result_json(output / "receipt.json")["inventory"]["unknown_tooling"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
