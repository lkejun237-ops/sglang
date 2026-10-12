"""Filesystem/subprocess fixtures for the Zing CI public CLI contract.

These fixtures never import the implementations under test or business models.
The small repositories use real Git history and real pytest child processes.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SUITES = [
    "taeh3-contracts",
    "adaln-export-contracts",
    "dit-pipeline-contracts",
    "benchmark-contracts",
    "realtime-contracts",
    "conditioning-contracts",
]
TOOLING_FILES = [
    f"test/registered/unit/ci/{name}"
    for name in (
        "test_zing_cpu_runner.py",
        "test_zing_change_selector.py",
        "test_zing_python_lint.py",
        "test_zing_ci_summary.py",
        "test_zing_cpu_inventory.py",
    )
]
UNIT = "python/sglang/multimodal_gen/test/unit"
BUSINESS = "benchmark/zing_h3_causal/test_fixture.py"


def write(repo, path, content):
    destination = repo / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination


def git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=check,
        timeout=15,
    )


def commit(repo):
    git(repo, "add", "-A")
    git(repo, "commit", "--allow-empty", "-qm", "fixture")
    return git(repo, "rev-parse", "HEAD").stdout.strip()


def make_repo(tmp_path, tooling=True):
    repo = tmp_path / "repo with spaces"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "CI Fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    for path in (".pre-commit-config.yaml", ".isort.cfg", "test/pytest.ini"):
        write(repo, path, (ROOT / path).read_text())
    if tooling:
        for index, path in enumerate(TOOLING_FILES):
            write(
                repo,
                path,
                f"def test_fixed_{index}():\n    assert {index} == {index}\n",
            )
    write(repo, "README.md", "fixture\n")
    commit(repo)
    return repo


def json_file(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def entry(path=BUSINESS, suite="benchmark-contracts", **overrides):
    result = {
        "path": path,
        "suite": suite,
        "status": "CPU_PASS",
        "approved": True,
        "allowed_skips": [],
        "sources": ["main"],
        "reason": "standard-library CPU fixture with real pytest execution",
        "dependencies": [],
    }
    result.update(overrides)
    return result


def manifest(repo, entries=None, recipes=None):
    return json_file(
        repo / "scripts/ci/zing/cpu_test_manifest.json",
        {
            "schema_version": 1,
            "tests": entries if entries is not None else [entry()],
            "recipes": recipes or {},
        },
    )


def cli(name, repo, *args, timeout=30, env=None):
    script = ROOT / "scripts/ci/zing" / name
    # A missing execution contract is an intentional assertion failure, never
    # an ImportError or Python's misleading "can't open file" failure.
    assert script.is_file(), f"Missing required Zing CI execution contract: {name}"
    environment = os.environ.copy()
    environment.update(
        SGLANG_LIGHTWEIGHT_RUNTIME="1",
        PYTHONPATH=str(repo / "python"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    environment["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    )
    environment.update(env or {})
    return subprocess.run(
        [sys.executable, str(script), "--repo", str(repo), *map(str, args)],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def result_json(path):
    assert path.is_file(), f"Missing machine-readable receipt: {path}"
    result = json.loads(path.read_text())
    assert isinstance(result, dict), result
    return result


def select(repo, base, head, event="pull_request"):
    output = repo.parent / "selection.json"
    process = cli(
        "select_changes.py",
        repo,
        "--base",
        base,
        "--head",
        head,
        "--event",
        event,
        "--output",
        output,
    )
    return process, output


def run_cpu(repo, entries=None, suite="benchmark-contracts", recipes=None, timeout=5):
    inventory = manifest(repo, entries, recipes)
    output = repo.parent / "cpu-results"
    process = cli(
        "run_cpu_tests.py",
        repo,
        "--manifest",
        inventory,
        "--suite",
        suite,
        "--output-dir",
        output,
        "--timeout",
        timeout,
    )
    return process, output


def run_tooling(repo, env=None):
    output = repo.parent / "tooling-results"
    process = cli("run_tooling_tests.py", repo, "--output-dir", output, env=env)
    return process, output


def copy_tooling_entrypoint(repo):
    source = ROOT / "scripts/ci/zing"
    assert (source / "run_tooling_tests.py").is_file(), (
        "Missing fixed tooling entrypoint"
    )
    shutil.copytree(source, repo / "scripts/ci/zing", dirs_exist_ok=True)


def conditioning_fixture(repo):
    """A minimal faithful reproduction of the existing runner argv trap."""
    roots = [
        f"{UNIT}/test_zing_conditioning_contract.py",
        f"{UNIT}/conditioning",
        "test/registered/unit/multimodal_gen",
    ]
    paths = [
        roots[0],
        roots[1] + "/test_review_fixture.py",
        roots[2] + "/test_conditioning_fixture.py",
    ]
    for path in paths:
        write(repo, path, "def test_real_cpu_case():\n    assert 6 * 7 == 42\n")
    runner = "benchmark/zing_h3_causal/run_conditioning_cpu_tests.py"
    write(
        repo,
        runner,
        """import os
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TESTS = ["test_zing_conditioning_contract.py", "conditioning"]
if __name__ != "__main__":
    (ROOT / "PARENT_IMPORT_FORBIDDEN").write_text("runner imported")
os.chdir(ROOT)
arguments = sys.argv[1:] or [
    *(str(ROOT / "python/sglang/multimodal_gen/test/unit" / name) for name in DEFAULT_TESTS),
    str(ROOT / "test/registered/unit/multimodal_gen"), "-q",
]
raise SystemExit(pytest.main(arguments))
""",
    )
    lock = "benchmark/zing_h3_causal/ISSUE_256_CPU_ENVIRONMENT.json"
    json_file(repo / lock, {"python": sys.version.split()[0], "dependencies": {}})
    history = commit(repo)
    recipe = {
        "runner": runner,
        "environment_lock": lock,
        "history_base": history,
        "reference_sha": history,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "default_roots": roots,
    }
    entries = [
        entry(path, "conditioning-contracts", sources=["conditioning"])
        for path in paths
    ]
    return entries, {"conditioning-contracts": recipe}
