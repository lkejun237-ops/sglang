"""Shared receipt, Git and child-process boundaries for Zing CPU CI."""

import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

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
HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_cpu_wheel(package, version):
    """Check distribution metadata without importing accelerator libraries."""
    if package.lower().replace("_", "-") not in {"torch", "torchvision"}:
        return
    local = version.partition("+")[2].lower()
    if any(part.startswith(("cu", "rocm", "hip", "xpu")) for part in local.split(".")):
        raise ValueError(
            "Accelerator wheel cannot satisfy a CPU recipe: " + package + " " + version
        )
    # macOS publishes plain versions; the Linux CPU index publishes +cpu.
    if sys.platform == "linux" and local != "cpu":
        raise ValueError("CPU recipe requires a +cpu wheel for " + package)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git(repo, *args):
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        timeout=30,
    )
    if result.returncode:
        raise ValueError(
            "Git operation failed: " + result.stderr.decode(errors="replace")
        )
    return result.stdout


def commit_sha(repo, value):
    # End option parsing explicitly: event fields never become Git options.
    return (
        git(repo, "rev-parse", "--verify", "--end-of-options", str(value) + "^{commit}")
        .decode()
        .strip()
    )


def changed_paths(repo, base, head):
    base_sha, head_sha = commit_sha(repo, base), commit_sha(repo, head)
    merge_base = git(repo, "merge-base", base_sha, head_sha).decode().strip()
    if not merge_base:
        raise ValueError("No merge base exists")
    raw = git(repo, "diff", "--name-status", "-z", "-M", merge_base, head_sha, "--")
    fields = raw.split(b"\0")
    if fields[-1] != b"":
        raise ValueError("Git diff is not NUL terminated")
    fields.pop()
    paths, current, index = [], [], 0
    while index < len(fields):
        status = fields[index].decode("ascii")
        index += 1
        width = 2 if status.startswith(("R", "C")) else 1
        if not status or status[0] not in "ACDMRTUXB" or index + width > len(fields):
            raise ValueError("Malformed Git name-status output")
        names = [os.fsdecode(item) for item in fields[index : index + width]]
        if any(not name for name in names):
            raise ValueError("Empty Git path")
        index += width
        paths.extend(names)
        if not status.startswith("D"):
            current.append(names[-1])
    return {
        "base_sha": base_sha,
        "head_sha": head_sha,
        "merge_base_sha": merge_base,
        "checkout_sha": commit_sha(repo, "HEAD"),
        "changed_paths": sorted(set(paths)),
        "current_paths": sorted(set(current)),
    }


def safe_path(repo, value):
    if not isinstance(value, str) or not value:
        raise ValueError("Expected a nonempty repository-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Path escapes repository: " + value)
    result = repo / path
    if not result.resolve().is_relative_to(repo.resolve()):
        raise ValueError("Symlink escapes repository: " + value)
    return result


def environment():
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "dependencies": {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        },
        "threads": {
            key: os.environ.get(key, "1")
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def provenance(repo, manifest=None, paths=()):
    tooling = {path.name: sha256(path) for path in sorted(HERE.glob("*.py"))}
    sources = {
        path: sha256(safe_path(repo, path))
        for path in paths
        if safe_path(repo, path).is_file()
    }
    result = {
        "checkout_sha": commit_sha(repo, "HEAD"),
        "environment": environment(),
        "tooling_sha256": tooling,
        "source_sha256": sources,
    }
    if manifest is not None:
        result["manifest_sha256"] = sha256(manifest)
    # Include staged/unstaged tracked changes and untracked candidate files;
    # HEAD alone cannot identify an uncommitted implementation.
    candidate = {
        "receipt_inputs": result,
        "tracked_diff_sha256": hashlib.sha256(
            git(repo, "diff", "HEAD", "--")
        ).hexdigest(),
    }
    untracked = git(repo, "ls-files", "--others", "--exclude-standard", "-z").split(
        b"\0"
    )
    candidate["untracked"] = {
        os.fsdecode(path): sha256(repo / os.fsdecode(path))
        for path in untracked
        if path
        and (repo / os.fsdecode(path)).is_file()
        and not os.fsdecode(path).endswith((".pyc", ".log", ".xml"))
    }
    result["candidate_fingerprint"] = hashlib.sha256(
        json.dumps(candidate, sort_keys=True).encode()
    ).hexdigest()
    return result


def execute_pytest(
    repo,
    path,
    roots,
    output,
    name,
    timeout=500,
    allowed_skips=(),
    runner=None,
    trial=1,
    extra_pythonpath=(),
    validated_cpu_recipe=False,
    required_files=(),
):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{name}-trial-{trial}"
    junit, details, log = (
        output / (stem + suffix) for suffix in (".xml", ".json", ".log")
    )
    for stale in (junit, details):
        stale.unlink(missing_ok=True)
    command = [sys.executable]
    command += [str(safe_path(repo, runner))] if runner else ["-m", "pytest"]
    command += [
        "-c",
        str(repo / "test/pytest.ini"),
        "--rootdir=" + str(repo),
        "--strict-config",
        "--strict-markers",
        "-v",
        "-o",
        f"faulthandler_timeout={min(120, timeout / 4):g}",
        "-p",
        "zing_pytest_receipt",
        f"--junitxml={junit}",
        *map(str, roots),
    ]
    env = os.environ.copy()
    env.update(
        SGLANG_LIGHTWEIGHT_RUNTIME="1",
        ZING_CASE_RECEIPT=str(details),
        PYTHONUNBUFFERED="1",
    )
    cuda_visibility_normalized = False
    if validated_cpu_recipe and env.get("CUDA_VISIBLE_DEVICES") == "":
        # A +cpu wheel cannot use CUDA. Leave visibility unset rather than
        # changing the caller's environment or selecting a GPU device number.
        env.pop("CUDA_VISIBLE_DEVICES")
        cuda_visibility_normalized = True
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        map(str, [HERE, repo / "python", *extra_pythonpath])
    )
    started = time.monotonic()
    record = {
        "path": path,
        "trial": trial,
        "command": command,
        "junit_path": str(junit),
        "case_receipt_path": str(details),
        "log_path": str(log),
        "source_sha256": sha256(repo / path) if (repo / path).is_file() else None,
        "status": "FAILED",
        "reason_code": None,
        "approved": False,
        "collected": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "skip_nodeids": [],
        "process_tree_cleaned": False,
        "cuda_visibility_normalized": cuda_visibility_normalized,
        "child_cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "child_nvidia_visible_devices": env.get("NVIDIA_VISIBLE_DEVICES"),
    }
    try:
        child = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        record["pid"] = child.pid
        try:
            data, _ = child.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the session group, even when the parent already exited and
            # a descendant is keeping stdout open. Do not kill unrelated jobs.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            # communicate() returns cumulative output after a timeout retry.
            # Appending it would duplicate progress and traceback diagnostics.
            data, _ = child.communicate(timeout=5)
            record.update(reason_code="TIMEOUT", process_tree_cleaned=True)
        record["exit_code"] = child.returncode
        log.write_bytes(data)
    except OSError as error:
        record.update(exit_code=127, reason_code="LAUNCH_FAILED")
        log.write_text(str(error))
    record["duration_seconds"] = time.monotonic() - started
    if record["reason_code"] == "TIMEOUT":
        return record
    # Pytest exits before initializing report plugins when loading conftest
    # itself raises ImportError. Exit 4 alone is also a CLI/config error, so
    # require the concrete diagnostic; never invent a JUnit file to hide it.
    diagnostic = log.read_text(errors="replace")
    if record["exit_code"] == 4 and "ImportError while loading conftest" in diagnostic:
        record.update(
            reason_code="COLLECTION_FAILED",
            collection_errors=[{"nodeid": "conftest", "error": diagnostic}],
        )
        return record
    try:
        tree = ET.parse(junit)
        if tree.getroot().tag not in {"testsuites", "testsuite"}:
            raise ValueError("Invalid JUnit root")
        cases = tree.findall(".//testcase")
        payload = json.loads(details.read_text())
        record.update(
            {
                key: payload[key]
                for key in (
                    "collected",
                    "passed",
                    "failed",
                    "skipped",
                    "skip_nodeids",
                    "nodeids",
                    "deselected",
                    "collection_errors",
                    "skip_reasons",
                    "collection_skips",
                    "completed_nodeids",
                    "incomplete_nodeids",
                )
            }
        )
        if record["collected"] > 0 and not cases:
            raise ValueError("JUnit lacks collected cases")
        collected_files = {nodeid.split("::", 1)[0] for nodeid in record["nodeids"]}
        record["uncollected_files"] = sorted(set(required_files) - collected_files)
        junit_failed = sum(
            case.find("failure") is not None or case.find("error") is not None
            for case in cases
        )
        junit_skipped = sum(case.find("skipped") is not None for case in cases)
        record["junit_failed"] = junit_failed
        record["junit_skipped"] = junit_skipped
        junit_nodeids = {
            property.attrib.get("value")
            for case in cases
            for property in case.findall("./properties/property")
            if property.attrib.get("name") == "zing_nodeid"
        }
        record["junit_missing_nodeids"] = sorted(
            set(record["completed_nodeids"]) - junit_nodeids
        )
        if payload["collection_errors"] or record["exit_code"] == 2:
            record["reason_code"] = "COLLECTION_FAILED"
        elif record["exit_code"] not in (0, 5) or record["failed"] or junit_failed:
            record["reason_code"] = "TEST_FAILED"
        elif record["collection_skips"]:
            record["reason_code"] = "UNKNOWN_COLLECTION_SKIP"
        elif record["deselected"]:
            record["reason_code"] = "UNREVIEWED_DESELECTION"
        elif record["incomplete_nodeids"]:
            record["reason_code"] = "INCOMPLETE_EXECUTION"
        elif record["uncollected_files"]:
            record["reason_code"] = "INCOMPLETE_FILE_COVERAGE"
        elif not record["collected"]:
            record["reason_code"] = "ZERO_CASES"
        elif record["junit_missing_nodeids"]:
            record["reason_code"] = "INVALID_JUNIT"
        else:
            allowed = {item["nodeid"]: item["reason"] for item in allowed_skips}
            if (
                any(
                    allowed.get(nodeid) != reason
                    for nodeid, reason in payload["skip_reasons"].items()
                )
                or junit_skipped != record["skipped"]
            ):
                record["reason_code"] = "UNKNOWN_SKIP"
            elif not record["passed"]:
                record["reason_code"] = "ZERO_CPU_PASSES"
            else:
                record.update(status="CPU_PASS", reason_code=None)
    except (OSError, ValueError, KeyError, ET.ParseError) as error:
        record["reason_code"] = record["reason_code"] or "INVALID_JUNIT"
        record["receipt_error"] = str(error)
    return record
