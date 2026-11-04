"""Inventory, triage and approved execution for Zing CPU contracts."""

import argparse
import ast
import importlib.metadata
import json
import re
import sys
from pathlib import Path

from common import (
    SUITES,
    TOOLING_FILES,
    UNIT,
    commit_sha,
    execute_pytest,
    provenance,
    require_cpu_wheel,
    safe_path,
    sha256,
    write_json,
)

STATUSES = {"CPU_PASS", "MIXED", "GPU_REQUIRED", "FAILED", "UNRUN"}


def discover(repo):
    paths = set()
    patterns = (
        UNIT + "/test_zing*.py",
        UNIT + "/realtime/**/test_zing*.py",
        UNIT + "/realtime/test_worker_connections.py",
        UNIT + "/realtime/test_cleanup_watchdog.py",
        UNIT + "/conditioning/**/test_*.py",
        "test/registered/**/test_zing*.py",
        "test/registered/unit/multimodal_gen/**/test_*.py",
        "benchmark/zing_h3_causal/**/test_*.py",
    )
    for pattern in patterns:
        for path in repo.glob(pattern):
            relative = path.relative_to(repo).as_posix()
            if (
                path.is_file()
                and "results" not in Path(relative).parts
                and not relative.startswith("test/registered/unit/ci/")
            ):
                paths.add(relative)
    return sorted(paths)


def inventory_check(repo, payload):
    if payload.get("schema_version") != 1 or not isinstance(payload.get("tests"), list):
        raise ValueError("Unsupported inventory schema")
    entries = payload["tests"]
    errors, seen = [], set()
    sources, approved, unapproved = {}, [], []
    for item in entries:
        path = item.get("path")
        safe_path(repo, path)
        if path in seen:
            errors.append("Duplicate inventory path: " + path)
        seen.add(path)
        if item.get("suite") not in SUITES:
            errors.append("Unknown suite: " + str(item.get("suite")))
        if item.get("status") not in STATUSES:
            errors.append("Unknown status: " + str(item.get("status")))
        if not isinstance(item.get("approved"), bool):
            errors.append("Approval must be explicit: " + path)
        if item.get("approved") and item.get("status") not in {"CPU_PASS", "MIXED"}:
            errors.append("Unvalidated status cannot be approved: " + path)
        origin = item.get("sources")
        if (
            not isinstance(origin, list)
            or not origin
            or any(source not in {"main", "conditioning"} for source in origin)
        ):
            errors.append("Missing or invalid source provenance: " + path)
        sources[path] = origin
        (approved if item.get("approved") else unapproved).append(path)
        skips = item.get("allowed_skips", [])
        if not isinstance(skips, list) or any(
            not isinstance(skip, dict)
            or not skip.get("nodeid")
            or not skip.get("reason")
            for skip in skips
        ):
            errors.append("Invalid reviewed skip list: " + path)
    discovered = discover(repo)
    tooling = sorted(
        path.relative_to(repo).as_posix()
        for path in (repo / "test/registered/unit/ci").glob("test_zing*.py")
        if path.is_file()
    )
    result = {
        "discovered": discovered,
        "unknown": sorted(set(discovered) - seen),
        "missing": sorted(path for path in seen if not safe_path(repo, path).is_file()),
        "outside_discovery": sorted(seen - set(discovered)),
        "approved": sorted(approved),
        "unapproved": sorted(unapproved),
        "sources": sources,
        "tooling_files": TOOLING_FILES,
        "unknown_tooling": sorted(set(tooling) - set(TOOLING_FILES)),
        "missing_tooling": sorted(set(TOOLING_FILES) - set(tooling)),
        "errors": errors,
    }
    for key in (
        "unknown",
        "missing",
        "outside_discovery",
        "unknown_tooling",
        "missing_tooling",
    ):
        if result[key]:
            errors.append(key + ": " + ", ".join(result[key]))
    return result


def conditioning_recipe(repo, payload, check_environment=True):
    recipe = payload.get("recipes", {}).get("conditioning-contracts")
    if not isinstance(recipe, dict):
        raise ValueError("Missing conditioning recipe")
    runner, lock = (
        safe_path(repo, recipe.get(key)) for key in ("runner", "environment_lock")
    )
    if not runner.is_file() or not lock.is_file():
        raise ValueError("Missing conditioning runner or environment lock")
    default_tests = None
    # Never import the dedicated runner: it mutates process-wide module stubs
    # and executes pytest. Read its literal test topology without side effects.
    for node in ast.parse(runner.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_TESTS"
            for target in node.targets
        ):
            default_tests = ast.literal_eval(node.value)
    if (
        not isinstance(default_tests, list)
        or not default_tests
        or any(not isinstance(root, str) for root in default_tests)
    ):
        raise ValueError("Dedicated runner lacks literal DEFAULT_TESTS")
    expected = [UNIT + "/" + name for name in default_tests] + [
        "test/registered/unit/multimodal_gen"
    ]
    if recipe.get("default_roots") != expected:
        raise ValueError("Conditioning recipe does not preserve complete default roots")
    expanded = set()
    for relative in expected:
        path = safe_path(repo, relative)
        if path.is_dir():
            expanded.update(
                p.relative_to(repo).as_posix()
                for p in path.rglob("test_*.py")
                if p.is_file()
            )
        elif path.is_file():
            expanded.add(relative)
        else:
            raise ValueError("Missing conditioning default root: " + relative)
    listed = {
        item["path"]
        for item in payload["tests"]
        if item["suite"] == "conditioning-contracts"
    }
    if expanded != listed:
        raise ValueError("Conditioning default-root files and inventory differ")
    baseline = commit_sha(repo, recipe.get("history_base", ""))
    reference = commit_sha(repo, recipe.get("reference_sha", ""))
    environment_lock = json.loads(lock.read_text())
    validated_dependencies = {}
    if check_environment:
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if recipe.get("python_version") != version:
            raise ValueError(
                "Conditioning requires Python " + str(recipe.get("python_version"))
            )
        for package, expected_version in {
            **environment_lock["dependencies"],
            **recipe.get("extra_dependencies", {}),
        }.items():
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError as error:
                raise ValueError(
                    "Missing conditioning dependency: " + package
                ) from error
            if actual.split("+")[0] != expected_version:
                raise ValueError(
                    f"Conditioning dependency version differs: {package} {actual} != {expected_version}"
                )
            require_cpu_wheel(package, actual)
            validated_dependencies[package] = actual
    return recipe, {
        "history_base": baseline,
        "reference_sha": reference,
        "environment_lock_sha256": sha256(lock),
        "expanded_files": sorted(expanded),
        "default_roots": expected,
        "dependencies": validated_dependencies,
    }


def ordinary_recipe(repo, recipe):
    """Verify an explicitly declared environment before any test launch."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if recipe.get("python_version") != version:
        raise ValueError(
            "CPU recipe requires Python " + str(recipe.get("python_version"))
        )
    hashes, dependencies, visiting = {}, {}, set()

    def read_lock(relative):
        path = safe_path(repo, relative)
        if relative in visiting:
            raise ValueError("Recursive requirements include: " + relative)
        if not path.is_file():
            raise ValueError("Missing CPU requirements lock: " + relative)
        visiting.add(relative)
        hashes[relative] = sha256(path)
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("--index-url ", "--extra-index-url ")):
                continue
            if line.startswith("-r "):
                included = (Path(relative).parent / line[3:].strip()).as_posix()
                read_lock(included)
                continue
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
            if not match:
                raise ValueError("CPU lock contains an unpinned requirement: " + line)
            package, expected = match.groups()
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError as error:
                raise ValueError("Missing CPU dependency: " + package) from error
            # CPU torch wheels carry a local +cpu suffix. An explicit local
            # version in a freeze remains exact; a public pin accepts only the
            # same public version, never a different release.
            equivalent = actual == expected or (
                "+" not in expected and actual.split("+")[0] == expected
            )
            if not equivalent:
                raise ValueError(
                    f"CPU dependency version differs: {package} {actual} != {expected}"
                )
            require_cpu_wheel(package, actual)
            dependencies[package] = actual
        visiting.remove(relative)

    read_lock(recipe.get("requirements"))
    if recipe.get("torch_requirements"):
        read_lock(recipe["torch_requirements"])
    return {
        "python_version": version,
        "requirements_sha256": hashes,
        "dependencies": dependencies,
    }


def verified_cpu_torch(recipe_evidence):
    # Only a successfully checked explicit recipe can supply this evidence.
    # Tooling and fixtures without a torch recipe retain their original env.
    return any(
        package.lower() == "torch" and version.partition("+")[2] == "cpu"
        for package, version in recipe_evidence.get("dependencies", {}).items()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest", type=Path, default=Path("scripts/ci/zing/cpu_test_manifest.json")
    )
    parser.add_argument("--check-inventory", action="store_true")
    parser.add_argument("--suite", choices=[*SUITES, "all"])
    parser.add_argument("--triage", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/zing-cpu"))
    parser.add_argument("--timeout", type=float, default=500)
    args = parser.parse_args()
    repo = args.repo.resolve()
    receipt = {"status": "error", "results": []}
    try:
        if args.trials < 1 or args.timeout <= 0:
            raise ValueError("Trials and timeout must be positive")
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            raise ValueError("Shard index/count are out of bounds")
        if args.shard_count > 1 and (
            args.triage
            or args.check_inventory
            or args.suite in {None, "all", "conditioning-contracts"}
        ):
            raise ValueError(
                "Multiple shards require CI mode and one ordinary CPU suite"
            )
        if args.file and not args.triage:
            raise ValueError("File-scoped execution is available only for triage")
        payload = json.loads(args.manifest.read_text())
        paths = [item["path"] for item in payload.get("tests", [])]
        receipt.update(provenance(repo, args.manifest, paths))
        inventory = inventory_check(repo, payload)
        receipt["inventory"] = inventory
        if inventory["errors"]:
            raise ValueError(
                "Inventory is incomplete: " + "; ".join(inventory["errors"])
            )
        if args.check_inventory:
            if "conditioning-contracts" in payload.get("recipes", {}):
                _, receipt["conditioning"] = conditioning_recipe(repo, payload, False)
            receipt["status"] = "success"
        else:
            if args.suite is None:
                raise ValueError("Execution requires an explicit suite")
            entries = [
                item
                for item in payload["tests"]
                if args.suite in ("all", item["suite"])
            ]
            if args.file:
                known = {item["path"] for item in entries}
                if set(args.file) - known:
                    raise ValueError(
                        "Requested files are absent from the selected inventory"
                    )
                entries = [item for item in entries if item["path"] in args.file]
            if not entries:
                raise ValueError("Selected suite has no inventoried files")
            if not args.triage:
                unresolved = [
                    item["path"]
                    for item in entries
                    if not item["approved"] and item["status"] != "GPU_REQUIRED"
                ]
                if unresolved:
                    raise ValueError(
                        "CI refuses unapproved candidates: " + ", ".join(unresolved)
                    )
                entries = [item for item in entries if item["approved"]]
                if not entries:
                    raise ValueError("Selected suite has no approved CPU files")
            # Approval above applies to the whole suite. Do not hide an UNRUN
            # or FAILED file in a different shard by slicing before validation.
            candidate_files = sorted(item["path"] for item in entries)
            selected_files = candidate_files[args.shard_index :: args.shard_count]
            receipt["shard"] = {
                "index": args.shard_index,
                "count": args.shard_count,
                "candidate_files": candidate_files,
                "selected_files": selected_files,
            }
            if not selected_files:
                raise ValueError("Selected shard contains no complete CPU files")
            if args.shard_count > 1:
                indexed = {item["path"]: item for item in entries}
                entries = [indexed[path] for path in selected_files]
            conditioning = [
                item for item in entries if item["suite"] == "conditioning-contracts"
            ]
            ordinary = [
                item for item in entries if item["suite"] != "conditioning-contracts"
            ]
            receipt["environment_recipes"] = {}
            for suite in sorted({item["suite"] for item in ordinary}):
                declared_recipe = payload.get("recipes", {}).get(suite)
                if declared_recipe is not None:
                    receipt["environment_recipes"][suite] = ordinary_recipe(
                        repo, declared_recipe
                    )
            recipe = None
            if conditioning:
                recipe, receipt["conditioning"] = conditioning_recipe(repo, payload)
            for trial in range(1, args.trials + 1):
                for index, item in enumerate(ordinary):
                    path = item["path"]
                    roots = [repo / path]
                    extra_path = [
                        repo / "benchmark/zing_h3_causal",
                        repo / UNIT,
                        repo / (UNIT + "/realtime"),
                    ]
                    record = execute_pytest(
                        repo,
                        path,
                        roots,
                        args.output_dir,
                        f"cpu-{index}",
                        timeout=args.timeout,
                        trial=trial,
                        allowed_skips=item["allowed_skips"],
                        extra_pythonpath=extra_path,
                        validated_cpu_recipe=verified_cpu_torch(
                            receipt["environment_recipes"].get(item["suite"], {})
                        ),
                    )
                    record["suite"] = item["suite"]
                    receipt["results"].append(record)
                    write_json(args.output_dir / "receipt.json", receipt)
                if conditioning:
                    roots = [repo / root for root in recipe["default_roots"]]
                    if args.file:
                        roots = [repo / item["path"] for item in conditioning]
                    record = execute_pytest(
                        repo,
                        recipe["runner"],
                        roots,
                        args.output_dir,
                        "conditioning",
                        timeout=args.timeout
                        if args.file
                        else min(1800, args.timeout * len(conditioning)),
                        trial=trial,
                        runner=recipe["runner"],
                        allowed_skips=[
                            skip
                            for item in conditioning
                            for skip in item["allowed_skips"]
                        ],
                        validated_cpu_recipe=verified_cpu_torch(
                            receipt["conditioning"]
                        ),
                        required_files=[item["path"] for item in conditioning],
                    )
                    record.update(
                        suite="conditioning-contracts",
                        files=[item["path"] for item in conditioning],
                        full_default_suite=not bool(args.file),
                    )
                    receipt["results"].append(record)
                    write_json(args.output_dir / "receipt.json", receipt)
            if receipt["results"] and all(
                record["status"] == "CPU_PASS" for record in receipt["results"]
            ):
                receipt["status"] = "success"
    except (OSError, ValueError, KeyError, SyntaxError) as error:
        receipt["error"] = str(error)
        print(str(error), file=sys.stderr)
    write_json(args.output_dir / "receipt.json", receipt)
    return 0 if receipt["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
