"""Select all six Zing CPU suites for conservative dependency changes."""

import argparse
from pathlib import Path

from common import SUITES, changed_paths, write_json

PREFIXES = (
    "python/sglang/multimodal_gen/",
    "python/sglang/srt/",
    "python/sglang/utils/",
    "python/sglang/test/",
    "python/sglang/kernels/",
    "benchmark/zing_h3_causal/",
    "test/registered/",
    "scripts/ci/zing/",
    ".github/actions/",
)
EXACT = {
    "python/sglang/__init__.py",
    "python/sglang/_platform_stubs.py",
    "python/sglang/utils.py",
    "python/sglang/version.py",
    "python/pyproject.toml",
    "test/run_suite.py",
    "test/pytest.ini",
    "test/conftest.py",
    "pytest.ini",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    "conftest.py",
    ".pre-commit-config.yaml",
    ".isort.cfg",
}


def relevant(path):
    return (
        path in EXACT
        or path.startswith(PREFIXES)
        or (path.startswith(".github/workflows/zing-") and path.endswith(".yml"))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument(
        "--event", choices=("pull_request", "workflow_dispatch"), default="pull_request"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    result = {"status": "error", "event": args.event, "suites": []}
    try:
        result.update(changed_paths(args.repo.resolve(), args.base, args.head))
        result["suites"] = (
            SUITES
            if args.event == "workflow_dispatch"
            or any(relevant(path) for path in result["changed_paths"])
            else []
        )
        result["status"] = "success"
        if args.github_output:
            with args.github_output.open("a") as output:
                output.write("run_cpu=" + str(bool(result["suites"])).lower() + "\n")
                for suite in SUITES:
                    output.write(
                        suite.replace("-", "_")
                        + "="
                        + str(suite in result["suites"]).lower()
                        + "\n"
                    )
    except (OSError, ValueError, UnicodeError) as error:
        result["error"] = str(error)
    write_json(args.output, result)
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
