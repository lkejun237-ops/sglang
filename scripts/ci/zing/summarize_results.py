"""Fail closed unless detector, lint, tooling and selected jobs all agree."""

import argparse
import json
from pathlib import Path

from common import SUITES, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--needs", type=Path, required=True)
    parser.add_argument(
        "--event", choices=("pull_request", "workflow_dispatch"), default="pull_request"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "status": "error",
        "errors": [],
        "business_ran": False,
        "tooling_ran": False,
    }
    try:
        selection = json.loads(args.selection.read_text())
        needs = json.loads(args.needs.read_text())
        selected = selection.get("suites")
        if (
            selection.get("status") != "success"
            or not isinstance(selected, list)
            or any(suite not in SUITES for suite in selected)
            or len(set(selected)) != len(selected)
        ):
            result["errors"].append("zing-changes")
            selected = []
        if args.event == "workflow_dispatch" and set(selected) != set(SUITES):
            result["errors"].append("zing-changes")
        for job in ("zing-changes", "zing-lint", "ci-tooling-contracts", *SUITES):
            outcome = needs.get(job, {}).get("result")
            expected = "success" if job not in SUITES or job in selected else "skipped"
            if outcome != expected:
                result["errors"].append(job)
        result["errors"] = sorted(set(result["errors"]))
        result["business_ran"] = bool(selected) and all(
            needs.get(suite, {}).get("result") == "success" for suite in selected
        )
        result["tooling_ran"] = (
            needs.get("ci-tooling-contracts", {}).get("result") == "success"
        )
        result["suites"] = selected
        result["message"] = (
            "Business CPU suites executed"
            if selected
            else "No relevant business CPU changes; tooling contracts executed"
        )
        if not result["errors"]:
            result["status"] = "success"
    except (OSError, ValueError, AttributeError, TypeError) as error:
        result["errors"].append("zing-changes")
        result["error"] = str(error)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
