"""Run the fixed tooling contracts, independently of business selection."""

import argparse
from pathlib import Path

from common import TOOLING_FILES, execute_pytest, provenance, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/zing-tooling")
    )
    parser.add_argument("--timeout", type=float, default=500)
    args = parser.parse_args()
    repo = args.repo.resolve()
    receipt = {"status": "error", "results": []}
    try:
        receipt.update(provenance(repo, paths=TOOLING_FILES))
        for index, path in enumerate(TOOLING_FILES):
            if not (repo / path).is_file():
                record = {
                    "path": path,
                    "status": "FAILED",
                    "reason_code": "MISSING_FILE",
                    "passed": 0,
                    "collected": 0,
                }
            else:
                record = execute_pytest(
                    repo,
                    path,
                    [repo / path],
                    args.output_dir,
                    f"tooling-{index}",
                    timeout=args.timeout,
                )
            receipt["results"].append(record)
        if all(record["status"] == "CPU_PASS" for record in receipt["results"]):
            receipt["status"] = "success"
    except (ValueError, OSError) as error:
        receipt["error"] = str(error)
    write_json(args.output_dir / "receipt.json", receipt)
    return 0 if receipt["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
