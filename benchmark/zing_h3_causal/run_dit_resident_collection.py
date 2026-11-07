# SPDX-License-Identifier: Apache-2.0
"""Collect both full-context DiT soaks without converting numerical FAIL to PASS.
Retirement boundary: see diagnostics/README.md. Keep this entry until the
current frozen acceptance and independent archive verification are complete.
"""

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

from run_isolated_stall_suite import Runner, stop, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.config.read_text())
    minimum = job.get("minimum_round_seconds", 600)
    if type(minimum) is not int or minimum not in (0, 600):
        raise ValueError("DiT collection duration must be 0 or 600 seconds")
    root = Path(job["root"])
    cfg = json.loads((root / "runner-config.json").read_text())
    if cfg.get("dit_only") is not True or len(cfg["common_captures"]) != 2:
        raise ValueError(
            "Expected an explicit DiT-only allocation and two full captures"
        )
    runner = Runner(cfg)
    cells, numerical_failures = [], []
    geometries = set()
    try:
        for index, capture in enumerate(cfg["common_captures"]):
            metadata = json.loads((Path(capture) / "COMPLETE.json").read_text())
            if metadata["input_frames"] != 15921 or metadata["capture_scope"] != "dit":
                raise ValueError(
                    "Resident collection requires original full long cases"
                )
            session = runner.torch.load(
                Path(capture) / "session.pt", map_location="cpu", weights_only=False
            )
            geometry = (session["params"].width, session["params"].height)
            if geometry not in {(832, 480), (1280, 736)} or geometry in geometries:
                raise ValueError(
                    "Resident collection requires distinct 480 and 736 captures"
                )
            geometries.add(geometry)
            del session
            label = f"SOAK-{index}" if minimum else f"D1-{index}"
            runner.phase(label)
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                "--nproc-per-node=4",
                "--master-addr=127.0.0.1",
                "--master-port=" + str(runner.ports["replay_master"]),
                str(runner.bench / "benchmark_dit_replay.py"),
                "--model",
                cfg["model"],
                "--pipeline-config",
                cfg["pipeline"],
                "--capture",
                capture,
                "--output",
                str(runner.out / label),
                "--measured-runs",
                "3",
                "--require-captured-kv-pool",
            ]
            if minimum:
                command += [
                    "--minimum-round-seconds",
                    str(minimum),
                    "--collect-soak-on-numerical-mismatch",
                ]
            write(runner.out / (label + "-command.json"), command)
            process = runner.start(label, command, runner.roles["dit"], group=False)
            try:
                if process.wait(timeout=14400):
                    raise RuntimeError("DiT resident collection execution failed")
            finally:
                # Execution/cleanup failures abort; only a completed numerical
                # FAIL is allowed to proceed to the other geometry.
                stop(process)
            ranks = []
            for rank in range(4):
                directory = runner.out / label / f"rank{rank}"
                summary = json.loads((directory / "summary.json").read_text())
                measured = json.loads((directory / "measured.json").read_text())
                if (
                    not summary["collection_completed"]
                    or summary["minimum_round_seconds"] != minimum
                    or len(measured) != 3
                ):
                    raise ValueError("Incomplete DiT collection evidence")
                if minimum:
                    if any(row["seconds"] < minimum for row in measured) or any(
                        not session["fully_replayed"]
                        or session["input_frames"] != 15921
                        or len(session["measurement"]["blocks"]) != metadata["chunks"]
                        for row in measured
                        for session in row["sessions"]
                    ):
                        raise ValueError("Incomplete resident session-work evidence")
                elif any(
                    not math.isfinite(row["seconds"])
                    or row["seconds"] <= 0
                    or len(row["blocks"]) != metadata["chunks"]
                    for row in measured
                ):
                    raise ValueError("Incomplete full-context DiT evidence")
                ranks.append(summary)
            strict = all(row["strict_replay_passed"] for row in ranks)
            if not strict:
                numerical_failures.append(label)
            cells.append(
                dict(label=label, capture=capture, geometry=geometry, ranks=ranks)
            )
            write(
                runner.out / "RESIDENT_COLLECTION.json",
                dict(
                    cells=cells,
                    collection_completed=len(cells) == 2,
                    numerical_status="FAIL" if numerical_failures else "PASS",
                    numerical_failures=numerical_failures,
                    full_acceptance_passed=False,
                    minimum_round_seconds=minimum,
                    collection_mode="resident_soak"
                    if minimum
                    else "single_complete_context",
                    scope="Four dedicated DiT GPUs and complete original contexts; collection is not numerical or stall acceptance",
                ),
            )
        # Do not fail early on the first geometry's numerical mismatch, and do
        # not let both completed collections erase that mismatch afterward.
        if numerical_failures:
            raise RuntimeError(
                "Both resident collections completed with numerical FAIL"
                if minimum
                else "Both full-context collections completed with numerical FAIL"
            )
        runner.phase("collection-complete")
    except Exception:
        write(
            runner.out / "COLLECTION_FAILURE.json", dict(error=traceback.format_exc())
        )
        raise
    finally:
        runner.close()


if __name__ == "__main__":
    main()
