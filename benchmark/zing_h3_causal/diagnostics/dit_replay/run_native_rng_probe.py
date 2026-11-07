# SPDX-License-Identifier: Apache-2.0
"""Untimed native 403-frame capture and three-chunk replay; no acceptance claim."""

import argparse
import json
import sys
from pathlib import Path

# Keep direct script execution compatible with the shared benchmark helpers.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_isolated_stall_suite import Runner, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.config.read_text())
    cfg = json.loads((Path(job["root"]) / "runner-config.json").read_text())
    runner = None
    try:
        runner = Runner(cfg)
        runner.launch_instance("native-rng", capture=True)
        for width, height in [(832, 480), (1280, 736)]:
            result = runner.client(
                f"native-{width}x{height}", width=width, height=height
            )
            request = json.loads((result / "requests.jsonl").read_text())
            if not request["success"] or request["output_frames"] != 403:
                raise RuntimeError(
                    "Native RNG capture did not complete the original short case"
                )
        runner.shutdown_instance()
        captures = sorted((runner.root / "captures").iterdir())
        if len(captures) != 2:
            raise ValueError("Expected exactly two native short-case captures")
        for index, capture in enumerate(captures):
            meta = json.loads((capture / "COMPLETE.json").read_text())
            if meta["input_frames"] != 403 or meta["capture_scope"] != "dit":
                raise ValueError("Native RNG diagnostic capture identity differs")
            if len(list((capture / "rng").glob("rank*/drain-*.pt"))) != 12:
                raise ValueError(
                    "Expected three native drain observations on each of four ranks"
                )
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                "--nproc-per-node=4",
                "--master-addr=127.0.0.1",
                f"--master-port={cfg['ports']['replay_master'] + index * 10}",
                str(runner.bench / "benchmark_dit_replay.py"),
                "--model",
                str(runner.model),
                "--pipeline-config",
                cfg["pipeline"],
                "--capture",
                str(capture),
                "--output",
                str(runner.out / f"replay-{index}"),
                "--diagnostic-blocks",
                "3",
                "--require-captured-kv-pool",
            ]
            runner.phase(f"rng-replay-{index}")
            process = runner.start(
                f"rng-replay-{index}", command, runner.roles["dit"], group=False
            )
            if process.wait(timeout=3600):
                raise RuntimeError("Native RNG prefix replay execution failed")
        if job.get("verify_terminal_recovery", False):
            # The two original captures stay immutable. Exercise profile and
            # disconnect/rejection recovery on a separate non-capture instance.
            runner.launch_instance("terminal-recovery")
            for width, height in [(832, 480), (1280, 736)]:
                runner.switch.touch()
                try:
                    runner.client(
                        f"terminal-profile-{width}x{height}",
                        width=width,
                        height=height,
                        diagnostic=True,
                    )
                finally:
                    runner.switch.unlink(missing_ok=True)
            runner.run_lifecycle()
            runner.shutdown_instance()
        write(
            runner.out / "DIAGNOSTIC_COLLECTION.json",
            dict(
                full_matrix_accepted=False,
                full_case_accepted=False,
                terminal_recovery_probe_executed=job.get(
                    "verify_terminal_recovery", False
                ),
                scope="Two natural403-frame native requests and first-three-chunk observations only; independent numerical/RNG analysis required",
            ),
        )
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
