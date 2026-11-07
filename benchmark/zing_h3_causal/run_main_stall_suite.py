# SPDX-License-Identifier: Apache-2.0
"""Run a frozen main runtime through the separately frozen measurement harness.

Online failures remain failures. Independent captured-input DiT cells still run
after owned online services shut down; they aren't retries of failed requests.

Retirement boundary: see diagnostics/README.md. Keep this entry until the
current frozen acceptance and independent archive verification are complete.
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from run_isolated_stall_suite import Runner, stop, write


class MainRunner(Runner):
    def environment(self, devices, label):
        env = super().environment(devices, label)
        env["SGLANG_ZING_FREEZE_STATIC_GC"] = "0"
        return env


def record_failure(runner, failures, stage, **details):
    failures.append(dict(stage=stage, traceback=traceback.format_exc(), **details))
    write(runner.out / "BASELINE_FAILURES.json", failures)


def collect_cold_restarts(runner, failures):
    """Match candidate S3; a failed restart never counts as a retried pass."""
    for width, height in [(832, 480), (1280, 736)]:
        for restart in range(3):
            label = f"S3-{width}x{height}-restart{restart}"
            expected = [label + "-cold", *[label + f"-warm{i}" for i in range(3)]]
            attempted, completed = [], []
            try:
                runner.launch_instance(label)
                for request in expected:
                    attempted.append(request)
                    runner.client(request, width=width, height=height)
                    completed.append(request)
            except Exception:
                record_failure(
                    runner,
                    failures,
                    label,
                    attempted=attempted,
                    completed=completed,
                    unexecuted=[name for name in expected if name not in attempted],
                )
            finally:
                # Cleanup failures propagate. Never overlap a later restart
                # with an instance whose termination was not confirmed.
                runner.shutdown_instance()


def collect_cold_components(runner, failures):
    for width, height in [(832, 480), (1280, 736)]:
        label = f"cold-components-{width}x{height}"
        runner.switch.touch()
        try:
            runner.launch_instance(label)
            runner.client(label, width=width, height=height, diagnostic=True)
        except Exception:
            record_failure(runner, failures, label)
        finally:
            try:
                runner.shutdown_instance()
            finally:
                runner.switch.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    runner = None
    failures = []
    try:
        runner = MainRunner(cfg)
        collect_cold_restarts(runner, failures)
        try:
            runner.launch_instance("instance")
            runner.client("instance-matrix", mode="full")
            runner.switch.touch()
            try:
                runner.client("diagnostic", mode="diagnostic")
            finally:
                runner.switch.unlink(missing_ok=True)
            runner.phase("lifecycle")
            with (runner.out / "lifecycle.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(runner.bench / "run_stall_lifecycle.py"),
                        "--url",
                        f"ws://127.0.0.1:{runner.ports['http']}/v1/realtime_video/generate",
                        "--media-root",
                        cfg["media"],
                        "--frozen-inputs",
                        cfg["frozen_inputs"],
                        "--output-dir",
                        str(runner.out / "lifecycle"),
                    ],
                    env=runner.environment([], "lifecycle"),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=2400,
                )
        except Exception:
            failures.append(dict(stage="online", traceback=traceback.format_exc()))
            write(runner.out / "BASELINE_FAILURES.json", failures)
        finally:
            runner.shutdown_instance()
        # An old-runtime profile/close failure must not prevent the formal
        # instance matrix from being attempted. Keep each cold diagnostic's
        # failure visible and still collect the independent captured-input D1.
        collect_cold_components(runner, failures)
        for index, capture in enumerate(cfg["common_captures"]):
            label = f"D1-{index}"
            runner.phase(label)
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                "--nproc-per-node=" + str(runner.sp),
                "--master-addr=127.0.0.1",
                "--master-port=" + str(runner.ports["replay_master"]),
                str(runner.bench / "run_main_dit_replay.py"),
                "--harness-source",
                str(runner.bench.parents[1]),
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
            write(runner.out / (label + "-command.json"), command)
            process = runner.start(label, command, runner.roles["dit"], group=False)
            try:
                if process.wait(timeout=14400):
                    raise RuntimeError("Baseline captured-input DiT replay failed")
            except Exception:
                failures.append(dict(stage=label, traceback=traceback.format_exc()))
                write(runner.out / "BASELINE_FAILURES.json", failures)
            finally:
                stop(process)
        write(
            runner.out / "BASELINE_COLLECTION.json",
            dict(
                time=time.time(),
                failures=failures,
                scope="Frozen main runtime, common S3/formal clients and captured-input D1; cold-component diagnostics follow formal requests; no resident soak claim",
            ),
        )
        if failures:
            raise RuntimeError(
                "Baseline contains failed stages; original data retained"
            )
        runner.phase("complete")
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
