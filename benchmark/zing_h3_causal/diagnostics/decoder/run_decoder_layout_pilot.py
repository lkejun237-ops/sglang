# SPDX-License-Identifier: Apache-2.0
"""Full captured Decoder parity and short instance pilot; never full acceptance."""

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

# Keep direct script execution compatible with the shared benchmark helpers.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from owned_process import owned_popen
from run_isolated_stall_suite import Runner, stop, write


def layout_config(job, world):
    root = Path(job["root"])
    cfg = json.loads((root / "runner-config.json").read_text())
    if job.get("decoder_world") != 2 or world not in [1, 2]:
        raise ValueError(
            "This pilot compares dedicated D1 and D2 on one eight-GPU allocation"
        )
    ports = [
        {key: value + offset for key, value in cfg["ports"].items()}
        for offset in [0, 1000]
    ]
    values = [value for block in ports for value in block.values()] + job[
        "parity_master_ports"
    ]
    if len(set(values)) != len(values) or any(
        type(v) is not int or not 1024 < v < 32768 for v in values
    ):
        raise ValueError(
            "All Decoder pilot and parity ports must be distinct and frozen"
        )
    return dict(
        cfg,
        root=str(root / "work" / f"d{world}"),
        ports=ports[world - 1],
        role_devices=dict(
            dit=[0, 1, 2, 3], encoder=[4], decoder=list(range(5, 5 + world))
        ),
        diagnostic_switch=job["diagnostic_switch"] + f"-d{world}",
        archive_dir=str(Path(job["output"]) / "raw-blobs"),
    )


def instance(job, world):
    root = Path(job["root"])
    runner = None
    errors = []
    indexed = root / "PILOT_REQUESTS.json"
    requests = json.loads(indexed.read_text()) if indexed.exists() else []
    try:
        runner = Runner(layout_config(job, world))
        write(
            runner.out / "LAYOUT_ALLOCATION.json",
            dict(allocated=8, active=5 + world, unused=3 - world),
        )
        for width, height in [(832, 480), (1280, 736)]:
            runner.launch_instance(f"d{world}-{width}x{height}")
            for phase in [
                "cold-component-diagnostic",
                "warmup",
                "formal0",
                "formal1",
                "formal2",
                "diagnostic0",
                "diagnostic1",
                "diagnostic2",
            ]:
                label = f"d{world}-{width}x{height}-{phase}"
                diagnostic = "diagnostic" in phase
                runner.phase(label)
                if diagnostic:
                    runner.switch.touch()
                try:
                    output = runner.client(
                        label, width=width, height=height, diagnostic=diagnostic
                    )
                finally:
                    runner.switch.unlink(missing_ok=True)
                row = json.loads((output / "requests.jsonl").read_text())
                if (
                    not row["success"]
                    or not row["delivery_complete"]
                    or row["output_frames"] != 403
                ):
                    raise RuntimeError(
                        "Decoder instance pilot did not deliver its complete original case"
                    )
                requests.append(
                    dict(
                        decoder_world=world,
                        width=width,
                        height=height,
                        phase=phase,
                        request_id=row["request_id"],
                        metrics=row["metrics"],
                        receipt=str(output / "requests.jsonl"),
                    )
                )
                write(indexed, requests)
            runner.shutdown_instance()
    except BaseException:
        errors.append(traceback.format_exc())
        raise
    finally:
        if runner is not None:
            runner.close()
        write(
            root / f"PILOT_D{world}_RESULT.json",
            dict(errors=errors, requests=len(requests), full_matrix_accepted=False),
        )


def run(job):
    root = Path(job["root"])
    cfg = json.loads((root / "runner-config.json").read_text())
    allocation = json.loads((root / "ALLOCATION.json").read_text())
    if (
        allocation["scheduler_allocated_gpu_count"] != 8
        or allocation["active_gpu_count"] != 7
    ):
        raise ValueError(
            "D1/D2 pilot requires eight allocated GPUs, at most seven active"
        )
    results = root / "work/results"
    results.mkdir(parents=True, exist_ok=False)
    owner = SimpleNamespace(
        c=cfg,
        ports=cfg["ports"],
        bench=root / "source/benchmark/zing_h3_causal",
        out=results,
        cuda_devices=allocation["allocated_cuda_ordinals"],
        switch=Path(job["diagnostic_switch"]),
        roles=cfg["role_devices"],
    )
    cells = json.loads((root / "COMMON_CAPTURES.json").read_text())
    if len(cells) != 2 or len(job["parity_master_ports"]) != 2:
        raise ValueError("Expected exactly two original full captured cases")
    errors = []

    def execute(label, command, env):
        process = None
        write(results / "STATUS.json", dict(phase=label))
        write(root / (label + "-command.json"), command)
        try:
            with (root / (label + ".log")).open("wb") as log:
                process = owned_popen(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                if process.wait(timeout=7200):
                    raise RuntimeError("Decoder layout diagnostic child failed")
        except BaseException:
            errors.append(dict(phase=label, error=traceback.format_exc()))
        finally:
            if process is not None:
                stop(process)

    for index, cell in enumerate(cells):
        label = f"decoder-parity-d2-{index}"
        write(
            results / (label + "-allocation.json"),
            dict(allocated=8, active=2, unused=6, scope="Isolated Decoder only"),
        )
        env = Runner.environment(owner, [5, 6], label)
        execute(
            label,
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                "--nproc-per-node=2",
                "--master-addr=127.0.0.1",
                f"--master-port={job['parity_master_ports'][index]}",
                str(owner.bench / "diagnostics/decoder/benchmark_decoder_parallel.py"),
                "--capture",
                cell["path"],
                "--pipeline",
                cfg["pipeline"],
                "--out",
                str(root / "work" / label),
            ],
            env,
        )
    for world in [1, 2]:
        child = layout_config(job, world)
        write(
            root / "ACTIVE_MEASUREMENT_ROOT.json",
            dict(root=child["root"], decoder_world=world),
        )
        execute(
            f"instance-d{world}",
            [
                sys.executable,
                str(Path(__file__)),
                "--config",
                str(root / "job-config.json"),
                "--world",
                str(world),
            ],
            Runner.environment(owner, list(range(5 + world)), f"instance-d{world}"),
        )
    (root / "ACTIVE_MEASUREMENT_ROOT.json").unlink(missing_ok=True)
    write(
        root / "PILOT_RESULT.json",
        dict(
            errors=errors,
            full_matrix_accepted=False,
            scope="Two full15921-frame D1/D2 parity cells and32 short403-frame instance requests; independent readback required",
        ),
    )
    if errors:
        raise RuntimeError(
            "Decoder pilot contains failures; original partial results retained"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--world", type=int, choices=[1, 2])
    args = parser.parse_args()
    job = json.loads(args.config.read_text())
    if args.preflight_only:
        print(json.dumps([layout_config(job, world) for world in [1, 2]], indent=2))
    elif args.world is not None:
        instance(job, args.world)
    else:
        run(job)


if __name__ == "__main__":
    main()
