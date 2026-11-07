# SPDX-License-Identifier: Apache-2.0
"""Same-allocation numerical context diagnosis, never a performance acceptance."""

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


def cells(config):
    root = Path(config["root"])
    cfg = json.loads((root / "runner-config.json").read_text())
    captures = cfg["common_captures"]
    ports = config["probe_master_ports"]
    blocks = config["probe_blocks"]
    if (
        len(captures) != 2
        or type(blocks) is not int
        or blocks < 1
        or len(ports) != 4
        or len(set(ports)) != 4
        or any(type(p) is not int or not 1024 < p < 32768 for p in ports)
        or set(ports).intersection(cfg["ports"].values())
    ):
        raise ValueError("Expected two captures and four distinct frozen probe ports")
    result = []
    for index, capture in enumerate(captures):
        complete = json.loads((Path(capture) / "COMPLETE.json").read_text())
        if complete["input_frames"] != 15921 or not 0 < blocks < complete["chunks"]:
            raise ValueError("Prefix diagnosis requires an original complete capture")
        for context in ["no_grad", "inference"]:
            label = f"capture{index}-{context}"
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                "--nproc-per-node=4",
                "--master-addr=127.0.0.1",
                f"--master-port={ports[len(result)]}",
                str(root / "source/benchmark/zing_h3_causal/benchmark_dit_replay.py"),
                "--model",
                cfg["model"],
                "--pipeline-config",
                cfg["pipeline"],
                "--capture",
                capture,
                "--output",
                str(root / "work/results" / label),
                "--measured-runs",
                "1",
                "--diagnostic-blocks",
                str(blocks),
                "--execution-context",
                context,
                "--require-captured-kv-pool",
            ]
            result.append(
                dict(label=label, context=context, capture=capture, command=command)
            )
    return cfg, result


def run(config):
    cfg, controls = cells(config)
    root = Path(config["root"])
    output = root / "work/results"
    output.mkdir(parents=True, exist_ok=False)
    allocation = json.loads((root / "ALLOCATION.json").read_text())
    devices = allocation["allocated_cuda_ordinals"]
    if len(devices) != 4 or allocation["active_gpu_count"] != 4:
        raise ValueError("This independent probe requires exactly four allocated GPUs")
    owner = SimpleNamespace(
        c=cfg,
        ports=cfg["ports"],
        bench=root / "source/benchmark/zing_h3_causal",
        out=output,
        cuda_devices=devices,
        roles=dict(dit=[0, 1, 2, 3], encoder=[], decoder=[]),
        switch=Path(config["diagnostic_switch"]),
    )
    env = Runner.environment(owner, [0, 1, 2, 3], "dit-context-probe")
    env["STALL_SHARED_ALLOCATOR_TRIM"] = "0"
    hardware = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import json,torch; assert torch.cuda.device_count()==4; "
            "print(json.dumps([dict(uuid='GPU-'+str(torch.cuda.get_device_properties(i).uuid).removeprefix('GPU-'),"
            "cc=list(torch.cuda.get_device_capability(i))) for i in range(4)]))",
        ],
        env=env,
        text=True,
    )
    hardware = json.loads(hardware)
    physical = {
        row[0].strip(): row[1].strip() for row in allocation["physical_inventory"]
    }
    if any(row["cc"] != cfg["expected_cc"] for row in hardware) or [
        row["uuid"] for row in hardware
    ] != [physical[d] for d in devices]:
        raise ValueError(
            "Actual probe GPU identity or capability differs from allocation"
        )
    write(output / "HARDWARE.json", hardware)
    results, errors = [], []
    for control in controls:
        label = control["label"]
        write(
            output / "STATUS.json",
            dict(phase=label, scope="Numerical prefix diagnosis only"),
        )
        write(output / (label + "-command.json"), control)
        process = None
        try:
            with (output / (label + ".log")).open("wb") as log:
                process = owned_popen(
                    control["command"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                if process.wait(timeout=1800):
                    raise RuntimeError("DiT context diagnostic child failed")
            ranks = []
            for rank in range(4):
                directory = output / label / f"rank{rank}"
                row = json.loads((directory / "DIAGNOSTIC_ONLY.json").read_text())
                if (
                    row["full_case_accepted"]
                    or row["formal_timing_collected"]
                    or (directory / "COMPLETE").exists()
                ):
                    raise ValueError(
                        "Prefix diagnosis must never claim complete-case acceptance"
                    )
                if (
                    row["replayed_chunks"] != config["probe_blocks"]
                    or row["execution_context"] != control["context"]
                ):
                    raise ValueError(
                        "Diagnostic control identity differs from its frozen command"
                    )
                ranks.append(
                    dict(
                        rank=rank,
                        diagnostic=row,
                        comparison=json.loads(
                            (directory / "replay-comparison.json").read_text()
                        ),
                    )
                )
            results.append(dict(control=control, ranks=ranks))
        except BaseException:
            errors.append(dict(control=control, error=traceback.format_exc()))
        finally:
            if process is not None:
                stop(process)
    write(
        output / "CONTEXT_PROBE_RESULT.json",
        dict(
            results=results,
            errors=errors,
            full_matrix_accepted=False,
            scope="Same four GPUs, fixed original captured prefix; execution context is the sole numerical control. No throughput or full-context acceptance.",
        ),
    )
    if errors:
        raise RuntimeError("DiT numerical diagnosis contains collection failures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.preflight_only:
        _, controls = cells(config)
        print(json.dumps(controls, indent=2))
    else:
        run(config)


if __name__ == "__main__":
    main()
