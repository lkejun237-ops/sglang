# SPDX-License-Identifier: Apache-2.0
"""Same-allocation Encoder-only static-GC switch, complete real long cases.

Three formal metrics requests per geometry/policy follow a full warmup. One
separate raw diagnostic enables CUDA stage timing and passive gen2 observations.
The DiT policy stays enabled in both arms; no production service is contacted.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Keep direct script execution compatible with the shared benchmark helpers.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from case_sets import load_case_set
from run_isolated_stall_suite import Runner, write


def policy_ports(ports, policy):
    """Fresh blocks avoid recently closed sockets from the other control."""
    if policy not in {"0", "1"}:
        raise ValueError("Unknown Encoder GC policy")
    blocks = [
        {key: port + offset for key, port in ports.items()} for offset in [0, 1000]
    ]
    values = [port for block in blocks for port in block.values()]
    if (
        not ports
        or any(type(port) is not int or not 1024 < port < 32768 for port in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("GC controls require distinct valid frozen port blocks")
    return blocks[int(policy)]


def validate_control_ports(config):
    expected = {policy: policy_ports(config["ports"], policy) for policy in ["0", "1"]}
    if config.get("encoder_gc_ports") != expected:
        raise ValueError("Encoder GC port blocks must be frozen in the job config")
    return expected


class EncoderGCRunner(Runner):
    def environment(self, devices, label):
        env = super().environment(devices, label)
        env["SGLANG_ZING_FREEZE_STATIC_GC"] = (
            self.c["encoder_gc_policy"] if label.endswith("-encoder") else "1"
        )
        env["STALL_GC_DIAGNOSTIC_SWITCH"] = str(self.switch)
        env["STALL_ROLE_GC_LABEL"] = label
        return env

    def launch_instance(self, label, **kwargs):
        super().launch_instance(label, **kwargs)
        ready = []
        for line in (self.out / (label + "-encoder.log")).read_text().splitlines():
            if '"encoder_service_ready"' in line:
                ready.append(json.loads(line))
        if len(ready) != 1 or ready[0]["rank"] != 0 or ready[0]["world"] != 1:
            raise RuntimeError("Missing exact dedicated Encoder readiness record")
        policy = ready[0]["static_gc"]
        if self.c["encoder_gc_policy"] == "1":
            if (
                not policy
                or not policy["enabled"]
                or not policy["automatic_gc_enabled"]
            ):
                raise RuntimeError("Encoder static GC did not activate")
        elif policy is not None:
            raise RuntimeError("Encoder static GC must stay disabled in the control")
        dit = {}
        for line in (self.out / (label + "-dit.log")).read_text().splitlines():
            if "ZING_STATIC_GC_READY_JSON " in line:
                row = json.loads(line.split("ZING_STATIC_GC_READY_JSON ", 1)[1])
                dit[row["pid"]] = row
        if len(dit) != self.sp or not all(
            r["enabled"] and r["automatic_gc_enabled"] for r in dit.values()
        ):
            raise RuntimeError("DiT static GC must stay enabled in both controls")
        write(
            self.out / (label + "-GC_POLICIES.json"),
            dict(encoder=ready, dit=list(dit.values())),
        )


def run(config):
    port_blocks = validate_control_ports(config)
    parent = Path(config["root"])
    cfg = json.loads((parent / "runner-config.json").read_text())
    allocation = json.loads((parent / "ALLOCATION.json").read_text())
    collection = load_case_set()
    cases = [
        case
        for case in collection["definition"]["cases"]
        if "long-session" in case["case_id"]
    ]
    if len(cases) != 1:
        raise ValueError("Expected one original complete long-session case")
    selector = parent / "encoder-gc-long-case.json"
    write(selector, dict(name=collection["definition"]["name"], cases=cases))
    frozen = json.loads((parent / "FROZEN_RGB.json").read_text())
    for policy in ["0", "1"]:
        name = "encoder-gc-off" if policy == "0" else "encoder-gc-on"
        control = dict(
            cfg,
            root=str(parent / name / "work"),
            encoder_gc_policy=policy,
            ports=port_blocks[policy],
            diagnostic_switch=config["diagnostic_switch"] + "-" + name,
            archive_dir=str(Path(config["output"]) / name / "raw-blobs"),
            raw_retention_budget_bytes=64 << 30,
        )
        runner = None
        try:
            runner = EncoderGCRunner(control)
            write(
                parent / "ACTIVE_MEASUREMENT_ROOT.json",
                dict(root=control["root"], encoder_gc_policy=policy),
            )
            write(runner.out / "ALLOCATION.json", allocation)
            write(runner.out / "CONTROL_CONFIG.json", control)
            boundaries = []
            for width, height in [(832, 480), (1280, 736)]:
                label = f"{name}-{width}x{height}"
                runner.launch_instance(label)
                for phase in ["warmup", "formal0", "formal1", "formal2", "diagnostic0"]:
                    diagnostic = phase == "diagnostic0"
                    if diagnostic:
                        runner.switch.touch()
                    started = time.time_ns()
                    try:
                        out = runner.client(
                            label + "-" + phase,
                            width=width,
                            height=height,
                            diagnostic=diagnostic,
                            case_set=selector,
                        )
                    finally:
                        runner.switch.unlink(missing_ok=True)
                    rows = [
                        json.loads(line)
                        for line in (out / "requests.jsonl").read_text().splitlines()
                        if line
                    ]
                    if (
                        len(rows) != 1
                        or rows[0]["output_frames"] != 15921
                        or rows[0]["input_sha256"]
                        != frozen[rows[0]["case_id"]]["sha256"]
                    ):
                        raise RuntimeError(
                            "Incomplete Encoder GC comparison input or output"
                        )
                    boundaries.append(
                        dict(
                            phase="passive-gc-diagnostic" if diagnostic else phase,
                            session_id=rows[0]["server_session_id"],
                            request_id=rows[0]["request_id"],
                            case_id=rows[0]["case_id"],
                            input_sha256=rows[0]["input_sha256"],
                            start=dict(unix_ns=started),
                            end=dict(unix_ns=time.time_ns()),
                        )
                    )
                    write(runner.out / "REQUEST_BOUNDARIES.json", boundaries)
                runner.shutdown_instance()
            runner.phase("collected")
        finally:
            if runner is not None:
                runner.close()
    (parent / "ACTIVE_MEASUREMENT_ROOT.json").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.preflight_only:
        print(json.dumps(validate_control_ports(config), sort_keys=True))
    else:
        run(config)


if __name__ == "__main__":
    main()
