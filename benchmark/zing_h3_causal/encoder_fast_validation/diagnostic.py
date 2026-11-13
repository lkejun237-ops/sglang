# SPDX-License-Identifier: Apache-2.0
"""Separate allocator/monitoring diagnostic requests from formal timing and videos."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener

from profile_worker import checkpoint


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def scrape(output, label):
    opener = build_opener(ProxyHandler({}))
    for role, port in [("encoder", 24110), ("dit", 24210), ("decoder", 24310)]:
        with opener.open(f"http://127.0.0.1:{port}/metrics", timeout=30) as response:
            content = response.read().decode()
        (output / f"{label}-{role}.prom").write_text(content)


async def client(root, target, case_set, group, owned, sample=False):
    cmd = [
        sys.executable,
        str(root / "source/benchmark/zing_h3_causal/bench_streaming_source.py"),
        "--url",
        group.input_ws,
        "--decoder-url",
        group.output_ws,
        "--case-set",
        str(case_set),
        "--media-root",
        str(root / "media"),
        "--output-dir",
        str(target),
        "--seed",
        "42",
        "--batch-pattern",
        "1",
        "--timeout",
        "7200",
        "--warmup-runs",
        "0",
        "--measured-runs",
        "1",
        "--no-stage-timing",
        "--media-mode",
        "metrics",
    ]
    rows = []
    with target.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                owned.check()
                if sample:
                    text = await asyncio.to_thread(
                        subprocess.check_output,
                        [
                            "nvidia-smi",
                            "--query-gpu=index,uuid,memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=15,
                    )
                    rows.append(
                        {"monotonic": time.monotonic(), "rows": text.splitlines()}
                    )
                await asyncio.sleep(0.2)
            if process.returncode:
                raise RuntimeError(f"Diagnostic request failed: {target.name}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                await asyncio.to_thread(process.wait, timeout=30)
    requests = [
        json.loads(line)
        for line in (target / "requests.jsonl").read_text().splitlines()
    ]
    assert (
        len(requests) == 1
        and requests[0]["success"]
        and requests[0]["delivery_complete"]
    )
    if sample:
        write(
            target / "driver-memory.json",
            {
                "interval_seconds": 0.2,
                "scope": "sampled whole-device driver usage, not exact peak",
                "samples": rows,
            },
        )


async def diagnose(root, output, group, owned, expected_workers):
    from case_sets import load_case_set

    directory = output / "diagnostics"
    deadline = time.monotonic() + 120
    while len(list((directory / "memory").glob("process-*.json"))) < expected_workers:
        if time.monotonic() >= deadline:
            raise RuntimeError("Allocator worker identities are incomplete")
        owned.check()
        await asyncio.sleep(0.5)
    for index, member in enumerate(load_case_set("product-case-20261103")["members"]):
        case = member["case_id"]
        case_set = directory / f"case-{index}.json"
        write(
            case_set,
            {
                "name": "single-frozen-product-case",
                "cases": [
                    {
                        "case_id": case,
                        "manifest": str(member["manifest"]),
                    }
                ],
            },
        )
        target = directory / case
        target.mkdir()
        await client(root, target / "warmup", case_set, group, owned)
        before = await asyncio.to_thread(
            checkpoint,
            directory,
            f"case-{index}-before",
            "reset",
            expected_workers,
            120,
        )
        scrape(target, "before")
        await client(root, target / "measured", case_set, group, owned, sample=True)
        after = await asyncio.to_thread(
            checkpoint, directory, f"case-{index}-after", "reset", expected_workers, 120
        )
        # Give the bounded asynchronous event collector time to publish completed observations.
        await asyncio.sleep(1)
        scrape(target, "after")
        write(
            target / "allocator.json",
            {
                "case_id": case,
                "requests": 1,
                "before": before,
                "after": after,
                "scope": "per-process whole-role allocator; Encoder includes Qwen/reference/source VAE; no allocation history",
            },
        )
        write(
            output / "STATUS.json",
            {
                "phase": "diagnostics",
                "case_id": case,
                "completed_cases": index + 1,
                "time": time.time(),
            },
        )
    await asyncio.to_thread(
        checkpoint, directory, "disable-for-timing", "disable", expected_workers, 120
    )
    write(
        directory / "GATE.json",
        {
            "passed": True,
            "cases": 8,
            "requests": 16,
            "allocator_observer_disabled_for_formal_timing": True,
        },
    )
