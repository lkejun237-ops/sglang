# SPDX-License-Identifier: Apache-2.0
"""One index for the existing Zing benchmark tools; each keeps its own CLI."""

import argparse
import os
import sys
from pathlib import Path

TOOLS = {
    "request": ("run_benchmark.py", "Complete checkpoint requests and client timing"),
    "source": (
        "bench_streaming_source.py",
        "Continuous source client against a running server",
    ),
    "dit": (
        "bench_dit_phases.py",
        "DiT phases and optional independent operator profile",
    ),
    "capture": ("profile_worker.py", "Opt-in memory/stage recording for a GPU worker"),
    "memory": ("analyze_memory.py", "Replay trusted allocator snapshots"),
    "report": ("build_benchmark_report.py", "Build/import offline comparison reports"),
}


def command(argv):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="\n".join(
            f"{key}: {description}" for key, (_, description) in TOOLS.items()
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("tool", choices=TOOLS)
    args = parser.parse_args(argv[:1])
    return [
        sys.executable,
        str(Path(__file__).with_name(TOOLS[args.tool][0])),
        *argv[1:],
    ]


if __name__ == "__main__":
    args = command(sys.argv[1:])
    os.execv(args[0], args)
