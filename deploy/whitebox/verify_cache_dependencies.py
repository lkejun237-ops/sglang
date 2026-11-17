"""Verify the pinned native cache ELF closure before a GPU job is scheduled."""

import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path


def verify(*, allow_missing_driver=False):
    distribution = importlib.metadata.distribution("mooncake-transfer-engine")
    if distribution.version != "0.3.13":
        raise RuntimeError("The native cache requires Mooncake 0.3.13")
    allowed = {"libcuda.so.1", "libnvidia-ml.so.1"} if allow_missing_driver else set()
    rows = []
    for name in distribution.files:
        # The cache uses these two entrypoints. Optional MoE/process-group
        # plugins target other Torch versions and are not imported by this path.
        if Path(name).name not in {"store.so", "engine.so"}:
            continue
        path = distribution.locate_file(name)
        result = subprocess.run(
            ["ldd", str(path)], text=True, capture_output=True, check=True
        )
        missing = [
            line.strip().split()[0]
            for line in result.stdout.splitlines()
            if "not found" in line
        ]
        if set(missing) - allowed:
            raise RuntimeError(
                "Unresolved native cache libraries: " + ", ".join(missing)
            )
        rows.append(dict(library=str(name), runtime_driver_dependencies=missing))
    if len(rows) != 2:
        raise RuntimeError("The native Store/transfer libraries are missing")
    return dict(
        mooncake_version=distribution.version,
        libraries=rows,
        gpu_execution_verified=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-missing-driver",
        action="store_true",
        help="CPU image build only; a real driver must be present at runtime",
    )
    print(
        json.dumps(
            verify(allow_missing_driver=parser.parse_args().allow_missing_driver),
            indent=2,
        )
    )
