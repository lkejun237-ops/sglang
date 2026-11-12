"""CPU-only before/after transport regression in an existing immutable runtime.

Run with a private config containing an immutable source archive URL and SHA.
The downloaded source is test input, never mounted into a serving container.
"""

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

os.umask(0o077)
root = Path(sys.argv[1]).resolve()
config = json.loads((root / "settings.private.json").read_text())
assert not (root / "RESULT.json").exists()
assert (
    json.loads(Path("/opt/whitebox/release.json").read_text())["source_sha"]
    == config["baseline_sha"]
)
with urllib.request.urlopen(config["url"], timeout=60) as response:
    data = response.read()
assert hashlib.sha256(data).hexdigest() == config["sha256"]
archive = root / "source.tar.gz"
archive.write_bytes(data)
source = root / "source"
assert not source.exists()
source.mkdir()
with tarfile.open(archive) as package:
    package.extractall(source, filter="data")
tests = source / "python/sglang/multimodal_gen/test/unit/realtime"
env = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",
    "SGLANG_LIGHTWEIGHT_RUNTIME": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
results = {}
for label, pythonpath, arguments in [
    (
        "before",
        "/opt/whitebox/python",
        [
            str(tests / "test_zing_source_decoder_overlap.py")
            + "::test_native_demand_does_not_serialize_next_dispatch_with_rgb_writer",
            str(tests / "test_realtime_output_transport.py")
            + "::test_decoder_materialization_does_not_hold_the_output_send_group",
        ],
    ),
    (
        "after",
        str(source / "python"),
        [
            str(tests / name)
            for name in [
                "test_source_demand_publisher.py",
                "test_websocket_transport.py",
                "test_zing_source_decoder_overlap.py",
                "test_realtime_output_transport.py",
                "test_realtime_async_output_critical_path.py",
                "test_realtime_control_telemetry.py",
                "test_zing_h3_source_adapter.py",
                "test_zing_source_encoder_api.py",
                "test_realtime_runtime.py",
                "test_async_vae_pipeline.py",
            ]
        ],
    ),
]:
    started = time.time()
    with (root / (label + ".log")).open("x") as log:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *arguments],
            cwd=source,
            env={**env, "PYTHONPATH": pythonpath},
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=900,
        )
    results[label] = {"exit_code": result.returncode, "seconds": time.time() - started}
    (root / "STATE.json").write_text(json.dumps(results))
    if label == "before":
        output = (root / "before.log").read_text()
        assert (
            result.returncode == 1 and "2 failed" in output and "TimeoutError" in output
        ), "Baseline must reproduce both blocked paths"
    else:
        assert result.returncode == 0, "Candidate regression failed; inspect logs"
(root / "RESULT.json").write_text(
    json.dumps(
        {
            "source_sha": config["source_sha"],
            "baseline_sha": config["baseline_sha"],
            "results": results,
            "gpu_used": False,
        }
    )
)
print(json.dumps(results), flush=True)
