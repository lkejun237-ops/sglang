# SPDX-License-Identifier: Apache-2.0
"""The CPU-visible benchmark CLI must describe the deployed RPC boundary."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    "relative", ["benchmark/zing_h3_causal/benchmark_taeh3_stream.py"]
)
def test_benchmark_help_offers_rpc_and_has_no_artifact_paths(relative):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "python"))
    completed = subprocess.run(
        [sys.executable, str(ROOT / relative), "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert "remote" in completed.stdout
    assert "artifact" not in completed.stdout
    assert "cache-root" not in completed.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
