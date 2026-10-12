# SPDX-License-Identifier: Apache-2.0
"""Check the real multimodal runtime import boundary in a fresh process."""

import os
import subprocess
import sys
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, stage="base-a", runner_config="cpu")


def test_public_server_args_import_in_fresh_process():
    # Other tests may already have imported attention before server_args, hiding
    # an import-order cycle. Keep the production dependency chain intact here.
    repository_root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env["SGLANG_LIGHTWEIGHT_RUNTIME"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(repository_root / "python"), env.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sglang.multimodal_gen.runtime.server_args import "
            "ServerArgs, get_global_server_args",
        ],
        cwd=repository_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Public server_args import exited with {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
