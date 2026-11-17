# SPDX-License-Identifier: Apache-2.0
"""CPU-only role-pool contracts, without importing unrelated GPU registries.

Only package initializers are bypassed. The actual Coordinator, native connection
manager, RolePool store, HTTP endpoints, and Room bound adapter execute unchanged.
Redis and WebSocket integration tests use ephemeral loopback-only servers.
"""

import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "python"))
for name in (
    "sglang",
    "sglang.multimodal_gen",
    "sglang.multimodal_gen.runtime",
    "sglang.multimodal_gen.runtime.realtime",
):
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / "python" / name.replace(".", "/"))]
    sys.modules[name] = package

import pytest

TESTS = [
    "python/sglang/multimodal_gen/test/unit/realtime/test_sglang_whitebox_boundary.py",
    "python/sglang/multimodal_gen/test/unit/realtime/test_golden_wire_parity.py",
    "python/sglang/multimodal_gen/test/unit/realtime/test_role_pool.py",
    "python/sglang/multimodal_gen/test/unit/realtime/test_role_pool_transport.py",
    "python/sglang/multimodal_gen/test/unit/realtime/test_worker_connections.py",
    "python/sglang/multimodal_gen/test/unit/realtime/test_worker_services.py",
]

if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                "--confcutdir=python/sglang/multimodal_gen/test/unit/realtime",
                "-o",
                "asyncio_default_fixture_loop_scope=function",
                *TESTS,
                # These eight adapter lifecycle cases import real Torch-backed
                # conditioning and remain mandatory in realtime-contracts via
                # run_cpu_tests.py. The lightweight role-pool job deliberately
                # installs no model dependencies; do not stub the real adapter
                # or turn these into hardware skips here.
                "--deselect=sglang/multimodal_gen/test/unit/realtime/"
                "test_worker_connections.py::"
                "test_encoder_initialization_is_ordered_before_quiesce_and_destroy",
                *sys.argv[1:],
            ]
        )
    )
