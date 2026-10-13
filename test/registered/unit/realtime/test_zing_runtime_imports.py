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


def test_session_stage_profile_is_relayed_after_cleanup_before_close(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    import msgspec.msgpack

    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
        realtime_video_api,
    )
    from sglang.multimodal_gen.runtime.utils import zing_stage_timing

    events = []
    session = SimpleNamespace(
        id="s", next_chunk_index=2, request=SimpleNamespace(return_stage_timing=True)
    )

    async def cleanup(*args):
        session.request = None
        session.next_chunk_index = 0
        events.append("cleanup")

    async def collect(
        directory, request_id, ranks, dispatch_count, *, generation_id=None
    ):
        assert request_id == "s" and dispatch_count == 2 and ranks == [0, 1]
        events.append("collect")
        return {
            "type": "profile_result",
            "session_id": "s",
            "status": "complete",
            "stages": [],
            "health": [],
        }

    async def send(payload):
        assert msgspec.msgpack.decode(payload)["status"] == "complete"
        events.append("profile")

    async def close(*args, **kwargs):
        events.append("close")

    monkeypatch.setenv("SGLANG_ZING_STAGE_TIMING", "cuda")
    monkeypatch.setenv("SGLANG_ZING_STAGE_TIMING_DIR", "/unused")
    monkeypatch.setattr(zing_stage_timing, "collect_profile", collect)
    monkeypatch.setattr(
        realtime_video_api,
        "get_global_server_args",
        lambda: SimpleNamespace(num_gpus=2),
    )
    monkeypatch.setattr(realtime_video_api, "_cleanup_realtime_session", cleanup)
    monkeypatch.setattr(realtime_video_api, "_close_realtime_websocket", close)
    asyncio.run(
        realtime_video_api._cleanup_realtime_session_then_close(
            SimpleNamespace(send_bytes=send),
            session,
            None,
            None,
            (1000, "generation complete"),
        )
    )
    assert events == ["cleanup", "collect", "profile", "close"]


def test_stage_profile_reports_disabled_collection_without_loading_workers(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    import msgspec.msgpack
    import pytest

    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
        realtime_video_api,
    )

    packets = []

    async def send(payload):
        packets.append(msgspec.msgpack.decode(payload))

    monkeypatch.setenv("SGLANG_ZING_STAGE_TIMING", "off")
    monkeypatch.setattr(
        realtime_video_api,
        "get_global_server_args",
        lambda: pytest.fail("Loaded worker topology while disabled"),
    )
    asyncio.run(
        realtime_video_api._send_stage_profile(SimpleNamespace(send_bytes=send), "s", 1)
    )
    assert packets[0]["status"] == "disabled"
    assert packets[0]["session_id"] == "s"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
