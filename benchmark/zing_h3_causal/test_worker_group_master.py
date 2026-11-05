# SPDX-License-Identifier: Apache-2.0
"""Exercise the real worker HTTP contract; model callbacks are CPU-only."""

import asyncio
import json
import socket
import threading
from types import SimpleNamespace

import pytest
from worker_group_master import LocalWorkers, WorkerGroupMaster, http_address

from sglang.multimodal_gen.runtime.realtime.worker_connections import (
    ServiceConnections,
    service_descriptor,
)
from sglang.multimodal_gen.runtime.realtime.worker_http import WorkerHTTPServer

ROLES = ("encoder", "dit", "decoder")


def test_cleanup_quiesces_session_then_disconnects_all_nccl_peers_together(tmp_path):
    async def run():
        group = WorkerGroupMaster(
            {role: "http://127.0.0.1:30000" for role in ROLES},
            launcher=LocalWorkers(None, tmp_path),
        )
        group.created = True
        started = {role: asyncio.Event() for role in ROLES}
        disconnected = set()
        quiesced = False

        async def rpc(role, path, payload=None, **kwargs):
            nonlocal quiesced
            if path == "/quiesce_group":
                assert role == "encoder" and not disconnected
                quiesced = True
                return dict(success=True, state="quiesced", worker_group_id=group.group)
            assert path == "/destroy_group" and quiesced
            started[role].set()
            # Private NCCL shutdown requires every peer, including Decoder.
            await asyncio.gather(*(event.wait() for event in started.values()))
            disconnected.add(role)
            return dict(state="disconnected")

        group.rpc = rpc
        await asyncio.wait_for(group.destroy(), timeout=2)
        assert disconnected == set(ROLES)

    asyncio.run(run())


@pytest.fixture
def workers():
    servers, managers, addresses, calls = [], {}, {}, []
    barrier = threading.Barrier(3)
    fail = set()
    holds = {}
    try:
        for role in ROLES:
            with socket.socket() as reserve:
                reserve.bind(("127.0.0.1", 0))
                port = reserve.getsockname()[1]
            http = f"http://127.0.0.1:{port}"
            fields = dict(http=port)
            if role == "encoder":
                fields.update(input_ws=30000, rendezvous=32000)
            elif role == "dit":
                fields.update(zmq=34000, rendezvous=35000)
            else:
                fields.update(zmq=37000, output_ws=38000)
            descriptor = service_descriptor(role, role, 0, "127.0.0.1", **fields)

            def build(group, topology, role=role):
                calls.append((role, "build", group))
                barrier.wait(timeout=5)
                if role in holds:
                    if not holds[role].wait(timeout=5):
                        raise TimeoutError("injected delayed build timed out")
                if role in fail:
                    raise RuntimeError("injected build failure")

            async def destroy(role=role):
                calls.append((role, "destroy"))
                await asyncio.sleep(0.02)

            manager = ServiceConnections(
                descriptor,
                ready=lambda: True,
                build=build,
                destroy=destroy,
                rollback=lambda role=role: calls.append((role, "rollback")),
                quiesce=(lambda: None) if role == "encoder" else None,
            )
            server = WorkerHTTPServer(role, "127.0.0.1", port, install=manager.install)
            server.ready = True
            servers.append(server)
            managers[role], addresses[role] = manager, http
        yield SimpleNamespace(
            addresses=addresses, managers=managers, calls=calls, fail=fail, holds=holds
        )
    finally:
        for server in reversed(servers):
            server.close()


def test_existing_workers_are_connected_without_launching_any_process(
    workers, tmp_path, monkeypatch
):
    def forbidden(*a, **kw):
        raise AssertionError("Explicit services must never be started again")

    monkeypatch.setattr("worker_group_master.subprocess.Popen", forbidden)

    async def run():
        launcher = LocalWorkers("nonexistent-config-is-not-read", tmp_path)
        async with WorkerGroupMaster(workers.addresses, launcher=launcher) as group:
            assert all(m.state == "connected" for m in workers.managers.values())
            assert group.input_ws == "ws://127.0.0.1:30000/v1/realtime_video/generate"
            assert group.output_ws == "ws://127.0.0.1:38000/v1/realtime_video/output"
            assert group.summary()["reused"] is False
        assert all(m.group is None for m in workers.managers.values())
        assert all(m.state == "idle" for m in workers.managers.values())
        assert len([c for c in workers.calls if c[1] == "destroy"]) == 3

    asyncio.run(run())


@pytest.mark.parametrize("supplied", [(), ("encoder",), ("encoder", "decoder")])
def test_launches_only_missing_roles(workers, tmp_path, monkeypatch, supplied):
    launched, signalled = [], []

    class Process:
        returncode = None

        def __init__(self, command, **kwargs):
            self.pid = 10000 + len(launched)
            launched.append(command[0])

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 0
            return 0

    monkeypatch.setattr("worker_group_master.subprocess.Popen", Process)
    monkeypatch.setattr(
        "worker_group_master.os.killpg", lambda pid, sig: signalled.append(pid)
    )
    config = tmp_path / "launch.json"
    config.write_text(
        json.dumps(
            dict(
                workers={
                    role: dict(http=workers.addresses[role], command=[role])
                    for role in ROLES
                }
            )
        )
    )

    async def run():
        explicit = {
            role: workers.addresses[role] if role in supplied else None
            for role in ROLES
        }
        launcher = LocalWorkers(config, tmp_path / "logs")
        async with WorkerGroupMaster(explicit, launcher=launcher):
            assert set(launched) == set(ROLES) - set(supplied)
        assert len(signalled) == len(launched)

    asyncio.run(run())


def test_missing_workers_without_launch_config_never_search_cluster(tmp_path):
    with pytest.raises(ValueError, match="no cluster lookup"):
        asyncio.run(
            WorkerGroupMaster({}, launcher=LocalWorkers(None, tmp_path)).prepare()
        )


def test_busy_service_is_not_restarted_or_destroyed(workers, tmp_path):
    workers.managers["dit"].query()

    async def run():
        with pytest.raises(RuntimeError, match="must be available"):
            async with WorkerGroupMaster(
                workers.addresses, launcher=LocalWorkers(None, tmp_path)
            ):
                pass
        assert not workers.calls

    asyncio.run(run())


def test_retained_group_can_be_reused_without_rebuilding(workers, tmp_path):
    async def run():
        async with WorkerGroupMaster(
            workers.addresses,
            launcher=LocalWorkers(None, tmp_path),
            worker_group_id="retained",
            keep_connections=True,
        ):
            pass
        assert all(m.state == "connected" for m in workers.managers.values())
        async with WorkerGroupMaster(
            workers.addresses,
            launcher=LocalWorkers(None, tmp_path),
            worker_group_id="retained",
        ) as reused:
            assert reused.summary()["reused"] is True
        assert len([c for c in workers.calls if c[1] == "build"]) == 3
        assert len([c for c in workers.calls if c[1] == "destroy"]) == 3

    asyncio.run(run())


def test_wrong_group_cannot_take_over_retained_connections(workers, tmp_path):
    async def run():
        async with WorkerGroupMaster(
            workers.addresses,
            launcher=LocalWorkers(None, tmp_path),
            worker_group_id="retained",
            keep_connections=True,
        ):
            pass
        with pytest.raises(RuntimeError, match="matching"):
            async with WorkerGroupMaster(
                workers.addresses,
                launcher=LocalWorkers(None, tmp_path),
                worker_group_id="other",
            ):
                pass
        assert all(m.group == "retained" for m in workers.managers.values())
        assert not [c for c in workers.calls if c[1] == "destroy"]

    asyncio.run(run())


def test_partial_build_failure_cleans_successful_services(workers, tmp_path):
    workers.fail.add("decoder")

    async def run():
        with pytest.raises(RuntimeError, match="build failed: decoder"):
            async with WorkerGroupMaster(
                workers.addresses, launcher=LocalWorkers(None, tmp_path)
            ):
                pass
        assert all(m.group is None for m in workers.managers.values())
        assert len([c for c in workers.calls if c[1] == "destroy"]) == 2
        assert ("decoder", "rollback") in workers.calls
        assert all(m.state == "idle" for m in workers.managers.values())

    asyncio.run(run())


@pytest.mark.parametrize(
    "value",
    [
        "localhost",
        "http://host:0",
        "http://user:pass@host:1234",
        "http://host:1234/path",
    ],
)
def test_invalid_management_addresses_are_rejected(value):
    with pytest.raises(ValueError):
        http_address(value)


def test_existing_media_client_receives_queried_ws_urls(workers, tmp_path):
    from encoder_test_master import run_test

    async def run():
        args = SimpleNamespace(
            output_kind="rgb",
            direct_ws=False,
            output=str(tmp_path / "result"),
            encoder_http=workers.addresses["encoder"],
            dit_http=workers.addresses["dit"],
            decoder_http=workers.addresses["decoder"],
            worker_launch_config=None,
            worker_log_dir=None,
            worker_group_id=None,
            worker_startup_timeout=5,
            worker_control_timeout=5,
            keep_connections=False,
        )

        async def client(args, *, output_sink):
            assert output_sink is None
            assert args.url == "ws://127.0.0.1:30000/v1/realtime_video/generate"
            assert args.decoder_url == "ws://127.0.0.1:38000/v1/realtime_video/output"
            assert all(m.state == "connected" for m in workers.managers.values())
            from pathlib import Path

            Path(args.output).mkdir()
            return dict(received_frames=120)

        result = await run_test(args, client)
        assert result["received_frames"] == 120
        assert result["worker_group"]["connections_kept"] is False
        assert all(m.group is None for m in workers.managers.values())

    asyncio.run(run())


def test_cancelled_build_waits_for_inflight_connection_before_cleanup(
    workers, tmp_path
):
    release = threading.Event()
    workers.holds["decoder"] = release

    async def run():
        group = WorkerGroupMaster(
            workers.addresses,
            launcher=LocalWorkers(None, tmp_path),
            startup_timeout=5,
            control_timeout=5,
        )
        task = asyncio.create_task(group.__aenter__())
        try:
            async with asyncio.timeout(5):
                while workers.managers["decoder"].state != "connecting":
                    await asyncio.sleep(0.005)
            task.cancel()
            await asyncio.sleep(0.05)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert all(m.group is None for m in workers.managers.values())
        finally:
            release.set()

    asyncio.run(run())


def test_partial_build_not_accepted_cleanup_does_not_acquire_a_new_lease(
    workers, tmp_path
):
    # This failure happens before Decoder's build callback, so the two
    # accepted roles must not wait on the fixture's three-role build barrier.
    for role in ("encoder", "dit"):
        workers.managers[role].build_callback = lambda group, topology, role=role: (
            workers.calls.append((role, "build", group))
        )

    async def run():
        group = WorkerGroupMaster(
            workers.addresses, launcher=LocalWorkers(None, tmp_path)
        )
        rpc = group.rpc

        async def expire_before_build(role, path, payload=None, **kwargs):
            if role == "decoder" and path == "/build_group":
                # No callback and no last_destroyed marker exist for this group.
                workers.managers[role].deadline = 0
            return await rpc(role, path, payload, **kwargs)

        group.rpc = expire_before_build
        with pytest.raises(RuntimeError, match="build failed: decoder"):
            async with group:
                pass
        assert ("decoder", "build") not in workers.calls
        assert all(m.state == "idle" for m in workers.managers.values())
        assert all(m.token is None for m in workers.managers.values())

    asyncio.run(run())
