# SPDX-License-Identifier: Apache-2.0
"""Two real Gloo ranks exercise DiT transport ownership and native broadcasts."""

import ast
import asyncio
import multiprocessing
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.realtime.worker_parallel import ParallelChannel

TensorMetadata = namedtuple("TensorMetadata", "device dtype size")


def coordinator(rank):
    # Use the actual runtime communication methods without importing CUDA-only
    # model/platform dependencies into this CPU transport test.
    path = (
        Path(__file__).resolve().parents[3] / "runtime/distributed/group_coordinator.py"
    )
    tree = ast.parse(path.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_split_tensor_dict", "_update_nested_dict")
    ]
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "GroupCoordinator"
    )
    nodes += [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("broadcast_object", "broadcast_tensor_dict")
    ]
    namespace = {"torch": torch, "TensorMetadata": TensorMetadata}
    exec(  # noqa: S102 - execute repository communication methods unchanged
        compile(
            "from __future__ import annotations\n"
            + ast.unparse(ast.Module(body=nodes, type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    group = SimpleNamespace(
        rank=rank,
        rank_in_group=rank,
        world_size=2,
        ranks=[0, 1],
        cpu_group=dist.group.WORLD,
        device_group=dist.group.WORLD,
        mq_broadcaster=None,
    )
    for name in ("broadcast_object", "broadcast_tensor_dict"):
        setattr(group, name, namespace[name].__get__(group))
    return group


class LocalChannel:
    def __init__(self):
        self.error = None
        self.sent = []
        self.paused = self.closed = False
        self.messages = [
            {"type": "session.init", "seed": 123, "width": 832, "height": 480},
            {
                "type": "conditioning",
                "event_id": 0,
                "tensors": {
                    "context": torch.arange(12).reshape(3, 4),
                    "roles": torch.arange(3),
                },
                "inputs": {"visual": {"0": torch.ones(2, 3)}, "audio": {}},
            },
            {"type": "source.moments", "tensor": torch.arange(8)},
            {
                "type": "conditioning",
                "event_id": 1,
                "tensors": {"context": torch.zeros(2, 4)},
            },
            {"type": "source.end"},
        ]

    def poll(self):
        return bool(self.messages)

    def receive(self):
        return self.messages.pop(0)

    def send(self, message):
        if self.error is not None:
            raise self.error
        self.sent.append(message)

    def pause_receiving(self):
        self.paused = True

    def close(self):
        self.closed = True


def exercise_lifecycle(group):
    from sglang.multimodal_gen.runtime.realtime.polling_task_queue import (
        PollingTaskQueue,
    )
    from sglang.multimodal_gen.runtime.realtime.worker_group_dit import DiTGroups

    class Groups(DiTGroups):
        def build(self, name, topology):
            self.seen = []
            self.link = ParallelChannel(group, LocalChannel() if self.leader else None)

            async def process(queue):
                self.seen.append(queue.popleft()["type"])
                self.link.send({"type": "input.processed"})

            self.service = SimpleNamespace(
                queue=PollingTaskQueue(process),
                worker=None,
                _close_decoder=lambda: None,
            )

        async def destroy(self, *, force=False):
            await super().destroy(force=force)
            if not force:
                self.running = False

    groups = Groups(
        {},
        store=None,
        device="cpu",
        timeout=15,
        create_session=None,
        codecs=None,
        parallel=group,
    )

    def commands():
        groups.submit("build", "group-a", {})
        groups.submit("destroy")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(commands) if group.rank_in_group == 0 else None
        asyncio.run(groups.run())
        if future is not None:
            future.result(timeout=5)
    assert groups.seen == [
        "session.init",
        "conditioning",
        "source.moments",
        "conditioning",
        "source.end",
    ]
    assert groups.service is None and groups.link is None


def exercise_progress_failure(channel):
    path = (
        Path(__file__).resolve().parents[3]
        / "runtime/pipelines_core/stages/model_specific_stages/zing_h3/dit_decoder.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DiTDecoderOutput"
    )
    namespace = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 - use actual adapter method
    output = object.__new__(namespace["DiTDecoderOutput"])
    output.parallel = channel

    def progress(**kwargs):
        assert channel.leader, "Only rank 0 may send decoder progress"
        raise RuntimeError("injected decoder progress failure")

    output.remote = SimpleNamespace(source_progress=progress)
    with pytest.raises(RuntimeError, match="injected decoder progress failure"):
        output.source_progress(1, [])


def _rank(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    try:
        group = coordinator(rank)
        local = LocalChannel() if rank == 0 else None
        channel = ParallelChannel(group, local)
        # HTTP pause is local and must not require a collective on the peer.
        if rank == 0:
            channel.pause_receiving_local()
            assert local.paused
        seen = []
        while channel.poll():
            message = channel.receive()
            seen.append(message["type"])
            if message["type"] == "conditioning" and message["event_id"] == 0:
                assert torch.equal(
                    message["tensors"]["context"], torch.arange(12).reshape(3, 4)
                )
                assert message["inputs"]["audio"] == {}
                assert torch.equal(message["inputs"]["visual"]["0"], torch.ones(2, 3))
            channel.send({"type": "input.processed"})
        assert seen == [
            "session.init",
            "conditioning",
            "source.moments",
            "conditioning",
            "source.end",
        ]
        if rank == 0:
            assert len(local.sent) == 5
            local.error = RuntimeError("injected send failure")
        with pytest.raises(RuntimeError, match="injected send failure"):
            channel.send({})
        assert channel.error == "injected send failure"
        # A follower-side setup failure must also reach rank 0, before inference.
        with pytest.raises(RuntimeError, match="follower setup failed"):
            channel.check_errors(
                RuntimeError("follower setup failed") if rank == 1 else None
            )
        channel.pause_receiving()
        channel.close()
        if rank == 0:
            assert local.closed
        # Rollback leaves the SP worker group reusable.
        channel = ParallelChannel(group, LocalChannel() if rank == 0 else None)
        assert channel.receive()["type"] == "session.init"
        channel.close()
        exercise_progress_failure(channel)
        exercise_lifecycle(group)
    finally:
        dist.destroy_process_group()


def test_two_rank_input_order_native_tensors_and_failure_cleanup(tmp_path):
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_rank, args=(rank, "file://" + str(tmp_path / "gloo")))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=35)
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def test_single_rank_preserves_local_channel():
    local = LocalChannel()
    channel = ParallelChannel(local=local)
    assert channel.receive()["seed"] == 123
    channel.send({"ok": True})
    channel.close()
    assert local.sent == [{"ok": True}] and local.closed


@pytest.mark.parametrize("decoder_failed", [False, True])
def test_dit_owner_preserves_failure_category_after_decoder_close(decoder_failed):
    """Transport cleanup allows reuse; model failure still quarantines the worker."""
    import threading
    from queue import Queue

    from sglang.multimodal_gen.runtime.realtime.polling_task_queue import (
        PollingTaskQueue,
    )
    from sglang.multimodal_gen.runtime.realtime.worker_group_dit import DiTGroups

    path = (
        Path(__file__).resolve().parents[3]
        / "runtime/pipelines_core/stages/model_specific_stages/zing_h3/dit_link.py"
    )
    node = next(
        item
        for item in ast.parse(path.read_text()).body
        if isinstance(item, ast.ClassDef) and item.name == "DiTInputService"
    )
    namespace = {"PollingTaskQueue": PollingTaskQueue}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )

    async def fail(_message):
        raise RuntimeError("Injected worker failure")

    closed = []
    service = namespace["DiTInputService"](
        SimpleNamespace(send=lambda message: None), None
    )
    service.owner = ("session", "generation")
    service.worker = SimpleNamespace(put=fail, close=lambda: None)
    service.decoder = SimpleNamespace(
        remote=SimpleNamespace(
            failure=RuntimeError("Disconnected") if decoder_failed else None
        ),
        close=lambda: closed.append(True),
    )
    service.queue.append(
        dict(
            type="conditioning",
            transfer_id=0,
            session_id="session",
            generation_id="generation",
        )
    )
    owner = DiTGroups.__new__(DiTGroups)
    owner.running = owner.leader = True
    owner.parallel = None
    owner.commands = Queue()
    owner.service = service
    owner.link = SimpleNamespace(error=None)
    owner.manager = SimpleNamespace(
        lock=threading.RLock(),
        state="connected",
        restart_required=False,
        token="token",
        group="group",
        topology={},
    )

    async def destroy(*, force):
        assert force
        owner.running = False

    owner.destroy = destroy
    asyncio.run(owner.run())
    assert closed == [True]
    assert service.decoder is None
    assert service.transport_failed is decoder_failed
    assert owner.manager.restart_required is (not decoder_failed)
    assert owner.manager.state == ("idle" if decoder_failed else "failed")
