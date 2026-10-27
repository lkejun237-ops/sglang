# SPDX-License-Identifier: Apache-2.0
"""Tier 2 streaming transport tests: 2-rank NCCL bridge, worker control, and SHM publisher."""

import importlib
import json

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _load_nccl_bridge():
    try:
        return importlib.import_module(
            "sglang.multimodal_gen.runtime.disaggregation.transport.nccl_bridge"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"sglang.multimodal_gen.runtime.disaggregation.transport.nccl_bridge not implemented: {exc}"
        )


def _load_worker_control():
    try:
        return importlib.import_module(
            "sglang.multimodal_gen.runtime.disaggregation.transport.worker_control"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"sglang.multimodal_gen.runtime.disaggregation.transport.worker_control not implemented: {exc}"
        )


def _load_shm_publisher():
    try:
        return importlib.import_module(
            "sglang.multimodal_gen.runtime.disaggregation.transport.shm_publisher"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"sglang.multimodal_gen.runtime.disaggregation.transport.shm_publisher not implemented: {exc}"
        )


# ==============================================================================
# Helper worker for multiprocessing Gloo bridge test
# ==============================================================================


def _bridge_worker_proc(rank, store_file, queue):
    try:
        nccl_bridge = importlib.import_module(
            "sglang.multimodal_gen.runtime.disaggregation.transport.nccl_bridge"
        )
        store = dist.FileStore(store_file, 2)
        group = nccl_bridge.create_bridge_group(
            store, rank, prefix="test-bridge", backend="gloo", timeout_seconds=10
        )
        assert not dist.is_initialized(), "Default dist world must not be initialized"
        if rank == 0:
            tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
            nccl_bridge.send_bridge_tensor(group, tensor, peer=1, tag=0)
        else:
            received = nccl_bridge.receive_bridge_tensor(
                group, shape=(4,), dtype=torch.float32, device="cpu", peer=0, tag=0
            )
            queue.put(received.tolist())
    except Exception as exc:
        queue.put(("ERROR", str(exc)))


# ==============================================================================
# 1. NCCL / Gloo 2-Rank Bridge tests
# ==============================================================================


def test_bridge_group_mp_spawn_send_recv(tmp_path):
    nccl_bridge = _load_nccl_bridge()
    store_file = str(tmp_path / "bridge_store")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(_bridge_worker_proc, args=(store_file, queue), nprocs=2, join=True)
    res = queue.get(timeout=5)
    assert res == [1.0, 2.0, 3.0, 4.0]


def test_nccl_bridge_does_not_mutate_stdlib_multiprocessing_spawn():
    _load_nccl_bridge()
    import multiprocessing.spawn as stdlib_mp_spawn
    import types

    assert type(stdlib_mp_spawn) is types.ModuleType, (
        f"multiprocessing.spawn class was mutated: {type(stdlib_mp_spawn)}"
    )
    assert not callable(stdlib_mp_spawn), (
        "multiprocessing.spawn must not be made callable"
    )


def test_validate_bridge_tensor_valid():
    nccl_bridge = _load_nccl_bridge()
    # 5D shape for moments or latents
    shape = (1, 48, 1, 16, 16)
    nccl_bridge.validate_bridge_tensor(
        shape,
        torch.float32,
        expected_ndim=5,
        expected_batch=1,
        expected_channels=48,
        max_temporal=2,
    )

    # 5D shape for decoder latents
    dec_shape = (1, 24, 2, 8, 8)
    nccl_bridge.validate_bridge_tensor(
        dec_shape,
        torch.bfloat16,
        expected_ndim=5,
        expected_batch=1,
        expected_channels=24,
        max_temporal=2,
    )


def test_validate_bridge_tensor_invalid_ndim():
    nccl_bridge = _load_nccl_bridge()
    shape = (1, 48, 16, 16)  # 4D
    with pytest.raises(ValueError, match="shape|dimensions|ndim"):
        nccl_bridge.validate_bridge_tensor(shape, torch.float32, expected_ndim=5)


def test_validate_bridge_tensor_invalid_batch():
    nccl_bridge = _load_nccl_bridge()
    shape = (2, 48, 1, 16, 16)  # batch=2 != 1
    with pytest.raises(ValueError, match="batch|dimensions"):
        nccl_bridge.validate_bridge_tensor(shape, torch.float32, expected_batch=1)


def test_validate_bridge_tensor_channel_mismatch():
    nccl_bridge = _load_nccl_bridge()
    shape = (1, 24, 1, 16, 16)
    with pytest.raises(ValueError, match="channel|dimensions"):
        nccl_bridge.validate_bridge_tensor(shape, torch.float32, expected_channels=48)


def test_validate_bridge_tensor_temporal_exceeded():
    nccl_bridge = _load_nccl_bridge()
    shape = (1, 48, 3, 16, 16)  # temporal=3 > max=2
    with pytest.raises(ValueError, match="temporal|dimensions|shape"):
        nccl_bridge.validate_bridge_tensor(shape, torch.float32, max_temporal=2)


def test_validate_bridge_tensor_unsupported_dtype():
    nccl_bridge = _load_nccl_bridge()
    shape = (1, 48, 1, 16, 16)
    with pytest.raises(ValueError, match="dtype"):
        nccl_bridge.validate_bridge_tensor(shape, torch.int32)


def test_validate_bridge_tensor_capacity_exceeded():
    nccl_bridge = _load_nccl_bridge()
    # 1 * 48 * 1 * 2048 * 2048 * 4 bytes ≈ 805 MB > 256 MiB
    shape = (1, 48, 1, 2048, 2048)
    with pytest.raises(ValueError, match="capacity|exceed"):
        nccl_bridge.validate_bridge_tensor(shape, torch.float32, max_bytes=256 << 20)


def test_encoder_and_decoder_transport_reuse_nccl_bridge():
    nccl_bridge = _load_nccl_bridge()
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_transport as dec_trans
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_transport as enc_trans

    assert hasattr(enc_trans, "nccl_bridge") or "nccl_bridge" in enc_trans.__dict__, (
        "encoder_transport must delegate to nccl_bridge"
    )
    assert hasattr(dec_trans, "nccl_bridge") or "nccl_bridge" in dec_trans.__dict__, (
        "decoder_transport must delegate to nccl_bridge"
    )


# ==============================================================================
# 2. Worker Control (RoleCommandControl & RoleControlError) tests
# ==============================================================================


class _MockDist:
    def __init__(self, world_size=2):
        self._world_size = world_size

    def get_world_size(self, group=None):
        return self._world_size

    def all_gather_object(self, object_list, obj, group=None):
        for i in range(len(object_list)):
            object_list[i] = obj

    def broadcast_object_list(self, object_list, src=0, group=None):
        pass


class _MockSocket:
    def __init__(self, messages=None):
        self.messages = list(messages or [])

    def poll(self, timeout=250):
        return bool(self.messages)

    def recv_multipart(self):
        if not self.messages:
            raise RuntimeError("No messages")
        return self.messages.pop(0)


def test_role_command_control_agree_normal():
    ctrl_mod = _load_worker_control()
    RoleCommandControl = ctrl_mod.RoleCommandControl

    control = RoleCommandControl(
        dist=_MockDist(world_size=2),
        rank=0,
        socket=None,
        idle_group="idle",
        operation_group="op",
        role_name="test_worker",
    )
    # Both ranks match, no error reported
    control.agree("startup", value=123)


def test_role_command_control_agree_cross_rank_error():
    ctrl_mod = _load_worker_control()
    RoleCommandControl = ctrl_mod.RoleCommandControl
    RoleControlError = ctrl_mod.RoleControlError

    class _ErrorDist(_MockDist):
        def all_gather_object(self, object_list, obj, group=None):
            object_list[0] = ("startup", None, False)
            object_list[1] = ("startup", None, True)  # rank 1 reported error

    control = RoleCommandControl(
        dist=_ErrorDist(world_size=2),
        rank=0,
        socket=None,
        idle_group="idle",
        operation_group="op",
        role_name="test_worker",
    )
    with pytest.raises(RoleControlError, match="failed on a participating rank"):
        control.agree("startup")


def test_role_command_control_agree_state_divergence():
    ctrl_mod = _load_worker_control()
    RoleCommandControl = ctrl_mod.RoleCommandControl
    RoleControlError = ctrl_mod.RoleControlError

    class _DivergentDist(_MockDist):
        def all_gather_object(self, object_list, obj, group=None):
            object_list[0] = ("step", "token_A", False)
            object_list[1] = ("step", "token_B", False)  # state mismatch

    control = RoleCommandControl(
        dist=_DivergentDist(world_size=2),
        rank=0,
        socket=None,
        idle_group="idle",
        operation_group="op",
        role_name="test_worker",
    )
    with pytest.raises(RoleControlError, match="match"):
        control.agree("step")


def test_role_command_control_next_command():
    ctrl_mod = _load_worker_control()
    RoleCommandControl = ctrl_mod.RoleCommandControl

    socket = _MockSocket([[json.dumps({"op": "ping"}).encode()]])
    control = RoleCommandControl(
        dist=_MockDist(world_size=2),
        rank=0,
        socket=socket,
        idle_group="idle",
        operation_group="op",
        role_name="test_worker",
    )
    cmd, payload = control.next()
    assert cmd == {"op": "ping"}
    assert payload is None


def test_role_command_control_next_invalid_command_raises():
    ctrl_mod = _load_worker_control()
    RoleCommandControl = ctrl_mod.RoleCommandControl
    RoleControlError = ctrl_mod.RoleControlError

    socket = _MockSocket([[b"not a valid json"]])
    control = RoleCommandControl(
        dist=_MockDist(world_size=2),
        rank=0,
        socket=socket,
        idle_group="idle",
        operation_group="op",
        role_name="test_worker",
    )
    with pytest.raises(RoleControlError, match="command"):
        control.next()


def test_workers_use_role_command_control():
    _load_worker_control()
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_worker as dec_worker
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker as enc_worker

    assert (
        hasattr(dec_worker, "RoleCommandControl")
        or "RoleCommandControl" in dec_worker.__dict__
    ), "decoder_worker must import and use RoleCommandControl"
    assert (
        hasattr(enc_worker, "RoleCommandControl")
        or "RoleCommandControl" in enc_worker.__dict__
    ), "encoder_worker must import and use RoleCommandControl"


# ==============================================================================
# 3. Shared Memory Frame Publisher tests
# ==============================================================================


def test_shm_publisher_reserve_success(monkeypatch):
    shm_mod = _load_shm_publisher()
    SharedMemoryFramePublisher = shm_mod.SharedMemoryFramePublisher

    publisher = SharedMemoryFramePublisher(timeout_seconds=2.0, max_in_flight=2)
    monkeypatch.setattr(
        shm_mod,
        "reserve_async_shared_memory_payload",
        lambda num_bytes: {"slot": 1, "num_bytes": num_bytes},
    )

    ref = publisher.reserve(
        frames=1, width=32, height=32, identity={"session_id": "sess_1"}
    )
    assert ref["kind"] == "raw_rgb24_frames"
    assert ref["batch_num_frames"] == [1]
    assert ref["width"] == 32
    assert ref["height"] == 32
    assert ref["channels"] == 3
    assert ref["session_id"] == "sess_1"


def test_shm_publisher_reserve_oversized_rejected():
    shm_mod = _load_shm_publisher()
    SharedMemoryFramePublisher = shm_mod.SharedMemoryFramePublisher

    publisher = SharedMemoryFramePublisher(timeout_seconds=2.0, max_in_flight=2)
    with pytest.raises(ValueError, match="capacity|payload"):
        # 1000 * 4096 * 4096 * 3 > max bytes
        publisher.reserve(
            frames=1000, width=4096, height=4096, identity={"session_id": "large"}
        )


def test_shm_publisher_slots_exhaustion_and_timeout(monkeypatch):
    shm_mod = _load_shm_publisher()
    SharedMemoryFramePublisher = shm_mod.SharedMemoryFramePublisher

    publisher = SharedMemoryFramePublisher(timeout_seconds=0.05, max_in_flight=2)
    monkeypatch.setattr(
        shm_mod,
        "reserve_async_shared_memory_payload",
        lambda num_bytes: {"slot": 1, "num_bytes": num_bytes},
    )

    ref1 = publisher.reserve(frames=1, width=32, height=32, identity={})
    ref2 = publisher.reserve(frames=1, width=32, height=32, identity={})

    with pytest.raises(TimeoutError, match="slots are full"):
        publisher.reserve(frames=1, width=32, height=32, identity={})


def test_shm_publisher_slot_released_on_allocation_failure(monkeypatch):
    shm_mod = _load_shm_publisher()
    SharedMemoryFramePublisher = shm_mod.SharedMemoryFramePublisher

    publisher = SharedMemoryFramePublisher(timeout_seconds=0.05, max_in_flight=1)

    def failing_alloc(num_bytes):
        raise OSError("SHM allocation failed")

    monkeypatch.setattr(shm_mod, "reserve_async_shared_memory_payload", failing_alloc)

    with pytest.raises(OSError, match="SHM allocation failed"):
        publisher.reserve(frames=1, width=32, height=32, identity={})

    # The semaphore slot must be released so next attempt does not time out
    monkeypatch.setattr(
        shm_mod,
        "reserve_async_shared_memory_payload",
        lambda num_bytes: {"slot": 1, "num_bytes": num_bytes},
    )
    ref = publisher.reserve(frames=1, width=32, height=32, identity={})
    assert ref["batch_num_frames"] == [1]


def test_shm_publisher_fail_releases_slot(monkeypatch):
    shm_mod = _load_shm_publisher()
    SharedMemoryFramePublisher = shm_mod.SharedMemoryFramePublisher

    monkeypatch.setattr(
        shm_mod,
        "publish_async_shared_memory_error",
        lambda ref, error: None,
        raising=False,
    )
    monkeypatch.setattr(
        shm_mod,
        "wait_for_async_shared_memory_terminal",
        lambda ref, timeout=None: None,
        raising=False,
    )
    monkeypatch.setattr(
        shm_mod,
        "reserve_async_shared_memory_payload",
        lambda num_bytes: {"slot": 1, "num_bytes": num_bytes},
    )

    publisher = SharedMemoryFramePublisher(timeout_seconds=0.1, max_in_flight=1)
    try:
        ref = publisher.reserve(frames=1, width=32, height=32, identity={})

        # Slots are full now (max_in_flight=1)
        with pytest.raises(TimeoutError, match="slots are full"):
            publisher.reserve(frames=1, width=32, height=32, identity={})

        # Report failure: accepts (reference, error)
        publisher.fail(ref, RuntimeError("decode failed"))

        # Wait for background _reclaim to release slot
        import time

        reclaimed = False
        for _ in range(50):
            if publisher._slots.acquire(blocking=False):
                publisher._slots.release()
                reclaimed = True
                break
            time.sleep(0.01)
        assert reclaimed, "Semaphore slot was not released after publisher.fail"

        # Slot should be available again
        ref2 = publisher.reserve(frames=1, width=32, height=32, identity={})
        assert ref2["batch_num_frames"] == [1]
    finally:
        if hasattr(publisher, "close"):
            publisher.close()
        elif hasattr(publisher, "_executor") and publisher._executor is not None:
            publisher._executor.shutdown(wait=False)


def test_decoder_worker_uses_shm_publisher():
    _load_shm_publisher()
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_worker as dec_worker

    assert (
        hasattr(dec_worker, "SharedMemoryFramePublisher")
        or "SharedMemoryFramePublisher" in dec_worker.__dict__
    ), "decoder_worker must import and use SharedMemoryFramePublisher"
