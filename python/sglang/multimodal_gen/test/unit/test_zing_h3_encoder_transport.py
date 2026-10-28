# SPDX-License-Identifier: Apache-2.0
"""Exercise an unregistered bridge group and bounded remote control on CPU."""

import queue
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_transport import (
    bridge_group,
    receive_moments,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
    expected_progressive_slices,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.remote_encoder import (
    RemoteEncoderSession,
    available_latents,
)


def _bridge_worker(rank, path):
    torch.set_num_threads(1)
    store = dist.FileStore(path, 2)
    group = bridge_group(store, rank, backend="gloo")
    expected = torch.arange(96, dtype=torch.float32).reshape(1, 48, 2, 1, 1)
    assert not dist.is_initialized()
    if rank:
        group.send([expected], 0, 0).wait()
    else:
        actual = receive_moments(group, [1, 48, 2, 1, 1], torch.float32, "cpu")
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_bridge_does_not_join_default_world(tmp_path):
    mp.spawn(_bridge_worker, args=(str(tmp_path / "bridge"),), nprocs=2, join=True)


@pytest.mark.parametrize("frames", range(1, 70))
def test_ready_watermarks_do_not_require_future_input(frames):
    assert available_latents(frames, True) == expected_progressive_slices(frames)[-1][1]
    complete = [
        end
        for _, end, watermark in expected_progressive_slices(85)
        if watermark <= frames
    ]
    assert available_latents(frames, False) == max(complete)


def _fake_producer(self):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.remote_encoder import (
        available_latents,
    )

    frames = emitted = 0
    self.started_slices = []
    while not self.stop.is_set():
        try:
            op, value, _ = self.commands.get(timeout=0.05)
        except queue.Empty:
            continue
        if op == "append":
            frames += value.shape[1]
        target = available_latents(frames, op == "finish")
        while emitted < target and not self.stop.is_set():
            if not self.permits.acquire(timeout=0.05):
                continue
            if self.stop.is_set():
                break
            end = min(target, emitted + (1 if emitted % 5 == 0 else 2))
            self.started_slices.append((emitted, end))
            dependency = min(
                frames,
                emitted // 5 * 17 + (1 if end % 5 == 1 else 9 if end % 5 == 3 else 17),
            )
            tensor = torch.zeros(1, 48, end - emitted, 1, 1)
            self.results.put(
                (
                    dict(
                        start=emitted,
                        end=end,
                        dependency=dependency,
                        shape=list(tensor.shape),
                        dtype="float32",
                    ),
                    tensor,
                )
            )
            emitted = end


@pytest.mark.parametrize("mode", ["serial", "overlap"])
def test_remote_queue_eos_and_disposal(monkeypatch, mode):
    from sglang.multimodal_gen.runtime.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state,
        "get_world_group",
        lambda: SimpleNamespace(
            rank_in_group=0, ranks=[0], cpu_group=None, device_group=None
        ),
    )
    monkeypatch.setattr(dist, "broadcast_object_list", lambda *a, **k: None)
    monkeypatch.setattr(dist, "broadcast", lambda *a, **k: None)
    monkeypatch.setattr(RemoteEncoderSession, "_produce", _fake_producer)
    codecs = SimpleNamespace(
        modules={"video_vae": SimpleNamespace(encoder=torch.nn.Linear(1, 1))}
    )
    session = RemoteEncoderSession(
        codecs,
        capacity_frames=34,
        config=dict(port=31001, mode=mode, timeout_seconds=2),
    )
    session.append(torch.zeros(3, 19, 2, 2, dtype=torch.uint8), frame_start=0)
    with pytest.raises(ValueError, match="capacity"):
        session.append(torch.zeros(3, 17, 2, 2, dtype=torch.uint8), frame_start=19)
    rows = []
    while session.ready:
        item = session.pop()
        rows.append((item.start, item.end, item.source_dependency_end))
    session.finish()
    while session.ready:
        item = session.pop()
        rows.append((item.start, item.end, item.source_dependency_end))
    assert rows == expected_progressive_slices(19)
    session.dispose()
    assert not session.thread.is_alive()
    assert session.state_usage().retained_bytes == 0
    session.dispose()


def test_remote_worker_token_can_use_client_session_identity(monkeypatch):
    from sglang.multimodal_gen.runtime.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state,
        "get_world_group",
        lambda: SimpleNamespace(
            rank_in_group=0, ranks=[0], cpu_group=None, device_group=None
        ),
    )
    monkeypatch.setattr(RemoteEncoderSession, "_produce", lambda self: None)
    codecs = SimpleNamespace(
        modules={"video_vae": SimpleNamespace(encoder=torch.nn.Linear(1, 1))}
    )
    session = RemoteEncoderSession(
        codecs,
        capacity_frames=34,
        config=dict(
            port=31001,
            mode="serial",
            timeout_seconds=1,
            session_id="client-session",
        ),
    )
    assert session.token == "client-session"
    session.dispose()


def test_receive_rejects_unbounded_or_invalid_tensor():
    for shape in ([1, 48, 3, 1, 1], [1, 48, 2, 100000, 100000], [1, 0, 2, 1, 1]):
        with pytest.raises(ValueError):
            receive_moments(None, shape, torch.float32, "cpu")


@pytest.mark.parametrize("mode", ["serial", "overlap"])
def test_cancel_while_producer_waits_for_credit_then_reset(monkeypatch, mode):
    from sglang.multimodal_gen.runtime.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state,
        "get_world_group",
        lambda: SimpleNamespace(
            rank_in_group=0, ranks=[0], cpu_group=None, device_group=None
        ),
    )
    monkeypatch.setattr(RemoteEncoderSession, "_produce", _fake_producer)
    codecs = SimpleNamespace(
        modules={"video_vae": SimpleNamespace(encoder=torch.nn.Linear(1, 1))}
    )
    session = RemoteEncoderSession(
        codecs,
        capacity_frames=34,
        config=dict(port=31001, mode=mode, timeout_seconds=1),
    )
    session.append(torch.zeros(3, 34, 2, 2, dtype=torch.uint8), frame_start=0)
    if mode == "overlap":
        deadline = time.monotonic() + 2
        while session.results.qsize() < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert session.results.qsize() == 3
        assert session.started_slices == [(0, 1), (1, 3), (3, 5)]
        assert not session.permits.acquire(blocking=False)
    old_thread, old_token = session.thread, session.token
    session.reset()
    assert not old_thread.is_alive()
    assert session.token != old_token
    assert session.accepted_frames == session.emitted_latents == 0
    assert session.results.empty() and session.commands.empty()
    session.dispose()
    assert not session.thread.is_alive()


@pytest.mark.parametrize("capacity_bytes,expected", [(4096, 25), (12 * 20, 11)])
def test_append_capacity_uses_formal_frames_and_bytes(
    monkeypatch, capacity_bytes, expected
):
    from sglang.multimodal_gen.runtime.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state, "get_world_group", lambda: SimpleNamespace(rank_in_group=1)
    )
    codecs = SimpleNamespace(
        modules={"video_vae": SimpleNamespace(encoder=torch.nn.Linear(1, 1))}
    )
    encoder = RemoteEncoderSession(
        codecs,
        capacity_frames=34,
        config=dict(port=31001, mode="overlap", capacity_bytes=capacity_bytes),
    )
    encoder.append(torch.zeros(3, 9, 2, 2, dtype=torch.uint8), frame_start=0)
    assert encoder.append_capacity_frames == expected
    encoder.prefetched_frames = 100  # Background progress cannot grant headroom.
    assert encoder.append_capacity_frames == expected
    encoder.finish()
    assert encoder.append_capacity_frames == 0
    encoder.dispose()


def test_producer_failure_is_broadcast_before_tensor_collective(monkeypatch):
    from sglang.multimodal_gen.runtime.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state,
        "get_world_group",
        lambda: SimpleNamespace(
            rank_in_group=0, ranks=[0], cpu_group=None, device_group=None
        ),
    )
    monkeypatch.setattr(RemoteEncoderSession, "_produce", lambda self: None)
    metadata = []
    monkeypatch.setattr(
        dist, "broadcast_object_list", lambda value, **kw: metadata.extend(value)
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Failed producer must not enter a tensor collective")

    monkeypatch.setattr(dist, "broadcast", unexpected)
    codecs = SimpleNamespace(
        modules={"video_vae": SimpleNamespace(encoder=torch.nn.Linear(1, 1))}
    )
    session = RemoteEncoderSession(
        codecs,
        capacity_frames=34,
        config=dict(port=31001, mode="overlap", timeout_seconds=1),
    )
    session.append(torch.zeros(3, 1, 2, 2, dtype=torch.uint8), frame_start=0)
    session.failure = RuntimeError("Injected transport failure")
    with pytest.raises(RuntimeError, match="Remote encoder failed"):
        session.pop()
    assert metadata == [{"error": "Remote encoder failed"}]
    with pytest.raises(RuntimeError, match="cleanup failed"):
        session.dispose()


@pytest.mark.parametrize("operation", ["dispose", "reset", "reset_in_except"])
def test_close_failure_is_not_erased_or_reused(monkeypatch, operation):
    from sglang.multimodal_gen.runtime.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state, "get_world_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    error = TimeoutError("Injected worker close ACK timeout")
    starts = []

    def producer(session):
        starts.append(session.token)
        assert session.stop.wait(2)
        session.failure = error

    monkeypatch.setattr(RemoteEncoderSession, "_produce", producer)
    codecs = SimpleNamespace(
        modules={"video_vae": SimpleNamespace(encoder=torch.nn.Linear(1, 1))}
    )
    session = RemoteEncoderSession(
        codecs,
        capacity_frames=34,
        config=dict(port=31001, mode="overlap", timeout_seconds=1),
    )
    token = session.token
    with pytest.raises(RuntimeError, match="cleanup failed") as caught:
        if operation == "reset_in_except":
            try:
                raise ValueError("Prior request error")
            except ValueError:
                session.reset()
        else:
            getattr(session, operation)()
    assert caught.value.__cause__ is error
    assert starts == [token] and session.token == token
    assert session.disposed and session.failure is error
    assert not session.thread.is_alive() and session.state_usage().retained_bytes == 0
    session.dispose()  # local cleanup remains idempotent
    with pytest.raises(ValueError, match="disposed"):
        session.reset()


def test_cleanup_in_finally_preserves_primary_error(monkeypatch):
    from sglang.multimodal_gen.runtime.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state, "get_world_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    failure = TimeoutError("Injected close timeout")

    def producer(session):
        assert session.stop.wait(2)
        session.failure = failure

    monkeypatch.setattr(RemoteEncoderSession, "_produce", producer)
    codecs = SimpleNamespace(
        modules={"video_vae": SimpleNamespace(encoder=torch.nn.Linear(1, 1))}
    )
    session = RemoteEncoderSession(
        codecs,
        capacity_frames=34,
        config=dict(port=31001, mode="overlap", timeout_seconds=1),
    )
    primary = ValueError("Original generation error")
    with pytest.raises(ValueError) as caught:
        try:
            raise primary
        finally:
            session.dispose()
    assert caught.value is primary and session.failure is failure and session.disposed
    assert not session.thread.is_alive()
