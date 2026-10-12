# SPDX-License-Identifier: Apache-2.0
"""CPU control tests run the production producer over real local ZeroMQ.

CUDA transport is fault-injected here; this is not a GPU overlap acceptance.
"""

import ast
import hashlib
import importlib.util
import json
import logging
import math
import queue
import sys
import tempfile
import threading
import time
import uuid
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import zmq

ROOT = Path(__file__).resolve().parents[2] / "python/sglang/multimodal_gen/runtime"


def production_class():
    path = (
        ROOT / "pipelines_core/stages/model_specific_stages/zing_h3/remote_encoder.py"
    )
    tree = ast.parse(path.read_text())
    nodes = [
        node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    ]
    recovery_spec = importlib.util.spec_from_file_location(
        "cross_dispatch_recovery", path.with_name("encoder_recovery.py")
    )
    recovery = importlib.util.module_from_spec(recovery_spec)
    recovery_spec.loader.exec_module(recovery)
    namespace = dict(
        torch=torch,
        dist=SimpleNamespace(),
        zmq=zmq,
        json=json,
        math=math,
        queue=queue,
        sys=sys,
        threading=threading,
        uuid=uuid,
        hashlib=hashlib,
        timedelta=timedelta,
        request_abort=recovery.request_abort,
        PROGRESSIVE_ENCODER_VERSION="h3-clip17-progressive-v1",
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("first", ["dispatch", "producer"])
def test_admission_identity_both_orders_and_wrong_content(first):
    ns = production_class()
    session = ns["RemoteEncoderSession"].__new__(ns["RemoteEncoderSession"])
    session.admission_lock = threading.Lock()
    session.admission_digests = {}
    session._check_admission(0, b"actual source", first)
    session._check_admission(
        0, b"actual source", "producer" if first == "dispatch" else "dispatch"
    )
    assert session.admission_digests == {}
    session._check_admission(1, b"actual source", first)
    with pytest.raises(ValueError, match="identity"):
        session._check_admission(
            1, b"different source", "producer" if first == "dispatch" else "dispatch"
        )


@pytest.mark.parametrize("capacity_bytes", [256 << 20, 100])
def test_real_ipc_prefetches_next_dispatch_and_credit_bounds(
    monkeypatch, tmp_path, capacity_bytes
):
    ns = production_class()
    cls = ns["RemoteEncoderSession"]
    context = zmq.Context()
    service = context.socket(zmq.REP)
    port = service.bind_to_random_port("tcp://127.0.0.1")
    service.setsockopt(zmq.RCVTIMEO, 2000)
    writer = context.socket(zmq.PUSH)
    ipc_dir = tempfile.TemporaryDirectory(prefix="zing249-", dir="/tmp")
    endpoint = "ipc://" + str(Path(ipc_dir.name) / "source.sock")
    writer.bind(endpoint)
    received, errors = [], []

    def serve():
        frames = emitted = 0
        try:
            while True:
                parts = service.recv_multipart()
                cmd = json.loads(parts[0])
                op = cmd["op"]
                if op in ("close", "abort"):
                    service.send_json(
                        {"ok": True, "released": True, "session": cmd["session"]}
                    )
                    return
                reply = {"ok": True, "ready": False}
                if op == "append":
                    frames += cmd["shape"][1]
                    received.append((cmd, parts[1]))
                target = ns["available_latents"](frames, op == "finish")
                if op == "pop":
                    end = emitted + (1 if emitted % 5 == 0 else 2)
                    dependency = emitted // 5 * 17 + {1: 1, 3: 9, 0: 17}[end % 5]
                    reply.update(
                        start=emitted,
                        end=end,
                        dependency=dependency,
                        shape=[1, 48, end - emitted, 1, 1],
                        dtype="float32",
                    )
                    emitted = end
                reply["ready"] = emitted < target
                service.send_json(reply)
        except BaseException as error:
            errors.append(error)

    server_thread = threading.Thread(target=serve)
    server_thread.start()
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(
        torch.cuda, "Stream", lambda **kw: SimpleNamespace(synchronize=lambda: None)
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    ns["dist"] = SimpleNamespace(TCPStore=lambda *a, **kw: SimpleNamespace(port=1234))
    ns["bridge_group"] = lambda *a, **kw: None
    ns["receive_moments"] = lambda bridge, shape, dtype, device: torch.zeros(
        shape, dtype=dtype
    )
    session = cls.__new__(cls)
    session.device, session.port, session.timeout = "cpu", port, 2
    session.admission_endpoint = endpoint
    session.profile_dir, session.token = None, "private-generation"
    session.capacity_frames, session.capacity_bytes = 51, capacity_bytes
    session.admission_lock, session.admission_digests = threading.Lock(), {}
    session.prefetched_frames, session.accepted_frames, session._consumed = 0, 1, 0
    session.commands, session.results = queue.Queue(52), queue.Queue(2)
    session.permits, session.stop = threading.Semaphore(2), threading.Event()
    session.failure, session.remote_usage = None, {}
    producer = threading.Thread(target=session._produce)
    producer.start()
    try:
        for start, count in [(0, 1), (1, 8), (9, 8)]:
            data = bytes(range(12)) * count
            writer.send_multipart(
                [
                    json.dumps(
                        dict(
                            op="append",
                            source_frame_start=start,
                            num_frames=count,
                            width=2,
                            height=2,
                        )
                    ).encode(),
                    data,
                ]
            )
        deadline = time.monotonic() + 2
        while (
            session.results.qsize() < 2
            and session.failure is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        if capacity_bytes == 100:
            # Each packet fits (12 and 96 bytes), but their 108-byte total does not.
            assert isinstance(session.failure, ValueError)
            assert "over-capacity" in str(session.failure)
            assert session.prefetched_frames == 1
            assert len(received) == 1
        else:
            assert session.failure is None
            assert session.results.qsize() == 2
            assert session.accepted_frames == 1  # Scheduler still in previous dispatch.
            assert session.prefetched_frames >= 9
            assert received[1][0]["dispatch_accepted_frames"] == 1
            # CHW worker bytes are the exact transpose of the original RGB source.
            assert (
                received[1][1]
                == torch.frombuffer(bytearray(bytes(range(12)) * 8), dtype=torch.uint8)
                .reshape(8, 2, 2, 3)
                .permute(3, 0, 1, 2)
                .contiguous()
                .numpy()
                .tobytes()
            )
            time.sleep(0.05)
            assert session.results.qsize() == 2  # No third completed/in-flight result.
    finally:
        session.stop.set()
        session.permits.release()
        producer.join(3)
        server_thread.join(3)
        writer.close(0)
        service.close(0)
        context.term()
        ipc_dir.cleanup()
    assert not producer.is_alive() and not server_thread.is_alive()
    assert errors == []
    assert (session.failure is None) == (capacity_bytes != 100)


def test_state_usage_includes_input_prefetched_ahead_of_dispatch():
    ns = production_class()
    ns["EncoderStateUsage"] = lambda retained, positions: (retained, positions)
    session = ns["RemoteEncoderSession"].__new__(ns["RemoteEncoderSession"])
    session.accepted_frames, session.prefetched_frames, session._consumed = 3, 5, 1
    session.disposed = False
    session.signature = (2, 2)
    session.results = queue.Queue(2)
    session.results.put((0, torch.zeros(4, dtype=torch.float32)))
    session.remote_usage = {"retained_bytes": 100, "temporal_positions": 2}
    assert session.state_usage() == (164, 2)


def test_dispose_clears_retention_when_prefetch_is_ahead_of_dispatch():
    ns = production_class()
    ns["EncoderStateUsage"] = lambda retained, positions: (retained, positions)
    session = ns["RemoteEncoderSession"].__new__(ns["RemoteEncoderSession"])
    session.accepted_frames, session.prefetched_frames, session._consumed = 3, 5, 1
    session.disposed = False
    session.signature = (2, 2)
    session.results, session.commands = queue.Queue(2), queue.Queue()
    session.results.put((0, torch.zeros(4, dtype=torch.float32)))
    session.commands.put(object())
    session.remote_usage = {"retained_bytes": 100, "temporal_positions": 2}
    session.stop, session.permits = threading.Event(), threading.Semaphore(0)
    session.thread = None
    session.admission_digests = {0: b"pending"}
    session.failure = None
    session.dispose()
    assert session.state_usage() == (0, 0)
    snapshot = session.queue_snapshot()
    assert snapshot["prefetched_frames"] == 5
    assert snapshot["dispatch_accepted_frames"] == 3
    for key in snapshot.keys() - {"prefetched_frames", "dispatch_accepted_frames"}:
        assert snapshot[key] == 0, key


def test_segmented_staging_preserves_quantization_and_provenance_boundaries():
    path = ROOT / "utils/realtime_video.py"
    tree = ast.parse(path.read_text())
    names = {"_stage_rgb24_segments", "_tensor_batch_to_rgb24_tensor"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    from collections.abc import Sequence

    namespace = dict(torch=torch, Sequence=Sequence)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    first = torch.tensor([-0.1, 0.5, 1.1]).reshape(1, 3, 1, 1, 1)
    second = torch.tensor([0.25, 0.75, 1.0, 0.0, 0.1, 0.9]).reshape(1, 3, 2, 1, 1)
    pixels, counts = namespace["_stage_rgb24_segments"]([first, second])
    assert counts == [1, 2]
    expected = torch.cat(
        [
            namespace["_tensor_batch_to_rgb24_tensor"](first),
            namespace["_tensor_batch_to_rgb24_tensor"](second),
        ],
        dim=1,
    )
    assert torch.equal(pixels, expected)
    assert pixels[0, 0, 0, 0].tolist() == [0, 127, 255]
    # Reject byte overflow before conversion/allocation, even with virtual tensors.
    enormous = torch.empty(1, 3, 1000, 1024, 1024, device="meta")
    with pytest.raises(ValueError, match="capacity"):
        namespace["_stage_rgb24_segments"]([enormous])


def test_new_chunk_plan_splits_only_initial_three_latents():
    path = ROOT / "realtime/zing_chunk_plan.py"
    tree = ast.parse(path.read_text())
    from dataclasses import dataclass

    namespace = dict(dataclass=dataclass, PackedRoleId=int, TemporalChunkId=int)
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    plan = namespace["ChunkPlan"]([3, 2], True)
    assert [plan.chunk(i).end - plan.chunk(i).start for i in range(10)] == [
        1,
        2,
        2,
        3,
        2,
        3,
        2,
        3,
        2,
        3,
    ]


@pytest.mark.parametrize("window", [None, {"window_chunks": 6, "sink_chunks": 3}])
def test_overlay_freezes_new_chunk_and_explicit_output_switch(tmp_path, window):
    from run_encoder_overlap_matrix import overlay

    base = tmp_path / "base"
    base.mkdir()
    (base / "zing_config.json").write_text(
        json.dumps(
            dict(vae_config={}, num_frame_per_block=[1, 2, 2], split_first_latent=False)
        )
    )
    (base / "zing_realtime_profile.json").write_text(
        json.dumps(dict(source_encoder={"mode": "progressive"}))
    )
    out = tmp_path / "overlay"
    out.mkdir()
    model = overlay(
        base,
        out,
        {"mode": "overlap"},
        chunk_plan={"num_frame_per_block": [3, 2], "split_first_latent": True},
        async_output=True,
        attention_window=window,
    )
    runtime = json.loads((model / "zing_config.json").read_text())
    assert (
        runtime["num_frame_per_block"] == [3, 2]
        and runtime["split_first_latent"] is True
    )
    assert (
        json.loads((model / "zing_realtime_profile.json").read_text())[
            "async_source_output"
        ]
        is True
    )
    assert json.loads((base / "zing_config.json").read_text())[
        "num_frame_per_block"
    ] == [1, 2, 2]
    assert (
        json.loads((model / "zing_realtime_profile.json").read_text()).get(
            "source_attention_window"
        )
        == window
    )
    assert "source_attention_window" not in json.loads(
        (base / "zing_realtime_profile.json").read_text()
    )


@pytest.mark.parametrize(
    "window",
    [
        None,
        {},
        {"window_chunks": True, "sink_chunks": 0},
        {"window_chunks": 22, "sink_chunks": 3},
        {"window_chunks": 3, "sink_chunks": 3},
        {"window_chunks": 6, "sink_chunks": -1},
    ],
)
def test_overlay_rejects_invalid_window_without_creating_model(tmp_path, window):
    from run_encoder_overlap_matrix import overlay

    if window is None:
        window = {"window_chunks": 6, "sink_chunks": 3, "extra": 1}
    with pytest.raises(ValueError, match="source_attention_window"):
        overlay(tmp_path, tmp_path, None, attention_window=window)
    assert not (tmp_path / "model").exists()


@pytest.mark.parametrize("failure", [False, True])
def test_production_worker_selects_taeh3_async_and_propagates_failure(failure):
    path = ROOT / "managers/gpu_worker.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GPUWorker"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_try_materialize_raw_frames_async"
    )
    calls = []

    class Tensor:
        device = SimpleNamespace(type="cuda")
        shape = (1, 3, 2, 1, 1)

        def unsqueeze(self, n):
            return self

        def dim(self):
            return 5

    def enqueue(output, **kw):
        calls.append((output, kw))
        if failure:
            raise RuntimeError("copy failed")
        return {"path": "private-reference"}, {"format": "rgb24"}

    ns = dict(
        torch=SimpleNamespace(
            Tensor=Tensor,
            cuda=SimpleNamespace(
                memory_allocated=lambda: 80,
                memory_reserved=lambda: 100,
                max_memory_allocated=lambda: 90,
                max_memory_reserved=lambda: 120,
            ),
        ),
        OutputBatch=SimpleNamespace,
        Req=SimpleNamespace,
        LOCAL_VAE_BACKEND="local",
        is_local_endpoint=lambda endpoint: True,
        time=time,
        json=json,
        logging=logging,
        logger=SimpleNamespace(
            isEnabledFor=lambda level: False, debug=lambda *args: None
        ),
        AsyncRawRGBFrameMaterializer=lambda: SimpleNamespace(
            enqueue=enqueue, snapshot=lambda: {}
        ),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
    worker = SimpleNamespace(
        rank=0,
        server_args=SimpleNamespace(
            num_gpus=1,
            realtime_vae_backend="local",
            scheduler_endpoint="ipc:///private",
            pipeline_config=SimpleNamespace(
                video_decoder_backend="taeh3", vae_config=SimpleNamespace()
            ),
        ),
        _async_raw_frame_materializer=None,
    )
    req = SimpleNamespace(
        extra={"zing_source_command": {"async_output": True}},
        is_warmup=False,
        realtime_session_id="session",
        realtime_generation_id="generation",
        request_id="request",
        block_idx=1,
        enable_frame_interpolation=False,
        enable_upscaling=False,
    )
    batch = SimpleNamespace(output=[Tensor()], metrics=None, usage={})
    if failure:
        with pytest.raises(RuntimeError, match="copy failed"):
            ns[method.name](worker, batch, req)
    else:
        assert ns[method.name](worker, batch, req)
        assert (
            batch.output is None
            and batch.raw_frame_shared_memory_ref["path"] == "private-reference"
        )
        assert batch.usage["zing_source_memory"]["gpu_peak_reserved_bytes"] == 120
    assert calls[0][1]["wait_for_slot"] is True
    assert calls[0][1]["generation_id"] == "generation"


@pytest.mark.parametrize("debug_enabled", [False, True])
def test_sync_source_output_peak_is_recorded_after_conversion(debug_enabled):
    path = ROOT / "managers/gpu_worker.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GPUWorker"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_materialize_raw_frame_transport"
    )
    logs = []
    memory = {"peak": 60}

    def convert(*args):
        memory["peak"] = 120
        return [[b"abcd", b"efgh"]], {}

    class Tensor:
        pass

    ns = dict(
        OutputBatch=SimpleNamespace,
        Req=SimpleNamespace,
        torch=SimpleNamespace(
            Tensor=Tensor,
            cuda=SimpleNamespace(
                memory_allocated=lambda: 80,
                memory_reserved=lambda: 100,
                max_memory_allocated=lambda: 90,
                max_memory_reserved=lambda: memory["peak"],
            ),
        ),
        RAW_RGB_CONTENT_TYPE="application/octet-stream",
        json=json,
        logging=logging,
        logger=SimpleNamespace(
            isEnabledFor=lambda level: debug_enabled,
            debug=lambda fmt, value: logs.append(json.loads(value)),
        ),
        post_process_sample=lambda: None,
        build_raw_rgb_frame_batches=convert,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
    worker = SimpleNamespace(
        is_output_rank=True, rank=0, _try_materialize_raw_frames_async=lambda *a: False
    )
    req = SimpleNamespace(
        extra={"zing_source_command": {}},
        realtime_session_id="session",
        request_id="request",
        block_idx=1,
    )
    req.extra["zing_source_command"]["async_output"] = False
    batch = SimpleNamespace(realtime_latents=None, output=Tensor(), usage={})
    ns[method.name](worker, batch, req)
    assert batch.output is None
    assert batch.usage["zing_source_memory"]["gpu_peak_reserved_bytes"] == 120
    if debug_enabled:
        assert logs[0]["memory"]["mode"] == "sync"
        assert logs[0]["memory"]["payload_bytes"] == 8
    else:
        assert logs == []


@pytest.mark.parametrize(
    "latent,source,dispatched,is_future",
    [
        (0, 9, 1, False),
        (1, 9, 1, True),
        (3, 17, 9, True),
        (1, 8, 1, True),
        (5, 18, 17, True),
    ],
)
def test_cross_dispatch_cuda_attribution_uses_actual_latent_dependency(
    tmp_path, latent, source, dispatched, is_future
):
    from analyze_encoder_overlap import trace_intervals

    path = tmp_path / "trace.json"
    annotation = f"Zing.encoder_prefetch.latent_start_{latent}.source_end_{source}.dispatch_end_{dispatched}"
    path.write_text(
        json.dumps(
            dict(
                baseTimeNanoseconds=1_000_000_000,
                traceEvents=[
                    dict(
                        ph="X",
                        ts=0,
                        dur=1,
                        pid=1,
                        tid=1,
                        cat="user_annotation",
                        name="Zing.clock_anchor.unix_ns1000000000",
                    ),
                    dict(
                        ph="X",
                        ts=10,
                        dur=10,
                        pid=1,
                        tid=1,
                        cat="user_annotation",
                        name=annotation,
                    ),
                    dict(
                        ph="X",
                        ts=11,
                        dur=2,
                        pid=1,
                        tid=1,
                        cat="cuda_runtime",
                        name="cudaLaunchKernel",
                        args={"correlation": 3},
                    ),
                    dict(
                        ph="X",
                        ts=30,
                        dur=4,
                        pid=2,
                        tid=2,
                        cat="kernel",
                        name="conv",
                        args={"correlation": 3},
                    ),
                ],
            )
        )
    )
    intervals, evidence = trace_intervals(path, "Zing.encoder_cross_dispatch")
    if is_future:
        assert intervals == [(1_000_030, 1_000_034)]
        assert evidence["clock_max_error_ns"] == 0
    else:
        assert intervals is None and evidence == "Missing requested scope"


@pytest.mark.parametrize("bind_fails", [False, True])
def test_admission_initialization_owns_endpoint_and_cleans_failed_bind(
    tmp_path, bind_fails
):
    path = ROOT / "entrypoints/openai/realtime/adapters/zing_h3_source_adapter.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ZingH3SourceStreamAdapter"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_initialize_source_admission"
    )
    calls = []

    def bind(endpoint):
        calls.append(("bind", endpoint))
        if bind_fails:
            raise RuntimeError("bind failed")

    socket = SimpleNamespace(
        setsockopt=lambda *args: calls.append(args),
        bind=bind,
        close=lambda: calls.append("close"),
    )
    context = SimpleNamespace(
        socket=lambda kind: socket, term=lambda: calls.append("term")
    )
    ns = dict(
        Path=Path,
        uuid=uuid,
        tempfile=tempfile,
        require=lambda ok, message: None,
        zmq=SimpleNamespace(
            PUSH=1, LINGER=2, SNDHWM=3, asyncio=SimpleNamespace(Context=lambda: context)
        ),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
    adapter = SimpleNamespace(
        source_encoder={"remote": {"mode": "overlap", "cross_dispatch_prefetch": True}}
    )
    session = SimpleNamespace(input_temp_dir=str(tmp_path))
    if bind_fails:
        with pytest.raises(RuntimeError, match="bind failed"):
            ns[method.name](adapter, session, SimpleNamespace(pending_frames=68))
        assert calls[-2:] == ["close", "term"]
        assert adapter.admission_socket is adapter.admission_context is None
    else:
        ns[method.name](adapter, session, SimpleNamespace(pending_frames=68))
        assert adapter.admission_socket is socket
        assert adapter.source_encoder["remote"]["admission_endpoint"].startswith(
            "ipc://" + str(tmp_path)
        )
        assert (2, 0) in calls and (3, 69) in calls
