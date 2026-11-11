# SPDX-License-Identifier: Apache-2.0
"""Unit tests for pooled Encoder/Decoder sharing across multiple DiT instances (AC-1 ~ AC-6)."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch


def _condition(rows: int = 2):
    return {
        "context": torch.ones(rows, 5120, dtype=torch.bfloat16),
        "row_role": torch.arange(rows, dtype=torch.int64) % 2,
    }


# ==============================================================================
# AC-1: ConditioningService & NcclSender Multi-Bridge Pooling
# ==============================================================================


def test_nccl_sender_supports_multiple_bridges_up_to_limit(monkeypatch):
    from sglang.multimodal_gen.runtime.conditioning import nccl
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.conditioning.nccl import NcclSender

    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(
        torch.cuda,
        "Stream",
        lambda device=None: SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(nccl.dist, "TCPStore", lambda *a, **kw: object())
    monkeypatch.setattr(nccl, "bridge_group", lambda *a, **kw: object())

    sender = NcclSender("cuda:0", 10.0, max_bridges=2)

    async def run_sync(fn, *args):
        return fn(*args)

    async def run():
        b1 = "1" * 32
        b2 = "2" * 32
        b3 = "3" * 32
        await sender.open(
            {"port": 40001, "bridge_id": b1, "timeout_s": 5.0}, "t1", run_sync
        )
        await sender.open(
            {"port": 40002, "bridge_id": b2, "timeout_s": 5.0}, "t2", run_sync
        )
        assert set(sender.bridges.keys()) == {b1, b2}

        # Duplicate bridge_id must be rejected
        with pytest.raises(ConditionError) as dup_err:
            await sender.open(
                {"port": 40003, "bridge_id": b1, "timeout_s": 5.0}, "t1", run_sync
            )
        assert dup_err.value.code == "CONDITION_BUSY"

        # Exceeding max_bridges must be rejected with CONDITION_BUSY
        with pytest.raises(ConditionError) as cap_err:
            await sender.open(
                {"port": 40003, "bridge_id": b3, "timeout_s": 5.0}, "t1", run_sync
            )
        assert cap_err.value.code == "CONDITION_BUSY"

    asyncio.run(run())


def test_nccl_sender_per_bridge_acquire_release_and_tenant_isolation():
    from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
    from sglang.multimodal_gen.runtime.conditioning.nccl import NcclSender

    sender = NcclSender("cuda:0", 10.0, max_bridges=4)
    b1 = "a" * 32
    b2 = "b" * 32
    sender.register_bridge_for_test(b1, "tenant_a", group=object())
    sender.register_bridge_for_test(b2, "tenant_b", group=object())

    # Wrong tenant is rejected with CONDITION_UNAUTHORIZED
    with pytest.raises(ConditionError) as unauth_err:
        sender.acquire(b1, "tenant_b")
    assert unauth_err.value.code == "CONDITION_UNAUTHORIZED"

    # Unknown bridge is rejected with CONDITION_UNAUTHORIZED
    with pytest.raises(ConditionError) as unknown_err:
        sender.acquire("c" * 32, "tenant_a")
    assert unknown_err.value.code == "CONDITION_UNAUTHORIZED"

    # Concurrent acquire on two distinct bridges succeeds
    sender.acquire(b1, "tenant_a")
    sender.acquire(b2, "tenant_b")

    # Re-acquiring busy bridge b1 fails with CONDITION_BUSY
    with pytest.raises(ConditionError) as busy_err:
        sender.acquire(b1, "tenant_a")
    assert busy_err.value.code == "CONDITION_BUSY"

    # Releasing b1 leaves b2 busy and allows b1 to be acquired again
    sender.release(b1)
    assert not sender.bridges[b1].busy
    assert sender.bridges[b2].busy
    sender.acquire(b1, "tenant_a")
    sender.release(b1)
    sender.release(b2)


def test_conditioning_service_nccl_multi_bridge_encode_and_fault_isolation(monkeypatch):
    from sglang.multimodal_gen.runtime.conditioning import nccl
    from sglang.multimodal_gen.runtime.conditioning.contract import safe_json
    from sglang.multimodal_gen.runtime.conditioning.inputs import (
        PREPROCESS_VERSION,
        content_identity,
    )
    from sglang.multimodal_gen.runtime.conditioning.nccl import _Records
    from sglang.multimodal_gen.runtime.conditioning.resources import ResourceLimits
    from sglang.multimodal_gen.runtime.conditioning.service import ConditioningService
    from sglang.multimodal_gen.runtime.conditioning.wire import encode_request

    class FakeEncoder:
        gpu_id = 0

        def __init__(self):
            self.result = _condition(2)

        def encode_device(self, value):
            return self.result

    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(
        torch.cuda,
        "Stream",
        lambda device=None: SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(nccl.dist, "TCPStore", lambda *a, **kw: object())
    groups = {}

    def fake_bridge_group(store, rank, **kw):
        grp = SimpleNamespace(id=len(groups) + 1)
        groups[grp.id] = grp
        return grp

    monkeypatch.setattr(nccl, "bridge_group", fake_bridge_group)

    profile = SimpleNamespace(
        profile_id="a" * 64, document={"preprocess_version": PREPROCESS_VERSION}
    )
    service = ConditioningService(
        profile=profile,
        encoder=FakeEncoder(),
        tokens={"token": {"tenant": "t"}},
        limits=ResourceLimits(max_bridges=2),
        result_transport="nccl",
    )

    b1 = "1" * 32
    b2 = "2" * 32
    sent_on_bridge = []

    def fake_send(condition, bridge_id=None):
        slot = service.nccl.bridges[bridge_id]
        if bridge_id == b1 and len(sent_on_bridge) >= 2:
            raise RuntimeError("Injected NCCL send failure on bridge 1")
        sent_on_bridge.append((bridge_id, slot.group))

    monkeypatch.setattr(service.nccl, "send", fake_send)

    async def open_bridge(bridge_id: str, port: int):
        body = json.dumps(
            {"port": port, "bridge_id": bridge_id, "timeout_s": 5.0}
        ).encode()
        messages = []
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        async def send(msg):
            messages.append(msg)

        await service(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/conditioning/bridge/open",
                "client": ("127.0.0.1", 1234),
                "headers": [
                    (b"authorization", b"Bearer token"),
                    (b"content-type", b"application/octet-stream"),
                    (b"content-length", str(len(body)).encode()),
                ],
            },
            receive,
            send,
        )
        return messages

    async def encode_on_bridge(bridge_id: str, prompt: str):
        digest = content_identity(prompt, [], service.profile.profile_id)
        body = encode_request(
            prompt, [], service.profile.profile_id, input_digest=digest
        )
        messages = []
        first = True
        disconnected = asyncio.Event()

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(msg):
            messages.append(msg)

        await service(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/conditioning/encode",
                "client": ("127.0.0.1", 1234),
                "headers": [
                    (b"authorization", b"Bearer token"),
                    (b"content-type", b"application/octet-stream"),
                    (b"content-length", str(len(body)).encode()),
                    (b"x-conditioning-bridge", bridge_id.encode()),
                ],
            },
            receive,
            send,
        )
        return digest, messages

    async def chunks(values):
        for v in values:
            yield v

    async def run():
        try:
            m1 = await open_bridge(b1, 41001)
            m2 = await open_bridge(b2, 41002)
            assert m1[0]["status"] == 200 and safe_json(m1[1]["body"]) == {
                "bridge_id": b1
            }
            assert m2[0]["status"] == 200 and safe_json(m2[1]["body"]) == {
                "bridge_id": b2
            }

            # Both bridges encode and receive their own bridge_id in metadata
            d1, msg1 = await encode_on_bridge(b1, "prompt-1")
            d2, msg2 = await encode_on_bridge(b2, "prompt-2")
            rec1 = _Records(chunks([b"".join(m.get("body", b"") for m in msg1)]))
            rec2 = _Records(chunks([b"".join(m.get("body", b"") for m in msg2)]))
            meta1 = await rec1.read()
            meta2 = await rec2.read()
            assert meta1["bridge_id"] == b1 and meta1["input_digest"] == d1
            assert meta2["bridge_id"] == b2 and meta2["input_digest"] == d2

            # Now inject failure on bridge 1; bridge 1 is poisoned, but bridge 2 still works!
            with pytest.raises(RuntimeError, match="Injected NCCL send failure"):
                await encode_on_bridge(b1, "prompt-fail")
            assert service.nccl.bridges[b1].failed is True
            assert service.nccl.bridges[b2].failed is False

            _, msg1_after = await encode_on_bridge(b1, "prompt-after-fail")
            assert msg1_after[0]["status"] == 503

            d2_after, msg2_after = await encode_on_bridge(b2, "prompt-still-healthy")
            assert msg2_after[0]["status"] == 200
            rec2_after = _Records(
                chunks([b"".join(m.get("body", b"") for m in msg2_after)])
            )
            assert (await rec2_after.read())["bridge_id"] == b2
            assert await rec2_after.read() == {"done": d2_after}
        finally:
            await service.close()

    asyncio.run(run())


# ==============================================================================
# AC-2: decoder_worker.py Multi-Session (--max-sessions M) & Persistent Bridges
# ==============================================================================


class _LoopExit(BaseException):
    pass


def _run_decoder_worker_script(commands, *, max_sessions: int = 2, service_cls=None):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        encoder_control,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_worker import (
        _output_identity,
    )

    path = (
        Path(__file__).resolve().parents[2]
        / "runtime/pipelines_core/stages/model_specific_stages/zing_h3/decoder_worker.py"
    )
    tree = ast.parse(path.read_text())
    main = next(
        x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "main"
    )
    script = iter(commands)
    channels_by_session = {}
    last_open_session = [None]
    tcp_store_ports = []
    services_created = []

    class FakeService:
        def __init__(self, *args, **kwargs):
            self.width = self.height = 8
            self.valid_frame_end = self.submitted_frames = self.buffered_latents = 0
            self.disposed = False
            self.drained = False
            services_created.append(self)

        def check(self):
            pass

        def append(
            self, latent, *, latent_start, valid_frame_end, source_dependency_end
        ):
            self.valid_frame_end = valid_frame_end

        def submit(self, frames, reference):
            self.submitted_frames += frames

        def finish(self, total_frames):
            if total_frames != self.valid_frame_end:
                raise ValueError("Frame mismatch")

        def drain(self, timeout):
            self.drained = True

        def dispose(self):
            self.disposed = True

    def take():
        try:
            command = dict(next(script))
        except StopIteration:
            raise _LoopExit
        if command.get("op") == "open":
            last_open_session[0] = command.get("session")
        ch = command.get("channel")
        if isinstance(ch, str) and ch.startswith("$channel:"):
            sess_key = ch.split(":", 1)[1]
            command["channel"] = channels_by_session[sess_key]
        return command

    replies = []

    def send(reply):
        if isinstance(reply, dict) and "channel" in reply and last_open_session[0]:
            channels_by_session[last_open_session[0]] = reply["channel"]
        replies.append(reply)

    socket = SimpleNamespace(
        selected=0,
        setsockopt=lambda *a: None,
        bind=lambda *a: None,
        poll=lambda *_: True,
        recv_multipart=lambda: [json.dumps(take()).encode()],
        send_json=send,
        close=lambda: None,
    )
    context = SimpleNamespace(socket=lambda *a: socket, term=lambda: None)
    args = SimpleNamespace(
        model=Path("model"),
        port=31003,
        decoder_config="{}",
        timeout_seconds=120,
        startup_timeout_seconds=600,
        max_sessions=max_sessions,
        worker_id="decoder-component-test",
        static_connection=True,
        service_count=1,
        http_port=None,
        http_host="127.0.0.1",
        output_port=None,
    )

    def all_gather_object(values, value, group=None):
        values[:] = [value]

    dist = SimpleNamespace(
        init_process_group=lambda *a, **kw: None,
        get_rank=lambda: 0,
        get_world_size=lambda *a, **kw: 1,
        new_group=lambda **kw: object(),
        broadcast_object_list=lambda *a, **kw: None,
        all_gather_object=all_gather_object,
        TCPStore=lambda host, port, *a, **kw: tcp_store_ports.append(port) or object(),
    )
    stub_cuda = SimpleNamespace(
        set_device=lambda n: None,
        current_device=lambda: 0,
        memory_allocated=lambda: 0,
        memory_reserved=lambda: 0,
        max_memory_allocated=lambda: 0,
    )
    namespace = dict(
        __doc__="test",
        __name__="sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_worker",
        __package__="sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3",
        diagnostic_print=lambda *a, **kw: None,
        argparse=SimpleNamespace(
            ArgumentParser=lambda **kw: SimpleNamespace(
                add_argument=lambda *a, **k: None, parse_args=lambda: args
            )
        ),
        Path=Path,
        json=json,
        logging=logging,
        time=SimpleNamespace(time_ns=lambda: 0),
        timedelta=timedelta,
        os=SimpleNamespace(environ={"LOCAL_RANK": "0"}),
        torch=SimpleNamespace(
            device=torch.device,
            float32=torch.float32,
            empty=torch.empty,
            cuda=stub_cuda,
        ),
        zmq=SimpleNamespace(Context=lambda: context, REP=1, LINGER=2, SNDTIMEO=3),
        dist=dist,
        initialize_runtime_imports=lambda: (
            SimpleNamespace(from_kwargs=lambda **kw: None),
            lambda server_args: None,
        ),
        load_decoder=lambda config: None,
        bridge_group=lambda *a, **kw: object(),
        receive_latent=lambda group, shape, dtype, device: torch.zeros(
            shape, dtype=dtype
        ),
        _DecodeService=service_cls or FakeService,
        _OutputPublisher=lambda **kw: SimpleNamespace(
            reserve=lambda **kw: dict(frames=kw["frames"], **kw["identity"]),
            close=lambda: None,
        ),
        _output_identity=_output_identity,
        remote_decode_progressive=lambda: False,
        service_ports=importlib.import_module(
            "sglang.multimodal_gen.runtime.realtime.worker_services"
        ).service_ports,
        EncoderCommandControl=encoder_control.EncoderCommandControl,
        EncoderControlError=encoder_control.EncoderControlError,
    )
    exec(
        compile(ast.Module(body=[main], type_ignores=[]), str(path), "exec"), namespace
    )
    with (
        # This harness replaces model loading with None and tests transport
        # ownership. The real startup decode is covered by codec warmup tests.
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codec_warmup.warmup_decoder"
        ) as warmup,
        patch(
            "sglang.multimodal_gen.runtime.realtime.worker_services.ServiceCommandSockets",
            lambda *args, **kwargs: socket,
        ),
        pytest.raises(_LoopExit),
    ):
        namespace["main"]()
    warmup.assert_called_once()
    assert warmup.call_args.args == (None,)
    assert warmup.call_args.kwargs["group"] is None
    assert callable(warmup.call_args.kwargs["agree"])
    return replies, tcp_store_ports, services_created


def test_decoder_worker_multi_session_isolation_and_capacity_limit():
    commands = [
        {"op": "ping"},
        {
            "op": "open",
            "session": "dit-1",
            "bridge_port": 41001,
            "capacity_latents": 10,
        },
        {"op": "ping"},
        {
            "op": "open",
            "session": "dit-2",
            "bridge_port": 41002,
            "capacity_latents": 10,
        },
        {"op": "ping"},
        {
            "op": "open",
            "session": "dit-3",
            "bridge_port": 41003,
            "capacity_latents": 10,
        },
        {
            "op": "append",
            "channel": "$channel:dit-1",
            "shape": [1, 24, 1, 2, 2],
            "dtype": "float32",
            "latent_start": 0,
            "valid_frame_end": 16,
            "source_dependency_end": 16,
        },
        {
            "op": "append",
            "channel": "$channel:dit-2",
            "shape": [1, 24, 2, 2, 2],
            "dtype": "float32",
            "latent_start": 0,
            "valid_frame_end": 32,
            "source_dependency_end": 32,
        },
        {
            "op": "output",
            "channel": "$channel:dit-1",
            "frames": 16,
            "generation_id": "gen-1",
            "request_id": "req-1",
            "chunk_index": 0,
        },
        {
            "op": "output",
            "channel": "$channel:dit-2",
            "frames": 32,
            "generation_id": "gen-2",
            "request_id": "req-2",
            "chunk_index": 0,
        },
        # Abort dit-1; dit-2 must stay alive and unaffected
        {"op": "abort", "session": "dit-1"},
        {"op": "ping"},
        {"op": "finish", "channel": "$channel:dit-2", "total_frames": 32},
        {"op": "close", "channel": "$channel:dit-2"},
        {"op": "ping"},
    ]
    replies, tcp_ports, services = _run_decoder_worker_script(commands, max_sessions=2)

    assert replies[0] == {
        "ok": True,
        "world_size": 1,
        "busy": False,
        "active_sessions": 0,
        "max_sessions": 2,
        "active_bridges": 0,
        "bridge_ids": [],
        "draining": False,
        "restart_required": False,
    }
    assert replies[1]["ok"] is True
    assert replies[2] == {
        "ok": True,
        "world_size": 1,
        "busy": False,
        "active_sessions": 1,
        "max_sessions": 2,
        "active_bridges": 0,
        "bridge_ids": [],
        "draining": False,
        "restart_required": False,
    }
    assert replies[3]["ok"] is True and replies[3]["channel"] != replies[1]["channel"]
    assert replies[4] == {
        "ok": True,
        "world_size": 1,
        "busy": True,
        "active_sessions": 2,
        "max_sessions": 2,
        "active_bridges": 0,
        "bridge_ids": [],
        "draining": False,
        "restart_required": False,
    }
    # 3rd session rejected with DECODER_BUSY
    assert replies[5]["ok"] is False and replies[5]["code"] == "DECODER_BUSY"
    # Outputs isolated per session
    assert (
        replies[8]["reference"]["session_id"] == "dit-1"
        and replies[8]["reference"]["frames"] == 16
    )
    assert (
        replies[9]["reference"]["session_id"] == "dit-2"
        and replies[9]["reference"]["frames"] == 32
    )
    # Abort dit-1 frees 1 slot while dit-2 continues
    assert replies[10] == {"ok": True, "session": "dit-1", "released": True}
    assert replies[11] == {
        "ok": True,
        "world_size": 1,
        "busy": False,
        "active_sessions": 1,
        "max_sessions": 2,
        "active_bridges": 0,
        "bridge_ids": [],
        "draining": False,
        "restart_required": False,
    }
    assert replies[12]["ok"] is True
    assert replies[13]["ok"] is True
    assert replies[14] == {
        "ok": True,
        "world_size": 1,
        "busy": False,
        "active_sessions": 0,
        "max_sessions": 2,
        "active_bridges": 0,
        "bridge_ids": [],
        "draining": False,
        "restart_required": False,
    }
    assert len(services) == 2 and all(s.disposed for s in services)


def test_decoder_worker_persistent_bridge_open_reused_across_sessions():
    commands = [
        {"op": "bridge_open", "bridge_id": "bridge-dit-1", "bridge_port": 41010},
        {
            "op": "open",
            "session": "s1",
            "bridge_id": "bridge-dit-1",
            "capacity_latents": 10,
        },
        {"op": "close", "channel": "$channel:s1"},
        {
            "op": "open",
            "session": "s2",
            "bridge_id": "bridge-dit-1",
            "capacity_latents": 10,
        },
        {"op": "close", "channel": "$channel:s2"},
    ]
    replies, tcp_ports, _ = _run_decoder_worker_script(commands, max_sessions=2)
    assert replies[0] == {"ok": True, "bridge_id": "bridge-dit-1"}
    assert replies[1]["ok"] is True and replies[2]["ok"] is True
    assert replies[3]["ok"] is True and replies[4]["ok"] is True
    # TCPStore was only created once during bridge_open!
    assert tcp_ports == [41010]


# ==============================================================================
# AC-3: encoder_worker.py Multi-Session (--max-sessions M) & Persistent Bridges
# ==============================================================================


def _run_encoder_worker_script(
    monkeypatch,
    commands,
    *,
    max_sessions: int = 2,
    rank: int = 0,
    extra_dist: dict | None = None,
):
    from test_zing_h3_encoder_abort_recovery import recovery

    path = (
        Path(__file__).resolve().parents[2]
        / "runtime/pipelines_core/stages/model_specific_stages/zing_h3/encoder_worker.py"
    )
    tree = ast.parse(path.read_text())
    main = next(
        x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "main"
    )
    script = iter(commands)
    sequence = 0

    class StopLoop(BaseException):
        pass

    def take():
        try:
            item = next(script)
        except StopIteration:
            raise StopLoop
        if isinstance(item, tuple):
            return item[0], item[1]
        return item, None

    replies, tcp_ports, created_sessions = [], [], []

    def recv_multipart():
        cmd, payload = take()
        parts = [json.dumps(cmd).encode()]
        if payload is not None:
            parts.append(payload)
        return parts

    socket = SimpleNamespace(
        setsockopt=lambda *a: None,
        bind=lambda *a: None,
        poll=lambda *_: True,
        recv_multipart=recv_multipart,
        send_json=lambda value: replies.append(value),
        close=lambda: None,
    )
    context = SimpleNamespace(socket=lambda *a: socket, term=lambda: None)

    class FakeEncoderSession:
        def __init__(self, codecs, capacity_frames, cache):
            self.cache = cache
            self.capacity_frames = capacity_frames
            self.buffered_frames = 0
            self.emitted_latents = 0
            self.ended = False
            self.closed = False
            self.appended = []
            created_sessions.append(self)

        @property
        def ready(self):
            return self.buffered_frames > 0

        def append(self, pixels, *, frame_start):
            self.appended.append((frame_start, int(pixels.shape[1])))
            self.buffered_frames += int(pixels.shape[1])

        def finish(self):
            self.ended = True

        def pop(self):
            if not self.ready:
                return None
            frames = self.buffered_frames
            self.buffered_frames = 0
            start = self.emitted_latents
            self.emitted_latents += 1
            return SimpleNamespace(
                values=torch.zeros(1, 48, 1, 2, 2, dtype=torch.float32),
                start=start,
                end=self.emitted_latents,
                source_dependency_end=frames,
            )

        def dispose(self):
            assert not self.closed
            self.closed = True

        def state_usage(self):
            return SimpleNamespace(
                retained_bytes=self.buffered_frames * 12,
                temporal_positions=self.buffered_frames,
            )

    module_name = "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.parallel_encoder"
    fake = ModuleType(module_name)
    fake.create_encoder_cache = lambda *a, **kw: SimpleNamespace(
        last_execution={"ok": True}
    )
    monkeypatch.setitem(sys.modules, module_name, fake)

    def broadcast_obj(values, src, **kwargs):
        nonlocal sequence
        if rank:
            sequence += 1
            cmd, _ = take()
            values[0] = ("COMMAND", sequence, cmd)

    def gather(output, value, **kwargs):
        output[:] = [value, value]

    class Parser:
        def __init__(self, **kw):
            pass

        def add_argument(self, *a, **kw):
            pass

        def parse_args(self):
            return SimpleNamespace(
                model=Path("model"),
                port=31001,
                timeout_seconds=120,
                startup_timeout_seconds=600,
                max_sessions=max_sessions,
                host="127.0.0.1",
                peer_host="127.0.0.1",
                auth_token_file=None,
                node_binding_file=None,
            )

    dist_kwargs = dict(
        init_process_group=lambda *a, **kw: None,
        get_rank=lambda: rank,
        new_group=lambda **kw: object(),
        barrier=lambda: None,
        broadcast=lambda *a, **kw: None,
        broadcast_object_list=broadcast_obj,
        get_world_size=lambda *a: 2,
        all_gather_object=gather,
        group=SimpleNamespace(WORLD=object()),
        destroy_process_group=lambda: None,
        TCPStore=lambda host, port, *a, **kw: tcp_ports.append(port) or object(),
    )
    if extra_dist:
        dist_kwargs.update(extra_dist)
    dist = SimpleNamespace(**dist_kwargs)
    control_spec = importlib.util.spec_from_file_location(
        "zing_encoder_control_cpu_pool", path.with_name("encoder_control.py")
    )
    control = importlib.util.module_from_spec(control_spec)
    control_spec.loader.exec_module(control)

    import contextlib

    namespace = dict(
        EncoderPeer=importlib.import_module(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_peer"
        ).EncoderPeer,
        EncoderCommandControl=control.EncoderCommandControl,
        EncoderControlError=control.EncoderControlError,
        logging=logging,
        diagnostic_print=lambda *a, **kw: None,
        prepare_static_gc=lambda *a, **kw: None,
        abort_owner=recovery.abort_owner,
        math=__import__("math"),
        trace_for_ids=lambda *a, **kw: None,
        stage_span=lambda *a, **kw: contextlib.nullcontext(None),
        handoff=lambda *a, **kw: None,
        __doc__="test",
        argparse=SimpleNamespace(ArgumentParser=Parser),
        Path=Path,
        initialize_runtime_imports=lambda: (
            SimpleNamespace(from_kwargs=lambda **kw: None),
            lambda a: None,
        ),
        torch=SimpleNamespace(
            uint8=torch.uint8,
            empty=lambda shape, dtype=torch.uint8, device=None: torch.empty(
                shape, dtype=dtype
            ),
            frombuffer=torch.frombuffer,
            profiler=SimpleNamespace(
                record_function=lambda *a, **kw: contextlib.nullcontext()
            ),
            cuda=SimpleNamespace(
                set_device=lambda n: None,
                current_stream=lambda: object(),
                Event=lambda: SimpleNamespace(
                    record=lambda *a: None, synchronize=lambda: None
                ),
                memory_allocated=lambda: 0,
                memory_reserved=lambda: 0,
                max_memory_allocated=lambda: 0,
                max_memory_reserved=lambda: 0,
            ),
        ),
        os=SimpleNamespace(environ={"LOCAL_RANK": str(rank)}),
        dist=dist,
        timedelta=timedelta,
        load_encoder=lambda root: SimpleNamespace(
            modules={"video_vae": object()},
            warmup_source_vae=lambda: None,
            runtime={
                "vae_config": {"encode_mode": "performance", "encode_dtype": "bfloat16"}
            },
        ),
        zmq=SimpleNamespace(Context=lambda: context, REP=1, LINGER=2, SNDTIMEO=3),
        json=json,
        time=SimpleNamespace(time_ns=lambda: 0),
        H3ProgressiveEncoderSession=FakeEncoderSession,
        bridge_group=lambda *a, **kw: SimpleNamespace(
            send=lambda tensors, dst, tag: SimpleNamespace(wait=lambda: None)
        ),
        print=lambda *a, **kw: None,
    )
    exec(
        compile(ast.Module(body=[main], type_ignores=[]), str(path), "exec"), namespace
    )
    with pytest.raises(StopLoop):
        namespace["main"]()
    return replies, tcp_ports, created_sessions


@pytest.mark.parametrize("rank", [0, 1])
def test_encoder_worker_multi_session_isolation_and_capacity_limit(monkeypatch, rank):
    payload_1 = b"x" * (3 * 2 * 2 * 2)
    payload_2 = b"y" * (3 * 4 * 2 * 2)
    commands = [
        {"op": "ping"},
        {"op": "open", "session": "dit-1", "bridge_port": 42001, "capacity_frames": 51},
        {"op": "ping"},
        {"op": "open", "session": "dit-2", "bridge_port": 42002, "capacity_frames": 51},
        {"op": "ping"},
        {"op": "open", "session": "dit-3", "bridge_port": 42003, "capacity_frames": 51},
        (
            {
                "op": "append",
                "session": "dit-1",
                "shape": [3, 2, 2, 2],
                "frame_start": 0,
                "capacity_bytes": 1024,
            },
            payload_1,
        ),
        (
            {
                "op": "append",
                "session": "dit-2",
                "shape": [3, 4, 2, 2],
                "frame_start": 0,
                "capacity_bytes": 1024,
            },
            payload_2,
        ),
        {"op": "pop", "session": "dit-1"},
        {"op": "pop", "session": "dit-2"},
        # Abort dit-1; dit-2 must remain valid and unaffected
        {"op": "abort", "session": "dit-1"},
        {"op": "ping"},
        {"op": "finish", "session": "dit-2"},
        {"op": "close", "session": "dit-2"},
        {"op": "ping"},
    ]
    replies, tcp_ports, sessions = _run_encoder_worker_script(
        monkeypatch, commands, max_sessions=2, rank=rank
    )
    assert len(sessions) == 2 and all(s.closed for s in sessions)
    assert sessions[0].appended == [(0, 2)]
    assert sessions[1].appended == [(0, 4)]
    if rank == 0:
        assert (
            replies[0]["busy"] is False
            and replies[0]["active_sessions"] == 0
            and replies[0]["max_sessions"] == 2
        )
        assert replies[1]["ok"] is True
        assert (
            replies[2]["busy"] is False
            and replies[2]["active_sessions"] == 1
            and replies[2]["max_sessions"] == 2
        )
        assert replies[3]["ok"] is True
        assert (
            replies[4]["busy"] is True
            and replies[4]["active_sessions"] == 2
            and replies[4]["max_sessions"] == 2
        )
        assert replies[5]["ok"] is False and replies[5]["code"] == "ENCODER_BUSY"
        assert replies[8]["ok"] is True and replies[8]["dependency"] == 2
        assert replies[9]["ok"] is True and replies[9]["dependency"] == 4
        assert replies[10]["ok"] is True and replies[10]["released"] is True
        assert replies[11]["busy"] is False and replies[11]["active_sessions"] == 1
        assert replies[12]["ok"] is True
        assert replies[13]["ok"] is True
        assert replies[14]["busy"] is False and replies[14]["active_sessions"] == 0


def test_encoder_worker_persistent_bridge_open_reused_across_sessions(monkeypatch):
    commands = [
        {"op": "bridge_open", "bridge_id": "enc-bridge-1", "bridge_port": 42010},
        {
            "op": "open",
            "session": "s1",
            "bridge_id": "enc-bridge-1",
            "capacity_frames": 51,
        },
        {"op": "close", "session": "s1"},
        {
            "op": "open",
            "session": "s2",
            "bridge_id": "enc-bridge-1",
            "capacity_frames": 51,
        },
        {"op": "close", "session": "s2"},
    ]
    replies, tcp_ports, sessions = _run_encoder_worker_script(
        monkeypatch, commands, max_sessions=2, rank=0
    )
    assert replies[0] == {"ok": True, "bridge_id": "enc-bridge-1"}
    assert all(r["ok"] is True for r in replies[1:])
    assert tcp_ports == [42010]
    assert len(sessions) == 2 and all(s.closed for s in sessions)


# ==============================================================================
# AC-4: RemoteLynnDecoder & RemoteEncoderSession Pool Routing, Failover & Bridge Reuse
# ==============================================================================


def test_remote_decoder_pool_routing_and_busy_failover_and_bridge_reuse(monkeypatch):
    from sglang.multimodal_gen.runtime.distributed import parallel_state
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        remote_decoder,
    )

    monkeypatch.setattr(
        parallel_state,
        "get_world_group",
        lambda: SimpleNamespace(
            rank_in_group=0, ranks=[0], world_size=1, cpu_group=None, device_group=None
        ),
    )
    remote_decoder._PERSISTENT_DECODER_BRIDGES.clear()

    tcp_stores = []
    monkeypatch.setattr(
        remote_decoder.dist,
        "TCPStore",
        lambda *a, **kw: tcp_stores.append(1) or SimpleNamespace(port=45001),
    )
    monkeypatch.setattr(remote_decoder, "bridge_group", lambda *a, **kw: "bridge_grp")

    # Track ZMQ operations per port
    port_Scripts = {
        31011: [
            # First session: ping shows 1 active session, open returns DECODER_BUSY
            {"ok": True, "busy": False, "active_sessions": 1, "max_sessions": 2},
            {"ok": True, "bridge_id": "b1"},
            {"ok": False, "code": "DECODER_BUSY"},
        ],
        31012: [
            # First session: ping shows 0 active sessions -> preferred!
            {"ok": True, "busy": False, "active_sessions": 0, "max_sessions": 2},
            {"ok": True, "bridge_id": "b2"},
            {"ok": True, "channel": 11},
            {"ok": True, "released": True},
            # Second session on 31012: reuses bridge without bridge_open!
            {"ok": True, "busy": False, "active_sessions": 0, "max_sessions": 2},
            {"ok": True, "channel": 12},
            {"ok": True, "released": True},
        ],
    }
    sent_by_port = {31011: [], 31012: []}

    class FakePortSocket:
        def __init__(self):
            self.port = None

        def setsockopt(self, *a):
            pass

        def connect(self, endpoint):
            self.port = int(endpoint.rsplit(":", 1)[1])

        def send_json(self, msg):
            sent_by_port[self.port].append(msg)

        def recv_json(self):
            return port_Scripts[self.port].pop(0)

        def close(self):
            pass

    monkeypatch.setattr(
        remote_decoder.zmq,
        "Context",
        lambda: SimpleNamespace(socket=lambda *a: FakePortSocket(), term=lambda: None),
    )

    codecs = SimpleNamespace(
        modules={
            "lynn_decoder": SimpleNamespace(
                protocol=SimpleNamespace(frame_count=lambda n: 4 * n)
            )
        },
        decoder=SimpleNamespace(module_key="lynn_decoder"),
    )

    # 1. Load-aware routing selects port 31012 (0 active vs 1 active on 31011)
    dec1 = remote_decoder.RemoteLynnDecoder(
        codecs,
        capacity_latents=10,
        provenance_for=lambda *a: None,
        config={"ports": [31011, 31012], "session_id": "sess-1"},
    )
    assert dec1.port == 31012
    assert dec1.channel == 11
    assert len(tcp_stores) == 1
    dec1.dispose()

    # 2. Second session on port 31012 reuses the persistent bridge without calling TCPStore/bridge_open again!
    dec2 = remote_decoder.RemoteLynnDecoder(
        codecs,
        capacity_latents=10,
        provenance_for=lambda *a: None,
        config={"port": 31012, "session_id": "sess-2"},
    )
    assert dec2.port == 31012
    assert dec2.channel == 12
    assert len(tcp_stores) == 1  # Still 1!
    open_msgs_31012 = [m for m in sent_by_port[31012] if m["op"] == "open"]
    assert len(open_msgs_31012) == 2
    assert "bridge_id" in open_msgs_31012[1] and "bridge_port" not in open_msgs_31012[1]
    dec2.dispose()

    # 3. Failover when first port returns DECODER_BUSY on open
    remote_decoder._PERSISTENT_DECODER_BRIDGES.clear()
    port_Scripts[31011] = [
        {"ok": True, "busy": False, "active_sessions": 0, "max_sessions": 2},
        {"ok": True, "bridge_id": "b1"},
        {"ok": False, "code": "DECODER_BUSY"},
    ]
    port_Scripts[31012] = [
        {"ok": True, "busy": False, "active_sessions": 0, "max_sessions": 2},
        {"ok": True, "bridge_id": "b2"},
        {"ok": True, "channel": 99},
        {"ok": True, "released": True},
    ]
    dec3 = remote_decoder.RemoteLynnDecoder(
        codecs,
        capacity_latents=10,
        provenance_for=lambda *a: None,
        config={"ports": [31011, 31012], "session_id": "sess-3"},
    )
    assert dec3.port == 31012
    assert dec3.channel == 99
    assert dec3.failure is None
    dec3.dispose()


def test_remote_encoder_pool_routing_and_busy_failover_and_bridge_reuse(monkeypatch):
    import contextlib

    from sglang.multimodal_gen.runtime.distributed import parallel_state
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        remote_encoder,
    )

    monkeypatch.setattr(
        parallel_state,
        "get_world_group",
        lambda: SimpleNamespace(
            rank_in_group=0, ranks=[0], world_size=1, cpu_group=None, device_group=None
        ),
    )
    remote_encoder._PERSISTENT_ENCODER_BRIDGES.clear()

    tcp_stores = []
    monkeypatch.setattr(
        remote_encoder.dist,
        "TCPStore",
        lambda *a, **kw: tcp_stores.append(1) or SimpleNamespace(port=46001),
    )
    monkeypatch.setattr(
        remote_encoder, "bridge_group", lambda *a, **kw: "enc_bridge_grp"
    )
    monkeypatch.setattr(
        remote_encoder.torch,
        "cuda",
        SimpleNamespace(
            set_device=lambda x: None,
            Stream=lambda **kw: SimpleNamespace(synchronize=lambda: None),
            stream=lambda x: contextlib.nullcontext(),
        ),
    )

    port_scripts = {
        31021: [
            # Ping says 1 active session; 31022 has 0 -> selects 31022
            {"ok": True, "busy": False, "active_sessions": 1, "max_sessions": 2},
        ],
        31022: [
            {"ok": True, "busy": False, "active_sessions": 0, "max_sessions": 2},
            {"ok": True, "bridge_id": "eb2"},
            {"ok": True},  # open
            {"ok": True},  # close
            # Second session on 31022
            {"ok": True},  # open (reuses bridge!)
            {"ok": True},  # close
        ],
    }
    sent_by_port = {31021: [], 31022: []}

    class FakeEncSocket:
        def __init__(self):
            self.port = None

        def setsockopt(self, *a):
            pass

        def connect(self, endpoint):
            self.port = int(endpoint.rsplit(":", 1)[1])

        def send_json(self, msg):
            sent_by_port[self.port].append(msg)

        def send_multipart(self, parts):
            sent_by_port[self.port].append(json.loads(parts[0]))

        def recv_json(self):
            return port_scripts[self.port].pop(0)

        def close(self):
            pass

    monkeypatch.setattr(
        remote_encoder.zmq,
        "Context",
        lambda: SimpleNamespace(socket=lambda *a: FakeEncSocket(), term=lambda: None),
    )

    codecs = SimpleNamespace(
        modules={"video_vae": SimpleNamespace(encoder=torch.nn.Linear(1, 1))}
    )

    # 1. Pool routing selects least-loaded port 31022
    s1 = remote_encoder.RemoteEncoderSession(
        codecs,
        capacity_frames=34,
        config={"ports": [31021, 31022], "mode": "overlap", "timeout_seconds": 2},
    )
    s1.dispose()
    assert s1.port == 31022
    assert len(tcp_stores) == 1

    # 2. Second session on 31022 reuses persistent bridge without TCPStore
    s2 = remote_encoder.RemoteEncoderSession(
        codecs,
        capacity_frames=34,
        config={"port": 31022, "mode": "overlap", "timeout_seconds": 2},
    )
    s2.dispose()
    assert len(tcp_stores) == 1
    open_msgs = [m for m in sent_by_port[31022] if m["op"] == "open"]
    assert len(open_msgs) == 2
    assert "bridge_id" in open_msgs[1] and "bridge_port" not in open_msgs[1]


# ==============================================================================
# AC-5 & AC-6: Single-GPU (SP=1), Arbitrary SP=k, and Multi-Instance Pool Launch
# ==============================================================================


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_single_gpu_and_arbitrary_sp_collectives_in_remote_clients_and_conditioning(
    monkeypatch, world_size
):
    from sglang.multimodal_gen.runtime.conditioning.worker import _cpu_timeout
    from sglang.multimodal_gen.runtime.distributed import parallel_state
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.remote_decoder import (
        RemoteLynnDecoder,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.remote_encoder import (
        RemoteEncoderSession,
    )

    group = SimpleNamespace(
        unique_name="world_group",
        rank_in_group=0,
        ranks=list(range(world_size)),
        world_size=world_size,
        cpu_group=object(),
        device_group=object(),
    )
    if world_size == 1:
        assert _cpu_timeout(group) is None

    monkeypatch.setattr(parallel_state, "get_world_group", lambda: group)

    obj_broadcasts = []
    tensor_broadcasts = []
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda values, **kw: obj_broadcasts.append(list(values)),
    )
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda tensor, **kw: tensor_broadcasts.append(tuple(tensor.shape)),
    )

    # Test RemoteEncoderSession.pop()
    monkeypatch.setattr(RemoteEncoderSession, "_produce", lambda self: None)
    codecs_enc = SimpleNamespace(
        modules={"video_vae": SimpleNamespace(encoder=torch.nn.Linear(1, 1))}
    )
    enc = RemoteEncoderSession(
        codecs_enc,
        capacity_frames=34,
        config={"port": 31001, "mode": "serial", "timeout_seconds": 1},
    )
    enc.accepted_frames = 17
    enc.results.put(
        (
            {
                "start": 0,
                "end": 1,
                "dependency": 1,
                "shape": [1, 48, 1, 1, 1],
                "dtype": "float32",
            },
            torch.zeros(1, 48, 1, 1, 1),
        )
    )
    out = enc.pop()
    assert out.start == 0 and out.end == 1
    if world_size == 1:
        assert tensor_broadcasts == []
        assert obj_broadcasts == []
    else:
        assert len(tensor_broadcasts) == 1
        assert len(obj_broadcasts) == 1

    # Test RemoteLynnDecoder.pop()
    obj_broadcasts.clear()
    monkeypatch.setattr(RemoteLynnDecoder, "_connect", lambda self: None)
    codecs_dec = SimpleNamespace(
        modules={
            "lynn_decoder": SimpleNamespace(
                protocol=SimpleNamespace(frame_count=lambda n: 4 * n)
            )
        },
        decoder=SimpleNamespace(module_key="lynn_decoder"),
    )
    dec = RemoteLynnDecoder(
        codecs_dec,
        capacity_latents=10,
        provenance_for=lambda pixels, dep: SimpleNamespace(
            pixels=pixels, source_dependency_end=dep
        ),
        config={"port": 31002, "session_id": "s1"},
    )
    dec.channel = 1
    dec.accepted_latents = 1
    dec.valid_frame_end = 4
    dec.dependency = 4
    dec.socket = SimpleNamespace(
        send_json=lambda *a: None,
        recv_json=lambda: {
            "ok": True,
            "reference": {"num_bytes": 12},
            "metadata": {"width": 2},
        },
        close=lambda: None,
    )
    slice_out = dec.pop()
    assert slice_out.frames == 4
    if world_size == 1:
        assert obj_broadcasts == []
    else:
        assert len(obj_broadcasts) == 1


def test_role_launcher_and_policies_with_multi_instance_ports_and_max_sessions(
    monkeypatch,
):
    from sglang.multimodal_gen.runtime.disaggregation import role_launcher
    from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

    server_args = SimpleNamespace(
        model_path="/tmp/model",
        encoder_gpus="0",
        decoder_gpus="3",
        disagg_max_sessions=3,
        pipeline_config=SimpleNamespace(decoder_config={"type": "lynn_streaming"}),
    )
    enc_cmd = role_launcher.build_role_command(
        server_args, RoleType.ENCODER, 31003, max_sessions=3
    )
    dec_cmd = role_launcher.build_role_command(
        server_args, RoleType.DECODER, 31004, max_sessions=3
    )
    assert (
        "--max-sessions" in enc_cmd
        and enc_cmd[enc_cmd.index("--max-sessions") + 1] == "3"
    )
    assert (
        "--max-sessions" in dec_cmd
        and dec_cmd[dec_cmd.index("--max-sessions") + 1] == "3"
    )

    profile = role_launcher._with_disagg_role_policies(
        {},
        encoder_port=31003,
        encoder_ports=[31003, 31005],
        decoder_port=31004,
        decoder_ports=[31004, 31006],
    )
    assert profile["source_encoder"]["remote"]["port"] == 31003
    assert profile["source_encoder"]["remote"]["ports"] == [31003, 31005]
    assert profile["source_decoder"]["remote"]["port"] == 31004
    assert profile["source_decoder"]["remote"]["ports"] == [31004, 31006]


@pytest.mark.skip(reason="Legacy shared-HTTP multi-DiT entrypoint is disabled")
def test_launch_pool_disagg_server_supports_multi_dit_and_heterogeneous_sp(monkeypatch):
    from sglang.multimodal_gen.runtime import launch_server as ls
    from sglang.multimodal_gen.runtime.disaggregation import role_launcher

    started_role_cmds = []
    launched_dit_args = []
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    monkeypatch.setattr(
        role_launcher,
        "start_role_workers",
        lambda args, **kw: started_role_cmds.append((args, kw)) or ["role_proc"],
    )
    monkeypatch.setattr(
        ls,
        "start_role_workers",
        lambda args, **kw: started_role_cmds.append((args, kw)) or ["role_proc"],
    )
    monkeypatch.setattr(
        ls,
        "_launch_server",
        lambda args, launch_http_server=True: (
            launched_dit_args.append((args, launch_http_server))
            or [f"dit_{args.num_gpus}"]
        ),
    )

    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(decoder_config={"type": "lynn_streaming"}),
        encoder_gpus=None,
        denoiser_gpus=None,
        decoder_gpus=None,
        disagg_encoder_port=None,
        disagg_decoder_port=None,
        disagg_encoder_ports=None,
        disagg_decoder_ports=None,
        model_path="dummy",
        num_gpus=2,
        nnodes=1,
        node_rank=0,
        port=30000,
        scheduler_port=31000,
        master_port=32000,
        ulysses_degree=2,
        ring_degree=1,
        sp_degree=2,
        tp_size=1,
        dp_size=1,
        cfg_parallel_degree=1,
        log_level="info",
    )

    # Heterogeneous DiT pool: 1 single-GPU DiT (SP=1), 1 2-GPU DiT (SP=2), 1 4-GPU DiT (SP=4)
    procs = ls.launch_pool_disagg_server(
        server_args,
        encoder_gpus=[[0]],
        denoiser_gpus=[[1], [2, 3], [4, 5, 6, 7]],
        decoder_gpus=[[8]],
        launch_http_server=False,
    )

    assert len(launched_dit_args) == 3
    dit_1, _ = launched_dit_args[0]
    dit_2, _ = launched_dit_args[1]
    dit_4, _ = launched_dit_args[2]

    # Single-GPU DiT (SP=1) auto-adjusts parallelism to 1
    assert dit_1.num_gpus == 1 and dit_1.denoiser_gpus == "1"
    assert dit_1.ulysses_degree == 1 and dit_1.sp_degree == 1
    assert dit_1.port == 30000
    # 2-GPU DiT (SP=2) — spaced by 10 so HTTP port and ZMQ broker (port+1) do not collide
    assert dit_2.num_gpus == 2 and dit_2.denoiser_gpus == "2,3"
    assert dit_2.ulysses_degree * dit_2.ring_degree == 2
    assert dit_2.port == 30010
    # 4-GPU DiT (SP=4)
    assert dit_4.num_gpus == 4 and dit_4.denoiser_gpus == "4,5,6,7"
    assert dit_4.ulysses_degree * dit_4.ring_degree == 4
    assert dit_4.port == 30020
    # Shared role worker max_sessions is set to len(denoiser_gpus) == 3
    assert started_role_cmds and started_role_cmds[0][1].get("max_sessions") == 3
    assert len(procs) >= 3


# ==============================================================================
# Round 1 Adversarial Review Regression Tests
# ==============================================================================


def test_nccl_sender_open_failure_cleans_up_slot_and_allows_retry(monkeypatch):
    from sglang.multimodal_gen.runtime.conditioning.nccl import NcclSender

    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(
        torch.cuda,
        "Stream",
        lambda device=None: SimpleNamespace(synchronize=lambda: None),
    )

    sender = NcclSender("cuda:0", 10.0, max_bridges=2)
    b1 = "1" * 32
    b2 = "2" * 32

    attempts = {"count": 0}

    def flaky_connect(doc, *args):
        slot = args[-1] if args else None
        attempts["count"] += 1
        if attempts["count"] <= 2:
            raise RuntimeError("Simulated TCPStore handshake timeout")
        if slot is not None:
            slot.store = object()
            slot.group = object()
            slot.stream = SimpleNamespace(synchronize=lambda: None)

    monkeypatch.setattr(sender, "_connect", flaky_connect)

    async def run_sync(fn, *args):
        return fn(*args)

    async def run():
        # 2 failed open attempts must NOT exhaust max_bridges=2 or block b1 retry
        for _ in range(2):
            with pytest.raises(RuntimeError, match="handshake timeout"):
                await sender.open(
                    {"port": 40001, "bridge_id": b1, "timeout_s": 5.0}, "t1", run_sync
                )
        assert b1 not in sender.bridges

        # Retry b1 and open b2 should both succeed now
        res1 = await sender.open(
            {"port": 40001, "bridge_id": b1, "timeout_s": 5.0}, "t1", run_sync
        )
        res2 = await sender.open(
            {"port": 40002, "bridge_id": b2, "timeout_s": 5.0}, "t2", run_sync
        )
        assert res1["ok"] and res2["ok"]
        assert set(sender.bridges.keys()) == {b1, b2}

    asyncio.run(run())


def test_encoder_worker_bridge_open_rank0_failure_propagates_via_agree(monkeypatch):
    def broken_tcp_store(host, port, *a, **kw):
        raise RuntimeError("TCPStore connection refused on bridge_open")

    agree_steps = []

    def gather_recording(output, value, **kwargs):
        if isinstance(value, tuple) and len(value) == 3:
            agree_steps.append((value[0], value[2]))
        output[:] = [value, value]

    # When TCPStore fails on rank 0 during bridge_open, bridge_open_prepared MUST
    # receive the error (value[2] is not None) so Rank 1+ aborts at bridge_open_prepared
    # instead of hanging later!
    with pytest.raises(Exception, match="participating rank"):
        _run_encoder_worker_script(
            monkeypatch,
            [
                {"op": "bridge_open", "bridge_id": "bad-bridge", "bridge_port": 45001},
            ],
            max_sessions=2,
            rank=0,
            extra_dist={
                "TCPStore": broken_tcp_store,
                "all_gather_object": gather_recording,
            },
        )
    bridge_prep = [err for step, err in agree_steps if step == "bridge_open_prepared"]
    assert bridge_prep and bridge_prep[0] is not None


def test_encoder_worker_unknown_abort_does_not_quarantine_multi_session_worker(
    monkeypatch,
):
    replies_r0, _, _ = _run_encoder_worker_script(
        monkeypatch,
        [
            {
                "op": "open",
                "session": "s-alpha",
                "bridge_port": 43001,
                "capacity_frames": 51,
            },
            {
                "op": "open",
                "session": "s-beta",
                "bridge_port": 43002,
                "capacity_frames": 51,
            },
            {"op": "abort", "session": "unknown-intruder"},
            {"op": "abort", "session": ""},
            {"op": "ping"},
            {"op": "close", "session": "s-alpha"},
            {"op": "close", "session": "s-beta"},
        ],
        max_sessions=2,
        rank=0,
    )
    # Unknown intruder abort leaves active sessions intact without quarantining
    assert replies_r0[2]["ok"] is True
    assert replies_r0[2]["current_owner_unchanged"] is True
    assert replies_r0[2]["released"] is False
    # Empty session token returns ENCODER_INVALID_TOKEN without quarantining
    assert replies_r0[3]["ok"] is False
    assert replies_r0[3]["code"] == "ENCODER_INVALID_TOKEN"
    # Worker is NOT quarantined
    assert replies_r0[4]["restart_required"] is False
    assert replies_r0[4]["active_sessions"] == 2


def test_remote_encoder_failover_on_busy_or_socket_error_and_abort_targets_correct_port(
    monkeypatch,
):
    import contextlib

    import zmq as real_zmq

    from sglang.multimodal_gen.runtime.distributed import parallel_state
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        remote_encoder,
    )

    monkeypatch.setattr(
        parallel_state,
        "get_world_group",
        lambda: SimpleNamespace(
            rank_in_group=0, ranks=[0], world_size=1, cpu_group=None, device_group=None
        ),
    )
    remote_encoder._PERSISTENT_ENCODER_BRIDGES.clear()

    monkeypatch.setattr(
        remote_encoder.dist,
        "TCPStore",
        lambda *a, **kw: SimpleNamespace(port=46099),
    )
    monkeypatch.setattr(
        remote_encoder, "bridge_group", lambda *a, **kw: "enc_bridge_grp"
    )
    monkeypatch.setattr(
        remote_encoder.torch,
        "cuda",
        SimpleNamespace(
            set_device=lambda x: None,
            Stream=lambda **kw: SimpleNamespace(synchronize=lambda: None),
            stream=lambda x: contextlib.nullcontext(),
        ),
    )

    aborted_ports = []
    monkeypatch.setattr(
        remote_encoder,
        "request_abort",
        lambda port, token, timeout, **kw: (
            aborted_ports.append(port) or {"ok": True, "released": True}
        ),
    )

    # Scenario 1: Port 31031 raises ZMQ error on bridge_open, Port 31032 returns ENCODER_BUSY, Port 31033 succeeds!
    port_scripts = {
        31031: [
            {"ok": True, "busy": False, "active_sessions": 0, "max_sessions": 2},
            real_zmq.Again("Port 31031 timed out on bridge_open"),
        ],
        31032: [
            {"ok": True, "busy": False, "active_sessions": 0, "max_sessions": 2},
            {"ok": True, "bridge_id": "b32"},
            {"ok": False, "code": "ENCODER_BUSY", "error": "busy"},
        ],
        31033: [
            {"ok": True, "busy": False, "active_sessions": 0, "max_sessions": 2},
            {"ok": True, "bridge_id": "b33"},
            {"ok": True},  # open succeeds
            {"ok": True},  # close
        ],
    }

    class FakeFailoverSocket:
        def __init__(self):
            self.port = None

        def setsockopt(self, *a):
            pass

        def connect(self, endpoint):
            self.port = int(endpoint.rsplit(":", 1)[1])

        def send_multipart(self, parts):
            pass

        def recv_json(self):
            item = port_scripts[self.port].pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        def close(self):
            pass

    monkeypatch.setattr(
        remote_encoder.zmq,
        "Context",
        lambda: SimpleNamespace(
            socket=lambda *a: FakeFailoverSocket(), term=lambda: None
        ),
    )

    codecs = SimpleNamespace(
        modules={"video_vae": SimpleNamespace(encoder=torch.nn.Linear(1, 1))}
    )
    s1 = remote_encoder.RemoteEncoderSession(
        codecs,
        capacity_frames=34,
        config={
            "ports": [31031, 31032, 31033],
            "mode": "overlap",
            "timeout_seconds": 2,
        },
    )
    s1.dispose()
    assert s1.port == 31033

    # Scenario 2: Port 31031 returns ENCODER_BUSY, Port 31032 times out during open recv_json -> abort MUST target 31032, NOT 31031!
    remote_encoder._PERSISTENT_ENCODER_BRIDGES.clear()
    port_scripts = {
        31031: [
            {"ok": True, "busy": False, "active_sessions": 0, "max_sessions": 2},
            {"ok": True, "bridge_id": "b31"},
            {"ok": False, "code": "ENCODER_BUSY", "error": "busy"},
        ],
        31032: [
            {"ok": True, "busy": False, "active_sessions": 0, "max_sessions": 2},
            {"ok": True, "bridge_id": "b32"},
            real_zmq.Again("Timeout waiting for open reply on 31032"),
        ],
    }
    s2 = remote_encoder.RemoteEncoderSession(
        codecs,
        capacity_frames=34,
        config={"ports": [31031, 31032], "mode": "overlap", "timeout_seconds": 2},
    )
    with pytest.raises(RuntimeError, match="Remote encoder cleanup failed"):
        s2.dispose()
    assert aborted_ports == [31032]


def test_decoder_worker_abort_drain_timeout_quarantines_without_releasing_or_destroying_pg(
    monkeypatch,
):
    disposed_while_running = []
    canceled_pending = []

    class StuckDecodeService:
        def __init__(self, *a, **kw):
            self.calls = getattr(StuckDecodeService, "_count", 0)
            StuckDecodeService._count = self.calls + 1
            self.valid_frame_end = 0
            self.submitted_frames = 0
            self.buffered_latents = 0
            self.thread_alive = True

        def check(self):
            pass

        def finish(self, total_frames):
            pass

        def cancel_pending(self):
            canceled_pending.append(True)

        def drain(self, timeout_s):
            raise TimeoutError("Timed out draining stuck decode service")

        def dispose(self):
            if self.thread_alive:
                disposed_while_running.append(True)

    replies, _, _ = _run_decoder_worker_script(
        [
            {"op": "bridge_open", "bridge_id": "dec-stuck-b1", "bridge_port": 44001},
            {
                "op": "open",
                "session": "stuck-sess",
                "bridge_id": "dec-stuck-b1",
                "capacity_latents": 10,
            },
            # 1) abort when drain times out must NOT release slot or dispose active stream
            {"op": "abort", "session": "stuck-sess"},
            {"op": "ping"},
            # 2) bridge_close(force=True) when drain times out must NOT destroy PG or claim closed
            {"op": "bridge_close", "bridge_id": "dec-stuck-b1", "force": True},
            # 3) Subsequent open is rejected with DECODER_RESTART_REQUIRED
            {
                "op": "open",
                "session": "next-sess",
                "bridge_id": "dec-stuck-b1",
                "capacity_latents": 10,
            },
        ],
        max_sessions=1,
        service_cls=StuckDecodeService,
    )
    assert replies[0] == {"ok": True, "bridge_id": "dec-stuck-b1"}
    assert replies[1]["ok"] is True
    # Abort reports DECODER_RESTART_REQUIRED with released=False and does NOT dispose active stream
    assert (
        replies[2]["ok"] is False
        and replies[2]["code"] == "DECODER_RESTART_REQUIRED"
        and replies[2]["released"] is False
        and replies[2]["restart_required"] is True
    )
    assert len(canceled_pending) >= 1
    # Ping reflects quarantined state (restart_required=True, busy=True, active_sessions=1, active_bridges=1)
    assert (
        replies[3]["restart_required"] is True
        and replies[3]["busy"] is True
        and replies[3]["active_sessions"] == 1
        and replies[3]["active_bridges"] == 1
    )
    # Forced bridge_close while decode thread is still alive refuses to destroy PG
    assert (
        replies[4]["ok"] is False
        and replies[4]["code"] == "DECODER_RESTART_REQUIRED"
        and replies[4]["closed"] is False
        and replies[4]["released"] is False
        and replies[4]["active_bridges"] == 1
    )
    # Subsequent open is blocked by quarantine
    assert (
        replies[5]["ok"] is False
        and replies[5]["code"] == "DECODER_RESTART_REQUIRED"
        and replies[5]["restart_required"] is True
    )


def test_decoder_worker_multi_session_serializes_decode_group_collectives(monkeypatch):
    import threading
    import time as _time

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        decoder_worker,
    )

    active_in_collective = 0
    max_concurrent_in_collective = 0
    collective_order = []
    lock = threading.Lock()

    def fake_all_gather_object(output, obj, group=None):
        nonlocal active_in_collective, max_concurrent_in_collective
        with lock:
            active_in_collective += 1
            max_concurrent_in_collective = max(
                max_concurrent_in_collective, active_in_collective
            )
            collective_order.append(obj[0])
        _time.sleep(0.03)
        with lock:
            active_in_collective -= 1
        output[:] = [obj, obj]

    monkeypatch.setattr(
        decoder_worker.dist, "all_gather_object", fake_all_gather_object
    )
    monkeypatch.setattr(decoder_worker.dist, "get_world_size", lambda group=None: 2)

    class _FakeStream:
        def push(self, latent):
            _time.sleep(0.02)
            return torch.zeros(1, 3, 4, 8, 8, dtype=torch.float32)

        def close(self):
            pass

    fake_module = SimpleNamespace(
        protocol=SimpleNamespace(
            latent_channels=16,
            frame_count=lambda n: 4 * n,
        ),
        spatial_ratio=4,
        new_stream=lambda **kw: _FakeStream(),
    )
    fake_publisher = SimpleNamespace(
        timeout=5.0,
        publish=lambda ref, tensor: None,
        fail=lambda ref, err: None,
    )
    step_gate_cls = getattr(decoder_worker, "_DecodeStepGate", None)
    gate = step_gate_cls() if step_gate_cls is not None else None
    gate_kw = {"step_gate": gate} if gate is not None else {}

    svc1 = decoder_worker._DecodeService(
        fake_module,
        capacity_latents=10,
        group=None,
        publisher=fake_publisher,
        decode_group=object(),
        device=torch.device("cpu"),
        **gate_kw,
    )
    svc2 = decoder_worker._DecodeService(
        fake_module,
        capacity_latents=10,
        group=None,
        publisher=fake_publisher,
        decode_group=object(),
        device=torch.device("cpu"),
        **gate_kw,
    )
    try:
        latent = torch.zeros(1, 24, 1, 2, 2, dtype=torch.float32)
        svc1.append(latent, latent_start=0, valid_frame_end=4, source_dependency_end=4)
        svc2.append(latent, latent_start=0, valid_frame_end=4, source_dependency_end=4)
        j1 = svc1.submit(4, {"id": "s1"})
        j2 = svc2.submit(4, {"id": "s2"})
        assert j1["done"].wait(timeout=5.0)
        assert j2["done"].wait(timeout=5.0)
        assert max_concurrent_in_collective == 1
    finally:
        svc1.dispose()
        svc2.dispose()


# ==============================================================================
# Round 2 Adversarial Review Regression Tests
# ==============================================================================


@pytest.mark.skip(reason="Legacy shared-HTTP multi-DiT entrypoint is disabled")
def test_launch_pool_disagg_server_with_real_disagg_dataclass_validation(monkeypatch):
    import dataclasses

    from sglang.multimodal_gen.runtime import launch_server as ls
    from sglang.multimodal_gen.runtime.disaggregation import role_launcher
    from sglang.multimodal_gen.runtime.server_args.disagg import DisaggServerArgsMixin

    @dataclasses.dataclass
    class _RealDisaggArgs(DisaggServerArgsMixin):
        model_path: str = "dummy"
        num_gpus: int = 2
        nnodes: int = 1
        node_rank: int = 0
        port: int = 30000
        scheduler_port: int = 31000
        master_port: int = 32000
        ulysses_degree: int = 2
        ring_degree: int = 1
        sp_degree: int = 2
        tp_size: int = 1
        dp_size: int = 1
        cfg_parallel_degree: int = 1
        log_level: str = "info"
        pipeline_config: object = dataclasses.field(
            default_factory=lambda: SimpleNamespace(
                decoder_config={"type": "lynn_streaming"}
            )
        )

        def __post_init__(self):
            self._validate_disagg_gpu_roles()

    launched_dit_args = []
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        role_launcher, "start_role_workers", lambda args, **kw: ["role_proc"]
    )
    monkeypatch.setattr(ls, "start_role_workers", lambda args, **kw: ["role_proc"])
    monkeypatch.setattr(
        ls,
        "_launch_server",
        lambda args, launch_http_server=True: (
            launched_dit_args.append(args) or [f"dit_{args.num_gpus}"]
        ),
    )

    real_args = _RealDisaggArgs()
    procs = ls.launch_pool_disagg_server(
        real_args,
        encoder_gpus=[[0]],
        denoiser_gpus=[[1], [2, 3]],
        decoder_gpus=[[4]],
        launch_http_server=False,
    )
    assert len(launched_dit_args) == 2
    assert launched_dit_args[0].denoiser_gpus == "1"
    assert launched_dit_args[0].num_gpus == 1
    assert launched_dit_args[1].denoiser_gpus == "2,3"
    assert launched_dit_args[1].num_gpus == 2
    assert len(procs) >= 2


def test_decode_step_gate_cancelled_future_ticket_does_not_bypass_active_ticket():
    import threading

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_worker import (
        _DecodeStepGate,
    )

    gate = _DecodeStepGate()
    t0 = gate.issue()
    t1 = gate.issue()
    t2 = gate.issue()

    # Thread 0 enters with ticket 0
    gate.enter(t0)

    t2_entered = threading.Event()

    def run_t2():
        gate.enter(t2)
        t2_entered.set()
        gate.leave(t2)

    th2 = threading.Thread(target=run_t2)
    th2.start()

    # Ticket 1 is cancelled BEFORE ticket 0 finishes
    gate.leave(t1)

    # Ticket 2 MUST NOT enter while ticket 0 is still active!
    assert not t2_entered.wait(timeout=0.05), (
        "Cancelling future ticket t1 allowed t2 to bypass active ticket t0!"
    )

    # Once ticket 0 leaves, ticket 2 should immediately enter and finish
    gate.leave(t0)
    assert t2_entered.wait(timeout=2.0)
    th2.join(timeout=2.0)


def test_remote_encoder_and_decoder_evict_persistent_bridge_on_open_failure(
    monkeypatch,
):
    import contextlib

    from sglang.multimodal_gen.runtime.distributed import parallel_state
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        remote_decoder,
        remote_encoder,
    )

    monkeypatch.setattr(
        parallel_state,
        "get_world_group",
        lambda: SimpleNamespace(
            unique_name="world_group",
            rank_in_group=0,
            ranks=[0],
            world_size=1,
            cpu_group=object(),
            device_group=object(),
        ),
    )

    # 1. RemoteLynnDecoder evicts broken cached bridge on open failure and reconnects
    remote_decoder._PERSISTENT_DECODER_BRIDGES.clear()
    remote_decoder._PERSISTENT_DECODER_BRIDGES[31051] = (
        "stale-bridge-id",
        "stale-store",
        "stale-group",
    )
    dec_tcp_stores = []
    monkeypatch.setattr(
        remote_decoder.dist,
        "TCPStore",
        lambda *a, **kw: dec_tcp_stores.append(1) or SimpleNamespace(port=47001),
    )
    monkeypatch.setattr(
        remote_decoder, "bridge_group", lambda *a, **kw: "fresh_dec_bridge"
    )
    dec_replies = [
        # First session uses cached stale-bridge-id -> worker rejects open with unknown bridge_id
        {"ok": False, "code": "DECODER_PROTOCOL_ERROR", "error": "Unknown bridge_id"},
        # Second session should re-run bridge_open and succeed!
        {"ok": True, "bridge_id": "fresh-id"},
        {"ok": True, "channel": 42},
        {"ok": True, "released": True},
    ]

    class _FakeDecSock:
        def setsockopt(self, *a):
            pass

        def connect(self, *a):
            pass

        def send_json(self, msg):
            pass

        def recv_json(self):
            return dec_replies.pop(0)

        def close(self):
            pass

    monkeypatch.setattr(
        remote_decoder.zmq,
        "Context",
        lambda: SimpleNamespace(socket=lambda *a: _FakeDecSock(), term=lambda: None),
    )
    codecs_dec = SimpleNamespace(
        modules={
            "lynn_decoder": SimpleNamespace(
                protocol=SimpleNamespace(frame_count=lambda n: 4 * n)
            )
        },
        decoder=SimpleNamespace(module_key="lynn_decoder"),
    )
    dec_fail = remote_decoder.RemoteLynnDecoder(
        codecs_dec,
        capacity_latents=10,
        provenance_for=lambda *a: None,
        config={"port": 31051, "session_id": "s-fail"},
    )
    assert dec_fail.failure is not None
    assert 31051 not in remote_decoder._PERSISTENT_DECODER_BRIDGES

    dec_ok = remote_decoder.RemoteLynnDecoder(
        codecs_dec,
        capacity_latents=10,
        provenance_for=lambda *a: None,
        config={"port": 31051, "session_id": "s-ok"},
    )
    assert dec_ok.failure is None
    assert dec_ok.channel == 42
    assert len(dec_tcp_stores) == 1
    dec_ok.dispose()

    # 2. RemoteEncoderSession evicts broken cached bridge on open failure and reconnects
    remote_encoder._PERSISTENT_ENCODER_BRIDGES.clear()
    remote_encoder._PERSISTENT_ENCODER_BRIDGES[31061] = (
        "stale-enc-bridge",
        "stale-store",
        "stale-group",
    )
    enc_tcp_stores = []
    monkeypatch.setattr(
        remote_encoder.dist,
        "TCPStore",
        lambda *a, **kw: enc_tcp_stores.append(1) or SimpleNamespace(port=47002),
    )
    monkeypatch.setattr(
        remote_encoder, "bridge_group", lambda *a, **kw: "fresh_enc_bridge"
    )
    monkeypatch.setattr(
        remote_encoder.torch,
        "cuda",
        SimpleNamespace(
            set_device=lambda x: None,
            Stream=lambda **kw: SimpleNamespace(synchronize=lambda: None),
            stream=lambda x: contextlib.nullcontext(),
        ),
    )
    monkeypatch.setattr(
        remote_encoder,
        "request_abort",
        lambda *a, **kw: {"ok": True, "released": True},
    )
    enc_replies = [
        # First session uses cached stale-enc-bridge -> worker rejects open
        {"ok": False, "code": "ENCODER_PROTOCOL_ERROR", "error": "Unknown bridge_id"},
        # Second session should re-run bridge_open and succeed!
        {"ok": True, "bridge_id": "fresh-enc-id"},
        {"ok": True},  # open
        {"ok": True},  # close
    ]

    class _FakeEncSock:
        def setsockopt(self, *a):
            pass

        def connect(self, *a):
            pass

        def send_multipart(self, parts):
            pass

        def recv_json(self):
            return enc_replies.pop(0)

        def close(self):
            pass

    monkeypatch.setattr(
        remote_encoder.zmq,
        "Context",
        lambda: SimpleNamespace(socket=lambda *a: _FakeEncSock(), term=lambda: None),
    )
    codecs_enc = SimpleNamespace(
        modules={"video_vae": SimpleNamespace(encoder=torch.nn.Linear(1, 1))}
    )
    s_fail = remote_encoder.RemoteEncoderSession(
        codecs_enc,
        capacity_frames=34,
        config={"port": 31061, "mode": "overlap", "timeout_seconds": 2},
    )
    with pytest.raises(RuntimeError):
        s_fail.dispose()
    assert 31061 not in remote_encoder._PERSISTENT_ENCODER_BRIDGES

    s_ok = remote_encoder.RemoteEncoderSession(
        codecs_enc,
        capacity_frames=34,
        config={"port": 31061, "mode": "overlap", "timeout_seconds": 2},
    )
    s_ok.dispose()
    assert len(enc_tcp_stores) == 1


def test_zing_h3_adaln_table_and_joint_forward_support_arbitrary_valid_sp_degrees(
    monkeypatch,
):
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
        ZingH3CausalPipelineConfig,
    )
    from sglang.multimodal_gen.runtime.models.dits import zing_h3
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        sampling,
    )

    # 1. ZingH3CausalPipelineConfig allows weights_free AdaLN table for SP=1, 2, 4, 8, etc.
    def make_args(sp):
        return SimpleNamespace(
            sp_degree=sp,
            ulysses_degree=sp,
            attention_backend=None,
            resolve_component_attention_backend=lambda _name: (None, None),
        )

    config = ZingH3CausalPipelineConfig(
        decoder_config={
            "checkpoint_path": "/model/lynn.pt",
            "model_dir": "/model/lynn",
        },
        zing_adaln_table_path="/table",
    )
    for valid_sp in (1, 2, 4, 7, 8, 14):
        config.validate_server_args(make_args(valid_sp))
    with pytest.raises(ValueError, match="divisible by ulysses_degree"):
        config.validate_server_args(make_args(3))

    # 2. _joint_eligible allows SP=8 when num_attention_heads (56) is divisible by 8,
    # and falls back with reason="sp_size" only when num_attention_heads % sp_size != 0.
    sampler = object.__new__(sampling.ZingH3Sampler)
    sampler.joint_forward = "source_first"
    sampler.do_cfg = False
    sampler.generator = SimpleNamespace(arch=SimpleNamespace(num_attention_heads=56))
    sampler.last_run_metrics = {"joint_fallbacks": {}}
    sampler._layout_index = lambda pos: SimpleNamespace(
        host={
            "row_role": torch.zeros(4, dtype=torch.long),
            "block_idx": torch.zeros(4, dtype=torch.long),
        }
    )
    sampler._has_block = lambda pos, source: False
    sampler._collective_layout_signature = lambda pos, source, target: "sig"
    sampler._agree_sp_boundary = lambda *a, **kw: kw.get("joint_eligible", False)

    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (8, 0))
    sampler._joint_eligible({}, SimpleNamespace(), 1, 2, None)
    assert "sp_size" not in sampler.last_run_metrics["joint_fallbacks"]

    monkeypatch.setattr(zing_h3, "get_ulysses_ctx", lambda: (9, 0))
    sampler._joint_eligible({}, SimpleNamespace(), 1, 2, None)
    assert sampler.last_run_metrics["joint_fallbacks"].get("sp_size") == 1


def test_disagg_external_role_port_cli_args():
    from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs
    from sglang.multimodal_gen.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    ServerArgs.add_disagg_cli_args(parser)
    parsed = parser.parse_args(
        [
            "--denoiser-gpus",
            "1",
            "--disagg-encoder-port",
            "31003",
            "--disagg-decoder-port",
            "31004",
            "--disagg-encoder-ports",
            "31003",
            "31013",
            "--disagg-decoder-ports",
            "31004",
            "31014",
            "--disagg-max-sessions",
            "2",
        ]
    )
    assert parsed.denoiser_gpus == "1"
    assert parsed.disagg_encoder_port == 31003
    assert parsed.disagg_decoder_port == 31004
    assert parsed.disagg_encoder_ports == [31003, 31013]
    assert parsed.disagg_decoder_ports == [31004, 31014]
    assert parsed.disagg_max_sessions == 2


def test_stage_timing_get_writer_supports_concurrent_pooled_directories(tmp_path):
    from sglang.multimodal_gen.runtime.utils import zing_stage_timing

    dir_a = tmp_path / "dit0-timing"
    dir_b = tmp_path / "dit1-timing"
    w_a1 = zing_stage_timing.get_writer(dir_a)
    w_a2 = zing_stage_timing.get_writer(dir_a)
    assert w_a1 is w_a2
    assert not w_a1.stop.is_set()
    w_b1 = zing_stage_timing.get_writer(dir_b)
    try:
        assert not w_b1.stop.is_set()
        assert w_a1.stop.is_set()
    finally:
        w_b1.close()


# ==============================================================================
# Round 3: Elastic Runtime Scale-Out & Scale-In (set_max_sessions & bridge_close)
# ==============================================================================


@pytest.mark.parametrize("rank", [0, 1])
def test_encoder_worker_dynamic_set_max_sessions_and_bridge_close(monkeypatch, rank):
    destroyed_groups = []
    commands = [
        # Start with max_sessions=1 (1 DiT instance)
        {"op": "ping"},
        {"op": "bridge_open", "bridge_id": "enc-dit-0", "bridge_port": 43000},
        {
            "op": "open",
            "session": "s-dit-0",
            "bridge_id": "enc-dit-0",
            "capacity_frames": 51,
        },
        # Second DiT rejected while max_sessions=1
        {"op": "bridge_open", "bridge_id": "enc-dit-1", "bridge_port": 43001},
        {
            "op": "open",
            "session": "s-dit-1",
            "bridge_id": "enc-dit-1",
            "capacity_frames": 51,
        },
        # Dynamically scale out max_sessions: 1 -> 2 while s-dit-0 is active!
        {"op": "set_max_sessions", "max_sessions": 2},
        # Now s-dit-1 opens and runs concurrently!
        {
            "op": "open",
            "session": "s-dit-1",
            "bridge_id": "enc-dit-1",
            "capacity_frames": 51,
        },
        {"op": "ping"},
        # Reject shrinking max_sessions below active_sessions (2) or < 1
        {"op": "set_max_sessions", "max_sessions": 1},
        {"op": "set_max_sessions", "max_sessions": 0},
        # Reject bridge_close while s-dit-1 is still using enc-dit-1 (without force)
        {"op": "bridge_close", "bridge_id": "enc-dit-1"},
        # Close s-dit-1 session cleanly, then scale in DiT #1 by closing its bridge
        {"op": "close", "session": "s-dit-1"},
        {"op": "bridge_close", "bridge_id": "enc-dit-1"},
        {"op": "set_max_sessions", "max_sessions": 1},
        {"op": "ping"},
        # Meanwhile s-dit-0 on DiT #0 was never interrupted and closes cleanly
        {"op": "close", "session": "s-dit-0"},
        {"op": "bridge_close", "bridge_id": "enc-dit-0"},
        {"op": "ping"},
    ]
    replies, tcp_ports, sessions = _run_encoder_worker_script(
        monkeypatch,
        commands,
        max_sessions=1,
        rank=rank,
        extra_dist={"destroy_process_group": lambda *a: destroyed_groups.append(a)},
    )
    assert len(sessions) == 2 and all(s.closed for s in sessions)
    if rank == 0:
        assert replies[0]["max_sessions"] == 1 and replies[0]["active_bridges"] == 0
        assert replies[1] == {"ok": True, "bridge_id": "enc-dit-0"}
        assert replies[2]["ok"] is True
        assert replies[3] == {"ok": True, "bridge_id": "enc-dit-1"}
        assert replies[4]["ok"] is False and replies[4]["code"] == "ENCODER_BUSY"
        assert replies[5] == {
            "ok": True,
            "max_sessions": 2,
            "active_sessions": 1,
            "busy": False,
        }
        assert replies[6]["ok"] is True
        assert (
            replies[7]["active_sessions"] == 2
            and replies[7]["max_sessions"] == 2
            and replies[7]["busy"] is True
            and replies[7]["active_bridges"] == 2
            and replies[7]["bridge_ids"] == ["enc-dit-0", "enc-dit-1"]
        )
        assert replies[8]["ok"] is False and replies[8]["code"] == "ENCODER_BUSY"
        assert (
            replies[9]["ok"] is False and replies[9]["code"] == "INVALID_MAX_SESSIONS"
        )
        assert replies[10]["ok"] is False and replies[10]["code"] == "ENCODER_BUSY"
        assert replies[11]["ok"] is True
        assert replies[12] == {
            "ok": True,
            "bridge_id": "enc-dit-1",
            "closed": True,
            "active_bridges": 1,
            "bridge_ids": ["enc-dit-0"],
        }
        assert replies[13] == {
            "ok": True,
            "max_sessions": 1,
            "active_sessions": 1,
            "busy": True,
        }
        assert (
            replies[14]["active_sessions"] == 1
            and replies[14]["max_sessions"] == 1
            and replies[14]["active_bridges"] == 1
            and replies[14]["bridge_ids"] == ["enc-dit-0"]
        )
        assert replies[15]["ok"] is True
        assert replies[16] == {
            "ok": True,
            "bridge_id": "enc-dit-0",
            "closed": True,
            "active_bridges": 0,
            "bridge_ids": [],
        }
        assert (
            replies[17]["active_sessions"] == 0 and replies[17]["active_bridges"] == 0
        )
        # 2 bridge process groups destroyed on rank 0
        assert len([g for g in destroyed_groups if g]) == 2


def test_decoder_worker_dynamic_set_max_sessions_and_bridge_close():
    commands = [
        {"op": "ping"},
        {"op": "bridge_open", "bridge_id": "dec-dit-0", "bridge_port": 44000},
        {
            "op": "open",
            "session": "s-dit-0",
            "bridge_id": "dec-dit-0",
            "capacity_latents": 10,
        },
        {"op": "bridge_open", "bridge_id": "dec-dit-1", "bridge_port": 44001},
        # Rejected while max_sessions=1
        {
            "op": "open",
            "session": "s-dit-1",
            "bridge_id": "dec-dit-1",
            "capacity_latents": 10,
        },
        # Scale out max_sessions: 1 -> 2 while s-dit-0 is active
        {"op": "set_max_sessions", "max_sessions": 2},
        {
            "op": "open",
            "session": "s-dit-1",
            "bridge_id": "dec-dit-1",
            "capacity_latents": 10,
        },
        {"op": "ping"},
        # Reject shrinking max_sessions below active_sessions (2) or < 1
        {"op": "set_max_sessions", "max_sessions": 1},
        {"op": "set_max_sessions", "max_sessions": -1},
        # Reject bridge_close while s-dit-1 is using dec-dit-1
        {"op": "bridge_close", "bridge_id": "dec-dit-1"},
        # Close s-dit-1, close dec-dit-1 bridge, scale in max_sessions: 2 -> 1
        {"op": "close", "channel": "$channel:s-dit-1"},
        {"op": "bridge_close", "bridge_id": "dec-dit-1"},
        {"op": "set_max_sessions", "max_sessions": 1},
        {"op": "ping"},
        # Close s-dit-0 and its bridge
        {"op": "close", "channel": "$channel:s-dit-0"},
        {"op": "bridge_close", "bridge_id": "dec-dit-0"},
        {"op": "ping"},
    ]
    replies, tcp_ports, services = _run_decoder_worker_script(commands, max_sessions=1)
    assert len(services) == 2 and all(s.disposed for s in services)
    assert replies[0]["max_sessions"] == 1 and replies[0]["active_bridges"] == 0
    assert replies[1] == {"ok": True, "bridge_id": "dec-dit-0"}
    assert replies[2]["ok"] is True
    assert replies[3] == {"ok": True, "bridge_id": "dec-dit-1"}
    assert replies[4]["ok"] is False and replies[4]["code"] == "DECODER_BUSY"
    assert replies[5] == {
        "ok": True,
        "max_sessions": 2,
        "active_sessions": 1,
        "busy": False,
    }
    assert replies[6]["ok"] is True
    assert (
        replies[7]["active_sessions"] == 2
        and replies[7]["max_sessions"] == 2
        and replies[7]["active_bridges"] == 2
        and replies[7]["bridge_ids"] == ["dec-dit-0", "dec-dit-1"]
    )
    assert replies[8]["ok"] is False and replies[8]["code"] == "DECODER_BUSY"
    assert replies[9]["ok"] is False and replies[9]["code"] == "INVALID_MAX_SESSIONS"
    assert replies[10]["ok"] is False and replies[10]["code"] == "DECODER_BUSY"
    assert replies[11]["ok"] is True
    assert replies[12] == {
        "ok": True,
        "bridge_id": "dec-dit-1",
        "closed": True,
        "active_bridges": 1,
        "bridge_ids": ["dec-dit-0"],
    }
    assert replies[13] == {
        "ok": True,
        "max_sessions": 1,
        "active_sessions": 1,
        "busy": True,
    }
    assert (
        replies[14]["active_sessions"] == 1
        and replies[14]["active_bridges"] == 1
        and replies[14]["bridge_ids"] == ["dec-dit-0"]
    )
    assert replies[15]["ok"] is True
    assert replies[16] == {
        "ok": True,
        "bridge_id": "dec-dit-0",
        "closed": True,
        "active_bridges": 0,
        "bridge_ids": [],
    }
    assert replies[17]["active_sessions"] == 0 and replies[17]["active_bridges"] == 0


def test_remote_encoder_and_decoder_close_persistent_bridges_and_admin_helpers(
    monkeypatch,
):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        remote_decoder,
        remote_encoder,
    )

    remote_encoder._PERSISTENT_ENCODER_BRIDGES.clear()
    remote_decoder._PERSISTENT_DECODER_BRIDGES.clear()
    remote_encoder._PERSISTENT_ENCODER_BRIDGES[31003] = (
        "enc-b-31003",
        "store-e",
        "grp-e",
    )
    remote_decoder._PERSISTENT_DECODER_BRIDGES[31004] = (
        "dec-b-31004",
        "store-d",
        "grp-d",
    )

    destroyed = []
    monkeypatch.setattr(
        remote_encoder.dist,
        "destroy_process_group",
        lambda grp=None: destroyed.append(grp),
    )

    sent = []

    class FakeAdminSock:
        def __init__(self):
            self.endpoint = None
            self.last_op = None

        def setsockopt(self, *a):
            pass

        def connect(self, endpoint):
            self.endpoint = endpoint

        def send_multipart(self, parts):
            msg = json.loads(parts[0].decode())
            self.last_op = msg["op"]
            sent.append((self.endpoint, msg))

        def send_json(self, msg):
            self.last_op = msg["op"]
            sent.append((self.endpoint, msg))

        def recv_json(self):
            if self.last_op == "set_max_sessions":
                return {
                    "ok": True,
                    "max_sessions": 3,
                    "active_sessions": 1,
                    "busy": False,
                }
            return {"ok": True, "closed": True, "active_bridges": 0}

        def close(self):
            pass

    monkeypatch.setattr(
        remote_encoder.zmq,
        "Context",
        lambda: SimpleNamespace(socket=lambda *a: FakeAdminSock(), term=lambda: None),
    )

    r_enc_max = remote_encoder.set_encoder_max_sessions(31003, 3)
    assert r_enc_max["ok"] is True and r_enc_max["max_sessions"] == 3
    r_dec_max = remote_decoder.set_decoder_max_sessions(31004, 3)
    assert r_dec_max["ok"] is True and r_dec_max["max_sessions"] == 3

    closed_enc = remote_encoder.close_persistent_encoder_bridges()
    assert closed_enc[31003]["ok"] is True
    assert not remote_encoder._PERSISTENT_ENCODER_BRIDGES

    closed_dec = remote_decoder.close_persistent_decoder_bridges()
    assert closed_dec[31004]["ok"] is True
    assert not remote_decoder._PERSISTENT_DECODER_BRIDGES

    assert destroyed == ["grp-e", "grp-d"]
    assert sent[0] == (
        "tcp://127.0.0.1:31003",
        {"op": "set_max_sessions", "max_sessions": 3},
    )
    assert sent[1] == (
        "tcp://127.0.0.1:31004",
        {"op": "set_max_sessions", "max_sessions": 3},
    )
    assert sent[2] == (
        "tcp://127.0.0.1:31003",
        {"op": "bridge_close", "bridge_id": "enc-b-31003", "force": False},
    )
    assert sent[3] == (
        "tcp://127.0.0.1:31004",
        {"op": "bridge_close", "bridge_id": "dec-b-31004", "force": False},
    )


# ==============================================================================
# Round 4: Encoder & Decoder Runtime Elastic Scaling (set_draining & Dynamic Port Pool)
# ==============================================================================


@pytest.mark.parametrize("rank", [0, 1])
def test_encoder_worker_graceful_draining_and_resume(monkeypatch, rank):
    payload_1 = b"x" * (3 * 2 * 2 * 2)
    commands = [
        {"op": "bridge_open", "bridge_id": "enc-b1", "bridge_port": 45001},
        {"op": "open", "session": "s1", "bridge_id": "enc-b1", "capacity_frames": 51},
        # Enter draining mode while s1 is active (active_sessions=1 < max_sessions=2)
        {"op": "set_draining", "draining": True},
        {"op": "ping"},
        # Invalid draining flag rejected without crashing worker
        {"op": "set_draining", "draining": "invalid"},
        # New session s2 is rejected with ENCODER_BUSY (draining=True)
        {"op": "open", "session": "s2", "bridge_id": "enc-b1", "capacity_frames": 51},
        # Meanwhile, active session s1 continues append, pop, finish, close uninterrupted!
        (
            {
                "op": "append",
                "session": "s1",
                "shape": [3, 2, 2, 2],
                "frame_start": 0,
                "capacity_bytes": 1024,
            },
            payload_1,
        ),
        {"op": "pop", "session": "s1"},
        {"op": "finish", "session": "s1"},
        {"op": "close", "session": "s1"},
        {"op": "ping"},
        # Resume from draining mode -> s2 is now admitted!
        {"op": "set_draining", "draining": False},
        {"op": "open", "session": "s2", "bridge_id": "enc-b1", "capacity_frames": 51},
        {"op": "close", "session": "s2"},
    ]
    replies, _, sessions = _run_encoder_worker_script(
        monkeypatch, commands, max_sessions=2, rank=rank
    )
    assert len(sessions) == 2 and all(s.closed for s in sessions)
    if rank == 0:
        assert replies[2] == {
            "ok": True,
            "draining": True,
            "active_sessions": 1,
            "max_sessions": 2,
            "busy": True,
        }
        assert (
            replies[3]["draining"] is True
            and replies[3]["busy"] is True
            and replies[3]["active_sessions"] == 1
        )
        assert (
            replies[4]["ok"] is False and replies[4]["code"] == "INVALID_DRAINING_FLAG"
        )
        assert (
            replies[5]["ok"] is False
            and replies[5]["code"] == "ENCODER_BUSY"
            and replies[5]["draining"] is True
        )
        assert replies[6]["ok"] is True
        assert replies[7]["ok"] is True and replies[7]["dependency"] == 2
        assert replies[8]["ok"] is True
        assert replies[9]["ok"] is True
        # After s1 closes, active_sessions == 0 (ready to shut down worker cleanly), still draining=True
        assert (
            replies[10]["active_sessions"] == 0
            and replies[10]["draining"] is True
            and replies[10]["busy"] is True
        )
        assert replies[11] == {
            "ok": True,
            "draining": False,
            "active_sessions": 0,
            "max_sessions": 2,
            "busy": False,
        }
        assert replies[12]["ok"] is True
        assert replies[13]["ok"] is True


def test_decoder_worker_graceful_draining_and_resume():
    commands = [
        {"op": "bridge_open", "bridge_id": "dec-b1", "bridge_port": 45002},
        {"op": "open", "session": "s1", "bridge_id": "dec-b1", "capacity_latents": 10},
        # Enter draining mode while s1 is active (active_sessions=1 < max_sessions=2)
        {"op": "set_draining", "draining": True},
        {"op": "ping"},
        {"op": "set_draining", "draining": 123},
        # New session s2 is rejected while draining
        {"op": "open", "session": "s2", "bridge_id": "dec-b1", "capacity_latents": 10},
        # Existing active session s1 continues append, output, finish, close uninterrupted!
        {
            "op": "append",
            "channel": "$channel:s1",
            "shape": [1, 24, 1, 2, 2],
            "dtype": "float32",
            "latent_start": 0,
            "valid_frame_end": 16,
            "source_dependency_end": 16,
        },
        {
            "op": "output",
            "channel": "$channel:s1",
            "frames": 16,
            "generation_id": "g1",
            "request_id": "r1",
            "chunk_index": 0,
        },
        {"op": "finish", "channel": "$channel:s1", "total_frames": 16},
        {"op": "close", "channel": "$channel:s1"},
        {"op": "ping"},
        # Resume from draining mode -> s2 is now admitted!
        {"op": "set_draining", "draining": False},
        {"op": "open", "session": "s2", "bridge_id": "dec-b1", "capacity_latents": 10},
        {"op": "close", "channel": "$channel:s2"},
    ]
    replies, _, services = _run_decoder_worker_script(commands, max_sessions=2)
    assert len(services) == 2 and all(s.disposed for s in services)
    assert replies[2] == {
        "ok": True,
        "draining": True,
        "active_sessions": 1,
        "max_sessions": 2,
        "busy": True,
    }
    assert (
        replies[3]["draining"] is True
        and replies[3]["busy"] is True
        and replies[3]["active_sessions"] == 1
    )
    assert replies[4]["ok"] is False and replies[4]["code"] == "INVALID_DRAINING_FLAG"
    assert (
        replies[5]["ok"] is False
        and replies[5]["code"] == "DECODER_BUSY"
        and replies[5]["draining"] is True
    )
    assert replies[6]["ok"] is True
    assert replies[7]["ok"] is True and replies[7]["reference"]["frames"] == 16
    assert replies[8]["ok"] is True
    assert replies[9]["ok"] is True
    assert (
        replies[10]["active_sessions"] == 0
        and replies[10]["draining"] is True
        and replies[10]["busy"] is True
    )
    assert replies[11] == {
        "ok": True,
        "draining": False,
        "active_sessions": 0,
        "max_sessions": 2,
        "busy": False,
    }
    assert replies[12]["ok"] is True
    assert replies[13]["ok"] is True


def test_remote_encoder_and_decoder_dynamic_port_pool_and_draining_helpers(
    monkeypatch, tmp_path
):
    from sglang.multimodal_gen.runtime.distributed import parallel_state
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        remote_decoder,
        remote_encoder,
    )

    monkeypatch.setattr(
        parallel_state,
        "get_world_group",
        lambda: SimpleNamespace(
            rank_in_group=1,
            ranks=[0, 1],
            world_size=2,
            cpu_group=None,
            device_group=None,
        ),
    )

    registry_file = tmp_path / "disagg_pool_registry.json"
    monkeypatch.setenv("SGLANG_DISAGG_POOL_REGISTRY", str(registry_file))
    remote_encoder.set_dynamic_encoder_ports(None)
    remote_decoder.set_dynamic_decoder_ports(None)

    # Write dynamic registry expanding Encoder to [31003, 31013] and Decoder to [31004, 31014]
    remote_encoder.update_disagg_pool_registry(
        registry_file,
        encoder_ports=[31003, 31013],
        decoder_ports=[31004, 31014],
    )

    codecs_enc = SimpleNamespace(
        modules={"video_vae": SimpleNamespace(encoder=torch.nn.Linear(1, 1))}
    )
    codecs_dec = SimpleNamespace(
        modules={
            "lynn_decoder": SimpleNamespace(
                protocol=SimpleNamespace(frame_count=lambda n: 4 * n)
            )
        },
        decoder=SimpleNamespace(module_key="lynn_decoder"),
    )

    # Even when initialized with a single static port, dynamic registry overrides/expands ports and enables _pool_mode!
    enc_sess = remote_encoder.RemoteEncoderSession(
        codecs_enc,
        capacity_frames=34,
        config={"port": 31003, "mode": "overlap"},
    )
    assert enc_sess.ports == [31003, 31013]
    assert enc_sess._pool_mode is True

    dec_sess = remote_decoder.RemoteLynnDecoder(
        codecs_dec,
        capacity_latents=10,
        provenance_for=lambda *a: None,
        config={"port": 31004, "session_id": "s-dyn"},
    )
    assert dec_sess.ports == [31004, 31014]
    assert dec_sess._pool_mode is True

    # Test set_encoder_draining & set_decoder_draining RPC helpers
    sent = []

    class FakeDrainSock:
        def __init__(self):
            self.endpoint = None

        def setsockopt(self, *a):
            pass

        def connect(self, endpoint):
            self.endpoint = endpoint

        def send_multipart(self, parts):
            sent.append((self.endpoint, json.loads(parts[0].decode())))

        def send_json(self, msg):
            sent.append((self.endpoint, msg))

        def recv_json(self):
            return {"ok": True, "draining": True, "active_sessions": 1, "busy": True}

        def close(self):
            pass

    monkeypatch.setattr(
        remote_encoder.zmq,
        "Context",
        lambda: SimpleNamespace(socket=lambda *a: FakeDrainSock(), term=lambda: None),
    )

    r_enc = remote_encoder.set_encoder_draining(31003, True)
    r_dec = remote_decoder.set_decoder_draining(31004, True)
    assert r_enc["ok"] is True and r_enc["draining"] is True
    assert r_dec["ok"] is True and r_dec["draining"] is True
    assert sent == [
        ("tcp://127.0.0.1:31003", {"op": "set_draining", "draining": True}),
        ("tcp://127.0.0.1:31004", {"op": "set_draining", "draining": True}),
    ]


# ==============================================================================
# Round 5: PR #461 Reviewer (@thughy) P1 Verification Tests
# ==============================================================================


@pytest.mark.skip(reason="Legacy shared-HTTP multi-DiT entrypoint is disabled")
def test_single_entrypoint_multi_dit_pooled_routing_and_live_elastic_scaling(
    monkeypatch, tmp_path
):
    import socket as _sock
    import threading

    import zmq

    from sglang.multimodal_gen.runtime import launch_server as ls
    from sglang.multimodal_gen.runtime.disaggregation import role_launcher
    from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
        ReleaseRealtimeSessionReq,
        SetLoraReq,
    )
    from sglang.multimodal_gen.runtime.pipelines_core import Req
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.remote_encoder import (
        update_disagg_pool_registry,
    )
    from sglang.multimodal_gen.runtime.scheduler_client import (
        AsyncSchedulerClient,
        SchedulerClient,
        set_dynamic_denoiser_ports,
    )

    def _free_port() -> int:
        with _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    # 1. Verify launch_pool_disagg_server(..., launch_http_server=True) launches all DiT workers
    #    with launch_http_server=False and starts ONE shared HTTP entrypoint on base_port with
    #    server_args.scheduler_ports populated with all DiT scheduler ports!
    launched_dits = []
    http_entry_args = []
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        role_launcher, "start_role_workers", lambda args, **kw: ["role_proc"]
    )
    monkeypatch.setattr(ls, "start_role_workers", lambda args, **kw: ["role_proc"])
    monkeypatch.setattr(
        ls,
        "_launch_server",
        lambda args, launch_http_server=True: (
            launched_dits.append((args, launch_http_server)) or [f"dit_{args.port}"]
        ),
    )
    monkeypatch.setattr(
        ls,
        "launch_http_server_only",
        lambda args: http_entry_args.append(args),
    )
    monkeypatch.setattr(ls, "shutdown_scheduler_processes", lambda *a, **kw: None)
    monkeypatch.setattr(ls, "stop_role_workers", lambda *a, **kw: None)

    base_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(decoder_config={"type": "lynn_streaming"}),
        encoder_gpus=None,
        denoiser_gpus=None,
        decoder_gpus=None,
        disagg_encoder_port=None,
        disagg_decoder_port=None,
        disagg_encoder_ports=None,
        disagg_decoder_ports=None,
        model_path="dummy",
        num_gpus=1,
        nnodes=1,
        node_rank=0,
        host="127.0.0.1",
        port=30000,
        scheduler_port=31000,
        scheduler_ports=None,
        master_port=32000,
        ulysses_degree=1,
        ring_degree=1,
        sp_degree=1,
        tp_size=1,
        dp_size=1,
        cfg_parallel_degree=1,
        webui=False,
        log_level="info",
    )
    ls.launch_pool_disagg_server(
        base_args,
        encoder_gpus=[[0]],
        denoiser_gpus=[[1], [2]],
        decoder_gpus=[[3]],
        launch_http_server=True,
    )
    assert len(launched_dits) == 2
    assert all(http_flag is False for _, http_flag in launched_dits)
    assert len(http_entry_args) == 1
    assert http_entry_args[0].port == 30000
    assert http_entry_args[0].scheduler_ports == [31000, 31010]
    assert http_entry_args[0].denoiser_scheduler_ports == [31000, 31010]

    # 2. Spin up 2 real ZMQ REP mock DiT scheduler servers and drive single-entrypoint E2E routing
    #    plus live scale-out (1 -> 2) and scale-in (2 -> 1) without interrupting active sessions!
    port_dit0 = _free_port()
    port_dit1 = _free_port()
    stop_evt = threading.Event()
    dit_hits = {port_dit0: [], port_dit1: []}
    dit_enabled = {port_dit0: True, port_dit1: True}

    def _run_mock_dit_scheduler(port: int, dit_name: str):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://127.0.0.1:{port}")
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        try:
            while not stop_evt.is_set():
                events = dict(poller.poll(20))
                if sock not in events:
                    continue
                if not dit_enabled[port]:
                    # Simulate scaled-in / stopped DiT not replying
                    _ = sock.recv()
                    continue
                batch = sock.recv_pyobj()
                dit_hits[port].append(batch)
                sock.send_pyobj(OutputBatch(output=dit_name))
        finally:
            sock.close()
            ctx.term()

    t0 = threading.Thread(
        target=_run_mock_dit_scheduler, args=(port_dit0, "dit-0"), daemon=True
    )
    t1 = threading.Thread(
        target=_run_mock_dit_scheduler, args=(port_dit1, "dit-1"), daemon=True
    )
    t0.start()
    t1.start()

    registry_file = tmp_path / "pool_registry.json"
    monkeypatch.setenv("SGLANG_DISAGG_POOL_REGISTRY", str(registry_file))
    set_dynamic_denoiser_ports(None)

    client_args = SimpleNamespace(
        host="127.0.0.1",
        dp_size=1,
        scheduler_port=port_dit0,
        scheduler_ports=[port_dit0, port_dit1],
        denoiser_scheduler_ports=[port_dit0, port_dit1],
        scheduler_rpc_timeout=1,
        log_requests=False,
        log_requests_level=0,
        log_requests_format="text",
        log_requests_target="stdout",
    )

    client = SchedulerClient()
    client.initialize(client_args)
    async_client = AsyncSchedulerClient()
    async_client.initialize(client_args)
    try:
        # Phase A: Static 2-DiT pool behind a single entrypoint — consecutive requests hit both DiTs!
        outputs = [client.forward([Req(prompt=f"p-{i}")]).output for i in range(4)]
        assert outputs == ["dit-0", "dit-1", "dit-0", "dit-1"]
        assert len(dit_hits[port_dit0]) == 2 and len(dit_hits[port_dit1]) == 2

        # Control requests fan out to all DiTs in the pool
        client.forward(SetLoraReq(lora_nickname="n", lora_path="p"))
        assert len(dit_hits[port_dit0]) == 3 and len(dit_hits[port_dit1]) == 3

        # AsyncSchedulerClient also distributes consecutive requests across both DiTs
        async def _run_async_checks():
            res = [
                (await async_client.forward([Req(prompt=f"ap-{i}")])).output
                for i in range(4)
            ]
            assert res == ["dit-0", "dit-1", "dit-0", "dit-1"]

        asyncio.run(_run_async_checks())

        # Phase B: Live Scale-Out (1 -> 2) & Scale-In (2 -> 1) with sticky realtime sessions!
        # Start with only DiT-0 in registry
        update_disagg_pool_registry(registry_file, denoiser_ports=[port_dit0])
        s0_req = Req(prompt="s0-step1", realtime_session_id="sess-0")
        assert client.forward([s0_req]).output == "dit-0"

        # Scale out to [port_dit0, port_dit1] while sess-0 is active
        update_disagg_pool_registry(
            registry_file, denoiser_ports=[port_dit0, port_dit1]
        )
        # New session sess-1 and sess-2 land on both DiTs, while sess-0 stays pinned to dit-0!
        s1_out = client.forward(
            [Req(prompt="s1-step1", realtime_session_id="sess-1")]
        ).output
        s2_out = client.forward(
            [Req(prompt="s2-step1", realtime_session_id="sess-2")]
        ).output
        assert {s1_out, s2_out} == {"dit-0", "dit-1"}
        assert (
            client.forward(
                [Req(prompt="s0-step2", realtime_session_id="sess-0")]
            ).output
            == "dit-0"
        )

        # Release sess-1 and sess-2, then scale in back to [port_dit0]
        client.forward(ReleaseRealtimeSessionReq(session_id="sess-1"))
        client.forward(ReleaseRealtimeSessionReq(session_id="sess-2"))
        update_disagg_pool_registry(registry_file, denoiser_ports=[port_dit0])
        assert (
            client.forward(
                [Req(prompt="s0-step3", realtime_session_id="sess-0")]
            ).output
            == "dit-0"
        )

        # Phase C: Automatic failover if a DiT in the pool becomes unreachable
        update_disagg_pool_registry(
            registry_file, denoiser_ports=[port_dit0, port_dit1]
        )
        dit_enabled[port_dit1] = False
        for i in range(4):
            out = client.forward([Req(prompt=f"failover-{i}")], timeout_ms=200).output
            assert out == "dit-0"
    finally:
        stop_evt.set()
        client.close()
        async_client.close()
        set_dynamic_denoiser_ports(None)
        t0.join(timeout=2.0)
        t1.join(timeout=2.0)


def test_conditioning_nccl_bridge_close_idle_eviction_and_12_round_elastic_churn(
    monkeypatch,
):
    from sglang.multimodal_gen.runtime.conditioning import nccl
    from sglang.multimodal_gen.runtime.conditioning.client import ConditioningClient
    from sglang.multimodal_gen.runtime.conditioning.contract import safe_json
    from sglang.multimodal_gen.runtime.conditioning.inputs import PREPROCESS_VERSION
    from sglang.multimodal_gen.runtime.conditioning.resources import ResourceLimits
    from sglang.multimodal_gen.runtime.conditioning.service import ConditioningService

    class FakeEncoder:
        gpu_id = 0

        def encode_device(self, prompt, media, device):
            return {
                "context": torch.ones((2, 5120), dtype=torch.bfloat16),
                "row_role": torch.zeros((2,), dtype=torch.int64),
            }

    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(
        torch.cuda,
        "Stream",
        lambda device=None: SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(
        nccl.dist, "TCPStore", lambda *a, **kw: SimpleNamespace(port=41000)
    )
    destroyed_groups = []
    monkeypatch.setattr(
        nccl.dist,
        "destroy_process_group",
        lambda grp=None: destroyed_groups.append(grp),
    )

    def fake_bridge_group(store, rank, **kw):
        return SimpleNamespace(rank=rank)

    monkeypatch.setattr(nccl, "bridge_group", fake_bridge_group)

    now = [1000.0]
    profile = SimpleNamespace(
        profile_id="a" * 64, document={"preprocess_version": PREPROCESS_VERSION}
    )
    service = ConditioningService(
        profile=profile,
        encoder=FakeEncoder(),
        tokens={"token": {"tenant": "t"}, "other": {"tenant": "t2"}},
        limits=ResourceLimits(max_bridges=3),
        result_transport="nccl",
        clock=lambda: now[0],
    )
    service.nccl.clock = lambda: now[0]
    service.nccl.idle_ttl_s = 60.0

    async def asgi_post(path: str, payload: dict, token: str = "token"):
        body = json.dumps(payload).encode()
        messages = []
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.sleep(3600)

        async def send(msg):
            messages.append(msg)

        scope = {
            "type": "http",
            "method": "POST",
            "path": path,
            "client": ("127.0.0.1", 50000),
            "headers": [
                (b"authorization", f"Bearer {token}".encode()),
                (b"content-type", b"application/octet-stream"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
        await service(scope, receive, send)
        status = messages[0]["status"]
        resp_body = b"".join(m.get("body", b"") for m in messages[1:])
        return status, safe_json(resp_body)

    async def run():
        # Resident DiT #0 opens bridge 0
        b0 = f"{0:032x}"
        st, doc = await asgi_post(
            "/v1/conditioning/bridge/open",
            {"port": 41000, "bridge_id": b0, "timeout_s": 5.0},
        )
        assert st == 200 and doc == {"bridge_id": b0}
        assert len(service.nccl.bridges) == 1

        # 12 rounds of 1 -> 3 -> 1 elastic DiT scale-out and scale-in churn!
        for r in range(1, 13):
            b_add1 = f"{(r * 10 + 1):032x}"
            b_add2 = f"{(r * 10 + 2):032x}"
            # Scale out: 1 -> 3
            st1, doc1 = await asgi_post(
                "/v1/conditioning/bridge/open",
                {"port": 41001, "bridge_id": b_add1, "timeout_s": 5.0},
            )
            st2, doc2 = await asgi_post(
                "/v1/conditioning/bridge/open",
                {"port": 41002, "bridge_id": b_add2, "timeout_s": 5.0},
            )
            assert st1 == 200 and doc1 == {"bridge_id": b_add1}
            assert st2 == 200 and doc2 == {"bridge_id": b_add2}
            assert len(service.nccl.bridges) == 3

            # Wrong tenant cannot close bridge
            st_unauth, _ = await asgi_post(
                "/v1/conditioning/bridge/close",
                {"bridge_id": b_add1},
                token="other",
            )
            assert st_unauth == 401

            # Busy bridge rejects non-forced close
            service.nccl.bridges[b_add1].busy = True
            st_busy, _ = await asgi_post(
                "/v1/conditioning/bridge/close",
                {"bridge_id": b_add1},
            )
            assert st_busy == 429
            service.nccl.bridges[b_add1].busy = False

            # Scale in: close b_add1 and b_add2 (3 -> 1), plus idempotent second close
            st_c1, doc_c1 = await asgi_post(
                "/v1/conditioning/bridge/close",
                {"bridge_id": b_add1},
            )
            st_c1_idem, doc_c1_idem = await asgi_post(
                "/v1/conditioning/bridge/close",
                {"bridge_id": b_add1},
            )
            st_c2, doc_c2 = await asgi_post(
                "/v1/conditioning/bridge/close",
                {"bridge_id": b_add2},
            )
            assert st_c1 == 200 and doc_c1["closed"] is True
            assert st_c1_idem == 200 and doc_c1_idem["closed"] is False
            assert st_c2 == 200 and doc_c2["closed"] is True
            assert set(service.nccl.bridges.keys()) == {b0}

        # Verify TTL-based idle bridge eviction when a scaled-in DiT crashed without calling /bridge/close
        b_stale1 = f"{901:032x}"
        b_stale2 = f"{902:032x}"
        await asgi_post(
            "/v1/conditioning/bridge/open",
            {"port": 41003, "bridge_id": b_stale1, "timeout_s": 5.0},
        )
        await asgi_post(
            "/v1/conditioning/bridge/open",
            {"port": 41004, "bridge_id": b_stale2, "timeout_s": 5.0},
        )
        assert len(service.nccl.bridges) == 3

        # Advance clock past idle_ttl_s (60s), while keeping resident b0 active
        now[0] += 30.0
        service.nccl.acquire(b0, "t")
        service.nccl.release(b0)
        now[0] += 40.0  # b_stale1 and b_stale2 are now 70s old (>60s), b0 is 40s old

        b_fresh = f"{903:032x}"
        st_fresh, doc_fresh = await asgi_post(
            "/v1/conditioning/bridge/open",
            {"port": 41005, "bridge_id": b_fresh, "timeout_s": 5.0},
        )
        assert st_fresh == 200 and doc_fresh == {"bridge_id": b_fresh}
        assert set(service.nccl.bridges.keys()) == {b0, b_fresh}

        # Verify ConditioningClient.close() calls /v1/conditioning/bridge/close
        closed_calls = []

        def fake_transport(method, path, body, timeout):
            closed_calls.append((method, path, safe_json(body)))
            return SimpleNamespace(
                status=200,
                body=json.dumps(
                    {"bridge_id": "c" * 32, "closed": True, "active_bridges": 1}
                ).encode(),
            )

        client = ConditioningClient(
            "http://127.0.0.1:29500",
            "token",
            "a" * 64,
            result_transport="nccl",
            transport=fake_transport,
        )
        client._nccl = nccl.NcclReceiver(client)
        client._nccl.bridge_id = "c" * 32
        client._nccl.group = SimpleNamespace(rank=0)
        close_res = client.close()
        assert close_res["closed"] is True
        assert closed_calls == [
            (
                "POST",
                "/v1/conditioning/bridge/close",
                {"bridge_id": "c" * 32, "force": False},
            )
        ]

        # Service shutdown closes all remaining NCCL bridges
        await service.close()
        assert len(service.nccl.bridges) == 0
        assert len(destroyed_groups) >= 27

    asyncio.run(run())
