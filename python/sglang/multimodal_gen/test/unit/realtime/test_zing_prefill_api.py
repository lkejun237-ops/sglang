# SPDX-License-Identifier: Apache-2.0
"""CPU protocol and request ownership; no GPU/model execution evidence."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import Mock

import msgspec.msgpack
import pytest
import torch
from fastapi import WebSocketDisconnect
from test_zing_h3_source_adapter import _reference, configured  # noqa: F401

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
    realtime_video_api as api,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
    zing_prefill as prefill,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters import (
    zing_h3_source_adapter as source,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.realtime.admission import (
    InMemorySessionLeaseStore,
    RealtimeAdmissionController,
)


def payload(request_id="case-a"):
    return dict(
        type="prefill",
        request_id=request_id,
        prompt="Render snow",
        size="32x32",
        zing_realtime_api_version=1,
        source_stream=dict(format="rgb24", width=32, height=32, fps=24),
        source_frame=bytes(32 * 32 * 3),
        references=[_reference()],
    )


class Socket:
    def __init__(self):
        self.incoming, self.outgoing = asyncio.Queue(), asyncio.Queue()
        self.closed = []
        self.scope, self.query_params, self.headers = {}, {}, {}
        self.client = None
        self.app = NS(state=NS())

    async def accept(self):
        pass

    async def receive_bytes(self):
        value = await self.incoming.get()
        if value is None:
            raise WebSocketDisconnect()
        return value

    async def send_bytes(self, wire):
        self.outgoing.put_nowait(msgspec.msgpack.decode(wire))

    async def close(self, code):
        self.closed.append(code)
        self.incoming.put_nowait(None)

    def submit(self, value):
        self.incoming.put_nowait(msgspec.msgpack.encode(value))

    async def receive(self):
        return await asyncio.wait_for(self.outgoing.get(), 3)


@pytest.fixture
def service(configured, monkeypatch):
    configured.realtime_session_lease_ttl_s = 60
    controller = RealtimeAdmissionController(
        InMemorySessionLeaseStore(max_active_sessions=1, ttl_s=60)
    )
    monkeypatch.setattr(api, "get_global_server_args", lambda: configured)
    monkeypatch.setattr(api, "_get_admission_controller", lambda _: controller)
    monkeypatch.setattr(
        api, "get_realtime_model_adapter", lambda _: source.ZingH3SourceStreamAdapter()
    )
    return configured, controller


def test_socket_serial_cases_busy_rejection_and_worker_affinity(service, monkeypatch):
    async def exercise():
        ws, started, finish = Socket(), asyncio.Event(), asyncio.Event()
        calls = []

        async def run(socket, data, size, session_id):
            calls.append((data["request_id"], session_id))
            started.set()
            await finish.wait()
            return dict(
                type="prefill.completed",
                request_id=data["request_id"],
                hit=False,
                cache_key="key",
            )

        monkeypatch.setattr(prefill, "run_prefill_case", run)
        task = asyncio.create_task(prefill.prefill_websocket(ws))
        assert (await ws.receive())["type"] == "prefill.ready"
        ws.submit(payload())
        await started.wait()
        ws.submit(payload("early"))
        busy = await ws.receive()
        assert busy["code"] == "prefill_busy" and busy["request_id"] == "early"
        assert [c[0] for c in calls] == ["case-a"]
        finish.set()
        assert (await ws.receive())["request_id"] == "case-a"
        ws.submit(payload("case-b"))
        assert (await ws.receive())["request_id"] == "case-b"
        assert [c[0] for c in calls] == ["case-a", "case-b"]
        assert calls[0][1] == calls[1][1]
        ws.incoming.put_nowait(None)
        await task

    asyncio.run(exercise())


def test_socket_invalid_case_can_retry_and_worker_failure_is_private(
    service, monkeypatch
):
    async def exercise():
        ws = Socket()

        async def run(socket, data, size, session_id):
            prefill.parse_prefill(data)
            raise RuntimeError("private-worker-secret")

        monkeypatch.setattr(prefill, "run_prefill_case", run)
        task = asyncio.create_task(prefill.prefill_websocket(ws))
        await ws.receive()
        ws.submit(dict(type="prefill"))
        assert (await ws.receive())["code"] == "prefill_invalid"
        ws.submit(payload())
        error = await ws.receive()
        assert error["code"] == "prefill_failed" and error["fatal"]
        assert "secret" not in str(error)
        await task
        assert ws.closed == [1011]

    asyncio.run(exercise())


def test_disconnect_waits_for_case_cleanup(service, monkeypatch):
    async def exercise():
        ws, started, draining, finish = (
            Socket(),
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )

        async def run(*args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                draining.set()
                await finish.wait()

        monkeypatch.setattr(prefill, "run_prefill_case", run)
        task = asyncio.create_task(prefill.prefill_websocket(ws))
        await ws.receive()
        ws.submit(payload())
        await started.wait()
        ws.incoming.put_nowait(None)
        await draining.wait()
        assert not task.done()
        finish.set()
        await task

    asyncio.run(exercise())


@pytest.mark.parametrize("hit", [False, True])
def test_case_uses_normal_request_and_cleans_before_receipt(service, monkeypatch, hit):
    args, controller = service
    args.zing_source_work_quantum = 1
    args.zing_source_defer_final = True
    args.zing_source_prompt_joint = True
    events, batches = [], []
    monkeypatch.setattr(source, "build_realtime_sampling_params", lambda *a, **kw: NS())

    def prepare(adapter, session, server_args, chunk):
        params = adapter.build_sampling_params(
            session,
            server_args,
            chunk,
            source.RealtimeChunkInputs(session.request.prompt),
            17,
        )
        return NS(extra={}, sampling_params=params)

    monkeypatch.setattr(
        source.BaseRealtimeModelAdapter, "prepare_next_request", prepare
    )

    async def forward(batch):
        batches.append(batch)
        events.append("dispatch")
        return [], OutputBatch(
            output=[],
            usage={
                "zing_prefill": dict(cache_key="key", hit=hit, entries=1, evictions=0)
            },
        )

    async def release(session_id):
        events.append("release")
        return True

    monkeypatch.setattr(api, "_process_realtime_generation_batch", forward)
    monkeypatch.setattr(api, "_release_scheduler_realtime_session", release)

    async def exercise():
        data = payload()
        result = await prefill.run_prefill_case(
            Socket(), data, len(msgspec.msgpack.encode(data)), "connection"
        )
        events.append("receipt")
        assert result["hit"] is hit and result["request_id"] == "case-a"
        assert events == ["dispatch", "release", "receipt"]
        command = batches[0].extra
        assert command["zing_prefill"]["source_frame"] == data["source_frame"]
        policy = command["zing_source_command"]
        assert policy["work_quantum"] == 1
        assert policy["defer_final"] is True
        assert policy["prompt_joint"] is True
        user = batches[0].sampling_params.zing_request["messages"][0]
        assert user["content"][0]["payload"] == data["references"][0]["payload"]
        assert user["content"][-1]["text"] == data["prompt"]
        lease = await controller.admit("next", "next", "next")
        await controller.release(lease)

    asyncio.run(exercise())


@pytest.mark.parametrize("bad_frame", [None, b"short"])
def test_invalid_s0_never_dispatches_and_releases_slot(service, monkeypatch, bad_frame):
    _, controller = service
    dispatch = Mock(side_effect=AssertionError("Invalid S0 dispatched"))
    monkeypatch.setattr(api, "_process_realtime_generation_batch", dispatch)

    async def release(_):
        return True

    monkeypatch.setattr(api, "_release_scheduler_realtime_session", release)

    async def exercise():
        data = payload()
        data["source_frame"] = bad_frame
        with pytest.raises(ValueError, match="RGB24"):
            await prefill.run_prefill_case(Socket(), data, 4096, "connection")
        dispatch.assert_not_called()
        lease = await controller.admit("next", "next", "next")
        await controller.release(lease)

    asyncio.run(exercise())


def test_worker_prefill_never_enters_source_generation(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        source_session,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )

    stage = object.__new__(ZingH3GenerationStage)
    stage.block0_cache = NS(hits=0, entries={"key": object()}, evictions=2)
    stage._build_request = lambda _: dict(prompt="scene")
    seen = []

    def prefill(state, request, **kwargs):
        seen.append(kwargs["source_preview"]["frames"])
        return NS(key="key")

    monkeypatch.setattr(source_session.ZingSourceStreamState, "prefill", prefill)
    monkeypatch.setattr(
        source_session.ZingSourceStreamState,
        "run",
        Mock(side_effect=AssertionError("Source generation ran")),
    )
    batch = NS(
        session=object(),
        extra={
            "zing_prefill": dict(source_frame=bytes(32 * 32 * 3)),
            "zing_source_command": dict(config=dict(width=32, height=32, fps=24)),
        },
    )
    result = stage._forward_realtime(batch, None)
    assert result.output == []
    assert result.usage["zing_prefill"] == dict(
        cache_key="key", hit=False, entries=1, evictions=2
    )
    assert seen[0].dtype == torch.uint8 and tuple(seen[0].shape) == (3, 1, 32, 32)


def test_prefill_route_reuses_one_websocket_for_two_cases(service, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    async def run(ws, data, size, session_id):
        prefill.parse_prefill(data)
        return dict(
            type="prefill.completed",
            request_id=data["request_id"],
            cache_key="key",
            hit=True,
        )

    monkeypatch.setattr(prefill, "run_prefill_case", run)
    app = FastAPI()
    app.include_router(api.router)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime_video/prefill") as ws:
            assert msgspec.msgpack.decode(ws.receive_bytes())["type"] == "prefill.ready"
            for name in ["a", "b"]:
                ws.send_bytes(msgspec.msgpack.encode(payload(name)))
                assert msgspec.msgpack.decode(ws.receive_bytes())["request_id"] == name


def test_internal_prefill_miss_then_hit_runs_only_prefix_and_releases_lease():
    from contextlib import contextmanager

    from test_zing_preparation_cache import inputs

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        ZingSourceStreamState,
    )
    from sglang.multimodal_gen.runtime.realtime.session import RealtimeSession

    events, entries = [], {}
    generator = torch.nn.Linear(1, 1)
    cache = NS(
        lookup=lambda key: entries.get(key),
        entries=entries,
        finish_fill=lambda: events.append("fill-done"),
    )

    def fill(key, *args):
        entries[key] = NS(key=key)

    cache.fill = fill

    @contextmanager
    def use(name, *args):
        events.append(name)
        yield generator

    sampler = NS(
        window_policy=NS(window_chunks=6, sink_chunks=3),
        rope_position_mode="relative",
        rope_max_frame_gap=12,
        processor=NS(pack_inference_layout=lambda *args: {}),
    )

    def initialize(*args, **kwargs):
        assert sampler.generator is generator
        events.append("prefix")
        return NS(cache=object(), close=lambda: events.append("state-close"))

    sampler.initialize = initialize
    lease = NS(kv_cache=object(), release=lambda: events.append("lease-release"))
    codecs = NS(
        remote_read_fs=None,
        video_decoder_name="lynn",
        prepare=Mock(return_value=inputs()),
    )
    stage = NS(
        block0_cache=cache,
        codecs=codecs,
        _use=use,
        _validate_sampling_params=Mock(),
        _build_realtime_sampler=lambda *a, **kw: sampler,
        _conditioning_consumer=lambda _: None,
        packer=NS(visual_channels=1),
        kv_pool=NS(acquire=lambda *args: lease),
        kv_request=lambda *a, **kw: object(),
    )
    batch = NS(extra={}, sampling_params=NS(), session=RealtimeSession())
    state = ZingSourceStreamState()
    first = state.prefill(
        dict(prompt="scene"),
        stage=stage,
        batch=batch,
        config=dict(width=32, height=32, fps=24),
    )
    assert events == [
        "transformer",
        "prefix",
        "fill-done",
        "state-close",
        "lease-release",
    ]
    assert (
        state.prefill(
            dict(prompt="scene"),
            stage=stage,
            batch=batch,
            config=dict(width=32, height=32, fps=24),
        )
        is first
    )
    codecs.prepare.assert_called_once()
    assert state.pixels is state.decoder is None
