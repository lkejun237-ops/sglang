# SPDX-License-Identifier: Apache-2.0
"""CPU protocol and request ownership; no GPU/model execution evidence."""

import asyncio
import json
import logging
import threading
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
from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import (  # noqa: F401
    cpu_dma,
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


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("hit", [False, True])
@pytest.mark.parametrize("force", [False, True])
def test_case_uses_normal_request_and_cleans_before_receipt(
    service, monkeypatch, hit, force, routed
):
    args, controller = service
    args.zing_source_work_quantum = 1
    args.zing_source_defer_final = True
    args.zing_source_prompt_joint = True
    events, batches = [], []
    execution_client = object() if routed else None
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

    async def forward(batch, *, execution_client=None):
        assert (execution_client is not None) is routed
        batches.append(batch)
        events.append("dispatch")
        return [], OutputBatch(
            output=[],
            usage={
                "zing_prefill": dict(cache_key="key", hit=hit, entries=1, evictions=0)
            },
        )

    async def release(session_id, *, execution_client=None):
        assert (execution_client is not None) is routed
        events.append("release")
        return True

    monkeypatch.setattr(api, "_process_realtime_generation_batch", forward)
    monkeypatch.setattr(api, "_release_scheduler_realtime_session", release)

    async def exercise():
        data = payload()
        data["force"] = force
        data.update(game_id="game", group_id="stage")
        ws = Socket()
        if routed:
            ws.app.state.realtime_execution_client = execution_client
            ws.app.state.realtime_admission_controller = controller
            ws.app.state.realtime_adapter_factory = lambda _: (
                source.ZingH3SourceStreamAdapter()
            )
        result = await prefill.run_prefill_case(
            ws, data, len(msgspec.msgpack.encode(data)), "connection"
        )
        events.append("receipt")
        assert result["hit"] is hit and result["request_id"] == "case-a"
        assert events == ["dispatch", "release", "receipt"]
        command = batches[0].extra
        assert command["zing_prefill"]["source_frame"] == data["source_frame"]
        assert command["zing_prefill"]["force"] is force
        assert command["zing_prefill"]["game_id"] == "game"
        assert command["zing_prefill"]["group_id"] == "stage"
        for field in ("force", "game_id", "group_id", "cache_write"):
            assert field not in batches[0].sampling_params.zing_request
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


@pytest.mark.parametrize("force", [False, True])
def test_worker_prefill_never_enters_source_generation(monkeypatch, force):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
        source_session,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )

    stage = object.__new__(ZingH3GenerationStage)
    stage.block0_cache = NS(
        hits=0, entries={"key": object()}, evictions=2, register=Mock()
    )
    stage._build_request = lambda _: dict(prompt="scene")
    seen = []

    def prefill(state, request, **kwargs):
        seen.append(kwargs["source_preview"]["frames"])
        seen.append(kwargs["force"])
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
            "zing_prefill": dict(source_frame=bytes(32 * 32 * 3), force=force),
            "zing_source_command": dict(config=dict(width=32, height=32, fps=24)),
        },
    )
    result = stage._forward_realtime(batch, None)
    assert result.output == []
    assert result.usage["zing_prefill"] == dict(
        cache_key="key", hit=False, entries=1, evictions=2
    )
    assert seen[0].dtype == torch.uint8 and tuple(seen[0].shape) == (3, 1, 32, 32)
    assert seen[1] is force


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


def test_internal_prefill_miss_then_hit_runs_only_prefix_and_releases_lease(caplog):
    caplog.set_level(logging.DEBUG)
    from contextlib import contextmanager

    from test_zing_preparation_cache import inputs

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        ZingSourceStreamState,
    )
    from sglang.multimodal_gen.runtime.realtime.session import RealtimeSession

    events, entries = [], {}
    generator = torch.nn.Linear(1, 1)
    cache = NS(
        identity={"model": {"dit": "fixture"}, "layout": {"sp": 1}},
        lookup=lambda key, **kwargs: entries.get(key),
        entries=entries,
        finish_fill=lambda: events.append("fill-done"),
    )

    def fill(key, *args, force=False):
        events.append(("force", force))
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
        admit_cached_preparation=Mock(),
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
        ("force", False),
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
    events.clear()
    forced = state.prefill(
        dict(prompt="scene"),
        stage=stage,
        batch=batch,
        config=dict(width=32, height=32, fps=24),
        force=True,
    )
    assert forced is not first and forced.key == first.key
    assert codecs.prepare.call_count == 2
    assert ("force", True) in events and "prefix" in events
    assert state.pixels is state.decoder is None
    values = [
        json.loads(r.getMessage().removeprefix("zing_cache_event "))
        for r in caplog.records
        if r.getMessage().startswith("zing_cache_event ")
    ]
    prefill = [v for v in values if v["event"] == "prefill"]
    assert [v["reason"] for v in prefill if v["result"] == "completed"] == [
        "recomputed",
        "reused",
        "recomputed",
    ]
    assert prefill[-2]["reason"] == "force_recompute"
    assert all(v["l2_persisted"] is None for v in prefill if v["result"] == "completed")
    assert all(v["event"] != "l2_write" for v in values)


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_force_is_a_strict_operation_flag(value):
    data = payload()
    data["force"] = value
    with pytest.raises(prefill.PrefillRequestError, match="boolean"):
        prefill.parse_prefill(data)


def test_force_is_not_part_of_the_model_request():
    data = payload()
    default, frame = prefill.parse_prefill(data)
    data["force"] = True
    forced, forced_frame = prefill.parse_prefill(data)
    assert forced.model_dump() == default.model_dump()
    assert frame == forced_frame


def test_forced_prefill_receipt_does_not_wait_for_real_background_writer(
    service, monkeypatch, tmp_path, cpu_dma
):
    from test_zing_preparation_cache import inputs

    from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import cpu_cache

    monkeypatch.setenv("SGLANG_ZING_DISK_CACHE_DIR", str(tmp_path))
    owner = cpu_cache([NS(k=torch.zeros(8, 2, 3))], rows=8, entries=1)
    started, release = threading.Event(), threading.Event()
    write = owner._write_disk

    def delayed(entry):
        started.set()
        assert release.wait(10)
        write(entry)

    monkeypatch.setattr(owner, "_write_disk", delayed)
    monkeypatch.setattr(
        source.BaseRealtimeModelAdapter,
        "prepare_next_request",
        lambda *a: NS(extra={}, sampling_params=NS()),
    )
    live = NS(
        prefix=NS(
            positions=torch.zeros(3, 3),
            layers=[(torch.ones(3, 2, 3), torch.ones(3, 2, 3))],
        ),
        _model_signature="model",
    )

    async def forward(batch):
        assert batch.extra["zing_prefill"]["force"] is True
        owner.fill("case", inputs(), "condition", "request", live, force=True)
        owner.finish_fill()
        return [], OutputBatch(
            output=[], usage={"zing_prefill": {"cache_key": "case", "hit": False}}
        )

    async def cleanup(_):
        return True

    monkeypatch.setattr(api, "_process_realtime_generation_batch", forward)
    monkeypatch.setattr(api, "_release_scheduler_realtime_session", cleanup)

    async def exercise():
        data = payload()
        data["force"] = True
        result = await asyncio.wait_for(
            prefill.run_prefill_case(Socket(), data, 4096, "connection"), 3
        )
        assert result["type"] == "prefill.completed"
        assert started.wait(3) and not release.is_set()
        assert owner.entries["case"].force
        assert owner.disk_cache.lookup("case") is None

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        owner.wait_for_disk()
    assert owner.disk_cache.lookup("case") is not None


def test_disk_management_api_is_scoped_and_never_dispatches_models(
    tmp_path, monkeypatch
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from sglang.multimodal_gen.runtime.entrypoints.openai import zing_cache_api
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache import (
        Block0DiskCache,
    )
    from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import (
        disk_commit,
    )

    owner = Block0DiskCache(
        tmp_path,
        max_bytes=1 << 24,
        identity={"model": {"fixture": "model"}, "layout": {"sp": 1}},
    )
    first, second = "a" * 64, "b" * 64
    disk_commit(owner, first)
    disk_commit(owner, second)
    owner.collect()
    monkeypatch.setattr(
        zing_cache_api,
        "get_global_server_args",
        lambda: NS(zing_block0_cache_dir=str(tmp_path)),
    )
    monkeypatch.setattr(
        api,
        "_process_realtime_generation_batch",
        Mock(side_effect=AssertionError("Model dispatched")),
    )
    app = FastAPI()
    app.include_router(zing_cache_api.router)
    with TestClient(app) as client:
        response = client.get("/v1/zing/cache", params={"limit": 1})
        assert response.status_code == 200
        assert response.json()["total_entries"] == 2
        assert response.json()["next_offset"] == 1
        assert client.delete("/v1/zing/cache/entries/first").status_code == 422
        deleted = client.delete(f"/v1/zing/cache/entries/{first}").json()
        assert deleted["deleted"] == [first] and not deleted["failed"]
        cleared = client.post("/v1/zing/cache/clear").json()
        assert cleared["deleted"] == [second]
        owner.collect()
        assert client.get("/v1/zing/cache").json()["total_entries"] == 0
        monkeypatch.setattr(
            zing_cache_api,
            "get_global_server_args",
            lambda: NS(zing_block0_cache_dir=None),
        )
        assert client.get("/v1/zing/cache").json() == {
            "detail": {"code": "cache_unavailable"}
        }


def test_l1_api_busy_status_and_cancel_drain_preserve_admission(service, monkeypatch):
    from fastapi import HTTPException

    from sglang.multimodal_gen.runtime.entrypoints.control_requests import ZingCacheReq
    from sglang.multimodal_gen.runtime.entrypoints.openai import zing_cache_api

    args, controller = service
    args.dp_size = 1
    monkeypatch.setattr(zing_cache_api, "get_global_server_args", lambda: args)

    async def exercise():
        started, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def forward(req):
            assert isinstance(req, ZingCacheReq)
            assert req.log_context["operation_id"]
            if req.operation != "status":
                assert req.log_context["game_id"] == "game"
            calls.append(req.operation)
            if req.operation == "status":
                return OutputBatch(output={"busy": True, "entries": 0})
            started.set()
            await release.wait()
            return OutputBatch(output={"results": [], "all_ready": True})

        monkeypatch.setattr(api.async_scheduler_client, "forward", forward)
        task = asyncio.create_task(
            zing_cache_api._worker_call(Socket(), "preload", game_id="game")
        )
        await started.wait()
        assert await controller.store.active_count() == 1
        with pytest.raises(HTTPException) as exc:
            await zing_cache_api._worker_call(Socket(), "unload", game_id="game")
        assert exc.value.status_code == 409
        assert (await zing_cache_api._worker_call(Socket(), "status"))["busy"]
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and await controller.store.active_count() == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await controller.store.active_count() == 0
        await zing_cache_api._worker_call(Socket(), "unload", game_id="game")
        assert calls == ["preload", "status", "unload"]
        args.dp_size = 2
        with pytest.raises(HTTPException) as exc:
            await zing_cache_api._worker_call(Socket(), "preload", game_id="game")
        assert exc.value.detail["code"] == "cache_requires_dp1"
        assert await controller.store.active_count() == 0

    asyncio.run(exercise())


def test_cache_management_routes_validate_and_forward_selectors(service, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from sglang.multimodal_gen.runtime.entrypoints.openai import zing_cache_api

    args, _ = service
    args.dp_size = 1
    monkeypatch.setattr(zing_cache_api, "get_global_server_args", lambda: args)
    seen = []

    async def forward(req):
        seen.append((req.operation, req.game_id, req.group_id, req.cache_key))
        return OutputBatch(output={"all_ready": True, "selected": []})

    monkeypatch.setattr(api.async_scheduler_client, "forward", forward)
    app = FastAPI()
    app.include_router(zing_cache_api.router)
    with TestClient(app) as client:
        assert client.post("/v1/zing/cache/preload", json={}).status_code == 422
        assert (
            client.post(
                "/v1/zing/cache/preload", json={"game_id": "g", "cache_key": "../bad"}
            ).status_code
            == 422
        )
        assert (
            client.get(
                "/v1/zing/cache/status", params={"group_id": "stage"}
            ).status_code
            == 422
        )
        body = {"game_id": "g", "group_id": "stage", "cache_key": "a" * 64}
        assert client.post("/v1/zing/cache/preload", json=body).status_code == 200
        assert client.post("/v1/zing/cache/unload", json=body).status_code == 200
        assert (
            client.get(
                "/v1/zing/cache/status", params={"cache_key": "a" * 64}
            ).status_code
            == 200
        )
    assert seen == [
        ("preload", "g", "stage", "a" * 64),
        ("unload", "g", "stage", "a" * 64),
        ("status", None, None, "a" * 64),
    ]


def test_cache_control_selects_index_and_rejects_busy_mutation(tmp_path, cpu_dma):
    from sglang.multimodal_gen.runtime.entrypoints.control_requests import ZingCacheReq
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.cache_index import (
        CacheIndex,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.disk_cache import (
        Block0DiskCache,
    )
    from sglang.multimodal_gen.test.unit.realtime.zing_cache_fixtures import (
        cpu_cache,
        disk_commit,
    )

    owner = cpu_cache(
        [NS(k=torch.zeros(8, 2, 2, dtype=torch.bfloat16)) for _ in range(2)],
        rows=8,
        entries=2,
    )
    owner.disk_cache = Block0DiskCache(tmp_path, identity=owner.identity)
    key = "a" * 64
    disk_commit(owner.disk_cache, key)
    owner.register(key, "game", "stage")
    active = {}

    def call(op, game=None, group=None, selected=None):
        return owner.control(ZingCacheReq(op, game, group, selected), busy=bool(active))

    assert call("preload", "game", "stage")["all_ready"]
    assert call("status", "game")["selected"] == [{"cache_key": key, "ready": True}]
    assert call("preload", "other", selected=key)["code"] == "invalid_cache_selection"
    assert not call("status", "unknown-game")["all_ready"]
    active["session"] = object()
    assert call("status", selected=key)["busy"]
    assert call("unload", "game")["code"] == "cache_busy"
    assert key in owner.entries
    active.clear()
    assert call("unload", "game")["results"] == [
        {"cache_key": key, "status": "unloaded"}
    ]
    assert CacheIndex(tmp_path).select("game") == [key]
    assert owner.disk_cache.lookup(key) is not None
