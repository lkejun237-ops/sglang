# SPDX-License-Identifier: Apache-2.0
"""Issue #256 v6 API contracts; GPU execution and external I/O stay at boundaries."""

if __name__ == "__main__":
    import runpy
    import sys
    from pathlib import Path

    test_file = Path(__file__).resolve()
    runner = (
        test_file.parents[4] / "benchmark/zing_h3_causal/run_conditioning_cpu_tests.py"
    )
    arguments = ["-x" if value == "-f" else value for value in sys.argv[1:]]
    sys.argv = [str(runner), str(test_file), *arguments]
    runpy.run_path(str(runner), run_name="__main__")


import asyncio
import importlib
import json
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import msgspec.msgpack
import pytest

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
    ZingH3CausalPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.conditioning.contract import _PUBLIC, ConditionError
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters import (
    zing_h3_realtime_adapter as zing,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _forbidden_gpu(*args, **kwargs):
    raise AssertionError("CPU contract tests must not construct GPU pipelines")


def test_remote_metadata_accepts_21_references_and_source_preview():
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.conditioning import (
        validate_remote_media_metadata,
    )

    media = [dict(type="video", clip=dict(start=0, end=0))] + [
        dict(type="image") for _ in range(21)
    ]
    validate_remote_media_metadata(media)
    try:
        validate_remote_media_metadata([*media, dict(type="image")])
    except ConditionError as error:
        assert error.code == "CONDITION_INVALID"
    else:
        raise AssertionError("Excess conditioning media must be rejected")


def _managers():
    # The CPU runner omits eager model registries. Patch only their GPU
    # constructors; the worker and scheduler methods below are real code.
    core = importlib.import_module("sglang.multimodal_gen.runtime.pipelines_core")
    with ExitStack() as stack:
        for name in ("ComposedPipelineBase", "LoRAPipeline", "build_pipeline"):
            if not hasattr(core, name):
                value = (
                    _forbidden_gpu
                    if name == "build_pipeline"
                    else type(name, (), {"__init__": _forbidden_gpu})
                )
                stack.enter_context(patch.object(core, name, value, create=True))
        worker = importlib.import_module(
            "sglang.multimodal_gen.runtime.managers.gpu_worker"
        )
        scheduler = importlib.import_module(
            "sglang.multimodal_gen.runtime.managers.scheduler"
        )
    return worker, scheduler


def _req(**kwargs):
    request = Req(
        sampling_params=SamplingParams(prompt="scene", num_outputs_per_prompt=1)
    )
    request.suppress_logs = True
    request.metrics = None
    for name, value in kwargs.items():
        setattr(request, name, value)
    return request


def _metadata(value):
    # Dynamic assignment makes these tests reach a behavior assertion on the
    # old dataclass, rather than failing on new constructor keywords.
    value.error_code = "CONDITION_BUSY"
    value.conditioning_notice = {
        "code": "CONDITION_TIMEOUT",
        "content": _PUBLIC["CONDITION_TIMEOUT"],
    }
    value.realtime_effective_prompt = {"event_id": 7, "prompt_version": 2}
    return value


class _Socket:
    def __init__(self):
        self.messages = []

    async def send_bytes(self, payload):
        self.messages.append(msgspec.msgpack.decode(payload))


class TestConditioningRPCOutputFields(CustomTestCase):
    def assert_metadata(self, source, result):
        for name in ("error_code", "conditioning_notice", "realtime_effective_prompt"):
            self.assertEqual(getattr(result, name, None), getattr(source, name), name)

    def test_t21b_req_conversion_retains_fields(self):
        worker, _ = _managers()
        req = _metadata(_req())
        self.assert_metadata(req, worker.GPUWorker._req_to_output_batch(req))

    def test_t21b_expanded_merge_retains_fields(self):
        worker, _ = _managers()
        source = _metadata(OutputBatch(error="private encoder details"))
        merged = OutputBatch()
        worker.GPUWorker._merge_expanded_singletons(merged, source)
        self.assert_metadata(source, merged)

    def test_t21b_scheduler_split_retains_fields(self):
        _, scheduler = _managers()
        source = _metadata(OutputBatch(output_file_paths=["a.mp4", "b.mp4"]))
        split = object.__new__(scheduler.Scheduler)._split_batched_output(
            source, [_req(), _req()]
        )
        self.assertEqual(len(split), 2)
        for result in split:
            self.assert_metadata(source, result)

    def test_t21b_dynamic_error_reconstruction_retains_fields(self):
        _, scheduler = _managers()
        source = _metadata(OutputBatch(error="private encoder details"))
        results = object.__new__(
            scheduler.Scheduler
        )._build_dynamic_batch_error_outputs([_req(), _req()], source)
        for result in results:
            self.assert_metadata(source, result)

    def test_scheduler_uses_effective_prompt_pair(self):
        _, scheduler = _managers()
        request = _req(
            realtime_session_id="s", realtime_event_id=11, realtime_prompt_version=3
        )
        output = _metadata(OutputBatch())
        scheduler.Scheduler._attach_realtime_request_metadata(output, request)
        self.assertEqual(output.realtime_request_metadata["event_id"], 7)
        self.assertEqual(output.realtime_request_metadata["prompt_version"], 2)


def _custom_request(media):
    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "scene"}, *media]},
            {
                "role": "target",
                "content": [
                    {
                        "type": "video",
                        "output": {"frames": 22, "width": 832, "height": 480},
                    }
                ],
            },
        ]
    }


class TestConditioningRPCInputPreflight(CustomTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        Path(self.directory.name, "zing_config.json").write_text(
            json.dumps(
                {
                    "num_frame_per_block": 5,
                    "split_first_latent": False,
                    "h3_media_config": {"fps": 24},
                }
            )
        )
        self.args = SimpleNamespace(
            model_path=self.directory.name,
            pipeline_config=ZingH3CausalPipelineConfig(),
            zing_conditioning_mode="remote",
            zing_source_decoder_port=None,
        )
        self.args_patch = patch.object(
            zing, "get_global_server_args", return_value=self.args
        )
        self.args_patch.start()
        self.addCleanup(self.args_patch.stop)

    def _session(self):
        session = GenerateSession()
        adapter = zing.ZingH3RealtimeAdapter()
        session.set_adapter(adapter)
        # An old implementation reaches this inert external connection; the
        # new API must perform metadata validation without any encoder RPC.
        session.conditioning_resolver = SimpleNamespace(initialize=AsyncMock())
        return adapter, session

    def test_t2_remote_initialization_rejects_multiframe_and_audio(self):
        for media in (
            {
                "type": "video",
                "id": "v",
                "uri": "/missing/video.mp4",
                "clip": {"start": 0, "end": 1},
            },
            {
                "type": "video",
                "id": "v",
                "uri": "/missing/video.mp4",
                "clip": {"start": 0, "end": None},
            },
            {"type": "audio", "id": "a", "uri": "/missing/audio.wav"},
        ):
            with self.subTest(media=media):
                adapter, session = self._session()
                request = RealtimeVideoGenerationsRequest(
                    type="init", prompt="scene", zing_request=_custom_request([media])
                )
                with self.assertRaises(ConditionError) as raised:
                    asyncio.run(adapter.on_init(session, request))
                self.assertEqual(raised.exception.code, "CONDITION_INVALID")
                session.conditioning_resolver.initialize.assert_not_awaited()

    def test_t2_remote_prompt_update_rejects_invalid_metadata_before_state_change(self):
        for media in (
            {
                "type": "video",
                "id": "v",
                "uri": "/missing/video.mp4",
                "clip": {"start": 2, "end": 3},
            },
            {"type": "audio", "id": "a", "uri": "/missing/audio.wav"},
        ):
            with self.subTest(media=media):
                adapter, session = self._session()
                request = RealtimeVideoGenerationsRequest(type="init", prompt="scene")
                asyncio.run(adapter.on_init(session, request))
                session.set_request(request)
                session.request.zing_request = _custom_request([media])
                with self.assertRaises(ConditionError) as raised:
                    adapter.ingest_event(
                        session,
                        RealtimeEvent(
                            type="event", kind="prompt", payload="next", event_id=9
                        ),
                    )
                self.assertEqual(raised.exception.code, "CONDITION_INVALID")
                self.assertIsNone(session.adapter_state.pending_prompt)

    def test_t2_nonzero_single_frame_video_is_accepted_without_decoding(self):
        adapter, session = self._session()
        media = {
            "type": "video",
            "id": "v",
            "uri": "/missing/video.mp4",
            "clip": {"start": 2.5, "end": 2.5},
        }
        request = RealtimeVideoGenerationsRequest(
            type="init", prompt="scene", zing_request=_custom_request([media])
        )
        asyncio.run(adapter.on_init(session, request))
        session.set_request(request)
        adapter.ingest_event(
            session,
            RealtimeEvent(type="event", kind="prompt", payload="next", event_id=9),
        )
        self.assertEqual(session.adapter_state.pending_prompt, ("next", 9))

    def test_t2_source_rejects_removed_user_media(self):
        from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters import (
            zing_h3_source_adapter as source,
        )

        request = RealtimeVideoGenerationsRequest(
            type="init",
            prompt="scene",
            zing_realtime_api_version=1,
            source_stream={"format": "rgb24", "width": 32, "height": 32, "fps": 24},
            size="32x32",
            user_media=[],
        )
        session = GenerateSession()
        adapter = source.ZingH3SourceStreamAdapter()
        session.set_adapter(adapter)
        with (
            patch.object(source, "get_global_server_args", return_value=self.args),
            self.assertRaises(ConditionError),
        ):
            asyncio.run(adapter.on_init(session, request))


class TestConditioningRPCBrowserErrors(CustomTestCase):
    def test_midstream_notice_preserves_update_event_id_in_browser_message(self):
        api = importlib.import_module(
            "sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_video_api"
        )
        result = OutputBatch()
        result.conditioning_notice = {
            "code": "CONDITION_BUSY",
            "event_id": 11,
            "detail": "private encoder host secret",
        }
        websocket = _Socket()
        asyncio.run(api._write_conditioning_notice(websocket, result))
        self.assertEqual(
            websocket.messages,
            [
                {
                    "type": "error",
                    "code": "CONDITION_BUSY",
                    "content": _PUBLIC["CONDITION_BUSY"],
                    "fatal": False,
                    "event_id": 11,
                }
            ],
        )

    def _worker_error(self):
        worker_module, _ = _managers()
        worker = object.__new__(worker_module.GPUWorker)
        worker.server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(supports_auto_residency=False)
        )
        worker._realtime_sessions = SimpleNamespace(attach=Mock(), touch=Mock())
        worker.is_output_rank = False
        worker._release_warmup_pool = Mock()
        worker._record_output_peak_memory = Mock()

        def fail():
            raise ConditionError("CONDITION_BUSY", "private GPU host secret")

        with patch.object(worker_module.current_platform, "is_cpu", return_value=True):
            return worker._execute_forward_common(
                _req(),
                forward_fn=fail,
                log_reqs=[],
                return_req=False,
                save_output_paths=Mock(),
                error_context="conditioning",
            )

    def test_t21_worker_exception_retains_public_code(self):
        result = self._worker_error()
        self.assertEqual(getattr(result, "error_code", None), "CONDITION_BUSY")

    def test_t21_real_process_generation_batch_raises_condition_error(self):
        from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
            process_generation_batch,
        )

        result = self._worker_error()
        with self.assertRaises(ConditionError) as raised:
            asyncio.run(
                process_generation_batch(
                    SimpleNamespace(forward=AsyncMock(return_value=result)), _req()
                )
            )
        self.assertEqual(raised.exception.code, "CONDITION_BUSY")
        self.assertEqual(raised.exception.public_message, _PUBLIC["CONDITION_BUSY"])

    def _run_loop(self, async_vae, outputs):
        api = importlib.import_module(
            "sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_video_api"
        )
        from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
            BaseRealtimeModelAdapter,
        )

        request = _req(
            realtime_session_id="s",
            realtime_event_id=11,
            realtime_prompt_version=3,
            block_idx=0,
        )

        class Adapter(BaseRealtimeModelAdapter):
            def create_state(self):
                return None

            async def wait_for_next_chunk(self, session):
                await asyncio.sleep(0)

            async def prepare_conditioning(self, session, args):
                pass

            def prepare_next_request(self, session, args, chunk):
                request.block_idx = chunk.index
                return request

            def on_chunk_complete(self, session, result):
                super().on_chunk_complete(session, result)

            def session_ready_message(self, session):
                return {"type": "ready"}

        class VAEClient:
            def __init__(self, *args, **kwargs):
                pass

            async def open(self, **kwargs):
                pass

            async def close(self):
                pass

            async def submit(self, latents, handoff, **kwargs):
                async def wait():
                    completed.append(
                        (request.realtime_event_id, request.realtime_prompt_version)
                    )
                    return SimpleNamespace(
                        completed_at=time.perf_counter(),
                        first_frame_ms=None,
                        num_frames=1,
                        **{
                            name: 0.0
                            for name in (
                                "queue_wait_ms",
                                "decode_ms",
                                "post_decode_ms",
                                "encode_ms",
                                "transfer_ms",
                                "serialize_ms",
                                "latent_send_ms",
                                "credit_wait_ms",
                            )
                        },
                    )

                return SimpleNamespace(
                    wait=wait, serialize_ms=0.0, latent_send_ms=0.0, credit_wait_ms=0.0
                )

        session = GenerateSession()
        session.set_adapter(Adapter())
        session.set_request(
            RealtimeVideoGenerationsRequest(
                type="init", prompt="scene", max_chunks=len(outputs)
            )
        )
        args = SimpleNamespace(
            batching_max_size=1,
            realtime_vae_worker_url="http://vae.invalid",
            realtime_vae_backend="exact_remote",
        )
        ws = _Socket()
        completed = []

        async def send_output(ws, session, chunk, batch, result, *args):
            # Output transport is external I/O. Preserve the actual batch
            # metadata observed after scheduler and API synchronization.
            completed.append((batch.realtime_event_id, batch.realtime_prompt_version))

        scheduler_client = SimpleNamespace(forward=AsyncMock(side_effect=outputs))
        with (
            patch.object(api, "async_scheduler_client", scheduler_client),
            patch.object(api, "get_global_server_args", return_value=args),
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.utils.get_global_server_args",
                return_value=args,
            ),
            patch.object(api, "RealtimeVAEClient", VAEClient),
            patch.object(api, "_send_output_and_log", send_output),
        ):
            if async_vae:
                asyncio.run(api._generate_loop_async_vae(ws, session, args))
            else:
                asyncio.run(api._generate_loop_local(ws, session))
        return ws.messages, completed, scheduler_client.forward.await_count

    def test_t21_both_real_loops_send_fatal_public_error(self):
        for async_vae in (False, True):
            with self.subTest(async_vae=async_vae):
                # The real worker produces the exception; the real scheduler
                # reconstructs each response before the real API sees it.
                _, scheduler = _managers()
                output = self._worker_error()
                result = object.__new__(
                    scheduler.Scheduler
                )._build_dynamic_batch_error_outputs([_req()], output)[0]
                messages, completed, calls = self._run_loop(async_vae, [result])
                self.assertEqual(calls, 1)
                self.assertFalse(completed)
                self.assertEqual(
                    messages,
                    [
                        {
                            "type": "error",
                            "code": "CONDITION_BUSY",
                            "content": _PUBLIC["CONDITION_BUSY"],
                            "fatal": True,
                        }
                    ],
                )

    def test_midstream_notice_is_nonfatal_and_output_continues(self):
        _, scheduler = _managers()
        for async_vae in (False, True):
            with self.subTest(async_vae=async_vae):
                outputs = []
                for index in range(4):
                    output = OutputBatch(raw_frame_batches=[[b"frame"]])
                    if async_vae:
                        output.realtime_latents = "external VAE input"
                        output.realtime_handoff = {"chunk_index": index}
                    output.realtime_effective_prompt = {
                        "event_id": 7,
                        "prompt_version": 2,
                    }
                    output.conditioning_notice = (
                        {
                            "code": "CONDITION_TIMEOUT",
                            "event_id": 11,
                            "detail": "private encoder host secret",
                        }
                        if index == 0
                        else None
                    )
                    scheduler.Scheduler._attach_realtime_request_metadata(
                        output,
                        _req(
                            realtime_session_id="s",
                            realtime_event_id=11,
                            realtime_prompt_version=3,
                        ),
                    )
                    outputs.append(output)
                messages, completed, calls = self._run_loop(async_vae, outputs)
                errors = [
                    message for message in messages if message.get("type") == "error"
                ]
                self.assertEqual(calls, 4)
                self.assertEqual(completed, [(7, 2)] * 4)
                self.assertEqual(
                    errors,
                    [
                        {
                            "type": "error",
                            "code": "CONDITION_TIMEOUT",
                            "content": _PUBLIC["CONDITION_TIMEOUT"],
                            "fatal": False,
                            "event_id": 11,
                        }
                    ],
                )


class TestConditioningRPCFrameHeaders(CustomTestCase):
    def test_t24_stable_frame_headers_use_each_effective_prompt_pair(self):
        from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
            RawRGBRealtimeOutputAdapter,
        )

        session = GenerateSession()
        request = _req(
            block_idx=4,
            request_id="current",
            realtime_event_id=19,
            realtime_prompt_version=5,
            realtime_output_format="raw",
        )
        output = OutputBatch(
            raw_frame_batches=[[b"\x01\x02\x03"], [b"\x04\x05\x06"]],
            raw_frame_metadata={
                "width": 1,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 3,
            },
            usage={
                "zing_realtime_blocks": [
                    {
                        "block_idx": 4,
                        "request_id": "old",
                        "event_id": 7,
                        "prompt_version": 2,
                    },
                    {
                        "block_idx": 5,
                        "request_id": "new",
                        "event_id": 13,
                        "prompt_version": 3,
                    },
                ]
            },
        )
        ws = _Socket()
        asyncio.run(RawRGBRealtimeOutputAdapter().send(ws, session, output, request))
        self.assertEqual(
            [
                (
                    message["chunk_index"],
                    message["request_id"],
                    message.get("event_id"),
                    message.get("prompt_version"),
                )
                for message in ws.messages
            ],
            [(4, "old", 7, 2), (5, "new", 13, 3)],
        )
        self.assertEqual(
            [message["payload"] for message in ws.messages],
            [b"\x01\x02\x03", b"\x04\x05\x06"],
        )

    def test_t24_source_frame_headers_keep_source_snapshot_pair_and_provenance(self):
        from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
            RawRGBRealtimeOutputAdapter,
        )
        from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
            PixelRange,
            Provenance,
            SourceSpan,
        )

        provenance = Provenance(PixelRange(0, 1), (SourceSpan(1, 0, 0, 1),), 1)
        block = {
            "block_idx": 4,
            "request_id": "source",
            "event_id": 7,
            "prompt_version": 2,
            "provenance": provenance,
        }
        request = _req(
            block_idx=4,
            request_id="current",
            realtime_event_id=19,
            realtime_prompt_version=5,
            realtime_output_format="raw",
            fps=24,
        )
        request.extra["zing_source_output_blocks"] = [block]
        output = OutputBatch(
            raw_frame_batches=[[b"\x01\x02\x03"]],
            raw_frame_metadata={
                "width": 1,
                "height": 1,
                "channels": 3,
                "bytes_per_frame": 3,
            },
            usage={
                "zing_realtime_blocks": [
                    {
                        "block_idx": 9,
                        "request_id": "wrong",
                        "event_id": 99,
                        "prompt_version": 99,
                    }
                ]
            },
        )
        ws = _Socket()
        asyncio.run(
            RawRGBRealtimeOutputAdapter().send(ws, GenerateSession(), output, request)
        )
        self.assertEqual(
            (
                ws.messages[0]["chunk_index"],
                ws.messages[0]["request_id"],
                ws.messages[0]["event_id"],
                ws.messages[0].get("prompt_version"),
            ),
            (4, "source", 7, 2),
        )
        for key, value in provenance.metadata().items():
            self.assertEqual(ws.messages[0][key], value)


class TestConditioningRPCFiniteErrors(CustomTestCase):
    def test_t25b_background_finite_job_uses_public_error_status(self):
        api = importlib.import_module(
            "sglang.multimodal_gen.runtime.entrypoints.openai.video_api"
        )
        scheduler_client = importlib.import_module(
            "sglang.multimodal_gen.runtime.scheduler_client"
        )
        from sglang.multimodal_gen.runtime.entrypoints.openai.stores import (
            AsyncDictStore,
        )

        for code, expected_status in (
            ("CONDITION_INVALID", 400),
            ("CONDITION_BUSY", 503),
            ("CONDITION_TIMEOUT", 503),
            ("CONDITION_UNAVAILABLE", 503),
        ):
            with self.subTest(code=code):
                output = OutputBatch(error="private worker host secret")
                output.error_code = code
                request = _req()
                request.sampling_params.cleanup_video_request = Mock()
                store = AsyncDictStore()

                async def run():
                    await store.upsert("job", {"status": "queued"})
                    await api._dispatch_job_async("job", request)
                    return await store.get("job")

                with (
                    patch.object(api, "VIDEO_STORE", store),
                    patch.object(
                        scheduler_client,
                        "async_scheduler_client",
                        SimpleNamespace(forward=AsyncMock(return_value=output)),
                    ),
                ):
                    job = asyncio.run(run())
                self.assertEqual(job["status"], "failed")
                self.assertEqual(
                    job["error"],
                    {
                        "code": code,
                        "message": _PUBLIC[code],
                        "status_code": expected_status,
                    },
                )

    def test_t25b_create_video_maps_prepare_condition_error_to_http(self):
        from fastapi import HTTPException
        from starlette.requests import Request

        api = importlib.import_module(
            "sglang.multimodal_gen.runtime.entrypoints.openai.video_api"
        )
        payload = json.dumps({"prompt": "scene"}).encode()

        async def receive():
            return {"type": "http.request", "body": payload, "more_body": False}

        for code, expected_status in (
            ("CONDITION_INVALID", 400),
            ("CONDITION_BUSY", 503),
        ):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                args = SimpleNamespace(
                    zing_conditioning_mode="remote",
                    input_save_path=directory,
                    output_path=directory,
                    pipeline_config=SimpleNamespace(
                        task_type=SimpleNamespace(requires_image_input=lambda: False)
                    ),
                )
                sampling = SimpleNamespace(
                    prepare_video_request_for_queue=Mock(
                        side_effect=ConditionError(code, "private encoder host secret")
                    ),
                    cleanup_video_request=Mock(),
                )
                request = Request(
                    {
                        "type": "http",
                        "headers": [(b"content-type", b"application/json")],
                    },
                    receive,
                )
                # Request materialization and model preparation are the
                # external model boundary; create_video error mapping is real.
                with (
                    patch.object(api, "get_global_server_args", return_value=args),
                    patch.object(
                        api, "_build_video_sampling_params", return_value=sampling
                    ),
                    patch.object(
                        api, "prepare_request", return_value=SimpleNamespace(extra={})
                    ),
                    self.assertRaises(HTTPException) as raised,
                ):
                    asyncio.run(api.create_video(request))
                self.assertEqual(raised.exception.status_code, expected_status)
                self.assertEqual(
                    raised.exception.detail, {"code": code, "message": _PUBLIC[code]}
                )


@pytest.mark.parametrize("limit", [0, 2, 32])
def test_remote_metadata_uses_pipeline_media_count(limit):
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.conditioning import (
        validate_remote_request,
    )

    configured = SimpleNamespace(
        zing_conditioning_mode="remote",
        pipeline_config=SimpleNamespace(zing_reference_max_images=limit),
    )
    media = [dict(type="video", clip=dict(start=0, end=0))] + [
        dict(type="image") for _ in range(limit)
    ]
    request = SimpleNamespace(zing_request=_custom_request(media))
    validate_remote_request(request, configured)
    with pytest.raises(ConditionError):
        validate_remote_request(
            SimpleNamespace(zing_request=_custom_request([*media, dict(type="image")])),
            configured,
        )
