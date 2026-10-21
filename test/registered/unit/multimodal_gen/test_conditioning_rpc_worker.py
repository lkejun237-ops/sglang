# SPDX-License-Identifier: Apache-2.0
"""Issue #256 v6 worker contracts; CPU receipts replace GPU/HTTP boundaries."""

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


import importlib
import importlib.util
import runpy
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import MISSING, fields
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

BASE = "sglang.multimodal_gen.runtime.conditioning"
STAGES = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3"
)
PROFILE = "a" * 64
DIGEST = "b" * 64


def condition(value=0):
    return {
        "context": torch.full((2, 5120), value, dtype=torch.bfloat16),
        "row_role": torch.tensor([0, 1], dtype=torch.int64),
    }


def media(value=0):
    return [
        {"type": "image", "frames": torch.full((3, 1, 2, 2), value, dtype=torch.uint8)}
    ]


class ReceiptBus:
    """Two CPU ranks synchronize actual provider logic at each receipt boundary."""

    def __init__(self, mutate=None):
        self.barrier = threading.Barrier(2, timeout=5)
        self.values = {}
        self.events = []
        self.mutate = mutate
        self.response = None
        self.tensor = None

    def gather(self, rank, receipt):
        self.values[rank] = receipt
        self.events.append((rank, "receipt", dict(receipt)))
        self.barrier.wait()
        result = [dict(self.values[i]) for i in range(2)]
        if self.mutate:
            result = self.mutate(result)
        self.barrier.wait()
        return result

    def response_broadcast(self, rank, value):
        if rank == 0:
            self.response = value
        self.events.append((rank, "response", value))
        self.barrier.wait()
        result = self.response
        self.barrier.wait()
        return result

    def tensor_broadcast(self, rank, tensor, name):
        self.events.append((rank, "gpu", name))
        if rank == 0:
            self.tensor = tensor.clone()
        self.barrier.wait()
        tensor.copy_(self.tensor)
        self.barrier.wait()
        return tensor


class TestConditioningRPCWorker(CustomTestCase):
    def api(self, module, name):
        # Missing APIs fail an explicit design assertion, never collection/import.
        spec = importlib.util.find_spec(BASE + "." + module)
        self.assertIsNotNone(spec, f"v6 requires conditioning/{module}.py")
        result = getattr(importlib.import_module(BASE + "." + module), name, None)
        self.assertTrue(callable(result), f"v6 requires {module}.{name}")
        return result

    def client(self, transport, **options):
        cls = self.api("client", "ConditioningClient")
        return cls(
            "https://conditioning",
            "private-token",
            PROFILE,
            transport=transport,
            timeout_s=10,
            sp_timeout_s=30,
            max_rows=16,
            **options,
        )

    def response(self, *, header=None, tensors=None):
        pack = self.api("wire", "pack_message")
        values = dict(
            schema_version=1,
            profile_id=PROFILE,
            input_digest=DIGEST,
            cache="miss",
            encode_ms=1,
            queue_ms=0,
            tensors=["context", "row_role"],
        )
        values.update(header or {})
        return NS(
            status=200, body=pack(values, condition() if tensors is None else tensors)
        )

    def test_t4_response_contract_rejects_profile_digest_shape_dtype_and_nonfinite(
        self,
    ):
        cls = self.api("client", "ConditioningClient")
        corruptions = [
            ({"profile_id": "c" * 64}, condition()),
            ({"input_digest": "c" * 64}, condition()),
            ({}, {**condition(), "context": torch.zeros(2, 5120)}),
            (
                {},
                {**condition(), "context": torch.zeros(2, 5119, dtype=torch.bfloat16)},
            ),
            (
                {},
                {
                    **condition(),
                    "context": torch.zeros(17, 5120, dtype=torch.bfloat16),
                    "row_role": torch.zeros(17, dtype=torch.int64),
                },
            ),
            ({}, {**condition(), "row_role": torch.zeros(2, dtype=torch.int32)}),
            ({}, {**condition(), "row_role": torch.zeros(1, dtype=torch.int64)}),
            (
                {},
                {
                    **condition(),
                    "context": torch.full(
                        (2, 5120), float("nan"), dtype=torch.bfloat16
                    ),
                },
            ),
            (
                {},
                {
                    **condition(),
                    "context": torch.full(
                        (2, 5120), float("inf"), dtype=torch.bfloat16
                    ),
                },
            ),
            ({}, {**condition(), "extra": torch.zeros(1)}),
        ]
        for header, tensors in corruptions:
            with self.subTest(
                header=header, shapes={k: tuple(v.shape) for k, v in tensors.items()}
            ):
                response = self.response(header=header, tensors=tensors)
                calls = []
                client = cls(
                    "https://conditioning",
                    "token",
                    PROFILE,
                    max_rows=16,
                    transport=lambda *args: calls.append(args) or response,
                )
                with self.assertRaises(ConditionError) as raised:
                    client.encode("prompt", media(), input_digest=DIGEST)
                self.assertEqual(raised.exception.code, "CONDITION_INVALID")
                self.assertEqual(len(calls), 1)

    def test_identity_changes_when_an_inference_tensor_is_rewritten_in_place(self):
        """Rewriting the same inference-mode buffer must never reuse the old digest."""
        worker = importlib.import_module(BASE + ".worker")
        state = worker.WorkerConditioningState()
        with torch.inference_mode():
            frames = torch.zeros(3, 1, 2, 2, dtype=torch.uint8)
        items = [{"type": "image", "frames": frames}]

        def identity():
            return worker._identity("p", items, PROFILE, state, "rgb-v1")

        first = identity()
        with torch.inference_mode():
            frames.fill_(7)
        self.assertNotEqual(identity(), first)

    def test_t14_profile_handshake_and_cpu_group_timeout_are_startup_gates(self):
        self.api("client", "ConditioningClient")
        client = self.client(lambda *a: NS(status=200, body=b'{"profile_id":"wrong"}'))
        with self.assertRaises(ConditionError) as raised:
            client.handshake()
        self.assertEqual(raised.exception.code, "CONDITION_PROFILE_MISMATCH")
        cls = self.api("client", "ConditioningClient")
        for timeout in (10, 11):
            with (
                self.subTest(timeout=timeout),
                self.assertRaises((ValueError, ConditionError)),
            ):
                cls(
                    "https://conditioning",
                    "token",
                    PROFILE,
                    timeout_s=timeout,
                    sp_timeout_s=10,
                )

    def ranks(
        self,
        *,
        fail_rpc=None,
        fail_allocate=None,
        mutate=None,
        prompts=None,
        cached=None,
    ):
        cls = self.api("worker", "RemoteConditioner")
        state_cls = self.api("worker", "WorkerConditioningState")
        bus, calls = ReceiptBus(mutate), []
        providers, states = [], [state_cls(), state_cls()]
        if cached:
            for state, is_cached in zip(states, cached):
                if is_cached:
                    state.identity = DIGEST
                    state.positive = condition()

        def encode(prompt, items, *, input_digest):
            calls.append((prompt, input_digest))
            if fail_rpc:
                raise ConditionError(fail_rpc)
            return condition(7)

        for rank in range(2):

            def allocate(shapes, device, positive, rank=rank):
                bus.events.append((rank, "allocate", None))
                if rank == fail_allocate:
                    raise MemoryError("Synthetic allocation failed")
                return {
                    name: (
                        positive[name].clone()
                        if positive is not None
                        else torch.empty(
                            shape,
                            dtype=torch.bfloat16 if name == "context" else torch.int64,
                        )
                    )
                    for name, shape in shapes.items()
                }

            providers.append(
                cls(
                    NS(encode=encode),
                    PROFILE,
                    rank=rank,
                    coordinate=lambda receipt, rank=rank: bus.gather(rank, receipt),
                    broadcast_receipt=lambda receipt, rank=rank: bus.response_broadcast(
                        rank, receipt
                    ),
                    allocate=allocate,
                    broadcast_tensor=lambda tensor, name, rank=rank: (
                        bus.tensor_broadcast(rank, tensor, name)
                    ),
                )
            )

        def run(rank):
            try:
                return providers[rank](
                    (prompts or ["prompt", "prompt"])[rank],
                    media(),
                    device="cpu",
                    state=states[rank],
                )
            except ConditionError as exc:
                return exc.code

        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(run, range(2)))
        return results, states, calls, bus.events

    def test_t15_busy_is_shared_before_allocation_or_gpu_broadcast(self):
        result, states, calls, events = self.ranks(fail_rpc="CONDITION_BUSY")
        self.assertEqual(result, ["CONDITION_BUSY"] * 2)
        self.assertEqual(len(calls), 1)
        self.assertFalse(any(kind in {"allocate", "gpu"} for _, kind, _ in events))
        self.assertTrue(all(state.positive is None for state in states))

    def test_t15b_any_rank_allocation_failure_prevents_gpu_collectives(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                results, states, calls, events = self.ranks(fail_allocate=rank)
                self.assertEqual(results, ["CONDITION_UNAVAILABLE"] * 2)
                self.assertEqual(len(calls), 1)
                self.assertFalse(any(kind == "gpu" for _, kind, _ in events))
                self.assertTrue(all(state.positive is None for state in states))

    def test_t15c_digest_or_cache_receipt_disagreement_rejects_before_rpc(self):
        for mutate in (
            lambda values: [values[0], {**values[1], "digest": "c" * 64}],
            lambda values: [values[0], {**values[1], "cached": True}],
        ):
            result, _, calls, events = self.ranks(mutate=mutate)
            self.assertEqual(result, ["CONDITION_INVALID"] * 2)
            self.assertEqual(calls, [])
            self.assertFalse(any(kind == "gpu" for _, kind, _ in events))

    def test_t15c_rank_local_validation_failure_still_joins_cpu_receipt(self):
        for prompts in (["rank-zero", "rank-one"], [None, "valid"]):
            with self.subTest(prompts=prompts):
                results, states, calls, events = self.ranks(prompts=prompts)
                self.assertEqual(results, ["CONDITION_INVALID"] * 2)
                self.assertEqual(calls, [])
                self.assertEqual(sum(kind == "receipt" for _, kind, _ in events), 2)
                self.assertFalse(
                    any(kind in {"allocate", "gpu"} for _, kind, _ in events)
                )

    def test_t16_success_has_three_cpu_stages_and_one_rpc(self):
        results, states, calls, events = self.ranks()
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(results[0]["context"], results[1]["context"]))
        self.assertEqual(results[0]["context"][0, 0].item(), 7)
        for rank in range(2):
            sequence = [kind for r, kind, _ in events if r == rank]
            self.assertEqual(
                sequence[:4], ["receipt", "response", "allocate", "receipt"]
            )
            self.assertEqual(sequence[4:], ["gpu", "gpu"])
            self.assertIs(states[rank].positive, results[rank])

    def test_t17_cached_chunk_avoids_rpc_transfer_and_frame_rehash(self):
        cls = self.api("worker", "RemoteConditioner")
        state = self.api("worker", "WorkerConditioningState")()
        calls = []
        provider = cls(
            NS(encode=lambda *a, **k: calls.append(k["input_digest"]) or condition()),
            PROFILE,
        )
        items = media()
        first = provider("old", items, device="cpu", state=state)
        # The inner media hash helper is a CPU boundary, not the conditioner.
        hash_frame = self.api("worker", "hash_frame")
        with patch(BASE + ".worker.hash_frame", wraps=hash_frame) as hashed:
            second = provider("old", items, device="cpu", state=state)
            third = provider("new", items, device="cpu", state=state)
            self.assertEqual(hashed.call_count, 0)
        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0], calls[1])

    def test_t17b_changed_prompt_media_or_order_causes_new_rpc(self):
        cls = self.api("worker", "RemoteConditioner")
        state = self.api("worker", "WorkerConditioningState")()
        calls = []
        provider = cls(
            NS(
                encode=lambda *a, **k: (
                    calls.append(k["input_digest"]) or condition(len(calls))
                )
            ),
            PROFILE,
        )
        items = media(0) + media(1)
        values = [
            provider("old", items, device="cpu", state=state),
            provider("new", items, device="cpu", state=state),
        ]
        changed = [items[0], media(2)[0]]
        values.append(provider("new", changed, device="cpu", state=state))
        values.append(
            provider("new", list(reversed(changed)), device="cpu", state=state)
        )
        self.assertEqual(len(set(calls)), 4)
        self.assertEqual([x["context"][0, 0].item() for x in values], [1, 2, 3, 4])

    def test_t19_remote_pipeline_filters_encoder_before_parent_loader(self):
        from sglang.multimodal_gen.runtime.pipelines.zing_h3_causal_pipeline import (
            ZingH3CausalPipeline,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
            ComposedPipelineBase,
        )

        seen = []

        def load(self, *args, **kwargs):
            seen.extend(kwargs["required_config_modules"])
            self.modules = {}

        args = NS(
            zing_conditioning_mode="remote",
            component_paths={},
            component_weights_paths={},
            component_direct_gpu_weight_loading={},
        )
        with patch.object(ComposedPipelineBase, "__init__", load):
            ZingH3CausalPipeline("trimmed-export", args)
        self.assertEqual(set(seen), {"video_vae", "audio_vae", "transformer"})

    def args(self, mode, **overrides):
        from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs

        result = object.__new__(ServerArgs)
        for item in fields(ServerArgs):
            if item.default is not MISSING:
                setattr(result, item.name, item.default)
            elif item.default_factory is not MISSING:
                setattr(result, item.name, item.default_factory())
        result.zing_conditioning_mode = mode
        result.conditioning_expected_profile = "profile.json"
        result.conditioning_cache_root = "legacy-root"
        result.conditioning_service_url = "https://conditioning"
        result.conditioning_config = "client.json"
        result.pipeline_config = NS(validate_server_args=lambda args: None)
        for name in (
            "_validate_scheduler_rpc_timeout",
            "_validate_pipeline",
            "_validate_offload",
            "_validate_direct_gpu_weight_loading",
            "_validate_parallelism",
            "_validate_cfg_parallel",
            "_validate_batching",
            "_validate_breakable_cuda_graph",
            "_validate_minimax_h3_adaln",
        ):
            setattr(result, name, lambda: None)
        for name, value in overrides.items():
            setattr(result, name, value)
        return result

    def test_t20_artifact_is_rejected_and_local_cache_defaults_to_256_mib(self):
        args = self.args("artifact")
        with self.assertRaises(
            ValueError, msg="Removed artifact mode must be rejected"
        ):
            args._validate_parameters()
        self.assertEqual(
            getattr(self.args("local"), "zing_conditioning_local_cache_bytes", None),
            256 << 20,
        )

    def test_t20_remote_requires_service_profile_and_config(self):
        for name in (
            "conditioning_expected_profile",
            "conditioning_service_url",
            "conditioning_config",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.args("remote", **{name: None})._validate_parameters()
        self.args("remote", conditioning_cache_root=None)._validate_parameters()

    def test_t22_local_first_prepare_and_prompt_update_share_cache(self):
        local_cls = self.api("worker", "LocalConditioner")
        stage_cls = importlib.import_module(STAGES + ".stage").ZingH3GenerationStage
        codec_cls = importlib.import_module(STAGES + ".codecs").ZingH3Codecs
        for cache_bytes, expected in ((256 << 20, 1), (0, 2)):
            with self.subTest(cache_bytes=cache_bytes):
                encoded = []
                local = local_cls(
                    lambda p, m: encoded.append(p) or condition(3),
                    PROFILE,
                    cache_bytes=cache_bytes,
                )
                stage = object.__new__(stage_cls)
                stage.conditioning_mode = "local"
                stage.conditioner = lambda batch, prompt, items: local(prompt, items)
                stage.codecs = object.__new__(codec_cls)
                stage.codecs.conditioning_mode = "local"
                stage.codecs.max_area_config = {"image": 4096, "video": 4096}
                (
                    stage.codecs.spatial_ratio,
                    stage.codecs.target_fps,
                    stage.codecs.audio_sample_rate,
                ) = 8, 30, 24000
                stage.codecs._parse_inference_request = lambda req: (
                    "prompt",
                    [],
                    {"type": "video"},
                    [],
                    None,
                )
                stage.codecs._normalize_inference_media = lambda *a, **k: (
                    [],
                    None,
                    16,
                    16,
                    1,
                    1,
                )
                stage.codecs.encode_condition = lambda p, m: (
                    encoded.append("bypass") or condition(99)
                )
                stage._use = lambda *a: nullcontext()
                batch = NS(extra={}, session=None)
                first = stage.codecs.prepare(
                    {}, stage._use, conditioning=stage._conditioning_consumer(batch)
                )
                second = stage.encode_condition_for_batch(batch, "prompt", [])
                self.assertEqual(len(encoded), expected)
                self.assertEqual(first.positive["context"][0, 0].item(), 3)
                self.assertTrue(
                    torch.equal(first.positive["context"], second["context"])
                )

    def test_t22_cache_is_cpu_owned_and_consumer_mutation_cannot_poison_it(self):
        local_cls = self.api("worker", "LocalConditioner")
        calls = []
        local = local_cls(
            lambda p, m: calls.append(p) or condition(3), PROFILE, cache_bytes=256 << 20
        )
        first = local("prompt", [])
        first["context"].fill_(99)
        second = local("prompt", [])
        self.assertEqual(second["context"][0, 0].item(), 3)
        self.assertEqual(calls, ["prompt"])
        self.assertEqual(second["context"].device.type, "cpu")

    def test_t22_oversized_local_result_is_returned_without_retention(self):
        local_cls = self.api("worker", "LocalConditioner")
        calls = []
        local = local_cls(
            lambda p, m: calls.append(p) or condition(3), PROFILE, cache_bytes=1
        )
        first, second = local("prompt", []), local("prompt", [])
        self.assertTrue(torch.equal(first["context"], second["context"]))
        self.assertEqual(calls, ["prompt", "prompt"])

    def test_t23_two_busy_responses_retry_then_return_success_before_deadline(self):
        now, calls = [0.0], []
        success = self.response()

        def transport(*args):
            calls.append(args)
            return (
                NS(status=429, body=b'{"code":"CONDITION_BUSY","message":"busy"}')
                if len(calls) < 3
                else success
            )

        client = self.client(
            transport,
            clock=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
            jitter=lambda: 0.5,
        )
        result = client.encode("prompt", media(), input_digest=DIGEST)
        self.assertEqual(len(calls), 3)
        self.assertTrue(torch.equal(result["context"], condition()["context"]))
        self.assertLess(now[0], 10)

    def test_t25_invalid_unauthorized_and_profile_mismatch_are_not_retried(self):
        self.api("client", "ConditioningClient")
        for status, code in (
            (400, "CONDITION_INVALID"),
            (401, "CONDITION_UNAUTHORIZED"),
            (409, "CONDITION_PROFILE_MISMATCH"),
        ):
            with self.subTest(status=status):
                calls = []

                def transport(*args):
                    calls.append(args)
                    return NS(
                        status=status,
                        body=('{"code":"' + code + '","message":"failed"}').encode(),
                    )

                client = self.client(transport)
                with self.assertRaises(ConditionError) as raised:
                    client.encode("prompt", media(), input_digest=DIGEST)
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(len(calls), 1)

    def test_t23_retry_deadline_is_bounded_and_retains_last_busy_code(self):
        self.api("client", "ConditioningClient")
        now, calls = [0.0], []

        def transport(*args):
            calls.append(args)
            return NS(status=429, body=b'{"code":"CONDITION_BUSY","message":"busy"}')

        client = self.client(
            transport,
            clock=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
            jitter=lambda: 0.5,
        )
        with self.assertRaises(ConditionError) as raised:
            client.encode("prompt", media(), input_digest=DIGEST)
        self.assertEqual(raised.exception.code, "CONDITION_BUSY")
        self.assertGreater(len(calls), 1)
        self.assertLessEqual(now[0], 10)

    def test_t23_slow_drip_response_has_total_deadline_and_closes_socket(self):
        now, timeouts = [0.0], []
        import httpx

        class Response(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                while True:
                    now[0] += 2
                    yield b"x"

            async def aclose(self):
                self.closed = True

        response = Response()

        def http(request):
            timeouts.append(request.extensions["timeout"]["read"])
            return httpx.Response(200, stream=response)

        async_client = httpx.AsyncClient
        client = self.client(None, clock=lambda: now[0])
        with patch(
            BASE + ".client.httpx.AsyncClient",
            side_effect=lambda **kwargs: async_client(
                transport=httpx.MockTransport(http), **kwargs
            ),
        ):
            with self.assertRaises(ConditionError) as raised:
                client._transport("GET", "/v1/conditioning/profile", None, 10)
        self.assertEqual(raised.exception.code, "CONDITION_TIMEOUT")
        self.assertTrue(response.closed)
        self.assertLessEqual(now[0], 10)
        self.assertTrue(timeouts)
        self.assertEqual(timeouts, sorted(timeouts, reverse=True))

    def test_t23_cpu_response_decode_counts_against_total_deadline(self):
        now = [0.0]
        response = self.response()
        client = self.client(lambda *args: response, clock=lambda: now[0])
        decode = self.api("wire", "decode_response")

        def delayed(*args, **kwargs):
            result = decode(*args, **kwargs)
            now[0] += 11
            return result

        with patch(BASE + ".wire.decode_response", delayed):
            with self.assertRaises(ConditionError) as raised:
                client.encode("prompt", media(), input_digest=DIGEST)
        self.assertEqual(raised.exception.code, "CONDITION_TIMEOUT")

    def test_t23_serialization_and_retries_send_only_remaining_server_deadline(self):
        now, deadlines = [0.0], []
        response = self.response()
        encode = self.api("wire", "encode_request")
        unpack = self.api("wire", "unpack_message")

        def slow_encode(*args, **kwargs):
            value = encode(*args, **kwargs)
            now[0] += 2
            return value

        def transport(method, path, body, timeout):
            deadlines.append(unpack(body)[0]["deadline_ms"])
            now[0] += 1
            if len(deadlines) == 1:
                return NS(status=429, body=b'{"code":"CONDITION_BUSY"}')
            return response

        client = self.client(
            transport,
            clock=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
            jitter=lambda: 0.5,
        )
        with patch(BASE + ".wire.encode_request", slow_encode):
            client.encode("prompt", media(), input_digest=DIGEST)
        self.assertEqual(len(deadlines), 2)
        self.assertLessEqual(deadlines[0], 8000)
        self.assertLess(deadlines[1], deadlines[0])

    def block_fixture(self, mode="immediate"):
        stage_cls = importlib.import_module(STAGES + ".stage").ZingH3GenerationStage
        state = importlib.import_module(STAGES + ".realtime").ZingH3RealtimeState()
        stage = object.__new__(stage_cls)
        stage.conditioning_mode = "remote"
        stage._validate_sampling_params = lambda params: None
        stage.codecs = NS(
            decoder=importlib.import_module(STAGES + ".decoder").DECODERS[
                "official_h3"
            ],
            video_decoder_name="video_vae",
            vae_latents_per_chunk=1,
            vae_frames_per_chunk=1,
            temporal_ratio=1,
            target_fps=24,
            spatial_ratio=1,
            va=NS(vae_token_drop=0),
        )
        transformer = torch.nn.Linear(1, 1)
        stage._use = lambda *a: nullcontext(transformer)
        state.inputs = NS(
            positive=condition(1),
            condition_media=[],
            pixel_frame_count=50,
            visual_shape=(50, 1, 1),
        )
        state.prompt, state.prompt_event, state.effective_prompt_version = (
            "old",
            "old-event",
            1,
        )
        state.sampling = NS(
            index=1,
            spans=[(index, index, index, index + 1) for index in range(50)],
            visual=torch.zeros(1, 3, 50, 1, 1),
            close=lambda: None,
        )
        state.requested_prompt = state.prompt
        updates = []
        pending = []

        def update_prompt(value, event_id, version):
            updates.append(value)
            pending[:] = [(event_id, version)]

        def step():
            state.sampling.prompt_updated = bool(pending)
            state.sampling.index += 1
            event, version = (
                pending.pop()
                if pending
                else (state.prompt_event, state.effective_prompt_version)
            )
            return NS(
                chunk=NS(end=state.sampling.index),
                prompt_event_id=event,
                prompt_version=version,
            )

        state.sampling.step = step
        state.sampling.update_prompt = update_prompt
        state.sampling.discard_pending_prompt = pending.clear
        state.sampling.retag_pending_prompt = lambda event, version: (
            pending.__setitem__(slice(None), [(event, version)])
        )
        state.sampler = NS()
        stage.codecs.decode_visual_range = lambda latent, start, end: torch.zeros(
            1, 3, end - start, 1, 1
        )
        calls = []

        def encode(batch, prompt, items):
            calls.append((prompt, batch.realtime_event_id))
            if batch.realtime_event_id == "failed-event":
                raise ConditionError("CONDITION_BUSY")
            return condition(2)

        stage.encode_condition_for_batch = encode

        def batch(event="failed-event", prompt="new", version=2):
            return NS(
                sampling_params=NS(
                    prompt=prompt, seed=1, zing_realtime_decode_mode=mode
                ),
                realtime_event_id=event,
                realtime_prompt_version=version,
                realtime_session_id="session",
                block_idx=state.sampling.index,
                request_id="request",
                metrics=None,
                extra={},
            )

        return stage, state, batch, calls, updates

    def test_t24_block_failed_update_preserves_frames_condition_and_effective_metadata(
        self,
    ):
        stage, state, batch, calls, updates = self.block_fixture()
        outputs = []
        for _ in range(4):
            try:
                outputs.append(stage._generate_realtime_block(batch(), state))
            except ConditionError as exc:
                self.fail(
                    f"Mid-session {exc.code} must preserve the old condition and continue producing frames"
                )
        self.assertEqual(calls, [("new", "failed-event")])
        self.assertEqual(
            updates,
            [],
            "Failed updates must not enter the rolling-prefix pending queue",
        )
        self.assertEqual(
            (state.prompt, state.prompt_event, state.effective_prompt_version),
            ("old", "old-event", 1),
        )
        notices = [getattr(out, "conditioning_notice", None) for out in outputs]
        self.assertEqual(
            [notice for notice in notices if notice],
            [{"code": "CONDITION_BUSY", "event_id": "failed-event"}],
        )
        for out in outputs:
            self.assertGreater(out.output.shape[2], 0)
            self.assertEqual(
                getattr(out, "realtime_effective_prompt", None),
                {"event_id": "old-event", "prompt_version": 1},
            )
        retry = stage._generate_realtime_block(batch("retry-event", version=3), state)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(updates), 1)
        self.assertEqual(
            retry.realtime_effective_prompt,
            {"event_id": "retry-event", "prompt_version": 3},
        )
        # Same text is a new control event without redundant RPC.
        again = stage._generate_realtime_block(
            batch("same-text-event", version=4), state
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            again.realtime_effective_prompt,
            {"event_id": "same-text-event", "prompt_version": 4},
        )

    def test_t24_stable_pending_block_records_effective_event_after_failure(self):
        stage, state, batch, calls, updates = self.block_fixture("stable")
        try:
            stage._sample_realtime_block(batch(), state)
        except ConditionError as exc:
            self.fail(f"A stable block must survive {exc.code}")
        stage._queue_realtime_output(
            batch(), state, state.sampling.index, "stable", False
        )
        self.assertEqual(state.pending_blocks[-1]["event_id"], "old-event")
        self.assertEqual(state.pending_blocks[-1]["prompt_version"], 1)

    def source_fixture(self):
        # Reuse CPU source encoder/DiT fixtures; actual run/ledger/bridge logic executes.
        root = Path(__file__).resolve().parents[4]
        helper_path = (
            root
            / "python/sglang/multimodal_gen/test/unit/realtime/test_zing_progressive_source_session.py"
        )
        spec = importlib.util.spec_from_file_location(
            "source_cpu_fixtures", helper_path
        )
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        plan = helper.ChunkPlan(5)
        state, stage = helper.session(plan)
        state.inputs = NS(condition_media=[], positive=condition(1))
        state.prompt, state.prompt_event = "old", "old-event"
        state.sampler.prompt = NS(
            condition=state.inputs.positive, event_id="old-event", prompt_version=1
        )
        calls, updates = [], []

        def encode(batch, prompt, items):
            calls.append((prompt, batch.realtime_event_id))
            if batch.realtime_event_id == "failed-event":
                raise ConditionError("CONDITION_TIMEOUT")
            return condition(2)

        stage.encode_condition_for_batch = encode

        def update_prompt(positive, event, version=None):
            updates.append(event)
            state.sampler.prompt = NS(
                condition=positive, event_id=event, prompt_version=version
            )

        state.sampler.update_prompt = update_prompt
        original_step = state.sampler.step

        def step():
            # Output metadata belongs to the sampler's generation-time snapshot.
            prompt = state.sampler.prompt
            result = original_step()
            result.prompt_event_id = prompt.event_id
            result.prompt_version = prompt.prompt_version
            return result

        state.sampler.step = step

        # Decode is a GPU boundary. Consume real queued block metadata as RGB frames.
        def decode(stage, outputs, metadata, timed, trace=None, **kwargs):
            while state.blocks:
                block = state.blocks.popleft()
                outputs.append(torch.zeros(3, 1, 1, 1))
                metadata.append(block)

        state._decode_ready = decode
        return state, stage, helper, calls, updates

    def test_t24_source_failed_update_keeps_command_sequence_and_emits_one_notice(self):
        state, stage, helper, calls, updates = self.source_fixture()
        outputs = []
        for sequence in range(4):
            value = helper.batch(state, sequence * 17, 17, sequence)
            (
                value.sampling_params.prompt,
                value.realtime_event_id,
                value.realtime_prompt_version,
            ) = "new", "failed-event", 2
            try:
                outputs.append(state.run(stage, value))
            except ConditionError as exc:
                self.fail(
                    f"Source command must continue after {exc.code}, including append and command_index"
                )
            self.assertEqual(state.command_index, sequence + 1)
            self.assertFalse(state.closed)
            self.assertEqual(state.ledger.accepted_frames, (sequence + 1) * 17)
        self.assertEqual(calls, [("new", "failed-event")])
        self.assertEqual(updates, [])
        self.assertEqual((state.prompt, state.prompt_event), ("old", "old-event"))
        self.assertEqual(
            [
                out.conditioning_notice
                for out in outputs
                if getattr(out, "conditioning_notice", None)
            ],
            [{"code": "CONDITION_TIMEOUT", "event_id": "failed-event"}],
        )
        for out in outputs:
            self.assertTrue(out.output)
            self.assertTrue(
                all(
                    item["event_id"] == "old-event"
                    for item in out.usage["zing_realtime_blocks"]
                )
            )
            self.assertIsNone(getattr(out, "realtime_effective_prompt", None))
        retry = helper.batch(state, 68, 17, 4)
        (
            retry.sampling_params.prompt,
            retry.realtime_event_id,
            retry.realtime_prompt_version,
        ) = "new", "retry-event", 3
        result = state.run(stage, retry)
        self.assertEqual(calls[-1], ("new", "retry-event"))
        self.assertEqual(updates, ["retry-event"])
        self.assertEqual(state.command_index, 5)
        self.assertTrue(
            all(
                item["event_id"] == "retry-event"
                for item in result.usage["zing_realtime_blocks"]
            )
        )

    def test_t24_source_failed_newer_event_supersedes_pending_prompt(self):
        state, stage, helper, calls, updates = self.source_fixture()
        state.sampler.pending_prompt = ("encoded-prompt", "pending-event")
        discarded = []

        def discard_pending_prompt():
            discarded.append(state.sampler.pending_prompt)
            state.sampler.pending_prompt = None

        state.sampler.discard_pending_prompt = discard_pending_prompt
        value = helper.batch(state, 0, 17, 0)
        (
            value.sampling_params.prompt,
            value.realtime_event_id,
            value.realtime_prompt_version,
        ) = "new", "failed-event", 2

        output = state.run(stage, value)

        self.assertIsNone(state.sampler.pending_prompt)
        self.assertEqual(discarded, [("encoded-prompt", "pending-event")])
        self.assertEqual(calls, [("new", "failed-event")])
        self.assertEqual(updates, [])
        self.assertEqual(
            output.conditioning_notice,
            {"code": "CONDITION_TIMEOUT", "event_id": "failed-event"},
        )

    def test_t24_source_block_metadata_preserves_old_version_then_switches_on_success(
        self,
    ):
        state, stage, helper, calls, updates = self.source_fixture()
        for sequence, event, version, expected in (
            (0, "failed-event", 2, 1),
            (1, "retry-event", 3, 3),
        ):
            value = helper.batch(state, sequence * 17, 17, sequence)
            value.sampling_params.prompt = "new"
            value.realtime_event_id = event
            value.realtime_prompt_version = version
            out = state.run(stage, value)
            metadata = out.usage["zing_realtime_blocks"]
            self.assertTrue(metadata)
            self.assertTrue(
                all(item.get("prompt_version") == expected for item in metadata)
            )
            self.assertEqual(state.sampler.prompt.prompt_version, expected)
