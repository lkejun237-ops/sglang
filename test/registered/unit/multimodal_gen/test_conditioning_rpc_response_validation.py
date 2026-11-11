# SPDX-License-Identifier: Apache-2.0
"""Malformed RPC response headers must preserve INVALID at the worker boundary."""

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
    # Prepare the CPU import boundaries without starting a nested test run.
    runpy.run_path(str(runner))
    import pytest

    raise SystemExit(pytest.main([str(test_file), *arguments]))

from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.conditioning.client import ConditioningClient
from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
from sglang.multimodal_gen.runtime.conditioning.inputs import content_identity
from sglang.multimodal_gen.runtime.conditioning.wire import pack_message
from sglang.multimodal_gen.runtime.conditioning.worker import (
    RemoteConditioner,
    WorkerConditioningState,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PROFILE_ID = "a" * 64


def response_worker(updates):
    digest = content_identity("p", [], PROFILE_ID)
    tensors = {
        "context": torch.zeros((1, 5120), dtype=torch.bfloat16),
        "row_role": torch.zeros(1, dtype=torch.int64),
    }
    header = dict(
        schema_version=1,
        profile_id=PROFILE_ID,
        input_digest=digest,
        cache="miss",
        encode_ms=0,
        queue_ms=0,
        tensors=["context", "row_role"],
    )
    header.update(updates)
    body = pack_message(header, tensors)
    calls = []

    def transport(*args):
        calls.append(args)
        return SimpleNamespace(status=200, body=body)

    client = ConditioningClient(
        "http://localhost", "dummy", PROFILE_ID, transport=transport
    )
    return RemoteConditioner(client, PROFILE_ID, rank=0), digest, tensors, calls


class TestConditioningResponseValidation(CustomTestCase):
    def assert_invalid_update(self, updates):
        for has_previous in (False, True):
            with self.subTest(has_previous=has_previous):
                worker, _, _, calls = response_worker(updates)
                state = WorkerConditioningState()
                if has_previous:
                    state.identity = content_identity("previous", [], PROFILE_ID)
                    state.positive = {
                        "context": torch.ones((1, 5120), dtype=torch.bfloat16),
                        "row_role": torch.zeros(1, dtype=torch.int64),
                    }
                previous_identity, previous_positive = state.identity, state.positive
                with self.assertRaises(ConditionError) as caught:
                    worker("p", [], device="cpu", state=state)
                self.assertEqual(state.identity, previous_identity)
                self.assertIs(state.positive, previous_positive)
                self.assertEqual(len(calls), 1)
                self.assertEqual(caught.exception.code, "CONDITION_INVALID")

    def test_cache_list_is_invalid(self):
        self.assert_invalid_update({"cache": []})

    def test_cache_object_is_invalid(self):
        self.assert_invalid_update({"cache": {}})

    def test_unrepresentable_encode_timing_is_invalid(self):
        self.assert_invalid_update({"encode_ms": 10**400})

    def test_unrepresentable_queue_timing_is_invalid(self):
        self.assert_invalid_update({"queue_ms": 10**400})

    def test_valid_cache_and_finite_timing_preserve_response(self):
        for cache in ("hit", "miss", "joined"):
            for timing in (0, 0.25, 10**100):
                with self.subTest(cache=cache, timing=timing):
                    worker, digest, expected, calls = response_worker(
                        {"cache": cache, "encode_ms": timing, "queue_ms": timing}
                    )
                    state = WorkerConditioningState()
                    result = worker("p", [], device="cpu", state=state)
                    self.assertEqual(state.identity, digest)
                    self.assertIs(state.positive, result)
                    self.assertEqual(len(calls), 1)
                    for name, tensor in expected.items():
                        self.assertEqual(result[name].dtype, tensor.dtype)
                        self.assertTrue(torch.equal(result[name], tensor))
