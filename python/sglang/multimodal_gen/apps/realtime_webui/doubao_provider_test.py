#!/usr/bin/env python3

import unittest
from types import SimpleNamespace

from doubao_provider import DoubaoStructuredProvider
from prompt_rewriter import PromptRewriteOutput


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        message = SimpleNamespace(content=response)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class DoubaoProviderTest(unittest.IsolatedAsyncioTestCase):
    async def test_generates_strict_structured_output_without_thinking(self):
        completions = FakeCompletions(
            ['{"prompt":"A snowy valley remains explorable.","change_type":"persistent"}']
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        provider = DoubaoStructuredProvider(
            client=client,
            model="doubao-test",
            request_timeout_seconds=1,
        )
        result = await provider.generate_structured(
            system_instruction="Return the schema only.",
            user_content="Make the valley snowy.",
            schema=PromptRewriteOutput,
            schema_name="prompt_rewrite",
        )
        self.assertEqual(result.change_type.value, "persistent")
        call = completions.calls[0]
        self.assertEqual(call["model"], "doubao-test")
        self.assertEqual(call["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(call["response_format"]["type"], "json_schema")

    async def test_invalid_json_gets_one_structure_retry(self):
        completions = FakeCompletions(
            [
                "not-json",
                '{"prompt":"A lightning bolt flashes once.","change_type":"one_time"}',
            ]
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        provider = DoubaoStructuredProvider(
            client=client,
            request_timeout_seconds=1,
            max_attempts=2,
        )
        result = await provider.generate_structured(
            system_instruction="Return the schema only.",
            user_content="Release lightning.",
            schema=PromptRewriteOutput,
            schema_name="prompt_rewrite",
        )
        self.assertEqual(result.change_type.value, "one_time")
        self.assertEqual(len(completions.calls), 2)
        self.assertIn(
            "STRUCTURE CORRECTION",
            completions.calls[1]["messages"][1]["content"],
        )

    async def test_api_failure_is_not_retried(self):
        completions = FakeCompletions([RuntimeError("ark unavailable")])
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        provider = DoubaoStructuredProvider(
            client=client,
            request_timeout_seconds=1,
            max_attempts=2,
        )
        with self.assertRaisesRegex(RuntimeError, "ark unavailable"):
            await provider.generate_structured(
                system_instruction="Return the schema only.",
                user_content="Release lightning.",
                schema=PromptRewriteOutput,
                schema_name="prompt_rewrite",
            )
        self.assertEqual(len(completions.calls), 1)


if __name__ == "__main__":
    unittest.main()
