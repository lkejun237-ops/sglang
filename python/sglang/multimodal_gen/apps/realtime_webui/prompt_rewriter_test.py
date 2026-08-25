#!/usr/bin/env python3

import os
import unittest
from types import SimpleNamespace

from prompt_rewriter import (
    PromptRewriter,
    build_user_message,
    build_world_rule_message,
)


class FakeModels:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = list(
            responses
            or [
                {
                    "prompt": "A stable cinematic camera follows the dragon through falling snow.",
                    "change_type": "persistent",
                }
            ]
        )

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(parsed=response)


class FakeDoubaoProvider:
    configured = True

    def __init__(self):
        self.calls = []

    async def warmup(self):
        return None

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        schema = kwargs["schema"]
        if kwargs["schema_name"] == "world_rule":
            return schema.model_validate(
                {
                    "name": "召唤飞船",
                    "prompt": "A glowing spacecraft descends beside the rider.",
                    "change_type": "one_time",
                }
            )
        return schema.model_validate(
            {
                "prompt": "Snow continuously covers the same valley.",
                "change_type": "persistent",
            }
        )


class PromptRewriterTest(unittest.IsolatedAsyncioTestCase):
    async def test_doubao_provider_rewrites_and_completes_rules(self):
        provider = FakeDoubaoProvider()
        rewriter = PromptRewriter(
            provider="doubao",
            doubao_provider=provider,
        )
        rewrite = await rewriter.rewrite("持续下雪", "A rider explores a valley.")
        rule = await rewriter.complete_world_rule(
            "召唤飞船", "A rider explores a valley.", "skill"
        )
        self.assertEqual(rewrite.change_type.value, "persistent")
        self.assertEqual(rule.change_type.value, "one_time")
        self.assertEqual([call["schema_name"] for call in provider.calls], [
            "prompt_rewrite",
            "world_rule",
        ])

    async def test_rewrite_returns_prompt_and_lifetime(self):
        models = FakeModels()
        client = SimpleNamespace(aio=SimpleNamespace(models=models))
        rewriter = PromptRewriter(client=client, request_timeout_seconds=1)
        result = await rewriter.rewrite("让世界下雪", "A dragon flies over a valley.")
        self.assertEqual(result.change_type.value, "persistent")
        self.assertIn("falling snow", result.prompt)
        self.assertIn("让世界下雪", models.calls[0]["contents"][0])
        self.assertIn("A dragon flies over a valley.", models.calls[0]["contents"][0])
        self.assertEqual(models.calls[0]["model"], "gemini-3.1-flash-lite")
        self.assertEqual(len(models.calls), 1)

    async def test_forbidden_process_words_are_accepted_without_retry(self):
        models = FakeModels(
            [
                {
                    "prompt": "The original dragon retains its previous position.",
                    "change_type": "persistent",
                }
            ]
        )
        client = SimpleNamespace(aio=SimpleNamespace(models=models))
        result = await PromptRewriter(client=client).rewrite(
            "保持位置", "A dragon flies over a valley."
        )
        self.assertIn("original", result.prompt)
        self.assertIn("previous", result.prompt)
        self.assertEqual(len(models.calls), 1)

    async def test_invalid_schema_gets_one_structure_retry(self):
        models = FakeModels(
            [
                {"prompt": "Missing its lifetime label."},
                {"prompt": "A dragon jumps once.", "change_type": "one_time"},
            ]
        )
        client = SimpleNamespace(aio=SimpleNamespace(models=models))
        result = await PromptRewriter(client=client).rewrite(
            "跳一下", "A dragon flies over a valley."
        )
        self.assertEqual(result.change_type.value, "one_time")
        self.assertEqual(len(models.calls), 2)
        self.assertIn("STRUCTURE CORRECTION", models.calls[1]["contents"][0])

    async def test_api_failure_is_not_retried(self):
        models = FakeModels([RuntimeError("vertex unavailable")])
        client = SimpleNamespace(aio=SimpleNamespace(models=models))
        with self.assertRaisesRegex(RuntimeError, "vertex unavailable"):
            await PromptRewriter(client=client).rewrite(
                "跳一下", "A dragon flies over a valley."
            )
        self.assertEqual(len(models.calls), 1)

    async def test_world_rule_completion_returns_name_prompt_and_lifetime(self):
        models = FakeModels(
            [
                {
                    "name": "召唤飞船",
                    "prompt": "A glowing spacecraft descends beside the rider.",
                    "change_type": "one_time",
                }
            ]
        )
        client = SimpleNamespace(aio=SimpleNamespace(models=models))
        result = await PromptRewriter(client=client).complete_world_rule(
            "召唤飞船", "A rider explores a valley.", "skill"
        )
        self.assertEqual(result.name, "召唤飞船")
        self.assertEqual(result.change_type.value, "one_time")
        self.assertIn("RULE KIND: skill", models.calls[0]["contents"][0])
        self.assertIn("召唤飞船", models.calls[0]["contents"][0])
        self.assertEqual(len(models.calls), 1)

    async def test_local_provider_expands_chinese_skill_into_visible_action(self):
        result = await PromptRewriter(provider="local").complete_world_rule(
            "喷火", "A rider explores a valley.", "skill"
        )
        self.assertEqual(result.name, "喷火")
        self.assertEqual(result.change_type.value, "one_time")
        self.assertIn("bright orange fire", result.prompt)
        self.assertIn("sparks", result.prompt)
        self.assertIn("A rider explores a valley.", result.prompt)

    async def test_vertex_failure_can_fall_back_to_local_skill_prompt(self):
        models = FakeModels([RuntimeError("vertex unavailable")])
        client = SimpleNamespace(aio=SimpleNamespace(models=models))
        previous = os.environ.get("VIDEO_PROMPT_REWRITE_FALLBACK_LOCAL")
        os.environ["VIDEO_PROMPT_REWRITE_FALLBACK_LOCAL"] = "true"
        try:
            result = await PromptRewriter(
                client=client,
                provider="vertex",
                request_timeout_seconds=1,
            ).complete_world_rule("喷火", "A rider explores a valley.", "skill")
        finally:
            if previous is None:
                os.environ.pop("VIDEO_PROMPT_REWRITE_FALLBACK_LOCAL", None)
            else:
                os.environ["VIDEO_PROMPT_REWRITE_FALLBACK_LOCAL"] = previous
        self.assertEqual(result.change_type.value, "one_time")
        self.assertIn("bright orange fire", result.prompt)

    def test_build_message_rejects_missing_state(self):
        with self.assertRaisesRegex(ValueError, "previous_prompt"):
            build_user_message("", "下雪")

    def test_build_world_rule_message_rejects_unknown_kind(self):
        with self.assertRaisesRegex(ValueError, "kind"):
            build_world_rule_message("A valley.", "下雪", "unknown")


if __name__ == "__main__":
    unittest.main()
