"""Deployment-scoped prompt budget; no GPU inference is needed."""

import pytest

from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
from sglang.multimodal_gen.runtime.conditioning.encoder import (
    encode_condition,
    validate_prompt_token_budget,
)


class ByteTokenizer:
    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": list(text.encode("utf-8"))}


def test_disabled_budget_preserves_existing_deployments():
    validate_prompt_token_budget(None, "existing prompt", None)


def test_exact_boundary_and_one_token_over():
    tokenizer = ByteTokenizer()
    validate_prompt_token_budget(tokenizer, "a" * 8192, 8192)
    with pytest.raises(ConditionError, match="Prompt tokens"):
        validate_prompt_token_budget(tokenizer, "a" * 8193, 8192)


def test_counts_tokenizer_output_instead_of_characters():
    # A short Unicode prompt can consume more tokenizer rows than characters.
    with pytest.raises(ConditionError, match="Prompt tokens"):
        validate_prompt_token_budget(ByteTokenizer(), "汉" * 3, 8)


@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.5, "8192"])
def test_invalid_deployment_budget(limit):
    with pytest.raises(ConditionError, match="Invalid prompt"):
        validate_prompt_token_budget(ByteTokenizer(), "a", limit)


def test_rejects_before_media_processing_or_model_forward():
    # The poisoned processor must never be accessed after a budget rejection.
    with pytest.raises(ConditionError, match="Prompt tokens"):
        encode_condition(
            {"tokenizer": ByteTokenizer(), "processor": object()},
            {},
            "a" * 9,
            [{"type": "image"}],
            max_prompt_tokens=8,
        )
