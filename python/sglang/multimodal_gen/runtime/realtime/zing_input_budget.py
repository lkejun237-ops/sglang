# SPDX-License-Identifier: Apache-2.0
"""Zing prompt/reference admission; no model execution or session state."""

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sglang.multimodal_gen.runtime.conditioning.contract import ConditionError
from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
    DEFAULT_MAX_REFERENCE_IMAGES,
    ReferenceLimits,
)

logger = logging.getLogger(__name__)
REFERENCE_TOKEN_ESTIMATE = 1700
CONDITION_TOKEN_OVERHEAD = 500


@dataclass(frozen=True)
class ZingInputBudget:
    max_prompt_tokens: int = 40000
    max_condition_tokens: int = 50000
    max_references: int = DEFAULT_MAX_REFERENCE_IMAGES

    def __post_init__(self):
        for name, config_name in (
            ("max_prompt_tokens", "zing_prompt_max_tokens"),
            ("max_condition_tokens", "zing_condition_max_tokens"),
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{config_name} must be a positive integer")
        ReferenceLimits(self.max_references)

    @classmethod
    def from_config(cls, config):
        return cls(
            max_prompt_tokens=getattr(config, "zing_prompt_max_tokens", 40000),
            max_condition_tokens=getattr(config, "zing_condition_max_tokens", 50000),
            max_references=getattr(
                config, "zing_reference_max_images", DEFAULT_MAX_REFERENCE_IMAGES
            ),
        )

    def validate(self, prompt_tokens, reference_count):
        if type(prompt_tokens) is not int or prompt_tokens < 0:
            raise ConditionError("CONDITION_INVALID", "Invalid prompt token count")
        if prompt_tokens > self.max_prompt_tokens:
            raise ConditionError(
                "CONDITION_PROMPT_TOO_LONG",
                f"Prompt tokens {prompt_tokens} exceed {self.max_prompt_tokens}",
            )
        if type(reference_count) is not int or reference_count < 0:
            raise ConditionError("CONDITION_INVALID", "Invalid reference count")
        if reference_count > self.max_references:
            raise ConditionError(
                "CONDITION_REFERENCE_LIMIT",
                f"Reference count {reference_count} exceeds {self.max_references}",
            )
        estimate = (
            prompt_tokens
            + REFERENCE_TOKEN_ESTIMATE * reference_count
            + CONDITION_TOKEN_OVERHEAD
        )

        if estimate > self.max_condition_tokens:
            raise ConditionError(
                "CONDITION_BUDGET_EXCEEDED",
                f"Estimated condition tokens {estimate} exceed {self.max_condition_tokens}",
            )

    def metadata(self):
        return dict(
            max_prompt_tokens=self.max_prompt_tokens,
            max_references=self.max_references,
            max_condition_tokens=self.max_condition_tokens,
            reference_token_estimate=REFERENCE_TOKEN_ESTIMATE,
            condition_token_overhead=CONDITION_TOKEN_OVERHEAD,
        )


@lru_cache(maxsize=1)
def _prompt_tokenizer(model_path):
    # Only CPU tokenizer files are loaded, never Qwen weights or remote code.
    # Match TokenizerLoader's Qwen2Tokenizer slow/fast choice.
    import json

    from transformers import AutoTokenizer

    root = Path(model_path)
    index = json.loads((root / "model_index.json").read_text())
    architecture = index["tokenizer"][1]
    return AutoTokenizer.from_pretrained(
        root / "tokenizer",
        local_files_only=True,
        padding_side="right",
        use_fast=architecture != "Qwen2Tokenizer",
    )


def count_prompt_tokens(model_path, prompt):
    try:
        tokenizer = _prompt_tokenizer(str(model_path))
        return len(tokenizer.encode(prompt, add_special_tokens=False, truncation=False))
    except Exception as exc:
        logger.exception("Zing prompt token counting failed")
        raise ConditionError(
            "CONDITION_UNAVAILABLE", "The prompt tokenizer is unavailable"
        ) from exc
