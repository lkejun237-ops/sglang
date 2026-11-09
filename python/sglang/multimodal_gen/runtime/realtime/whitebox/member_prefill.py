# SPDX-License-Identifier: Apache-2.0
"""One admission-frozen condition on a member's existing Prefill connection."""

import copy
import hashlib

from .render_contract import RenderError, canonical, segment_parameters

PROTOCOL = "worldkit-member-prefill-v1"


def validate_profile(value):
    if value != {"protocol": PROTOCOL}:
        raise ValueError("Invalid frozen member Prefill profile")


def condition_digest(value):
    if not isinstance(value, dict):
        raise RenderError("CONDITION_INVALID")
    parsed = segment_parameters(value)
    condition = {key: parsed[key] for key in ("prompt", "seed", "references")}
    return hashlib.sha256(canonical(condition)).hexdigest()


class MemberPrefill:
    """Not a browser opt-in: constructed only from the verified member factory."""

    def __init__(self, owner):
        self.owner = owner

    def conditions(self):
        self.owner.config.engine_factory.check()
        assets = self.owner.inference.assets
        if assets is None or not assets.committed.is_set():
            raise RenderError("CONDITION_INVALID")
        return dict(
            prompt=assets.prompt,
            seed=assets.seed,
            references=copy.deepcopy(self.owner.initial_assets),
        )

    def descriptor(self):
        conditions = self.conditions()
        return dict(
            protocol=PROTOCOL,
            prompt_sha256=hashlib.sha256(conditions["prompt"].encode()).hexdigest(),
            seed=conditions["seed"],
            references=conditions["references"],
        )

    def validate_conditions(self, value):
        if condition_digest(value) != condition_digest(self.conditions()):
            raise RenderError("PREFILL_INPUT_MISMATCH")

    def validate_start(self, command):
        self.validate_conditions(command)
        prefill = self.owner.control_prefill
        if prefill is None or not getattr(self.owner.engine, "shared_world", False):
            raise RenderError("CAPABILITY_UNAVAILABLE")
        first = prefill.resolve(command.get("context_id"))
        if command.get("start_boundary") != {"source_frame_index": first[1][0].index}:
            raise RenderError("PREFILL_INPUT_MISMATCH")
        worker = self.owner.model_binding.assignment.denoiser.worker_epoch
        expected = condition_digest(command)
        if not any(
            request.get("context_id") == command["context_id"]
            and request.get("condition_sha256") == expected
            and request.get("ready_emitted") is True
            and request.get("receipt", {}).get("worker_epoch") == worker
            for request in prefill.requests.values()
        ):
            raise RenderError("PREFILL_INPUT_MISMATCH")
