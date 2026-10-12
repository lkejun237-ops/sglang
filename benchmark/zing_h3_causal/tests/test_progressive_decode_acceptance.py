from types import SimpleNamespace

import pytest
import torch
from benchmark_taeh3_replay import validate_progressive
from run_encoder_overlap_matrix import role_environment, record_role_environment


def test_decode_role_policy_is_explicit_and_recorded(tmp_path):
    import json

    key = "SGLANG_ZING_EXPERIMENTAL_PROGRESSIVE_DECODE"
    base = {key: "1"}
    assert role_environment({}, "server", "0", base)[key] == "0"
    for role in ("server", "consumer"):
        assert (
            role_environment({"progressive_decode": True}, role, "0", base)[key] == "1"
        )
    env = role_environment({"progressive_decode": True}, "encoder", "1", base)
    assert env[key] == "0"
    record_role_environment(tmp_path / "env.json", encoder=env)
    assert json.loads((tmp_path / "env.json").read_text())["encoder"][key] == "0"
    with pytest.raises(ValueError, match="boolean"):
        role_environment({"progressive_decode": "1"}, "server", "0")


@pytest.mark.parametrize("mismatch", [False, True])
def test_strict_decoder_gate_preserves_failure(mismatch):
    class Stream:
        def __init__(self, progressive):
            self.progressive = progressive

        def push(self, x):
            return torch.empty(1, 3, 0, 1, 1)

        def finish(self, frames):
            return torch.full(
                (1, 3, frames, 1, 1), float(mismatch and self.progressive)
            )

        def close(self):
            pass

    model = SimpleNamespace(new_stream=lambda progressive: Stream(progressive))
    result = validate_progressive(model, [torch.zeros(1, 24, 7, 1, 1)], 18)
    assert result["passed"] is not mismatch
    assert result["atol"] == result["rtol"] == 0
    assert [x["frames"] for x in result["cases"]] == list(range(1, 19))
