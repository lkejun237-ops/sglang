# SPDX-License-Identifier: Apache-2.0
"""The tile acceptance gate cannot silently tolerate differences."""

import json

import torch
from validate_progressive_tiles import encoder_server_args, exact


def test_exact_gate_checks_bytes_shape_dtype_and_finiteness():
    assert exact(torch.ones(2), torch.ones(2))["passed"]
    assert not exact(torch.zeros(2), torch.zeros(1, 2))["passed"]
    assert not exact(torch.zeros(2), torch.zeros(2).double())["passed"]
    assert not exact(torch.tensor([0.0]), torch.tensor([-0.0]))["passed"]
    a = torch.tensor([1.0], dtype=torch.float64)
    assert not exact(a, a + 1e-14)["passed"]
    for value in (float("nan"), float("inf")):
        x = torch.tensor([value])
        assert not exact(x, x)["passed"]


def test_encoder_probe_does_not_require_decoder_checkpoint(tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "ZingH3CausalPipeline",
                "_diffusers_version": "0.36.0",
                "video_vae": ["sglang", "MiniMaxH3VideoVAE"],
            }
        )
    )
    (tmp_path / "video_vae").mkdir()
    args = encoder_server_args(str(tmp_path))
    assert args.pipeline_config.video_decoder_backend == "native"
    assert args.pipeline_config.taeh3_checkpoint_path is None
    args.pipeline_config.validate_server_args(args)


def test_auto_probe_uses_unset_environment(monkeypatch):
    import os

    import validate_progressive_tiles as probe

    name = "MINIMAX_H3_PROGRESSIVE_ENCODER_TILING"
    monkeypatch.setenv(name, "parallel")
    monkeypatch.setattr(
        probe, "create_session", lambda *args, **kwargs: os.environ.get(name)
    )
    assert probe.factory("auto")(None, capacity_frames=8) is None
    assert probe.factory("replicated")(None, capacity_frames=8) == "replicated"
