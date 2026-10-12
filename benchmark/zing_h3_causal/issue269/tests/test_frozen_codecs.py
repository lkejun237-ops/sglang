# SPDX-License-Identifier: Apache-2.0
"""Exercise snapshot identity and session reset with CPU codec fixtures."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def harness(monkeypatch, tmp_path):
    prefix = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3."

    class Codecs:
        def __init__(self):
            self.runtime = {"vae_config": {"encode_mode": "performance"}}
            self.bias = 0

        def encode_condition(self, prompt, media):
            return {"context": torch.tensor([self.bias + 3.0])}

        def encode_visual(self, frames):
            return frames.float() + self.bias

        def sample_streaming_visual_moments(self, moments, **kwargs):
            return moments + self.bias

    class Source:
        def __init__(self):
            self.policy = {}

        def run(self, stage, batch):
            frames = batch.frames
            return [
                stage.codecs.encode_condition(
                    "prompt", [{"id": "ref", "frames": frames}]
                ),
                stage.codecs.encode_visual(frames),
                stage.codecs.sample_streaming_visual_moments(
                    frames.float(), latent_start=0, noise_version="indexed-cpu-v1"
                ),
            ]

        def _decode_ready(self, *args, **kwargs):
            return None

    class Encoder:
        def pop(self):
            return None

    class Decoder:
        def append(self, *args, **kwargs):
            return None

    class Ready:
        def append(self, *args, **kwargs):
            return None

    for name, cls in [
        ("codecs", Codecs),
        ("source_session", Source),
        ("progressive_encoder", Encoder),
        ("streaming_codecs", Decoder),
        ("incremental", Ready),
    ]:
        module = types.ModuleType(prefix + name)
        exported = {
            "codecs": "ZingH3Codecs",
            "source_session": "ZingSourceStreamState",
            "progressive_encoder": "H3ProgressiveEncoderSession",
            "streaming_codecs": "StableVideoDecoder",
            "incremental": "SourceReadySession",
        }[name]
        setattr(module, exported, cls)
        monkeypatch.setitem(sys.modules, prefix + name, module)
    monkeypatch.setenv("ZING269_ROOT", str(tmp_path))
    monkeypatch.setenv("ZING269_ARM", "test")
    monkeypatch.setenv("ZING269_FREEZE", "capture")
    monkeypatch.setenv("ZING269_BENCHMARK_OFF", "0")
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    original_to = torch.Tensor.to

    def cpu_to(self, *args, **kwargs):
        return (
            self
            if args == ("cuda",) and not kwargs
            else original_to(self, *args, **kwargs)
        )

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    spec = importlib.util.spec_from_file_location(
        "issue269_frozen", Path(__file__).parents[1] / "frozen_codecs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install()
    return Codecs, Source, tmp_path


def test_later_session_reuses_first_snapshot_and_resets_indices(harness):
    Codecs, Source, root = harness
    codecs = Codecs()
    source = Source()
    stage = types.SimpleNamespace(codecs=codecs)
    batch = types.SimpleNamespace(
        realtime_session_id="first",
        block_idx=0,
        frames=torch.ones(1, 2, 3, dtype=torch.uint8),
    )
    first = source.run(stage, batch)
    codecs.bias = 7
    batch.realtime_session_id = "second"
    second = source.run(stage, batch)
    assert torch.equal(first[0]["context"], second[0]["context"])
    assert all(torch.equal(a, b) for a, b in zip(first[1:], second[1:]))
    rows = [
        json.loads(line)
        for line in (root / "test-rank0.jsonl").read_text().splitlines()
    ]
    refs = [r for r in rows if r["operation"] == "encode_visual"]
    assert [r["index"] for r in refs] == [0, 0]
    assert refs[0]["used"] == refs[1]["used"]
    assert refs[0]["observed"] != refs[1]["observed"]


def test_changed_raw_reference_is_rejected(harness):
    Codecs, Source, _ = harness
    source = Source()
    stage = types.SimpleNamespace(codecs=Codecs())
    batch = types.SimpleNamespace(
        realtime_session_id="first",
        block_idx=0,
        frames=torch.ones(1, 2, 3, dtype=torch.uint8),
    )
    source.run(stage, batch)
    batch.realtime_session_id = "second"
    batch.frames = batch.frames + 1
    with pytest.raises(AssertionError, match="input mismatch"):
        source.run(stage, batch)


def test_benchmark_probe_accepts_legacy_policy_without_receipt(harness, monkeypatch):
    """Historical encoder contexts yield None, unlike the latest runtime."""
    from contextlib import contextmanager

    _ = harness
    prefix = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3"
    package = types.ModuleType(prefix)
    package.__path__ = []
    package.codecs = sys.modules[prefix + ".codecs"]
    package.progressive_encoder = sys.modules[prefix + ".progressive_encoder"]
    monkeypatch.setitem(sys.modules, prefix, package)

    @contextmanager
    def legacy_policy(*args, **kwargs):
        previous = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = True
        try:
            yield
        finally:
            torch.backends.cudnn.benchmark = previous

    monkeypatch.setattr(
        package.codecs, "_visual_encode_policy", legacy_policy, raising=False
    )
    monkeypatch.setenv("ZING269_BENCHMARK_OFF", "1")
    spec = importlib.util.spec_from_file_location(
        "issue269_legacy", Path(__file__).parents[1] / "frozen_codecs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install()
    with package.codecs._visual_encode_policy("performance") as receipt:
        assert not torch.backends.cudnn.benchmark
        assert receipt["cudnn_benchmark"] is False
