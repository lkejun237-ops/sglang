# SPDX-License-Identifier: Apache-2.0
"""The service must finish its first (kernel-compiling) encode before it listens."""

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

from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.runtime.conditioning import cli
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class RecordingEncoder:
    def __init__(self, events):
        self.events = events
        self.values = []

    def __call__(self, value):
        self.events.append("warm")
        self.values.append(value)
        return {"context": torch.zeros(1, 4)}


def test_warm_encoder_runs_one_real_shaped_image_encode():
    events = []
    encoder = RecordingEncoder(events)
    cli.warm_encoder(encoder)
    assert events == ["warm"]
    value = encoder.values[0]
    assert isinstance(value.prompt, str) and value.prompt
    (item,) = value.media
    assert item["type"] == "image"
    assert item["frames"].shape == (3, 1, 480, 832)
    assert item["frames"].dtype == torch.uint8


def test_serve_warms_before_it_starts_listening(monkeypatch):
    import sys
    import types

    events = []
    encoder = RecordingEncoder(events)
    limits = SimpleNamespace(
        max_condition_rows=16384, max_active_requests=8, max_header_bytes=65536
    )
    monkeypatch.setattr(
        cli,
        "load_configuration",
        lambda args: (
            {"native": {"model_path": "m", "gpu_id": 0}, "tokens": {}},
            object(),
            limits,
        ),
    )
    from sglang.multimodal_gen.runtime.conditioning import native, service

    monkeypatch.setattr(native, "NativeConditionEncoder", lambda *a, **k: encoder)
    monkeypatch.setattr(service, "ConditioningService", lambda **k: "app")
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda app, **kwargs: events.append("listen")
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    cli.main(["serve", "--config", "c", "--profile", "p"])
    assert events == ["warm", "listen"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
