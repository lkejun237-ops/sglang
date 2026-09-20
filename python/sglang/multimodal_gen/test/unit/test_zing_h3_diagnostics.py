# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_zing_h3_causal_renoise import (
    ZingH3CausalRenoiseSchedule,
)
from sglang.multimodal_gen.runtime.utils import zing_diagnostics as diagnostics


def test_capture_preserves_rng_and_records_actual_renoise(tmp_path):
    schedule = ZingH3CausalRenoiseSchedule(
        [1000, 750, 500, 250], num_train_timesteps=1000, shift=5
    )
    sample = torch.arange(30, dtype=torch.float32).reshape(5, 6).t()
    prediction = torch.full_like(sample, 0.25)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        expected = schedule.step(prediction, sample, step_index=0)
        expected_state = torch.random.get_rng_state()
        torch.manual_seed(42)
        trace = diagnostics.ZingDiagnostic(tmp_path / "capture")
        token = diagnostics._current.set(trace)
        try:
            actual = schedule.step(prediction, sample, step_index=0)
        finally:
            diagnostics._current.reset(token)
        assert torch.equal(torch.random.get_rng_state(), expected_state)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        entry = json.loads((tmp_path / "capture/events.jsonl").read_text())
        data = torch.load(tmp_path / "capture" / entry["file"], weights_only=True)
        assert entry["tensors"]["sample"]["stride"] == list(sample.stride())
        assert entry["tensors"]["sample"]["contiguous"] is False
        torch.manual_seed(42)
        expected_noise = torch.randn_like(actual[1].to(sample.dtype))
        assert torch.equal(data["noise"], expected_noise)
        assert torch.equal(data["next_sample"], actual[0])


def test_tensor_hash_is_logical_order_but_stride_is_recorded(tmp_path):
    trace = diagnostics.ZingDiagnostic(tmp_path / "capture")
    tensor = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).t()
    trace.record(
        "layout", strided=tensor, contiguous=tensor.contiguous(), empty=tensor[:0]
    )
    entry = json.loads((tmp_path / "capture/events.jsonl").read_text())
    a, b = entry["tensors"]["strided"], entry["tensors"]["contiguous"]
    assert a["sha256"] == b["sha256"]
    assert a["stride"] != b["stride"]


def test_failed_capture_resets_request_context(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_DIAGNOSTICS_DIR", str(tmp_path))

    def fail_record(*args, **kwargs):
        raise OSError("Diagnostic storage unavailable")

    monkeypatch.setattr(diagnostics.ZingDiagnostic, "record", fail_record)

    @diagnostics.diagnostic_request
    def forward(self, batch):
        pytest.fail("A failed initial capture must not start generation")

    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(output_file_name="sample.mp4", seed=42)
    )
    previous = diagnostics.current_diagnostic()
    with pytest.raises(OSError, match="Diagnostic storage unavailable"):
        forward(None, batch)
    assert diagnostics.current_diagnostic() is previous


def test_prefix_module_hooks_preserve_output_and_are_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_DIAGNOSTICS_PREFIX_MODULES", "1")
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
    value = torch.ones(2, 4)
    expected = model.blocks[0](value)
    trace = diagnostics.ZingDiagnostic(tmp_path / "capture")
    with trace.prefix_modules(model):
        trace.phase = "prefix"
        actual = model.blocks[0](value)
        trace.phase = "block2.step0"
        model.blocks[0](value)
    assert torch.equal(expected, actual)
    assert all(not block._forward_hooks for block in model.blocks)
    events = (tmp_path / "capture/events.jsonl").read_text().splitlines()
    assert len(events) == 1
    assert json.loads(events[0])["name"] == "module.blocks.0"


def test_selected_module_phase_and_exception_cleanup(tmp_path, monkeypatch):
    monkeypatch.delenv("SGLANG_ZING_DIAGNOSTICS_PREFIX_MODULES", raising=False)
    monkeypatch.setenv("SGLANG_ZING_DIAGNOSTICS_MODULE_PHASE", "block4.step0")
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList(
        [torch.nn.Sequential(torch.nn.Linear(4, 4)) for _ in range(3)]
    )
    model.video_patch_proj = torch.nn.Linear(4, 4)
    model.audio_patch_proj = torch.nn.Linear(4, 4)
    trace = diagnostics.ZingDiagnostic(tmp_path / "capture")
    with pytest.raises(RuntimeError, match="stop"):
        with trace.prefix_modules(model):
            trace.phase = "prefix"
            model.blocks[0](torch.ones(2, 4))
            trace.phase = "block4.step0"
            model.video_patch_proj(torch.ones(2, 4))
            model.audio_patch_proj(torch.ones(1, 4))
            for block in model.blocks:
                block(torch.ones(2, 4))
            raise RuntimeError("stop")
    assert all(not module._forward_hooks for module in model.modules())
    events = [
        json.loads(x)
        for x in (tmp_path / "capture/events.jsonl").read_text().splitlines()
    ]
    assert {e["phase"] for e in events} == {"block4.step0"}
    assert {e["name"] for e in events} == {
        "module.blocks.0",
        "module.blocks.0.0",
        "module.video_patch_proj",
        "module.audio_patch_proj",
        "module.blocks.1",
        "module.blocks.1.0",
        "module.blocks.2",
    }


def test_hook_registration_failure_cleans_previous_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_ZING_DIAGNOSTICS_PREFIX_MODULES", "1")
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(2)])

    def fail(*args, **kwargs):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(model.blocks[1], "register_forward_hook", fail)
    trace = diagnostics.ZingDiagnostic(tmp_path / "capture")
    with pytest.raises(RuntimeError, match="registration failed"):
        with trace.prefix_modules(model):
            pytest.fail("registration should fail")
    assert not model.blocks[0]._forward_hooks


def test_selected_phase_sequence_alignment_uses_its_forward_rows(tmp_path):
    import importlib.util
    from pathlib import Path

    repo = next(
        p
        for p in Path(__file__).resolve().parents
        if (p / "benchmark/zing_h3_causal/compare_sp_diagnostics.py").exists()
    )
    spec = importlib.util.spec_from_file_location(
        "compare_diagnostics",
        repo / "benchmark/zing_h3_causal/compare_sp_diagnostics.py",
    )
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    left = diagnostics.ZingDiagnostic(tmp_path / "left")
    right = diagnostics.ZingDiagnostic(tmp_path / "right")
    full = torch.arange(20).reshape(5, 4)
    for trace, output in [(left, full), (right, full[3:])]:
        trace.phase = "prefix"
        trace.record("forward_input", packed={"position_ids": torch.zeros(8, 3)})
        trace.phase = "block4.step0"
        trace.record("forward_input", packed={"position_ids": torch.zeros(5, 3)})
        trace.record("module.blocks.0", output=output)
    result = comparator.compare(
        left.directory,
        right.directory,
        right_sequence_size=2,
        right_sequence_rank=1,
        module_phase="block4.step0",
    )
    assert result["first_tensor_divergence"] is None
    assert result["sequence_comparison"]["module_phase"] == "block4.step0"
    default = comparator.compare(
        left.directory, right.directory, right_sequence_size=2, right_sequence_rank=1
    )
    assert default["first_tensor_divergence"] is not None


@pytest.mark.parametrize("all_layers", [False, True])
def test_cache_all_layers_opt_in_and_head_alignment(tmp_path, monkeypatch, all_layers):
    import importlib.util
    from pathlib import Path

    monkeypatch.setenv(
        "SGLANG_ZING_DIAGNOSTICS_ALL_CACHE_LAYERS", "1" if all_layers else "0"
    )
    repo = next(
        p
        for p in Path(__file__).resolve().parents
        if (p / "benchmark/zing_h3_causal/compare_sp_diagnostics.py").exists()
    )
    spec = importlib.util.spec_from_file_location(
        "compare_diagnostics",
        repo / "benchmark/zing_h3_causal/compare_sp_diagnostics.py",
    )
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    layers = tuple(
        (torch.arange(24).reshape(3, 4, 2) + i, torch.ones(3, 4, 2) * i)
        for i in range(3)
    )
    left = diagnostics.ZingDiagnostic(tmp_path / "left")
    right = diagnostics.ZingDiagnostic(tmp_path / "right")
    for trace, values in [
        (left, layers),
        (right, tuple((k[:, 2:], v[:, 2:]) for k, v in layers)),
    ]:
        chunk = SimpleNamespace(
            block=2, frames=1, positions=torch.arange(3), layers=values
        )
        trace.cache(
            SimpleNamespace(_history=lambda: [chunk], active=None, last_block=2)
        )
    entry = json.loads((left.directory / "events.jsonl").read_text())
    assert ("chunks.0.layers.1.0" in entry["tensors"]) == all_layers
    assert "chunks.0.layer0.0" in entry["tensors"]
    assert "chunks.0.layer_last.1" in entry["tensors"]
    result = comparator.compare(
        left.directory, right.directory, right_sp_size=2, right_rank=1
    )
    assert result["first_tensor_divergence"] is None


@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("audio_first", [False, True])
def test_projection_compact_rows_alignment(tmp_path, rank, audio_first):
    import importlib.util
    from pathlib import Path

    repo = next(
        p
        for p in Path(__file__).resolve().parents
        if (p / "benchmark/zing_h3_causal/compare_sp_diagnostics.py").exists()
    )
    spec = importlib.util.spec_from_file_location(
        "compare_diagnostics",
        repo / "benchmark/zing_h3_causal/compare_sp_diagnostics.py",
    )
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    roles = torch.tensor(
        ([11] * 16 + [10] * 896) if audio_first else ([10] * 896 + [11] * 16)
    )
    video = torch.arange(896 * 2).reshape(896, 2)
    audio = torch.arange(16 * 2).reshape(16, 2)
    left = diagnostics.ZingDiagnostic(tmp_path / "left")
    right = diagnostics.ZingDiagnostic(tmp_path / "right")
    for trace in (left, right):
        trace.phase = "block4.step0"
        trace.record(
            "forward_input",
            packed={"position_ids": torch.zeros(912, 3), "row_role": roles},
        )
        for name, value, role in [
            ("video_patch_proj", video, 10),
            ("audio_patch_proj", audio, 11),
        ]:
            if trace is right:
                start, end = rank * 228, (rank + 1) * 228
                value = value[
                    int((roles[:start] == role).sum()) : int(
                        (roles[:end] == role).sum()
                    )
                ]
            trace.record("module." + name, input=(value,), output=(value + 1, None))
    result = comparator.compare(
        left.directory,
        right.directory,
        right_sequence_size=4,
        right_sequence_rank=rank,
        module_phase="block4.step0",
    )
    assert result["first_tensor_divergence"] is None
