# SPDX-License-Identifier: Apache-2.0
"""Real CPU sampler checks; CUDA event validity is checked by the GPU smoke."""

from types import SimpleNamespace

import pytest
import torch
from dit_phase_timing import PhaseTimer, summarize

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
    ZingH3Packer,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import model  # noqa: F401
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
    request,
)


@pytest.mark.parametrize("ratio, spatial", [(16, (30, 52)), (8, (60, 104))])
def test_conditioning_geometry_uses_production_vae_config(ratio, spatial):
    from bench_dit_phases import check_conditioning

    arch = SimpleNamespace(patch_size=(1, 2, 2), latents_dim=24)
    vae = SimpleNamespace(
        spatial_compression_ratio=ratio,
        temporal_compression_ratio=4,
        vae_clip_length=16,
    )
    shape = (31, *spatial)
    inputs = SimpleNamespace(
        audio_shape=None,
        visual=dict.fromkeys(range(6)),
        source_visual=SimpleNamespace(shape=(1, 24, *shape)),
        visual_shape=shape,
        pixel_frame_count=120,
    )
    runtime = dict(h3_media_config=dict(max_short_edge=2048))
    assert check_conditioning(inputs, arch, vae, runtime) == shape
    inputs.visual_shape = (31, spatial[0] * 2, spatial[1] * 2)
    with pytest.raises(ValueError, match="latent geometry"):
        check_conditioning(inputs, arch, vae, runtime)


def test_capture_preserves_rng_and_copies_dataclass_to_snapshot(monkeypatch, tmp_path):
    import capture_dit_conditioning

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
        ZingH3Codecs,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
        H3InferenceInputs,
    )

    value = H3InferenceInputs(
        output_type="video",
        conditions=[],
        media=[],
        positive={"context": torch.ones(3, 12)},
        visual={"ref": torch.ones(1)},
        audio={},
        source_visual=torch.ones(1, 2, 15, 2, 2),
        visual_shape=(15, 2, 2),
        audio_shape=None,
        pixel_frame_count=120,
        sample_count=None,
        sample_rate=48000,
    )
    monkeypatch.setattr(ZingH3Codecs, "prepare", lambda *args, **kwargs: value)
    path = tmp_path / "conditioning.pt"
    monkeypatch.setenv("ZING_BENCH_CONDITIONING_OUTPUT", str(path))
    capture_dit_conditioning.install()
    before = torch.get_rng_state().clone()
    assert object.__new__(ZingH3Codecs).prepare({"sample_id": "capture-test"}) is value
    assert torch.equal(before, torch.get_rng_state())
    actual = torch.load(path, weights_only=False)
    assert isinstance(actual, H3InferenceInputs)
    assert torch.equal(actual.positive["context"], value.positive["context"])
    assert actual.source_visual.device.type == "cpu"


def test_real_sampler_phase_counts_history_and_unchanged_output(model):  # noqa: F811
    _, inputs, _, _ = request(model, source=True, audio=False)
    inputs.visual_shape = (15, 2, 2)
    inputs.source_visual = torch.randn(1, 2, 15, 2, 2)
    visual = torch.randn(1, 2, 15, 2, 2)
    packer = ZingH3Packer(
        model.arch, num_frame_per_block=[3, 2], split_first_latent=True
    )
    sampler = ZingH3Sampler(
        model,
        packer,
        solver="few_step",
        denoising_step_list=[1000, 750, 500, 250],
        window_chunks=2,
        sink_chunks=1,
    )
    torch.manual_seed(42)
    expected, _ = sampler.sample(inputs, visual)
    rows = []
    for mode in ("block", "forward"):
        timer = PhaseTimer(sampler, mode=mode, run=0, run_kind="measured")

        def measure(call, **fields):
            result = call()
            timer.rows.append(
                dict(
                    run=0,
                    run_kind="measured",
                    mode=mode,
                    chunk=timer.chunk,
                    latent_frames=timer.frames,
                    wall_ms=1,
                    cuda_ms=1,
                    **fields,
                )
            )
            return result

        timer.measure = measure
        torch.manual_seed(42)
        with timer:
            actual, _ = sampler.sample(inputs, visual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        rows.extend(timer.rows)
    denoise = [r for r in rows if r["kind"] == "denoise"]
    commits = [r for r in rows if r["kind"] == "clean_commit"]
    assert len(denoise) == 7 * 4 and len(commits) == 7
    assert [r["latent_frames"] for r in commits] == [1, 2, 2, 3, 2, 3, 2]
    assert {r["step"] for r in denoise} == {0, 1, 2, 3}
    assert all(
        r["kv_rows"] == r["q_rows"] + r["prefix_rows"] + r["history_rows"]
        for r in denoise
    )
    # One spatial token per latent: chunk0 is a 2-row source+target sink;
    # chunk2 adds four old rows, evicted only after chunk3's step zero.
    assert [
        (r["prefix_rows"], r["history_rows"]) for r in denoise if r["chunk"] == 3
    ] == [
        (3, 9),
        (3, 5),
        (3, 5),
        (3, 5),
    ]
    assert len(summarize(rows)["cycles"]) == 2
