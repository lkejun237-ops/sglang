# SPDX-License-Identifier: Apache-2.0
"""Realtime deployment plans, checked against real small-model sampling."""

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import msgspec.msgpack
import pytest
import torch

from sglang.multimodal_gen.configs.zing_sampling import (
    parse_noise_grid,
    validate_initial_blocks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.realtime_sampling import (
    ZingH3RealtimeSampler,
)
from sglang.multimodal_gen.test.unit.realtime.test_zing_h3_source_adapter import (  # noqa: F401
    configured,
)
from sglang.multimodal_gen.test.unit.test_zing_h3_model import model  # noqa: F401
from sglang.multimodal_gen.test.unit.test_zing_h3_model import (  # noqa: F401
    cpu_execution_options,
    request,
)


@pytest.mark.parametrize("count", [0, 3])
@pytest.mark.parametrize("grid", [[1000, 500], [1000, 750]])
def test_deployment_grids(count, grid):
    regular = parse_noise_grid(grid)
    initial = parse_noise_grid("1000,750,500,250") if count else None
    validate_initial_blocks("few_step", regular, initial, count)
    assert regular == grid


@pytest.mark.parametrize(
    "count,cli_steps,profile_count",
    [
        (0, None, 0),
        (0, 2, 0),
        (3, 2, 0),
        (0, None, 3),
        (0, 4, 3),
        (0, 2, 3),
        (0, None, None),
    ],
)
@pytest.mark.parametrize("regular_grid", [[1000, 750], [1000, 500]])
@torch.no_grad()
def test_source_ready_schedule_matches_worker_execution(
    configured,  # noqa: F811
    monkeypatch,
    model,  # noqa: F811
    count,
    cli_steps,
    profile_count,
    regular_grid,
):
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters import (
        zing_h3_source_adapter as source,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        ZingSourceStreamState,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )
    from sglang.multimodal_gen.test.unit.realtime.test_zing_h3_source_adapter import (
        initialize,
    )

    configured.zing_solver = "few_step" if cli_steps else None
    configured.zing_denoising_step_list = (
        (regular_grid if cli_steps == 2 else [1000, 750, 500, 250][:cli_steps])
        if cli_steps
        else None
    )
    configured.zing_initial_block_count = count
    configured.zing_initial_denoising_step_list = (
        [1000, 750, 500, 250] if count else None
    )
    if profile_count is None:
        path = Path(configured.model_path) / "zing_realtime_profile.json"
        profile = json.loads(path.read_text())
        profile.pop("initial_block_count")
        path.write_text(json.dumps(profile))
    elif profile_count:
        path = Path(configured.model_path) / "zing_realtime_profile.json"
        profile = json.loads(path.read_text())
        profile.update(
            denoising_step_list=[1000, 500],
            initial_block_count=3,
            initial_denoising_step_list=[1000, 750, 500, 250],
        )
        path.write_text(json.dumps(profile))
    adapter, session = initialize()
    monkeypatch.setattr(
        source,
        "build_realtime_sampling_params",
        lambda *a, **kw: SimpleNamespace(guidance_scale=1, **kw),
    )
    params = adapter.build_sampling_params(
        session,
        configured,
        SimpleNamespace(request_id="test"),
        SimpleNamespace(prompt="snow"),
        17,
    )
    packer, inputs, _, _ = request(model, audio=False)
    inputs.visual_shape = (10, 2, 2)
    inputs.source_visual = torch.randn(1, 2, 10, 2, 2)
    # Match the existing pipeline's stage configuration; do not replace the sampler.
    stage = SimpleNamespace(
        modules={"transformer": model},
        packer=packer,
        runtime=dict(timestep_shift=8, audio_timestep_shift=8, num_train_timestep=1000),
        deployment_sampling={
            key: getattr(configured, "zing_" + key)
            for key in (
                "solver",
                "denoising_step_list",
                "initial_block_count",
                "initial_denoising_step_list",
            )
        },
    )
    # Exercise the real command assembly and worker initialize before media encoding.
    import msgspec.msgpack

    monkeypatch.setattr(
        source.BaseRealtimeModelAdapter,
        "prepare_next_request",
        lambda *a: SimpleNamespace(extra={}, block_idx=0),
    )

    # A worker's first source command always follows an accepted source append.
    class Socket:
        async def send_bytes(self, payload):
            pass

    payload = dict(
        client_seq=0,
        source_frame_start=0,
        num_frames=1,
        pts_us=0,
        data=bytes(32 * 32 * 3),
    )
    asyncio.run(
        adapter.ingest_wire_event(
            Socket(),
            session,
            dict(type="event", kind="source.append", event_id=1, payload=payload),
            len(payload["data"]) + 200,
        )
    )
    batch = adapter.prepare_next_request(session, configured, None)
    batch.extra = msgspec.msgpack.decode(msgspec.msgpack.encode(batch.extra))
    batch.sampling_params = params
    samplers = []

    def build(actual_params, **options):
        sampler = ZingH3GenerationStage._build_realtime_sampler(
            stage, actual_params, **options
        )
        samplers.append(sampler)
        return sampler

    class MediaBoundary(Exception):
        pass

    def prepare(*args, **kwargs):
        raise MediaBoundary()

    stage._build_realtime_sampler = build
    stage._validate_sampling_params = lambda params: None
    stage._build_request = lambda batch: None
    stage._conditioning_consumer = lambda batch: None
    stage.codecs = SimpleNamespace(video_decoder_name="test", prepare=prepare)
    with pytest.raises(MediaBoundary):
        ZingSourceStreamState().initialize(stage, batch, adapter.config)
    (sampler,) = samplers
    assert stage.deployment_sampling["initial_block_count"] == count
    state = sampler.initialize(inputs, torch.randn_like(inputs.source_visual))
    try:
        while state.index < len(state.spans):
            sampler.step(state)
        advertised = adapter.ready["sampling_schedule"]
        assert [
            item["noise_grid"] for item in sampler.last_run_metrics["block_schedules"]
        ] == [
            advertised["initial_denoising_step_list"]
            if i < advertised["initial_block_count"]
            else advertised["denoising_step_list"]
            for i in range(5)
        ]
        if cli_steps == 2:
            assert advertised["denoising_step_list"] == regular_grid
        expected = [
            len(advertised["initial_denoising_step_list"])
            if i < advertised["initial_block_count"]
            else adapter.ready["num_inference_steps"]
            for i in range(5)
        ]
        assert [
            item["denoise_steps"]
            for item in sampler.last_run_metrics["block_schedules"]
        ] == expected
        mixed = count or (
            (3 if profile_count is None else profile_count) if cli_steps is None else 0
        )
        assert expected == ([4, 4, 4, 2, 2] if mixed else [cli_steps or 4] * 5)
    finally:
        state.close()
        asyncio.run(adapter.close_conditioning(session))
        assert (
            session.id
            not in session.conditioning_resources.ledger.snapshot()["tenants"]
        )
        adapter.dispose(session)


@pytest.mark.parametrize("count", [0, 3])
def test_deployment_cli(count):
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    argv = [
        "--model-path",
        "unused",
        "--zing-solver",
        "few_step",
        "--zing-denoising-step-list",
        "1000,500",
        "--zing-initial-block-count",
        str(count),
    ]
    if count:
        argv += ["--zing-initial-denoising-step-list", "1000,750,500,250"]
    args = parser.parse_args(argv)
    args.pipeline_config = SimpleNamespace(zing_adaln_table_path=None)
    ServerArgs._validate_zing_sampling(args)
    assert args.zing_denoising_step_list == [1000, 500]
    assert args.zing_initial_block_count == count


@pytest.mark.parametrize(
    "options, expected",
    [
        ([], ("few_step", [1000, 500], [1000, 750, 500, 250])),
        (["--zing-initial-block-count", "0"], (None, None, None)),
        (
            ["--zing-solver", "euler", "--zing-initial-block-count", "0"],
            ("euler", None, None),
        ),
        (
            [
                "--zing-solver",
                "few_step",
                "--zing-denoising-step-list",
                "1000,500",
                "--zing-initial-block-count",
                "0",
            ],
            ("few_step", [1000, 500], None),
        ),
        (
            ["--zing-initial-block-count", "3"],
            ("few_step", [1000, 500], [1000, 750, 500, 250]),
        ),
        (
            ["--zing-denoising-step-list", "1000,500"],
            ("few_step", [1000, 500], [1000, 750, 500, 250]),
        ),
        (
            ["--zing-initial-denoising-step-list", "1000,800,600"],
            ("few_step", [1000, 500], [1000, 800, 600]),
        ),
        (
            [
                "--zing-denoising-step-list",
                "1000",
                "--zing-initial-denoising-step-list",
                "1000,800,600",
            ],
            ("few_step", [1000], [1000, 800, 600]),
        ),
        (
            [
                "--zing-initial-block-count",
                "3",
                "--zing-denoising-step-list",
                "1000,500",
            ],
            ("few_step", [1000, 500], [1000, 750, 500, 250]),
        ),
    ],
)
def test_deployment_cli_defaults(options, expected):
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "unused", *options])
    args.pipeline_config = SimpleNamespace(zing_adaln_table_path=None)
    ServerArgs._validate_zing_sampling(args)
    assert (
        args.zing_solver,
        args.zing_denoising_step_list,
        args.zing_initial_denoising_step_list,
    ) == expected


@pytest.mark.parametrize(
    "override",
    [
        {"zing_solver": "euler"},
        {"zing_denoising_step_list": [1000, 500, 500]},
        {"zing_initial_denoising_step_list": [750, 500]},
        {"zing_denoising_step_list": []},
        {"zing_initial_denoising_step_list": []},
    ],
)
def test_on_defaults_do_not_hide_conflicting_configuration(override):
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    values = dict(
        zing_solver=None,
        zing_denoising_step_list=None,
        zing_initial_denoising_step_list=None,
        zing_initial_block_count=3,
        pipeline_config=SimpleNamespace(zing_adaln_table_path=None),
    )
    values.update(override)
    with pytest.raises(ValueError):
        ServerArgs._validate_zing_sampling(SimpleNamespace(**values))


def test_on_defaults_have_independent_grids():
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    def resolve():
        args = SimpleNamespace(
            zing_solver=None,
            zing_denoising_step_list=None,
            zing_initial_denoising_step_list=None,
            zing_initial_block_count=3,
            pipeline_config=SimpleNamespace(zing_adaln_table_path="mixed-table"),
        )
        ServerArgs._validate_zing_sampling(args)
        return args

    first, second = resolve(), resolve()
    first.zing_denoising_step_list.append(250)
    first.zing_initial_denoising_step_list.append(125)
    assert second.zing_denoising_step_list == [1000, 500]
    assert second.zing_initial_denoising_step_list == [1000, 750, 500, 250]


@pytest.mark.parametrize("count", [0, 3])
@torch.no_grad()
def test_actual_target_steps_and_new_generation(model, count):  # noqa: F811
    packer, inputs, _, _ = request(model, audio=False)
    packer.num_frame_per_block = [3, 2]
    packer.split_first_latent = True
    inputs.visual_shape = (10, 2, 2)
    inputs.source_visual = torch.randn(1, 2, 10, 2, 2)
    noise = torch.randn_like(inputs.source_visual)
    sampler = ZingH3RealtimeSampler(
        model,
        packer,
        solver="few_step",
        denoising_step_list=[1000, 500],
        initial_denoising_step_list=[1000, 750, 500, 250] if count else None,
        initial_block_count=count,
        window_chunks=None,
        sink_chunks=0,
        rope_position_mode="absolute",
    )
    calls = []
    schedules = [sampler.video_scheduler]
    if count:
        schedules.append(sampler.initial_schedulers[0])
    for schedule in schedules:
        original = schedule.step

        def tracked(*args, _original=original, **kwargs):
            calls.append(kwargs["step_index"])
            return _original(*args, **kwargs)

        schedule.step = tracked
    expected = [4, 4, 4, 2, 2] if count else [2, 2, 2, 2, 2]
    for _ in range(2):
        calls.clear()
        state = sampler.initialize(inputs, noise)
        cache = state.cache
        try:
            while state.index < len(state.spans):
                sampler.step(state)
                assert state.cache is cache
            assert [
                r["latent_frames"] for r in sampler.last_run_metrics["block_schedules"]
            ] == [1, 2, 2, 3, 2]
            assert [
                r["denoise_steps"] for r in sampler.last_run_metrics["block_schedules"]
            ] == expected
            assert calls == [step for steps in expected for step in range(steps)]
        finally:
            state.close()


@pytest.mark.parametrize("count", [0, 3])
@torch.no_grad()
def test_ready_chunks_use_generation_index(model, count):  # noqa: F811
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import fixture_provenance

    packer, inputs, _, _ = request(model, audio=False)
    packer.num_frame_per_block = [3, 2]
    packer.split_first_latent = True
    packer.num_frame_per_block = [3, 2]
    packer.split_first_latent = True
    inputs.visual_shape = (10, 2, 2)
    inputs.source_visual = torch.randn(1, 2, 10, 2, 2)
    noise = torch.randn_like(inputs.source_visual)
    sampler = ZingH3RealtimeSampler(
        model,
        packer,
        solver="few_step",
        denoising_step_list=[1000, 500],
        initial_denoising_step_list=[1000, 750, 500, 250] if count else None,
        initial_block_count=count,
        joint_forward="cross_chunk",
    )
    session = SourceReadySession(
        sampler,
        inputs,
        seed=42,
        max_pending_chunks=20,
        max_pending_bytes=1000000,
        total_latent_frames=10,
    )
    try:
        for chunk in packer.chunk_plan.chunks(10):
            session.append(
                inputs.source_visual[:, :, chunk.start : chunk.end],
                noise[:, :, chunk.start : chunk.end],
                provenance=fixture_provenance(
                    chunk.chunk_id + 1, 20, chunk.start, chunk.end
                ),
            )
        executed = []
        for _ in session.drain_ready():
            executed.append(
                sampler.last_run_metrics["last_block_schedule"]["denoise_steps"]
            )
        assert executed == ([4, 4, 4, 2, 2] if count else [2, 2, 2, 2, 2])
    finally:
        session.dispose()


def test_deployment_schedule_is_only_used_by_realtime(model):  # noqa: F811
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
        ZingH3Sampler,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    stage = object.__new__(ZingH3GenerationStage)
    stage.modules = {"transformer": model}
    stage.packer, _, _, _ = request(model, audio=False)
    stage.runtime = {
        "timestep_shift": 8.0,
        "audio_timestep_shift": 3.0,
        "num_train_timestep": 1000,
    }
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "unused"])
    ServerArgs._validate_zing_sampling(args)
    stage.deployment_sampling = {
        key: getattr(args, "zing_" + key)
        for key in (
            "solver",
            "denoising_step_list",
            "initial_denoising_step_list",
            "initial_block_count",
        )
    }
    params = SimpleNamespace(
        zing_solver="few_step",
        zing_denoising_step_list=[1000, 500],
        num_inference_steps=50,
        guidance_scale=1.0,
    )
    offline = stage._build_sampler(params)
    realtime = stage._build_realtime_sampler(params)
    assert type(offline) is ZingH3Sampler
    assert type(realtime) is ZingH3RealtimeSampler
    expected = ZingH3Sampler(
        model,
        stage.packer,
        solver="few_step",
        denoising_step_list=[1000, 500],
        video_shift=8.0,
    )
    torch.testing.assert_close(
        offline.video_scheduler.timesteps, expected.video_scheduler.timesteps
    )
    assert realtime.denoising_step_list == [1000, 500]
    assert realtime.initial_block_count == 3


def test_realtime_step_restores_schedulers_on_failure(model, monkeypatch):  # noqa: F811
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
        ZingH3Sampler,
    )

    packer, _, _, _ = request(model, audio=False)
    sampler = ZingH3RealtimeSampler(
        model,
        packer,
        solver="few_step",
        denoising_step_list=[1000, 500],
        initial_denoising_step_list=[1000, 750, 500, 250],
        initial_block_count=3,
    )
    regular = sampler.video_scheduler, sampler.audio_scheduler

    def fail(self, state, *, inputs=None):
        assert self.video_scheduler is self.initial_schedulers[0]
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(ZingH3Sampler, "step", fail)
    with pytest.raises(RuntimeError, match="sampling failed"):
        sampler.step(SimpleNamespace(index=0))
    assert (sampler.video_scheduler, sampler.audio_scheduler) == regular


@pytest.fixture
def mixed_grids(request):
    return getattr(request, "param", ([1000, 500], [1000, 750, 500, 250]))


@pytest.fixture
def static_mixed(model, tmp_path, mixed_grids):  # noqa: F811
    from dataclasses import replace

    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
    from sglang.multimodal_gen.runtime.models.dits.zing_h3 import ZingH3DiTModel
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import (
        ZingAdalnTable,
        initial_blocks_plan,
        video_plan,
        write_table,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_model import (
        verify_synthetic_table,
    )

    arch = replace(model.arch, ar_condition_timestep=0, reference_visual_timestep=0.999)
    resident = ZingH3DiTModel(ZingH3DiTConfig(arch_config=arch), {}).eval()
    resident.load_state_dict(model.state_dict())
    model = resident
    packer, inputs, _, _ = request(model, audio=False)
    packer.num_frame_per_block, packer.split_first_latent = [3, 2], True
    inputs.visual_shape = (10, 2, 2)
    inputs.source_visual = torch.randn(1, 2, 10, 2, 2)
    inputs.visual = {"ref": torch.randn(1, 2, 1, 2, 2)}
    inputs.media = [{"id": "ref", "type": "image"}]
    inputs.conditions = [{"content_id": "ref", "role": "reference"}]
    noise = torch.randn_like(inputs.source_visual)
    options = dict(
        solver="few_step",
        denoising_step_list=mixed_grids[0],
        initial_denoising_step_list=mixed_grids[1],
        initial_block_count=3,
        guidance_scale=2,
        ar_condition_timestep=0,
        reference_visual_timestep=0.999,
        joint_forward="off",
    )
    baseline = ZingH3RealtimeSampler(model, packer, **options)
    plans = [
        video_plan("few_step", s.timesteps, 0, 0.999, sigmas=s.sigmas)
        for s in (baseline.video_scheduler, baseline.initial_schedulers[0])
    ]
    plan = initial_blocks_plan(*plans, 3)
    # Deduplicate overlap, including when neither grid is a subset of the other.
    expected_batches = {tuple(batch) for part in plans for batch in part["batches"]}
    assert {tuple(batch) for batch in plan["batches"]} == expected_batches
    assert len(plan["batches"]) == len(expected_batches)
    path = tmp_path / "table"
    write_table(
        path,
        arch=model.arch,
        tags=model.modality_tags,
        embedder=model.time_embedder,
        projections=[
            *((i, b.adaln_proj) for i, b in enumerate(model.blocks)),
            ("final", model.final_layer.adaln_proj),
        ],
        plan=plan,
        identity={"synthetic": True},
        provenance={"fixture": True},
    )
    table = ZingAdalnTable(
        path, arch=model.arch, tags=model.modality_tags, max_bytes=1 << 20
    )
    verify_synthetic_table(table, tmp_path)
    with torch.device("meta"):
        candidate = ZingH3DiTModel(
            ZingH3DiTConfig(arch_config=replace(model.arch)), {}, adaln_table=table
        )
    assert not any("adaln_proj" in n for n, _ in candidate.named_parameters())
    candidate.to_empty(device="cpu").eval()
    candidate.load_state_dict(
        {k: v for k, v in model.state_dict().items() if "adaln_proj" not in k}
    )
    sampler = ZingH3RealtimeSampler(candidate, packer, **options)
    return baseline, sampler, inputs, noise, table


def assert_same_kv(first, second, layers):
    for layer in range(layers):
        for a, b in zip(first.history(layer), second.history(layer)):
            assert torch.equal(a, b)


@pytest.mark.parametrize("joint", ["source_first", "cross_chunk"])
@pytest.mark.parametrize("configuration", ["profile", "cli"])
@pytest.mark.parametrize(
    "mixed_grids",
    [
        pytest.param(([1000, 500], [1000, 750, 500, 250]), id="default-4-to-2"),
        pytest.param(([1000, 500], [1000, 800, 600, 200]), id="custom-4-to-2"),
        pytest.param(([1000], [1000, 800, 600]), id="3-to-1"),
        pytest.param(([1000, 600, 200], [1000]), id="1-to-3"),
    ],
    indirect=True,
)
@torch.no_grad()
def test_source_custom_plan_matches_dynamic_cache(
    configured,  # noqa: F811
    monkeypatch,
    static_mixed,
    joint,
    configuration,
    mixed_grids,
):
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters import (
        zing_h3_source_adapter as source,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs
    from sglang.multimodal_gen.test.unit.realtime.test_zing_h3_source_adapter import (
        initialize,
    )

    regular, initial = mixed_grids
    expected = dict(
        solver="few_step",
        denoising_step_list=regular,
        initial_block_count=3,
        initial_denoising_step_list=initial,
    )
    path = Path(configured.model_path) / "zing_realtime_profile.json"
    profile = json.loads(path.read_text())
    profile["joint_forward"] = joint
    if configuration == "profile":
        profile.update(expected)
    else:
        # CLI must replace the fixture's explicit single-stage four-step profile.
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed = parser.parse_args(
            [
                "--model-path",
                configured.model_path,
                "--zing-denoising-step-list",
                ",".join(map(str, regular)),
                "--zing-initial-denoising-step-list",
                ",".join(map(str, initial)),
            ]
        )
        ServerArgs._validate_zing_sampling(parsed)
        for field in expected:
            setattr(configured, "zing_" + field, getattr(parsed, "zing_" + field))
        configured.is_arg_explicitly_set = lambda name: (
            name in {"zing_denoising_step_list", "zing_initial_denoising_step_list"}
        )
    path.write_text(json.dumps(profile))
    adapter, api_session = initialize()
    assert adapter.ready["sampling_schedule"] == expected
    assert adapter.ready["num_inference_steps"] == len(regular)
    monkeypatch.setattr(
        source,
        "build_realtime_sampling_params",
        lambda *a, **kw: SimpleNamespace(guidance_scale=1, **kw),
    )
    params = adapter.build_sampling_params(
        api_session,
        configured,
        SimpleNamespace(request_id="test"),
        SimpleNamespace(prompt="snow"),
        17,
    )
    baseline, original, inputs, noise, table = static_mixed
    baseline.generator.adaln_projection_cache_enabled = True
    # Exercise the same internal command that is serialized to the worker.
    monkeypatch.setattr(
        source.BaseRealtimeModelAdapter,
        "prepare_next_request",
        lambda *a: SimpleNamespace(extra={}, block_idx=0),
    )
    batch = adapter.prepare_next_request(api_session, configured, None)
    command = msgspec.msgpack.decode(msgspec.msgpack.encode(batch.extra))[
        "zing_source_command"
    ]
    assert command["sampling_schedule"] == expected
    stage = SimpleNamespace(
        modules={"transformer": original.generator},
        packer=original.processor,
        runtime=dict(
            timestep_shift=5,
            audio_timestep_shift=3,
            num_train_timestep=1000,
            ar_condition_timestep=0,
            reference_visual_timestep=0.999,
        ),
    )
    sampler = ZingH3GenerationStage._build_realtime_sampler(
        stage, params, source_sampling_schedule=command["sampling_schedule"]
    )
    baseline.guidance_scale, baseline.do_cfg, baseline.joint_forward = 1, False, joint
    sessions = [ready_session(s, inputs, noise) for s in (baseline, sampler)]
    iterators = [s.drain_ready() for s in sessions]
    steps = []
    try:
        for index in range(5):
            results = [next(it) for it in iterators]
            assert torch.equal(results[0].visual, results[1].visual)
            assert_same_kv(sessions[0].cache, sessions[1].cache, 2)
            assert torch.equal(sessions[0].rng.cpu_rng, sessions[1].rng.cpu_rng)
            receipt = sampler.last_run_metrics["last_block_schedule"]
            assert receipt["noise_grid"] == (initial if index < 3 else regular)
            steps.append(
                sampler.last_run_metrics["last_block_schedule"]["denoise_steps"]
            )
        assert steps == [len(initial)] * 3 + [len(regular)] * 2
        assert sessions[0].cache.adaln_projection_cache.hits > 0
        for it in iterators:
            with pytest.raises(StopIteration):
                next(it)
        if joint == "cross_chunk":
            assert sampler.last_run_metrics["forward_counts"]["cross_chunk"] == 4
    finally:
        for it in iterators:
            it.close()
        for session in sessions:
            session.dispose()
        adapter.dispose(api_session)
    assert table.active_plans == 0 and table.stats()["fallback"] == 0
    # A valid single plan must still fail against a mixed-plan static table.
    single = {
        **adapter.sampling_schedule,
        "initial_block_count": 0,
        "initial_denoising_step_list": None,
    }
    mismatched = ZingH3GenerationStage._build_realtime_sampler(
        stage, params, source_sampling_schedule=single
    )
    with pytest.raises(ValueError, match="schedule"):
        mismatched.initialize(inputs, noise)
    assert table.active_plans == 0
    # Even a different valid schedule with no new timestep must be rejected.
    changed = {**expected, "initial_denoising_step_list": regular}
    mismatched = ZingH3GenerationStage._build_realtime_sampler(
        stage, params, source_sampling_schedule=changed
    )
    with pytest.raises(ValueError, match="schedule"):
        mismatched.initialize(inputs, noise)
    assert table.active_plans == 0


@pytest.mark.parametrize("joint", ["off", "source_first", "cross_chunk"])
@torch.no_grad()
def test_static_mixed_matches_resident_and_resets_generation(static_mixed, joint):
    baseline, sampler, inputs, noise, table = static_mixed
    for s in (baseline, sampler):
        s.joint_forward = joint
        if joint != "off":
            s.guidance_scale, s.do_cfg = 1, False
    for _ in range(2):
        states = [s.initialize(inputs, noise) for s in (baseline, sampler)]
        try:
            for index in range(len(states[0].spans)):
                for s, state in zip((baseline, sampler), states):
                    torch.manual_seed(100 + index)
                    # This goes through step's temporary initial-scheduler binding.
                    s.step(state)
                assert torch.equal(states[0].visual, states[1].visual)
                assert torch.isfinite(states[1].visual).all()
                assert_same_kv(states[0].cache, states[1].cache, 2)
                assert table.active_plans == 1
            assert [
                r["denoise_steps"] for r in sampler.last_run_metrics["block_schedules"]
            ] == [4, 4, 4, 2, 2]
            if joint != "off":
                assert sampler.last_run_metrics["forward_counts"]["source_first"] == 5
                # Finite step() cannot look ahead into another streaming call.
                assert sampler.last_run_metrics["forward_counts"]["cross_chunk"] == 0
        finally:
            for state in states:
                state.close()
        assert table.active_plans == 0
    assert table.hits > 0 and table.stats()["fallback"] == 0


def ready_session(sampler, inputs, noise, **kwargs):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )
    from sglang.multimodal_gen.test.unit.test_zing_h3_chunks import fixture_provenance

    session = SourceReadySession(
        sampler,
        inputs,
        seed=42,
        max_pending_chunks=10,
        max_pending_bytes=1000000,
        total_latent_frames=noise.shape[2],
        **kwargs,
    )
    for chunk in sampler.processor.chunk_plan.chunks(noise.shape[2]):
        session.append(
            inputs.source_visual[:, :, chunk.start : chunk.end],
            noise[:, :, chunk.start : chunk.end],
            provenance=fixture_provenance(
                chunk.chunk_id + 1, 20, chunk.start, chunk.end
            ),
        )
    return session


@pytest.mark.parametrize("mode", ["step", "drain"])
@pytest.mark.parametrize("joint", ["off", "source_first", "cross_chunk"])
@pytest.mark.parametrize("count", [0, 3])
@torch.no_grad()
def test_static_ready_matches_resident_and_releases(static_mixed, mode, joint, count):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import batch_key

    baseline, sampler, inputs, noise, table = static_mixed
    for s in (baseline, sampler):
        s.joint_forward = joint
        if joint != "off":
            s.guidance_scale, s.do_cfg = 1, False
        if not count:
            s.initial_block_count = 0
            s.initial_schedulers = None
    if not count:
        table.manifest["schedule"].pop("initial_blocks")
    # [0, 1] can exist as a prefix variant, but this fixture's prefix uses
    # [0, 0.999]. Joint must only look up singleton projections, even if the
    # unused combined clean/noisy batch is absent.
    first_t = sampler.regular_schedulers[0].timesteps[0].item()
    combined_key = batch_key(torch.unique(torch.tensor([1.0, first_t]), sorted=True))
    table.keys.pop(combined_key, None)
    sessions = [ready_session(s, inputs, noise) for s in (baseline, sampler)]
    iterators = [s.drain_ready() for s in sessions] if mode == "drain" else None
    actual_steps = []
    try:
        for index in range(5):
            if mode == "step":
                if index == 1:
                    prompt = dict(inputs.positive)
                    prompt["context"] = prompt["context"] * 0.5
                    for session in sessions:
                        session.update_prompt(prompt, 1)
                results = [s.step() for s in sessions]
            else:
                results = [next(it) for it in iterators]
            assert torch.equal(results[0].visual, results[1].visual)
            assert_same_kv(sessions[0].cache, sessions[1].cache, 2)
            assert table.active_plans == 1
            actual_steps.append(
                sampler.last_run_metrics["last_block_schedule"]["denoise_steps"]
            )
        if iterators:
            for it in iterators:
                with pytest.raises(StopIteration):
                    next(it)
        for session in sessions:
            session.finish()
        assert table.active_plans == 0
        assert actual_steps == ([4, 4, 4, 2, 2] if count else [2] * 5)
        for s in (baseline, sampler):
            counts = s.last_run_metrics["forward_counts"]
            assert counts.get("initial_joint", 0) == int(joint != "off")
            if joint == "cross_chunk" and mode == "drain":
                # Includes the third -> fourth block boundary, not a fallback.
                assert counts["cross_chunk"] == 4
                assert counts["source_first"] == 0
            elif joint != "off":
                assert counts["source_first"] == 4
                assert counts["cross_chunk"] == 0
        assert table.hits > 0 and table.stats()["fallback"] == 0
    finally:
        if iterators:
            for it in iterators:
                it.close()
        for session in sessions:
            session.dispose()


@pytest.mark.parametrize("entry", ["finite", "ready"])
@pytest.mark.parametrize("joint", ["off", "source_first", "cross_chunk"])
@pytest.mark.parametrize(
    "fault", ["single", "sigma", "count", "missing", "prefix", "clean", "pool"]
)
@torch.no_grad()
def test_static_mixed_rejects_before_kv_and_cleans_up(
    static_mixed, entry, joint, fault
):
    from sglang.multimodal_gen.runtime.models.dits.zing_h3_adaln_table import batch_key

    _, sampler, inputs, noise, table = static_mixed
    sampler.joint_forward = joint
    if joint != "off":
        sampler.guidance_scale, sampler.do_cfg = 1, False
    schedule = table.manifest["schedule"]
    if fault == "single":
        table.manifest["schedule"] = schedule["initial_blocks"]["schedule"]
    elif fault == "sigma":
        schedule["initial_blocks"]["schedule"]["video_sigmas"]["raw_hex"] = "00"
    elif fault == "count":
        schedule["initial_blocks"]["count"] = 2
    elif fault == "missing":
        t = sampler.initial_schedulers[0].timesteps[2].item()
        del table.keys[batch_key(torch.tensor([t]))]
    elif fault == "prefix":
        del table.keys[batch_key(torch.tensor([0.0, 0.999]))]
    elif fault == "clean":
        del table.keys[batch_key(torch.tensor([1.0]))]

    def acquire(*_):
        assert fault == "pool", "Incompatible AdaLN plan acquired KV storage"
        raise RuntimeError("synthetic pool exhaustion")

    expected = RuntimeError if fault == "pool" else ValueError
    if entry == "finite":
        with pytest.raises(expected):
            sampler.initialize(inputs, noise, acquire_storage=acquire)
    else:
        session = ready_session(sampler, inputs, noise, acquire_storage=acquire)
        with pytest.raises(expected):
            session.step()
        assert session.closed and session.cache is None
    assert table.active_plans == 0 and table.hits == 0


@pytest.mark.parametrize("joint", ["off", "source_first", "cross_chunk"])
@torch.no_grad()
def test_static_ready_cancel_releases_plan(static_mixed, joint):
    _, sampler, inputs, noise, table = static_mixed
    sampler.joint_forward = joint
    if joint != "off":
        sampler.guidance_scale, sampler.do_cfg = 1, False
    session = ready_session(sampler, inputs, noise)
    iterator = session.drain_ready()
    next(iterator)
    assert table.active_plans == 1
    iterator.close()
    assert session.closed and table.active_plans == 0
    session.dispose()
    assert table.active_plans == 0


@pytest.mark.parametrize("count", [0, 3])
@pytest.mark.parametrize(
    "mixed_grids",
    [([1000, 500], [1000, 750, 500, 250]), ([1000, 600, 200], [1000])],
    indirect=True,
)
def test_table_builder_cli_plan(
    static_mixed, tmp_path, monkeypatch, count, mixed_grids
):
    import importlib.util
    import json
    from pathlib import Path

    _, _, _, _, table = static_mixed
    path = (
        Path(__file__).resolve().parents[5]
        / "benchmark/zing_h3_causal/build_adaln_table.py"
    )
    spec = importlib.util.spec_from_file_location("mixed_table_builder", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    output = tmp_path / "plan.json"
    argv = [
        str(path),
        "plan",
        "--solver",
        "few_step",
        "--denoising-step-list",
        *map(str, mixed_grids[0]),
        "--condition",
        "0",
        "--reference",
        "0.999",
        "--output",
        str(output),
    ]
    if count:
        argv += [
            "--initial-block-count",
            "3",
            "--initial-denoising-step-list",
            *map(str, mixed_grids[1]),
        ]
    monkeypatch.setattr("sys.argv", argv)
    builder.main()
    plan = json.loads(output.read_text())
    expected = dict(table.manifest["schedule"])
    if not count:
        expected.pop("initial_blocks")
    assert plan["schedule"] == expected
    if count:
        assert len(plan["batches"]) == len(table.keys)
