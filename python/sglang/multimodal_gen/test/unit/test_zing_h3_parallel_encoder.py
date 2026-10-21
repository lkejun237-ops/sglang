# SPDX-License-Identifier: Apache-2.0
"""Real CPU collectives and CNNs; CUDA/BF16 acceptance remains separate."""

import faulthandler
import gc
import os
import sys
import weakref
from datetime import timedelta
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae import (
    AutoencoderKL,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.parallel_encoder import (
    create_encoder_cache,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.progressive import (
    CausalEncoderCache,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_cnn import (
    EncoderFCN3D,
)


def _model(stack):
    torch.manual_seed(7)
    model = nn.Module()
    model.encoder = (
        EncoderFCN3D(
            ch=32,
            ch_mult=[1, 1, 1],
            space_down=[1, 2, 2],
            time_down=[1, 2, 2],
            num_res_blocks=1,
            in_channels=3,
            z_channels=24,
            double_z=True,
            causal=True,
            use_t_isolated_gn=True,
        )
        .double()
        .eval()
    )
    model.quant_conv = nn.Conv3d(48, 48, 1).double().eval()
    model.encoder_tiling = True
    model.stack_tiling = stack
    model.tile_size, model.tile_overlap_min, model.vae_ratio = 16, 8, 4
    model._blend_weight_cache = {}
    for name in ("split_tiles", "blend", "_assemble_tiles", "_local_tile_indices"):
        setattr(model, name, MethodType(getattr(AutoencoderKL, name), model))
    return model


def _worker(rank, world, rendezvous, stack):
    faulthandler.enable(all_threads=True)

    def stage(name):
        print(
            f"[parallel-encoder rank={rank}/{world} stack={stack}] {name}",
            file=sys.stderr,
            flush=True,
        )

    stage("initializing")
    torch.set_num_threads(1)
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import klvae

    original_state = klvae.get_tile_parallel_state
    original_group = klvae.get_decode_parallel_group_coordinator
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=90),
    )
    group_ref = weakref.ref(dist.group.WORLD)
    stage("initialized")
    try:
        model = _model(stack)
        # Force a conflicting serving group; explicit worker injection must win.
        klvae.get_tile_parallel_state = lambda: (0, 1)
        klvae.get_decode_parallel_group_coordinator = lambda: (_ for _ in ()).throw(
            AssertionError("Serving group must not be consulted")
        )
        ref = CausalEncoderCache(model, tile_mode="replicated")
        os.environ["MINIMAX_H3_PROGRESSIVE_ENCODER_TILING"] = "parallel"
        candidate = create_encoder_cache(
            model,
            dist.group.WORLD,
            control_group=dist.group.WORLD,
            encode_mode="deterministic",
            encode_dtype="float64",
        )
        assert candidate.available_group.device_group is dist.group.WORLD
        assert type(candidate) is CausalEncoderCache
        # Include a second clip, temporal phase changes, and an idle-rank case.
        for size in (24, 16):
            candidate = create_encoder_cache(
                model,
                dist.group.WORLD,
                control_group=dist.group.WORLD,
                encode_mode="deterministic",
                encode_dtype="float64",
            )
            for _ in range(2):
                for frames in (1, 8, 8):
                    x = torch.randn(1, 3, frames, size, size, dtype=torch.float64)
                    with torch.no_grad():
                        expected, state, work = ref.encode(x)
                        actual, pending, count = candidate.encode(x)
                    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                    ref.commit(state, work)
                    candidate.commit(pending, count)
                    total = torch.tensor(count)
                    dist.all_reduce(total)
                    assert total.item() == work * (
                        1
                        if candidate.last_execution["effective_mode"] == "parallel"
                        else world
                    )
                ref.clear()
                candidate.clear()
                assert not candidate.states
        stage("cache checks complete")
        # Exercise the merged session contract, including policy validation,
        # rank agreement, execution metadata, EOS and cache commit/clear.
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
            H3ProgressiveEncoderSession,
        )

        codecs = SimpleNamespace(
            modules={"video_vae": model},
            vae_frames_per_chunk=17,
            vae_latents_per_chunk=5,
            temporal_ratio=4,
            runtime={
                "vae_config": {
                    "encode_mode": "deterministic",
                    "encode_dtype": "float64",
                }
            },
        )
        reference = H3ProgressiveEncoderSession(
            codecs,
            capacity_frames=34,
            cache=CausalEncoderCache(model, tile_mode="replicated"),
        )
        session = H3ProgressiveEncoderSession(
            codecs,
            capacity_frames=34,
            cache=create_encoder_cache(
                model,
                dist.group.WORLD,
                control_group=dist.group.WORLD,
                encode_mode="deterministic",
                encode_dtype="float64",
            ),
        )
        pixels = torch.randint(0, 256, (3, 10, 24, 24), dtype=torch.uint8)
        for item in (reference, session):
            item.append(pixels, frame_start=0)
            item.finish()
        while reference.ready:
            expected, actual = reference.pop(), session.pop()
            torch.testing.assert_close(
                actual.values, expected.values, atol=1e-12, rtol=1e-12
            )
            assert session.cache.last_execution["latent_end"] == actual.end
            assert actual.source_dependency_end == expected.source_dependency_end
        assert not session.ready
        stage("session checks complete")
        reference.dispose()
        session.dispose()
        stage("sessions disposed")
        # dispose()/clear() release tensors, but caches still own Gloo group
        # references. Drop every owner (including the loop alias) while Python
        # is alive, before destroying the group and exiting the spawned worker.
        del item, reference, session, ref, candidate, codecs, model
        gc.collect()
        dist.monitored_barrier(timeout=timedelta(seconds=30), wait_all_ranks=True)
        stage("fixtures released; ranks synchronized")
    finally:
        klvae.get_tile_parallel_state = original_state
        klvae.get_decode_parallel_group_coordinator = original_group
        stage("destroying process group")
        dist.destroy_process_group()
        stage("process group destroyed")
    # Keep this outside finally so an original assertion/encoding failure is
    # never replaced by the success-path lifecycle check.
    gc.collect()
    assert group_ref() is None, "Encoder fixtures retained the process group"
    stage("worker complete")


@pytest.mark.parametrize("world,stack", [(1, True), (2, False), (2, True), (4, True)])
def test_distributed_tiles_match_unmodified_cnn(tmp_path, world, stack):
    mp.spawn(
        _worker,
        args=(world, "file://" + str(tmp_path / "rdzv"), stack),
        nprocs=world,
        join=True,
    )


def test_worker_preserves_model_setup_error(tmp_path, monkeypatch):
    def fail_model(stack):
        raise RuntimeError("Injected model setup failure")

    monkeypatch.setattr(sys.modules[__name__], "_model", fail_model)
    with pytest.raises(RuntimeError, match="Injected model setup failure"):
        _worker(0, 1, "file://" + str(tmp_path / "rdzv"), True)
    assert not dist.is_initialized()
