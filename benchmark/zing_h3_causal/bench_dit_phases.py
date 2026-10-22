# SPDX-License-Identifier: Apache-2.0
"""Resident SP1 checkpoint-subset DiT benchmark using trusted frozen conditioning.

Run once per GPU/process. Encoding, decoding and network delivery are excluded.
The input is a locally generated H3InferenceInputs snapshot, never user uploads.
"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

from dit_phase_timing import PhaseTimer, check_forward_rows, summarize


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def keep_layer(name, layers):
    match = re.search(r"(?:^|\.)(?:transformer_blocks|blocks)\.(\d+)\.", name)
    # Token-refiner depth is independent and stays unchanged.
    if "token_refiner." in name:
        return True
    return match is None or int(match[1]) < layers


def check_conditioning(inputs, arch, vae_arch, runtime):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.media_input import (
        ZingH3MediaInput,
    )

    geometry = SimpleNamespace(
        spatial_ratio=vae_arch.spatial_compression_ratio,
        temporal_ratio=vae_arch.temporal_compression_ratio,
        vae_frames_per_chunk=vae_arch.vae_clip_length,
        vae_latents_per_chunk=math.ceil(
            vae_arch.vae_clip_length / vae_arch.temporal_compression_ratio
        ),
        patch_size=arch.patch_size,
        max_short_edge=runtime["h3_media_config"]["max_short_edge"],
    )
    height, width, frames, pixels = ZingH3MediaInput._inference_geometry(
        geometry, dict(type="video", output=dict(frames=120, height=480, width=832))
    )
    expected = (
        frames,
        height // geometry.spatial_ratio,
        width // geometry.spatial_ratio,
    )
    if (
        inputs.audio_shape is not None
        or inputs.source_visual is None
        or len(inputs.visual) != 6
    ):
        raise ValueError("Expected video-only source plus six-reference conditioning")
    if inputs.pixel_frame_count != pixels or tuple(inputs.visual_shape) != expected:
        raise ValueError(
            f"Expected 120-frame 832x480 conditioning with latent geometry {expected}; got pixels={inputs.pixel_frame_count}, latent={inputs.visual_shape}"
        )
    if tuple(inputs.source_visual.shape) != (1, arch.latents_dim, *expected):
        raise ValueError("Source and target latent geometry differ")
    return expected


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="Exported checkpoint root")
    p.add_argument(
        "--conditioning",
        type=Path,
        required=True,
        help="Trusted local H3InferenceInputs torch snapshot",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--measured-runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window-chunks", type=int, default=6)
    p.add_argument("--sink-chunks", type=int, default=3)
    p.add_argument(
        "--profile-cycle",
        action="store_true",
        help="After timing, collect an independent last complete 3+2 cycle CPU/CUDA trace",
    )
    p.add_argument(
        "--denoising-step-list", nargs="+", type=int, default=[1000, 750, 500, 250]
    )
    p.add_argument(
        "--attention-backend", choices=("torch_sdpa", "fa4"), default="torch_sdpa"
    )
    return p


def run(args):
    if args.layers < 1 or args.warmup_runs < 1 or args.measured_runs < 1:
        raise ValueError("layers, warmup-runs and measured-runs must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    # Eager, separate source/denoise/commit calls are an explicit comparison group.
    os.environ.update(
        SGLANG_ZING_EXPERIMENTAL_ATTENTION=args.attention_backend,
        SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES="0",
        SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN="0",
        SGLANG_ZING_DENOISE_CUDA_GRAPH="0",
        SGLANG_ZING_CHUNK_CUDA_GRAPH="0",
        SGLANG_ZING_ADALN_PROJECTION_CACHE="1",
        SGLANG_ZING_ADALN_PROJECTION_CACHE_BYTES="134217728",
        SGLANG_ZING_EXPERIMENTAL_FA4_METADATA_REUSE="1",
    )
    import torch
    from safetensors import safe_open

    from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
        MiniMaxH3VideoVAEConfig,
    )
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import ZingH3DiTConfig
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
    from sglang.multimodal_gen.runtime.models.dits.zing_h3 import ZingH3DiTModel
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
        ZingH3Packer,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
        ZingH3Sampler,
    )

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This benchmark supports SP1 only")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    torch.cuda.set_device(0)
    config = ZingH3DiTConfig()
    transformer = args.model / "transformer"
    config.update_model_arch(json.loads((transformer / "config.json").read_text()))
    full_depth = config.arch_config.num_layers
    if args.layers > full_depth:
        raise ValueError("Requested depth exceeds checkpoint depth")
    config.arch_config.num_layers = args.layers
    runtime = json.loads((args.model / "zing_config.json").read_text())
    inputs = torch.load(args.conditioning, map_location="cpu", weights_only=False)
    vae_config = MiniMaxH3VideoVAEConfig()
    vae_config.update_model_arch(
        json.loads((args.model / "video_vae/config.json").read_text())
    )
    latent_shape = check_conditioning(
        inputs, config.arch_config, vae_config.arch_config, runtime
    )
    inventory = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version,power.limit,clocks.sm,clocks.mem,temperature.gpu",
            "--format=csv",
        ],
        text=True,
    )
    topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    import importlib.metadata

    versions = {}
    for name in ("flash-attn-4", "triton", "numpy", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    receipt = dict(
        scope="real-checkpoint reduced-depth DiT; frozen real conditioning; no codecs or server",
        layers=args.layers,
        layer_ids=list(range(args.layers)),
        full_depth=full_depth,
        sp=1,
        graph=False,
        compile_adaln=False,
        joint_forward="off",
        offload=False,
        attention_backend=args.attention_backend,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        nccl=torch.cuda.nccl.version(),
        versions=versions,
        device=torch.cuda.get_device_name(0),
        cc=list(torch.cuda.get_device_capability(0)),
        inventory=inventory,
        topology=topology,
        conditioning_sha256=sha256(args.conditioning),
        model_config_sha256=sha256(transformer / "config.json"),
        runtime_config_sha256=sha256(args.model / "zing_config.json"),
        vae_config_sha256=sha256(args.model / "video_vae/config.json"),
        latent_shape=latent_shape,
        rope_config=runtime["rope_config"],
        source_revision=os.environ.get("SGLANG_COMMIT", "unknown"),
        source_archive_sha256=os.environ.get("SGLANG_BENCH_SOURCE_ARCHIVE_SHA256"),
        kernel_route_envs={
            name: os.environ.get("SGLANG_ZING_" + name)
            for name in (
                "FUSED_HISTORY_ROPE",
                "GROUPED_HISTORY_ROPE",
                "CHUNK_ROPE_REUSE",
                "HISTORY_ROPE_PLAN",
                "FUSED_CURRENT_ROPE",
                "FUSED_CURRENT_ROPE_KV",
                "FA4_FIXED",
            )
        },
        options={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        chunk_plan=dict(num_frame_per_block=[3, 2], split_first_latent=True),
        timing="block: sync only at chunk boundaries; forward: isolated synchronized calls; CUDA event is current-stream elapsed",
        kv_rows="logical packed rows across modalities, before SP partition; not a kernel-specific Q/KV shape",
    )
    (args.output / "config.json").write_text(json.dumps(receipt, indent=2))
    # The iterator filters keys before get_tensor, so unselected blocks are never
    # materialized. Hash the exact selected checkpoint bytes for cross-GPU reuse.
    selected = {}

    def weights():
        for path in sorted(transformer.glob("*.safetensors")):
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if keep_layer(key, args.layers):
                        tensor = handle.get_tensor(key)
                        selected[key] = dict(
                            shape=list(tensor.shape),
                            dtype=str(tensor.dtype),
                            sha256=hashlib.sha256(
                                tensor.contiguous().view(torch.uint8).numpy().tobytes()
                            ).hexdigest(),
                        )
                        yield key, tensor

    torch.cuda.reset_peak_memory_stats()
    model = maybe_load_fsdp_model(
        model_cls=ZingH3DiTModel,
        init_params=dict(config=config, hf_config={}),
        weight_dir_list=[],
        device=torch.device("cuda:0"),
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        pin_cpu_memory=False,
        strict=True,
        weights_iterator=weights(),
    )
    receipt.update(
        selected_weights=selected,
        parameters_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
        load_peak_allocated=torch.cuda.max_memory_allocated(),
        arch=json.loads((transformer / "config.json").read_text()),
        effective_backend=model.experimental_attention_backend,
        adaln_projection_cache=model.adaln_projection_cache_enabled,
        adaln_projection_cache_bytes=model.adaln_projection_cache_bytes,
        fa4_metadata_reuse=model.experimental_fa4_metadata_reuse,
    )
    (args.output / "config.json").write_text(json.dumps(receipt, indent=2))
    for name in ("positive", "visual", "audio"):
        setattr(inputs, name, {k: v.cuda() for k, v in getattr(inputs, name).items()})
    inputs.source_visual = inputs.source_visual.cuda()
    packer = ZingH3Packer(
        model.arch,
        num_frame_per_block=[3, 2],
        split_first_latent=True,
        spatial_extent=runtime["rope_config"]["spatial_extent"],
        frame_scale=runtime["rope_config"]["frame_scale"],
        visual_token_frame_spans=runtime["rope_config"]["visual_token_frame_spans"],
    )
    sampler = ZingH3Sampler(
        model,
        packer,
        solver="few_step",
        denoising_step_list=args.denoising_step_list,
        num_train_timesteps=runtime["num_train_timestep"],
        video_shift=runtime["timestep_shift"],
        audio_shift=runtime["audio_timestep_shift"],
        guidance_scale=1,
        window_chunks=args.window_chunks,
        sink_chunks=args.sink_chunks,
        joint_forward="off",
        ar_condition_timestep=model.arch.ar_condition_timestep,
        reference_visual_timestep=model.arch.reference_visual_timestep,
    )
    rows, runs = [], []
    expected_output_sha256 = None
    # Separate whole-block baseline from intrusive per-forward isolation.
    for mode in ("block", "forward"):
        for index in range(args.warmup_runs + args.measured_runs):
            run_kind = "warmup" if index < args.warmup_runs else "measured"
            torch.manual_seed(args.seed)
            visual = torch.randn(
                (1, model.arch.latents_dim, *inputs.visual_shape), device="cuda"
            )
            torch.cuda.synchronize()
            ready = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            with (
                torch.no_grad(),
                PhaseTimer(sampler, mode=mode, run=index, run_kind=run_kind) as timer,
            ):
                output, audio = sampler.sample(inputs, visual)
            if mode == "forward":
                check_forward_rows(
                    timer.rows,
                    sampler.last_run_metrics["block_spans"],
                    len(args.denoising_step_list),
                    sampler.last_run_metrics["forward_counts"],
                )
            # Finite checks are outside every timing boundary.
            if audio is not None or not bool(torch.isfinite(output).all()):
                raise ValueError("Invalid video-only latent output")
            output_sha256 = hashlib.sha256(
                output.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            if expected_output_sha256 is None:
                expected_output_sha256 = output_sha256
            elif output_sha256 != expected_output_sha256:
                raise ValueError("Output changed across repetitions or timing modes")
            rows.extend(timer.rows)
            runs.append(
                dict(
                    mode=mode,
                    run=index,
                    run_kind=run_kind,
                    finite=True,
                    output_sha256=output_sha256,
                    shape=list(output.shape),
                    ready_allocated=ready,
                    peak_allocated=torch.cuda.max_memory_allocated(),
                    peak_reserved=torch.cuda.max_memory_reserved(),
                    forward_counts=sampler.last_run_metrics["forward_counts"],
                )
            )
            with (args.output / "samples.jsonl").open("a") as stream:
                for row in timer.rows:
                    stream.write(json.dumps(row) + "\n")
            (args.output / "runs.json").write_text(json.dumps(runs, indent=2))
            print("DIT_PHASE_RUN " + json.dumps(runs[-1]), flush=True)
            del output, visual
    summary = summarize(rows)
    if not summary["cycles"]:
        raise ValueError("No complete stable 3+2 cycle was measured")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.profile_cycle:
        from dit_phase_profile import profile_cycle

        profile_cycle(
            sampler,
            inputs,
            args,
            summary["cycles"][-1]["chunks"],
            expected_output_sha256,
        )
    print(
        "DIT_PHASE_COMPLETE "
        + json.dumps(dict(groups=len(summary["groups"]), cycles=len(summary["cycles"])))
    )


if __name__ == "__main__":
    run(parser().parse_args())
