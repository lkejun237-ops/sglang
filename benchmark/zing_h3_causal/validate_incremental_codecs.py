# SPDX-License-Identifier: Apache-2.0
"""Real-weight codec acceptance against the existing finite H3 path.

No DiT or independent-native quality claim. Synthetic, nonrepeating RGB input
exercises real learned VAE weights; source moments avoid changing posterior RNG.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import (
    _consume_vae_checkpoint_arch_metadata,
)
from sglang.multimodal_gen.runtime.loader.utils import load_model_state_dict
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
    ZingH3Codecs,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.streaming_codecs import (
    SourcePixelBuffer,
    StableVideoDecoder,
)

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.streaming_noise import (
    STREAMING_NOISE_VERSION,
    indexed_video_noise,
)
from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
    PixelRange,
    Provenance,
    SourceSpan,
)


def source_provenance(pixels, dependency):
    spans = []
    start = pixels.start
    while start < pixels.end:
        end = min(pixels.end, (start // 17 + 1) * 17)
        spans.append(SourceSpan(start // 17 + 1, start, start, end - start))
        start = end
    return Provenance(pixels, tuple(spans), dependency)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(actual, expected):
    a, b = actual.float().cpu(), expected.float().cpu()
    delta = a - b
    result = dict(
        max_abs=delta.abs().max().item(),
        rrmse=(delta.square().sum() / b.square().sum().clamp_min(1e-30)).sqrt().item(),
    )
    # Same implementation, weights and precision: require exact equivalence.
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    return result


def codec_soak(c, frames, output):
    """Measure codec ownership using real kernels and indexed posterior sampling.

    No full-video inputs, latents or outputs are retained. This deliberately
    does not exercise the production AR sampler or service transport.
    """
    source = SourcePixelBuffer(c, capacity_frames=34)
    decoder = StableVideoDecoder(c, capacity_latents=16)
    observations = []
    output_hash = hashlib.sha256()
    emitted = 0

    def accept_output(result):
        nonlocal emitted
        assert result.start == emitted
        emitted = result.end
        output_hash.update(result.values.cpu().numpy().tobytes())

    def handoff(result):
        latent = c.sample_streaming_visual_moments(
            result.values,
            latent_start=result.start,
            noise_version=STREAMING_NOISE_VERSION,
        )
        decoder.append(
            latent,
            latent_start=result.start,
            source_dependency_end=result.source_dependency_end,
        )
        while decoder.ready:
            accept_output(decoder.pop())

    for start in range(0, frames, 17):
        end = min(frames, start + 17)
        t = torch.arange(start, end).view(1, -1, 1, 1)
        y = torch.arange(64).view(1, 1, -1, 1)
        x = torch.arange(64).view(1, 1, 1, -1)
        channel = torch.arange(3).view(3, 1, 1, 1)
        pixels = ((x + 3 * y + 7 * t + channel * 61 + (t * t) // 13) % 256).byte()
        source.append(pixels, frame_start=start)
        while source.ready:
            handoff(source.pop())
        torch.cuda.synchronize()
        # Current RSS, not peak RSS (ru_maxrss would hide releases).
        rss_pages = int(Path("/proc/self/statm").read_text().split()[1])
        observations.append(
            dict(
                accepted_frames=end,
                emitted_frames=emitted,
                source_buffered=source.buffered_frames,
                decoder_buffered=decoder.buffered_latents,
                allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(),
                host_rss=rss_pages * os.sysconf("SC_PAGE_SIZE"),
            )
        )
    source.finish()
    for result in source.drain():
        handoff(result)
    decoder.finish(total_frames=frames)
    for result in decoder.drain():
        accept_output(result)
    assert emitted == frames
    source.dispose()
    decoder.dispose()
    assert source._pixels is decoder._latent is None
    assert not decoder._dependencies
    steady = observations[20:]
    assert steady, "Soak requires more than 20 native clips"
    allocated_span = max(r["allocated"] for r in steady) - min(
        r["allocated"] for r in steady
    )
    host_span = max(r["host_rss"] for r in steady) - min(r["host_rss"] for r in steady)
    result = dict(
        frames=frames,
        size=[64, 64],
        posterior_noise_version=STREAMING_NOISE_VERSION,
        output_sha256=output_hash.hexdigest(),
        observations=observations,
        allocated_span=allocated_span,
        host_rss_span=host_span,
    )
    (output / "soak.json").write_text(json.dumps(result, indent=2))
    assert allocated_span < 16 * 2**20, "Codec GPU memory did not plateau"
    assert host_span < 64 * 2**20, "Codec host memory did not plateau"
    return result


@torch.inference_mode()
def run(args):
    root, output = Path(args.model), Path(args.output)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world > 1:
        from sglang.multimodal_gen.runtime.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank)
        initialize_model_parallel(sequence_parallel_degree=world, ulysses_degree=world)
        output = output / f"rank{rank}"
    output.mkdir(parents=True, exist_ok=False)
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs,
        set_global_server_args,
    )

    set_global_server_args(
        ServerArgs.from_kwargs(
            model_path=str(root),
            num_gpus=1,
            attention_backend="torch_sdpa",
            warmup_mode="off",
        )
    )
    config = MiniMaxH3VideoVAEConfig()
    config.update_model_arch(json.loads((root / "video_vae/config.json").read_text()))
    # H3 constructs non-persistent rotary buffers; a meta construction would
    # leave those buffers unmaterialized after state-dict assignment.
    model = MiniMaxH3VideoVAE(config)
    weights = root / "video_vae/source/model.safetensors"
    state = load_file(str(weights))
    _consume_vae_checkpoint_arch_metadata(state, config, model.state_dict())
    load_model_state_dict(model, state, assign=True, strict=True)
    del state
    model = model.cuda().eval()
    c = object.__new__(ZingH3Codecs)
    c.modules = {"video_vae": model}
    c.va = config.arch_config
    c.temporal_ratio = c.va.temporal_compression_ratio
    c.vae_frames_per_chunk = c.va.vae_clip_length
    c.vae_latents_per_chunk = math.ceil(c.vae_frames_per_chunk / c.temporal_ratio)
    c.runtime = json.loads((root / "zing_config.json").read_text())
    props = torch.cuda.get_device_properties(local_rank)
    report = dict(
        torch=torch.__version__,
        decode_parallel_size=world,
        rank=rank,
        hardware=dict(
            name=props.name,
            memory_mib=props.total_memory // 2**20,
            cc=[props.major, props.minor],
        ),
        weights_sha256=digest(weights),
        config_sha256=digest(root / "zing_config.json"),
        noise_version=STREAMING_NOISE_VERSION,
        cases=[],
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2))
    for frames, height, width in [
        (args.frames, args.height, args.width),
        *[(n, 64, 64) for n in range(1, 35)],
    ]:
        # Every frame varies; no looped or padded input duration.
        t = torch.arange(frames).view(1, frames, 1, 1)
        y = torch.arange(height).view(1, 1, height, 1)
        x = torch.arange(width).view(1, 1, 1, width)
        channels = torch.arange(3).view(3, 1, 1, 1)
        pixels = ((x + 3 * y + 7 * t + channels * 61 + (t * t) // 13) % 256).byte()
        captured = []
        adaptive = model._adaptive_encode

        def capture(clip):
            moments = adaptive(clip)
            captured.append(moments.cpu())
            return moments

        model._adaptive_encode = capture
        try:
            latent = c.encode_visual(pixels)
        finally:
            model._adaptive_encode = adaptive
        expected_moments = torch.cat(captured, 2)[:, :, : latent.shape[2]]
        case = dict(
            frames=frames,
            height=height,
            width=width,
            input_sha256=hashlib.sha256(pixels.numpy().tobytes()).hexdigest(),
            encode=[],
            decode=[],
        )
        indexed_latent = c.sample_streaming_visual_moments(
            expected_moments.cuda(),
            latent_start=0,
            noise_version=STREAMING_NOISE_VERSION,
        )
        case["indexed_posterior"] = []
        for batches in ([1], [17], [3, 11, 2, 19, 7]):
            stream = SourcePixelBuffer(c, capacity_frames=36)
            cursor = index = 0
            actual, sampled = [], []

            def accept_encoded(result):
                actual.append(result.values.cpu())
                sampled.append(
                    c.sample_streaming_visual_moments(
                        result.values,
                        latent_start=result.start,
                        noise_version=STREAMING_NOISE_VERSION,
                    ).cpu()
                )

            while cursor < frames:
                end = min(frames, cursor + batches[index % len(batches)])
                stream.append(pixels[:, cursor:end], frame_start=cursor)
                cursor, index = end, index + 1
                while stream.ready:
                    accept_encoded(stream.pop())
            stream.finish()
            for r in stream.drain():
                accept_encoded(r)
            case["encode"].append(
                dict(batches=batches, **compare(torch.cat(actual, 2), expected_moments))
            )
            case["indexed_posterior"].append(
                dict(batches=batches, **compare(torch.cat(sampled, 2), indexed_latent))
            )
            stream.dispose()
        cpu_rng, gpu_rng = (
            torch.get_rng_state().clone(),
            torch.cuda.get_rng_state().clone(),
        )
        noise_args = dict(
            seed=42, purpose="target_initial", device=latent.device, dtype=latent.dtype
        )
        whole_noise = indexed_video_noise(
            tuple(latent.shape), latent_start=0, **noise_args
        )
        pieces = []
        for start in range(0, latent.shape[2], 5):
            shape = list(latent.shape)
            shape[2] = min(5, latent.shape[2] - start)
            value = indexed_video_noise(tuple(shape), latent_start=start, **noise_args)
            compare(
                value,
                indexed_video_noise(tuple(shape), latent_start=start, **noise_args),
            )
            pieces.append(value)
        case["target_initial"] = compare(torch.cat(pieces, 2), whole_noise)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
        case["indexed_latent_sha256"] = hashlib.sha256(
            indexed_latent.cpu().numpy().tobytes()
        ).hexdigest()
        case["target_noise_sha256"] = hashlib.sha256(
            whole_noise.cpu().numpy().tobytes()
        ).hexdigest()
        # Reference uses one-rank tiling even in a two-rank run. Each rank
        # compares distributed complete-tile decode against that reference.
        model.parallel_tiling = False
        expected = c.decode_visual(latent, frames).cpu()
        model.parallel_tiling = world > 1
        legacy_latent = latent
        for kind, batch in (
            ("legacy", 1),
            ("legacy", 2),
            ("legacy", 5),
            ("indexed", 5),
        ):
            if kind == "indexed":
                latent = indexed_latent
                model.parallel_tiling = False
                expected = c.decode_visual(latent, frames).cpu()
                model.parallel_tiling = world > 1
            stream = StableVideoDecoder(
                c, capacity_latents=16, provenance_for=source_provenance
            )
            memory = []
            for start in range(0, latent.shape[2], batch):
                end = min(start + batch, latent.shape[2])
                stream.append(
                    latent[:, :, start:end],
                    latent_start=start,
                    source_dependency_end=frames,
                )
                while stream.ready:
                    r = stream.pop()
                    assert r.provenance.pixels == PixelRange(r.start, r.end)
                    assert r.provenance.source_dependency_end == frames
                    compare(r.values, expected[:, :, r.start : r.end])
                    del r
                torch.cuda.synchronize()
                memory.append(
                    dict(
                        accepted=end,
                        buffered=stream.buffered_latents,
                        allocated=torch.cuda.memory_allocated(),
                        reserved=torch.cuda.memory_reserved(),
                    )
                )
            stream.finish(total_frames=frames)
            for r in stream.drain():
                compare(r.values, expected[:, :, r.start : r.end])
            assert stream.buffered_latents == 0 and stream._latent is None
            stream.dispose()
            case["decode"].append(
                dict(kind=kind, batch=batch, exact=True, memory=memory)
            )
        latent = legacy_latent
        report["cases"].append(case)
        (output / "results.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps(dict(frames=frames, height=height, width=width, exact=True)),
            flush=True,
        )
    if args.soak_frames:
        report["soak"] = codec_soak(c, args.soak_frames, output)
        (output / "results.json").write_text(json.dumps(report, indent=2))
    (output / "PASSED").write_text(
        "Real weights / finite SG reference only; no native/DiT quality claim.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--soak-frames", type=int, default=0)
    parser.add_argument("--frames", type=int, default=243)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    run(parser.parse_args())
