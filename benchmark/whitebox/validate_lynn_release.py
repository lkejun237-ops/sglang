"""Compare conservative/proven release on the same trained decoder and SP2 GPUs.

Run before serving starts in the already allocated candidate Job. Only release
timing differs; this is not a native-Zing vs SGLang generation quality test.
"""

import argparse
import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
    ZingH3CausalPipelineConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.loader import (
    load_lynn_decoder,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
    decoder as decoder_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_decoder import (
    SourceLynnDecoder,
)
from sglang.multimodal_gen.runtime.realtime.zing_contracts import Provenance, SourceSpan


class Group:
    def __init__(self, rank, cpu_group):
        self.rank_in_group, self.world_size = rank, dist.get_world_size()
        self.cpu_group, self.device_group = cpu_group, dist.group.WORLD

    def all_gather_into(self, output, tensor):
        from sglang.multimodal_gen.runtime.distributed.group_coordinator import (
            GroupCoordinator,
        )

        return GroupCoordinator.all_gather_into(self, output, tensor)

    def all_reduce(self, tensor, op=dist.ReduceOp.SUM):
        dist.all_reduce(tensor, op=op, group=self.device_group)
        return tensor


def provenance(pixels, dependency):
    return Provenance(
        pixels, (SourceSpan(1, pixels.start, pixels.start, pixels.count),), dependency
    )


def run_policy(model, latent, frames, proven):
    adapter = decoder_module.DECODERS["lynn_streaming"]
    source = SourceLynnDecoder(
        SimpleNamespace(modules={adapter.module_key: model}, decoder=adapter),
        capacity_latents=20,
        provenance_for=provenance,
    )
    outputs, released, cursor = [], [], 0
    try:
        while cursor < latent.shape[2]:
            end = min(cursor + 5, latent.shape[2])
            valid = min(frames, model.protocol.frame_count(end))
            source.append(
                latent[:, :, cursor:end],
                latent_start=cursor,
                source_dependency_end=min(frames, valid + 17),
            )
            if proven:
                source.prove_valid_frames(valid)
            result = source.pop()
            assert result.start == (released[-1] if released else 0)
            released.append(result.end)
            outputs.append(result.values.cpu())
            cursor = end
        source.finish(total_frames=frames)
        result = source.pop()
        outputs.append(result.values.cpu())
        assert result.end == frames and source.drained
        return torch.cat(outputs, dim=2), released
    finally:
        source.dispose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-config", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    assert dist.get_world_size() == 2
    cpu_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=180))
    decoder_module.sp_tile_group = lambda: Group(rank, cpu_group)
    try:
        config = ZingH3CausalPipelineConfig()
        config.update_pipeline_config(
            json.loads(Path(args.pipeline_config).read_text())
        )
        spec = config.resolve_decoder_spec()
        assert spec.type == "lynn_streaming" and spec.precision == "bf16"
        model = load_lynn_decoder(spec, device=torch.device("cuda", rank))
        ratio = model.spatial_ratio
        assert args.height % ratio == args.width % ratio == 0
        cases = []
        for frames in (1, 16, 17, 18, 37):
            count = 1
            while model.protocol.frame_count(count) < frames:
                count += 1
            latent = torch.randn(
                1,
                24,
                count,
                args.height // ratio,
                args.width // ratio,
                generator=torch.Generator().manual_seed(42 + frames),
            ).to(device=device)
            old, old_ends = run_policy(model, latent, frames, False)
            new, new_ends = run_policy(model, latent, frames, True)
            assert old.shape == new.shape == (1, 3, frames, args.height, args.width)
            equal = int(torch.equal(old, new))
            passed = torch.tensor(equal, device=device)
            dist.all_reduce(passed, op=dist.ReduceOp.MIN)
            assert passed.item() == 1, f"Release changed pixels at {frames} frames"
            cases.append(
                dict(
                    frames=frames,
                    bitwise_equal=True,
                    conservative_ends=old_ends,
                    proven_ends=new_ends,
                    pixel_sha256=hashlib.sha256(
                        new.float().numpy().tobytes()
                    ).hexdigest(),
                )
            )
            del old, new, latent
        if rank == 0:
            result = dict(
                status="passed",
                scope="trained-decoder-same-sp2-release-policy",
                checkpoint=str(spec.checkpoint_path),
                precision=spec.precision,
                width=args.width,
                height=args.height,
                cases=cases,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
            output = Path(args.result)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n")
            print("LYNN_PROVEN_RELEASE_GPU_PASS", json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
