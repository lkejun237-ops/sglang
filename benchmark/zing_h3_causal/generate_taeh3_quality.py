# SPDX-License-Identifier: Apache-2.0
"""Generate all review cases with one model load per decoder backend."""

import argparse
import json
import os
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("native", "taeh3"), required=True)
    parser.add_argument("--sp-size", type=int, default=4)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    args = parser.parse_args()
    os.environ.setdefault("SGLANG_ZING_EXPERIMENTAL_ATTENTION", "fa4")
    os.environ.setdefault("SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES", "1")
    os.environ.setdefault("SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN", "1")
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

    began = time.perf_counter()
    generator = DiffGenerator.from_pretrained(
        model_path=args.model,
        num_gpus=args.sp_size,
        sp_degree=args.sp_size,
        ulysses_degree=args.sp_size,
        ring_degree=1,
        encoder_parallel="replicate",
        local_mode=True,
        attention_backend="torch_sdpa",
        use_fsdp_inference=False,
        dit_layerwise_offload=False,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        performance_mode="speed",
        pin_cpu_memory=False,
        vae_cpu_offload=False,
        warmup_mode="off",
        pipeline_config={
            "video_decoder_backend": args.backend,
            "taeh3_checkpoint_path": args.weights,
            "taeh3_decode_precision": args.dtype,
        },
    )
    load_seconds = time.perf_counter() - began
    try:
        for case in sorted(Path(args.cases).glob("*.json")):
            request = json.loads(case.read_text())
            target = request["messages"][1]["content"][0]["output"]
            prompt = request["messages"][0]["content"][0]["text"]
            out = Path(args.output) / case.stem / args.backend
            out.mkdir(parents=True, exist_ok=True)
            began = time.perf_counter()
            result = generator.generate(
                sampling_params_kwargs={
                    "prompt": prompt,
                    "height": target["height"],
                    "width": target["width"],
                    "num_frames": target["frames"],
                    "fps": 24,
                    "num_inference_steps": 4,
                    "guidance_scale": 1.0,
                    "seed": 42,
                    "zing_request": request,
                    "zing_solver": "few_step",
                    "zing_denoising_step_list": [1000, 750, 500, 250],
                    "zing_return_latents": True,
                    "save_output": True,
                    "return_file_paths_only": False,
                    "output_path": str(out),
                    "output_file_name": "sglang-0",
                }
            )
            seconds = time.perf_counter() - began
            if result is None or result.trajectory_latents is None:
                raise RuntimeError("Quality generation returned no latent")
            sample = result.samples
            if isinstance(sample, (tuple, list)) and len(sample) == 2:
                sample = sample[0]
            frames = torch.as_tensor(sample).detach().cpu()
            if frames.ndim == 4:
                frames = frames.unsqueeze(0)
            latent = torch.as_tensor(result.trajectory_latents).detach().cpu()
            audio = (
                None
                if result.audio is None
                else torch.as_tensor(result.audio).detach().cpu()
            )
            torch.save(
                {"frames": frames, "visual_latents": latent, "audio": audio},
                out / "sglang.pt",
            )
            report = {
                "video_decoder_backend": args.backend,
                "taeh3_decode_precision": args.dtype,
                "sp_size": args.sp_size,
                "request": request,
                "generation_seconds": seconds,
                "generation_seconds_all": [seconds],
                "load_seconds_shared": load_seconds,
                "timing_scope": "quality run only; not a warm latency benchmark",
                "shapes": {
                    "frames": list(frames.shape),
                    "visual_latents": list(latent.shape),
                },
                "metrics": result.metrics,
                "usage": getattr(result, "usage", None),
            }
            (out / "sglang.json").write_text(json.dumps(report, indent=2))
            print(
                json.dumps(
                    {"case": case.stem, "backend": args.backend, "seconds": seconds}
                ),
                flush=True,
            )
            del result, frames, latent, audio, sample
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
