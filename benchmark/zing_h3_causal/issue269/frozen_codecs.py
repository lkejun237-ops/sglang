# SPDX-License-Identifier: Apache-2.0
"""Untimed issue 269 component diagnosis. Trusted local snapshots only."""

import functools
import hashlib
import json
import os
from pathlib import Path


def install():
    import torch
    import torch.distributed as dist
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
        ZingH3Codecs,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
        H3ProgressiveEncoderSession,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        ZingSourceStreamState,
    )

    def map_tensors(value, transform):
        if isinstance(value, torch.Tensor):
            return transform(value)
        if isinstance(value, dict):
            return {k: map_tensors(v, transform) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(map_tensors(v, transform) for v in value)
        if isinstance(value, list):
            return [map_tensors(v, transform) for v in value]
        return value

    root = Path(os.environ["ZING269_ROOT"])
    arm = os.environ["ZING269_ARM"]
    mode = os.environ["ZING269_FREEZE"]
    canonical = root / "canonical"
    canonical.mkdir(exist_ok=True)
    counters = {}
    session = None
    session_index = -1

    def hashed(v):
        if isinstance(v, torch.Tensor):
            cpu = v.detach().contiguous().cpu()
            return {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "sha256": hashlib.sha256(
                    cpu.view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
        if isinstance(v, dict):
            return {k: hashed(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [hashed(x) for x in v]
        return (
            v if v is None or isinstance(v, (int, float, str, bool)) else str(type(v))
        )

    def log(row):
        row.update(arm=arm, rank=dist.get_rank(), session_index=session_index)
        with (root / f"{arm}-rank{dist.get_rank()}.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    original_run = ZingSourceStreamState.run

    @functools.wraps(original_run)
    def run(self, stage, batch):
        nonlocal session, session_index, counters
        if batch.realtime_session_id != session:
            session = batch.realtime_session_id
            session_index += 1
            counters = {}
        value = original_run(self, stage, batch)
        log(
            {
                "operation": "dispatch",
                "block": batch.block_idx,
                "counters": counters,
                "policy": self.policy,
            }
        )
        return value

    ZingSourceStreamState.run = run
    for name in [
        "encode_condition",
        "encode_visual",
        "sample_streaming_visual_moments",
    ]:
        original = getattr(ZingH3Codecs, name)

        def wrap(original, name):
            @functools.wraps(original)
            def call(self, *args, **kwargs):
                index = counters.get(name, 0)
                counters[name] = index + 1
                input_digest = (
                    hashed(args[0])
                    if name != "encode_condition"
                    else hashed(
                        [
                            args[0],
                            [{"id": x["id"], "frames": x["frames"]} for x in args[1]],
                        ]
                    )
                )
                result = original(self, *args, **kwargs)
                observed = hashed(result)
                file = canonical / f"{name}-{index:03d}.pt"
                if mode == "capture" and session_index == 0 and dist.get_rank() == 0:
                    torch.save(
                        {
                            "value": map_tensors(
                                result, lambda t: t.detach().cpu().clone()
                            ),
                            "input": input_digest,
                            "kwargs": kwargs,
                        },
                        file,
                    )
                if mode in ["capture", "replay"]:
                    dist.barrier()
                    saved = torch.load(file, map_location="cpu", weights_only=False)
                    # source moments are deliberately variable; source position must remain fixed.
                    if name != "sample_streaming_visual_moments":
                        assert saved["input"] == input_digest, (
                            name,
                            index,
                            "input mismatch",
                        )
                    else:
                        assert saved["kwargs"] == kwargs, (
                            name,
                            index,
                            "source position mismatch",
                        )
                    result = map_tensors(saved["value"], lambda t: t.to("cuda"))
                log(
                    {
                        "operation": name,
                        "index": index,
                        "input": input_digest,
                        "observed": observed,
                        "used": hashed(result),
                        "kwargs": kwargs,
                    }
                )
                return result

            return call

        setattr(ZingH3Codecs, name, wrap(original, name))
    pop = H3ProgressiveEncoderSession.pop

    def observed_pop(self):
        result = pop(self)
        if result is not None:
            log(
                {
                    "operation": "source_moments",
                    "start": result.start,
                    "end": result.end,
                    "value": hashed(result.values),
                    "execution": self.cache.last_execution,
                }
            )
        return result

    H3ProgressiveEncoderSession.pop = observed_pop
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.streaming_codecs import (
        StableVideoDecoder,
    )

    append = StableVideoDecoder.append

    def observed_append(self, latent, *, latent_start, source_dependency_end):
        log(
            {
                "operation": "target_latent",
                "start": latent_start,
                "value": hashed(latent),
            }
        )
        dest = root / f"{arm}-latents-rank{dist.get_rank()}-session{session_index}"
        dest.mkdir(exist_ok=True)
        torch.save(latent.detach().cpu(), dest / f"{latent_start:03d}.pt")
        return append(
            self,
            latent,
            latent_start=latent_start,
            source_dependency_end=source_dependency_end,
        )

    StableVideoDecoder.append = observed_append
    # Change exactly one numeric policy variable in controlled raw-encoding arms.
    init = ZingH3Codecs.__init__

    def configured(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if os.environ.get("ZING269_ENCODE_MODE"):
            self.runtime["vae_config"]["encode_mode"] = os.environ[
                "ZING269_ENCODE_MODE"
            ]

    ZingH3Codecs.__init__ = configured
    if os.environ.get("ZING269_BENCHMARK_OFF") == "1":
        from contextlib import contextmanager

        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3 import (
            codecs,
            progressive_encoder,
        )

        policy = codecs._visual_encode_policy

        @contextmanager
        def benchmark_off(*args, **kwargs):
            with policy(*args, **kwargs) as receipt:
                torch.backends.cudnn.benchmark = False
                receipt = {} if receipt is None else receipt
                receipt["cudnn_benchmark"] = False
                yield receipt

        codecs._visual_encode_policy = benchmark_off
        progressive_encoder._visual_encode_policy = benchmark_off

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
        SourceReadySession,
    )

    ready_append = SourceReadySession.append

    def observed_ready_append(self, source, noise, *args, **kwargs):
        log(
            {
                "operation": "dit_inputs",
                "source": hashed(source),
                "noise": hashed(noise),
            }
        )
        return ready_append(self, source, noise, *args, **kwargs)

    SourceReadySession.append = observed_ready_append
