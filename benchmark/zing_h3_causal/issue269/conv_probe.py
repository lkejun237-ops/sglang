"""Untimed reference convolution diagnosis; injected only into the next probe."""

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

    active = False
    call = 0
    reference = 0
    seen = set()
    root = Path(os.environ["ZING269_ROOT"]) / ("conv-" + os.environ["ZING269_ARM"])
    root.mkdir(exist_ok=True)

    def digest(t):
        cpu = t.detach().contiguous().cpu()
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "sha256": hashlib.sha256(
                cpu.view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }

    conv = torch.nn.functional.conv3d

    def observed_conv(input, weight, bias=None, *args, **kwargs):
        nonlocal call
        out = conv(input, weight, bias, *args, **kwargs)
        if active and reference <= 6:
            key = (tuple(input.shape), tuple(weight.shape), str(input.dtype))
            # All calls on the first reference; first instance of each geometry thereafter.
            if reference == 1 or key not in seen:
                seen.add(key)
                row = {
                    "reference": reference,
                    "call": call,
                    "input": digest(input),
                    "weight": digest(weight),
                    "bias": None if bias is None else digest(bias),
                    "output": digest(out),
                    "args": args,
                    "kwargs": kwargs,
                    "benchmark": torch.backends.cudnn.benchmark,
                    "deterministic": torch.backends.cudnn.deterministic,
                }
                dest = root / f"conv-rank{dist.get_rank()}.jsonl"
                with dest.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                # First-reference leaf values allow exact numeric follow-up without a rerun.
                if reference == 1 and call < 5:
                    torch.save(
                        {
                            "input": input.detach().cpu(),
                            "weight": weight.detach().cpu(),
                            "bias": None if bias is None else bias.detach().cpu(),
                            "output": out.detach().cpu(),
                            "args": args,
                            "kwargs": kwargs,
                        },
                        root / f"conv-rank{dist.get_rank()}-{call}.pt",
                    )
            call += 1
        return out

    torch.nn.functional.conv3d = observed_conv
    encode = ZingH3Codecs.encode_visual

    @functools.wraps(encode)
    def observed_encode(self, frames):
        nonlocal active, call, reference
        reference += 1
        call = 0
        active = reference <= 6
        if active:
            torch.save(frames, root / f"reference-{reference}-rank{dist.get_rank()}.pt")
        try:
            return encode(self, frames)
        finally:
            active = False

    ZingH3Codecs.encode_visual = observed_encode
