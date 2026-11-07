# SPDX-License-Identifier: Apache-2.0
"""Exercise real Zing graph capture with pending compiled kernel finalizers.

Run in its own GPU process. Adapted from pytorch/pytorch#199117. The deliberately
low GC threshold turns a timing-dependent capture invalidation into a regression
case. This is a correctness check, not a throughput or model accuracy benchmark.
"""

import argparse
import gc
import json
import os
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--unprotected", action="store_true", help="Diagnose raw torch capture"
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="zing-capture-gc-") as cache:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache + "/inductor"
        os.environ["TRITON_CACHE_DIR"] = cache + "/triton"
        import torch

        from sglang.multimodal_gen.runtime.models.dits.zing_h3_denoise_graph import (
            CudaGraphBackend,
        )

        gc.collect()
        gc.disable()
        x = torch.randn(1024, device="cuda")
        compiled = torch.compile(lambda x: x.sin() + x.cos(), mode="max-autotune")
        with torch.inference_mode():
            for _ in range(3):
                compiled(x).cpu()
        del compiled
        torch.compiler.reset()
        gc.set_threshold(10**9, 10**9, 10**9)
        gc.enable()
        backend = CudaGraphBackend()
        policies = []

        def forward():
            if backend.capturing or args.unprotected:
                policies.append(gc.isenabled())
                gc.set_threshold(1, 1, 1)
                garbage = [[] for _ in range(3000)]
                y = x + len(garbage) / 3000
            else:
                y = x + 1
            return (y,)

        before = torch.cuda.current_stream()
        with torch.inference_mode():
            if args.unprotected:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph, stream=torch.cuda.Stream(), capture_error_mode="thread_local"
                ):
                    (y,) = forward()
            else:
                graph, (y,) = backend.capture(forward)
            graph.replay()
            torch.testing.assert_close(y, x + 1)
        assert gc.isenabled(), "Automatic GC was not restored after capture"
        assert torch.cuda.current_stream() == before
        if not args.unprotected:
            assert policies == [False], policies
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                dict(
                    passed=True,
                    torch=torch.__version__,
                    cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(),
                    automatic_gc_after=gc.isenabled(),
                    automatic_gc_during_capture=policies,
                    scope="Compiled kernel finalizer safety, not model performance",
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
