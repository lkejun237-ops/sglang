# SPDX-License-Identifier: Apache-2.0
"""Set OUT to an output directory; run on one CUDA device, outside serving.

CUDA-event intervals include enqueue gaps. Kernel sums are separate diagnostics;
logical bytes describe unique tensors, not measured device memory traffic.
Grouped and packing alternatives are benchmarks only.
HISTORY_ROPE_WORKLOADS optionally names a JSON list of named chunk-size cases.
"""

import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
import torch
import sglang.multimodal_gen.runtime.models.dits.zing_h3  # noqa: F401 - initialize runtime imports
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import _apply_rope_into
from sglang.multimodal_gen.runtime.models.dits.zing_h3_rope import (
    apply_history_rope_into,
    HISTORY_ROPE_ROUTES,
)
import sglang.multimodal_gen.runtime.models.dits.zing_h3_rope as module
from history_rope_grouped_prototype import grouped

out = Path(os.environ["OUT"])
out.mkdir(parents=True, exist_ok=True)
os.environ["SGLANG_ZING_FUSED_HISTORY_ROPE"] = "1"
torch.set_grad_enabled(False)
torch.manual_seed(179)
meta = dict(
    device=torch.cuda.get_device_name(),
    memory=torch.cuda.get_device_properties(0).total_memory,
    cc=torch.cuda.get_device_capability(),
    torch=torch.__version__,
    cuda=torch.version.cuda,
    triton=importlib.metadata.version("triton"),
    file=module.__file__,
    hash=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
)
records = []
workloads = [
    dict(name="small", sizes=[17]),
    dict(name="single", sizes=[390]),
    dict(name="synthetic-22-chunks", sizes=[5040] + [780] * 21),
]
if path := os.environ.get("HISTORY_ROPE_WORKLOADS"):
    workloads = json.loads(Path(path).read_text())
    meta["workloads_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert workloads and all(
    isinstance(case.get("name"), str)
    and case.get("sizes")
    and all(isinstance(n, int) and n > 0 for n in case["sizes"])
    for case in workloads
)
meta["workloads"] = workloads
for heads in (14, 28, 56):
    for case_index, case in enumerate(workloads):
        sizes = case["sizes"]
        xs = [
            torch.randn(n, heads, 128, device="cuda", dtype=torch.bfloat16)
            for n in sizes
        ]
        cs = [torch.randn(n, 96, device="cuda", dtype=torch.bfloat16) for n in sizes]
        dest = torch.empty(sum(sizes), heads, 128, device="cuda", dtype=torch.bfloat16)
        outputs = dest.split(sizes)

        def eager():
            for x, c, y in zip(xs, cs, outputs):
                _apply_rope_into(x, c, y)

        def fused():
            for x, c, y in zip(xs, cs, outputs):
                apply_history_rope_into(x, c, y)

        def packed():
            apply_history_rope_into(torch.cat(xs), torch.cat(cs), dest)

        eager()
        expected = dest.clone()

        def grouped_call():
            grouped(xs, cs, dest)

        for name, fn in [
            ("eager", eager),
            ("fused", fused),
            ("pack_fused", packed),
            ("grouped_prototype", grouped_call),
        ]:
            try:
                fn()
                assert torch.equal(dest, expected), (heads, sizes, name)
            except Exception as error:
                if name != "grouped_prototype":
                    raise
                records.append(
                    dict(
                        heads=heads,
                        case=case["name"],
                        sizes=sizes,
                        mode=name,
                        error=repr(error),
                    )
                )
                print("PROTOTYPE_FAILURE", repr(error), flush=True)
                continue
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            samples = []
            wall = []
            for _ in range(20):
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                start = time.perf_counter()
                a.record()
                fn()
                b.record()
                b.synchronize()
                samples.append(a.elapsed_time(b))
                wall.append((time.perf_counter() - start) * 1000)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as p:
                fn()
                torch.cuda.synchronize()
            path = out / f"micro-h{heads}-case{case_index}-{name}.trace.json"
            p.export_chrome_trace(str(path))
            kernels = [
                e
                for e in json.loads(path.read_text())["traceEvents"]
                if e.get("cat") == "kernel"
            ]
            records.append(
                dict(
                    heads=heads,
                    case=case["name"],
                    sizes=sizes,
                    mode=name,
                    ms=samples,
                    wall_ms=wall,
                    kernels=len(kernels),
                    kernel_us=sum(e.get("dur", 0) for e in kernels),
                    rows=sum(sizes),
                    logical_bytes=sum(x.numel() * x.element_size() for x in xs) * 2
                    + sum(c.numel() * c.element_size() for c in cs),
                )
            )
            print("MICRO", records[-1], flush=True)
(out / "micro.json").write_text(
    json.dumps(
        dict(metadata=meta, records=records, routes=dict(HISTORY_ROPE_ROUTES)), indent=2
    )
)
