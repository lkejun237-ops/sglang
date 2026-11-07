# SPDX-License-Identifier: Apache-2.0
"""Measurement-only deserialization/GC adapter for main before these PR helpers.

Transformer, sampling, attention, CUDA Graph and KV methods remain untouched.
Both runtime and harness Git identities are verified by the outer job wrapper.

Retirement boundary: see diagnostics/README.md. Keep this entry until the
current frozen acceptance and independent archive verification are complete.
"""

import argparse
import importlib.util
import json
import runpy
import sys
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--harness-source", type=Path, required=True)
    args, replay_args = parser.parse_known_args()
    package = "sglang.multimodal_gen.runtime"
    path = (
        args.harness_source
        / "python"
        / Path(
            *package.split("."),
            "pipelines_core/stages/model_specific_stages/zing_h3/component_capture.py",
        )
    )
    spec = importlib.util.spec_from_file_location("typed_capture_mapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.component_capture as capture

    capture.map_tensors = module.map_tensors
    name = package + ".utils.zing_static_gc"
    if importlib.util.find_spec(name) is not None:
        raise RuntimeError(
            "Baseline already contains static GC support; review adapter"
        )
    shim = types.ModuleType(name)
    shim.prepare_static_gc = lambda _: None
    sys.modules[name] = shim
    print(
        "BASELINE_MEASUREMENT_ADAPTER",
        json.dumps(
            dict(
                typed_capture_mapper=str(path),
                static_gc="unchanged baseline, no freeze",
                scope="Input deserialization and measurement-only compatibility; production files unchanged",
            )
        ),
        flush=True,
    )
    entry = args.harness_source / "benchmark/zing_h3_causal/benchmark_dit_replay.py"
    sys.argv = [str(entry), *replay_args]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
