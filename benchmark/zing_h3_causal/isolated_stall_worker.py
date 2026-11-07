# SPDX-License-Identifier: Apache-2.0
"""Benchmark-owned role entry; all children use the same frozen source/probes."""

import multiprocessing as mp
import os
import runpy
import sys
from pathlib import Path

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker import (
    initialize_runtime_imports,
)

initialize_runtime_imports()
from stall_stage_probe import enable_cuda_trace, install_memory_probes

install_memory_probes()
enable_cuda_trace()

if os.environ.get("SGLANG_ZING_COMPONENT_CAPTURE_DIR"):
    from diagnostics.dit_replay.dit_rng_probe import install_native

    install_native(os.environ["SGLANG_ZING_COMPONENT_CAPTURE_DIR"])

if os.environ.get("STALL_RETAIN_STAGE_TIMING") == "1":
    from stall_stage_probe import retain_isolated_stage_files

    from sglang.multimodal_gen.runtime.utils import zing_stage_timing

    retain_isolated_stage_files(zing_stage_timing)
    print(
        "STALL_STAGE_TIMING_RETENTION finite isolated benchmark; native cleanup replaced by runner-owned retention",
        flush=True,
    )

if "STALL_GC_DIAGNOSTIC_SWITCH" in os.environ:
    from gc_pause_observer import install_passive_gc_diagnostic

    install_passive_gc_diagnostic()


def main():
    role = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    package = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3"
    if role == "server":
        from sglang.multimodal_gen.runtime.launch_server import (
            dispatch_launch,
            prepare_server_args,
        )

        mp.set_start_method("spawn", force=True)
        args = prepare_server_args(sys.argv[1:])
        args.disagg_encoder_port = int(os.environ["STALL_ENCODER_PORT"])
        args.disagg_decoder_port = int(os.environ["STALL_DECODER_PORT"])
        dispatch_launch(args)
    elif role == "encoder":
        from sglang.multimodal_gen.runtime.realtime.zing_nccl import (
            configure_zing_nccl,
        )

        # Match the production zing_source_encoder_server entry before the
        # underlying encoder worker initializes its communicators.
        configure_zing_nccl()
        runpy.run_module(package + ".encoder_worker", run_name="__main__")
    elif role == "qwen":
        runpy.run_module(
            "sglang.multimodal_gen.runtime.conditioning.cli", run_name="__main__"
        )
    elif role == "decoder":
        import importlib.util

        from stall_stage_probe import decoder_source, lifecycle_memory

        path = Path(importlib.util.find_spec(package + ".decoder_worker").origin)
        text = decoder_source(path)
        needle = 'if __name__ == "__main__":\n    main()'
        if text.count(needle) != 1:
            raise ValueError("Decoder entry anchor differs from frozen source")
        text = text.replace(
            needle,
            'if __name__ == "__main__":\n    lifecycle_memory(_DecodeService, "__init__", "dispose", "Decoder_process")\n    main()',
        )
        exec(
            compile(text, str(path), "exec"),
            dict(
                __name__="__main__",
                __file__=str(path),
                __package__=package,
                lifecycle_memory=lifecycle_memory,
            ),
        )
    else:
        raise ValueError("Unknown isolated stall role")


if __name__ == "__main__":
    main()
