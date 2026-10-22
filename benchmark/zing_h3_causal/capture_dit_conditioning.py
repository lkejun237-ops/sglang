# SPDX-License-Identifier: Apache-2.0
"""Capture production codec inputs during one existing validator request.

This is an untimed preparation run. The temporary startup hook propagates into
the validator's spawned workers, without changing serving code or sampling.
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def install():
    import dataclasses
    import functools
    import json

    import torch
    from bench_dit_phases import sha256

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker import (
        initialize_runtime_imports,
    )

    initialize_runtime_imports()
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
        ZingH3Codecs,
    )

    original = ZingH3Codecs.prepare

    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if dataclasses.is_dataclass(value):
            return dataclasses.replace(
                value,
                **{
                    f.name: cpu(getattr(value, f.name))
                    for f in dataclasses.fields(value)
                },
            )
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cpu(v) for v in value)
        return value

    @functools.wraps(original)
    def prepare(self, request, *args, **kwargs):
        result = original(self, request, *args, **kwargs)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            destination = Path(os.environ["ZING_BENCH_CONDITIONING_OUTPUT"])
            if destination.exists():
                raise FileExistsError(
                    "Capture requires exactly one request and a new output path"
                )
            torch.save(cpu(result), destination)
            destination.with_suffix(".json").write_text(
                json.dumps(
                    dict(
                        snapshot_sha256=sha256(destination),
                        request=request,
                        visual_shape=list(result.visual_shape),
                        audio_shape=result.audio_shape,
                        reference_ids=list(result.visual),
                    ),
                    indent=2,
                )
            )
        return result

    ZingH3Codecs.prepare = prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A validator command is required after --")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zing-conditioning-") as directory:
        Path(directory, "sitecustomize.py").write_text(
            "try:\n import capture_dit_conditioning; capture_dit_conditioning.install()\n"
            "except BaseException:\n import traceback, os; traceback.print_exc(); os._exit(90)\n"
        )
        env = dict(
            os.environ,
            ZING_BENCH_CONDITIONING_OUTPUT=str(args.output.resolve()),
            PYTHONPATH=os.pathsep.join(
                [
                    directory,
                    str(Path(__file__).resolve().parent),
                    os.environ.get("PYTHONPATH", ""),
                ]
            ),
        )
        subprocess.run(command, env=env, check=True)
    if not args.output.exists():
        raise RuntimeError("Validator completed without capturing conditioning")


if __name__ == "__main__":
    main()
