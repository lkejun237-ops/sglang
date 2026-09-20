#!/usr/bin/env python3
"""Read-only native reference observer. Copies alter timing, never sampling logic.

Usage: python capture_native_pipeline.py --runner VALIDATOR.py --diagnostics DIR
 --native-archive zing-97d60848.tar.gz -- reference [normal validator arguments]
The reference entry point, native defaults, placements and RNG calls are unchanged.
"""

import argparse
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

import torch


def tensor_hash(value):
    return hashlib.sha256(
        value.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    ).hexdigest()


def rng_snapshot():
    result = {"cpu": tensor_hash(torch.random.get_rng_state())}
    # Do not initialize CUDA or a fresh device merely to inspect its RNG.
    if torch.cuda.is_initialized():
        result["cuda_local"] = tensor_hash(
            torch.cuda.get_rng_state(torch.cuda.current_device())
        )
    return result


class Recorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.index = 0
        self.phase = "prepare"
        self.forward_counts = {}

    def record(self, name, **values):
        before = rng_snapshot()
        metadata = {}

        def freeze(value, path):
            if isinstance(value, torch.Tensor):
                cpu = value.detach().to(device="cpu", copy=True).contiguous()
                metadata[path] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "stride": list(value.stride()),
                    "device": str(value.device),
                    "sha256": tensor_hash(cpu),
                }
                return cpu
            if isinstance(value, dict):
                return {
                    key: freeze(item, f"{path}.{key}") for key, item in value.items()
                }
            if isinstance(value, (tuple, list, torch.Size)):
                return [freeze(item, f"{path}.{i}") for i, item in enumerate(value)]
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            raise TypeError(
                f"Unsupported observed value at {path}: {type(value).__name__}"
            )

        payload = {key: freeze(value, key) for key, value in values.items()}
        filename = f"{self.index:05d}.pt"
        torch.save(payload, self.directory / filename)
        after = rng_snapshot()
        record = {
            "index": self.index,
            "phase": self.phase,
            "name": name,
            "file": filename,
            "tensors": metadata,
            "rng": before,
            "observer_rng_after": after,
            "observer_preserved_rng": before == after,
        }
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        self.index += 1
        if before != after:
            raise RuntimeError("Read-only observer unexpectedly changed RNG state")


def condition_values(inputs):
    fields = (
        "positive",
        "visual",
        "audio",
        "source_visual",
        "visual_shape",
        "audio_shape",
        "output_type",
    )
    return {name: getattr(inputs, name) for name in fields if hasattr(inputs, name)}


def install_observers(processor_class, pipeline_class, recorder):
    originals = {}

    def replace(cls, name, wrapper):
        original = getattr(cls, name)
        originals[(cls, name)] = original
        setattr(cls, name, functools.wraps(original)(wrapper(original)))

    def prepare(original):
        def wrapped(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            recorder.record("conditioning_ready", **condition_values(result))
            return result

        return wrapped

    def sample(original):
        def wrapped(self, inputs, visual, audio):
            recorder.phase = "sample_start"
            recorder.record("conditioning_on_sample_device", **condition_values(inputs))
            recorder.record("initial_noise", video=visual, audio=audio)
            result = original(self, inputs, visual, audio)
            recorder.phase = "sample_complete"
            recorder.record("final_latents", video=result[0], audio=result[1])
            return result

        return wrapped

    def forward(original):
        def wrapped(self, packed, cache, update):
            mode = str(update)
            count = recorder.forward_counts.get(mode, 0)
            recorder.forward_counts[mode] = count + 1
            recorder.phase = f"{mode}.{count:04d}"
            recorder.record("forward_input", update=update, packed=packed)
            result = original(self, packed, cache, update)
            recorder.record(
                "forward_output", update=update, video=result[0], audio=result[1]
            )
            return result

        return wrapped

    replace(processor_class, "prepare_inference", prepare)
    replace(pipeline_class, "_sample", sample)
    replace(pipeline_class, "_forward", forward)

    def restore():
        for (cls, name), original in originals.items():
            setattr(cls, name, original)

    return restore


def verify_native(root, archive_path):
    count = 0
    with tarfile.open(archive_path) as archive:
        for member in archive:
            if (
                member.isfile()
                and member.name.startswith("src/")
                and member.name.endswith(".py")
            ):
                actual = root / member.name
                if (
                    not actual.is_file()
                    or hashlib.sha256(actual.read_bytes()).digest()
                    != hashlib.sha256(archive.extractfile(member).read()).digest()
                ):
                    raise ValueError("Native source mismatch: " + member.name)
                count += 1
    if not count:
        raise ValueError("Archive contained no src Python files")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--native-archive", type=Path, required=True)
    parser.add_argument("reference_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.reference_arguments
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv or argv[0] != "reference":
        parser.error("Supply -- reference and the unchanged native reference arguments")
    spec = importlib.util.spec_from_file_location(
        "observed_reference_validator", args.runner
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    reference = runner.parser().parse_args(argv)
    if reference.sp_size != 1:
        raise ValueError("This observer supports native SP1 only")
    root = Path(reference.minwm_root).resolve()
    verified = verify_native(root, args.native_archive)
    sys.path.insert(0, str(root / "src"))
    from dataloader.processors.h3_packed import H3PackedProcessor
    from pipeline.h3_inference import UnifiedH3InferencePipeline

    recorder = Recorder(args.diagnostics)
    restore = install_observers(H3PackedProcessor, UnifiedH3InferencePipeline, recorder)
    report = {
        "scope": "Read-only snapshots of native reference run; diagnostics are not benchmark timings",
        "reference_arguments": argv,
        "native_source_files_verified": verified,
        "native_archive_sha256": hashlib.sha256(
            args.native_archive.read_bytes()
        ).hexdigest(),
        "runner_sha256": hashlib.sha256(args.runner.read_bytes()).hexdigest(),
        "request_sha256": hashlib.sha256(
            Path(reference.request).read_bytes()
        ).hexdigest(),
        "no_algorithm_or_placement_override": True,
        "limitations": [
            "No explicit scheduler renoise noise captures in this minimal observer. Forward-to-forward RNG states are recorded, but exact intermediate RNG draws are not reconstructed.",
            "Synchronous CPU snapshots change timing and memory pressure; do not use these runs for performance claims.",
            "Forward phases use update mode and occurrence count. Align packed block_idx/t/row_role rather than assuming SG event indices match.",
        ],
    }
    try:
        runner.run_reference(reference)
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        restore()
        report.update(
            event_count=recorder.index, forward_counts=recorder.forward_counts
        )
        (args.diagnostics / "capture-report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
