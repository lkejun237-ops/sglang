# SPDX-License-Identifier: Apache-2.0
"""Lynn decoder SP tile-parallel parity on fixed latents (PR #298, design §5.1).

``decode`` runs under ``torchrun --nproc-per-node P``. It initializes the real
SGLang SP group (sp = ulysses = P), builds the decoder through the registry
adapter, and decodes every staged latent on four paths: pre-clamp
``decode_step``, offline ``ZingH3Codecs.decode_visual``, a finite realtime
stream with business chunks and ``SourceLynnDecoder``. P=1 stores the
pre-clamp frames as the SP1 baseline; P>1 compares against them and across
ranks, and injects a single-rank failure. Precomputed ``reference/R`` frames
are an optional vs-R gate. ``report`` summarizes on CPU.

Numeric mismatches never stop a run; every comparison is a metrics row.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from lynn_acceptance_manifest import run_report

LIFECYCLE_SAMPLE = "video-480p-checkpoint_generated-vbh000169_h3"


class Recorder:
    """Per-rank metrics rows, fsync per row."""

    def __init__(self, args, rank, world_size):
        self.args, self.rank, self.world_size = args, rank, world_size
        self.output = Path(args.output) / f"sp{world_size}"
        self.output.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(
            (Path(args.output) / "acceptance-manifest.json").read_text()
        )
        if world_size not in manifest["sp"] or rank not in range(world_size):
            raise ValueError("Worker is outside the acceptance manifest")
        if sha256(Path(args.inputs) / "inputs.json") != manifest["inputs_json_sha256"]:
            raise ValueError("Staged inputs changed after acceptance preparation")
        if (
            bool(args.reference_output) != manifest["reference"]
            or args.lifecycle_sample != manifest["lifecycle_sample"]
        ):
            raise ValueError("Worker gates differ from the acceptance manifest")
        self.run_id = manifest["run_id"]
        self.env = {
            "rank": rank,
            "world_size": world_size,
            "argv": sys.argv,
            "run_id": self.run_id,
        }

    def write(self, **row):
        row = {
            "sp": self.world_size,
            "rank": self.rank,
            "time": time.time(),
            "run_id": self.run_id,
            **row,
        }
        row = {
            k: (repr(v) if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in row.items()
        }
        with (self.output / f"metrics-rank{self.rank}.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def save_env(self):
        (self.output / f"env-rank{self.rank}.json").write_text(
            json.dumps(self.env, indent=1, default=str)
        )

    def abort(self, reason):
        self.env["abort"] = reason
        self.save_env()
        print(f"ABORT sp{self.world_size} rank{self.rank}: {reason}", flush=True)
        raise SystemExit(3)


def sha256(path):
    digest_ = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest_.update(block)
    return digest_.hexdigest()


def load_inputs(run):
    """Verify each staged latent against inputs.json, its manifest row and canonical latent sha."""
    from safetensors import safe_open

    root = Path(run.args.inputs)
    staged = json.loads((root / "inputs.json").read_text())
    rows = {
        row["id"]: row
        for row in map(json.loads, (root / "manifest.jsonl").read_text().splitlines())
    }
    samples = []
    for entry in staged["latents"]:
        path = root / entry["path"]
        file_digest = sha256(path)
        if (
            file_digest != entry["sha256"]
            or file_digest != entry["canonical_latent_sha256"]
        ):
            run.abort(f"input sha256 mismatch: {path}")
        with safe_open(str(path), framework="pt") as handle:
            if set(handle.keys()) != {"latents"}:
                run.abort(f"latent contract violated: {path}")
            identity = json.loads(handle.metadata()["identity"])
            latent = handle.get_tensor("latents")
        row = rows[entry["id"]]
        if identity["video_sha256"] != row["sha256"]:
            run.abort(f"latent identity does not match manifest: {path}")
        latent = latent.unsqueeze(0) if latent.ndim == 4 else latent
        samples.append(
            SimpleNamespace(
                id=entry["id"],
                latent=latent,
                frames=int(row["frame_count"]),
                media_type=row["media_type"],
            )
        )
    run.env["inputs"] = {
        "inputs_json_sha256": sha256(root / "inputs.json"),
        "samples": [sample.id for sample in samples],
    }
    return samples


@torch.no_grad()
def compare(actual, expected):
    """Exact float32 comparison over all frames."""
    result = {
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
    }
    if actual.shape != expected.shape:
        result.update(
            shape_equal=False,
            equal=False,
            max_abs=None,
            rmse=None,
            mismatched=None,
            finite=None,
        )
        return result
    max_abs, squared, mismatched, first, frame_max, finite, equal = (
        0.0,
        0.0,
        0,
        None,
        [],
        True,
        True,
    )
    for start in range(0, actual.shape[2], 8):
        left = actual[:, :, start : start + 8].float()
        right = expected[:, :, start : start + 8].float()
        finite &= bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        equal &= torch.equal(left, right)
        diff = (left - right).abs()
        frame_max += [float(x) for x in diff.amax(dim=(0, 1, 3, 4))]
        max_abs = max(max_abs, float(diff.max()))
        squared += float(diff.double().pow(2).sum())
        unequal = diff != 0
        count = int(unequal.sum())
        if count and first is None:
            index = [int(i) for i in unequal.nonzero()[0]]
            index[2] += start
            first = index
        mismatched += count
    rmse = math.sqrt(squared / actual.numel())
    result.update(
        shape_equal=True,
        equal=bool(equal and finite),
        finite=finite,
        max_abs=max_abs,
        rmse=rmse,
        psnr_db=None if rmse == 0 else 20 * math.log10(1.0 / rmse),
        mismatched=mismatched,
        first_mismatch=first,
        frame_max=frame_max,
    )
    return result


def business_chunks(count):
    """H3 production split: blocks [3, 2] with the first latent split off."""
    sizes, head = [], [1, 2]
    while sum(sizes) < count:
        sizes.append(head.pop(0) if head else (2 if len(sizes) % 2 == 0 else 3))
    sizes[-1] -= sum(sizes) - count
    return sizes


def split_by(latent, sizes):
    start = 0
    for size in sizes:
        yield latent[:, :, start : start + size]
        start += size


def frames_path(output, tag, sample_id):
    return Path(output) / "reference" / tag / f"{sample_id}.safetensors"


def save_frames(path, frames):
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    save_file({"frames": frames.contiguous()}, str(temporary))
    os.replace(temporary, path)


def load_frames(path):
    from safetensors.torch import load_file

    return load_file(str(path))["frames"]


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().cpu().numpy().tobytes()).hexdigest()


def state_bytes(state):
    return sum(
        t.numel() * t.element_size()
        for layers in state.tiles.values()
        for kv in layers
        for t in kv
    )


def run_decode(args):
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
        ZingH3CausalPipelineConfig,
    )
    from sglang.multimodal_gen.runtime.distributed import (
        get_sp_group,
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
        ZingH3Codecs,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder import (
        decoder_adapter,
        sp_tile_group,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_decoder import (
        SourceLynnDecoder,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
        Provenance,
        SourceSpan,
    )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=world_size, ulysses_degree=world_size
    )
    sp = get_sp_group()
    rank, device = sp.rank_in_group, sp.device
    run = Recorder(args, rank, world_size)
    run.env.update(torch=torch.__version__, cuda=torch.version.cuda, device=str(device))
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        run.env.update(
            device_name=props.name,
            memory_mib=props.total_memory // 2**20,
            compute_capability=list(torch.cuda.get_device_capability(device)),
        )
    import diffusers
    import huggingface_hub

    run.env["packages"] = {
        m.__name__: {"version": m.__version__, "file": m.__file__}
        for m in (diffusers, huggingface_hub)
    }
    group = sp_tile_group()
    if (group is None) != (world_size == 1) or (
        group is not None and group.world_size != world_size
    ):
        run.abort(f"sp_tile_group does not match SP {world_size}: {group}")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if world_size > 1:
            torch.distributed.barrier(group=sp.cpu_group)

    shim = SimpleNamespace(args=args, env=run.env, abort=run.abort)
    samples = load_inputs(shim)
    if args.checkpoint_sha256 and rank == 0:
        actual = sha256(args.checkpoint)
        run.env["checkpoint_sha256"] = actual
        if actual != args.checkpoint_sha256:
            run.abort(f"checkpoint sha256 {actual} != {args.checkpoint_sha256}")
    decoder_config = {
        "type": "lynn_streaming",
        "checkpoint_path": args.checkpoint,
        "model_dir": args.model_dir,
    }
    spec = ZingH3CausalPipelineConfig(
        decoder_config=decoder_config
    ).resolve_decoder_spec()
    adapter = decoder_adapter(spec)
    # Residency moves devices only; the loader already fixed the mixed dtypes.
    model = adapter.build(spec)[adapter.module_key].to(device)
    run.env.update(
        attention=type(model.attention).__name__,
        compute_dtype=str(model.compute_dtype),
        tile_config=[
            model.config.tile_size,
            model.config.tile_overlap,
            model.config.tile_batch_size,
        ],
    )
    codecs = object.__new__(ZingH3Codecs)
    codecs.decoder, codecs.modules = adapter, {adapter.module_key: model}
    protocol = model.protocol
    run.save_env()

    def provenance(pixels, dependency):
        return Provenance(
            pixels,
            (SourceSpan(1, pixels.start, pixels.start, pixels.count),),
            dependency,
        )

    @torch.no_grad()
    def preclamp(latent):
        state, steps = model.init_stream(), []
        with torch.autocast(device.type, dtype=model.compute_dtype):
            for z_t in latent.split(1, dim=2):
                frames, state = model.decode_step(z_t, state, group)
                steps.append(frames.float().cpu())
        return torch.cat(steps, 2), state

    def cross_rank(label, sample, tensor):
        value = digest(tensor)
        values = [None] * world_size
        if world_size > 1:
            torch.distributed.all_gather_object(values, value, group=sp.cpu_group)
        else:
            values = [value]
        run.write(
            gate="ranks-equal",
            sample=sample.id,
            variant=label,
            ok=len(set(values)) == 1,
            digests=values,
        )

    # Warm-up (kernels, allocator, NCCL communicator); not timed.
    warm = samples[0].latent[:, :, :2].to(device)
    preclamp(warm)
    sync()

    for sample in samples:
        latent, frames = sample.latent.to(device), sample.frames
        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
        sync()
        started = time.perf_counter()
        steps, state = preclamp(latent)
        sync()
        seconds = time.perf_counter() - started
        tiles = len(state.tiles)
        run.write(
            gate="timing",
            sample=sample.id,
            variant="decode_step",
            seconds=seconds,
            latents=latent.shape[2],
            frames=steps.shape[2],
            owned_tiles=tiles,
            state_bytes=state_bytes(state),
            peak_allocated=torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        )
        del state
        cross_rank("preclamp", sample, steps)
        if rank == 0:
            if args.reference_output:
                reference = frames_path(args.reference_output, "R", sample.id)
                row = (
                    compare(steps, load_frames(reference))
                    if reference.is_file()
                    else {
                        "ok": False,
                        "equal": False,
                        "max_abs": None,
                        "missing": str(reference),
                    }
                )
                run.write(gate="vs-R", sample=sample.id, variant="preclamp", **row)
            sp1 = frames_path(args.output, "SP1", sample.id)
            if world_size == 1:
                save_frames(sp1, steps)
            elif sp1.is_file():
                run.write(
                    gate="vs-SP1",
                    sample=sample.id,
                    variant="preclamp",
                    **compare(steps, load_frames(sp1)),
                )
        expected = steps[:, :, :frames].clamp(0, 1)
        del steps

        sync()
        started = time.perf_counter()
        offline = codecs.decode_visual(latent, frames).cpu()
        sync()
        run.write(
            gate="timing",
            sample=sample.id,
            variant="offline",
            seconds=time.perf_counter() - started,
            frames=offline.shape[2],
        )
        run.write(
            gate="paths",
            sample=sample.id,
            variant="offline-vs-preclamp",
            **compare(offline, expected),
        )
        cross_rank("offline", sample, offline)

        sizes = business_chunks(latent.shape[2])
        stream = adapter.new_stream(model, frame_limit=frames)
        pieces = [stream.push(chunk).cpu() for chunk in split_by(latent, sizes)]
        pieces.append(stream.finish(frames).cpu())
        finite = torch.cat(pieces, dim=2)
        run.write(
            gate="paths",
            sample=sample.id,
            variant="finite-vs-offline",
            chunks=sizes,
            **compare(finite, offline),
        )
        cross_rank("finite", sample, finite)
        del finite, pieces, stream

        source = SourceLynnDecoder(
            codecs, capacity_latents=64, provenance_for=provenance
        )
        pieces, intervals, start = [], [], 0
        try:
            for chunk in split_by(latent, sizes):
                end = start + chunk.shape[2]
                source.append(
                    chunk,
                    latent_start=start,
                    source_dependency_end=min(frames, protocol.frame_count(end)),
                )
                start = end
                while source.ready:
                    piece = source.pop()
                    pieces.append(piece.values.cpu())
                    intervals.append([piece.start, piece.end])
            source.finish(total_frames=frames)
            while source.ready:
                piece = source.pop()
                pieces.append(piece.values.cpu())
                intervals.append([piece.start, piece.end])
        finally:
            source.dispose()
        contiguous = (
            intervals[0][0] == 0
            and intervals[-1][1] == frames
            and all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
        )
        run.write(
            gate="paths",
            sample=sample.id,
            variant="source-vs-offline",
            intervals_ok=contiguous,
            **compare(torch.cat(pieces, dim=2), offline),
        )
        del pieces, offline, expected

        if world_size > 1 and sample.id == args.lifecycle_sample:
            failure_check(
                run, model, adapter, latent, frames, rank, world_size, sync, device
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()


def failure_check(run, model, adapter, latent, frames, rank, world_size, sync, device):
    """Last rank fails one decode_step mid-stream: every rank must raise and close, then recover."""
    baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    expected = adapter.decode(model, latent, frames).cpu()
    stream = adapter.new_stream(model, frame_limit=frames)
    stream.push(latent[:, :, :3])
    original = model._decode_tiles
    if rank == world_size - 1:

        def fail(*args, **kwargs):
            model._decode_tiles = original
            raise RuntimeError("injected tile failure")

        model._decode_tiles = fail
    started, raised = time.perf_counter(), None
    try:
        stream.push(latent[:, :, 3:])
    except Exception as exc:
        raised = f"{type(exc).__name__}: {exc}"[:300]
    model._decode_tiles = original
    seconds = time.perf_counter() - started
    del stream
    sync()
    recovered = adapter.decode(model, latent, frames).cpu()
    after = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    run.write(
        gate="failure",
        variant="single-rank-failure",
        ok=raised is not None and torch.equal(recovered, expected),
        raised=raised,
        seconds_to_raise=seconds,
        memory_before=baseline,
        memory_after=after,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("decode", "report"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--inputs")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--model-dir")
    parser.add_argument(
        "--reference-output",
        help="directory with reference/R/<sample>.safetensors; omit to skip vs-R",
    )
    parser.add_argument("--lifecycle-sample", default=LIFECYCLE_SAMPLE)
    args = parser.parse_args()
    if args.role == "report":
        raise SystemExit(run_report(args))
    try:
        run_decode(args)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        with contextlib.suppress(Exception):
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
