# SPDX-License-Identifier: Apache-2.0
"""Resident Decoder lifetime probe through the real service and SHM publisher.

Run the two runtimes in separate owned GPU processes with the same inputs.
This is a Decoder attribution case, not an end-to-end throughput acceptance.
"""

import argparse
import hashlib
import inspect
import json
import os
import sys
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

# Keep direct script execution compatible with the shared benchmark helpers.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def allocator_environment():
    return {
        name: os.environ.get(name)
        for name in (
            "PYTORCH_ALLOC_CONF",
            "PYTORCH_CUDA_ALLOC_CONF",
            "PYTORCH_HIP_ALLOC_CONF",
            "PYTORCH_NO_CUDA_MEMORY_CACHING",
        )
    }


def allocator_snapshot(torch):
    snapshot = torch.cuda.memory._snapshot()
    if not isinstance(snapshot, dict) or not {
        "segments",
        "allocator_settings",
    }.issubset(snapshot):
        raise RuntimeError("The allocator snapshot does not expose its settings")
    return snapshot


def consume_output(service, publisher, frames, identity, *, materialize, cancel):
    reference = publisher.reserve(
        frames=frames, width=service.width, height=service.height, identity=identity
    )
    submitted = False
    try:
        service.submit(frames, reference)
        submitted = True
        return materialize(reference)
    except Exception as error:
        try:
            if not submitted:
                # No producer owns the reservation, so schedule its reclamation.
                publisher.fail(reference, error)
        finally:
            cancel(reference)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument("--runtime-revision", required=True)
    parser.add_argument("--cpu-ready", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    from frozen_source import verify_source

    verify_source(args.runtime_source, args.runtime_revision)
    import torch

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker import (
        initialize_runtime_imports,
    )

    initialize_runtime_imports()
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.decoder_worker import (
        _DecodeService,
        load_decoder,
    )

    runtime_file = Path(inspect.getsourcefile(_DecodeService)).resolve()
    if not runtime_file.is_relative_to(args.runtime_source.resolve()):
        raise ValueError("Decoder imported from a different frozen runtime")
    cases = json.loads(args.cases.read_text())
    if set(cases) != {"480", "736"} or any(
        set(cases[key]) != {"short", "long"} for key in cases
    ):
        raise ValueError("Both geometries need a complete short and long capture")
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3_decoder import (
        normalize_decoder_spec,
    )

    spec = normalize_decoder_spec(
        json.loads(args.pipeline.read_text())["decoder_config"],
        video_decoder_backend=None,
        vae_decode_precision="fp32",
    )
    if spec.type != "lynn_streaming" or spec.weights not in {"ema", "student"}:
        raise ValueError(
            "The resident probe requires an explicit Lynn weights selection"
        )
    from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.execution import (
        LynnExecutionOptions,
    )

    identity = dict(
        probe_sha256=sha(__file__),
        runtime_revision=args.runtime_revision,
        runtime_decoder_sha256=sha(runtime_file),
        cases_sha256=sha(args.cases),
        pipeline_sha256=sha(args.pipeline),
        short_sessions=40,
        long_rounds=3,
        selected_weights=spec.weights,
        precision=spec.precision,
        execution_options=asdict(LynnExecutionOptions.from_env()),
        expected_gpu_uuid=args.expected_gpu_uuid,
        allocator_environment=allocator_environment(),
    )
    if args.validate_only:
        if torch.cuda.device_count() or torch.cuda.is_initialized():
            raise RuntimeError("CPU validation must not initialize or see CUDA")
        files = []
        for path in [Path(spec.checkpoint_path), Path(spec.model_dir) / "config.json"]:
            stat = path.stat()
            files.append(
                dict(
                    path=str(path),
                    bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    resolved=str(path.resolve()),
                    sha256=sha(path),
                    model_asset=True,
                )
            )
        payload = torch.load(
            spec.checkpoint_path, map_location="cpu", mmap=True, weights_only=False
        )
        if (
            spec.weights not in payload
            or payload["config"]["training_config"]["dtype"] != "bfloat16"
        ):
            raise ValueError(
                "Decoder checkpoint weights or compute dtype differ from the profile"
            )
        from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.loader import (
            _decoder_config,
            read_architecture,
        )

        read_architecture(spec.model_dir)
        _decoder_config(payload["config"]["decoder_config"])
        del payload
        for geometry, pair in cases.items():
            for kind, directory in pair.items():
                root = Path(directory)
                complete = json.loads((root / "COMPLETE.json").read_text())
                frames = 403 if kind == "short" else 15921
                if complete["input_frames"] != frames or complete["chunks"] < 1:
                    raise ValueError(
                        "A capture is incomplete or has the wrong case length"
                    )
                if {p.name for p in root.glob("chunk-*.pt")} != {
                    f"chunk-{i:05d}.pt" for i in range(complete["chunks"])
                }:
                    raise ValueError(
                        "Capture chunk files do not match the complete receipt"
                    )
                for index in [0, complete["chunks"] - 1]:
                    row = torch.load(
                        root / f"chunk-{index:05d}.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                    value, chunk = row["expected_visual"], row["chunk"]
                    width, height = (832, 480) if geometry == "480" else (1280, 736)
                    if (
                        value.ndim != 5
                        or value.shape[1] != 24
                        or list(value.shape[-2:]) != [height // 16, width // 16]
                        or not torch.isfinite(value).all()
                        or (index == 0 and chunk.provenance.pixels.start != 0)
                        or (
                            index == complete["chunks"] - 1
                            and chunk.provenance.pixels.end != frames
                        )
                    ):
                        raise ValueError(
                            "Captured Decoder inputs do not match the geometry"
                        )
                for path in [
                    root / "COMPLETE.json",
                    root / "session.pt",
                    *sorted(root.glob("chunk-*.pt")),
                ]:
                    stat = path.stat()
                    files.append(
                        dict(
                            path=str(path),
                            bytes=stat.st_size,
                            mtime_ns=stat.st_mtime_ns,
                            resolved=str(path.resolve()),
                            sha256=sha(path),
                        )
                    )
        if args.cpu_ready.exists():
            raise ValueError("CPU readiness already exists; use a new frozen root")
        write(
            args.cpu_ready,
            dict(passed=True, identity=identity, files=files, cuda_count=0),
        )
        print("DECODER_RESIDENT_CPU_VALIDATED", flush=True)
        return
    ready = json.loads(args.cpu_ready.read_text())
    if not ready["passed"] or ready["identity"] != identity:
        raise ValueError("CPU readiness does not bind this exact probe and runtime")
    for row in ready["files"]:
        path = Path(row["path"])
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns, str(path.resolve())) != (
            row["bytes"],
            row["mtime_ns"],
            row["resolved"],
        ):
            raise ValueError("Frozen capture metadata changed after CPU validation")
        if row.get("model_asset") and sha(path) != row["sha256"]:
            raise ValueError("Decoder model bytes changed after CPU validation")
    if torch.cuda.device_count() != 1:
        raise ValueError("The caller must assign exactly one owned Decoder GPU")
    torch.cuda.set_device(0)
    if str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix(
        "gpu-"
    ) != args.expected_gpu_uuid.lower().removeprefix("gpu-"):
        raise ValueError("Decoder GPU differs from the frozen allocation")
    allocator_configuration = dict(
        environment=allocator_environment(),
        backend=torch.cuda.memory.get_allocator_backend(),
        settings=allocator_snapshot(torch)["allocator_settings"],
    )
    torch.set_num_threads(4)
    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.disaggregation.transport.shm_publisher import (
        SharedMemoryFramePublisher,
    )
    from sglang.multimodal_gen.runtime.realtime.cleanup_watchdog import CleanupWatchdog
    from sglang.multimodal_gen.runtime.utils.realtime_video import (
        cancel_async_raw_rgb_frame_reference,
        materialize_async_raw_rgb_frame_batches,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    rendezvous = args.output / "gloo-rendezvous"
    dist.init_process_group(
        "gloo", init_method=rendezvous.resolve().as_uri(), rank=0, world_size=1
    )
    publications = []

    class Publisher(SharedMemoryFramePublisher):
        def publish(self, reference, frames):
            entered = time.perf_counter_ns()
            stream = torch.cuda.current_stream()
            try:
                if (
                    "-warmup-" in reference["session_id"]
                    and not torch.isfinite(frames).all().item()
                ):
                    raise RuntimeError("Decoder warmup produced non-finite frames")
                return super().publish(reference, frames)
            finally:
                publications.append(
                    dict(
                        session=reference["session_id"],
                        chunk=reference["chunk_index"],
                        thread_native_id=threading.get_native_id(),
                        stream=stream.cuda_stream,
                        entered_ns=entered,
                        returned_ns=time.perf_counter_ns(),
                        scope="Host publish includes outstanding decode-stream work, conversion, D2H and SHM; not pure GPU decode",
                    )
                )

    publisher = Publisher(timeout_seconds=120, max_in_flight=10)
    results = []
    service = None
    raw_output = None
    session = None
    blocks = []
    try:
        model = load_decoder(json.loads(args.pipeline.read_text())["decoder_config"])
        if asdict(model.execution.options) != identity["execution_options"]:
            raise ValueError(
                "Loaded Decoder execution options differ from CPU readiness"
            )
        for geometry in ["480", "736"]:
            loaded = {}
            for kind, directory in cases[geometry].items():
                root = Path(directory)
                complete = json.loads((root / "COMPLETE.json").read_text())
                rows = []
                for i in range(complete["chunks"]):
                    row = torch.load(
                        root / f"chunk-{i:05d}.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                    rows.append(
                        (
                            row["expected_visual"].cuda(),
                            row["chunk"].chunk.start,
                            row["chunk"].provenance,
                        )
                    )
                loaded[kind] = rows
            expected = {}
            for kind, phase, index in [
                ("short", "warmup", 0),
                ("long", "warmup", 0),
                *[("short", "resident", i) for i in range(40)],
                *[("long", "measured", i) for i in range(3)],
            ]:
                session = f"{geometry}-{kind}-{phase}-{index}"
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                began = time.perf_counter()
                service = _DecodeService(
                    model,
                    capacity_latents=10,
                    group=None,
                    publisher=publisher,
                    decode_group=dist.group.WORLD,
                    device=torch.device("cuda", 0),
                )
                digest = hashlib.sha256()
                blocks, frame_end = [], 0
                work_seconds = time.perf_counter() - began
                raw_path = (
                    args.output / (session + ".rgb") if phase == "warmup" else None
                )
                raw_output = raw_path.open("xb") if raw_path is not None else None
                for chunk_index, (latent, start, provenance) in enumerate(loaded[kind]):
                    entered = time.perf_counter()
                    service.append(
                        latent,
                        latent_start=start,
                        valid_frame_end=provenance.pixels.end,
                        source_dependency_end=provenance.source_dependency_end,
                    )
                    expected_size = (832, 480) if geometry == "480" else (1280, 736)
                    if (service.width, service.height) != expected_size:
                        raise ValueError(
                            "Loaded Decoder geometry differs from the frozen case"
                        )
                    count = provenance.pixels.end - frame_end
                    batches = consume_output(
                        service,
                        publisher,
                        count,
                        dict(
                            session_id=session,
                            generation_id=session,
                            request_id=f"{session}-{chunk_index}",
                            chunk_index=chunk_index,
                        ),
                        materialize=materialize_async_raw_rgb_frame_batches,
                        cancel=cancel_async_raw_rgb_frame_reference,
                    )
                    returned = time.perf_counter()
                    work_seconds += returned - entered
                    delivered = sum(map(len, batches))
                    if delivered != count:
                        raise RuntimeError(
                            "Decoder did not deliver the complete declared chunk"
                        )
                    chunk_hash = hashlib.sha256()
                    for batch in batches:
                        for frame in batch:
                            digest.update(frame)
                            chunk_hash.update(frame)
                            if raw_output is not None:
                                raw_output.write(frame)
                    blocks.append(
                        dict(
                            chunk_index=chunk_index,
                            frame_start=frame_end,
                            frames=count,
                            seconds=returned - entered,
                            sha256=chunk_hash.hexdigest(),
                        )
                    )
                    frame_end = provenance.pixels.end
                    del batches, batch, frame
                cleanup = time.perf_counter()
                service.finish(frame_end)
                service.drain(120)
                service.dispose()
                if service.thread_alive:
                    raise RuntimeError(
                        "Decoder thread is still alive; no next session is allowed"
                    )
                work_seconds += time.perf_counter() - cleanup
                service = None
                if raw_output is not None:
                    raw_output.close()
                    raw_output = None
                wall_seconds = time.perf_counter() - began
                torch.cuda.synchronize()
                publication_path = args.output / (session + "-publications.json")
                streams = sorted({item["stream"] for item in publications})
                thread_ids = sorted({item["thread_native_id"] for item in publications})
                write(publication_path, publications)
                publications.clear()
                stats = torch.cuda.memory_stats()
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                import psutil

                row = dict(
                    session=session,
                    geometry=geometry,
                    kind=kind,
                    phase=phase,
                    index=index,
                    output_frames=frame_end,
                    chunks=len(blocks),
                    streams=streams,
                    decoder_threads=thread_ids,
                    publication_file=publication_path.name,
                    sha256=digest.hexdigest(),
                    matches_warmup=expected.get(kind) in (None, digest.hexdigest()),
                    raw_rgb=str(raw_path) if raw_path is not None else None,
                    raw_rgb_bytes=raw_path.stat().st_size
                    if raw_path is not None
                    else None,
                    seconds_including_cleanup_excluding_hash=work_seconds,
                    wall_seconds_including_hash_and_raw_retention=wall_seconds,
                    blocks=blocks,
                    allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    allocator={
                        k: stats[k]
                        for k in [
                            "num_alloc_retries",
                            "num_ooms",
                            "segment.all.current",
                            "inactive_split_bytes.all.current",
                            "active_bytes.all.current",
                        ]
                        if k in stats
                    },
                    rss_bytes=psutil.Process().memory_info().rss,
                    live_threads=psutil.Process().num_threads(),
                    cuda_mem_get_info_free_bytes=free_bytes,
                    cuda_mem_get_info_total_bytes=total_bytes,
                    scope="Whole Decoder process includes the fixed captured latent inputs; no empty_cache, manual GC, process restart or padded context between sessions",
                )
                expected.setdefault(kind, digest.hexdigest())
                results.append(
                    {key: value for key, value in row.items() if key != "blocks"}
                )
                write(args.output / (session + ".json"), row)
                write(
                    args.output / "STATUS.json",
                    dict(
                        phase=session, completed_sessions=len(results), time=time.time()
                    ),
                )
                if phase == "resident" and index in [0, 15, 31, 39]:
                    write(
                        args.output / (session + "-allocator-snapshot.json"),
                        allocator_snapshot(torch),
                    )
                print(
                    "DECODER_RESIDENT_SESSION",
                    session,
                    frame_end,
                    row["reserved_bytes"],
                    flush=True,
                )
            del loaded, rows, latent
        write(
            args.output / "RESULT.json",
            dict(
                collection_completed=True,
                all_repeats_exact=all(row["matches_warmup"] for row in results),
                identity=identity,
                allocator_configuration=allocator_configuration,
                results=[
                    {k: v for k, v in row.items() if k != "blocks"} for row in results
                ],
                device=dict(
                    name=torch.cuda.get_device_name(),
                    capability=list(torch.cuda.get_device_capability()),
                    uuid=str(torch.cuda.get_device_properties(0).uuid),
                ),
                torch=torch.__version__,
                cudnn=torch.backends.cudnn.version(),
                loaded_parameter_dtypes=sorted(
                    {str(p.dtype) for p in model.parameters()}
                ),
                scope="Decoder-only attribution collection; output parity and resource comparisons require independent analysis; no end-to-end or stall acceptance",
            ),
        )
    except BaseException:
        write(
            args.output / "FAILURE.json",
            dict(
                traceback=traceback.format_exc(),
                session=session,
                completed_sessions=len(results),
                current_blocks=blocks,
                publications=publications,
                collection_completed=False,
            ),
        )
        raise
    finally:
        if raw_output is not None:
            raw_output.close()
        with CleanupWatchdog(120):
            try:
                if service is not None:
                    try:
                        service.drain(120)
                    finally:
                        service.dispose()
            finally:
                try:
                    publisher.close()
                finally:
                    dist.destroy_process_group()


if __name__ == "__main__":
    main()
