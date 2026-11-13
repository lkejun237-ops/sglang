# SPDX-License-Identifier: Apache-2.0
"""CPU prepare / large-GPU latent generation / CPU collection for Decoder A/B.

All paths and hashes come from an immutable experiment spec. Model assets are
referenced read-only; only the newly created owned root is written.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def environment(root):
    return dict(
        os.environ,
        PYTHONPATH=f"{root}/kernels/src:{root}/deps:{root}/source/python:{root}/source/benchmark/zing_h3_causal",
        SGLANG_LIGHTWEIGHT_RUNTIME="1",
        SGLANG_ZING_EXPERIMENTAL_ATTENTION="fa4",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        NCCL_NET="Socket",
        NCCL_IB_DISABLE="1",
        SGLANG_DIFFUSION_STAGE_LOGGING="0",
    )


def download_ranges(url, path, total, expected_sha, block_manifest=None):
    """Resume a stopped contiguous writer, then checkpoint complete HTTP ranges."""
    if path.exists() and path.stat().st_size == total and sha(path) == expected_sha:
        print("FROZEN_INPUT_ALREADY_VERIFIED", flush=True)
        return
    marker = path.with_name(path.name + ".ranges.json")
    if block_manifest is not None:
        block_bytes = block_manifest["block_bytes"]
        checks = block_manifest["blocks"]
        if (
            block_manifest["bytes"] != total
            or block_manifest["sha256"] != expected_sha
            or len(checks) != (total + block_bytes - 1) // block_bytes
        ):
            raise ValueError("Source block manifest identity differs")
        state = dict(
            total=total,
            sha256=expected_sha,
            prefix=0,
            block_bytes=block_bytes,
            completed=[],
        )
        if path.exists():
            with path.open("rb") as stream:
                for i, expected in enumerate(checks):
                    data = stream.read(min(block_bytes, total - i * block_bytes))
                    if hashlib.sha256(data).hexdigest() == expected:
                        state["completed"].append(i * block_bytes)
        marker = path.with_name(path.name + ".verified-blocks.json")
        write(marker, state)
        print(
            f"VERIFIED_EXISTING_INPUT_BLOCKS {len(state['completed'])}/{len(checks)}",
            flush=True,
        )
    else:
        if marker.exists():
            state = json.loads(marker.read_text())
            if state["total"] != total or state["sha256"] != expected_sha:
                raise ValueError("Download checkpoint identity differs")
        else:
            state = dict(
                total=total,
                sha256=expected_sha,
                prefix=path.stat().st_size if path.exists() else 0,
                block_bytes=16 << 20,
                completed=[],
            )
            if not 0 <= state["prefix"] <= total:
                raise ValueError("Invalid contiguous download prefix")
            write(marker, state)
    ranges = list(range(state["prefix"], total, state["block_bytes"]))
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)

    def fetch_once(start):
        end = min(total, start + state["block_bytes"])
        request = Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
        try:
            with urlopen(request, timeout=120) as response:
                if (
                    response.status != 206
                    or response.headers.get("Content-Range")
                    != f"bytes {start}-{end - 1}/{total}"
                ):
                    raise ValueError("Frozen input range response differs")
                offset = start
                while offset < end:
                    data = response.read(min(1 << 20, end - offset))
                    if not data or os.pwrite(fd, data, offset) != len(data):
                        raise ValueError("Incomplete frozen range write")
                    offset += len(data)
        except Exception as exc:
            raise RuntimeError(
                f"Frozen range download failed: {type(exc).__name__}, http={getattr(exc, 'code', None)}, start={start}, end={end}"
            ) from None
        return start

    def fetch(start):
        for attempt in range(3):
            try:
                return fetch_once(start)
            except RuntimeError:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    failures = []
    try:
        pending = [n for n in ranges if n not in state["completed"]]
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(fetch, n) for n in pending]
            for future in as_completed(futures):
                try:
                    completed = future.result()
                except RuntimeError as exc:
                    failures.append(str(exc))
                    print(str(exc), flush=True)
                    continue
                state["completed"].append(completed)
                temporary = marker.with_suffix(".next.json")
                write(temporary, state)
                temporary.replace(marker)
                done = state["prefix"] + sum(
                    min(state["block_bytes"], total - n) for n in state["completed"]
                )
                print(f"FROZEN_INPUT_DOWNLOAD {done}/{total}", flush=True)
    finally:
        os.close(fd)
    if failures:
        raise RuntimeError("Frozen input has failed ranges: " + "; ".join(failures))
    if path.stat().st_size != total or sha(path) != expected_sha:
        raise ValueError("Completed frozen download hash differs")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "generate", "collect"))
    p.add_argument("--exchange", type=Path, default=Path("/work"))
    args = p.parse_args()
    exchange = args.exchange
    spec = json.loads((exchange / "SPEC.json").read_text())
    root = Path(spec["owned_root"])
    if not root.is_absolute() or not root.name.startswith("pr693-frozen-"):
        raise ValueError("Invalid owned experiment root")
    if sha(__file__) != spec["stage_sha256"]:
        raise ValueError("Stage source differs from frozen spec")
    if args.mode == "prepare":
        resume = set(spec.get("resume_prepared_archives", []))
        if not resume <= {
            "runtime.tar.gz",
            "deps.tar.gz",
            "kernels.tar.gz",
            "encoder.tar.gz",
        }:
            raise ValueError("Invalid preparation resume boundary")
        root.mkdir(parents=True, exist_ok=bool(resume))
        for name, folder in (
            ("runtime.tar.gz", "source"),
            ("deps.tar.gz", "deps"),
            ("kernels.tar.gz", "kernels"),
            ("encoder.tar.gz", "results"),
            ("model-metadata.tar.gz", "model-metadata"),
        ):
            path = exchange / name
            remote = spec.get("remote_inputs", {}).get(name)
            if remote is not None:
                path = root / ("download-" + name)
                download_ranges(
                    remote, path, spec["file_sizes"][name], spec["files"][name]
                )
            if sha(path) != spec["files"][name]:
                raise ValueError("Archive hash differs: " + name)
            dest = root / folder
            if name in resume:
                if not dest.is_dir():
                    raise ValueError("Missing completed preparation directory")
                print("REUSED_PREPARED_ARCHIVE " + name, flush=True)
                continue
            if (
                name == "model-metadata.tar.gz"
                and dest.exists()
                and spec.get("replace_incomplete_metadata")
            ):
                if (root / "PREPARED.json").exists() or (root / "model").exists():
                    raise ValueError("Cannot replace metadata after model preparation")
                dest.rename(root / "model-metadata-r4-incomplete")
            dest.mkdir()
            with tarfile.open(path) as bundle:
                bundle.extractall(dest, filter="data")
        model = root / "model"
        model.mkdir()
        original = Path(spec["model"])
        config = json.loads((original / "zing_config.json").read_text())
        if config.get("checkpoint_sha256") != spec["checkpoint_sha256"]:
            raise ValueError("B300 model is not the frozen checkpoint")
        # Pin the original H100 runtime/geometry/profile bytes while referring
        # to the same checkpoint's existing CPFS weights, without copying them.
        for path in original.iterdir():
            if path.name not in ("zing_config.json", "zing_realtime_profile.json"):
                (model / path.name).symlink_to(path.resolve())
        for name in ("zing_config.json", "zing_realtime_profile.json"):
            shutil.copyfile(root / "model-metadata" / name, model / name)
        if sha(model / "zing_config.json") != spec["runtime_config_sha256"]:
            raise ValueError("Frozen H100 runtime configuration changed")
        for name in ("transformer/config.json", "video_vae/config.json"):
            if sha(original / name) != sha(root / "model-metadata" / name):
                actual = json.loads((original / name).read_text())
                expected = json.loads((root / "model-metadata" / name).read_text())
                write(
                    exchange / "MODEL_CONFIG_DIFF.json",
                    dict(
                        file=name,
                        keys={
                            k: dict(expected=expected.get(k), actual=actual.get(k))
                            for k in set(actual) | set(expected)
                            if actual.get(k) != expected.get(k)
                        },
                    ),
                )
                raise ValueError("Checkpoint geometry/config differs: " + name)
        command = [
            sys.executable,
            str(root / "source/benchmark/zing_h3_causal/run_sm90_frozen_review.py"),
            "--help",
        ]
        subprocess.run(command, env=environment(root), check=True)
        inventory = sorted((root / "results/encoded").glob("*/COMPLETE.json"))
        cases = sorted((root / "results/cases").glob("*/streaming-case.json"))
        if len(inventory) != 8 or len(cases) != 8:
            raise ValueError("Preparation requires all eight encoded cases")
        receipt = dict(
            passed=True,
            source_sha=spec["source_sha"],
            files=spec["files"],
            stage_sha256=spec["stage_sha256"],
            stage_source_sha=spec.get("stage_source_sha", spec["source_sha"]),
            checkpoint_sha256=spec["checkpoint_sha256"],
            runtime_config_sha256=sha(model / "zing_config.json"),
            case_ids=[json.loads(c.read_text())["case_id"] for c in cases],
            kernels_revision=spec["kernels_revision"],
            adaln_mode="resident",
            scope="CPU preflight; no CUDA inference executed",
        )
        write(root / "PREPARED.json", receipt)
        write(exchange / "PREPARED.json", receipt)
        print("CPU_PREPARED", flush=True)
    elif args.mode == "generate":
        if not json.loads((root / "PREPARED.json").read_text())["passed"]:
            raise ValueError("CPU preparation did not pass")
        env = environment(root)
        hardware_script = "import torch,json,subprocess; uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().splitlines(); assert len(uuids)==1; assert torch.cuda.device_count()==1; p=torch.cuda.get_device_properties(0); assert (p.major,p.minor)==(10,3) and p.total_memory>=250*(1<<30); print(json.dumps(dict(gpu_uuid=uuids[0],name=p.name,cc=[p.major,p.minor],memory_bytes=p.total_memory,torch=torch.__version__,cuda=torch.version.cuda)))"
        hardware = json.loads(
            subprocess.check_output(
                [sys.executable, "-c", hardware_script], env=env, text=True
            )
        )
        write(root / "GENERATOR_HARDWARE.json", hardware)
        command = [
            sys.executable,
            "-u",
            str(root / "source/benchmark/zing_h3_causal/run_sm90_frozen_review.py"),
            "--model",
            str(root / "model"),
            "--lynn",
            spec["lynn"],
            "--media-root",
            str(root / "unused-media"),
            "--output",
            str(root / "results"),
            "--source-sha",
            spec["source_sha"],
            "--adaln-mode",
            "resident",
            "--phase",
            "generate",
        ]
        write(root / "GENERATOR_COMMAND.json", command)
        subprocess.run(command, env=env, check=True)
        completed = sorted((root / "results/latents").glob("*/COMPLETE.json"))
        if len(completed) != 8:
            raise ValueError("Generator did not complete all eight cases")
        receipt = json.loads((root / "results/experiment.json").read_text())
        receipt.setdefault("phase_sources", {})["encoded"] = receipt["source_sha"]
        receipt["phase_sources"]["latents"] = spec["source_sha"]
        receipt["generator_hardware"] = hardware
        receipt["generator_adaln_mode"] = "resident"
        receipt["scope"] = (
            "B300 generates once; H100 Decoder A/B consumes identical tensors; no E2E or cross-hardware performance claim"
        )
        write(root / "results/experiment.json", receipt)
        write(
            root / "GENERATED.json",
            dict(passed=True, source_sha=spec["source_sha"], cases=8),
        )
        print("GPU_GENERATED_ALL_CASES", flush=True)
    else:
        if not json.loads((root / "GENERATED.json").read_text())["passed"]:
            raise ValueError("Generation did not pass")
        archive = exchange / "latents.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=1) as bundle:
            bundle.add(root / "results/latents", arcname="latents")
            bundle.add(root / "results/experiment.json", arcname="experiment.json")
            for name in (
                "PREPARED.json",
                "GENERATOR_HARDWARE.json",
                "GENERATOR_COMMAND.json",
                "GENERATED.json",
            ):
                bundle.add(root / name, arcname="generator/" + name)
        write(
            exchange / "COLLECTED.json",
            dict(
                passed=True,
                bytes=archive.stat().st_size,
                sha256=sha(archive),
                source_sha=spec["source_sha"],
            ),
        )
        print("CPU_COLLECTED", flush=True)


if __name__ == "__main__":
    main()
