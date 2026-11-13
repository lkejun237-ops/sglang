# SPDX-License-Identifier: Apache-2.0
"""Prepare on CPU, then restore and run the same immutable tree on GPU."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            result.update(data)
    return result.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "measure"])
    parser.add_argument("--exchange", type=Path, default=Path("/work"))
    parser.add_argument(
        "--root", type=Path, default=Path("/tmp/pr691-encoder-product-runtime/run")
    )
    args = parser.parse_args()
    root, exchange = args.root, args.exchange
    spec = json.loads((exchange / "SPEC.json").read_text())
    assert sha(Path(__file__)) == spec["files"]["stage.py"]
    archive = exchange / "prepared.tar.gz"
    if args.mode == "prepare":
        root.mkdir(parents=True, exist_ok=False)
        for name, folder in [
            ("runtime.tar.gz", "source"),
            ("harness.tar.gz", "harness"),
            ("deps.tar.gz", "deps"),
            ("kernels.tar.gz", "kernels"),
        ]:
            print("PREPARING_ARCHIVE", name, flush=True)
            assert sha(exchange / name) == spec["files"][name], name
            destination = root / folder
            destination.mkdir()
            with tarfile.open(exchange / name) as bundle:
                bundle.extractall(destination, filter="data")
        write(root / "RUN_CONFIG.json", spec["config"])
        write(
            root / "SOURCE.json",
            {
                "runtime_revision": spec["runtime_revision"],
                "harness_revision": spec["harness_revision"],
                "kernels_revision": spec["kernels_revision"],
                "package_hashes": spec["files"],
            },
        )
        for label in ["model", "lynn"]:
            source = Path(spec["assets"][label])
            print("PREPARING_ASSET", label, str(source), flush=True)
            if label == "model" and spec["config"]["compute_capability"] == [10, 3]:
                candidates = [source]
                inventory = []
                for candidate in candidates:
                    try:
                        config = json.loads(
                            (candidate / "zing_config.json").read_text()
                        )
                        inventory.append(
                            {
                                "path": str(candidate),
                                "checkpoint": config.get("checkpoint_sha256"),
                                "complete": all(
                                    p.exists() for p in candidate.iterdir()
                                ),
                            }
                        )
                    except (OSError, ValueError):
                        continue
                write(exchange / "MODEL_DISCOVERY.json", inventory)
                matches = [
                    row["path"]
                    for row in inventory
                    if row["checkpoint"] == spec["checkpoint_sha256"]
                    and row["complete"]
                ]
                assert matches, (
                    "Selected export does not match the frozen checkpoint; see MODEL_DISCOVERY.json"
                )
                source = Path(matches[0])
                spec["assets"]["model"] = str(source)
            assert source.is_dir(), source
            (root / label).mkdir()
            for file in source.iterdir():
                target = root / label / file.name
                if file.name in (
                    "zing_config.json",
                    "zing_realtime_profile.json",
                    "model_index.json",
                ):
                    shutil.copy2(file, target)
                else:
                    target.symlink_to(file.resolve(), target_is_directory=file.is_dir())
        media = root / "media"
        media.mkdir()
        source = Path(spec["assets"]["media"])
        for item in json.loads((exchange / "asset-downloads.json").read_text()):
            print("PREPARING_INPUT", item["sha256"], flush=True)
            target = media / item["relative"]
            target.parent.mkdir(parents=True, exist_ok=True)
            original = source / item["relative"]
            if not original.is_file() and item.get("object_key"):
                original = source / item["object_key"]
            if not original.is_file():
                original = (
                    Path(spec.get("input_cache", str(exchange / "inputs")))
                    / item["sha256"]
                )
            if original.is_file():
                assert (
                    original.stat().st_size == item["bytes"]
                    and sha(original) == item["sha256"]
                ), original
                target.symlink_to(original.resolve())
            else:
                import urllib.request

                original = root / "download-cache" / item["sha256"]
                original.parent.mkdir(exist_ok=True)
                if not original.exists():
                    try:
                        with (
                            urllib.request.urlopen(
                                item["url"], timeout=120
                            ) as response,
                            original.open("xb") as target_file,
                        ):
                            shutil.copyfileobj(response, target_file)
                    except Exception:
                        raise RuntimeError(
                            "Frozen input download failed: " + item["relative"]
                        ) from None
                assert (
                    original.stat().st_size == item["bytes"]
                    and sha(original) == item["sha256"]
                )
                target.symlink_to(original.resolve())
        write(root / "ASSET_PATHS.json", spec["assets"])
        wheels = exchange / "wheels"
        for name, expected in spec["wheels"].items():
            assert sha(wheels / name) == expected
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--find-links",
                str(wheels),
                "--target",
                str(root / "transport-deps"),
                "uvicorn==0.35.0",
                "websockets==15.0.1",
                "redis==6.4.0",
            ],
            check=True,
        )
    else:
        prepared = json.loads((exchange / "PREPARED.json").read_text())
        spec["assets"] = prepared["asset_paths"]
        assert prepared["passed"] and sha(archive) == prepared["archive_sha256"]
        assert not root.parent.exists()
        # Absolute links only point to pre-existing read-only assets.
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                path = Path(member.name)
                assert not path.is_absolute() and ".." not in path.parts
                if member.issym():
                    target = Path(
                        os.path.normpath(
                            str(root.parent.parent / path.parent / member.linkname)
                        )
                    )
                    assert (
                        target.is_relative_to(root.parent)
                        or str(target) in prepared["external_link_targets"]
                    )
                elif member.islnk():
                    assert (
                        not Path(member.linkname).is_absolute()
                        and ".." not in Path(member.linkname).parts
                    )
                else:
                    assert member.isfile() or member.isdir()
            bundle.extractall(root.parent.parent, filter="fully_trusted")
    env = dict(
        os.environ,
        PYTHONPATH=f"{root}/kernels/src:{root}/transport-deps:{root}/deps:{root}/source/python:{root}/source/benchmark/zing_h3_causal:{root}/harness",
        PYTHONDONTWRITEBYTECODE="1",
        SGLANG_LIGHTWEIGHT_RUNTIME="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        NCCL_IB_DISABLE="1",
        NCCL_NET="Socket",
        NCCL_DEBUG="WARN",
        NCCL_GRAPH_REGISTER="0",
        SGLANG_ZING_EXPERIMENTAL_ATTENTION="fa4",
        SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES="1",
        SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN="1",
        SGLANG_ZING_DENOISE_CUDA_GRAPH="1",
        SGLANG_ZING_DENOISE_CUDA_GRAPH_SP="1",
        SGLANG_ZING_SOURCE_PROMPT_JOINT="0",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    for name in [
        "HF_RMSNORM",
        "FUSED_MROPE",
        "FUSED_GATE_UP",
        "TEXT_FA4",
        "VISION_FA4",
        "FP8",
    ]:
        env["SGLANG_H3_QWEN_" + name] = "0"
    if args.mode == "prepare":
        env.update(
            CUDA_VISIBLE_DEVICES="",
            NVIDIA_VISIBLE_DEVICES="void",
            SGLANG_DIFFUSION_PLATFORM_OVERRIDE="cpu",
        )
    else:
        env.pop("SGLANG_DIFFUSION_PLATFORM_OVERRIDE", None)
        env.pop("NVIDIA_VISIBLE_DEVICES", None)
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    output = exchange if args.mode == "prepare" else root / "results"
    command = [
        sys.executable,
        "-u",
        str(root / "harness/runner.py"),
        args.mode,
        "--root",
        str(root),
        "--output",
        str(output),
        "--revision",
        spec["runtime_revision"],
    ]
    if args.mode == "prepare":
        subprocess.run(command, env=env, cwd=root / "source", check=True)
        external_targets = sorted(
            {
                str(p.resolve())
                for p in root.rglob("*")
                if p.is_symlink() and not p.resolve().is_relative_to(root.parent)
            }
        )
        with tarfile.open(archive, "w:gz", dereference=False) as bundle:
            bundle.add(root.parent, arcname=root.parent.name)
        original = root.parent.with_name(root.parent.name + "-cpu-original")
        assert not original.exists()
        root.parent.rename(original)
        with tarfile.open(archive) as bundle:
            bundle.extractall(root.parent.parent, filter="fully_trusted")
        ready = json.loads((root / "CPU_READY.json").read_text())
        for relative, expected in ready["controls"].items():
            assert sha(root / relative) == expected, relative
        shutil.rmtree(original)
        write(
            exchange / "PREPARED.json",
            {
                "passed": True,
                "restore_cpu_verified": True,
                "archive_sha256": sha(archive),
                "archive_bytes": archive.stat().st_size,
                "runtime_revision": spec["runtime_revision"],
                "asset_paths": spec["assets"],
                "external_link_targets": external_targets,
            },
        )
        print("ENCODER_PRODUCT_PREPARED", flush=True)
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "python/sglang/multimodal_gen/test/unit/test_zing_h3_fast_single_gpu_encoder.py",
            ],
            env=env,
            cwd=root / "source",
            check=True,
        )
        process = subprocess.Popen(command, env=env, cwd=root / "source")
        try:
            while process.poll() is None:
                for name in ["STATUS.json", "SUMMARY.json", "FAILURES.json"]:
                    path = output / name
                    if path.exists():
                        shutil.copy2(path, exchange / name)
                for path in output.rglob("*"):
                    if not path.is_file() or path.suffix not in (
                        ".json",
                        ".jsonl",
                        ".log",
                        ".txt",
                    ):
                        continue
                    destination = exchange / "live" / path.relative_to(output)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if (
                        not destination.exists()
                        or path.stat().st_size != destination.stat().st_size
                    ):
                        shutil.copy2(path, destination)
                time.sleep(15)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
            if output.exists():
                shutil.copytree(output, exchange / "results", dirs_exist_ok=True)
        raise SystemExit(process.returncode)


if __name__ == "__main__":

    def terminate(*unused):
        raise KeyboardInterrupt("Termination requested")

    signal.signal(signal.SIGTERM, terminate)
    main()
