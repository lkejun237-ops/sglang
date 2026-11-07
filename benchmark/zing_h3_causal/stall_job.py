# SPDX-License-Identifier: Apache-2.0
"""Prepare a frozen isolated suite on CPU, run it on allocated GPUs, collect raw metadata.

The JSON configuration supplies paths and exact published source identities. This
entry never creates cloud resources or restarts an existing deployment.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def instance_devices(config):
    """Count explicit instance devices; CPU Runner admission checks role sharing."""
    if config.get("dit_resident_collection") is True:
        minimum = config.get("minimum_round_seconds", 600)
        if type(minimum) is not int or minimum not in (0, 600):
            raise ValueError("DiT collection duration must be 0 or 600 seconds")
        allocated = config.get("allocated_gpu_count", 8)
        roles = dict(dit=[0, 1, 2, 3], encoder=[], decoder=[])
        if (
            type(allocated) is not int
            or allocated not in [4, 8]
            or config.get("role_devices", roles) != roles
            or config.get("allow_shared_role_devices", False)
        ):
            raise ValueError("DiT-only collection requires four dedicated model GPUs")
        return allocated, 4, roles
    decoder_world = config.get("decoder_world", 1)
    if type(decoder_world) is not int or decoder_world not in [1, 2, 3]:
        raise ValueError("Expected one, two or three Decoder GPUs")
    roles = config.get(
        "role_devices",
        dict(dit=[0, 1, 2, 3], encoder=[4], decoder=list(range(5, 5 + decoder_world))),
    )
    if (
        not isinstance(roles, dict)
        or set(roles) != {"dit", "encoder", "decoder"}
        or any(not isinstance(values, list) or not values for values in roles.values())
    ):
        raise ValueError("Expected explicit nonempty lists for all three model roles")
    indices = [index for values in roles.values() for index in values]
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("Expected nonnegative CUDA ordinals")
    active = len(set(indices))
    allocated = config.get("allocated_gpu_count", 8)
    if (
        type(allocated) is not int
        or allocated not in [4, 6, 8]
        or active > allocated
        or sorted(set(indices)) != list(range(active))
        or len(roles["decoder"]) != decoder_world
    ):
        raise ValueError(
            "Expected four, six or eight allocated GPUs and contiguous role devices"
        )
    return allocated, active, roles


def assets(root):
    verified = []
    rows = json.loads((root / "ASSETS.json").read_text())
    if (root / "COMMON_CAPTURE_FILES.json").exists():
        rows += json.loads((root / "COMMON_CAPTURE_FILES.json").read_text())
    for row in rows:
        path = Path(row["path"])
        stat = path.stat()
        actual = dict(
            resolved=str(path.resolve()),
            bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            inode=stat.st_ino,
        )
        changed = any(actual[key] != row[key] for key in actual)
        if changed and sha(path) != row["sha256"]:
            raise ValueError(f"Frozen asset changed: {path}")
        verified.append(dict(path=str(path), stat_unchanged=not changed))
    return verified


def environment(root):
    runtime = (
        root / "runtime-source"
        if (root / "runtime-source").exists()
        else root / "source"
    )
    env = dict(os.environ)
    env.update(
        PYTHONPATH=f"{root}/deps:{runtime}/python:{root}/source/benchmark/zing_h3_causal",
        SGLANG_LIGHTWEIGHT_RUNTIME="1",
        SGLANG_ZING_FREEZE_STATIC_GC="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        NCCL_MIN_NCHANNELS="4",
    )
    return env


def extract_source(source, archive_path, object_path, revision):
    source.mkdir()
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        if any(
            Path(m.name).is_absolute() or ".." in Path(m.name).parts or m.islnk()
            for m in members
        ):
            raise ValueError("Invalid source archive member")
        archive.extractall(source, filter="data")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    with Path(object_path).open("rb") as stream:
        subprocess.run(
            ["git", "-C", str(source), "index-pack", "--stdin"],
            stdin=stream,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "update-ref",
            "refs/heads/frozen",
            revision,
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "symbolic-ref", "HEAD", "refs/heads/frozen"],
        check=True,
    )


def prepare(config, root, output):
    if (
        sum(
            bool(config.get(key))
            for key in [
                "runtime_source_revision",
                "encoder_gc_comparison",
                "dit_context_probe",
                "decoder_layout_pilot",
                "native_rng_probe",
                "dit_resident_collection",
            ]
        )
        > 1
    ):
        raise ValueError(
            "Main baseline, Encoder GC and DiT context probe are separate suites"
        )
    root.mkdir(mode=0o700)
    output.mkdir(parents=True, exist_ok=True)
    for name, digest in config["source_files"].items():
        if sha(name) != digest:
            raise ValueError(f"Source package hash mismatch: {name}")
    source = root / "source"
    extract_source(
        source,
        config["source_archive"],
        config["source_objects"],
        config["source_revision"],
    )
    runtime = source
    if config.get("runtime_source_revision"):
        runtime = root / "runtime-source"
        extract_source(
            runtime,
            config["runtime_source_archive"],
            config["source_objects"],
            config["runtime_source_revision"],
        )
    reuse = Path(config["reuse_root"])
    for name in ["deps", "model", "lynn", "media", "freeze"]:
        (root / name).symlink_to(reuse / name, target_is_directory=True)
    for name in [
        "ASSETS.json",
        "MODEL_IDENTITY.json",
        "FROZEN_RGB.json",
        "pipeline.json",
        "conditioning-profile.json",
    ]:
        shutil.copyfile(reuse / name, root / name)
    cfg = json.loads((reuse / "runner-config.json").read_text())
    _, active, roles = instance_devices(config)
    cfg.update(
        root=str(root / "work"),
        source=str(runtime),
        deps=str(root / "deps"),
        model=str(root / "model"),
        pipeline=str(root / "pipeline.json"),
        frozen_inputs=str(root / "FROZEN_RGB.json"),
        conditioning_profile=str(root / "conditioning-profile.json"),
        cpu_receipt=str(root / "CPU_READY.json"),
        before_import_gpu_apps=str(root / "GPU_BEFORE_IMPORT.csv"),
        diagnostic_switch=config["diagnostic_switch"],
        archive_dir=str(output / "raw-blobs"),
        ports=config["ports"],
        role_devices=roles,
        allow_shared_role_devices=config.get("allow_shared_role_devices", False),
        dit_only=config.get("dit_resident_collection") is True,
    )
    for key in ["candidate_control_config", "baseline_control_config"]:
        cfg.pop(key, None)
    if (
        config.get("runtime_source_revision")
        or config.get("dit_context_probe")
        or config.get("decoder_layout_pilot")
        or config.get("dit_resident_collection")
    ):
        captures = sorted(Path(config["capture_root"]).iterdir())
        if len(captures) != 2 or any(not p.is_dir() for p in captures):
            raise ValueError("Expected two complete common DiT captures")
        rows = []
        for capture in captures:
            complete = json.loads((capture / "COMPLETE.json").read_text())
            if complete["input_frames"] != 15921 or complete["capture_scope"] != "dit":
                raise ValueError("Incomplete common DiT capture")
            for path in sorted(capture.rglob("*")):
                if not path.is_file():
                    continue
                stat = path.stat()
                rows.append(
                    dict(
                        path=str(path),
                        resolved=str(path.resolve()),
                        bytes=stat.st_size,
                        mtime_ns=stat.st_mtime_ns,
                        inode=stat.st_ino,
                        sha256=sha(path),
                    )
                )
        cfg["common_captures"] = list(map(str, captures))
        write(root / "COMMON_CAPTURE_FILES.json", rows)
        if config.get("decoder_layout_pilot"):
            identities = json.loads(Path(config["capture_index"]).read_text())
            if {row["path"] for row in identities} != set(map(str, captures)):
                raise ValueError("Decoder parity index differs from frozen captures")
            write(root / "COMMON_CAPTURES.json", identities)
        if config.get("runtime_source_revision") or (
            config.get("dit_resident_collection") and config.get("matched_allocation")
        ):
            shutil.copyfile(
                config["matched_allocation"], root / "MATCHED_ALLOCATION.json"
            )
    ports = list(cfg["ports"].values())
    if len(set(ports)) != len(ports) or any(
        type(port) is not int or not 1024 < port < 32768 for port in ports
    ):
        raise ValueError("Expected distinct listening ports below the ephemeral range")
    write(root / "runner-config.json", cfg)
    shutil.copyfile(__file__, root / "job.py")
    write(root / "job-config.json", config)
    env = environment(root)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    bench = source / "benchmark/zing_h3_causal"
    code = (
        "import torch;from frozen_source import verify_source;"
        f"verify_source({str(source)!r},{config['source_revision']!r});"
        f"verify_source({str(runtime)!r},{config.get('runtime_source_revision', config['source_revision'])!r});"
        "from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker;"
        "from sglang.multimodal_gen.runtime.pipelines.zing_h3_causal_pipeline import ZingH3CausalPipeline;"
        "from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker import initialize_runtime_imports;"
        "initialize_runtime_imports();"
        "from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.loader import load_lynn_decoder;"
        "from run_isolated_stall_suite import role_layout;import json;"
        f"cfg=json.load(open({str(root / 'runner-config.json')!r}));"
        f"assert cfg['frozen_inputs']=={str(root / 'FROZEN_RGB.json')!r};"
        f"assert cfg['conditioning_profile']=={str(root / 'conditioning-profile.json')!r};"
        f"assert role_layout(cfg,{active})=={roles!r};"
        "assert cfg['sp']==4 and cfg['mem_fraction_static']==0.8;"
        "assert torch.cuda.device_count()==0 and not torch.cuda.is_initialized();"
        "print('ACTUAL_IMPORTS_ZERO_GPU_PASS')"
    )
    with (root / "cpu-imports.log").open("w") as log:
        subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=240,
        )
    if config.get("native_rng_probe"):
        # Installing observations must leave ordinary CPU execution untouched
        # and must not initialize CUDA before the scheduler grants any GPU.
        code = (
            "import torch;from diagnostics.dit_replay.dit_rng_probe import install;"
            "from sglang.multimodal_gen.runtime.models.dits.zing_h3 import FixedForwardPlan,ZingH3DiTModel;"
            "from sglang.multimodal_gen.runtime.layers.linear import ColumnParallelLinear,ReplicatedLinear,RowParallelLinear;"
            "original=FixedForwardPlan.execute;"
            "methods=('_embed_inputs','_block_forward','_should_fuse_pre_linear_quant');"
            "before=[getattr(ZingH3DiTModel,k) for k in methods];"
            "linears=(ColumnParallelLinear,ReplicatedLinear,RowParallelLinear);"
            "linear_before=[cls.forward for cls in linears];"
            "plan=FixedForwardPlan({'x':torch.ones(2)},(),None,lambda v:(v['x']+1,None));"
            "restore=install();result=plan.execute();restore();"
            "assert torch.equal(result[0],torch.full((2,),2.));"
            "assert FixedForwardPlan.execute is original;"
            "assert before==[getattr(ZingH3DiTModel,k) for k in methods];"
            "assert linear_before==[cls.forward for cls in linears];"
            "assert torch.cuda.device_count()==0 and not torch.cuda.is_initialized();"
            "print('PROBE_INSTALL_RESTORE_CPU_EXECUTION_ZERO_GPU_PASS')"
        )
        if config.get("verify_terminal_recovery", False):
            code += (
                ";import inspect;from run_isolated_stall_suite import Runner;"
                "inspect.signature(Runner.client).bind(None,'terminal-profile',"
                "width=832,height=480,diagnostic=True);"
                "inspect.signature(Runner.run_lifecycle).bind(None);"
                "print('TERMINAL_PROFILE_AND_LIFECYCLE_API_BINDING_PASS')"
            )
        with (root / "cpu-probe-imports.log").open("w") as log:
            subprocess.run(
                [sys.executable, "-c", code],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=240,
            )
    if config.get("runtime_source_revision"):
        with (root / "cpu-baseline-adapter.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(bench / "run_main_dit_replay.py"),
                    "--harness-source",
                    str(source),
                    "--help",
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
    if config.get("encoder_gc_comparison"):
        with (root / "cpu-encoder-gc-comparison.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(bench / "diagnostics/encoder/run_encoder_gc_comparison.py"),
                    "--config",
                    str(root / "job-config.json"),
                    "--preflight-only",
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
    if config.get("dit_context_probe"):
        with (root / "cpu-dit-context-probe.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(bench / "diagnostics/dit_replay/run_dit_context_probe.py"),
                    "--config",
                    str(root / "job-config.json"),
                    "--preflight-only",
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
    if config.get("decoder_layout_pilot"):
        with (root / "cpu-decoder-layout-pilot.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(bench / "diagnostics/decoder/run_decoder_layout_pilot.py"),
                    "--config",
                    str(root / "job-config.json"),
                    "--preflight-only",
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
    contracts = config.get(
        "cpu_contracts", ["benchmark/zing_h3_causal/tests/test_stream_stalls.py"]
    )
    if config.get("dit_resident_collection") and not {
        "benchmark/zing_h3_causal/tests/test_stream_stalls.py",
        "benchmark/zing_h3_causal/tests/test_replay_collection_decision.py",
    }.issubset(contracts):
        raise ValueError(
            "DiT collection preflight requires resident and verdict contracts"
        )
    for path in contracts:
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("CPU contract path must stay inside the frozen source")
    for trial in [1, 2]:
        with (root / f"cpu-contracts-{trial}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    *[str(source / path) for path in contracts],
                    "--junitxml",
                    str(root / f"cpu-contracts-{trial}.xml"),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        junit = ET.parse(root / f"cpu-contracts-{trial}.xml")
        if (
            not junit.findall(".//testcase")
            or junit.findall(".//skipped")
            or junit.findall(".//failure")
            or junit.findall(".//error")
        ):
            raise RuntimeError("CPU contracts must complete without failures or skips")
    assets(root)
    files = [
        root / name
        for name in [
            "job.py",
            "job-config.json",
            "runner-config.json",
            "ASSETS.json",
            "MODEL_IDENTITY.json",
            "FROZEN_RGB.json",
            "pipeline.json",
            "conditioning-profile.json",
            "cpu-imports.log",
            "cpu-contracts-1.log",
            "cpu-contracts-2.log",
            "cpu-contracts-1.xml",
            "cpu-contracts-2.xml",
        ]
    ]
    if config.get("runtime_source_revision"):
        files += [
            root / "COMMON_CAPTURE_FILES.json",
            root / "MATCHED_ALLOCATION.json",
            root / "cpu-baseline-adapter.log",
        ]
    if config.get("encoder_gc_comparison"):
        files.append(root / "cpu-encoder-gc-comparison.log")
    if config.get("dit_context_probe"):
        files += [
            root / "COMMON_CAPTURE_FILES.json",
            root / "cpu-dit-context-probe.log",
        ]
    if config.get("decoder_layout_pilot"):
        files += [
            root / "COMMON_CAPTURE_FILES.json",
            root / "COMMON_CAPTURES.json",
            root / "cpu-decoder-layout-pilot.log",
        ]
    if config.get("native_rng_probe"):
        files.append(root / "cpu-probe-imports.log")
    if config.get("dit_resident_collection"):
        files.append(root / "COMMON_CAPTURE_FILES.json")
        if (root / "MATCHED_ALLOCATION.json").exists():
            files.append(root / "MATCHED_ALLOCATION.json")
    ready = dict(
        passed=True,
        source_revision=config["source_revision"],
        time=time.time(),
        controls={str(path): sha(path) for path in files},
        scope="Actual CPU preparation only; GPU collection and acceptance remain separate",
        cpu_contracts={path: sha(source / path) for path in contracts},
    )
    if config.get("runtime_source_revision"):
        ready["runtime_source_revision"] = config["runtime_source_revision"]
    write(root / "CPU_READY.json", ready)
    write(output / "CPU_READY.json", ready)
    print("CPU_READY_SHA256", sha(root / "CPU_READY.json"), flush=True)


def collect(root, output):
    rows = []
    archive_path = output / "GPU_COLLECTION.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink() or ".secret." in path.name:
                continue
            if path.relative_to(root).parts[0] in {
                "source",
                "runtime-source",
                "deps",
                "model",
                "lynn",
                "media",
                "freeze",
            }:
                continue
            if path.suffix not in {
                ".json",
                ".jsonl",
                ".csv",
                ".log",
                ".txt",
                ".xml",
                ".html",
                ".js",
                ".css",
            }:
                continue
            relative = str(path.relative_to(root))
            archive.add(path, arcname=relative, recursive=False)
            rows.append(
                dict(path=relative, bytes=path.stat().st_size, sha256=sha(path))
            )
    write(
        output / "GPU_COLLECTION.json",
        dict(archive_sha256=sha(archive_path), files=rows, time=time.time()),
    )


def run(config, root, output):
    ready = json.loads((root / "CPU_READY.json").read_text())
    if not ready["passed"] or ready["source_revision"] != config["source_revision"]:
        raise ValueError("CPU source identity mismatch")
    for path, digest in ready["controls"].items():
        if sha(path) != digest:
            raise ValueError(f"Frozen control changed: {path}")
    sys.path.insert(0, str(root / "source/benchmark/zing_h3_causal"))
    from frozen_source import verify_source

    verify_source(root / "source", config["source_revision"])
    if config.get("runtime_source_revision"):
        if ready.get("runtime_source_revision") != config["runtime_source_revision"]:
            raise ValueError("CPU runtime source identity mismatch")
        verify_source(root / "runtime-source", config["runtime_source_revision"])
    from owned_process import owned_popen
    from run_isolated_stall_suite import stop

    assets(root)
    before = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv"],
        text=True,
    )
    (root / "GPU_BEFORE_IMPORT.csv").write_text(before)
    inventory = list(
        csv.reader(
            io.StringIO(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,uuid,name,memory.total,memory.used",
                        "--format=csv,noheader",
                    ],
                    text=True,
                )
            )
        )
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if config.get("dit_context_probe"):
        allocated = active = 4
    else:
        allocated, active, _ = instance_devices(config)
    devices = (
        visible.split(",")
        if visible is not None
        else [str(i) for i in range(allocated)]
    )
    if (
        len(devices) != allocated
        or len(set(devices)) != allocated
        or any(not d.isascii() or not d.isdecimal() for d in devices)
    ):
        raise ValueError(f"Expected {allocated} allocated numeric CUDA devices")
    physical = {row[0].strip(): row for row in inventory}
    if any(
        physical[d][1].strip() in before
        or int(physical[d][-1].strip().split()[0]) >= 32
        for d in devices
    ):
        raise RuntimeError("Allocated GPU is still occupied")
    allocation = dict(
        time=time.time(),
        scheduler_allocated_gpu_count=allocated,
        active_gpu_count=active,
        unused_gpu_count=allocated - active,
        allocated_cuda_ordinals=devices,
        physical_inventory=inventory,
        all_allocated_devices_unoccupied_before_import=True,
    )
    if (root / "MATCHED_ALLOCATION.json").exists():
        prior = json.loads((root / "MATCHED_ALLOCATION.json").read_text())
        prior_map = {
            row[0].strip(): row[1].strip() for row in prior["physical_inventory"]
        }
        if [physical[d][1].strip() for d in devices] != [
            prior_map[d] for d in prior["allocated_cuda_ordinals"]
        ]:
            raise ValueError(
                "Baseline requires the exact candidate GPU UUID allocation"
            )
        allocation["matched_all_candidate_uuids"] = True
    write(root / "ALLOCATION.json", allocation)
    write(output / "ALLOCATION.json", allocation)
    env = environment(root)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[:active])
    finished = threading.Event()

    def heartbeat():
        while not finished.wait(30):
            try:
                active_root = root / "ACTIVE_MEASUREMENT_ROOT.json"
                work = (
                    Path(json.loads(active_root.read_text())["root"])
                    if active_root.exists()
                    else root / "work"
                )
                results = work / "results"
                status = results / "STATUS.json"
                row = (
                    json.loads(status.read_text())
                    if status.exists()
                    else dict(phase="startup")
                )
                requests = []
                for path in results.rglob("requests.jsonl"):
                    requests.extend(
                        json.loads(line)
                        for line in path.read_text().splitlines()
                        if line
                    )
                row.update(
                    time=time.time(),
                    completed_requests=len(requests),
                    failed_requests=sum(not r.get("success") for r in requests),
                )
                active = results / row["phase"] / "active-request.json"
                if not active.exists():
                    active = results / (row["phase"] + "-active-request.json")
                if active.exists():
                    row["active"] = json.loads(active.read_text())
                    row["active_output_frames"] = sum(
                        event["header"]["num_frames"]
                        for path in Path(row["active"]["output"]).rglob("events.jsonl")
                        for line in path.read_text().splitlines()
                        if line
                        for event in [json.loads(line)]
                        if event.get("kind") == "frames"
                    )
                write(output / "HEARTBEAT.json", row)
                print("HEARTBEAT", json.dumps(row), flush=True)
            except Exception as error:
                print("HEARTBEAT_READ_ERROR", type(error).__name__, flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    process = None
    error = None
    try:
        process = owned_popen(
            [
                sys.executable,
                str(
                    root
                    / "source/benchmark/zing_h3_causal"
                    / (
                        "run_main_stall_suite.py"
                        if config.get("runtime_source_revision")
                        else "diagnostics/encoder/run_encoder_gc_comparison.py"
                        if config.get("encoder_gc_comparison")
                        else "diagnostics/dit_replay/run_dit_context_probe.py"
                        if config.get("dit_context_probe")
                        else "diagnostics/decoder/run_decoder_layout_pilot.py"
                        if config.get("decoder_layout_pilot")
                        else "diagnostics/dit_replay/run_native_rng_probe.py"
                        if config.get("native_rng_probe")
                        else "run_dit_resident_collection.py"
                        if config.get("dit_resident_collection")
                        else "run_isolated_stall_suite.py"
                    )
                ),
                "--config",
                str(
                    root
                    / (
                        "job-config.json"
                        if config.get("encoder_gc_comparison")
                        or config.get("dit_context_probe")
                        or config.get("decoder_layout_pilot")
                        or config.get("native_rng_probe")
                        or config.get("dit_resident_collection")
                        else "runner-config.json"
                    )
                ),
            ],
            env=env,
            start_new_session=True,
        )
        if process.wait(timeout=config["timeout_seconds"]):
            raise RuntimeError("Isolated suite failed; inspect original child logs")
    except BaseException:
        error = traceback.format_exc()
        (root / "GPU_FAILURE.txt").write_text(error)
        raise
    finally:
        cleanup_error = None
        try:
            if process is not None:
                stop(process)
        except BaseException:
            cleanup_error = traceback.format_exc()
            raise
        finally:
            finished.set()
            thread.join(timeout=10)
            write(
                output / "GPU_EXECUTION_RESULT.json",
                dict(
                    time=time.time(),
                    error=error,
                    cleanup_error=cleanup_error,
                    regression_accepted=False,
                    scope="CPU readback still required",
                ),
            )
            collect(root, output)
            write(
                output / "GPU_RESULT.json",
                dict(
                    time=time.time(),
                    collection_completed=error is None and cleanup_error is None,
                    regression_accepted=False,
                    error=error,
                    cleanup_error=cleanup_error,
                    scope="Metadata collected; independent CPU acceptance and raw retention still required",
                ),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root, output = Path(config["root"]), Path(config["output"])
    (prepare if args.mode == "prepare" else run)(config, root, output)


if __name__ == "__main__":
    main()
