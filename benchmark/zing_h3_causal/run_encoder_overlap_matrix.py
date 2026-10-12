# SPDX-License-Identifier: Apache-2.0
"""Run a prepared, at-most-eight-GPU experiment serially on one node.

Input JSON: model, request, source_rgb, soak_rgb, taeh3_weights. Prepare RGB and
local reference URIs in a zero-GPU CPU preparation job first. No cloud submission is performed.
Outputs are append-only; existing cells are never silently reused.
"""

import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

from encoder_execution_evidence import (
    capture_session,
    endpoint_binding,
    input_geometry,
    validate_correctness,
    validate_encoder_acceptance,
    validate_profile_binding,
    validate_serial_overlap,
    validate_serial_overlap_parity,
)
from encoder_overlap_config import layouts, selected_layouts
from prepare_encoder_overlap import digest

BENCH = Path(__file__).resolve().parent
WORKER = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2))


_TILE_ENV = "MINIMAX_H3_PROGRESSIVE_ENCODER_TILING"
_DECODE_ENV = "SGLANG_ZING_EXPERIMENTAL_PROGRESSIVE_DECODE"


def role_environment(config, role, devices, base=None):
    mode = config.get("encoder_tile_mode", "parallel")
    if mode not in ("auto", "parallel"):
        raise ValueError("encoder_tile_mode must be auto or parallel")
    progressive = config.get("progressive_decode", False)
    if type(progressive) is not bool:
        raise ValueError("progressive_decode must be boolean")
    env = dict(os.environ if base is None else base)
    env[_DECODE_ENV] = "1" if progressive and role != "encoder" else "0"
    # A shell-wide experimental override must never alter the shared baseline.
    env[_TILE_ENV] = mode if role == "encoder" else "auto"
    env["CUDA_VISIBLE_DEVICES"] = devices
    return env


def record_role_environment(path, **roles):
    write_json(
        path,
        {
            role: {
                key: env[key]
                for key in (_TILE_ENV, _DECODE_ENV, "CUDA_VISIBLE_DEVICES")
            }
            for role, env in roles.items()
        },
    )


def prerequisite_gate(name, check, *, exploratory, records, path):
    """Only explicitly named acceptance gates may be bypassed for collection."""
    try:
        check()
    except (RuntimeError, FileNotFoundError) as exc:
        records[name] = dict(passed=False, error=f"{type(exc).__name__}: {exc}")
        write_json(path, records)
        if not exploratory:
            raise
    else:
        records[name] = dict(passed=True)
        write_json(path, records)


def command_run(cmd, output, env=None, timeout=14400):
    write_json(output.with_suffix(".command.json"), cmd)
    with output.open("w") as log:
        subprocess.run(
            cmd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )


def lifecycle_gate(cmd, out, env, *, exploratory):
    """Only an isolated probe process failure may be collected experimentally."""
    receipt = dict(
        passed=False,
        exploratory=exploratory,
        log=str(out / "lifecycle.log"),
        artifacts=str(out / "lifecycle"),
    )
    try:
        command_run(cmd, out / "lifecycle.log", env)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        receipt.update(
            error=f"{type(exc).__name__}: {exc}",
            exit_code=getattr(exc, "returncode", None),
            timeout_seconds=getattr(exc, "timeout", None),
        )
        write_json(out / "lifecycle-gate.json", receipt)
        if not exploratory:
            raise
    else:
        receipt.update(passed=True, exit_code=0)
        write_json(out / "lifecycle-gate.json", receipt)
    return receipt


def snapshot_process_groups(children):
    """Capture PID identities while both owned launchers are still alive."""
    import psutil

    groups = {child.pid for child in children}
    if any(child.poll() is not None for child in children):
        raise RuntimeError("Cannot snapshot an exited lifecycle process group")
    members = []
    for proc in psutil.process_iter():
        try:
            if os.getpgid(proc.pid) in groups:
                proc.create_time()  # Cache identity before cancellation can reap it.
                members.append(proc)
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
    if not groups.issubset({proc.pid for proc in members}):
        raise RuntimeError("Lifecycle process group changed during ownership snapshot")
    return groups, members


def stop_failed_process_groups(snapshot):
    """Signal captured process identities, never a potentially reused PGID."""
    import psutil

    groups, members = snapshot
    zombies = {}

    def live(processes):
        result = []
        for proc in processes:
            try:
                if not proc.is_running():
                    continue
                if proc.status() == psutil.STATUS_ZOMBIE:
                    zombies[(proc.pid, proc.create_time())] = dict(
                        pid=proc.pid,
                        create_time=proc.create_time(),
                        status="zombie",
                        stopped=True,
                    )
                else:
                    result.append(proc)
            except psutil.NoSuchProcess:
                continue
        return result

    members = live(members)
    for proc in members:
        try:
            proc.terminate()  # psutil checks cached identity against PID reuse.
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(members, timeout=60)
    alive = live(alive)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=30)
    if live(alive):
        raise RuntimeError("Failed lifecycle processes did not terminate")
    for proc in psutil.process_iter():
        try:
            if os.getpgid(proc.pid) in groups:
                if (
                    proc.pid,
                    proc.create_time(),
                ) in zombies and proc.status() == psutil.STATUS_ZOMBIE:
                    continue
                # A new descendant or reused group is not safe to signal; never
                # continue the matrix with unverified group cleanup.
                raise RuntimeError("Lifecycle process group cleanup is unverified")
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
    return dict(confirmed_stopped=True, zombies=list(zombies.values()))


@contextlib.contextmanager
def process(cmd, log_path, env):
    write_json(log_path.with_suffix(".command.json"), cmd)
    with log_path.open("w") as log:
        child = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            yield child
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=30)


def wait_server(child):
    for _ in range(1800):
        if child.poll() is not None:
            raise RuntimeError("Server exited before readiness")
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:30000/health", timeout=2
            ) as reply:
                if reply.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError("Server readiness timed out")


def wait_encoder(child, port):
    import zmq

    context = zmq.Context()
    try:
        for _ in range(600):
            if child.poll() is not None:
                raise RuntimeError("Encoder worker exited before readiness")
            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 1000)
            socket.connect(f"tcp://127.0.0.1:{port}")
            try:
                socket.send_json(dict(op="ping"))
                if socket.recv_json().get("ok"):
                    return
            except zmq.Again:
                pass
            finally:
                socket.close()
        raise TimeoutError("Encoder readiness timed out")
    finally:
        context.term()


def overlay(
    base,
    out,
    remote,
    cudnn_benchmark_limit=None,
    *,
    chunk_plan=None,
    async_output=False,
    attention_window=None,
):
    if cudnn_benchmark_limit is not None and (
        type(cudnn_benchmark_limit) is not int or cudnn_benchmark_limit != 1
    ):
        raise ValueError("cudnn_benchmark_limit must be 1 or null")
    if type(async_output) is not bool:
        raise ValueError("async_output must be boolean")
    if attention_window is not None and not (
        isinstance(attention_window, dict)
        and set(attention_window) == {"window_chunks", "sink_chunks"}
        and type(attention_window["window_chunks"]) is int
        and type(attention_window["sink_chunks"]) is int
        and 1 <= attention_window["window_chunks"] <= 21
        and 0 <= attention_window["sink_chunks"] < attention_window["window_chunks"]
    ):
        raise ValueError("Invalid source_attention_window deployment experiment")
    if chunk_plan is not None and chunk_plan != {
        "num_frame_per_block": [3, 2],
        "split_first_latent": True,
    }:
        raise ValueError(
            "This pipeline experiment requires [3,2] with split_first_latent=true"
        )
    model = out / "model"
    model.mkdir()
    for p in base.iterdir():
        if p.name not in ("zing_realtime_profile.json", "zing_config.json"):
            (model / p.name).symlink_to(p.resolve())
    runtime = json.loads((base / "zing_config.json").read_text())
    original_chunk_plan = {
        key: runtime[key]
        for key in ("num_frame_per_block", "split_first_latent")
        if key in runtime
    }
    if chunk_plan is not None:
        runtime.update(chunk_plan)
    original_limit = runtime["vae_config"].get("cudnn_benchmark_limit")
    if cudnn_benchmark_limit is not None:
        runtime["vae_config"]["cudnn_benchmark_limit"] = cudnn_benchmark_limit
    write_json(model / "zing_config.json", runtime)
    write_json(
        out / "encoder-policy.json",
        dict(
            original_chunk_plan=original_chunk_plan,
            effective_chunk_plan={key: runtime[key] for key in original_chunk_plan},
            original_model_config_sha256=digest(base / "zing_config.json"),
            effective_model_config_sha256=digest(model / "zing_config.json"),
            original_cudnn_benchmark_limit=original_limit,
            requested_cudnn_benchmark_limit=cudnn_benchmark_limit,
            effective_cudnn_benchmark_limit=runtime["vae_config"].get(
                "cudnn_benchmark_limit"
            ),
            baseline="Shared/reference and dedicated Encoder use this same frozen policy; no comparison to old-policy output or timing is implied",
        ),
    )
    profile = json.loads((base / "zing_realtime_profile.json").read_text())
    profile["model_config_sha256"] = digest(model / "zing_config.json")
    assert profile["source_encoder"]["mode"] == "progressive"
    profile["source_encoder"].pop("remote", None)
    if remote:
        profile["source_encoder"]["remote"] = remote
    profile["async_source_output"] = async_output
    if attention_window is not None:
        profile["source_attention_window"] = dict(attention_window)
    profile["measure_source_cuda"] = False
    write_json(model / "zing_realtime_profile.json", profile)
    return model


def server_command(model, config, sp, *, profile_monitoring=False):
    entry = (
        [
            str(BENCH / "encoder_overlap_profile_server.py"),
            "--realtime-session-max-lifetime-s",
            "3600",
            "--realtime-session-idle-timeout-s",
            "3600",
        ]
        if profile_monitoring
        else ["-m", "sglang.multimodal_gen.runtime.launch_server"]
    )
    return [
        sys.executable,
        *entry,
        "--model-path",
        str(model),
        "--pipeline-config-path",
        str(config),
        "--num-gpus",
        str(sp),
        "--sp-degree",
        str(sp),
        "--ulysses-degree",
        str(sp),
        "--ring-degree",
        "1",
        "--encoder-parallel",
        "replicate",
        "--attention-backend",
        "torch_sdpa",
        "--performance-mode",
        "speed",
        "--warmup-mode",
        "off",
        "--dit-cpu-offload",
        "false",
        "--dit-layerwise-offload",
        "false",
        "--text-encoder-cpu-offload",
        "false",
        "--vae-cpu-offload",
        "false",
        "--use-fsdp-inference",
        "false",
        "--realtime-max-sessions",
        "1",
        "--realtime-max-sessions-per-worker",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
    ]


def client_command(config, out, *, paced, profile=False, soak=False):
    width, height = input_geometry(config)
    pattern = config.get("source_batch_pattern", [1, 3, 17, 2, 5])
    if (
        not isinstance(pattern, list)
        or not pattern
        or any(type(n) is not int or n < 1 for n in pattern)
    ):
        raise ValueError(
            "source_batch_pattern must be a nonempty list of positive integers"
        )
    request = json.loads(Path(config["request"]).read_text())
    content = request["messages"][0]["content"]
    prompt = "\n".join(item["text"] for item in content if item["type"] == "text")
    references = [item for item in content if item["type"] == "image"]
    if len(references) != 6:
        raise ValueError("The default case requires all six reference images")
    if [r["id"] for r in references] != [f"<Picture {i}>" for i in range(1, 7)]:
        raise ValueError("Reference image order differs from the frozen case")
    if (
        not Path(config["prompt_video_frame"]).is_file()
        or config["prompt_video_id"] != "<Video 1>"
    ):
        raise ValueError("The default case requires its original source prompt frame")
    cmd = [
        sys.executable,
        str(BENCH / "continuous_source_client.py"),
        "--source-rgb",
        config["soak_rgb" if soak else "source_rgb"],
        "--output",
        str(out),
        "--prompt",
        prompt,
        "--width",
        str(width),
        "--height",
        str(height),
        "--batch-pattern",
        *map(str, pattern),
        "--timeout",
        "3600",
    ]
    for ref in references:
        path = unquote(urlparse(ref["uri"]).path)
        if not Path(path).is_file():
            raise ValueError("A reference asset is unavailable on the worker")
        cmd += ["--reference", path, "--reference-id", ref["id"]]
    # main derives the prompt source preview from the first actual RGB packet.
    # The prepared snapshot remains an asset check, never a legacy wire field.
    if paced:
        cmd += ["--paced"]
    if profile:
        cmd += ["--profile"]
    return cmd


def endpoint_phase_groups(*, capture=False, soak=False):
    if capture:
        return [(False, ["capture"])]
    if soak:
        return [(False, ["warmup", "measured-0"])]
    return [
        (False, ["warmup"] + [f"measured-{i}" for i in range(5)]),
        (True, ["profile-warmup", "profile"]),
    ]


def endpoint_cell(
    root,
    config,
    spec,
    paced,
    *,
    capture=False,
    soak=False,
    defer_profile=False,
    collection_phase="all",
):
    if collection_phase not in ("all", "formal", "profile"):
        raise ValueError("Invalid endpoint collection phase")
    if collection_phase != "all" and (capture or soak or defer_profile):
        raise ValueError("Split collection requires an independent performance cell")
    if defer_profile and (capture or soak):
        raise ValueError("Only performance cells can defer profiling")
    name, sp, enc, mode = spec
    label = (
        name
        + ("-paced" if paced else "-unpaced")
        + ("-capture" if capture else "-soak" if soak else "")
    )
    out = root / label
    out.mkdir()
    env = role_environment(config, "server", ",".join(map(str, range(sp))))
    env.pop("SGLANG_ZING_DIAGNOSTICS_DIR", None)
    env.pop("SGLANG_ZING_COMPONENT_CAPTURE_DIR", None)
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(map(str, range(sp))),
        SGLANG_DIFFUSION_PROFILE_ALL_RANKS="1",
        SGLANG_DIFFUSION_STAGE_LOGGING="1",
        SGLANG_DIFFUSION_COLLECT_RANK_MEMORY="1",
        SGLANG_DIFFUSION_TORCH_PROFILER_DIR=str(out / "traces"),
    )
    if capture:
        env["SGLANG_ZING_COMPONENT_CAPTURE_DIR"] = str(out / "capture")
    for key in ("cross_dispatch_prefetch", "async_source_output"):
        if type(config.get(key, False)) is not bool:
            raise ValueError(f"{key} must be boolean")
    remote = (
        dict(
            port=31001,
            mode=mode,
            cross_dispatch_prefetch=config.get("cross_dispatch_prefetch", False)
            and mode == "overlap",
            capacity_bytes=256 << 20,
            profile_dir=str(out / "encoder-traces"),
        )
        if enc
        else None
    )
    model = overlay(
        Path(config["model"]),
        out,
        remote,
        config.get("cudnn_benchmark_limit"),
        chunk_plan=config.get("source_chunk_plan"),
        async_output=config.get("async_source_output", False) and mode == "overlap",
        attention_window=config.get("source_attention_window"),
    )
    decoder_config = out / "decoder.json"
    write_json(
        decoder_config,
        dict(
            video_decoder_backend="taeh3",
            taeh3_checkpoint_path=config["taeh3_weights"],
            taeh3_decode_precision="fp16",
        ),
    )
    roles = {"server": env}
    if enc:
        encoder_env = role_environment(
            config, "encoder", ",".join(map(str, range(sp, sp + enc))), env
        )
        roles["encoder"] = encoder_env
    record_role_environment(out / "role-environment.json", **roles)

    def run_groups(groups):
        for profile_monitoring, phases in groups:
            lifecycle = None
            server_log = out / (
                "profile-server.log" if profile_monitoring else "server.log"
            )
            with contextlib.ExitStack() as stack:
                if enc:
                    worker = stack.enter_context(
                        process(
                            [
                                sys.executable,
                                "-m",
                                "torch.distributed.run",
                                "--standalone",
                                "--nproc-per-node",
                                str(enc),
                                "-m",
                                WORKER,
                                "--model",
                                str(model),
                                "--port",
                                "31001",
                            ],
                            out
                            / (
                                "profile-encoder.log"
                                if profile_monitoring
                                else "encoder.log"
                            ),
                            encoder_env,
                        )
                    )
                    wait_encoder(worker, 31001)
                write_json(
                    out
                    / (
                        "profile-monitoring-policy.json"
                        if profile_monitoring
                        else "monitoring-policy.json"
                    ),
                    dict(
                        profile_monitoring=profile_monitoring,
                        scope=(
                            "Separate profile process and its untimed warmup"
                            if profile_monitoring
                            else "Production monitoring defaults"
                        ),
                        session_max_lifetime_s=3600 if profile_monitoring else 600,
                        session_idle_timeout_s=3600 if profile_monitoring else 60,
                        server_ping_timeout_s=3600 if profile_monitoring else 20,
                        profile_client_ping_timeout_s=3600,
                        client_total_timeout_s=3600,
                    ),
                )
                server = stack.enter_context(
                    process(
                        server_command(
                            model,
                            decoder_config,
                            sp,
                            profile_monitoring=profile_monitoring,
                        ),
                        server_log,
                        env,
                    )
                )
                wait_server(server)
                for phase in phases:
                    command_run(
                        client_command(
                            config,
                            out / phase,
                            paced=paced,
                            profile=phase == "profile",
                            soak=soak,
                        ),
                        out / f"{phase}.log",
                        env,
                    )
                    summary = json.loads((out / phase / "summary.json").read_text())
                    frames = 2880 if soak else 720
                    if not summary["ended"] or any(
                        summary[k] != frames
                        for k in ("submitted", "accepted", "consumed", "received")
                    ):
                        raise RuntimeError("Incomplete endpoint output")
                    # Cleanup acknowledgement prevents the next session racing the old encoder.
                    events = [
                        json.loads(line)
                        for line in (out / phase / "events.jsonl")
                        .read_text()
                        .splitlines()
                    ]
                    sid = next(
                        e["message"]["session_id"]
                        for e in events
                        if e["kind"] == "ready"
                    )
                    for _ in range(600):
                        if any(
                            "server.session_cleanup_done" in line and sid in line
                            for line in (server_log).read_text().splitlines()
                        ):
                            break
                        if server.poll() is not None:
                            raise RuntimeError("Server exited during cleanup")
                        time.sleep(0.1)
                    else:
                        raise TimeoutError("Session cleanup acknowledgement is missing")
                    write_json(out / "progress.json", dict(phase=phase, completed=True))
                if (
                    enc
                    and not paced
                    and not capture
                    and not soak
                    and not profile_monitoring
                ):
                    owned = snapshot_process_groups([server, worker])
                    lifecycle = lifecycle_gate(
                        [
                            sys.executable,
                            str(BENCH / "continuous_source_lifecycle_probe.py"),
                            "--width",
                            str(input_geometry(config)[0]),
                            "--height",
                            str(input_geometry(config)[1]),
                            "--rgb",
                            config["source_rgb"],
                            "--case-request",
                            config["request"],
                            "--server-log",
                            str(server_log),
                            "--output",
                            str(out / "lifecycle"),
                        ],
                        out,
                        env,
                        exploratory=bool(config.get("exploratory")),
                    )
                    if not lifecycle["passed"]:
                        # Never reuse a process that participated in a failed
                        # cancellation probe, including a dead launcher's children.
                        lifecycle["cleanup"] = stop_failed_process_groups(owned)
                        write_json(out / "lifecycle-gate.json", lifecycle)
            if not capture and not soak:
                write_json(
                    out
                    / (
                        "PROFILE_COMPLETE.json"
                        if profile_monitoring
                        else "FORMAL_COMPLETE.json"
                    ),
                    dict(
                        phases=phases,
                        completed=True,
                        measurement_completed=not profile_monitoring,
                        lifecycle=lifecycle,
                        collection_status=(
                            "measurements_complete_lifecycle_failed"
                            if lifecycle is not None and not lifecycle["passed"]
                            else "profile_complete"
                            if profile_monitoring
                            else "measurements_complete"
                        ),
                    ),
                )

    def complete():
        lifecycle = (
            json.loads((out / "lifecycle-gate.json").read_text())
            if (out / "lifecycle-gate.json").is_file()
            else None
        )
        if not capture and not soak:
            if not all(
                (out / name).is_file()
                for name in ("FORMAL_COMPLETE.json", "PROFILE_COMPLETE.json")
            ):
                raise RuntimeError(
                    "Endpoint formal and profile collection must both finish"
                )
        write_json(
            out / "COMPLETE.json",
            dict(
                sp=sp,
                encoder_gpus=enc,
                mode=mode,
                paced=paced,
                capture=capture,
                soak=soak,
                exploratory=bool(config.get("exploratory")),
                merge_eligible=False,
                performance_validated=False,
                lifecycle=lifecycle,
                collection_status=(
                    "collection_complete_lifecycle_failed"
                    if lifecycle is not None and not lifecycle["passed"]
                    else "collection_complete"
                ),
            ),
        )
        return out

    groups = endpoint_phase_groups(capture=capture, soak=soak)
    if collection_phase != "all":
        run_groups(groups[:1] if collection_phase == "formal" else groups[1:])
        return out  # Only phase receipt; never claim both collections finished.
    if defer_profile:
        run_groups(groups[:1])

        # This continuation owns the freshly created cell. It is intentionally
        # in-memory only: no automatic resume or reuse of historical directories.
        def profile_after_formals():
            if (out / "PROFILE_COMPLETE.json").exists():
                raise RuntimeError("Endpoint profile collection already finished")
            run_groups(groups[1:])
            return complete()

        return profile_after_formals
    run_groups(groups)
    return complete()


def endpoint_matrix(root, config, specs, *, collection_phase="all"):
    if collection_phase != "all":
        for spec in specs:
            for paced in (False, True):
                endpoint_cell(
                    root, config, spec, paced, collection_phase=collection_phase
                )
        return
    profiles = []
    for spec in specs:
        for paced in (False, True):
            profiles.append(
                endpoint_cell(root, config, spec, paced, defer_profile=True)
            )
    for profile in profiles:
        profile()


def endpoint_binding(config):
    """Exact frozen configuration and source identity for split collections."""
    source = BENCH.parents[1]
    sha = hashlib.sha256()
    paths = sorted((source / "python" / "sglang").rglob("*.py")) + sorted(
        BENCH.glob("*.py")
    )
    for path in paths:
        sha.update(str(path.relative_to(source)).encode())
        sha.update(bytes.fromhex(digest(path)))
    return dict(
        config=config,
        source_sha256=sha.hexdigest(),
        source_files=len(paths),
        request_sha256=digest(Path(config["request"])),
        source_rgb_sha256=digest(Path(config["source_rgb"])),
        model_config_sha256=digest(Path(config["model"]) / "zing_config.json"),
        realtime_profile_sha256=digest(
            Path(config["model"]) / "zing_realtime_profile.json"
        ),
    )


def validate_profile_binding(root, config, specs, binding):
    path = root / "endpoint-binding.json"
    if not path.is_file() or json.loads(path.read_text()) != binding:
        raise ValueError("Profile inputs or source differ from the formal collection")
    for spec in specs:
        for pace in ("unpaced", "paced"):
            cell = root / f"{spec[0]}-{pace}"
            receipt = json.loads((cell / "FORMAL_COMPLETE.json").read_text())
            expected = ["warmup"] + [f"measured-{i}" for i in range(5)]
            if receipt.get("phases") != expected or not receipt.get(
                "measurement_completed"
            ):
                raise ValueError("Profile requires all five formal runs and warmup")
            for phase in expected:
                row = json.loads((cell / phase / "summary.json").read_text())
                if row.get("ended") is not True or any(
                    row.get(k) != 720
                    for k in ("submitted", "accepted", "consumed", "received")
                ):
                    raise ValueError("Profile cannot bind incomplete formal output")


def validate_serial_overlap(root, spec, config=None, *, require_parity=True):
    """Require five complete paired runs before scheduling a separate soak."""
    name, sp, enc, mode = spec
    if not enc or mode != "overlap":
        raise ValueError(
            "Soak requires an explicitly selected dedicated overlap layout"
        )
    if config is not None:
        input_path = root / "input.json"
        if not input_path.is_file():
            raise RuntimeError("Paired performance input receipt is missing")
        previous = json.loads(input_path.read_text())
        # Paths refer to the immutable prepared assets/model overlay, never a rebuilt request.
        if any(
            previous.get(key) != config.get(key)
            for key in (
                "model",
                "request",
                "source_rgb",
                "soak_rgb",
                "taeh3_weights",
                "cudnn_benchmark_limit",
                "encoder_tile_mode",
                "progressive_decode",
                "source_batch_pattern",
                "source_chunk_plan",
                "source_attention_window",
                "cross_dispatch_prefetch",
                "async_source_output",
                "width",
                "height",
            )
        ):
            raise RuntimeError("Soak inputs differ from the paired performance receipt")
    rows = []
    for execution in ("serial", "overlap"):
        directory = root / f"enc{enc}-sp{sp}-{execution}-paced"
        for i in range(5):
            path = directory / f"measured-{i}" / "summary.json"
            if not path.is_file():
                raise RuntimeError(
                    "Five paired performance runs are required before soak"
                )
            row = json.loads(path.read_text())
            if not row.get("ended") or any(
                row.get(key) != 720
                for key in ("submitted", "accepted", "consumed", "received")
            ):
                raise RuntimeError("Incomplete paired performance output")
            rows.append(row)
    if any(not row.get("output_sha256") for row in rows):
        raise RuntimeError("Paired performance output hash evidence is missing")
    if require_parity:
        validate_serial_overlap_parity(rows)
    return rows


def validate_serial_overlap_parity(rows):
    """Only numerical equality is an optional exploratory gate."""
    if len({row["output_sha256"] for row in rows}) != 1:
        raise RuntimeError("Serial/overlap output parity failed before soak")


def validate_correctness(root, specs, config):
    receipt = root / "input.json"
    if not receipt.is_file():
        raise RuntimeError("Correctness input receipt is missing")
    previous = json.loads(receipt.read_text())
    if any(
        previous.get(key) != config.get(key)
        for key in (
            "model",
            "request",
            "source_rgb",
            "taeh3_weights",
            "cudnn_benchmark_limit",
            "encoder_tile_mode",
            "progressive_decode",
            "source_batch_pattern",
            "source_chunk_plan",
            "source_attention_window",
            "cross_dispatch_prefetch",
            "async_source_output",
            "width",
            "height",
        )
    ):
        raise ValueError("Correctness inputs differ from the performance configuration")
    for name, _, enc, _ in specs:
        if not enc:
            continue
        path = root / f"correctness-{name}.json"
        if not path.is_file():
            raise RuntimeError(
                "Shared-versus-dedicated correctness evidence is missing"
            )
        row = json.loads(path.read_text())
        if (
            row.get("status") != "passed"
            or row.get("tolerance") != dict(atol=0, rtol=0)
            or not isinstance(row.get("wire_rgb"), dict)
            or row["wire_rgb"].get("bytes")
            != 720 * input_geometry(config)[0] * input_geometry(config)[1] * 3
        ):
            raise RuntimeError("Shared-versus-dedicated strict correctness gate failed")


def capture_session(cell):
    paths = list((cell / "capture").glob("*/COMPLETE.json"))
    if len(paths) != 1:
        raise RuntimeError("Expected one complete capture session")
    return paths[0].parent


def correctness_cells(root, config, specs):
    commands = []
    baselines = {}
    for spec in specs:
        name, sp, enc, _ = spec
        if not enc:
            continue
        if sp not in baselines:
            baselines[sp] = endpoint_cell(
                root, config, (f"shared-sp{sp}", sp, 0, "serial"), False, capture=True
            )
        baseline = baselines[sp]
        candidate = endpoint_cell(root, config, spec, False, capture=True)
        commands.append(
            [
                sys.executable,
                str(BENCH / "compare_encoder_overlap_capture.py"),
                "--reference",
                str(capture_session(baseline)),
                "--candidate",
                str(capture_session(candidate)),
                "--reference-rgb",
                str(baseline / "capture" / "output.rgb"),
                "--candidate-rgb",
                str(candidate / "capture" / "output.rgb"),
                "--output",
                str(root / f"correctness-{name}.json"),
            ]
        )
    if not commands:
        raise ValueError("Correctness capture needs at least one dedicated layout")
    write_json(
        root / "capture-comparisons.json",
        dict(
            commands=commands,
            execution="Run these commands in the zero-GPU CPU postprocessing job; timing stays blocked until every strict comparison passes",
        ),
    )


def transport_cell(root, config, enc):
    cell = root / f"transport-enc{enc}"
    cell.mkdir()
    model = overlay(
        Path(config["model"]), cell, None, config.get("cudnn_benchmark_limit")
    )
    encoder_env = role_environment(
        config, "encoder", ",".join(map(str, range(1, enc + 1)))
    )
    consumer_env = role_environment(config, "consumer", "0")
    record_role_environment(
        cell / "role-environment.json", encoder=encoder_env, consumer=consumer_env
    )
    with process(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            str(enc),
            "-m",
            WORKER,
            "--model",
            str(model),
            "--port",
            "31001",
        ],
        cell / "encoder.log",
        encoder_env,
    ) as worker:
        wait_encoder(worker, 31001)
        command_run(
            [
                sys.executable,
                str(BENCH / "probe_encoder_transport_gpu.py"),
                "--width",
                str(input_geometry(config)[0]),
                "--height",
                str(input_geometry(config)[1]),
                *(["--exploratory"] if config.get("exploratory") else []),
                "--model",
                str(model),
                "--source-rgb",
                config["source_rgb"],
                "--port",
                "31001",
                "--output",
                str(cell / "results"),
            ],
            cell / "probe.log",
            consumer_env,
        )


def required_gpus(stage, specs, dit_sizes, encoder_sizes):
    requirements = [1]
    if stage == "transport":
        requirements.extend(enc + 1 for enc in encoder_sizes)
    if stage in ("dit", "components", "all"):
        requirements.extend(dit_sizes)
    if stage in ("encoder", "components", "all"):
        requirements.extend(encoder_sizes)
    if stage in ("correctness", "overlap", "soak", "all"):
        requirements.extend(sp + enc for _, sp, enc, _ in specs)
    return max(requirements)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--stage",
        choices=(
            "capture",
            "correctness",
            "transport",
            "encoder",
            "dit",
            "decoder",
            "components",
            "overlap",
            "soak",
            "all",
        ),
        default="overlap",
    )
    p.add_argument(
        "--layout",
        action="append",
        choices=[row[0] for row in layouts()],
        help="Repeat to select endpoint layouts; alternatively set layouts in input JSON",
    )
    p.add_argument(
        "--dit-sp", type=int, nargs="+", choices=(1, 2, 4, 8), default=[1, 2, 4, 8]
    )
    p.add_argument(
        "--encoder-gpus", type=int, nargs="+", choices=(1, 2, 4), default=[2]
    )
    p.add_argument(
        "--capture-root",
        type=Path,
        help="Existing completed component capture directory",
    )
    p.add_argument(
        "--performance-root",
        type=Path,
        help="Completed paired performance run required for --stage soak",
    )
    p.add_argument(
        "--exploratory",
        action="store_true",
        help="Collect despite failed numerical acceptance; never marks performance validated or merge eligible",
    )
    p.add_argument(
        "--collection-phase", choices=("all", "formal", "profile"), default="all"
    )
    args = p.parse_args()
    if args.collection_phase != "all" and args.stage != "overlap":
        p.error("Split collection is supported only by --stage overlap")
    if args.collection_phase == "profile" and args.performance_root is None:
        p.error("Profile collection requires --performance-root")
    config = json.loads(args.input.read_text())
    config.setdefault("encoder_tile_mode", "parallel")
    config.setdefault("cudnn_benchmark_limit", 1)
    specs = selected_layouts(args.layout or config.get("layouts", []))
    if args.stage in ("correctness", "overlap", "soak", "all") and not specs:
        p.error("Endpoint stages require explicit --layout or input layouts")
    if args.stage == "soak" and (
        len(specs) != 1 or specs[0][3] != "overlap" or args.performance_root is None
    ):
        p.error("Soak requires one explicit overlap --layout and --performance-root")
    width, height = input_geometry(config)
    limit = config.get("cudnn_benchmark_limit")
    if limit is not None and (type(limit) is not int or limit != 1):
        raise ValueError("cudnn_benchmark_limit must be 1 or null")
    role_environment(config, "server", "")
    config["exploratory"] = args.exploratory
    binding = endpoint_binding(config) if args.stage == "overlap" else None
    if args.collection_phase == "profile":
        validate_profile_binding(args.performance_root, config, specs, binding)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    hardware = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import torch,json; print(json.dumps(dict(devices=[dict(index=i,name=torch.cuda.get_device_name(i),"
                "cc=torch.cuda.get_device_capability(i),memory_bytes=torch.cuda.get_device_properties(i).total_memory) "
                "for i in range(torch.cuda.device_count())],torch=torch.__version__,cuda=torch.version.cuda)))",
            ],
            text=True,
        )
    )
    required = required_gpus(args.stage, specs, args.dit_sp, args.encoder_gpus)
    if not required <= len(hardware["devices"]) <= 8:
        raise RuntimeError(
            f"Selected stage needs at least {required} and at most eight visible GPUs"
        )
    write_json(root / "hardware.json", hardware)
    write_json(root / "input.json", config)
    if binding is not None:
        write_json(root / "endpoint-binding.json", binding)
    write_json(
        root / "matrix-plan.json",
        dict(
            stage=args.stage,
            collection_phase=args.collection_phase,
            performance_root=str(args.performance_root)
            if args.performance_root
            else None,
            exploratory=args.exploratory,
            merge_eligible=False,
            performance_validated=False,
            layouts=specs,
            dit_sp=args.dit_sp,
            encoder_gpus=args.encoder_gpus,
            visible_gpus=len(hardware["devices"]),
            required_gpus=required,
        ),
    )
    with (root / "topology.txt").open("w") as f:
        subprocess.run(["nvidia-smi", "topo", "-m"], stdout=f, check=True)
    for key, count in (("source_rgb", 720), ("soak_rgb", 2880)):
        assert Path(config[key]).stat().st_size == count * width * height * 3
    if args.stage == "transport":
        for enc in args.encoder_gpus:
            transport_cell(root, config, enc)
    if args.stage in ("encoder", "components", "all"):
        for enc in args.encoder_gpus:
            env = role_environment(config, "encoder", ",".join(map(str, range(enc))))
            record_role_environment(
                root / f"encoder-{enc}.role-environment.json", encoder=env
            )
            command_run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc-per-node",
                    str(enc),
                    str(BENCH / "benchmark_parallel_encoder.py"),
                    "--width",
                    str(width),
                    "--height",
                    str(height),
                    "--tile-mode",
                    config.get("encoder_tile_mode", "parallel"),
                    *(
                        [
                            "--cudnn-benchmark-limit",
                            str(config["cudnn_benchmark_limit"]),
                        ]
                        if config.get("cudnn_benchmark_limit") is not None
                        else []
                    ),
                    "--model",
                    config["model"],
                    "--source-rgb",
                    config["source_rgb"],
                    "--output",
                    str(root / f"encoder-{enc}"),
                ],
                root / f"encoder-{enc}.log",
                env,
            )
    if args.stage in ("capture", "dit", "decoder", "components", "all"):
        capture = args.capture_root
        if capture is None:
            capture_root = endpoint_cell(
                root, config, ("shared-sp1", 1, 0, "serial"), False, capture=True
            )
            captures = list((capture_root / "capture").glob("*/COMPLETE.json"))
            if len(captures) != 1:
                raise RuntimeError("Expected exactly one complete component capture")
            capture = captures[0].parent
        if not (capture / "COMPLETE.json").is_file():
            raise RuntimeError("Component capture is incomplete")
        if args.stage in ("dit", "components", "all"):
            for sp in args.dit_sp:
                env = dict(
                    os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, range(sp)))
                )
                command_run(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc-per-node",
                        str(sp),
                        str(BENCH / "benchmark_dit_replay.py"),
                        "--model",
                        config["model"],
                        "--capture",
                        str(capture),
                        "--output",
                        str(root / f"dit-sp{sp}"),
                    ],
                    root / f"dit-sp{sp}.log",
                    env,
                )
        if args.stage in ("decoder", "components", "all"):
            command_run(
                [
                    sys.executable,
                    str(BENCH / "benchmark_taeh3_replay.py"),
                    "--capture",
                    str(capture),
                    "--weights",
                    config["taeh3_weights"],
                    "--output",
                    str(root / "decoder"),
                    *(
                        ["--progressive"]
                        if config.get("progressive_decode", False)
                        else []
                    ),
                ],
                root / "decoder.log",
                role_environment(config, "server", "0"),
            )
    if args.stage in ("correctness", "overlap", "soak", "all"):
        gates = Path(config.get("encoder_acceptance_root", root))
        sizes = sorted({enc for _, _, enc, _ in specs if enc})
        gate_records = {}
        gate_path = root / "gate-results.json"
        if args.exploratory:
            # Runtime/source/policy/group/gather/tile proofs remain mandatory;
            # only completeness and exact numerical parity may be exploratory.
            validate_encoder_acceptance(
                gates,
                digest(Path(config["source_rgb"])),
                sizes,
                config.get("cudnn_benchmark_limit"),
                require_numerical=False,
                width=width,
                height=height,
            )
        prerequisite_gate(
            "encoder_numerical",
            lambda: validate_encoder_acceptance(
                gates,
                digest(Path(config["source_rgb"])),
                sizes,
                config.get("cudnn_benchmark_limit"),
                width=width,
                height=height,
            ),
            exploratory=args.exploratory,
            records=gate_records,
            path=gate_path,
        )
        if args.stage == "correctness":
            correctness_cells(root, config, specs)
        else:
            prerequisite_gate(
                "capture_correctness",
                lambda: validate_correctness(
                    Path(config.get("correctness_root", root)), specs, config
                ),
                exploratory=args.exploratory,
                records=gate_records,
                path=gate_path,
            )
        if args.stage == "soak":
            paired_rows = validate_serial_overlap(
                args.performance_root, specs[0], config, require_parity=False
            )
            prerequisite_gate(
                "serial_overlap_rgb_parity",
                lambda: validate_serial_overlap_parity(paired_rows),
                exploratory=args.exploratory,
                records=gate_records,
                path=gate_path,
            )
            write_json(
                root / "provisional-selection.json",
                dict(
                    layout=specs[0],
                    performance_root=str(args.performance_root),
                    acceptance="pending growth and media review",
                ),
            )
            endpoint_cell(root, config, specs[0], True, soak=True)
        elif args.stage != "correctness":
            endpoint_matrix(root, config, specs, collection_phase=args.collection_phase)
    write_json(
        root / "COLLECTION_COMPLETE.json",
        dict(
            stage=args.stage,
            collection_phase=args.collection_phase,
            performance_root=str(args.performance_root)
            if args.performance_root
            else None,
            lifecycle_failed_receipts=[
                str(path)
                for path in sorted(root.glob("*/lifecycle-gate.json"))
                if json.loads(path.read_text()).get("passed") is False
            ],
            exploratory=args.exploratory,
            merge_eligible=False,
            performance_validated=False,
            acceptance=(
                "Exploratory collection; failed gates remain failed"
                if args.exploratory
                else "Collection only; numerical, stability, trace, lifecycle and media gates require separate review"
            ),
        ),
    )


if __name__ == "__main__":
    main()
