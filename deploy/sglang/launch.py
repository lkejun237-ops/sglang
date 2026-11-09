"""Prepare, render and launch existing SGLang workers without application bindings."""

import argparse
import base64
import hashlib
import ipaddress
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

from source import digest, prepare, sha, validate_manifest, write_json

MODEL_ID = "zing-h3-dmd-1102-step2000-generator"
PREFIX = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3."
)
MODULES = {
    "encoder": "sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server",
    "dit": PREFIX + "dit_worker",
    "decoder": PREFIX + "decoder_worker",
}
PORTS = {"encoder": 31000, "dit": 33000, "decoder": 36000}
# No inherited PYTHONPATH, PYTHONHOME, LD_PRELOAD, credential or arbitrary runtime overrides.
ENV_ALLOW = {
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "NCCL_IB_HCA",
    "NCCL_IB_DISABLE",
    "NCCL_DEBUG",
    "TORCH_EXTENSIONS_DIR",
    "HF_HOME",
    "OMP_NUM_THREADS",
    "MINIMAX_H3_PROGRESSIVE_FUSED_NORM_INPUT",
    "MINIMAX_H3_PROGRESSIVE_FUSED_CONV_INPUT",
    "SGLANG_LYNN_PACKED_QKV",
    "SGLANG_LYNN_FUSED_QK_NORM",
    "SGLANG_LYNN_COMPILED_FFN",
}
CONFIG_KEYS = {
    "schema",
    "role",
    "model_id",
    "model",
    "asset_receipt",
    "asset_receipt_sha256",
    "pipeline",
    "decoder_config",
    "quantization",
    "environment",
    "startup_timeout",
    "hardware",
}


def load_config(path):
    c = json.loads(Path(path).read_text())
    if (
        c.get("schema") != 1
        or set(c) - CONFIG_KEYS
        or c.get("role") not in MODULES
        or c.get("model_id") != MODEL_ID
    ):
        raise ValueError("Unknown configuration field, role or model identity")
    if set(c.get("environment", {})) - (
        ENV_ALLOW
        - {
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "HOME",
            "PATH",
            "LD_LIBRARY_PATH",
        }
    ):
        raise ValueError("Environment key is not allowlisted")
    if any(
        not isinstance(v, str) or "\x00" in v for v in c.get("environment", {}).values()
    ):
        raise ValueError("Environment values must be strings without NUL")
    if c.get("quantization") not in (None, "fp8"):
        raise ValueError("Unsupported quantization")
    if c["role"] != "dit" and c.get("quantization"):
        raise ValueError("Quantization is a DiT option")
    if c["role"] == "dit" and not c.get("pipeline"):
        raise ValueError("DiT requires an explicitly verified pipeline")
    if c["role"] == "decoder" and not c.get("decoder_config"):
        raise ValueError("Decoder requires its own config")
    if c["role"] == "decoder" and c.get("pipeline"):
        raise ValueError("Decoder does not accept a pipeline")
    for key in ("model", "asset_receipt"):
        if not Path(c[key]).is_absolute():
            raise ValueError("Asset paths must be absolute")
    sha(c["asset_receipt_sha256"])
    if not 1 <= c.get("startup_timeout", 900) <= 3600:
        raise ValueError("Startup timeout must be in [1, 3600] seconds")
    return c


def verify_assets(c):
    path = Path(c["asset_receipt"])
    if digest(path) != c["asset_receipt_sha256"]:
        raise ValueError("Asset receipt digest mismatch")
    a = json.loads(path.read_text())
    if (
        a.get("schema") != 1
        or a["model_id"] != MODEL_ID
        or a["component"] != "generator"
        or a["ema"] is not False
        or Path(a["model"]).resolve() != Path(c["model"]).resolve()
    ):
        raise ValueError("Model identity or component mismatch")
    # Receipt is produced by the asset owner after verifying the original export.
    # Hash every listed artifact, including all weight shards and external components.
    files = a["files"]
    if not files or any(not Path(p).is_absolute() for p in files):
        raise ValueError("Asset receipt requires absolute artifact paths")
    required = [str(Path(c["model"]) / "zing_config.json")]
    required += [c[k] for k in ("pipeline", "decoder_config") if c.get(k)]
    if any(p not in files for p in required):
        raise ValueError("Role config is not covered by asset receipt")
    weights = [
        p
        for p in Path(c["model"]).rglob("*")
        if p.is_file() and p.suffix in {".safetensors", ".pt", ".bin"}
    ]
    if not weights or any(str(p) not in files for p in weights):
        raise ValueError("Model weight files must be covered by asset receipt")
    for p, expected in files.items():
        if digest(p) != sha(expected):
            raise ValueError("Asset digest mismatch: " + p)
    if c["role"] == "dit":
        pipeline = json.loads(Path(c["pipeline"]).read_text())
        adaln = a["adaln"]
        if adaln["mode"] != pipeline.get("zing_adaln_mode", "weights_free"):
            raise ValueError("AdaLN mode mismatch")
        if adaln["mode"] == "weights_free":
            table = pipeline.get("zing_adaln_table_path")
            if (
                not table
                or table != adaln["table"]
                or any(
                    str(Path(table) / n) not in files
                    for n in ("manifest.json", "table.safetensors")
                )
            ):
                raise ValueError("Verified matching AdaLN table is required")
            table_manifest = json.loads((Path(table) / "manifest.json").read_text())
            if table_manifest.get("schedule") != adaln.get(
                "schedule"
            ) or table_manifest.get("arithmetic") != adaln.get("cuda_provenance"):
                raise ValueError("AdaLN manifest provenance mismatch")
            if (
                not adaln.get("schedule")
                or not adaln.get("cuda_provenance")
                or adaln.get("model_id") != MODEL_ID
            ):
                raise ValueError("AdaLN model/schedule/CUDA provenance is required")
        elif adaln["mode"] != "resident" or pipeline.get("zing_adaln_table_path"):
            raise ValueError("Invalid resident AdaLN configuration")
    return a


def reachable_ip():
    value = os.environ.get("POD_IP")
    if not value:
        # Kernel route selection only; UDP connect sends no packet.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.0.0.121", 9))
            value = s.getsockname()[0]
    address = ipaddress.ip_address(value)
    if address.is_loopback or address.is_unspecified or address.is_multicast:
        raise ValueError("Platform did not provide a usable advertise IP")
    return str(address)


def command(c, source, host, worker_id):
    role = c["role"]
    argv = [sys.executable, "-m"]
    if role == "decoder":
        argv += [
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=1",
            "--module",
        ]
    argv += [
        MODULES[role],
        "--model",
        c["model"],
        "--service-count",
        "1",
        "--http-port",
        str(PORTS[role]),
        "--advertise-host",
        host,
        "--worker-id",
        worker_id,
    ]
    if c.get("pipeline"):
        argv += ["--pipeline-config-path", c["pipeline"]]
    if role == "dit":
        argv += ["--sp-degree", "1"]
        if c.get("quantization"):
            argv += ["--quantization", c["quantization"]]
    if role == "decoder":
        argv += [
            "--decoder-config",
            json.dumps(
                json.loads(Path(c["decoder_config"]).read_text()), separators=(",", ":")
            ),
        ]
    env = {k: v for k, v in os.environ.items() if k in ENV_ALLOW}
    env.update(c.get("environment", {}))
    env.update(
        PYTHONPATH=str(source / "python"),
        PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "1"),
    )
    return argv, env


def probe_source(source, env, role, hardware=None):
    # Run under the exact interpreter/env/cwd used for worker startup.
    code = """import importlib.util,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
result={}
for name in ('sglang',sys.argv[2]):
 spec=importlib.util.find_spec(name)
 path=pathlib.Path(spec.origin).resolve()
 if not path.is_relative_to(root/'python'): raise RuntimeError('Import escaped verified source: '+str(path))
 import hashlib
 result[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
import torch
if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('Exactly one allocated GPU is required')
properties=torch.cuda.get_device_properties(0)
hardware=json.loads(sys.argv[3])
if hardware.get('capability') and list(torch.cuda.get_device_capability(0))!=hardware['capability']: raise RuntimeError('GPU capability does not match the role profile')
if properties.total_memory < hardware.get('minimum_memory_gib',0)*1024**3: raise RuntimeError('GPU memory does not match the role profile')
result.update(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,capability=list(torch.cuda.get_device_capability(0)),gpu=torch.cuda.get_device_name(0),total_memory_bytes=properties.total_memory)
print(json.dumps(result))"""
    return json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                code,
                str(source),
                MODULES[role],
                json.dumps(hardware or {}),
            ],
            cwd=source,
            env=env,
            text=True,
        )
    )


def run(c, source, commit, archive_sha):
    sha(commit, 40)
    sha(archive_sha)
    assets = verify_assets(c)
    attestation = json.loads((source.parent / "verified-source.json").read_text())
    if attestation != {"commit": commit, "archive_sha256": archive_sha}:
        raise ValueError("Missing verified archive attestation")
    host = reachable_ip()
    worker_id = c["role"] + "-" + uuid.uuid4().hex
    argv, env = command(c, source, host, worker_id)
    imports = probe_source(source, env, c["role"], c.get("hardware"))
    ports = {
        "encoder": (30000, 31000, 32000),
        "dit": (33000, 34000, 35000),
        "decoder": (36000, 37000, 38000),
    }
    for port in ports[c["role"]]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("0.0.0.0", port))
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,memory.used",
            "--format=csv,noheader",
        ],
        env=env,
        text=True,
    ).strip()
    receipt = {
        "schema": 1,
        "release_id": commit + "-" + archive_sha[:12],
        "launcher_sha256": digest(Path(__file__)),
        "boot_epoch": uuid.uuid4().hex,
        "worker_boot_epoch": None,
        "image_requested": os.environ.get("SGLANG_IMAGE_DIGEST"),
        "image_actual": None,
        "role_config_sha256": hashlib.sha256(
            json.dumps(c, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "status": "starting",
        "commit": commit,
        "archive_sha256": archive_sha,
        "model_id": MODEL_ID,
        "asset_receipt_sha256": c["asset_receipt_sha256"],
        "model_path": c["model"],
        "pipeline": c.get("pipeline"),
        "decoder_config": c.get("decoder_config"),
        "adaln": assets.get("adaln"),
        "role": c["role"],
        "worker_id": worker_id,
        "ip": host,
        "http": "http://" + host + ":" + str(PORTS[c["role"]]),
        "argv": argv,
        "environment": {
            k: v for k, v in env.items() if k not in {"HOME", "PATH", "LD_LIBRARY_PATH"}
        },
        "runtime": imports,
        "gpu": gpu,
        "pod": os.environ.get("POD_NAME", socket.gethostname()),
        "pod_uid": os.environ.get("POD_UID"),
        "job": os.environ.get("PAI_JOB_ID")
        or os.environ.get("DLC_JOB_ID")
        or os.environ.get("JOB_ID"),
        "pid": None,
    }
    output = source.parent / "receipt.json"
    child = subprocess.Popen(argv, cwd=source, env=env, start_new_session=True)
    receipt["pid"] = child.pid

    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(output)
        print(json.dumps(receipt), flush=True)

    def stop(signum, frame):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    try:
        save()
        deadline = time.monotonic() + c.get("startup_timeout", 900)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError("Worker exited before becoming healthy")
            try:
                with opener.open(receipt["http"] + "/health", timeout=2) as response:
                    health = json.load(response)
                if health.get("ready") is True and health.get("role") == c["role"]:
                    receipt.update(status="ready", health=health)
                    save()
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Worker startup health deadline exceeded")
        code = child.wait()
        receipt.update(status="exited", exit_code=code)
        save()
        return code if code >= 0 else 128 - code
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["exit_reason"] = type(error).__name__
        save()
        raise
    finally:
        # The torchrun parent can exit before its worker descendants.
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def bootstrap_command(manifest, config):
    payload = base64.b64encode(
        Path(__file__).with_name("source.py").read_bytes()
    ).decode()
    code = (
        "import base64;exec(compile(base64.b64decode("
        + repr(payload)
        + "),'<source-bootstrap>','exec'))"
    )
    return [
        "/opt/sglang/bin/python",
        "-c",
        code,
        "--manifest",
        json.dumps(manifest),
        "--config",
        json.dumps(config),
    ]


def render(c, manifest, site, output):
    validate_manifest(manifest)
    if site.get("schema") != 1:
        raise ValueError("Unsupported site configuration schema")
    allowed = (
        {"schema", "image", "namespace", "volumes", "volume_mounts"}
        if c["role"] != "dit"
        else {"schema", "image", "job_spec", "data_sources"}
    )
    if set(site) - allowed:
        raise ValueError("Unknown site configuration field")
    for name in ("launch.py", "source.py", "select_gpu.py"):
        if manifest.get("tools", {}).get(name) != digest(
            Path(__file__).with_name(name)
        ):
            raise ValueError(
                "Render tools differ from the selected archive; use that commit checkout"
            )
    if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", site["image"]):
        raise ValueError("A verified image digest is required")
    c = {
        **c,
        "hardware": {
            "capability": [9, 0],
            "minimum_memory_gib": 130 if c["role"] == "dit" else 75,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    argv = bootstrap_command(manifest, c)
    name = "sglang-" + c["role"] + "-" + uuid.uuid4().hex[:12]
    if c["role"] in ("encoder", "decoder"):
        # Discovery and selection occur on the target node, never on the user's kube context.
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": name, "namespace": site["namespace"]},
            "spec": {
                "nodeName": "world-model-b27mys7b-0000",
                "runtimeClassName": "nvidia",
                "hostNetwork": True,
                "restartPolicy": "Never",
                "containers": [
                    {
                        "name": c["role"],
                        "image": site["image"],
                        "imagePullPolicy": "Never",
                        "command": argv,
                        "env": [
                            {"name": "SGLANG_IMAGE_DIGEST", "value": site["image"]},
                            {
                                "name": "NVIDIA_VISIBLE_DEVICES",
                                "value": "__ALLOCATED_GPU_UUID__",
                            },
                            {
                                "name": "NVIDIA_DRIVER_CAPABILITIES",
                                "value": "compute,utility",
                            },
                            *[
                                {"name": n, "valueFrom": {"fieldRef": {"fieldPath": p}}}
                                for n, p in [
                                    ("POD_IP", "status.podIP"),
                                    ("POD_NAME", "metadata.name"),
                                    ("POD_UID", "metadata.uid"),
                                ]
                            ],
                        ],
                        "volumeMounts": site["volume_mounts"],
                    }
                ],
                "volumes": site["volumes"],
            },
        }
        # Node local advisory lock prevents two invocations of this launcher selecting the same card.
        script = """#!/bin/sh
set -eu
exec 9>/tmp/sglang-role-allocation.lock
flock -n 9
"""
        selector = base64.b64encode(
            Path(__file__).with_name("select_gpu.py").read_bytes()
        ).decode()
        selector_code = (
            "import base64;exec(compile(base64.b64decode("
            + repr(selector)
            + "),'<gpu-selector>','exec'))"
        )
        script += "GPU_UUID=$(python3 -c " + shlex.quote(selector_code) + ")\n"
        script += "POD_JSON=" + shlex.quote(json.dumps(pod)) + "\n"
        script += """printf '%s' "$POD_JSON" | sed "s/__ALLOCATED_GPU_UUID__/$GPU_UUID/g" | kubectl create --dry-run=client -f - >/dev/null
printf '%s' "$POD_JSON" | sed "s/__ALLOCATED_GPU_UUID__/$GPU_UUID/g" | kubectl create -f -
"""
        script += (
            "kubectl -n "
            + shlex.quote(site["namespace"])
            + " get pod "
            + shlex.quote(name)
            + " -o json\n"
        )
        (output / "h100-node.sh").write_text(script)
        request = {
            "NodeIdList": ["e01-cn-2zd4xi8m505"],
            "Name": name,
            "RepeatMode": "Once",
            "Timeout": 60,
            "ContentEncoding": "PlainText",
            "CommandContent": script,
        }
        write_json(output / "eflo.json", request)
    else:
        # Explicit resource fields copied from a currently verified PAI specification.
        spec = site["job_spec"]
        if set(spec) - {
            "Type",
            "PodCount",
            "QuotaId",
            "RestartPolicy",
            "ResourceConfig",
        } or (
            spec.get("PodCount") != 1
            or spec.get("ResourceConfig", {}).get("GPU") != "1"
            or spec.get("Type") != "Worker"
            or spec.get("QuotaId") != "quota3l8bnv9a35n"
            or spec.get("RestartPolicy") != "Never"
        ):
            raise ValueError("PAI DiT requires one pod with exactly one GPU")
        spec = {**spec, "Image": site["image"]}
        job = {
            "DisplayName": name,
            "JobType": "PyTorchJob",
            "WorkspaceId": "1453237",
            "ResourceId": "quota3l8bnv9a35n",
            "JobSpecs": [spec],
            "DataSources": site["data_sources"],
            "UserCommand": "export SGLANG_IMAGE_DIGEST="
            + shlex.quote(site["image"])
            + "\nexec "
            + shlex.join(argv),
        }
        write_json(output / "pai.json", job)
    write_json(output / "role.json", c)
    write_json(output / "source.json", manifest)
    write_json(
        output / "render.json",
        {
            "name": name,
            "role": c["role"],
            "source": manifest,
            "model_id": MODEL_ID,
            "status": "rendered-not-submitted",
            "schema": 1,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--repo", type=Path, default=Path("."))
    a.add_argument("--commit", required=True)
    a.add_argument("--oss-prefix", required=True)
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("render", help="Offline dry-run; never submits a job")
    a.add_argument("--config", type=Path, required=True)
    a.add_argument("--manifest", type=Path, required=True)
    a.add_argument("--site", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("run")
    a.add_argument("--config", type=Path, required=True)
    a.add_argument("--source", type=Path, required=True)
    a.add_argument("--commit", required=True)
    a.add_argument("--archive-sha256", required=True)
    args = p.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.repo, args.commit, args.output, args.oss_prefix)))
    elif args.action == "render":
        render(
            load_config(args.config),
            json.loads(args.manifest.read_text()),
            json.loads(args.site.read_text()),
            args.output,
        )
    else:
        if args.source.resolve() != Path(__file__).resolve().parents[2]:
            raise ValueError("Launcher is not running from the selected source")
        return run(
            load_config(args.config),
            args.source.resolve(),
            args.commit,
            args.archive_sha256,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
