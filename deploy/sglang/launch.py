"""Prepare, render and launch existing SGLang workers without application bindings."""

import argparse
import base64
import hashlib
import ipaddress
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from pathlib import Path

from source import digest, prepare, sha, validate_manifest, write_json

MODEL_ID = "zing-h3-dmd-1102-step2000-generator"
MODEL_IDS = {
    MODEL_ID,
    "model1027",
    "zing-h3-dmd-1103-step2000-generator",
    "zing-h3-dmd-1115-step1700-generator",
}
ROLE_POOL_SECRETS = {"ROLE_POOL_SIGNING_KEY", "ROLE_POOL_REGISTRATION_TOKEN"}
SECRET_ENV = ROLE_POOL_SECRETS | {
    "WHITEBOX_CACHE_RECORDING_SECRET",
    "SGLANG_CACHE_ADMIN_TOKEN",
}
ROLE_PORTS = {
    "encoder": {"port": 30000, "http_port": 31000, "rendezvous_port": 32000},
    "dit": {"http_port": 33000, "port": 34000, "rendezvous_port": 35000},
    "decoder": {"http_port": 36000, "port": 37000, "output_port": 38000},
}
HARDWARE_PROFILES = {
    "h100": {"capability": [9, 0], "minimum_memory_gib": 75},
    "h200": {"capability": [9, 0], "minimum_memory_gib": 130},
    "b300": {"capability": [10, 3], "minimum_memory_gib": 240},
}
DIT_OPTIONS = {
    "mem_fraction_static",
    "zing_denoising_step_list",
    "zing_solver",
    "zing_initial_denoising_step_list",
    "zing_initial_block_count",
    "zing_block0_cache_mode",
    "zing_block0_cache_backend",
    "zing_block0_mooncake_config",
    "zing_block0_cache_dir",
    "zing_block0_cache_host_gb",
    "zing_block0_cache_storage",
    "zing_block0_cache_disk_gb",
    "zing_block0_cache_host_total_gb",
    "zing_block0_cache_budget_root",
    "zing_block0_cache_min_free_gb",
    "zing_block0_cache_inflight_gb",
    "zing_block0_cache_max_writers",
    "zing_block0_cache_gc_high_watermark",
    "zing_block0_cache_gc_low_watermark",
    "zing_block0_cache_capacity_dir",
    "zing_source_joint_forward",
    "timeout_seconds",
}
UNSUPPORTED_DIT_OPTIONS = {
    "zing_block0_cache_steps",
    "zing_block0_cache_lv_threshold",
    "zing_block0_cache_require_same_condition",
}


def allowed_env(key):
    return key in ENV_ALLOW or key.startswith(
        ("SGLANG_ZING_", "SGLANG_H3_QWEN_", "MINIMAX_H3_PROGRESSIVE_")
    )


def role_ports(c):
    ports = {**ROLE_PORTS[c["role"]], **c.get("ports", {})}
    if (
        set(ports) != set(ROLE_PORTS[c["role"]])
        or any(type(p) is not int or not 1024 <= p <= 65535 for p in ports.values())
        or len(set(ports.values())) != len(ports)
    ):
        raise ValueError("Invalid or overlapping role ports")
    return ports


def role_service_count(c):
    count = c.get("service_count", 1)
    if type(count) is not int or count < 1:
        raise ValueError("service_count must be a positive integer")
    if c.get("role") == "dit" and count != 1:
        raise ValueError("DiT supports only service_count=1")
    return count


def role_port_ranges(c):
    count = role_service_count(c)
    bases = role_ports(c)
    ranges = {}
    used = set()
    for name, base in bases.items():
        if base + count - 1 > 65535:
            raise ValueError("Invalid or overlapping role ports")
        values = list(range(base, base + count))
        if used & set(values):
            raise ValueError("Invalid or overlapping role ports")
        used.update(values)
        ranges[name] = values
    return ranges


def role_allocated_ports(c):
    return [port for values in role_port_ranges(c).values() for port in values]


def role_hardware(c):
    profile = c.get("profile")
    if profile in HARDWARE_PROFILES:
        expected = dict(HARDWARE_PROFILES[profile])
    else:
        expected = dict(HARDWARE_PROFILES["h200" if c["role"] == "dit" else "h100"])
    if c.get("hardware") is not None:
        hw = c["hardware"]
        if (
            not isinstance(hw, dict)
            or set(hw) - {"capability", "minimum_memory_gib"}
            or hw.get("capability") not in ([9, 0], [10, 3])
            or type(hw.get("minimum_memory_gib")) is not int
            or hw["minimum_memory_gib"] < 1
        ):
            raise ValueError("Invalid role hardware gate")
        if profile in HARDWARE_PROFILES and hw != expected:
            raise ValueError("Hardware gate conflicts with role profile")
        return dict(hw)
    return expected


PREFIX = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3."
)
MODULES = {
    "encoder": "sglang.multimodal_gen.runtime.entrypoints.zing_source_encoder_server",
    "dit": PREFIX + "dit_worker",
    "decoder": PREFIX + "decoder_worker",
}
PORTS = {"encoder": 31000, "dit": 33000, "decoder": 36000}
# Runtime credentials are explicit and redacted; source and device allocation stay isolated.
ENV_ALLOW = {
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_EC2_METADATA_DISABLED",
    "WORLD_MODEL_METRIC_LANE",
    "SGLANG_ROLE_POOL_WORKER_CONFIG",
    "ROLE_POOL_SIGNING_KEY",
    "ROLE_POOL_REGISTRATION_TOKEN",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MIN_CTAS",
    "NCCL_GRAPH_REGISTER",
    "NCCL_NET",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTORCH_CUDA_ALLOC_CONF",
    "CUDA_CACHE_PATH",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "XDG_CACHE_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "TOKENIZERS_PARALLELISM",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "SGLANG_DIFFUSION_STAGE_LOGGING",
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
    "SGLANG_LYNN_SM90_FAST_DECODE",
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
    "ports",
    "dit_options",
    "profile",
    "resolution",
    "service_count",
    "worker_config",
    "secret_files",
    "stage_from",
}


def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def validate_config(c):
    if (
        c.get("schema") != 1
        or set(c) - CONFIG_KEYS
        or c.get("role") not in MODULES
        or c.get("model_id") not in MODEL_IDS
    ):
        raise ValueError("Unknown configuration field, role or model identity")
    forbidden = {
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "HOME",
        "PATH",
        "LD_LIBRARY_PATH",
    }
    if any(not allowed_env(k) or k in forbidden for k in c.get("environment", {})):
        raise ValueError("Environment key is not allowlisted")
    if SECRET_ENV & c.get("environment", {}).keys():
        raise ValueError(
            "Secrets must be injected by the platform, never stored in role specs"
        )
    environment = c.get("environment", {})
    credential_file = environment.get("AWS_SHARED_CREDENTIALS_FILE")
    if credential_file is not None and (
        not isinstance(credential_file, str)
        or "\0" in credential_file
        or not Path(credential_file).is_absolute()
    ):
        raise ValueError("AWS credentials file must be an absolute mounted path")
    if environment.get("AWS_EC2_METADATA_DISABLED", "true") not in ("true", "false"):
        raise ValueError("AWS_EC2_METADATA_DISABLED must be true or false")
    if set(c.get("secret_files", {})) - SECRET_ENV or any(
        not Path(v).is_absolute() for v in c.get("secret_files", {}).values()
    ):
        raise ValueError("Secret files require known secret names and absolute paths")
    if c.get("worker_config"):
        w = c["worker_config"]
        required = {
            "role",
            "service_count",
            "coordinator_url",
            "pool_id",
            "control_epoch",
            "compatibility",
            "network_domain",
            "capacity",
            "profiles",
            "profile_limits",
            "hardware_ids",
            "signing_secret_env",
            "role_token_env",
        }
        if required - set(w) or set(w) - required - {"schema", "heartbeat_seconds"}:
            raise ValueError("Unknown or incomplete worker agent configuration")
        if (
            w["signing_secret_env"] != "ROLE_POOL_SIGNING_KEY"
            or w["role_token_env"] != "ROLE_POOL_REGISTRATION_TOKEN"
        ):
            raise ValueError(
                "Worker agent secrets must use the supported platform environment names"
            )
        if w.get("role") != c["role"] or w.get("service_count") != role_service_count(
            c
        ):
            raise ValueError("Worker agent role/service count mismatch")
    role_port_ranges(c)
    if c.get("profile") not in (None, "auto", "h100", "h200", "b300"):
        raise ValueError("Unknown hardware profile")
    if c["role"] == "dit" and c.get("profile") == "h100":
        raise ValueError("DiT profile must be auto, h200 or b300")
    if c.get("hardware") is not None:
        role_hardware(c)
    options = c.get("dit_options", {})
    if not isinstance(options, dict) or any(
        type(v) not in (str, float, int) for v in options.values()
    ):
        raise ValueError("DiT options must be string or numeric native CLI values")
    if (
        "mem_fraction_static" in options
        and not 0 < float(options["mem_fraction_static"]) <= 1
    ):
        raise ValueError("Memory fraction must be in (0, 1]")
    if set(options) & UNSUPPORTED_DIT_OPTIONS:
        raise ValueError(
            "Unsupported native DiT options: "
            + ", ".join(sorted(set(options) & UNSUPPORTED_DIT_OPTIONS))
        )
    if set(options) - DIT_OPTIONS or (options and c["role"] != "dit"):
        raise ValueError("Unknown or non-DiT options")
    if (
        "zing_block0_mooncake_config" in options
        and not Path(str(options["zing_block0_mooncake_config"])).is_absolute()
    ):
        raise ValueError("Mooncake configuration path must be absolute")
    if c.get("resolution", "480") not in ("480", "736"):
        raise ValueError("Resolution must be 480 or 736")
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
    for key in (
        "model",
        "asset_receipt",
        *(("stage_from",) if c.get("stage_from") else ()),
    ):
        if not Path(c[key]).is_absolute():
            raise ValueError("Asset paths must be absolute")
    sha(c["asset_receipt_sha256"])
    if not 1 <= c.get("startup_timeout", 900) <= 3600:
        raise ValueError("Startup timeout must be in [1, 3600] seconds")
    return c


def asset_files(model):
    for directory, _, names in os.walk(model, followlinks=True):
        for name in names:
            path = Path(directory) / name
            if path.is_file():
                yield path


def stage_assets(c):
    """Copy hot serving assets locally; leave large peripheral components on the mount."""
    source = Path(c["stage_from"])
    destination = Path(c["model"]).parent
    if destination.is_symlink() or Path(c["model"]).is_symlink():
        raise ValueError("Staging destination cannot traverse a symlink")
    if not (source / "model").is_dir() or Path(c["model"]).name != "model":
        raise ValueError(
            "Staging requires a source model directory and a destination named model"
        )
    if (
        source.resolve() == destination.resolve()
        or source.resolve().is_relative_to(destination.resolve())
        or destination.resolve().is_relative_to(source.resolve())
    ):
        raise ValueError("Staging source and destination must not overlap")
    role = c.get("role", "dit")
    base_files = {
        "zing_config.json",
        "model_index.json",
        "zing_realtime_profile.json",
        "EXPORT_COMPLETE.json",
    }
    if role == "encoder":
        selected = base_files | {"video_vae", "processor"}
        copy_adaln = False
        copy_lynn = False
    elif role == "decoder":
        selected = set(base_files)
        copy_adaln = False
        copy_lynn = True
    else:
        selected = base_files | {"transformer"}
        copy_adaln = True
        copy_lynn = False
    paths = []
    for name in selected:
        path = source / "model" / name
        if path.is_dir():
            paths.extend(asset_files(path))
        elif path.is_file():
            paths.append(path)
    if copy_adaln and (source / "adaln-table-unsplit").is_dir():
        paths.extend(asset_files(source / "adaln-table-unsplit"))
    if copy_lynn and (source / "lynn").is_dir():
        paths.extend(asset_files(source / "lynn"))

    def copy(path):
        target = destination / path.relative_to(source)
        # Do not follow a previous destination symlink back into shared storage.
        if any(
            parent.is_symlink()
            for parent in (target, *target.parents)
            if parent == destination or destination in parent.parents
        ):
            raise ValueError("Staging destination cannot traverse a symlink")
        if (
            path.suffix in {".safetensors", ".pt", ".bin"}
            and not target.is_symlink()
            and target.is_file()
            and target.stat().st_size == path.stat().st_size
        ):
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp-" + uuid.uuid4().hex)
        try:
            shutil.copyfile(path, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(copy, paths))
    model = destination / "model"
    model.mkdir(parents=True, exist_ok=True)
    extra_roots = []
    if not copy_lynn:
        extra_roots.append(source / "lynn")
    if not copy_adaln:
        extra_roots.append(source / "adaln-table-unsplit")
    for path in [
        p for p in (source / "model").iterdir() if p.name not in selected
    ] + extra_roots:
        if not path.exists():
            continue
        target = (
            model / path.name
            if path.parent == source / "model"
            else destination / path.name
        )
        if target.is_symlink() and target.resolve() == path.resolve():
            continue
        if target.exists() or target.is_symlink():
            raise ValueError("Staging refuses to replace an existing peripheral asset")
        target.symlink_to(path.resolve(), target_is_directory=path.is_dir())
    profile = model / "zing_realtime_profile.json"
    data = json.loads(profile.read_text()) if profile.exists() else {}
    data["source_attention_window"] = {"window_chunks": 6, "sink_chunks": 0}
    profile.write_text(json.dumps(data, indent=2) + "\n")


def verify_assets(c):
    path = Path(c["asset_receipt"])
    if digest(path) != c["asset_receipt_sha256"]:
        raise ValueError("Asset receipt digest mismatch")
    a = json.loads(path.read_text())
    if (
        not isinstance(a, dict)
        or a.get("schema") != 1
        or a.get("model_id") != c["model_id"]
        or a.get("component") != "generator"
        or a.get("ema") is not False
        or not isinstance(a.get("model"), str)
        or Path(a["model"]).resolve() != Path(c["model"]).resolve()
    ):
        raise ValueError("Model identity or component mismatch")
    # Receipt is produced by the asset owner after verifying the original export.
    # String entries are fully hashed; owner-supplied size entries avoid rereading cold weights.
    files = a.get("files")
    if (
        not isinstance(files, dict)
        or not files
        or any(not Path(p).is_absolute() for p in files)
    ):
        raise ValueError("Asset receipt requires absolute artifact paths")
    required = [str(Path(c["model"]) / "zing_config.json")]
    required += [c[k] for k in ("pipeline", "decoder_config") if c.get(k)]
    if any(p not in files for p in required):
        raise ValueError("Role config is not covered by asset receipt")
    weights = [
        p
        for p in asset_files(c["model"])
        if p.suffix in {".safetensors", ".pt", ".bin"}
    ]
    if not weights or any(str(p) not in files for p in weights):
        raise ValueError("Model weight files must be covered by asset receipt")
    for p, expected in files.items():
        full_hash = True
        if isinstance(expected, dict):
            if (
                set(expected) != {"bytes", "sha256"}
                or type(expected["bytes"]) is not int
                or expected["bytes"] < 0
            ):
                raise ValueError("Invalid sized asset receipt entry: " + p)
            expected_hash = sha(expected["sha256"])
            if Path(p).stat().st_size != expected["bytes"]:
                raise ValueError("Asset size mismatch: " + p)
            full_hash = p in required or Path(p).suffix.lower() in {
                ".json",
                ".yaml",
                ".yml",
            }
        else:
            expected_hash = sha(expected)
        if full_hash and digest(p) != expected_hash:
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
                or adaln.get("model_id") != c["model_id"]
            ):
                raise ValueError("AdaLN model/schedule/CUDA provenance is required")
        elif adaln["mode"] != "resident" or pipeline.get("zing_adaln_table_path"):
            raise ValueError("Invalid resident AdaLN configuration")
    return a


def reachable_ip():
    vpc = ipaddress.ip_network("10.0.0.0/8")
    route_succeeded = False
    for peer in ("10.0.1.37", "10.0.0.121"):
        try:
            # Kernel route selection only; UDP connect sends no packet.
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((peer, 9))
                address = ipaddress.ip_address(s.getsockname()[0])
            route_succeeded = True
            if address in vpc:
                return str(address)
        except OSError:
            pass
    value = os.environ.get("POD_IP")
    if value:
        address = ipaddress.ip_address(value)
        if not (
            address.is_loopback or address.is_unspecified or address.is_multicast
        ) and (address in vpc or not route_succeeded):
            return str(address)
    raise ValueError("Platform did not provide a usable advertise IP")


def command(c, source, host, worker_id):
    role = c["role"]
    native_args = [
        "--model",
        c["model"],
        "--service-count",
        str(role_service_count(c)),
        "--http-port",
        str(role_ports(c)["http_port"]),
        "--advertise-host",
        host,
        "--worker-id",
        worker_id,
    ]
    for name, port in role_ports(c).items():
        if name != "http_port":
            native_args += ["--" + name.replace("_", "-"), str(port)]
    if c.get("pipeline"):
        native_args += ["--pipeline-config-path", c["pipeline"]]
    if role == "dit":
        native_args += ["--sp-degree", "1"]
        for name, value in c.get("dit_options", {}).items():
            native_args += ["--" + name.replace("_", "-"), str(value)]
        if c.get("quantization"):
            native_args += ["--quantization", c["quantization"]]
    if role == "decoder":
        native_args += [
            "--decoder-config",
            json.dumps(
                json.loads(Path(c["decoder_config"]).read_text()), separators=(",", ":")
            ),
        ]
    env = {k: v for k, v in os.environ.items() if allowed_env(k)}
    if role == "encoder":
        env.setdefault("SGLANG_ZING_SOURCE_PREPARE", "1")
    env.update(c.get("environment", {}))
    if role == "encoder":
        # Room negotiates preparation per request. Advertise support on its
        # standalone ingress too; an explicit deployment override still wins.
        env.setdefault("SGLANG_ZING_SOURCE_PREPARE", "1")
    for key, path in c.get("secret_files", {}).items():
        env[key] = Path(path).read_text().strip()
        if not env[key] or "\x00" in env[key]:
            raise ValueError("Invalid platform secret file")
    if c.get("worker_config"):
        env["SGLANG_ROLE_POOL_WORKER_CONFIG"] = str(
            source.parent / "role_pool_worker.json"
        )
    env.update(
        PYTHONPATH=str(source / "python"),
        PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "1"),
        WORLD_MODEL_METRIC_SERVICE=role,
        WORLD_MODEL_METRIC_LANE=env.get("WORLD_MODEL_METRIC_LANE", "default"),
        PROMETHEUS_MULTIPROC_DIR=str(source.parent / "prometheus" / worker_id),
    )
    # #471 owns hardware policy and its native argv; no policy is copied here.
    argv = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.tools.zing_h3_profile_launcher",
        "--mode",
        "independent",
        "--component",
        role,
        "--profile",
        c.get("profile", "auto"),
        "--resolution",
        c.get("resolution", "480"),
        "--sp-degree",
        "1",
        "--",
        *native_args,
    ]
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
    boot_t0 = time.monotonic()
    stage_t0 = boot_t0
    if c.get("stage_from"):
        stage_assets(c)
    stage_s = round(time.monotonic() - stage_t0, 3) if c.get("stage_from") else 0.0
    verify_t0 = time.monotonic()
    assets = verify_assets(c)
    verify_s = round(time.monotonic() - verify_t0, 3)
    attestation = json.loads((source.parent / "verified-source.json").read_text())
    if attestation != {"commit": commit, "archive_sha256": archive_sha}:
        raise ValueError("Missing verified archive attestation")
    host = reachable_ip()
    worker_id = c["role"] + "-" + uuid.uuid4().hex
    argv, env = command(c, source, host, worker_id)
    # Unique boot identity: never reuse or clear another live worker's counters.
    # Set before the import probe and before torchrun forks any ranks.
    Path(env["PROMETHEUS_MULTIPROC_DIR"]).mkdir(parents=True, exist_ok=False)
    imports = probe_source(source, env, c["role"], c.get("hardware"))
    for port in role_allocated_ports(c):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
    worker_config = c.get("worker_config")
    if worker_config:
        worker_config = dict(worker_config)
        allocated = re.findall(r"GPU-[A-Za-z0-9-]+", gpu)
        if len(set(allocated)) != 1:
            raise ValueError("Managed worker requires exactly one observed GPU UUID")
        if worker_config["hardware_ids"] == ["__ALLOCATED_GPU_UUID__"]:
            worker_config["hardware_ids"] = sorted(set(allocated))
        if worker_config["hardware_ids"] != sorted(set(allocated)):
            raise ValueError("Managed hardware identity differs from allocated GPU")
        write_json(source.parent / "role_pool_worker.json", worker_config)
    receipt = {
        "schema": 1,
        "worker_config": worker_config,
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
        "model_id": c["model_id"],
        "asset_receipt_sha256": c["asset_receipt_sha256"],
        "model_path": c["model"],
        "pipeline": c.get("pipeline"),
        "decoder_config": c.get("decoder_config"),
        "adaln": assets.get("adaln"),
        "role": c["role"],
        "worker_id": worker_id,
        "ip": host,
        "http": "http://" + host + ":" + str(role_ports(c)["http_port"]),
        "metrics": {
            "url": "http://"
            + host
            + ":"
            + str(role_ports(c)["http_port"])
            + "/metrics",
            "service": c["role"],
            "lane": env["WORLD_MODEL_METRIC_LANE"],
            "multiprocess_dir": env["PROMETHEUS_MULTIPROC_DIR"],
            "enabled": env.get("SGLANG_ZING_COMPONENT_METRICS", "1") != "0",
        },
        "argv": argv,
        "environment": {
            k: v
            for k, v in env.items()
            if k not in {"HOME", "PATH", "LD_LIBRARY_PATH"} | SECRET_ENV
        },
        "runtime": imports,
        "gpu": gpu,
        "pod": os.environ.get("POD_NAME", socket.gethostname()),
        "pod_uid": os.environ.get("POD_UID"),
        "job": os.environ.get("PAI_JOB_ID")
        or os.environ.get("DLC_JOB_ID")
        or os.environ.get("JOB_ID"),
        "pid": None,
        "warmed_up": False,
        "timings_s": {
            "stage_assets_s": stage_s,
            "verify_assets_s": verify_s,
        },
    }
    output = source.parent / "receipt.json"
    options = c.get("dit_options", {})
    if c.get("role") == "dit" and options.get("zing_block0_cache_storage") == "local":
        for key in (
            "zing_block0_cache_dir",
            "zing_block0_cache_budget_root",
            "zing_block0_cache_capacity_dir",
        ):
            if options.get(key):
                Path(options[key]).mkdir(parents=True, exist_ok=True)
    spawn_t0 = time.monotonic()
    child = subprocess.Popen(argv, cwd=source, env=env, start_new_session=True)
    receipt["pid"] = child.pid

    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(output)
        line = json.dumps(receipt)
        print(line, flush=True)
        if len(line) > 1800:
            omitted = {"adaln", "argv", "environment"}
            compact = {
                key: value for key, value in receipt.items() if key not in omitted
            }
            compact_line = json.dumps(compact, separators=(",", ":"))
            if len(compact_line) > 1800 or receipt.get("worker_config") is not None:
                omitted.update(
                    {
                        "runtime",
                        "launcher_sha256",
                        "release_id",
                        "image_requested",
                        "image_actual",
                        "pipeline",
                        "decoder_config",
                        "model_path",
                    }
                )
                compact = {
                    key: value for key, value in receipt.items() if key not in omitted
                }
                compact_line = json.dumps(compact, separators=(",", ":"))
            print(compact_line, flush=True)

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
                    now_mono = time.monotonic()
                    receipt["timings_s"] = {
                        "stage_assets_s": stage_s,
                        "verify_assets_s": verify_s,
                        "worker_ready_s": round(now_mono - spawn_t0, 3),
                        "total_boot_s": round(now_mono - boot_t0, 3),
                    }
                    receipt.update(
                        status="ready",
                        health=health,
                        warmed_up=bool(health.get("warmed_up", True)),
                    )
                    if health.get("warmup") is not None:
                        receipt["warmup"] = health["warmup"]
                    save()
                    break
            except (OSError, ValueError, HTTPException):
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Worker startup health deadline exceeded")
        while True:
            try:
                code = child.wait(timeout=15)
                break
            except subprocess.TimeoutExpired:
                # Keep a bounded log tail useful to the cross-cluster receipt collector.
                save()
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
        zlib.compress(Path(__file__).with_name("source.py").read_bytes(), 9)
    ).decode()
    code = (
        "import base64,zlib;exec(compile(zlib.decompress(base64.b64decode("
        + repr(payload)
        + ")),'<source-bootstrap>','exec'))"
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
    is_pai = "job_spec" in site
    if c["role"] == "dit" and not is_pai:
        raise ValueError("DiT requires a PAI job_spec site")
    allowed = (
        {
            "schema",
            "image",
            "namespace",
            "volumes",
            "volume_mounts",
            "model_id",
            "image_pull_policy",
        }
        if not is_pai
        else {
            "schema",
            "image",
            "job_spec",
            "data_sources",
            "model_id",
            "workspace_id",
            "image_auth_file",
        }
    )
    if site.get("image_auth_file") is not None:
        if (
            not isinstance(site["image_auth_file"], str)
            or not site["image_auth_file"]
            or not Path(site["image_auth_file"]).is_absolute()
        ):
            raise ValueError("image_auth_file must be an absolute path")
    if site.get("model_id", c["model_id"]) != c["model_id"]:
        raise ValueError("Site and role model identity mismatch")
    if site.get("image_pull_policy", "Never") not in {"Never", "IfNotPresent"}:
        raise ValueError("Unsupported image pull policy")
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
        "hardware": role_hardware(c),
    }
    output.mkdir(parents=True, exist_ok=False)
    argv = bootstrap_command(manifest, c)
    name = "sglang-" + c["role"] + "-" + uuid.uuid4().hex[:12]
    if not is_pai:
        cache = "/var/lib/sglang-orchestrate/cache"
        volumes = list(site["volumes"])
        mounts = list(site["volume_mounts"])
        existing = next((m for m in mounts if m["mountPath"] == cache), None)
        if existing:
            volume = next((v for v in volumes if v["name"] == existing["name"]), {})
            if (
                not existing.get("readOnly")
                or volume.get("hostPath", {}).get("path") != cache
            ):
                raise ValueError(
                    "Source cache requires a matching read-only hostPath mount"
                )
        else:
            if any(v["name"] == "sglang-source-cache" for v in volumes):
                raise ValueError("Source cache volume name is already in use")
            volumes.append(
                {
                    "name": "sglang-source-cache",
                    "hostPath": {"path": cache, "type": "Directory"},
                }
            )
            mounts.append(
                {"name": "sglang-source-cache", "mountPath": cache, "readOnly": True}
            )
        # Discovery and selection occur on the target node, never on the user's kube context.
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": name, "namespace": site["namespace"]},
            "spec": {
                "nodeName": "world-model-b27mys7b-0000",
                "runtimeClassName": "nvidia",
                "hostNetwork": True,
                "dnsPolicy": "ClusterFirstWithHostNet",
                "restartPolicy": "Never",
                "containers": [
                    {
                        "name": c["role"],
                        "image": site["image"],
                        "imagePullPolicy": site.get("image_pull_policy", "Never"),
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
                        "volumeMounts": mounts,
                    }
                ],
                "volumes": volumes,
            },
        }
        # Node local advisory lock prevents two invocations of this launcher selecting the same card.
        script = """#!/bin/sh
set -eu
exec 9>/tmp/sglang-role-allocation.lock
flock -n 9
"""
        repo, digest_ref = site["image"].split("@sha256:", 1)
        cached_tag = repo + ":cached-" + digest_ref[:12]
        script += (
            "/var/lib/rancher/k3s/data/current/bin/ctr -a /run/k3s/containerd/containerd.sock -n k8s.io images tag "
            + shlex.quote(site["image"])
            + " "
            + shlex.quote(cached_tag)
            + " >/dev/null 2>&1 || true\n"
        )
        selector = base64.b64encode(
            zlib.compress(Path(__file__).with_name("select_gpu.py").read_bytes(), 9)
        ).decode()
        selector_code = (
            "import base64,zlib;exec(compile(zlib.decompress(base64.b64decode("
            + repr(selector)
            + ")),'<gpu-selector>','exec'))"
        )
        script += "GPU_UUID=$(python3 -c " + shlex.quote(selector_code) + ")\n"
        script += (
            "POD_JSON=" + shlex.quote(json.dumps(pod, separators=(",", ":"))) + "\n"
        )
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
        workspace_id = site.get("workspace_id", "1453237")
        quota_id = spec.get("QuotaId")
        if (
            set(spec)
            - {
                "Type",
                "PodCount",
                "QuotaId",
                "RestartPolicy",
                "ResourceConfig",
                "ImageConfig",
            }
            or spec.get("PodCount") != 1
            or spec.get("ResourceConfig", {}).get("GPU") != "1"
            or spec.get("Type") != "Worker"
            or not isinstance(quota_id, str)
            or not quota_id
            or (
                c.get("profile") != "b300"
                and workspace_id == "1453237"
                and quota_id != "quota3l8bnv9a35n"
            )
            or spec.get("RestartPolicy") != "Never"
        ):
            raise ValueError("PAI DiT requires one pod with exactly one GPU")
        spec = {**spec, "Image": site["image"]}
        if site.get("image_auth_file"):
            auth_path = Path(site["image_auth_file"])
            auth_value = auth_path.read_text().strip()
            if not auth_value or "\x00" in auth_value:
                raise ValueError("Invalid PAI image auth secret file")
            image_cfg = dict(spec.get("ImageConfig") or {})
            image_cfg.setdefault("DockerRegistry", site["image"].split("/", 1)[0])
            image_cfg["Auth"] = auth_value
            spec["ImageConfig"] = image_cfg
        job = {
            "DisplayName": name,
            "JobType": "PyTorchJob",
            "WorkspaceId": workspace_id,
            "ResourceId": quota_id,
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
            "model_id": c["model_id"],
            "status": "rendered-not-submitted",
            "schema": 1,
        },
    )


def generate_assets(config, extra_files, output, sized_weights=False):
    """Hash local exports; original checkpoint provenance remains owner supplied."""
    model = Path(config["model"])
    files = {p.absolute() for p in asset_files(model)}
    files.update(Path(p).absolute() for p in extra_files)
    files.update(
        Path(config[k]).absolute()
        for k in ("pipeline", "decoder_config")
        if config.get(k)
    )
    adaln = {"mode": "resident"}
    if config.get("pipeline"):
        pipeline = json.loads(Path(config["pipeline"]).read_text())
        adaln["mode"] = pipeline.get("zing_adaln_mode", "weights_free")
        if adaln["mode"] == "weights_free":
            table = Path(pipeline["zing_adaln_table_path"]).absolute()
            manifest = json.loads((table / "manifest.json").read_text())
            files.update(
                (table / name) for name in ("manifest.json", "table.safetensors")
            )
            adaln.update(
                table=str(table),
                model_id=config["model_id"],
                schedule=manifest["schedule"],
                cuda_provenance=manifest["arithmetic"],
            )

    def _entry(path):
        digest_hex = digest(path)
        if sized_weights and path.suffix in {".safetensors", ".pt", ".bin"}:
            return {"bytes": path.stat().st_size, "sha256": digest_hex}
        return digest_hex

    receipt = dict(
        schema=1,
        model_id=config["model_id"],
        component="generator",
        ema=False,
        model=str(model.absolute()),
        files={str(p): _entry(p) for p in sorted(files)},
        adaln=adaln,
    )
    # Validate before publishing a new receipt, including all role-required assets.
    temporary = output.with_name(output.name + ".tmp-" + uuid.uuid4().hex)
    try:
        write_json(temporary, receipt)
        verify_assets(
            {
                **config,
                "asset_receipt": str(temporary),
                "asset_receipt_sha256": digest(temporary),
            }
        )
        if output.exists():
            raise FileExistsError(output)
        write_json(output, receipt)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "asset_receipt": str(output.resolve()),
        "asset_receipt_sha256": digest(output),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("assets", help="Generate or verify local asset hashes")
    a.add_argument("--config", type=Path, required=True)
    a.add_argument("--output", type=Path)
    a.add_argument("--extra-file", type=Path, action="append", default=[])
    a.add_argument(
        "--sized-weights",
        action="store_true",
        help="Emit {bytes, sha256} entries for weight files to speed up cold mount verification",
    )
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
    if args.action == "assets":
        c = load_config(args.config)
        print(
            json.dumps(
                generate_assets(
                    c, args.extra_file, args.output, sized_weights=args.sized_weights
                )
                if args.output
                else verify_assets(c)
            )
        )
    elif args.action == "prepare":
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
            args.source.absolute(),
            args.commit,
            args.archive_sha256,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
