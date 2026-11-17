"""SGLang multi-role environment orchestration. Planning is offline; mutations require --execute.

No local kube context, SSH aliases, or automatic create retries.
State files are private operator evidence, not a source of arbitrary shell commands.
"""

import argparse
import asyncio
import base64
import contextlib
import copy
import datetime

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None
import hashlib
import importlib.util
import io
import ipaddress
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import launch
from source import digest, prepare

ROLES = ("encoder", "dit", "decoder")
SHANGHAI_OSS_BUCKET = "oss://leap-world-shanghai"
THAILAND_OSS_BUCKET = "oss://leap-worldmodel-thailand-single-az"
LEGACY_THAILAND_OSS_BUCKET = "oss://leap-worldmodel-thailand"
ENVIRONMENT_OSS_HISTORY_PREFIXES = {
    "selftest": f"{SHANGHAI_OSS_BUCKET}/sglang/deployments/selftest",
    "test-shanghai-copy": f"{SHANGHAI_OSS_BUCKET}/sglang/deployments/test-shanghai-copy",
    "prod-shanghai": f"{SHANGHAI_OSS_BUCKET}/sglang/deployments/prod-shanghai",
    "prod-thailand": f"{THAILAND_OSS_BUCKET}/sglang/deployments/prod-thailand",
}
TARGET = {
    "environment": "test-shanghai-copy",
    "region": "cn-shanghai",
    "node_id": "e01-cn-2zd4xi8m505",
    "h100_ip": "10.0.0.121",
    "workspace_id": "1453237",
    "quota_id": "quota3l8bnv9a35n",
    "coordinator_url": "http://10.0.1.37:18181",
    "pool_id": "adaptive-test-20261109",
    "room_url": "http://10.0.1.34:8092",
    "wtp_url": "https://shcopy.world-test-platform.loopit.com.cn",
    "oss_prefix": "oss://leap-world-shanghai/sglang/orchestration",
}
REPO = Path(__file__).resolve().parents[2]


def resolve_environment_region(target=None, *, environment=None, region=None):
    env_name = str(
        environment
        or (target.get("environment") if isinstance(target, dict) else None)
        or "test-shanghai-copy"
    ).strip()
    if env_name == "prod-thailand":
        return "ap-southeast-7"
    if env_name in ("test-shanghai-copy", "prod-shanghai"):
        return "cn-shanghai"
    eff_region = str(
        region
        or (target.get("region") if isinstance(target, dict) else None)
        or "cn-shanghai"
    ).strip()
    return eff_region or "cn-shanghai"


def resolve_oss_history_prefix(target=None, *, environment=None, region=None):
    """Resolve the fixed OSS deployment history prefix separated by environment tier."""
    env_name = str(
        environment
        or (target.get("environment") if isinstance(target, dict) else None)
        or "test-shanghai-copy"
    ).strip()
    if not env_name:
        raise ValueError("Environment name is required to resolve OSS history prefix")
    eff_region = resolve_environment_region(target, environment=env_name, region=region)
    if env_name in ("test-shanghai-copy", "prod-shanghai", "prod-thailand"):
        return ENVIRONMENT_OSS_HISTORY_PREFIXES[env_name]
    bucket = (
        THAILAND_OSS_BUCKET if eff_region == "ap-southeast-7" else SHANGHAI_OSS_BUCKET
    )
    if env_name == "selftest":
        return f"{bucket}/sglang/deployments/selftest"
    safe_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", env_name).strip("-")
    if not safe_slug or safe_slug in (
        "test-shanghai-copy",
        "prod-shanghai",
        "prod-thailand",
    ):
        raise ValueError(f"Invalid custom selftest environment name: {env_name!r}")
    return f"{bucket}/sglang/deployments/selftest/{safe_slug}"


def _load_profile_launcher():
    path = REPO / "python/sglang/multimodal_gen/tools/zing_h3_profile_launcher.py"
    spec = importlib.util.spec_from_file_location("sglang_orch_profile_launcher", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_role_launcher():
    path = REPO / "python/sglang/multimodal_gen/tools/zing_h3_role_launcher.py"
    spec = importlib.util.spec_from_file_location("sglang_orch_role_launcher", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        temp.replace(path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check_target(value):
    if not isinstance(value, dict):
        raise ValueError("Target must be a JSON object")
    allowed_keys = set(TARGET) | {"topology_profile", "history_dir", "sync_history_oss"}
    if set(value) - allowed_keys:
        raise ValueError("Unsupported target configuration keys")
    if "sync_history_oss" in value and not isinstance(value["sync_history_oss"], bool):
        raise ValueError("target.sync_history_oss must be a boolean")
    env_name = str(value.get("environment", "")).strip()
    if not env_name:
        raise ValueError("Target environment is required")
    if env_name == "test-shanghai-copy":
        if {k: value.get(k) for k in TARGET} != TARGET:
            raise ValueError("Only the pinned test-shanghai-copy target is permitted")
    else:
        required = {
            "environment",
            "region",
            "workspace_id",
            "quota_id",
            "coordinator_url",
            "pool_id",
            "room_url",
            "wtp_url",
            "oss_prefix",
        }
        if not required <= set(value) or any(
            not isinstance(value[k], str) or not value[k].strip() for k in required
        ):
            raise ValueError("Target is missing required environment fields")
        if not value["oss_prefix"].startswith("oss://"):
            raise ValueError("Target oss_prefix must be an oss:// URI")
        raw_oss = value["oss_prefix"].rstrip("/")
        if raw_oss == LEGACY_THAILAND_OSS_BUCKET or raw_oss.startswith(
            LEGACY_THAILAND_OSS_BUCKET + "/"
        ):
            raise ValueError(
                f"Legacy bucket {LEGACY_THAILAND_OSS_BUCKET} is forbidden; use {THAILAND_OSS_BUCKET}"
            )
        for url_key in ("coordinator_url", "room_url", "wtp_url"):
            uri = urlsplit(value[url_key])
            if uri.scheme not in ("http", "https") or not uri.hostname or uri.port == 0:
                raise ValueError(f"Invalid target {url_key}")
        h100_ip = value.get("h100_ip")
        if h100_ip is not None:
            if not isinstance(h100_ip, str):
                raise ValueError("Invalid h100_ip")
            ipaddress.ip_address(h100_ip)
    if "topology_profile" in value and value["topology_profile"] is not None:
        _load_role_launcher().resolve_topology_workaround(value["topology_profile"])


def confirm_execution(action, plan_summary, *, yes=False):
    print(
        json.dumps(
            {
                "action": action,
                "confirmation_required": not bool(yes),
                "plan": plan_summary,
            },
            sort_keys=True,
        )
    )
    if yes:
        return True
    if not sys.stdin or not sys.stdin.isatty():
        raise ValueError(
            f"Action '{action}' with --execute requires interactive confirmation or explicit --yes"
        )
    reply = input(f"Confirm '{action}' execution? [y/N]: ").strip().lower()
    if reply not in ("y", "yes"):
        raise RuntimeError(f"Execution of '{action}' cancelled by operator")
    return True


def expand_minimal_spec(raw):
    """Expand a minimal environment probe input into a full validated spec."""
    if not isinstance(raw, dict):
        raise ValueError("Spec must be a JSON object")
    if {"roles", "sites", "pool", "route"} <= set(raw):
        return validate_spec(copy.deepcopy(raw))
    spec = copy.deepcopy(raw)
    spec.setdefault("schema", 1)
    target = spec.setdefault("target", copy.deepcopy(TARGET))
    check_target(target)
    model_id = spec.setdefault("model_id", "model1027")
    topology = spec.pop("topology", None)
    if topology is None:
        sites_in = spec.get("sites", {})
        if "b300" in sites_in:
            topology = "b300"
        elif set(sites_in) == {"h200"}:
            topology = "h200"
        else:
            topology = "h100_h200"
    if topology not in ("h100_h200", "h200", "b300"):
        raise ValueError(f"Unsupported topology '{topology}'")
    default_model_dir = spec.pop("model", f"/assets/{model_id}")
    default_receipt = spec.pop("asset_receipt", "/assets/assets.json")
    default_receipt_sha = spec.pop("asset_receipt_sha256", "a" * 64)
    default_image = spec.pop("image", "test@sha256:" + "c" * 64)
    default_stage_from = spec.pop("stage_from", None)
    existing = spec.setdefault("existing_services", {})
    roles_in = spec.get("roles", {})
    roles = {}
    for role in ROLES:
        r_in = copy.deepcopy(roles_in.get(role, {}))
        mode = r_in.get("mode")
        if mode == "reuse" or role in existing:
            if role not in existing:
                existing[role] = {
                    k: r_in[k]
                    for k in (
                        "http",
                        "worker_id",
                        "service_index",
                        "service_count",
                        "gpu",
                        "boot_epoch",
                        "pai",
                        "job_id",
                        "job_name",
                    )
                    if k in r_in
                }
            continue
        prof = r_in.get(
            "profile",
            (
                "b300"
                if topology == "b300"
                else ("h200" if (topology == "h200" or role == "dit") else "h100")
            ),
        )
        role_cfg = {
            "schema": 1,
            "role": role,
            "mode": "create",
            "profile": prof,
            "model_id": model_id,
            "model": r_in.get("model", default_model_dir),
            "asset_receipt": r_in.get("asset_receipt", default_receipt),
            "asset_receipt_sha256": r_in.get(
                "asset_receipt_sha256", default_receipt_sha
            ),
            "environment": dict(r_in.get("environment", {})),
        }
        stage_from = r_in.get("stage_from", default_stage_from)
        if stage_from:
            role_cfg["stage_from"] = stage_from
        for opt_key in (
            "service_count",
            "ports",
            "secret_files",
            "resolution",
            "startup_timeout",
        ):
            if opt_key in r_in:
                role_cfg[opt_key] = r_in[opt_key]
        if role == "dit":
            role_cfg["pipeline"] = r_in.get("pipeline", "/assets/dit.json")
            role_cfg["dit_options"] = dict(
                r_in.get(
                    "dit_options",
                    {
                        "mem_fraction_static": 0.85,
                        "zing_denoising_step_list": "1000,500",
                    },
                )
            )
        elif role == "decoder":
            role_cfg["decoder_config"] = r_in.get(
                "decoder_config", "/assets/decoder.json"
            )
        roles[role] = role_cfg
    spec["roles"] = roles
    if "sites" not in spec:
        if topology == "b300":
            spec["sites"] = {
                "b300": {
                    "schema": 1,
                    "image": default_image,
                    "workspace_id": target["workspace_id"],
                    "job_spec": {
                        "Type": "Worker",
                        "PodCount": 1,
                        "QuotaId": target["quota_id"],
                        "RestartPolicy": "Never",
                        "ResourceConfig": {"GPU": "1"},
                    },
                    "data_sources": [],
                }
            }
        elif topology == "h200":
            spec["sites"] = {
                "h200": {
                    "schema": 1,
                    "image": default_image,
                    "job_spec": {
                        "Type": "Worker",
                        "PodCount": 1,
                        "QuotaId": target["quota_id"],
                        "RestartPolicy": "Never",
                        "ResourceConfig": {
                            "GPU": "1",
                            "CPU": "16",
                            "Memory": "128Gi",
                            "SharedMemory": "64Gi",
                        },
                    },
                    "data_sources": [],
                }
            }
        else:
            spec["sites"] = {
                "h100": {
                    "schema": 1,
                    "image": default_image,
                    "namespace": "sglang-test",
                    "volumes": [],
                    "volume_mounts": [],
                },
                "h200": {
                    "schema": 1,
                    "image": default_image,
                    "job_spec": {
                        "Type": "Worker",
                        "PodCount": 1,
                        "QuotaId": target["quota_id"],
                        "RestartPolicy": "Never",
                        "ResourceConfig": {"GPU": "1"},
                    },
                    "data_sources": [],
                },
            }
    spec.setdefault("route", "test-480")
    if "pool" not in spec:
        limits = {
            "source_sizes": [[832, 480]],
            "target_sizes": [[832, 480]],
            "max_fps": 24,
            "max_references": 6,
            "max_reference_bytes": 1000,
            "max_reference_pixels": 1000,
            "max_prompt_chars": 1000,
            "max_preview_pixels": 1000,
        }
        capacities = {}
        for role in ROLES:
            if role in existing:
                slots = existing[role].get("service_count", 1)
            else:
                slots = launch.role_service_count(roles[role])
            capacities[role] = {"slots": slots}
        spec["pool"] = {
            "control_epoch": "epoch-test",
            "compatibility": f"{model_id}-test",
            "network_domain": target["region"],
            "capacity": capacities,
            "profiles": {r: {"480": {"slots": 1}} for r in ROLES},
            "profile_limits": {"480": limits},
        }
    return validate_spec(spec)


def load_spec(path):
    return validate_spec(read(path))


def owned_roles(spec):
    reused = set(spec.get("existing_services", {}))
    for role, cfg in spec.get("roles", {}).items():
        if isinstance(cfg, dict) and cfg.get("mode") == "reuse":
            reused.add(role)
    return tuple(r for r in ROLES if r not in reused)


def role_site(spec, role):
    c = spec["roles"][role]
    profile = c.get("profile")
    if profile in ("h100", "h200", "b300") and profile in spec["sites"]:
        return spec["sites"][profile]
    for single_key in ("b300", "h200"):
        if single_key in spec["sites"] and len(spec["sites"]) == 1:
            return spec["sites"][single_key]
    default_key = "h200" if role == "dit" else "h100"
    if default_key not in spec["sites"]:
        raise ValueError(f"Missing site configuration for {role}")
    return spec["sites"][default_key]


def is_pai_role(spec, role):
    return "job_spec" in role_site(spec, role)


def service_endpoints(base_http, service_count=1):
    uri = urlsplit(base_http)
    return [
        f"{uri.scheme}://{uri.hostname}:{uri.port + offset}"
        for offset in range(service_count)
    ]


def reused_resource(run_id, model_id, role, entry):
    res = dict(
        run_id=run_id,
        generation=0,
        name="reused-" + role + "-" + entry["worker_id"],
        reused=True,
        mode="reuse",
        status="reused",
        service_index=entry.get("service_index", 0),
        service_count=entry.get("service_count", 1),
        pai=bool(entry.get("pai") or entry.get("job_id")),
        receipt=dict(
            schema=1,
            role=role,
            status="ready",
            model_id=model_id,
            worker_id=entry["worker_id"],
            http=entry["http"],
            gpu=entry.get("gpu", f"GPU-reused-{role}"),
        ),
    )
    if entry.get("job_id"):
        res["job_id"] = entry["job_id"]
    if entry.get("job_name"):
        res["job_name"] = entry["job_name"]
    return res


def validate_spec(spec, isolated=True, launch_module=launch):
    if spec.get("schema") != 1 or set(spec) - {
        "schema",
        "target",
        "roles",
        "sites",
        "model_id",
        "pool",
        "route",
        "existing_services",
    }:
        raise ValueError("Unsupported orchestration spec")
    check_target(spec["target"])
    target = spec["target"]
    existing = spec.get("existing_services", {})
    if not isinstance(existing, dict):
        raise ValueError("existing_services must be a dict")
    roles_dict = spec.get("roles", {})
    if not isinstance(roles_dict, dict):
        raise ValueError("roles must be a dict")
    # Normalize per-role mode="reuse" into existing_services if specified in roles.
    for role, cfg in list(roles_dict.items()):
        if not isinstance(cfg, dict):
            raise ValueError("Role config must be a dict")
        mode = cfg.get("mode")
        if mode not in (None, "create", "reuse"):
            raise ValueError(f"Unsupported role mode '{mode}' for {role}")
        if mode == "reuse":
            if role not in existing:
                existing = dict(existing)
                spec["existing_services"] = existing
                existing[role] = {
                    k: cfg[k]
                    for k in (
                        "http",
                        "worker_id",
                        "service_index",
                        "service_count",
                        "gpu",
                        "boot_epoch",
                        "pai",
                        "job_id",
                        "job_name",
                    )
                    if k in cfg
                }
            roles_dict.pop(role, None)
    if set(existing) - set(ROLES):
        raise ValueError("Unknown existing service role")
    owned = owned_roles(spec)
    if (
        spec.get("model_id")
        not in getattr(launch_module, "MODEL_IDS", launch.MODEL_IDS)
        or not set(owned) <= set(roles_dict)
        or set(roles_dict) - set(ROLES)
        or (not existing and set(roles_dict) != set(ROLES))
    ):
        raise ValueError("Exactly three roles and a supported model are required")
    if (
        not isinstance(spec.get("sites"), dict)
        or not spec["sites"]
        or set(spec["sites"]) - {"h100", "h200", "b300"}
    ):
        raise ValueError("Unsupported site keys")
    ports = []
    h100_ip = target.get("h100_ip")
    for role, entry in existing.items():
        if (
            not isinstance(entry, dict)
            or set(entry)
            - {
                "http",
                "worker_id",
                "service_index",
                "service_count",
                "gpu",
                "boot_epoch",
                "pai",
                "job_id",
                "job_name",
            }
            or not isinstance(entry.get("http"), str)
            or not isinstance(entry.get("worker_id"), str)
            or not entry["worker_id"]
        ):
            raise ValueError("Invalid existing service specification")
        count = entry.get("service_count", 1)
        # A retry may keep its DiT in the creating run while the original run
        # owns E/D. Pin the single-GPU PAI identity; never claim its ownership.
        if role == "dit" and (
            entry.get("pai") is not True
            or not entry.get("job_id")
            or not entry.get("job_name")
            or count != 1
            or entry.get("service_index", 0) != 0
        ):
            raise ValueError("DiT reuse requires a pinned single-service PAI job")
        if ("job_id" in entry or "job_name" in entry) and (
            not entry.get("pai")
            or not isinstance(entry.get("job_id"), str)
            or not entry["job_id"]
            or (
                "job_name" in entry
                and (not isinstance(entry["job_name"], str) or not entry["job_name"])
            )
        ):
            raise ValueError(
                "Pinned PAI existing service requires pai=true and non-empty job_id"
            )
        idx = entry.get("service_index", 0)
        if (
            type(count) is not int
            or count < 1
            or type(idx) is not int
            or not 0 <= idx < count
        ):
            raise ValueError("Invalid existing service_count or service_index")
        validate_endpoint(
            entry["http"], role, pai=bool(entry.get("pai")), target=target
        )
        uri = urlsplit(entry["http"])
        if h100_ip and uri.hostname == h100_ip:
            base_port = uri.port - idx
            if base_port < 1024 or base_port + count - 1 > 65535:
                raise ValueError("Invalid existing service port range")
            ports.extend(range(base_port, base_port + count))
    topology_profile = target.get("topology_profile")
    workaround_env = (
        _load_role_launcher().resolve_topology_workaround(topology_profile)[
            "environment"
        ]
        if topology_profile
        else {}
    )
    observed_nccl = {}
    allocated_ports_fn = getattr(
        launch_module, "role_allocated_ports", launch.role_allocated_ports
    )
    allowed_role_keys = getattr(launch_module, "CONFIG_KEYS", None)
    for role in owned:
        c = spec["roles"][role]
        role_for_validation = {k: v for k, v in c.items() if k != "mode"}
        if isinstance(allowed_role_keys, set):
            if "profile" not in allowed_role_keys:
                role_for_validation.pop("profile", None)
            if (
                "service_count" not in allowed_role_keys
                and role_for_validation.get("service_count", 1) == 1
            ):
                role_for_validation.pop("service_count", None)
        if c.get("role") != role or c.get("model_id") != spec["model_id"]:
            raise ValueError("Mixed role or model identity")
        # Reuse the exact launcher validator without creating temporary config files.
        launch_module.validate_config(role_for_validation)
        env = c.setdefault("environment", {})
        if workaround_env:
            for k, v in workaround_env.items():
                if k in env and env[k] != v:
                    raise ValueError(
                        f"Topology profile '{topology_profile}' requires {k}={v}"
                    )
                env[k] = v
        for name in ("NCCL_MIN_NCHANNELS", "NCCL_MIN_CTAS"):
            if name in env:
                if name in observed_nccl and observed_nccl[name] != env[name]:
                    raise ValueError(
                        f"Cross-role {name} mismatch: {observed_nccl[name]} vs {env[name]}"
                    )
                observed_nccl[name] = env[name]
        if isolated and (
            c.get("worker_config") or env.get("SGLANG_ROLE_POOL_WORKER_CONFIG")
        ):
            raise ValueError(
                "Initial roles must be isolated; use attach after selftest"
            )
        site = role_site(spec, role)
        if site.get("model_id", spec["model_id"]) != spec["model_id"]:
            raise ValueError("Mixed site model identity")
        if not is_pai_role(spec, role):
            ports.extend(allocated_ports_fn(role_for_validation))
    if len(set(ports)) != len(ports):
        raise ValueError("H100 hostNetwork roles must use disjoint ports")
    return spec


def load_state(path, launch_module=None):
    path = Path(path)
    state = read(path)
    if state.get("schema") != 1 or not re.fullmatch(
        r"[0-9a-f]{32}", state.get("run_id", "")
    ):
        raise ValueError("Invalid orchestration state")
    check_target(state["target"])
    if launch_module is None:
        rb_tools = path.parent / ".rollback-tools"
        if (rb_tools / "launch.py").is_file() and (rb_tools / "source.py").is_file():
            launch_module = _load_rollback_launch_module(rb_tools)
        else:
            launch_module = launch
    validate_spec(
        copy.deepcopy(state["spec"]), isolated=False, launch_module=launch_module
    )
    if state.get("generation", -1) < 0:
        raise ValueError("Invalid run generation")
    state.setdefault("deployment_id", f"deploy-{state['run_id'][:12]}")
    existing = state["spec"].get("existing_services", {})
    for role, resource in state.get("resources", {}).items():
        if role not in ROLES or resource.get("run_id") != state["run_id"]:
            raise ValueError("Resource ownership does not match this run")
        if resource.get("reused") is True:
            if (
                role not in existing
                or resource.get("receipt", {}).get("worker_id")
                != existing[role]["worker_id"]
                or resource.get("receipt", {}).get("http") != existing[role]["http"]
            ):
                raise ValueError("Reused resource identity does not match spec")
            continue
        if type(resource.get("generation")) is not int or resource["generation"] < 0:
            raise ValueError("Resource ownership does not match this run")
        if resource["name"] != "sglang-" + role + "-" + state["run_id"][
            :12
        ] + "-" + str(resource["generation"]):
            raise ValueError("Resource name does not match this run")
        if not is_pai_role(state["spec"], role) and not re.fullmatch(
            r"[a-z0-9][a-z0-9-]*", resource["namespace"]
        ):
            raise ValueError("Invalid resource namespace")
    return state


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_OSS_FATAL_ERROR_MARKERS = (
    "nosuchbucket",
    "accessdenied",
    "invalidaccesskeyid",
    "signaturedoesnotmatch",
    "securitytokenexpired",
    "requesttimetooskewed",
    "slowdown",
    "serviceunavailable",
    "internalerror",
    "requesttimeout",
    "badgateway",
    "gatewaytimeout",
    "connection",
    "timed out",
    "timeout",
    "dial tcp",
    "tls handshake",
    "invalid usage",
    "statuscode=401",
    "statuscode=403",
    "statuscode=429",
    "statuscode=500",
    "statuscode=502",
    "statuscode=503",
    "statuscode=504",
)
_OSS_NOT_FOUND_MARKERS = (
    "errorcode=nosuchkey",
    "errorcode: nosuchkey",
    '"nosuchkey"',
    "nosuchkey",
    "objectnotexist",
    "the specified key does not exist",
    "statuscode=404",
)


def _strip_ansi(text):
    return _ANSI_ESCAPE_RE.sub("", text or "")


def _format_oss_cli_error(proc):
    if proc is None:
        return "subprocess returned None"
    out = _strip_ansi(getattr(proc, "stdout", "") or "").strip()
    err = _strip_ansi(getattr(proc, "stderr", "") or "").strip()
    parts = [p for p in (err, out) if p]
    return " | ".join(parts) if parts else "unknown OSS CLI failure"


def _is_oss_not_found(returncode, stdout="", stderr=""):
    if returncode == 0:
        return False
    combined = _strip_ansi(f"{stdout or ''}\n{stderr or ''}").lower()
    if any(marker in combined for marker in _OSS_FATAL_ERROR_MARKERS):
        return False
    return any(marker in combined for marker in _OSS_NOT_FOUND_MARKERS)


class Cloud:
    """Explicit target cloud adapter; all remote mutations pass through explicit calls."""

    def __init__(self, timeout=1200, target=None, aliyun_profile=None):
        self.timeout = timeout
        self.target = copy.deepcopy(target or TARGET)
        self.aliyun_profile = (
            aliyun_profile
            if aliyun_profile is not None
            else os.environ.get("ALIYUN_PROFILE")
        )
        self.ua = (
            "AlibabaCloud-Agent-Skills/alibabacloud-pai-dlc-job/" + uuid.uuid4().hex
        )

    def _require_aliyun_profile(self):
        profile = (self.aliyun_profile or "").strip()
        if not profile:
            raise ValueError(
                "An explicit Aliyun CLI profile is required (--aliyun-profile or ALIYUN_PROFILE)"
            )
        return profile

    def cli(self, product, operation, **args):
        profile = self._require_aliyun_profile()
        region = self.target.get("region", "cn-shanghai")
        argv = [
            "aliyun",
            product,
            operation,
            "--profile",
            profile,
            "--region",
            region,
        ]
        if product == "pai-dlc":
            argv += ["--user-agent", self.ua]
        for key, value in args.items():
            argv += [
                "--" + key,
                json.dumps(value) if isinstance(value, (dict, list)) else str(value),
            ]
        try:
            output = subprocess.check_output(
                argv, text=True, timeout=60, stderr=subprocess.PIPE
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            # CommandContent may carry short-lived credentials for the VPC relay.
            raise RuntimeError(
                "Cloud CLI failed: " + product + " " + operation
            ) from None
        return json.loads(output)

    def eflo(self, script, submitted=None):
        node_id = self.target.get("node_id")
        if not node_id:
            raise ValueError("Target has no node_id for Eflo commands")
        request = dict(
            NodeIdList=[node_id],
            Name="sglang-observe-" + uuid.uuid4().hex[:12],
            RepeatMode="Once",
            Timeout=180,
            ContentEncoding="Base64",
            CommandContent=base64.b64encode(script.encode()).decode(),
        )
        reply = self.cli("eflo-controller", "RunCommand", **request)
        invoke = reply["InvokeId"]
        if submitted:
            submitted(invoke)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            result = self.cli(
                "eflo-controller",
                "DescribeInvocations",
                InvokeId=invoke,
                IncludeOutput="true",
                ContentEncoding="PlainText",
            )
            rows = result.get("Invocations", {}).get("Invocation", [])
            for invocation in rows:
                if invocation.get("InvokeId") != invoke:
                    raise RuntimeError("Eflo invocation identity mismatch")
                nodes = invocation.get("InvokeNodes", {}).get("InvokeNode", [])
                for row in nodes:
                    if row.get("NodeId") != node_id:
                        raise RuntimeError("Eflo returned a different node")
                    status = row.get("InvocationStatus")
                    if status == "Success":
                        if row.get("ExitCode") != 0 or row.get("Dropped", 0):
                            tail = str(row.get("Output") or "").strip()[-600:]
                            detail = f": {tail}" if tail else ""
                            raise RuntimeError(
                                "Eflo node command failed or output was truncated"
                                + detail
                            )
                        return row["Output"]
                    if status in {
                        "Failed",
                        "Timeout",
                        "Cancelled",
                        "Error",
                        "Aborted",
                        "Invalid",
                        "Terminated",
                    }:
                        tail = str(row.get("Output") or "").strip()[-600:]
                        detail = f": {tail}" if tail else ""
                        raise RuntimeError("Eflo invocation failed: " + status + detail)
            time.sleep(2)
        raise TimeoutError("Eflo invocation is UNKNOWN; reconcile InvokeId " + invoke)

    def upload_source(self, local_path, uri):
        profile = self._require_aliyun_profile()
        region = resolve_environment_region(self.target)
        subprocess.run(
            [
                "aliyun",
                "--profile",
                profile,
                "--endpoint",
                f"oss-{region}.aliyuncs.com",
                "oss",
                "cp",
                "--force",
                str(local_path),
                uri,
            ],
            check=True,
            timeout=300,
        )

    def oss_history_prefix(self):
        return resolve_oss_history_prefix(self.target)

    def download_oss_optional(self, uri, local_path):
        """Download an OSS object if it exists.

        Returns False ONLY on explicit OSS object NotFound (404 NoSuchKey / ObjectNotExist).
        Raises RuntimeError on auth, bucket, network, throttling, timeout, or unknown CLI errors.
        """
        profile = self._require_aliyun_profile()
        region = resolve_environment_region(self.target)
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run(
                [
                    "aliyun",
                    "--profile",
                    profile,
                    "--endpoint",
                    f"oss-{region}.aliyuncs.com",
                    "oss",
                    "cp",
                    "--force",
                    uri,
                    str(local_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            local_path.unlink(missing_ok=True)
            raise RuntimeError(f"OSS download failed for {uri}: {exc}") from exc
        if proc is None:
            local_path.unlink(missing_ok=True)
            raise RuntimeError(f"OSS download returned no process status for {uri}")
        rc = getattr(proc, "returncode", 1)
        stdout = getattr(proc, "stdout", "") or ""
        stderr = getattr(proc, "stderr", "") or ""
        combined_lower = _strip_ansi(f"{stdout}\n{stderr}").lower()
        if rc == 0:
            if "finishwitherror" in combined_lower or not local_path.is_file():
                local_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"OSS download incomplete for {uri}: {_format_oss_cli_error(proc)}"
                )
            return True
        local_path.unlink(missing_ok=True)
        if _is_oss_not_found(rc, stdout, stderr):
            return False
        raise RuntimeError(
            f"OSS download failed for {uri} (rc={rc}): {_format_oss_cli_error(proc)}"
        )

    def download_oss_required(self, uri, local_path, *, expected_sha256=None):
        local_path = Path(local_path)
        if not self.download_oss_optional(uri, local_path):
            raise FileNotFoundError(f"Required OSS object not found: {uri}")
        if expected_sha256 is not None and digest(local_path) != expected_sha256:
            local_path.unlink(missing_ok=True)
            raise ValueError(
                f"OSS object SHA256 mismatch for {uri} (expected {expected_sha256})"
            )
        return local_path

    def list_oss_objects(self, uri_prefix):
        """List object URIs under `uri_prefix` (fail-closed on any non-NotFound error)."""
        profile = self._require_aliyun_profile()
        region = resolve_environment_region(self.target)
        normalized_prefix = uri_prefix.rstrip("/") + "/"
        try:
            proc = subprocess.run(
                [
                    "aliyun",
                    "--profile",
                    profile,
                    "--endpoint",
                    f"oss-{region}.aliyuncs.com",
                    "oss",
                    "ls",
                    normalized_prefix,
                    "--short-format",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise RuntimeError(
                f"OSS list failed for {normalized_prefix}: {exc}"
            ) from exc
        if proc is None:
            raise RuntimeError(
                f"OSS list returned no process status for {normalized_prefix}"
            )
        rc = getattr(proc, "returncode", 1)
        stdout = getattr(proc, "stdout", "") or ""
        stderr = getattr(proc, "stderr", "") or ""
        combined_lower = _strip_ansi(f"{stdout}\n{stderr}").lower()
        if rc != 0:
            if _is_oss_not_found(rc, stdout, stderr):
                return []
            raise RuntimeError(
                f"OSS list failed for {normalized_prefix} (rc={rc}): {_format_oss_cli_error(proc)}"
            )
        if (
            "finishwitherror" in combined_lower
            or "error:" in _strip_ansi(stderr).lower()
        ):
            raise RuntimeError(
                f"OSS list reported errors for {normalized_prefix}: {_format_oss_cli_error(proc)}"
            )
        clean_out = _strip_ansi(stdout)
        uris = []
        for raw_line in clean_out.splitlines():
            line = raw_line.strip()
            if line.startswith(normalized_prefix) and not line.endswith("/"):
                uris.append(line)
        if not uris and "object number is:" not in clean_out.lower():
            raise RuntimeError(
                f"Unexpected OSS list output for {normalized_prefix}: {_format_oss_cli_error(proc)}"
            )
        return sorted(set(uris))

    def sync_history_artifacts(self, history_dir, deployment_id, summary_dir=None):
        """Archive immutable per-deployment records and index shards to OSS.

        Write order (record before index shard) ensures any reader observing an index
        shard can immediately fetch its content-addressed record without torn reads,
        and concurrent writers never overwrite a shared mutable `index.json`.
        """
        prefix = self.oss_history_prefix().rstrip("/")
        history_dir = Path(history_dir)
        record_file = history_dir / f"{deployment_id}.json"
        if not record_file.is_file():
            raise FileNotFoundError(
                f"Local history record '{record_file}' not found for OSS sync"
            )
        record = read(record_file)
        record_sha256 = digest(record_file)
        record_key = f"records/{deployment_id}/{record_sha256}.json"
        uploaded = []

        # 1. Immutable content-addressed deployment record snapshot.
        immutable_record_uri = f"{prefix}/{record_key}"
        self.upload_source(record_file, immutable_record_uri)
        uploaded.append(immutable_record_uri)

        # 2. Direct per-deployment record path for human inspection & compatibility.
        direct_record_uri = f"{prefix}/{deployment_id}.json"
        self.upload_source(record_file, direct_record_uri)
        uploaded.append(direct_record_uri)

        # 3. Human/operator deployment credentials & verification summary files.
        if summary_dir is not None:
            summary_dir = Path(summary_dir)
            for name in (
                "deployment.json",
                "DEPLOYMENT.md",
                "traffic_verification.json",
            ):
                local_file = summary_dir / name
                if local_file.is_file():
                    uri = f"{prefix}/{deployment_id}/{name}"
                    self.upload_source(local_file, uri)
                    uploaded.append(uri)

        # 4. Commit point: upload immutable per-entry index shard (never overwrite shared index.json).
        entry = build_history_index_entry(
            record, record_sha256=record_sha256, record_key=record_key
        )
        with tempfile.TemporaryDirectory() as td:
            shard_tmp = Path(td) / "entry.json"
            save(shard_tmp, entry)
            entry_sha256 = digest(shard_tmp)
            entry_uri = f"{prefix}/entries/{deployment_id}/{entry_sha256}.json"
            self.upload_source(shard_tmp, entry_uri)
            uploaded.append(entry_uri)

        return uploaded

    def _download_and_parse_index_shards(self, shard_uris, scratch_dir):
        if not shard_uris:
            return []

        def _fetch_one(idx_and_uri):
            idx, uri = idx_and_uri
            stem = uri.rsplit("/", 1)[-1]
            expected_sha = (
                stem[:-5]
                if stem.endswith(".json") and re.fullmatch(r"[0-9a-f]{64}", stem[:-5])
                else None
            )
            local_shard = scratch_dir / f"shard-{idx}.json"
            self.download_oss_required(uri, local_shard, expected_sha256=expected_sha)
            entry = read(local_shard)
            if not isinstance(entry, dict) or not entry.get("deployment_id"):
                raise ValueError(f"Corrupted deployment index shard at {uri}")
            return entry

        max_workers = min(8, len(shard_uris))
        if max_workers <= 1:
            return [_fetch_one((0, shard_uris[0]))]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(_fetch_one, enumerate(shard_uris)))

    def _fetch_and_merge_oss_index_entries(self, prefix):
        shard_uris = [
            u
            for u in self.list_oss_objects(f"{prefix}/entries/")
            if u.endswith(".json")
        ]
        collected = []
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            legacy_index_path = td_path / "legacy_index.json"
            if self.download_oss_optional(f"{prefix}/index.json", legacy_index_path):
                legacy_payload = read(legacy_index_path)
                if not isinstance(legacy_payload, dict) or not isinstance(
                    legacy_payload.get("deployments"), list
                ):
                    raise ValueError(
                        f"Corrupted legacy deployment index at {prefix}/index.json"
                    )
                collected.extend(legacy_payload["deployments"])
            collected.extend(self._download_and_parse_index_shards(shard_uris, td_path))
        return merge_history_index_entries(collected)

    def _fetch_single_oss_deployment_record(self, prefix, dep_id, dest_dir):
        dest_path = dest_dir / f"{dep_id}.json"
        dep_shards = [
            u
            for u in self.list_oss_objects(f"{prefix}/entries/{dep_id}/")
            if u.endswith(".json")
        ]
        if dep_shards:
            with tempfile.TemporaryDirectory() as td:
                entries = merge_history_index_entries(
                    self._download_and_parse_index_shards(dep_shards, Path(td))
                )
            if entries:
                latest = entries[-1]
                record_key = latest.get("record_key")
                if record_key:
                    self.download_oss_required(
                        f"{prefix}/{record_key.lstrip('/')}",
                        dest_path,
                        expected_sha256=latest.get("record_sha256"),
                    )
                    return
        self.download_oss_optional(f"{prefix}/{dep_id}.json", dest_path)

    def _fetch_indexed_oss_deployment_record(self, prefix, entry, dest_dir):
        dep_id = entry["deployment_id"]
        dest_path = dest_dir / f"{dep_id}.json"
        record_key = entry.get("record_key")
        if record_key:
            uri = f"{prefix}/{record_key.lstrip('/')}"
            if self.download_oss_optional(uri, dest_path):
                expected_sha = entry.get("record_sha256")
                if expected_sha and digest(dest_path) != expected_sha:
                    dest_path.unlink(missing_ok=True)
                    raise ValueError(
                        f"OSS deployment record SHA256 mismatch for {uri} (expected {expected_sha})"
                    )
            return
        self.download_oss_optional(f"{prefix}/{dep_id}.json", dest_path)

    def fetch_oss_history_store(self, dest_dir, deployment_ids=None):
        prefix = self.oss_history_prefix().rstrip("/")
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        if deployment_ids is not None and len(deployment_ids) > 0:
            for dep_id in dict.fromkeys(d for d in deployment_ids if d):
                self._fetch_single_oss_deployment_record(prefix, dep_id, dest_dir)
            return dest_dir

        merged_entries = self._fetch_and_merge_oss_index_entries(prefix)
        save(dest_dir / "index.json", {"schema": 1, "deployments": merged_entries})

        if deployment_ids is None and merged_entries:

            def _fetch_rec(entry):
                self._fetch_indexed_oss_deployment_record(prefix, entry, dest_dir)

            max_workers = min(8, len(merged_entries))
            if max_workers <= 1:
                _fetch_rec(merged_entries[0])
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    list(pool.map(_fetch_rec, merged_entries))
        return dest_dir

    def stage_h100_source(self, manifest):
        profile = self._require_aliyun_profile()
        region = self.target.get("region", "cn-shanghai")
        # Signing URLs are short-lived transport credentials: never persist or print them.
        signed = subprocess.check_output(
            [
                "aliyun",
                "--profile",
                profile,
                "--endpoint",
                f"oss-{region}-internal.aliyuncs.com",
                "oss",
                "sign",
                manifest["uri"],
                "--timeout",
                "3600",
            ],
            text=True,
            timeout=60,
        )
        urls = [
            line.strip()
            for line in signed.splitlines()
            if line.strip().startswith(("https://", "http://"))
        ]
        if len(urls) != 1:
            raise ValueError("OSS signing did not return exactly one URL")
        checksum = manifest["sha256"]
        from source import sha

        sha(checksum)
        cache = "/var/lib/sglang-orchestrate/cache/" + checksum
        archive = cache + "/source.tar"
        script = (
            """set -eu
mkdir -p CACHE
exec 9>CACHE/.stage.lock
flock 9
if [ -f ARCHIVE ] && printf '%s  %s\n' SHA ARCHIVE | sha256sum -c - >/dev/null 2>&1; then exit 0; fi
TMP=$(mktemp CACHE/source.tar.XXXXXX)
trap 'rm -f "$TMP"' EXIT
curl -fsSL -o "$TMP" -- URL
printf '%s  %s\n' SHA "$TMP" | sha256sum -c - >/dev/null
mv "$TMP" ARCHIVE
""".replace("CACHE", shlex.quote(cache))
            .replace("ARCHIVE", shlex.quote(archive))
            .replace("SHA", shlex.quote(checksum))
            .replace("URL", shlex.quote(urls[0]))
        )
        try:
            self.eflo(script)
        except Exception:
            # CLI exception argv can contain CommandContent, including the signed URL.
            raise RuntimeError(
                "H100 source staging failed; no Pods were submitted"
            ) from None

    def submit_job(self, body):
        profile = self._require_aliyun_profile()
        region = self.target.get("region", "cn-shanghai")
        argv = [
            "aliyun",
            "pai-dlc",
            "POST",
            "/api/v1/jobs",
            "--profile",
            profile,
            "--region",
            region,
            "--user-agent",
            self.ua,
            "--header",
            "Content-Type=application/json",
            "--body",
            json.dumps(body),
        ]
        return json.loads(subprocess.check_output(argv, text=True, timeout=60))["JobId"]

    def get_job(self, job_id):
        return self.cli("pai-dlc", "GetJob", JobId=job_id)

    def receipt(self, resource, role):
        is_pai = (
            "job_id" in resource
            or "expected_job" in resource
            or bool(resource.get("pai"))
            or role == "dit"
        )
        pod_id = None
        if not is_pai:
            pod = self.pod(resource)
            if pod["metadata"]["uid"] != resource["uid"]:
                raise ValueError("Pod UID changed")
            extract = (
                """import sys,json,subprocess
last={}
for line in sys.stdin:
 try: value=json.loads(line[line.find('{'):])
 except ValueError: continue
 if isinstance(value,dict) and value.get('schema')==1 and value.get('role')==ROLE and 'archive_sha256' in value: last=value
if last and 'argv' not in last:
 try:
  raw=subprocess.check_output(['kubectl','-n',NS,'exec',POD,'--','sh','-c','cat /tmp/sglang-release-*/receipt.json 2>/dev/null || true'],universal_newlines=True,timeout=10).strip()
  if raw:
   full=json.loads(raw)
   if isinstance(full,dict) and full.get('schema')==1 and full.get('role')==ROLE and full.get('archive_sha256')==last.get('archive_sha256') and full.get('worker_id')==last.get('worker_id'):
    last={**full,**last,'argv':full.get('argv')}
 except Exception: pass
print(json.dumps(last))
""".replace("ROLE", repr(role))
                .replace("NS", repr(resource["namespace"]))
                .replace("POD", repr(resource["name"]))
            )
            text = self.eflo(
                "kubectl -n "
                + shlex.quote(resource["namespace"])
                + " logs "
                + shlex.quote(resource["name"])
                + " --tail=1000 | python3 -c "
                + shlex.quote(extract)
            )
        else:
            job = self.get_job(resource["job_id"])
            verify_job(job, resource, target=self.target)
            pods = [
                p
                for p in job.get("Pods", [])
                if "aimaster" not in p.get("PodId", "").lower()
            ]
            if len(pods) != 1:
                raise RuntimeError(f"Expected exactly one {role} Worker Pod")
            pod_id = pods[0]["PodId"]
            logs = self.cli(
                "pai-dlc",
                "GetPodLogs",
                JobId=resource["job_id"],
                PodId=pod_id,
                MaxLines=100,
            )
            if logs.get("JobId") != resource["job_id"]:
                raise ValueError("PAI log identity mismatch")
            text = "\n".join(logs["Logs"])
        try:
            parsed = last_receipt(text, role)
        except RuntimeError:
            if is_pai and job.get("Status") in (
                "Failed",
                "Stopped",
                "Succeeded",
            ):
                raise ValueError(
                    "Owned PAI job terminated before ready receipt"
                ) from None
            raise
        if not is_pai and "argv" not in parsed:
            try:
                raw_disk = self.eflo(
                    "kubectl -n "
                    + shlex.quote(resource["namespace"])
                    + " exec "
                    + shlex.quote(resource["name"])
                    + " -- sh -c 'cat /tmp/sglang-release-*/receipt.json 2>/dev/null || true'"
                ).strip()
                if raw_disk:
                    full_disk = json.loads(raw_disk)
                    if (
                        isinstance(full_disk, dict)
                        and full_disk.get("schema") == 1
                        and full_disk.get("role") == role
                        and full_disk.get("archive_sha256")
                        == parsed.get("archive_sha256")
                        and full_disk.get("worker_id") == parsed.get("worker_id")
                        and isinstance(full_disk.get("argv"), list)
                        and full_disk["argv"]
                    ):
                        parsed = {**full_disk, **parsed, "argv": full_disk["argv"]}
            except Exception:
                pass
        if is_pai and "argv" not in parsed:
            for line in text.splitlines():
                cand = line[line.find("{") :]
                try:
                    full_line = json.loads(cand)
                except ValueError:
                    continue
                if (
                    isinstance(full_line, dict)
                    and full_line.get("schema") == 1
                    and full_line.get("role") == role
                    and full_line.get("archive_sha256") == parsed.get("archive_sha256")
                    and full_line.get("worker_id") == parsed.get("worker_id")
                    and isinstance(full_line.get("argv"), list)
                    and full_line["argv"]
                ):
                    parsed = {**full_line, **parsed, "argv": full_line["argv"]}
        return parsed

    def pod(self, resource):
        return json.loads(
            self.eflo(
                "kubectl -n "
                + shlex.quote(resource["namespace"])
                + " get pod "
                + shlex.quote(resource["name"])
                + " -o json"
            )
        )


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise ValueError("Cross-environment HTTP redirects are forbidden")


def vpc_url(url):
    try:
        return ipaddress.ip_address(
            urlsplit(url).hostname or ""
        ) in ipaddress.ip_network("10.0.0.0/8")
    except ValueError:
        return False


def http(url, payload=None, token=None, cloud=None):
    headers = {
        "Content-Type": "application/json",
        **({"Authorization": "Bearer " + token} if token else {}),
    }
    request = Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers=headers,
    )
    try:
        with build_opener(ProxyHandler({}), NoRedirect()).open(
            request, timeout=5
        ) as response:
            return json.load(response)
    except HTTPError:
        # An HTTP response proves connectivity; never replay an application rejection.
        raise
    except OSError:
        if cloud is None or not vpc_url(url):
            raise
    # The Eflo host runs Python 3.6. Keep syntax and urllib behavior compatible,
    # and propagate HTTP application status codes (e.g. 403 on managed endpoints) cleanly.
    parameters = json.dumps(dict(url=url, payload=payload, headers=headers))
    code = """import json,urllib.error,urllib.request
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,request,fp,code,message,headers,new_url):
  raise ValueError('Cross-environment HTTP redirects are forbidden')
p=json.loads(PARAMETERS)
data=None if p['payload'] is None else json.dumps(p['payload']).encode()
r=urllib.request.Request(p['url'],data=data,headers=p['headers'])
try:
 with urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect()).open(r,timeout=60) as response:
  print(json.dumps(json.load(response)))
except urllib.error.HTTPError as e:
 body=''
 try: body=e.read().decode('utf-8','replace')
 except Exception: pass
 print(json.dumps({'__eflo_http_error__':int(e.code),'reason':str(e.reason),'body':body}))
""".replace("PARAMETERS", repr(parameters))
    relayed = json.loads(cloud.eflo("python3 -c " + shlex.quote(code)))
    if isinstance(relayed, dict) and "__eflo_http_error__" in relayed:
        raise HTTPError(
            url,
            int(relayed["__eflo_http_error__"]),
            str(relayed.get("reason", "HTTP Error")),
            {},
            io.BytesIO(str(relayed.get("body", "")).encode("utf-8")),
        )
    return relayed


def last_receipt(text, role):
    receipts = []
    if not isinstance(text, str):
        raise ValueError("Expected text worker logs")
    for line in text.splitlines():
        candidate = line[line.find("{") :]
        try:
            value = json.loads(candidate)
        except ValueError:
            marker = ',"runtime":{'
            if marker not in candidate:
                continue
            try:
                value = json.loads(candidate[: candidate.index(marker)] + "}")
            except ValueError:
                continue
            if "gpu" not in value:
                tail = candidate[candidate.index(marker) :]
                gpu_matches = []
                for raw_match in re.findall(r'"gpu"\s*:\s*"((?:\\.|[^"\\])*)"', tail):
                    try:
                        gpu_matches.append(json.loads('"' + raw_match + '"'))
                    except ValueError:
                        continue
                with_uuid = [
                    item
                    for item in gpu_matches
                    if re.search(r"GPU-[A-Za-z0-9-]+", item)
                ]
                if with_uuid:
                    value["gpu"] = with_uuid[-1]
                elif gpu_matches:
                    value["gpu"] = gpu_matches[-1]
            if "gpu" not in value and isinstance(value.get("worker_config"), dict):
                value["gpu"] = ", ".join(value["worker_config"].get("hardware_ids", []))
        if (
            isinstance(value, dict)
            and value.get("schema") == 1
            and value.get("role") == role
            and "archive_sha256" in value
        ):
            receipts.append(value)
    if not receipts:
        raise RuntimeError("Runtime receipt not available")
    return receipts[-1]


def contains_expected(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k in actual and contains_expected(actual[k], v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(contains_expected(a, e) for a, e in zip(actual, expected))
        )
    return actual == expected


def verify_job(job, resource, target=None):
    t = target or TARGET
    expected = resource.get("expected_job") or {}
    workspace_id = expected.get("WorkspaceId", t.get("workspace_id"))
    quota_id = expected.get("ResourceId", t.get("quota_id"))
    if (
        str(job.get("WorkspaceId")) != str(workspace_id)
        or job.get("ResourceId") != quota_id
        or job.get("DisplayName") != resource["name"]
        or job.get("JobId") != resource["job_id"]
    ):
        raise ValueError("PAI job identity does not match this run")
    if expected and not contains_expected(job, expected):
        raise ValueError("PAI source command, mounts or resource spec changed")
    specs = job.get("JobSpecs", [])
    if (
        len(specs) != 1
        or specs[0].get("PodCount") != 1
        or str(specs[0].get("ResourceConfig", {}).get("GPU")) != "1"
    ):
        raise ValueError("PAI job is not the owned single-GPU topology")


def probe(spec, output, cloud):
    """Read-only host inventory and paginated PAI metadata; never reserve GPUs."""
    spec = expand_minimal_spec(spec)
    target = spec["target"]
    output.mkdir(parents=True, exist_ok=False)
    h100_roles = [r for r in owned_roles(spec) if not is_pai_role(spec, r)]
    # Only the owned H100 assets are observed on H100; PAI mounts remain PAI evidence.
    models = sorted({spec["roles"][r]["model"] for r in h100_roles})
    code = """import json,subprocess,os,socket
def r(a):
 if a[0]=='crictl': a=a[:1]+['--config','/dev/null']+a[1:]
 return subprocess.check_output(a,universal_newlines=True)
def containers(items):
 return [dict(image=c.get('image',''),env=[{k:e[k] for k in ('name','value') if k in e} for e in c.get('env',[]) if e['name']=='NVIDIA_VISIBLE_DEVICES']) for c in items]
def statuses(items):
 return [dict(imageID=c.get('imageID','')) for c in items]
models=MODELS
ports=PORTS
busy=[]
for port in ports:
 s=socket.socket()
 try: s.bind(('0.0.0.0',port))
 except OSError: busy.append(port)
 finally: s.close()
pods=json.loads(r(['kubectl','get','pods','-A','-o','json']))
pods={'items':[dict(spec=dict(nodeName=p.get('spec',{}).get('nodeName'),runtimeClassName=p.get('spec',{}).get('runtimeClassName'),containers=containers(p.get('spec',{}).get('containers',[])),initContainers=containers(p.get('spec',{}).get('initContainers',[]))),status=dict(phase=p.get('status',{}).get('phase'),containerStatuses=statuses(p.get('status',{}).get('containerStatuses',[])),initContainerStatuses=statuses(p.get('status',{}).get('initContainerStatuses',[])))) for p in pods['items'] if p.get('status',{}).get('phase') not in ('Succeeded','Failed')]}
images=json.loads(r(['crictl','-r','unix:///run/k3s/containerd/containerd.sock','images','-o','json']))
images={'images':[dict(repoDigests=i.get('repoDigests',[])) for i in images.get('images',[])]}
print(json.dumps(dict(gpus=r(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used','--format=csv,noheader,nounits']),processes=r(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader']),pods=pods,images=images,models={p:os.path.isdir(p) for p in models},busy_ports=busy),separators=(',',':')))
""".replace("MODELS", repr(models)).replace(
        "PORTS",
        repr(
            sorted(
                {
                    p
                    for r in h100_roles
                    for p in launch.role_allocated_ports(
                        {k: v for k, v in spec["roles"][r].items() if k != "mode"}
                    )
                }
            )
        ),
    )

    def list_pai_jobs():
        items, page = [], 1
        while True:
            reply = cloud.cli(
                "pai-dlc",
                "ListJobs",
                WorkspaceId=target["workspace_id"],
                ShowOwn="false",
                PageNumber=page,
                PageSize=100,
            )
            items.extend(reply["Jobs"])
            if len(items) >= reply["TotalCount"]:
                break
            if not reply["Jobs"]:
                raise RuntimeError("PAI pagination did not advance")
            page += 1
        return items

    if target.get("node_id"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            eflo_future = pool.submit(cloud.eflo, "python3 -c " + shlex.quote(code))
            pai_future = pool.submit(list_pai_jobs)
            inventory = json.loads(eflo_future.result())
            jobs = pai_future.result()
    else:
        inventory = dict(
            gpus="",
            processes="",
            pods={"items": []},
            images={"images": []},
            models={},
            busy_ports=[],
        )
        jobs = list_pai_jobs()
    gpus = []
    for row in inventory["gpus"].splitlines():
        if not row.strip():
            continue
        device, name, total, used = (v.strip() for v in row.split(","))
        gpus.append(
            dict(uuid=device, memory=int(used), name=name, total_memory_mib=int(total))
        )
    import select_gpu

    free = []
    while gpus:
        try:
            free.append(
                select_gpu.select(
                    gpus, inventory["processes"].splitlines() + free, inventory["pods"]
                )
            )
        except ValueError as error:
            if str(error) != "No unclaimed idle GPU is available":
                raise
            break
    pai_roles = [r for r in owned_roles(spec) if is_pai_role(spec, r)]
    quota_jobs = [j for j in jobs if j.get("ResourceId") == target["quota_id"]]
    active_Statuses = {"Running", "Launching", "Starting", "Creating"}
    queuing_jobs = [
        j.get("JobId") for j in quota_jobs if j.get("Status") in ("Queuing", "Pending")
    ]
    running_gpus = 0
    running_cpus = 0
    observed_pai_images = set()
    for j in quota_jobs:
        if j.get("Status") not in active_Statuses and j.get("Status") not in (
            "Queuing",
            "Pending",
        ):
            continue
        detail = None
        if isinstance(j.get("JobSpecs"), list):
            detail = j
        else:
            try:
                fetched = cloud.get_job(j["JobId"])
                if isinstance(fetched, dict) and isinstance(
                    fetched.get("JobSpecs"), list
                ):
                    detail = fetched
            except Exception:
                detail = None
        if detail:
            for js in detail.get("JobSpecs", []):
                pods_n = int(js.get("PodCount", 1) or 1)
                rc = js.get("ResourceConfig") or {}
                if j.get("Status") in active_Statuses:
                    try:
                        running_gpus += pods_n * int(rc.get("GPU", 0) or 0)
                    except (TypeError, ValueError):
                        pass
                    try:
                        running_cpus += pods_n * int(rc.get("CPU", 0) or 0)
                    except (TypeError, ValueError):
                        pass
                if js.get("Image"):
                    observed_pai_images.add(js["Image"])
        elif j.get("Status") in active_Statuses:
            running_gpus += 1
    req_pai_gpus = len(pai_roles)
    req_pai_cpus = 0
    for r in pai_roles:
        rc = role_site(spec, r).get("job_spec", {}).get("ResourceConfig", {})
        try:
            req_pai_cpus += int(rc.get("CPU", 16) or 16)
        except (TypeError, ValueError):
            req_pai_cpus += 16
    evidence = dict(
        schema=1,
        observed_at=now(),
        target=target,
        h100=inventory,
        idle_gpu_uuids=free,
        h200_jobs=[
            {
                k: j.get(k)
                for k in ("JobId", "DisplayName", "Status", "UserId", "ResourceId")
            }
            for j in quota_jobs
        ],
        h200_inventory={
            "quota_id": target["quota_id"],
            "running_gpus": running_gpus,
            "running_cpus": running_cpus,
            "queuing_jobs": queuing_jobs,
            "requested_pai_roles": pai_roles,
            "requested_pai_gpus": req_pai_gpus,
            "requested_pai_cpus": req_pai_cpus,
            "observed_images": sorted(observed_pai_images),
        },
    )
    for role in owned_roles(spec):
        save(output / (role + ".json"), spec["roles"][role])
    for site in sorted(spec["sites"]):
        save(output / (site + "-site.json"), spec["sites"][site])
    evidence["blockers"] = []
    for r, entry in spec.get("existing_services", {}).items():
        if entry.get("job_id"):
            try:
                reused_job = cloud.get_job(entry["job_id"])
            except Exception as exc:
                evidence["blockers"].append(
                    f"Reused PAI {r} job {entry['job_id']} query failed: {type(exc).__name__}"
                )
                continue
            if isinstance(reused_job, dict) and "Status" in reused_job:
                if reused_job.get("Status") != "Running" or (
                    entry.get("job_name")
                    and reused_job.get("DisplayName") != entry["job_name"]
                ):
                    evidence["blockers"].append(
                        f"Reused PAI {r} job {entry['job_id']} is not Running with expected DisplayName"
                    )
    if queuing_jobs and pai_roles:
        evidence["blockers"].append(
            f"PAI quota {target['quota_id']} already has queuing jobs: {', '.join(queuing_jobs)}"
        )
    if (
        target.get("quota_id") == "quota3l8bnv9a35n"
        and pai_roles
        and running_gpus + req_pai_gpus > 8
    ):
        evidence["blockers"].append(
            f"PAI H200 quota {target['quota_id']} has {running_gpus}/8 GPUs in use; cannot fit {req_pai_gpus} new role(s)"
        )
    if (
        target.get("quota_id") == "quota3l8bnv9a35n"
        and pai_roles
        and running_cpus > 0
        and running_cpus + req_pai_cpus > 192
    ):
        evidence["blockers"].append(
            f"PAI H200 quota {target['quota_id']} CPU request ({running_cpus}+{req_pai_cpus} vCPU) risks CPU starvation queuing; lower ResourceConfig.CPU to 16"
        )
    if len(free) < len(h100_roles):
        evidence["blockers"].append(
            "Fewer than two unclaimed idle H100 GPUs"
            if len(h100_roles) == 2
            else f"Fewer than {len(h100_roles)} unclaimed idle H100 GPUs"
        )
    if h100_roles and inventory["busy_ports"]:
        evidence["blockers"].append("Configured H100 ports are occupied")
    if h100_roles and not all(inventory["models"].values()):
        evidence["blockers"].append("Configured H100 model directory is missing")
    images = {
        v
        for image in inventory["images"].get("images", [])
        for v in image.get("repoDigests", [])
    }
    for pod in inventory["pods"]["items"]:
        for container in pod.get("spec", {}).get("containers", []) + pod.get(
            "spec", {}
        ).get("initContainers", []):
            images.add(container.get("image", ""))
        for container in pod.get("status", {}).get("containerStatuses", []) + pod.get(
            "status", {}
        ).get("initContainerStatuses", []):
            images.add(container.get("imageID", ""))
    images = {image.split("://", 1)[-1] for image in images}
    inventory["observed_images"] = sorted(images - {""})
    if h100_roles and "h100" in spec["sites"]:
        pinned = spec["sites"]["h100"]["image"]
        if pinned not in images and pinned.rsplit("@", 1)[-1] not in {
            image.rsplit("@", 1)[-1] for image in images
        }:
            evidence["blockers"].append("Pinned H100 image digest is not present")
    evidence["configuration_status"] = (
        "template-with-blockers"
        if evidence["blockers"]
        else (
            "h200-quota-matched; h200-assets-unverified"
            if not h100_roles
            else "h100-inventory-matched; h200-assets-unverified"
        )
    )
    # Retain GPU allocation metadata, never arbitrary Pod environment or credentials.
    for pod in evidence["h100"]["pods"]["items"]:
        pod.pop("metadata", None)
        pod.pop("status", None)
        pod["spec"] = {
            "nodeName": pod.get("spec", {}).get("nodeName"),
            "containers": [
                {
                    "env": [
                        e
                        for e in c.get("env", [])
                        if e["name"] == "NVIDIA_VISIBLE_DEVICES"
                    ]
                }
                for c in pod.get("spec", {}).get("containers", [])
            ],
        }
    save(output / "probe.json", evidence)
    # Templates stay explicit: an inventory cannot invent PAI quota availability or asset provenance.
    save(output / "spec.json", spec)
    return evidence


def create_plan(
    spec, output, commit, repo=REPO, *, deployment_id=None, rollback_of=None
):
    plan_t0 = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    target = spec["target"]
    manifest = prepare(repo, commit, output / "source", target["oss_prefix"])
    if manifest["tools"].get("orchestrate.py") != digest(Path(__file__)):
        raise ValueError("Orchestrator differs from the selected source commit")
    run_id = uuid.uuid4().hex
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    deploy_id = deployment_id or f"deploy-{ts}-{run_id[:8]}"
    resources = {
        role: reused_resource(run_id, spec["model_id"], role, entry)
        for role, entry in spec.get("existing_services", {}).items()
    }
    state = dict(
        schema=1,
        deployment_id=deploy_id,
        target=copy.deepcopy(target),
        run_id=run_id,
        created_at=now(),
        source=manifest,
        spec=spec,
        resources=resources,
        phase="planned",
        generation=0,
        timings_sec={},
    )
    if rollback_of:
        state["rollback_of"] = rollback_of
    render_roles(state, output)
    state["timings_sec"]["plan_sec"] = round(time.monotonic() - plan_t0, 3)
    save(output / "state.json", state)
    write_deployment_summary(state, output)
    save_history_record(state, output)
    return state


def render_roles(state, output, launch_module=launch):
    fallback_auth_file = os.environ.get("SGLANG_PAI_IMAGE_AUTH_FILE", "").strip()
    for role in owned_roles(state["spec"]):
        directory = output / (role + "-" + str(state["generation"]))
        site = copy.deepcopy(role_site(state["spec"], role))
        pai = is_pai_role(state["spec"], role)
        if pai:
            has_inline_auth = bool(
                (site.get("job_spec", {}).get("ImageConfig") or {}).get("Auth")
            )
            if (
                not has_inline_auth
                and not site.get("image_auth_file")
                and fallback_auth_file
            ):
                site["image_auth_file"] = fallback_auth_file
        role_cfg = {
            k: v for k, v in state["spec"]["roles"][role].items() if k != "mode"
        }
        if launch_module is not launch:
            if "profile" not in launch_module.CONFIG_KEYS:
                role_cfg.pop("profile", None)
            if (
                "service_count" not in launch_module.CONFIG_KEYS
                and role_cfg.get("service_count", 1) == 1
            ):
                role_cfg.pop("service_count", None)
            if pai and "image_auth_file" in site:
                try:
                    auth_path = Path(site.pop("image_auth_file"))
                    auth_val = auth_path.read_text().strip()
                    if auth_val:
                        img_cfg = dict(site["job_spec"].get("ImageConfig") or {})
                        img_cfg.setdefault(
                            "DockerRegistry", site["image"].split("/", 1)[0]
                        )
                        img_cfg["Auth"] = auth_val
                        site["job_spec"]["ImageConfig"] = img_cfg
                except OSError:
                    pass
        launch_module.render(
            role_cfg,
            state["source"],
            site,
            directory,
        )
        rendered = read(directory / "render.json")
        name = (
            "sglang-"
            + role
            + "-"
            + state["run_id"][:12]
            + "-"
            + str(state["generation"])
        )
        # Replace only the generated platform resource name, not arbitrary script content.
        filenames = ("pai.json",) if pai else ("eflo.json", "h100-node.sh")
        for filename in filenames:
            path = directory / filename
            path.write_text(path.read_text().replace(rendered["name"], name))
        rendered["name"] = name
        rendered["payload_sha256"] = {
            filename: digest(directory / filename) for filename in filenames
        }
        save(directory / "render.json", rendered)


def _sanitize_pai_job_for_state(job_payload):
    cleaned = copy.deepcopy(job_payload)
    for spec in cleaned.get("JobSpecs") or []:
        if not isinstance(spec, dict):
            continue
        img_cfg = spec.get("ImageConfig")
        if isinstance(img_cfg, dict) and "Auth" in img_cfg:
            img_cfg.pop("Auth", None)
            if not img_cfg:
                spec.pop("ImageConfig", None)
    return cleaned


def submit(state, directory, cloud):
    if state["phase"] not in ("planned", "attach-planned"):
        raise ValueError(
            "Run is already submitted or UNKNOWN; never retry creates automatically"
        )
    owned = owned_roles(state["spec"])
    for role in owned:
        plan = directory / (role + "-" + str(state["generation"]))
        receipt = read(plan / "render.json")
        if any(
            digest(plan / name) != expected
            for name, expected in receipt["payload_sha256"].items()
        ):
            raise ValueError("Rendered platform payload changed after planning")
    state_path = directory / "state.json"
    timings = state.setdefault("timings_sec", {})
    upload_t0 = time.monotonic()
    # Write-ahead marker covers an upload timeout too; no create is repeated on resume.
    state["phase"] = "uploading"
    save(state_path, state)
    if digest(directory / "source/source.tar") != state["source"]["sha256"]:
        raise ValueError("Local source archive changed after planning")
    cloud.upload_source(directory / "source/source.tar", state["source"]["uri"])
    if any(not is_pai_role(state["spec"], role) for role in owned):
        cloud.stage_h100_source(state["source"])
    timings["upload_sec"] = round(time.monotonic() - upload_t0, 3)
    for role, entry in state["spec"].get("existing_services", {}).items():
        state["resources"].setdefault(
            role,
            reused_resource(state["run_id"], state["spec"]["model_id"], role, entry),
        )
    submit_t0 = time.monotonic()
    pai_roles_to_submit = [role for role in owned if is_pai_role(state["spec"], role)]
    k3s_roles_to_submit = [
        role for role in owned if not is_pai_role(state["spec"], role)
    ]
    # Submit H100 K3s Pods first sequentially (node-side select_gpu uses an advisory flock,
    # and a failed/unknown K3s create stops immediately before creating any PAI jobs).
    for role in k3s_roles_to_submit:
        plan = directory / (role + "-" + str(state["generation"]))
        resource = dict(
            run_id=state["run_id"],
            generation=state["generation"],
            name=read(plan / "render.json")["name"],
            namespace=role_site(state["spec"], role)["namespace"],
            status="submitting",
        )
        state["resources"][role] = resource
        state["phase"] = "submitting"
        save(state_path, state)

        def submitted(invoke, res=resource):
            res["invoke_id"] = invoke
            save(state_path, state)

        pod = json.loads(
            cloud.eflo(
                (plan / "h100-node.sh")
                .read_text()
                .replace("kubectl create -f -", "kubectl create -f - >/dev/null"),
                submitted=submitted,
            )
        )
        if (
            pod["metadata"]["name"] != resource["name"]
            or pod["metadata"]["namespace"] != resource["namespace"]
        ):
            raise ValueError("Created Pod identity mismatch")
        resource["uid"] = pod["metadata"]["uid"]
        resource["status"] = "submitted"
        save(state_path, state)
    # Submit PAI jobs in parallel after write-ahead recording so all H200 roles boot concurrently.
    if pai_roles_to_submit:
        pai_payloads = {}
        for role in pai_roles_to_submit:
            plan = directory / (role + "-" + str(state["generation"]))
            raw_pai_job = read(plan / "pai.json")
            pai_payloads[role] = raw_pai_job
            state["resources"][role] = dict(
                run_id=state["run_id"],
                generation=state["generation"],
                name=read(plan / "render.json")["name"],
                expected_job=_sanitize_pai_job_for_state(raw_pai_job),
                status="submitting",
            )
        state["phase"] = "submitting"
        save(state_path, state)
        submit_errors = {}
        with ThreadPoolExecutor(max_workers=len(pai_roles_to_submit)) as pool:
            futures = {
                role: pool.submit(cloud.submit_job, pai_payloads[role])
                for role in pai_roles_to_submit
            }
            for role in pai_roles_to_submit:
                try:
                    job_id = futures[role].result()
                    if not isinstance(job_id, str) or not job_id:
                        raise ValueError(f"Invalid PAI JobId returned for {role}")
                    state["resources"][role]["job_id"] = job_id
                    state["resources"][role]["status"] = "submitted"
                except Exception as exc:
                    submit_errors[role] = exc
        save(state_path, state)
        if submit_errors:
            first_role = next(iter(submit_errors))
            raise submit_errors[first_role]
    timings["submit_sec"] = round(time.monotonic() - submit_t0, 3)
    state["phase"] = "waiting"
    save(state_path, state)
    wait_t0 = time.monotonic()
    wait_ready(state, directory, cloud)
    timings["wait_ready_sec"] = round(time.monotonic() - wait_t0, 3)
    timings["up_total_sec"] = round(
        timings.get("plan_sec", 0.0)
        + timings.get("upload_sec", 0.0)
        + timings.get("submit_sec", 0.0)
        + timings.get("wait_ready_sec", 0.0),
        3,
    )
    timings["within_10min_sla"] = timings["up_total_sec"] < 600.0
    save(state_path, state)
    write_deployment_summary(state, directory)
    save_history_record(state, directory)


def wait_ready(state, directory, cloud):
    target = state.get("target", TARGET)
    deadline = time.monotonic() + cloud.timeout
    while time.monotonic() < deadline:
        errors = {}

        def check_role(role):
            resource = state["resources"][role]
            if resource.get("reused") is True:
                receipt = copy.deepcopy(resource["receipt"])
                entry_cfg = state["spec"].get("existing_services", {}).get(role, {})
                pinned_job_id = resource.get("job_id") or entry_cfg.get("job_id")
                pinned_job_name = resource.get("job_name") or entry_cfg.get("job_name")
                if pinned_job_id:
                    reused_job = cloud.get_job(pinned_job_id)
                    if (
                        str(reused_job.get("WorkspaceId"))
                        != str(target.get("workspace_id"))
                        or reused_job.get("ResourceId") != target.get("quota_id")
                        or reused_job.get("JobId") != pinned_job_id
                        or (
                            pinned_job_name
                            and reused_job.get("DisplayName") != pinned_job_name
                        )
                    ):
                        raise ValueError(
                            "Reused PAI job identity does not match pinned specification"
                        )
                    if reused_job.get("Status") != "Running":
                        raise RuntimeError("Reused PAI job is not Running")
                validate_endpoint(
                    receipt["http"],
                    role,
                    pai=bool(resource.get("pai")),
                    target=target,
                )
                health = http(receipt["http"] + "/health", cloud=cloud)
                if health.get("ready") is not True or health.get("role") != role:
                    raise RuntimeError("Reused worker is not ready")
                if health.get("warmed_up") is False:
                    raise RuntimeError("Reused worker has not completed startup warmup")
                try:
                    conn = http(
                        receipt["http"] + "/get_connection_status?reserve=false",
                        cloud=cloud,
                    )
                    service = conn.get("service", {})
                    if not service and resource.get("pai") and resource.get("job_id"):
                        # Native non-reserving queries intentionally omit descriptors.
                        # Authenticate the endpoint against its pinned PAI job instead
                        # of acquiring a lease from a read-only inspection.
                        pinned = {**resource, "name": reused_job["DisplayName"]}
                        verify_job(reused_job, pinned, target)
                        observed = cloud.receipt(pinned, role)
                        validate_endpoint(
                            observed["http"], role, pai=True, target=target
                        )
                        endpoints = service_endpoints(
                            observed["http"], resource.get("service_count", 1)
                        )
                        if (
                            observed.get("worker_id") != receipt["worker_id"]
                            or observed.get("role") != role
                            or observed.get("model_id") != receipt["model_id"]
                            or observed.get("status") != "ready"
                            or endpoints[resource.get("service_index", 0)]
                            != receipt["http"]
                        ):
                            raise ValueError(
                                "Reused PAI receipt does not match the service"
                            )
                        receipt.update(observed, http=receipt["http"])
                        service = observed
                    if (
                        service.get("worker_id") != receipt["worker_id"]
                        or service.get("role") != role
                    ):
                        raise ValueError("Reused service worker_id or role mismatch")
                    receipt.update(
                        status="ready",
                        health=health,
                        managed=False,
                        **(
                            {"warmed_up": health["warmed_up"] is True}
                            if "warmed_up" in health
                            else {}
                        ),
                        warmup=health.get("warmup"),
                    )
                except (HTTPError, RuntimeError):
                    signing_key = os.environ.get("ROLE_POOL_SIGNING_KEY", "")
                    control_epoch = (
                        state.get("spec", {}).get("pool", {}).get("control_epoch", "")
                    )
                    if not signing_key or not control_epoch:
                        raise
                    contracts = pool_contract()
                    authority = contracts.Credentials(signing_key, control_epoch)
                    body = dict(control_epoch=control_epoch)
                    token = authority.sign(
                        "identity", body, expires_at=time.time() + 30
                    )
                    identity = http(
                        receipt["http"] + "/managed/identity",
                        body,
                        token,
                        cloud=cloud,
                    )
                    service = identity.get("service", {})
                    expected_be = (
                        state["spec"]
                        .get("existing_services", {})
                        .get(role, {})
                        .get("boot_epoch")
                    )
                    if (
                        service.get("worker_id") != receipt["worker_id"]
                        or service.get("role") != role
                        or identity.get("control_epoch", control_epoch) != control_epoch
                        or (
                            expected_be
                            and expected_be != "reused"
                            and identity.get("boot_epoch") != expected_be
                        )
                    ):
                        raise ValueError(
                            "Reused managed service identity or boot_epoch mismatch"
                        ) from None
                    operator_token = os.environ.get("ROLE_POOL_OPERATOR_TOKEN", "")
                    if operator_token and target.get("coordinator_url"):
                        proc_status = http(
                            target["coordinator_url"] + "/v1/role-pool/process-status",
                            dict(
                                process_id=receipt["worker_id"],
                                boot_epoch=identity["boot_epoch"],
                            ),
                            operator_token,
                            cloud=cloud,
                        )
                        proc_obj = proc_status.get("process", {})
                        if proc_obj.get("state") != "ready":
                            raise RuntimeError(
                                "Reused managed worker is not ready in Coordinator"
                            ) from None
                        proc_manifest = proc_obj.get("manifest") or {}
                        expected_pool = state.get("spec", {}).get("pool", {})
                        if (
                            "compatibility" in proc_manifest
                            and expected_pool.get("compatibility")
                            and proc_manifest["compatibility"]
                            != expected_pool["compatibility"]
                        ) or (
                            "network_domain" in proc_manifest
                            and expected_pool.get("network_domain")
                            and proc_manifest["network_domain"]
                            != expected_pool["network_domain"]
                        ):
                            raise ValueError(
                                "Reused managed worker compatibility or network_domain does not match spec.pool"
                            ) from None
                    receipt.update(
                        status="ready",
                        health=health,
                        managed=True,
                        boot_epoch=identity["boot_epoch"],
                        service=service,
                        **(
                            {"warmed_up": health["warmed_up"] is True}
                            if "warmed_up" in health
                            else {}
                        ),
                        warmup=health.get("warmup"),
                    )
                return receipt
            existing_worker_config = (resource.get("receipt") or {}).get(
                "worker_config"
            )
            receipt = cloud.receipt(resource, role)
            c = state["spec"]["roles"][role]
            if (
                receipt["commit"] != state["source"]["commit"]
                or receipt["archive_sha256"] != state["source"]["sha256"]
                or receipt["model_id"] != c["model_id"]
                or receipt["asset_receipt_sha256"] != c["asset_receipt_sha256"]
            ):
                raise ValueError("Runtime source or model identity mismatch")
            validate_endpoint(
                receipt["http"],
                role,
                pai=is_pai_role(state["spec"], role),
                target=target,
            )
            if urlsplit(receipt["http"]).port != launch.role_ports(c)["http_port"]:
                raise ValueError("Runtime HTTP port differs from the frozen role spec")
            # Renderer adds the hardware gate to the effective config.
            plan_config = read(
                directory / (role + "-" + str(state["generation"])) / "role.json"
            )
            expected_config = hashlib.sha256(
                json.dumps(plan_config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            expected_configs = {expected_config}
            if (plan_config.get("worker_config") or {}).get("hardware_ids") == [
                "__ALLOCATED_GPU_UUID__"
            ]:
                allocated_config = copy.deepcopy(plan_config)
                allocated_config["worker_config"]["hardware_ids"] = sorted(
                    set(re.findall(r"GPU-[A-Za-z0-9-]+", receipt.get("gpu", "")))
                )
                expected_configs.add(
                    hashlib.sha256(
                        json.dumps(
                            allocated_config, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest()
                )
            if receipt.get("role_config_sha256") not in expected_configs:
                raise ValueError("Runtime role config identity mismatch")
            if receipt["status"] in ("failed", "exited"):
                raise ValueError("Owned worker failed or exited")
            health = http(receipt["http"] + "/health", cloud=cloud)
            if (
                receipt["status"] != "ready"
                or health.get("ready") is not True
                or health.get("role") != role
            ):
                raise RuntimeError("Owned worker is not ready")
            if health.get("warmed_up") is False or receipt.get("warmed_up") is False:
                raise RuntimeError("Owned worker has not completed startup warmup")
            if "warmed_up" in health:
                receipt["warmed_up"] = health["warmed_up"] is True
            elif "warmed_up" in receipt:
                receipt["warmed_up"] = receipt["warmed_up"] is True
            if health.get("warmup") is not None:
                receipt["warmup"] = health["warmup"]
            if (
                state.get("attachment_mode") == "in-place"
                and existing_worker_config is not None
                and receipt.get("worker_config") is None
            ):
                receipt = {**receipt, "worker_config": existing_worker_config}
            return receipt

        with ThreadPoolExecutor(max_workers=len(ROLES)) as pool:
            futures = {role: pool.submit(check_role, role) for role in ROLES}
            for role in ROLES:
                try:
                    receipt = futures[role].result()
                    state["resources"][role].update(
                        receipt=receipt, health_observed_at=now(), status="ready"
                    )
                except (OSError, RuntimeError) as error:
                    errors[role] = type(error).__name__
        state["waiting"] = errors
        save(directory / "state.json", state)
        if not errors:
            state["phase"] = "ready"
            save(directory / "state.json", state)
            write_deployment_summary(state, directory)
            save_history_record(state, directory)
            return
        time.sleep(3)
    raise TimeoutError(
        "Three-role readiness deadline exceeded; resources retained in state"
    )


def validate_endpoint(url, role, *, pai=None, target=None):
    t = target or TARGET
    if pai is None:
        pai = role == "dit"
    uri = urlsplit(url)
    if (
        uri.scheme != "http"
        or uri.username
        or uri.password
        or uri.path
        or uri.query
        or uri.fragment
    ):
        raise ValueError("Invalid role HTTP endpoint")
    h100_ip = t.get("h100_ip")
    if not pai and h100_ip and uri.hostname != h100_ip:
        raise ValueError("H100 endpoint is outside the configured node")
    if pai or not h100_ip:
        address = ipaddress.ip_address(uri.hostname or "")
        if address not in ipaddress.ip_network("10.0.0.0/8") or str(address) == h100_ip:
            raise ValueError("Role endpoint is outside the configured private network")
    if uri.port is None:
        raise ValueError("Role HTTP endpoint requires a port")


class VPCUnreachable(OSError):
    """Fail the local health preflight promptly instead of waiting its full GPU timeout."""


async def native_selftest(addresses, worker_ids, directory):
    path = REPO / "benchmark/zing_h3_causal/worker_group_master.py"
    module_spec = importlib.util.spec_from_file_location(
        "sglang_owned_group_master", path
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    launcher = module.LocalWorkers(None, directory / "selftest")

    def request(address, path, **kwargs):
        try:
            reply = module.http_request(address, path, **kwargs)
        except HTTPError:
            raise
        except OSError:
            if path == "/health" and vpc_url(address):
                raise VPCUnreachable(
                    "Local worker health preflight cannot reach the VPC"
                ) from None
            raise RuntimeError(
                "Native selftest transport failed after health preflight; refusing replay"
            ) from None
        if path == "/get_connection_status" and "service" in reply:
            role = next(r for r in ROLES if addresses[r] == address)
            if reply["service"]["worker_id"] != worker_ids[role]:
                raise ValueError(
                    "Selftest endpoint no longer belongs to the owned worker"
                )
        return reply

    async with module.WorkerGroupMaster(
        addresses, launcher=launcher, request=request
    ) as group:
        return group.summary()


def remote_selftest(state, addresses, worker_ids, cloud):
    encoder = state["resources"]["encoder"]
    attestation = dict(
        commit=state["source"]["commit"], archive_sha256=state["source"]["sha256"]
    )
    # When encoder is an owned K3s H100 Pod, exec inside its verified container.
    if not encoder.get("reused") and "uid" in encoder and "namespace" in encoder:
        if cloud.pod(encoder)["metadata"]["uid"] != encoder["uid"]:
            raise ValueError("Encoder Pod UID changed before selftest")
        package = Path(encoder["receipt"]["runtime"]["sglang"]["path"])
        if not package.is_absolute():
            raise ValueError("Encoder receipt has no absolute verified package path")
        root = str(package.parents[2])
        code = (
            """import asyncio,hashlib,json,sys
from pathlib import Path
root=Path(ROOT)
if json.loads((root.parent/'verified-source.json').read_text())!=ATTESTATION: raise ValueError('Source attestation mismatch')
if hashlib.sha256((root/'deploy/sglang/orchestrate.py').read_bytes()).hexdigest()!=EXPECTED: raise ValueError('Selftest orchestrator source mismatch')
sys.path.insert(0,str(root/'deploy/sglang'))
import orchestrate
result=asyncio.run(orchestrate.native_selftest(ADDRESSES,WORKERS,root.parent))
print(json.dumps(result))
""".replace("ROOT", repr(root))
            .replace("ATTESTATION", repr(attestation))
            .replace("EXPECTED", repr(digest(Path(__file__))))
            .replace("ADDRESSES", repr(addresses))
            .replace("WORKERS", repr(worker_ids))
        )
        command = [
            "kubectl",
            "-n",
            encoder["namespace"],
            "exec",
            encoder["name"],
            "--",
            "/opt/sglang/bin/python",
            "-c",
            code,
        ]
        return json.loads(cloud.eflo(shlex.join(command)))

    # For all-PAI H200 deployments (or reused non-managed services), stage the verified
    # archive on the VPC Eflo CPU relay and execute native_selftest inside a 0-GPU cached
    # SGLang runtime container (which carries numpy/PIL/runtime deps), falling back to
    # host Python 3.12+ only when containerd or the cached image is unavailable.
    cloud.stage_h100_source(state["source"])
    archive = (
        "/var/lib/sglang-orchestrate/cache/" + state["source"]["sha256"] + "/source.tar"
    )
    expected_orch = state["source"].get("tools", {}).get("orchestrate.py") or digest(
        Path(__file__)
    )
    relay_code = (
        """import asyncio,hashlib,json,shutil,sys,tarfile,tempfile
from pathlib import Path
archive=Path(__ARCHIVE__)
if hashlib.sha256(archive.read_bytes()).hexdigest()!=__SHA__: raise ValueError('Relay archive SHA256 mismatch')
work=Path(tempfile.mkdtemp(prefix='sglang-selftest-relay-'))
try:
 root=work/'source'
 root.mkdir()
 with tarfile.open(archive) as tar:
  for m in tar.getmembers():
   if m.name.startswith(('deploy/sglang/','benchmark/zing_h3_causal/','python/')):
    tar.extract(m,path=root)
 if hashlib.sha256((root/'deploy/sglang/orchestrate.py').read_bytes()).hexdigest()!=__EXPECTED__:
  raise ValueError('Relay orchestrator source mismatch')
 sys.path.insert(0,str(root/'python'))
 sys.path.insert(0,str(root/'deploy/sglang'))
 import orchestrate
 result=asyncio.run(orchestrate.native_selftest(__ADDRESSES__,__WORKERS__,work))
 print(json.dumps(result))
finally:
 shutil.rmtree(work,ignore_errors=True)
""".replace("__ARCHIVE__", repr(archive))
        .replace("__SHA__", repr(state["source"]["sha256"]))
        .replace("__EXPECTED__", repr(expected_orch))
        .replace("__ADDRESSES__", repr(addresses))
        .replace("__WORKERS__", repr(worker_ids))
    )
    sites = state.get("spec", {}).get("sites") or {}
    preferred_site = (
        sites.get("h100")
        or sites.get("h200")
        or (next(iter(sites.values())) if sites else {})
    )
    site_image = str((preferred_site or {}).get("image") or "")
    cached_image = ""
    if "@sha256:" in site_image:
        img_repo, img_digest = site_image.split("@sha256:", 1)
        cached_image = f"{img_repo}:cached-{img_digest[:12]}"
    cname = "sglang-selftest-" + uuid.uuid4().hex[:12]
    quoted_code = shlex.quote(relay_code)
    script = (
        'CTR=/var/lib/rancher/k3s/data/current/bin/ctr; [ -x "$CTR" ] || CTR=ctr\n'
        "SOCK=/run/k3s/containerd/containerd.sock\n"
        f"IMG={shlex.quote(site_image)}\n"
        f"CACHED={shlex.quote(cached_image)}\n"
        'if [ -n "$CACHED" ] && [ -S "$SOCK" ] && command -v "$CTR" >/dev/null 2>&1; then\n'
        '  "$CTR" -a "$SOCK" -n k8s.io images tag "$IMG" "$CACHED" >/dev/null 2>&1 || true\n'
        "fi\n"
        'if [ -n "$CACHED" ] && [ -S "$SOCK" ] && "$CTR" -a "$SOCK" -n k8s.io images ls -q 2>/dev/null | grep -Fxq "$CACHED"; then\n'
        '  "$CTR" -a "$SOCK" -n k8s.io run --rm --net-host '
        "--mount type=bind,src=/var/lib/sglang-orchestrate/cache,dst=/var/lib/sglang-orchestrate/cache,options=rbind:ro "
        f'"$CACHED" {shlex.quote(cname)} /opt/sglang/bin/python -c {quoted_code}\n'
        "else\n"
        f'  PY=/usr/bin/python3.12; [ -x "$PY" ] || PY=python3; "$PY" -c {quoted_code}\n'
        "fi\n"
    )
    return json.loads(cloud.eflo(script))


def select_deterministic_admission_id(
    target_service_ids, avoid_service_ids=None, *, prefix="selftest"
):
    contracts = pool_contract()
    avoid = {r: set((avoid_service_ids or {}).get(r, ())) for r in target_service_ids}
    for _ in range(50000):
        candidate = f"{prefix}-{uuid.uuid4().hex}"
        ok = True
        for role, target_sid in target_service_ids.items():
            target_hash = contracts.digest([candidate, target_sid])
            if target_hash >= "20" + "f" * 62:
                ok = False
                break
            if any(
                target_hash >= contracts.digest([candidate, other_sid])
                for other_sid in avoid.get(role, ())
                if other_sid != target_sid
            ):
                ok = False
                break
        if ok:
            return candidate
    return f"{prefix}-{uuid.uuid4().hex}"


def _verify_managed_endpoint_identity(
    identity,
    receipt,
    role,
    index,
    control_epoch,
    expected_cfg,
    expected_boot_epoch=None,
):
    svc = identity.get("service", {})
    expected_service_id = f"{receipt['worker_id']}/{index}"
    service_id = svc.get("service_id", expected_service_id)
    deployment = identity.get("deployment")
    if (
        svc.get("worker_id") != receipt["worker_id"]
        or svc.get("role") != role
        or not isinstance(service_id, str)
        or not service_id.endswith(f"/{index}")
        or identity.get("control_epoch", control_epoch) != control_epoch
        or (
            isinstance(deployment, dict)
            and (
                (
                    "hardware_ids" in deployment
                    and deployment["hardware_ids"] != expected_cfg["hardware_ids"]
                )
                or (
                    "pool_id" in deployment
                    and deployment["pool_id"] != expected_cfg["pool_id"]
                )
                or (
                    "compatibility" in deployment
                    and deployment["compatibility"] != expected_cfg["compatibility"]
                )
                or (
                    "network_domain" in deployment
                    and deployment["network_domain"] != expected_cfg["network_domain"]
                )
            )
        )
        or (
            expected_boot_epoch is not None
            and identity.get("boot_epoch") != expected_boot_epoch
        )
    ):
        raise ValueError("In-place attach returned mismatched worker identity")
    return service_id, identity["boot_epoch"]


def _preflight_or_reconcile_owned_endpoints(
    state, directory, configs, authority, cloud, *, in_place=True
):
    owned = owned_roles(state["spec"])
    control_epoch = state["spec"]["pool"]["control_epoch"]
    attachment = state.setdefault("attachment", {})
    attached_identities = attachment.setdefault("identities", {})
    endpoint_identities = attachment.setdefault("endpoint_identities", {})
    endpoint_intents = attachment.setdefault("endpoint_intents", {})
    for role in owned:
        receipt = state["resources"][role]["receipt"]
        role_cfg = state["spec"]["roles"][role]
        if set(role_cfg.get("secret_files", {})) != launch.SECRET_ENV:
            raise ValueError(
                "Attach requires pre-mounted platform secret files for every role"
            )
        role_ep_ids = endpoint_identities.setdefault(role, {})
        role_prior = attached_identities.get(role)
        role_boot_epoch = (
            role_prior.get("boot_epoch") if isinstance(role_prior, dict) else None
        )
        for existing_ep_info in role_ep_ids.values():
            if role_boot_epoch is None and isinstance(existing_ep_info, dict):
                role_boot_epoch = existing_ep_info.get("boot_epoch")
        for index, endpoint in enumerate(
            service_endpoints(receipt["http"], launch.role_service_count(role_cfg))
        ):
            recorded_ep = role_ep_ids.get(endpoint)
            known_managed = in_place and (
                isinstance(recorded_ep, dict) or role in attached_identities
            )
            if not known_managed:
                try:
                    status = http(
                        endpoint + "/get_connection_status?reserve=false", cloud=cloud
                    )
                    if status.get("state") != "available":
                        raise RuntimeError(
                            "Owned service is busy; attach cannot restart it"
                        )
                    continue
                except (HTTPError, RuntimeError) as exc:
                    if not in_place or "Owned service is busy" in str(exc):
                        raise
            expected_boot = (
                recorded_ep.get("boot_epoch")
                if isinstance(recorded_ep, dict)
                else role_boot_epoch
            )
            id_body = dict(control_epoch=control_epoch)
            id_token = authority.sign("identity", id_body, expires_at=time.time() + 30)
            identity = http(
                endpoint + "/managed/identity",
                id_body,
                id_token,
                cloud=cloud,
            )
            service_id, observed_boot = _verify_managed_endpoint_identity(
                identity,
                receipt,
                role,
                index,
                control_epoch,
                configs[role],
                expected_boot,
            )
            obs_body = dict(
                control_epoch=control_epoch,
                boot_epoch=observed_boot,
            )
            obs_token = authority.sign("observe", obs_body, expires_at=time.time() + 30)
            snapshot = http(
                endpoint + "/managed/observe",
                obs_body,
                obs_token,
                cloud=cloud,
            )
            if (
                snapshot.get("state") != "idle"
                or snapshot.get("owner") is not None
                or snapshot.get("restart_required")
            ):
                raise RuntimeError("Owned service is busy; attach cannot restart it")
            role_boot_epoch = observed_boot
            role_ep_ids[endpoint] = dict(
                process_id=receipt["worker_id"],
                service_id=service_id,
                boot_epoch=observed_boot,
            )
            endpoint_intents.get(role, {}).pop(endpoint, None)
            state["attachment_mode"] = "in-place"
            save(directory / "state.json", state)


def _attach_owned_endpoints_in_place(state, directory, configs, authority, cloud):
    owned = owned_roles(state["spec"])
    control_epoch = state["spec"]["pool"]["control_epoch"]
    attachment = state.setdefault("attachment", {})
    attached_identities = attachment.setdefault("identities", {})
    endpoint_identities = attachment.setdefault("endpoint_identities", {})
    endpoint_intents = attachment.setdefault("endpoint_intents", {})
    for role in owned:
        receipt = state["resources"][role]["receipt"]
        role_cfg = state["spec"]["roles"][role]
        count = launch.role_service_count(role_cfg)
        role_ep_ids = endpoint_identities.setdefault(role, {})
        role_intents = endpoint_intents.setdefault(role, {})
        prior = attached_identities.get(role)
        boot_epoch = prior.get("boot_epoch") if isinstance(prior, dict) else None
        for existing_ep_info in role_ep_ids.values():
            if boot_epoch is None and isinstance(existing_ep_info, dict):
                boot_epoch = existing_ep_info.get("boot_epoch")
        primary_service_id = None
        for index, endpoint in enumerate(service_endpoints(receipt["http"], count)):
            confirmed = role_ep_ids.get(endpoint)
            if isinstance(confirmed, dict) and confirmed.get("boot_epoch"):
                if (
                    confirmed.get("process_id") != receipt["worker_id"]
                    or not str(confirmed.get("service_id", "")).endswith(f"/{index}")
                    or (
                        boot_epoch is not None
                        and confirmed.get("boot_epoch") != boot_epoch
                    )
                ):
                    raise ValueError(
                        "In-place attach returned mismatched worker identity"
                    )
                boot_epoch = confirmed["boot_epoch"]
                if index == 0:
                    primary_service_id = confirmed["service_id"]
                continue
            role_intents[endpoint] = dict(
                process_id=receipt["worker_id"],
                service_id=f"{receipt['worker_id']}/{index}",
                control_epoch=control_epoch,
                requested_at=now(),
            )
            state["attachment_mode"] = "in-place"
            save(directory / "state.json", state)
            body = dict(
                control_epoch=control_epoch,
                worker_config=configs[role],
            )
            token = authority.sign("attach", body, expires_at=time.time() + 30)
            identity = http(endpoint + "/managed/attach", body, token, cloud=cloud)
            service_id, observed_boot = _verify_managed_endpoint_identity(
                identity,
                receipt,
                role,
                index,
                control_epoch,
                configs[role],
                boot_epoch,
            )
            boot_epoch = observed_boot
            if index == 0:
                primary_service_id = service_id
            role_ep_ids[endpoint] = dict(
                process_id=receipt["worker_id"],
                service_id=service_id,
                boot_epoch=boot_epoch,
            )
            role_intents.pop(endpoint, None)
            save(directory / "state.json", state)
        receipt["worker_config"] = copy.deepcopy(configs[role])
        state["attachment_mode"] = "in-place"
        attached_identities[role] = dict(
            process_id=receipt["worker_id"],
            boot_epoch=boot_epoch,
            service_id=primary_service_id or f"{receipt['worker_id']}/0",
        )
        save(directory / "state.json", state)


def _attach_owned_in_place(state, directory, cloud):
    configs = worker_configs(state)
    for role, config in configs.items():
        save(directory / (role + "-role_pool_worker.json"), config)
    owned = owned_roles(state["spec"])
    if not owned:
        return configs
    if set(os.environ) & {"ROLE_POOL_SIGNING_KEY", "ROLE_POOL_OPERATOR_TOKEN"} != {
        "ROLE_POOL_SIGNING_KEY",
        "ROLE_POOL_OPERATOR_TOKEN",
    }:
        raise ValueError(
            "Attach verification requires local signing and operator credentials"
        )
    contracts = pool_contract()
    authority = contracts.Credentials(
        os.environ["ROLE_POOL_SIGNING_KEY"],
        state["spec"]["pool"]["control_epoch"],
    )
    _preflight_or_reconcile_owned_endpoints(
        state, directory, configs, authority, cloud, in_place=True
    )
    _attach_owned_endpoints_in_place(state, directory, configs, authority, cloud)
    deadline = time.monotonic() + (cloud.timeout if cloud else 60)
    while True:
        try:
            verify_pool(state, directory, cloud=cloud)
            break
        except (OSError, RuntimeError):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Worker pool registration deadline exceeded"
                ) from None
            time.sleep(2)
    return configs


def coordinator_selftest(state, directory, cloud):
    target = state.get("target", TARGET)
    room_token = os.environ.get("WHITEBOX_ROOM_TOKEN", "")
    if not room_token:
        raise ValueError(
            "Coordinator lease selftest for managed reused services requires WHITEBOX_ROOM_TOKEN"
        )
    owned = owned_roles(state["spec"])
    attached_identities = state.get("attachment", {}).get("identities", {})
    if any(r not in attached_identities for r in owned):
        _attach_owned_in_place(state, directory, cloud)
    worker_ids = {r: state["resources"][r]["receipt"]["worker_id"] for r in ROLES}
    target_service_ids = {
        r: f"{worker_ids[r]}/{state['resources'][r].get('service_index', 0)}"
        for r in ROLES
    }
    avoid_service_ids = {r: set() for r in ROLES}
    raw_avoid = os.environ.get("WHITEBOX_AVOID_SERVICE_IDS", "")
    if raw_avoid:
        for role, items in json.loads(raw_avoid).items():
            if role in avoid_service_ids and isinstance(items, list):
                avoid_service_ids[role].update(items)
    profile = sorted(state["spec"]["pool"]["profile_limits"])[0]
    for _ in range(4):
        admission_id = select_deterministic_admission_id(
            target_service_ids,
            avoid_service_ids,
            prefix="selftest-" + state["run_id"][:8],
        )
        admit_body = dict(
            admission_id=admission_id,
            user_id="selftest-user-" + state["run_id"][:12],
            player_id="selftest-player-" + state["run_id"][:12],
            generation_id="selftest-gen-" + uuid.uuid4().hex[:12],
            profile=profile,
            compatibility=state["spec"]["pool"]["compatibility"],
            network_domain=state["spec"]["pool"]["network_domain"],
            ttl=120,
            created_at=time.time(),
            operation="prefill",
        )
        allocation = http(
            target["coordinator_url"] + "/v1/role-pool/admit",
            admit_body,
            room_token,
            cloud=cloud,
        )
        if allocation.get("state") != "bound" or not allocation.get("group_id"):
            raise RuntimeError("Coordinator admit did not bind a ready worker group")
        state["active_leases"] = {
            r: dict(
                admission_id=admission_id,
                allocation_id=allocation["allocation_id"],
                worker_group_id=allocation["group_id"],
            )
            for r in ROLES
        }
        save(directory / "state.json", state)
        closed = http(
            target["coordinator_url"] + "/v1/role-pool/close",
            dict(
                admission_id=admission_id,
                allocation_id=allocation["allocation_id"],
                sdk_cleanup=None,
            ),
            room_token,
            cloud=cloud,
        )
        if closed.get("state") != "released" or closed.get("roles") != {
            r: "released" for r in ROLES
        }:
            raise RuntimeError("Coordinator close did not confirm three-role release")
        state.pop("active_leases", None)
        save(directory / "state.json", state)
        topo = allocation.get("topology", {})
        if all(topo.get(r, {}).get("worker_id") == worker_ids[r] for r in ROLES):
            return dict(
                worker_group_id=allocation["group_id"],
                allocation_id=allocation["allocation_id"],
                admission_id=admission_id,
                connections_kept=False,
                topology={
                    r: dict(
                        worker_id=topo[r]["worker_id"],
                        service_id=topo[r]["service_id"],
                        endpoints=topo[r].get("endpoints", {}),
                    )
                    for r in ROLES
                },
                release_roles=closed["roles"],
            )
        for r in ROLES:
            sid = topo.get(r, {}).get("service_id")
            if sid and sid != target_service_ids[r]:
                avoid_service_ids[r].add(sid)
    raise RuntimeError(
        "Coordinator lease selftest could not select the exact target three-role tuple"
    )


async def selftest(state, directory, cloud=None):
    target = state.get("target", TARGET)
    owned = owned_roles(state["spec"])
    has_managed_reused = any(
        state["resources"][r].get("reused") is True
        and state["resources"][r].get("receipt", {}).get("managed") is True
        for r in ROLES
    )
    if state["phase"] != "ready" or (
        not has_managed_reused
        and any(state["spec"]["roles"][r].get("worker_config") for r in owned)
    ):
        raise ValueError("Isolated native selftest requires unregistered ready roles")
    addresses = {role: state["resources"][role]["receipt"]["http"] for role in ROLES}
    for role, address in addresses.items():
        resource = state["resources"][role]
        pai = (
            bool(resource.get("pai"))
            if resource.get("reused") is True
            else is_pai_role(state["spec"], role)
        )
        validate_endpoint(address, role, pai=pai, target=target)
    worker_ids = {r: state["resources"][r]["receipt"]["worker_id"] for r in ROLES}
    selftest_t0 = time.monotonic()
    state["selftest"] = dict(
        status="running",
        observed_at=now(),
        source=state["source"],
        worker_ids=worker_ids,
    )
    save(directory / "state.json", state)
    execution = "operator"
    scope = "native-build-quiesce-destroy"
    if has_managed_reused:
        result = await asyncio.to_thread(coordinator_selftest, state, directory, cloud)
        execution = "coordinator-lease"
        scope = "coordinator-lease-build-quiesce-release"
    else:
        try:
            result = await native_selftest(addresses, worker_ids, directory)
        except OSError as error:
            if (
                cloud is None
                or isinstance(error, HTTPError)
                or not any(vpc_url(url) for url in addresses.values())
            ):
                raise
            result = await asyncio.to_thread(
                remote_selftest, state, addresses, worker_ids, cloud
            )
            enc_res = state["resources"]["encoder"]
            execution = (
                "owned-encoder-pod"
                if (not enc_res.get("reused") and "uid" in enc_res)
                else "vpc-cpu-relay"
            )
    if (
        not isinstance(result, dict)
        or not result.get("worker_group_id")
        or result.get("connections_kept") is not False
        or set(result.get("topology", {})) != set(ROLES)
        or any(
            result["topology"][role].get("worker_id") != worker_ids[role]
            for role in ROLES
        )
    ):
        raise ValueError(
            "Selftest summary does not match the owned workers and teardown contract"
        )
    selftest_sec = round(time.monotonic() - selftest_t0, 3)
    state.setdefault("timings_sec", {})["selftest_sec"] = selftest_sec
    state["selftest"] = dict(
        status="passed",
        scope=scope,
        observed_at=now(),
        elapsed_sec=selftest_sec,
        source=state["source"],
        worker_ids=worker_ids,
        group=result,
        execution=execution,
    )
    save(directory / "selftest.json", state["selftest"])
    save(directory / "state.json", state)
    write_deployment_summary(state, directory)
    save_history_record(state, directory)


def worker_configs(state):
    spec = state["spec"]
    target = state.get("target", spec.get("target", TARGET))
    pool = spec["pool"]
    required = {
        "control_epoch",
        "compatibility",
        "network_domain",
        "capacity",
        "profiles",
        "profile_limits",
    }
    if set(pool) != required:
        raise ValueError(
            "Pool requires explicit measured capacity/profile limits and authority identity"
        )
    contract = pool_contract()
    for limits in pool["profile_limits"].values():
        contract.profile_limits(limits)
    result = {}
    all_ids = []
    owned = set(owned_roles(spec))
    for role in ROLES:
        resource = state["resources"][role]
        receipt = resource["receipt"]
        ids = re.findall(r"GPU-[A-Za-z0-9-]+", receipt.get("gpu", ""))
        if len(set(ids)) != 1:
            raise ValueError("Exactly one observed GPU UUID per role is required")
        all_ids.extend(set(ids))
        if role not in owned:
            continue
        capacity = contract.vector(pool["capacity"][role])
        if set(pool["profiles"][role]) != set(pool["profile_limits"]):
            raise ValueError("Every role must advertise the same measured profiles")
        for costs in pool["profiles"][role].values():
            costs = contract.vector(costs)
            if set(costs) != set(capacity) or any(
                costs[k] > capacity[k] for k in costs
            ):
                raise ValueError("Profile exceeds process capacity")
        result[role] = dict(
            schema=1,
            role=role,
            service_count=launch.role_service_count(spec["roles"][role]),
            coordinator_url=target["coordinator_url"],
            pool_id=target["pool_id"],
            control_epoch=pool["control_epoch"],
            compatibility=pool["compatibility"],
            network_domain=pool["network_domain"],
            capacity=pool["capacity"][role],
            profiles=pool["profiles"][role],
            profile_limits=pool["profile_limits"],
            hardware_ids=sorted(set(ids)),
            signing_secret_env="ROLE_POOL_SIGNING_KEY",
            role_token_env="ROLE_POOL_REGISTRATION_TOKEN",
        )
    if len(set(all_ids)) != 3:
        raise ValueError("Roles must use three distinct observed GPUs")
    return result


def pool_contract():
    path = REPO / "python/sglang/multimodal_gen/runtime/realtime/role_pool/contracts.py"
    spec = importlib.util.spec_from_file_location("sglang_pool_wire_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attach(state, directory, cloud, execute=False, in_place=False):
    target = state.get("target", TARGET)
    if state.get("selftest", {}).get("status") != "passed" or state["phase"] != "ready":
        raise ValueError("Attach requires successful isolated selftest")
    tested = state["selftest"]
    if tested.get("source") != state["source"] or tested.get("worker_ids") != {
        r: state["resources"][r]["receipt"]["worker_id"] for r in ROLES
    }:
        raise ValueError("Selftest identity is stale")
    if cloud is not None:
        refreshed = False
        for role in owned_roles(state["spec"]):
            res = state["resources"][role]
            rcpt = res.get("receipt") or {}
            if not re.findall(r"GPU-[A-Za-z0-9-]+", rcpt.get("gpu", "")):
                latest = cloud.receipt(res, role)
                if (
                    latest.get("worker_id") == rcpt.get("worker_id")
                    and latest.get("archive_sha256") == rcpt.get("archive_sha256")
                    and re.findall(r"GPU-[A-Za-z0-9-]+", latest.get("gpu", ""))
                ):
                    rcpt["gpu"] = latest["gpu"]
                    refreshed = True
        if refreshed:
            save(directory / "state.json", state)
    configs = worker_configs(state)
    for role, config in configs.items():
        save(directory / (role + "-role_pool_worker.json"), config)
    if not execute:
        return configs
    attach_t0 = time.monotonic()
    if set(os.environ) & {"ROLE_POOL_SIGNING_KEY", "ROLE_POOL_OPERATOR_TOKEN"} != {
        "ROLE_POOL_SIGNING_KEY",
        "ROLE_POOL_OPERATOR_TOKEN",
    }:
        raise ValueError(
            "Attach verification requires local signing and operator credentials"
        )
    # Validate existing route and all inputs before stopping any owned resource.
    capacity = http(
        target["coordinator_url"]
        + "/v1/capacity?"
        + urlencode({"model_revision": state["spec"]["route"]}),
        cloud=cloud,
    )
    if (
        capacity.get("backend") != "role_pool"
        or capacity.get("profile") not in state["spec"]["pool"]["profile_limits"]
    ):
        raise ValueError("Existing copy route does not match measured profiles")
    contracts = pool_contract()
    authority = contracts.Credentials(
        os.environ["ROLE_POOL_SIGNING_KEY"],
        state["spec"]["pool"]["control_epoch"],
    )
    _preflight_or_reconcile_owned_endpoints(
        state, directory, configs, authority, cloud, in_place=in_place
    )
    state["pre_attach_capacity"] = copy.deepcopy(capacity)
    if in_place:
        _attach_owned_endpoints_in_place(state, directory, configs, authority, cloud)
    else:
        # Fallback: restart owned idle resources with the agent config.
        staged = copy.deepcopy(state)
        staged["generation"] += 1
        for role, config in configs.items():
            # A recreated PAI worker can receive a different physical GPU.
            staged["spec"]["roles"][role]["worker_config"] = {
                **config,
                "hardware_ids": ["__ALLOCATED_GPU_UUID__"],
            }
        render_roles(staged, directory)
        down(state, directory, cloud)
        staged.setdefault("history", []).append(copy.deepcopy(state["resources"]))
        staged["resources"] = {
            role: copy.deepcopy(res)
            for role, res in state["resources"].items()
            if res.get("reused") is True
        }
        staged["phase"] = "attach-planned"
        state.clear()
        state.update(staged)
        save(directory / "state.json", state)
        submit(state, directory, cloud)
    deadline = time.monotonic() + cloud.timeout
    while True:
        try:
            verify_pool(state, directory, cloud=cloud)
            break
        except (OSError, RuntimeError):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Worker pool registration deadline exceeded"
                ) from None
            time.sleep(2)
    attach_sec = round(time.monotonic() - attach_t0, 3)
    timings = state.setdefault("timings_sec", {})
    timings["attach_sec"] = attach_sec
    end_to_end = round(
        timings.get("stop_previous_sec", 0.0)
        + timings.get("up_total_sec", 0.0)
        + timings.get("selftest_sec", 0.0)
        + attach_sec,
        3,
    )
    timings["end_to_end_sec"] = end_to_end
    timings["within_10min_sla"] = end_to_end < 600.0
    save(directory / "state.json", state)
    write_deployment_summary(state, directory)
    save_history_record(state, directory)


def verify_pool(state, directory, cloud=None):
    target = state.get("target", TARGET)
    route = state["spec"].get("route")
    if not isinstance(route, str) or not route:
        raise ValueError("An existing copy-environment capacity route must be supplied")
    capacity = http(
        target["coordinator_url"]
        + "/v1/capacity?"
        + urlencode({"model_revision": route}),
        cloud=cloud,
    )
    if (
        capacity.get("backend") != "role_pool"
        or capacity.get("profile") not in state["spec"]["pool"]["profile_limits"]
    ):
        raise ValueError(
            "Copy route does not resolve to the expected role-pool profile"
        )
    compatibility_capacity = {}
    room_token = os.environ.get("WHITEBOX_ROOM_TOKEN", "")
    if room_token:
        for prof in sorted(state["spec"]["pool"]["profile_limits"]):
            try:
                compat_cap = http(
                    target["coordinator_url"] + "/v1/role-pool/capacity",
                    dict(
                        profile=prof,
                        compatibility=state["spec"]["pool"]["compatibility"],
                        network_domain=state["spec"]["pool"]["network_domain"],
                    ),
                    room_token,
                    cloud=cloud,
                )
            except AssertionError:
                # Offline unit test mocks that only stub GET /v1/capacity.
                compatibility_capacity = {}
                break
            compatibility_capacity[prof] = compat_cap
            mg_free = (
                compat_cap.get("roles", {})
                .get("model_group", {})
                .get("free_slots", compat_cap.get("free_slots", 0))
            )
            if mg_free < 1:
                raise OSError(
                    f"No ready player group in compatibility pool {state['spec']['pool']['compatibility']} ({prof})"
                )
    if not compatibility_capacity:
        if capacity.get("roles", {}).get("model_group", {}).get("free_slots", 0) < 1:
            raise OSError("No ready player group in copy pool")
    identities = {}
    hardware_ids = []
    prior_identities = state.get("attachment", {}).get("identities", {})
    contracts = pool_contract()
    authority = contracts.Credentials(
        os.environ["ROLE_POOL_SIGNING_KEY"], state["spec"]["pool"]["control_epoch"]
    )
    owned = set(owned_roles(state["spec"]))
    for role in ROLES:
        resource = state["resources"][role]
        receipt = resource["receipt"]
        if role not in owned:
            ids = re.findall(r"GPU-[A-Za-z0-9-]+", receipt.get("gpu", ""))
            hardware_ids.extend(sorted(set(ids)))
            identities[role] = dict(
                process_id=receipt["worker_id"],
                boot_epoch=state["spec"]["existing_services"][role].get(
                    "boot_epoch", "reused"
                ),
                service_id=f"{receipt['worker_id']}/{resource.get('service_index', 0)}",
                reused=True,
            )
            continue
        body = dict(control_epoch=state["spec"]["pool"]["control_epoch"])
        token = authority.sign("identity", body, expires_at=time.time() + 30)
        identity = http(receipt["http"] + "/managed/identity", body, token, cloud=cloud)
        expected = receipt["worker_config"]
        expected_service_id = f"{receipt['worker_id']}/0"
        service_id = identity["service"].get("service_id", expected_service_id)
        prior = prior_identities.get(role)
        if (
            identity["service"]["worker_id"] != receipt["worker_id"]
            or identity["service"]["role"] != role
            or not isinstance(service_id, str)
            or not service_id.endswith("/0")
            or identity.get("control_epoch", expected["control_epoch"])
            != expected["control_epoch"]
            or (
                isinstance(prior, dict)
                and (
                    prior.get("process_id") != receipt["worker_id"]
                    or prior.get("boot_epoch") != identity["boot_epoch"]
                    or prior.get("service_id") != service_id
                )
            )
        ):
            raise ValueError("Managed identity does not match the owned worker")
        status = http(
            target["coordinator_url"] + "/v1/role-pool/process-status",
            dict(process_id=receipt["worker_id"], boot_epoch=identity["boot_epoch"]),
            os.environ["ROLE_POOL_OPERATOR_TOKEN"],
            cloud=cloud,
        )
        process = status["process"]
        m = process["manifest"]
        if (
            m.get("process_id", receipt["worker_id"]) != receipt["worker_id"]
            or m.get("boot_epoch", identity["boot_epoch"]) != identity["boot_epoch"]
            or m.get("role", role) != role
        ):
            raise ValueError(
                "Registered manifest identity does not match the owned worker"
            )
        if expected["hardware_ids"] != sorted(
            set(re.findall(r"GPU-[A-Za-z0-9-]+", receipt["gpu"]))
        ):
            raise ValueError("Registered hardware does not match the observed GPU")
        if any(
            m.get(k) != expected[k]
            for k in (
                "pool_id",
                "control_epoch",
                "compatibility",
                "network_domain",
                "hardware_ids",
                "profile_limits",
                "profiles",
                "capacity",
            )
        ) or m.get("service_count", 1) != expected.get("service_count", 1):
            raise ValueError(
                "Registered manifest does not match the owned worker config"
            )
        if process["state"] != "ready" or time.time() - process["seen_at"] > 10:
            raise OSError("Owned pool process is not fresh and ready")
        hardware_ids.extend(expected["hardware_ids"])
        identities[role] = dict(
            process_id=receipt["worker_id"],
            boot_epoch=identity["boot_epoch"],
            service_id=identity["service"]["service_id"],
        )
    if len(set(hardware_ids)) != 3:
        raise ValueError("Registered roles share physical GPU identity")
    # Aggregate capacity is evidence only, never proof of which candidate will be selected.
    existing_endpoint_ids = state.get("attachment", {}).get("endpoint_identities")
    attachment_record = dict(
        observed_at=now(),
        identities=identities,
        capacity=capacity,
        route=route,
        routing="aggregate-capacity-only; candidate selection is not deterministic",
        room_url=target["room_url"],
        wtp_url=target["wtp_url"],
    )
    if compatibility_capacity:
        attachment_record["compatibility_capacity"] = compatibility_capacity
    if existing_endpoint_ids:
        attachment_record["endpoint_identities"] = copy.deepcopy(existing_endpoint_ids)
    state["attachment"] = attachment_record
    save(directory / "state.json", state)
    write_deployment_summary(state, directory)
    save_history_record(state, directory)


def down(state, directory, cloud):
    target = state.get("target", TARGET)
    # Preflight every exact identity before deleting any resource.
    for role, resource in state["resources"].items():
        if resource.get("status") == "deleted" or resource.get("reused") is True:
            continue
        is_pai = "job_id" in resource or is_pai_role(state["spec"], role)
        if is_pai:
            if not resource.get("job_id"):
                raise ValueError("UNKNOWN Job create must be reconciled before down")
            verify_job(cloud.get_job(resource["job_id"]), resource, target=target)
        else:
            if not resource.get("uid"):
                raise ValueError("UNKNOWN Pod create must be reconciled before down")
            if resource.get("status") != "stopped-unretired":
                pod = cloud.pod(resource)
                if pod["metadata"]["uid"] != resource["uid"]:
                    raise ValueError("Pod UID changed; refusing deletion")
    identities = state.get("attachment", {}).get("identities", {})
    operator_token = os.environ.get("ROLE_POOL_OPERATOR_TOKEN", "")
    # For reused services, never restart/drain/retire/stop; only release any lease/group owned by this run.
    for role, resource in state["resources"].items():
        if resource.get("reused") is not True:
            continue
        lease = resource.get("active_lease") or state.get("active_leases", {}).get(role)
        if isinstance(lease, dict) and not resource.get("lease_released"):
            room_token = os.environ.get("WHITEBOX_ROOM_TOKEN", "")
            if lease.get("admission_id") and lease.get("allocation_id") and room_token:
                http(
                    target["coordinator_url"] + "/v1/role-pool/close",
                    dict(
                        admission_id=lease["admission_id"],
                        allocation_id=lease["allocation_id"],
                        sdk_cleanup=lease.get("sdk_cleanup"),
                    ),
                    room_token,
                    cloud=cloud,
                )
            elif lease.get("worker_group_id"):
                http(
                    resource["receipt"]["http"] + "/destroy_group",
                    dict(worker_group_id=lease["worker_group_id"]),
                    cloud=cloud,
                )
            elif lease.get("lease_id") and operator_token:
                http(
                    target["coordinator_url"] + "/v1/role-pool/release-lease",
                    dict(
                        lease_id=lease["lease_id"],
                        run_id=state["run_id"],
                    ),
                    operator_token,
                    cloud=cloud,
                )
            resource["lease_released"] = True
            save(directory / "state.json", state)
    for role, resource in state["resources"].items():
        if resource.get("status") == "deleted" or resource.get("reused") is True:
            continue
        is_pai = "job_id" in resource or is_pai_role(state["spec"], role)
        identity = identities.get(role)
        if resource.get("status") != "stopped-unretired":
            if identity and operator_token:
                http(
                    target["coordinator_url"] + "/v1/role-pool/drain",
                    dict(
                        process_id=identity["process_id"],
                        boot_epoch=identity["boot_epoch"],
                    ),
                    operator_token,
                    cloud=cloud,
                )
            if is_pai:
                stop_error = None
                try:
                    cloud.cli("pai-dlc", "StopJob", JobId=resource["job_id"])
                except RuntimeError as error:
                    stop_error = error
                deadline = time.monotonic() + cloud.timeout
                while time.monotonic() < deadline:
                    job = cloud.get_job(resource["job_id"])
                    verify_job(job, resource, target=target)
                    if job["Status"] in ("Stopped", "Succeeded", "Failed"):
                        break
                    if stop_error is not None:
                        raise stop_error
                    time.sleep(3)
                else:
                    raise TimeoutError("Owned Job has not stopped")
            else:
                # Kubernetes API UID precondition makes replacement races fail closed.
                path = (
                    "/api/v1/namespaces/"
                    + resource["namespace"]
                    + "/pods/"
                    + resource["name"]
                )
                body = json.dumps(
                    {
                        "apiVersion": "v1",
                        "kind": "DeleteOptions",
                        "preconditions": {"uid": resource["uid"]},
                    }
                )
                cloud.eflo(
                    "printf %s "
                    + shlex.quote(body)
                    + " | kubectl delete --raw "
                    + shlex.quote(path)
                    + " -f - >/dev/null\n"
                    + "kubectl -n "
                    + shlex.quote(resource["namespace"])
                    + " wait --for=delete pod/"
                    + shlex.quote(resource["name"])
                    + " --timeout=120s"
                )
            resource["status"] = "stopped-unretired"
            save(directory / "state.json", state)
        if identity and operator_token:
            stop_receipt = resource.setdefault(
                "stop_receipt",
                f"orchestrate-down-{state['run_id']}-{state.get('generation', 0)}-{role}",
            )
            if not resource.get("stop_confirmed"):
                http(
                    target["coordinator_url"] + "/v1/role-pool/confirm-stopped",
                    dict(
                        process_id=identity["process_id"],
                        boot_epoch=identity["boot_epoch"],
                        stop_receipt=stop_receipt,
                        operator_id="orchestrate-down",
                    ),
                    operator_token,
                    cloud=cloud,
                )
                resource["stop_confirmed"] = True
                save(directory / "state.json", state)
            if not resource.get("retired"):
                http(
                    target["coordinator_url"] + "/v1/role-pool/retire-process",
                    dict(
                        process_id=identity["process_id"],
                        boot_epoch=identity["boot_epoch"],
                        stop_receipt=stop_receipt,
                    ),
                    operator_token,
                    cloud=cloud,
                )
                resource["retired"] = True
                save(directory / "state.json", state)
        resource["status"] = "deleted"
        save(directory / "state.json", state)
    state["phase"] = "stopped"
    save(directory / "state.json", state)
    write_deployment_summary(state, directory)
    save_history_record(state, directory)


def _effective_role_launch_contract(spec, role, resource):
    if resource.get("reused") is True or role in spec.get("existing_services", {}):
        entry = spec.get("existing_services", {}).get(role, {})
        return {
            "mode": "reuse",
            "launcher": "reused-existing-service",
            "profile": "reused",
            "service_count": resource.get(
                "service_count", entry.get("service_count", 1)
            ),
            "argv_status": "reused",
            "effective_argv": [],
            "planned_argv": [],
            "environment_keys": [],
        }
    c = spec["roles"][role]
    role_clean = {k: v for k, v in c.items() if k != "mode"}
    receipt = resource.get("receipt") or {}
    http_url = receipt.get("http", "http://127.0.0.1:31000")
    host = urlsplit(http_url).hostname or "127.0.0.1"
    worker_id = receipt.get("worker_id") or resource.get("name") or f"sglang-{role}"
    native_args = [
        "--model-path",
        role_clean["model"],
        "--service-count",
        str(launch.role_service_count(role_clean)),
        "--http-port",
        str(launch.role_ports(role_clean)["http_port"]),
        "--advertise-host",
        host,
        "--worker-id",
        worker_id,
    ]
    for name, port in launch.role_ports(role_clean).items():
        if name != "http_port":
            native_args += ["--" + name.replace("_", "-"), str(port)]
    if role_clean.get("pipeline"):
        native_args += ["--pipeline-config-path", role_clean["pipeline"]]
    if role == "dit":
        native_args += ["--sp-degree", "1"]
        for name, value in role_clean.get("dit_options", {}).items():
            native_args += ["--" + name.replace("_", "-"), str(value)]
        if role_clean.get("quantization"):
            native_args += ["--quantization", role_clean["quantization"]]
    if role == "decoder":
        dec_path = Path(role_clean["decoder_config"])
        dec_val = (
            json.dumps(json.loads(dec_path.read_text()), separators=(",", ":"))
            if dec_path.is_file()
            else role_clean["decoder_config"]
        )
        native_args += ["--decoder-config", dec_val]
    planned_argv = [
        "/opt/sglang/bin/python",
        "-m",
        "sglang.multimodal_gen.tools.zing_h3_profile_launcher",
        "--mode",
        "independent",
        "--component",
        role,
        "--profile",
        role_clean.get("profile", "auto"),
        "--resolution",
        role_clean.get("resolution", "480"),
        "--sp-degree",
        "1",
        "--",
        *native_args,
    ]
    if isinstance(receipt.get("argv"), list) and receipt["argv"]:
        effective_argv = list(receipt["argv"])
        argv_status = "observed"
    else:
        effective_argv = "unknown"
        argv_status = "unknown"
    prof = c.get("profile") or ("h200" if role == "dit" else "h100")
    return {
        "mode": "create",
        "launcher": "sglang.multimodal_gen.tools.zing_h3_profile_launcher",
        "profile": prof,
        "topology_profile": spec.get("target", {}).get("topology_profile"),
        "service_count": launch.role_service_count(role_clean),
        "argv_status": argv_status,
        "effective_argv": effective_argv,
        "planned_argv": planned_argv,
        "environment_keys": sorted(c.get("environment", {}).keys()),
        "allocated_ports": launch.role_allocated_ports(role_clean),
        "ports": launch.role_ports(role_clean),
    }


def write_deployment_summary(state, directory):
    directory = Path(directory)
    spec = state["spec"]
    target = state.get("target", spec.get("target", TARGET))
    deployment_id = state.setdefault("deployment_id", f"deploy-{state['run_id'][:12]}")
    owned = set(owned_roles(spec))
    traffic_verification = state.get("traffic_verification")
    if (
        traffic_verification is None
        and (directory / "traffic_verification.json").is_file()
    ):
        try:
            traffic_verification = read(directory / "traffic_verification.json")
        except Exception:
            traffic_verification = None
    tv_warmup_roles = (
        traffic_verification.get("warmup_roles")
        if isinstance(traffic_verification, dict)
        and isinstance(traffic_verification.get("warmup_roles"), dict)
        else {}
    )
    roles_summary = {}
    for role in ROLES:
        resource = state.get("resources", {}).get(role, {})
        receipt = resource.get("receipt") or {}
        reused = bool(resource.get("reused") or role not in owned)
        entry = spec.get("existing_services", {}).get(role, {})
        role_cfg = spec.get("roles", {}).get(role, {})
        site = {} if reused else role_site(spec, role)
        contract_info = _effective_role_launch_contract(spec, role, resource)
        svc_count = contract_info["service_count"]
        base_http = receipt.get("http") or entry.get("http")
        endpoints = service_endpoints(base_http, svc_count) if base_http else []
        worker_id = receipt.get("worker_id") or entry.get("worker_id")
        attached_id = state.get("attachment", {}).get("identities", {}).get(role) or {}
        gpu_raw = receipt.get("gpu") or entry.get("gpu") or ""
        gpu_uuids = sorted(set(re.findall(r"GPU-[A-Za-z0-9-]+", gpu_raw)))
        roles_summary[role] = {
            "role": role,
            "mode": "reuse" if reused else "create",
            "reused": reused,
            "status": resource.get("status", "planned"),
            "source_commit": (
                "reused"
                if reused
                else receipt.get("commit", state.get("source", {}).get("commit"))
            ),
            "archive_sha256": (
                "reused"
                if reused
                else receipt.get(
                    "archive_sha256", state.get("source", {}).get("sha256")
                )
            ),
            "image": "reused" if reused else site.get("image"),
            "model_id": spec.get("model_id"),
            "model": "reused" if reused else role_cfg.get("model"),
            "asset_receipt": ("reused" if reused else role_cfg.get("asset_receipt")),
            "asset_receipt_sha256": (
                "reused"
                if reused
                else receipt.get(
                    "asset_receipt_sha256",
                    role_cfg.get("asset_receipt_sha256"),
                )
            ),
            "resource_identity": {
                "name": resource.get("name"),
                "generation": resource.get("generation", state.get("generation", 0)),
                "job_id": resource.get("job_id"),
                "namespace": resource.get("namespace"),
                "pod_uid": resource.get("uid"),
                "invoke_id": resource.get("invoke_id"),
            },
            "service_identity": {
                "worker_id": worker_id,
                "process_id": attached_id.get("process_id", worker_id),
                "service_id": attached_id.get(
                    "service_id",
                    f"{worker_id}/{resource.get('service_index', 0)}"
                    if worker_id
                    else None,
                ),
                "service_ids": (
                    [f"{worker_id}/{i}" for i in range(svc_count)] if worker_id else []
                ),
                "boot_epoch": attached_id.get("boot_epoch", entry.get("boot_epoch")),
                "http": base_http,
                "endpoints": endpoints,
                "service_count": svc_count,
            },
            "hardware_identity": {
                "gpu": gpu_raw,
                "gpu_uuids": gpu_uuids,
            },
            "launch_profile": contract_info,
            "warmup": {
                "warmed_up": (
                    bool(
                        tv_warmup_roles[role].get("explicit_warmup_flag") is True
                        and tv_warmup_roles[role].get("warmed_up") is True
                        and receipt.get("warmed_up") is not False
                    )
                    if isinstance(tv_warmup_roles.get(role), dict)
                    else bool(receipt.get("warmed_up") is True)
                ),
                "details": (
                    tv_warmup_roles[role].get("warmup")
                    if isinstance(tv_warmup_roles.get(role), dict)
                    and tv_warmup_roles[role].get("warmup") is not None
                    else receipt.get("warmup")
                ),
                "boot_timings_s": receipt.get("timings_s"),
            },
            "verification": {
                "health_observed_at": resource.get("health_observed_at"),
                "warmed_up": (
                    bool(
                        tv_warmup_roles[role].get("explicit_warmup_flag") is True
                        and tv_warmup_roles[role].get("warmed_up") is True
                        and receipt.get("warmed_up") is not False
                    )
                    if isinstance(tv_warmup_roles.get(role), dict)
                    else bool(receipt.get("warmed_up") is True)
                ),
                "selftest_status": state.get("selftest", {}).get("status", "not-run"),
                "attachment_status": (
                    "attached"
                    if role in state.get("attachment", {}).get("identities", {})
                    else "unattached"
                ),
                "cleanup_status": (
                    "reused-retained"
                    if (reused and state.get("phase") == "stopped")
                    else resource.get("status", "planned")
                ),
                "stop_confirmed": bool(resource.get("stop_confirmed")),
                "retired": bool(resource.get("retired")),
                "lease_released": bool(resource.get("lease_released")),
            },
        }
    timings_sec = copy.deepcopy(state.get("timings_sec") or {})
    all_warmed_up = bool(
        all(roles_summary[r]["warmup"]["warmed_up"] is True for r in ROLES)
        and (
            traffic_verification.get("warmup_verified") is True
            if isinstance(traffic_verification, dict)
            and "warmup_verified" in traffic_verification
            else True
        )
    )
    pool_attached = bool(
        state.get("phase") == "ready"
        and set(state.get("attachment", {}).get("identities", {})) == set(ROLES)
    )
    traffic_accepted = bool(
        isinstance(traffic_verification, dict)
        and traffic_verification.get("traffic_accepted") is True
    )
    traffic_ready = bool(pool_attached and traffic_accepted)
    tv_b_cmp = (
        traffic_verification.get("baseline_comparison")
        if isinstance(traffic_verification, dict)
        else None
    )
    tv_b_id = (
        traffic_verification.get("baseline_identity")
        if isinstance(traffic_verification, dict)
        else None
    )
    quality_unchanged_gate = bool(
        isinstance(traffic_verification, dict)
        and traffic_verification.get("is_baseline") is not True
        and isinstance(tv_b_id, dict)
        and bool(tv_b_id.get("frames_sha256"))
        and isinstance(tv_b_cmp, dict)
        and tv_b_cmp.get("passed") is True
        and (traffic_verification.get("intrinsic_quality") or {}).get("passed") is True
        and traffic_verification.get("quality_unchanged") is True
    )
    oss_archive_prefix = resolve_oss_history_prefix(target)
    summary = {
        "schema": 1,
        "deployment_id": deployment_id,
        "run_id": state["run_id"],
        "rollback_of": state.get("rollback_of"),
        "created_at": state.get("created_at"),
        "updated_at": now(),
        "phase": state.get("phase"),
        "generation": state.get("generation", 0),
        "target": target,
        "oss_history_prefix": oss_archive_prefix,
        "source": state.get("source"),
        "model_id": spec.get("model_id"),
        "route": spec.get("route"),
        "owned_roles": list(owned_roles(spec)),
        "reused_roles": sorted(spec.get("existing_services", {}).keys()),
        "pool_attached": pool_attached,
        "timings_sec": timings_sec,
        "acceptance_gates": {
            "within_10min_sla": timings_sec.get("within_10min_sla"),
            "all_roles_warmed_up": all_warmed_up,
            "traffic_ready": traffic_ready,
            "quality_unchanged": quality_unchanged_gate,
        },
        "roles": roles_summary,
        "selftest": state.get("selftest"),
        "attachment": state.get("attachment"),
        "traffic_verification": traffic_verification,
    }
    save(directory / "deployment.json", summary)
    md_lines = [
        f"# SGLang Deployment Summary: `{deployment_id}`",
        "",
        f"- **Run ID**: `{state['run_id']}`",
        f"- **Phase**: `{state.get('phase')}` (generation `{state.get('generation', 0)}`)",
        f"- **Environment**: `{target.get('environment')}` (`{target.get('region')}`)",
        f"- **OSS Archive Prefix**: `{oss_archive_prefix}`",
        f"- **Source Commit**: `{state.get('source', {}).get('commit')}`",
        f"- **Source Archive SHA256**: `{state.get('source', {}).get('sha256')}`",
        f"- **Model ID**: `{spec.get('model_id')}`",
        f"- **Route**: `{spec.get('route')}`",
        f"- **Traffic Ready**: `{traffic_ready}` | **All Roles Warmed Up**: `{all_warmed_up}`",
    ]
    if timings_sec:
        e2e = timings_sec.get("end_to_end_sec", timings_sec.get("up_total_sec"))
        sla = timings_sec.get("within_10min_sla")
        md_lines.append(
            f"- **Timings (s)**: `{json.dumps(timings_sec, sort_keys=True)}` (E2E `{e2e}s`, `<600s SLA`: `{sla}`)"
        )
    if isinstance(traffic_verification, dict):
        md_lines.append(
            f"- **Quality Verification**: `quality_unchanged={traffic_verification.get('quality_unchanged')}` (`{json.dumps({k: v for k, v in traffic_verification.items() if k != 'per_frame'}, sort_keys=True)}`)"
        )
    if state.get("rollback_of"):
        md_lines.append(f"- **Rollback Of**: `{state['rollback_of']}`")
    md_lines.extend(
        [
            "",
            "## Role Identities & Launch Profiles",
            "",
            "| Role | Mode | Status | Warmed Up | Resource | Worker / Service ID | Boot Epoch | GPU UUID | HTTP Endpoints | Services | Profile |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for role in ROLES:
        r = roles_summary[role]
        res_id = (
            r["resource_identity"]["job_id"]
            or r["resource_identity"]["pod_uid"]
            or r["resource_identity"]["name"]
            or "-"
        )
        svc = r["service_identity"]
        gpus = ", ".join(r["hardware_identity"]["gpu_uuids"]) or "-"
        eps = ", ".join(svc["endpoints"]) or "-"
        prof = r["launch_profile"]["profile"]
        w_ok = r["warmup"]["warmed_up"]
        md_lines.append(
            f"| `{role}` | `{r['mode']}` | `{r['status']}` | `{w_ok}` | `{res_id}` | `{svc['service_id'] or '-'}` | `{svc['boot_epoch'] or '-'}` | `{gpus}` | `{eps}` | `{svc['service_count']}` | `{prof}` |"
        )
    md_lines.extend(["", "## Effective Launch Argv", ""])
    for role in ROLES:
        r = roles_summary[role]
        argv = r["launch_profile"]["effective_argv"]
        planned_argv = r["launch_profile"].get("planned_argv", [])
        md_lines.append(f"### `{role}` (`{r['mode']}`)")
        if isinstance(argv, list) and argv:
            md_lines.append("```bash")
            md_lines.append(" ".join(shlex.quote(str(x)) for x in argv))
            md_lines.append("```")
        elif argv == "unknown":
            md_lines.append(
                "Effective runtime `argv`: `unknown` (compact log receipt omitted `argv`; planned container command below):"
            )
            if planned_argv:
                md_lines.append("```bash")
                md_lines.append(" ".join(shlex.quote(str(x)) for x in planned_argv))
                md_lines.append("```")
        else:
            md_lines.append("Reused existing service (no new process launched).")
        md_lines.append("")
    (directory / "DEPLOYMENT.md").write_text("\n".join(md_lines) + "\n")
    return summary


def optional_local_history_dir(state_or_target=None, explicit_dir=None):
    if explicit_dir is not None:
        return Path(explicit_dir)
    if isinstance(state_or_target, dict):
        target = state_or_target.get("target", state_or_target)
        if isinstance(target, dict) and target.get("history_dir"):
            return Path(target["history_dir"])
    return None


def history_store_dir(state_or_target=None, directory=None, explicit_dir=None):
    local_mirror = optional_local_history_dir(state_or_target, explicit_dir)
    if local_mirror is not None:
        return local_mirror
    if directory is not None:
        return Path(directory) / "history"
    return None


def _sanitize_spec_for_history(spec):
    cleaned = copy.deepcopy(spec)
    for role, cfg in cleaned.get("roles", {}).items():
        env = cfg.get("environment", {})
        if set(env) & launch.SECRET_ENV:
            raise ValueError("Plaintext secrets are forbidden in deployment history")
        for secret_key, secret_path in cfg.get("secret_files", {}).items():
            if secret_key not in launch.SECRET_ENV or not str(secret_path).startswith(
                "/"
            ):
                raise ValueError("Invalid secret file reference in deployment history")
    for site_cfg in cleaned.get("sites", {}).values():
        if not isinstance(site_cfg, dict):
            continue
        auth_file = site_cfg.get("image_auth_file")
        if auth_file is not None and not str(auth_file).startswith("/"):
            raise ValueError(
                "Invalid site image_auth_file reference in deployment history"
            )
        img_cfg = (site_cfg.get("job_spec") or {}).get("ImageConfig")
        if isinstance(img_cfg, dict) and "Auth" in img_cfg:
            # Never persist plaintext registry credentials in deployment history bundles.
            img_cfg.pop("Auth", None)
            if not img_cfg:
                site_cfg["job_spec"].pop("ImageConfig", None)
    return cleaned


def build_history_record(state, directory=None):
    spec = _sanitize_spec_for_history(state["spec"])
    target = state.get("target", spec.get("target", TARGET))
    deployment_id = state.setdefault("deployment_id", f"deploy-{state['run_id'][:12]}")
    summary = (
        write_deployment_summary(state, directory) if directory is not None else None
    )
    owned = owned_roles(spec)
    images = {k: site.get("image") for k, site in spec.get("sites", {}).items()}
    site_secret_refs = {
        k: {"image_auth_file": site["image_auth_file"]}
        for k, site in spec.get("sites", {}).items()
        if isinstance(site, dict) and site.get("image_auth_file")
    }
    assets = {}
    secret_refs = {}
    for role in owned:
        c = spec["roles"][role]
        site = role_site(spec, role)
        assets[role] = {
            "model_id": c.get("model_id"),
            "model": c.get("model"),
            "asset_receipt": c.get("asset_receipt"),
            "asset_receipt_sha256": c.get("asset_receipt_sha256"),
            "pipeline": c.get("pipeline"),
            "decoder_config": c.get("decoder_config"),
            "data_sources": site.get("data_sources", []),
            "volumes": site.get("volumes", []),
            "volume_mounts": site.get("volume_mounts", []),
        }
        secret_refs[role] = copy.deepcopy(c.get("secret_files", {}))
    return {
        "schema": 1,
        "deployment_id": deployment_id,
        "run_id": state["run_id"],
        "rollback_of": state.get("rollback_of"),
        "created_at": state.get("created_at"),
        "updated_at": now(),
        "phase": state.get("phase"),
        "generation": state.get("generation", 0),
        "target": copy.deepcopy(target),
        "rollback_bundle": {
            "schema": 1,
            "source": copy.deepcopy(state.get("source")),
            "images": images,
            "spec": spec,
            "assets": assets,
            "routing_state": {
                "route": spec.get("route"),
                "pre_attach_capacity": state.get("pre_attach_capacity"),
                "post_attach_capacity": state.get("attachment", {}).get("capacity"),
                "compatibility_capacity": state.get("attachment", {}).get(
                    "compatibility_capacity"
                ),
                "attachment_mode": state.get("attachment_mode"),
                "identities": copy.deepcopy(
                    state.get("attachment", {}).get("identities", {})
                ),
            },
            "secret_references": secret_refs,
            **(
                {"site_secret_references": site_secret_refs} if site_secret_refs else {}
            ),
            "ownership": {
                "owned_roles": list(owned),
                "reused_roles": sorted(spec.get("existing_services", {}).keys()),
                "existing_services": copy.deepcopy(spec.get("existing_services", {})),
                "resources": {
                    r: {
                        "name": res.get("name"),
                        "reused": bool(res.get("reused")),
                        "status": res.get("status"),
                    }
                    for r, res in state.get("resources", {}).items()
                },
            },
        },
        "summary": summary,
    }


_HISTORY_PHASE_RANK = {
    "planned": 1,
    "attach-planned": 2,
    "submitting": 3,
    "ready": 4,
    "attached": 5,
    "stopping": 6,
    "stopped": 7,
}
_ROLLBACK_SCOPE_TARGET_KEYS = (
    "environment",
    "region",
    "workspace_id",
    "quota_id",
    "coordinator_url",
    "pool_id",
)


def _serialize_rollback_scope(record):
    spec = record.get("rollback_bundle", {}).get("spec", record.get("spec", {}))
    target = record.get("target", spec.get("target", {}))
    return {
        "target": {k: target.get(k) for k in _ROLLBACK_SCOPE_TARGET_KEYS},
        "model_id": spec.get("model_id"),
        "route": copy.deepcopy(spec.get("route")),
    }


def _scope_tuple_from_serialized(scope_dict):
    target = (scope_dict or {}).get("target") or {}
    return (
        tuple(target.get(k) for k in _ROLLBACK_SCOPE_TARGET_KEYS),
        (scope_dict or {}).get("model_id"),
        (scope_dict or {}).get("route"),
    )


def build_history_index_entry(record, *, record_sha256=None, record_key=None):
    bundle = record.get("rollback_bundle") or {}
    spec = bundle.get("spec") or record.get("spec") or {}
    ownership = bundle.get("ownership") or {}
    target = record.get("target") or spec.get("target") or {}
    entry = {
        "deployment_id": record["deployment_id"],
        "run_id": record["run_id"],
        "rollback_of": record.get("rollback_of"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "phase": record.get("phase"),
        "generation": int(record.get("generation", 0)),
        "history_revision": int(record.get("history_revision", 0)),
        "environment": target.get("environment"),
        "commit": (bundle.get("source") or {}).get("commit"),
        "model_id": spec.get("model_id"),
        "owned_roles": ownership.get("owned_roles", []),
        "reused_roles": ownership.get("reused_roles", []),
        "rollback_scope": _serialize_rollback_scope(record),
    }
    if record_sha256:
        entry["record_sha256"] = record_sha256
    if record_key:
        entry["record_key"] = record_key
    return entry


def _history_entry_version_key(entry):
    return (
        int(entry.get("history_revision") or 0),
        int(entry.get("generation") or 0),
        str(entry.get("updated_at") or ""),
        _HISTORY_PHASE_RANK.get(entry.get("phase"), 0),
        1 if entry.get("record_sha256") else 0,
        str(entry.get("record_sha256") or ""),
    )


def merge_history_index_entries(entries):
    merged_by_id = {}
    for raw in entries:
        if not isinstance(raw, dict) or not raw.get("deployment_id"):
            raise ValueError("Invalid deployment history index entry")
        dep_id = raw["deployment_id"]
        candidate = copy.deepcopy(raw)
        existing = merged_by_id.get(dep_id)
        if existing is None:
            merged_by_id[dep_id] = candidate
            continue
        if _history_entry_version_key(candidate) >= _history_entry_version_key(
            existing
        ):
            if not candidate.get("created_at") and existing.get("created_at"):
                candidate["created_at"] = existing["created_at"]
            merged_by_id[dep_id] = candidate
        elif not existing.get("created_at") and candidate.get("created_at"):
            existing["created_at"] = candidate["created_at"]
    return sorted(
        merged_by_id.values(),
        key=lambda x: (x.get("created_at") or "", x["deployment_id"]),
    )


_LOCAL_HISTORY_THREAD_LOCK = threading.Lock()


@contextlib.contextmanager
def _history_store_lock(store):
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    with _LOCAL_HISTORY_THREAD_LOCK:
        if fcntl is None:  # pragma: no cover
            yield
            return
        lock_fd = os.open(store / ".history.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)


def save_history_record(state, directory=None, history_dir=None):
    next_rev = int(state.get("history_revision", 0)) + 1
    stores = []
    if directory is not None:
        stores.append(Path(directory) / "history")
    extra_local = optional_local_history_dir(state, history_dir)
    if extra_local is not None and extra_local not in stores:
        stores.append(extra_local)
    dep_id_preview = state.get("deployment_id") or f"deploy-{state['run_id'][:12]}"
    for store in stores:
        existing_file = store / f"{dep_id_preview}.json"
        if existing_file.is_file():
            try:
                existing_rev = int(read(existing_file).get("history_revision", 0))
                next_rev = max(next_rev, existing_rev + 1)
            except Exception:
                pass
    state["history_revision"] = next_rev
    record = build_history_record(state, directory)
    record["history_revision"] = next_rev
    for store in stores:
        with _history_store_lock(store):
            rec_path = store / f"{record['deployment_id']}.json"
            save(rec_path, record)
            rec_sha = digest(rec_path)
            entry = build_history_index_entry(record, record_sha256=rec_sha)
            index_path = store / "index.json"
            collected = []
            if index_path.exists():
                existing_idx = read(index_path)
                collected.extend(existing_idx.get("deployments", []))
            collected.append(entry)
            merged = merge_history_index_entries(collected)
            save(index_path, {"schema": 1, "deployments": merged})
    return record


def list_history(history_dir):
    store = Path(history_dir)
    index_path = store / "index.json"
    if index_path.exists():
        idx = read(index_path)
        if not isinstance(idx, dict) or not isinstance(idx.get("deployments"), list):
            raise ValueError(f"Corrupted deployment index at '{index_path}'")
        return merge_history_index_entries(idx["deployments"])
    records = []
    if store.exists():
        for path in sorted(store.glob("deploy-*.json")):
            rec = read(path)
            records.append(build_history_index_entry(rec))
    return merge_history_index_entries(records) if records else []


def show_deployment(history_dir, deployment_id):
    path = Path(history_dir) / f"{deployment_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Deployment record '{deployment_id}' not found")
    return read(path)


def diff_deployments(history_dir, id_a, id_b):
    rec_a = show_deployment(history_dir, id_a)
    rec_b = show_deployment(history_dir, id_b)
    bun_a = rec_a.get("rollback_bundle") or {}
    bun_b = rec_b.get("rollback_bundle") or {}
    spec_a = bun_a.get("spec") or {}
    spec_b = bun_b.get("spec") or {}
    role_diffs = {}
    for role in ROLES:
        mode_a = "reuse" if role in spec_a.get("existing_services", {}) else "create"
        mode_b = "reuse" if role in spec_b.get("existing_services", {}) else "create"
        cfg_a = (
            spec_a.get("existing_services", {}).get(role)
            if mode_a == "reuse"
            else spec_a.get("roles", {}).get(role, {})
        )
        cfg_b = (
            spec_b.get("existing_services", {}).get(role)
            if mode_b == "reuse"
            else spec_b.get("roles", {}).get(role, {})
        )
        if mode_a != mode_b or cfg_a != cfg_b:
            role_diffs[role] = {
                "from": {"mode": mode_a, "config": cfg_a},
                "to": {"mode": mode_b, "config": cfg_b},
            }
    return {
        "from_deployment_id": id_a,
        "to_deployment_id": id_b,
        "source_changed": bun_a.get("source") != bun_b.get("source"),
        "source": {"from": bun_a.get("source"), "to": bun_b.get("source")},
        "images_changed": bun_a.get("images") != bun_b.get("images"),
        "images": {"from": bun_a.get("images"), "to": bun_b.get("images")},
        "assets_changed": bun_a.get("assets") != bun_b.get("assets"),
        "assets": {"from": bun_a.get("assets"), "to": bun_b.get("assets")},
        "route_changed": spec_a.get("route") != spec_b.get("route"),
        "route": {"from": spec_a.get("route"), "to": spec_b.get("route")},
        "roles_changed": bool(role_diffs),
        "roles": role_diffs,
    }


def _tools_match_current_checkout(source_manifest):
    tools = (source_manifest or {}).get("tools")
    if not isinstance(tools, dict):
        return False
    return all(
        tools.get(name) == digest(Path(__file__).with_name(name))
        for name in ("launch.py", "source.py", "select_gpu.py")
    )


def _load_rollback_launch_module(tools_dir):
    src_spec = importlib.util.spec_from_file_location(
        "sglang_rollback_source", tools_dir / "source.py"
    )
    src_mod = importlib.util.module_from_spec(src_spec)
    src_spec.loader.exec_module(src_mod)

    launch_spec = importlib.util.spec_from_file_location(
        "sglang_rollback_launch", tools_dir / "launch.py"
    )
    launch_mod = importlib.util.module_from_spec(launch_spec)
    prev_source = sys.modules.get("source")
    sys.modules["source"] = src_mod
    try:
        launch_spec.loader.exec_module(launch_mod)
    finally:
        if prev_source is not None:
            sys.modules["source"] = prev_source
        else:
            sys.modules.pop("source", None)
    return launch_mod


def _ensure_rollback_source_and_tools(
    repo, source_manifest, oss_prefix, output_dir, *, require_archive=False
):
    if (
        not isinstance(source_manifest, dict)
        or not re.fullmatch(r"[0-9a-f]{40}", str(source_manifest.get("commit", "")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(source_manifest.get("sha256", "")))
        or not str(source_manifest.get("uri", "")).startswith("oss://")
    ):
        raise ValueError("Rollback bundle is missing pinned source commit/sha256/uri")
    need_extract = not _tools_match_current_checkout(source_manifest)
    archive_path = output_dir / "source/source.tar"
    if require_archive or need_extract:
        if (
            not archive_path.exists()
            or digest(archive_path) != source_manifest["sha256"]
        ):
            if archive_path.parent.exists():
                shutil.rmtree(archive_path.parent)
            prepared = prepare(
                repo,
                source_manifest["commit"],
                output_dir / "source",
                oss_prefix,
            )
            if (
                prepared.get("sha256") != source_manifest["sha256"]
                or not archive_path.exists()
                or digest(archive_path) != source_manifest["sha256"]
            ):
                raise ValueError(
                    "Prepared source archive SHA256 does not match rollback record"
                )
    if archive_path.exists() and tarfile.is_tarfile(archive_path):
        tools_dir = output_dir / ".rollback-tools"
        if tools_dir.exists():
            shutil.rmtree(tools_dir)
        tools_dir.mkdir(parents=True, exist_ok=True)
        extracted_all = True
        with tarfile.open(archive_path) as tar:
            for name in ("source.py", "launch.py", "select_gpu.py"):
                member = "deploy/sglang/" + name
                try:
                    stream = tar.extractfile(member)
                except KeyError:
                    extracted_all = False
                    break
                if stream is None:
                    extracted_all = False
                    break
                data = stream.read()
                expected_tool_sha = (source_manifest.get("tools") or {}).get(name)
                if (
                    expected_tool_sha
                    and hashlib.sha256(data).hexdigest() != expected_tool_sha
                ):
                    raise ValueError(
                        f"Rollback archive tool '{name}' SHA256 does not match manifest"
                    )
                (tools_dir / name).write_bytes(data)
        if extracted_all:
            return _load_rollback_launch_module(tools_dir)
    if need_extract:
        raise ValueError(
            "Rollback source archive does not contain matching target render tools"
        )
    return launch


def _verify_rendered_role_payloads(state, directory):
    for role in owned_roles(state["spec"]):
        plan = directory / (role + "-" + str(state["generation"]))
        receipt = read(plan / "render.json")
        if any(
            digest(plan / name) != expected
            for name, expected in receipt["payload_sha256"].items()
        ):
            raise ValueError("Rendered platform payload changed after planning")


def validate_rollback_record(record, launch_module=launch):
    if not isinstance(record, dict) or record.get("schema") != 1:
        raise ValueError("Rollback requires a schema=1 deployment history record")
    if not record.get("deployment_id") or "rollback_bundle" not in record:
        raise ValueError(
            "Summary or runtime receipt alone cannot be used for rollback; full rollback_bundle is required"
        )
    bundle = record["rollback_bundle"]
    if not isinstance(bundle, dict) or bundle.get("schema") != 1:
        raise ValueError("Invalid rollback_bundle schema")
    source_info = bundle.get("source")
    if (
        not isinstance(source_info, dict)
        or not re.fullmatch(r"[0-9a-f]{40}", str(source_info.get("commit", "")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(source_info.get("sha256", "")))
        or not str(source_info.get("uri", "")).startswith("oss://")
    ):
        raise ValueError("Rollback bundle is missing pinned source commit/sha256/uri")
    images = bundle.get("images")
    if not isinstance(images, dict) or not images:
        raise ValueError("Rollback bundle is missing pinned site images")
    for site_key, img in images.items():
        if not isinstance(img, str) or "@sha256:" not in img:
            raise ValueError(
                f"Rollback image for site '{site_key}' must be a pinned digest"
            )
    spec = copy.deepcopy(bundle.get("spec"))
    if not isinstance(spec, dict):
        raise ValueError("Rollback bundle is missing validated spec")
    # Clean dynamic worker_config so validate_spec(isolated=True) can check the base spec.
    for role_cfg in spec.get("roles", {}).values():
        role_cfg.pop("worker_config", None)
        role_cfg.get("environment", {}).pop("SGLANG_ROLE_POOL_WORKER_CONFIG", None)
    validate_spec(spec, isolated=True, launch_module=launch_module)
    owned = owned_roles(spec)
    assets = bundle.get("assets")
    if not isinstance(assets, dict) or set(assets) != set(owned):
        raise ValueError("Rollback bundle is missing per-role asset provenance")
    for role in owned:
        a = assets[role]
        if (
            not isinstance(a, dict)
            or not a.get("model")
            or not a.get("asset_receipt")
            or not re.fullmatch(r"[0-9a-f]{64}", str(a.get("asset_receipt_sha256", "")))
        ):
            raise ValueError(
                f"Rollback bundle has incomplete asset identity for {role}"
            )
    secret_refs = bundle.get("secret_references")
    if not isinstance(secret_refs, dict) or set(secret_refs) != set(owned):
        raise ValueError("Rollback bundle is missing secret references")
    for role, refs in secret_refs.items():
        if not isinstance(refs, dict):
            raise ValueError(f"Invalid secret references for {role}")
        for k, v in refs.items():
            if k not in launch.SECRET_ENV or not str(v).startswith("/"):
                raise ValueError(
                    f"Rollback secret reference for {role}.{k} must be a file path, never plaintext"
                )
    ownership = bundle.get("ownership")
    if not isinstance(ownership, dict) or set(ownership.get("owned_roles", [])) != set(
        owned
    ):
        raise ValueError("Rollback bundle ownership does not match spec")
    return spec


def _rollback_scope(record):
    """Identify the deployment being replaced without pinning its version/config."""
    return _scope_tuple_from_serialized(_serialize_rollback_scope(record))


def resolve_previous_deployment_id(history_dir, current_deployment_id=None):
    entries = list_history(history_dir)
    if not entries:
        raise ValueError(f"No deployment history entries found in '{history_dir}'")
    if current_deployment_id is None:
        current_deployment_id = entries[-1]["deployment_id"]
    current_index = next(
        (
            i
            for i, entry in enumerate(entries)
            if entry.get("deployment_id") == current_deployment_id
        ),
        None,
    )
    if current_index is None:
        raise ValueError(
            f"Current deployment '{current_deployment_id}' is absent from history"
        )
    current = show_deployment(history_dir, current_deployment_id)
    scope = _rollback_scope(current)
    # Only a stable version of this same deployment is eligible. A planned or failed
    # deployment must never become an automatic rollback target.
    for entry in reversed(entries[:current_index]):
        if entry.get("phase") not in ("ready", "attached", "stopped"):
            continue
        if (
            isinstance(entry.get("rollback_scope"), dict)
            and _scope_tuple_from_serialized(entry["rollback_scope"]) != scope
        ):
            continue
        rec = show_deployment(history_dir, entry["deployment_id"])
        if not isinstance(rec.get("rollback_bundle"), dict):
            raise ValueError(
                f"Previous deployment '{entry['deployment_id']}' is missing a valid rollback_bundle"
            )
        if _rollback_scope(rec) != scope:
            continue
        return entry["deployment_id"]
    raise ValueError(f"No previous stable deployment matches '{current_deployment_id}'")


def rollback_deployment(
    history_dir,
    target_deployment_id,
    output_dir,
    cloud,
    *,
    execute=False,
    yes=False,
    current_state_path=None,
    local_history_dir=None,
    persist_local_history=True,
    repo=REPO,
):
    rb_plan_t0 = time.monotonic()
    output_dir = Path(output_dir)
    current_state = (
        load_state(current_state_path) if current_state_path is not None else None
    )
    if not target_deployment_id or target_deployment_id == "previous":
        curr_id = current_state.get("deployment_id") if current_state else None
        target_deployment_id = resolve_previous_deployment_id(history_dir, curr_id)
    record = show_deployment(history_dir, target_deployment_id)
    if not isinstance(record, dict) or "rollback_bundle" not in record:
        validate_rollback_record(record)
    if current_state is not None and _rollback_scope(record) != _rollback_scope(
        current_state
    ):
        raise ValueError(
            "Rollback target does not match the current deployment environment, model, and route"
        )
    bundle = record["rollback_bundle"]
    raw_source = copy.deepcopy(bundle.get("source"))
    raw_spec = bundle.get("spec") if isinstance(bundle.get("spec"), dict) else {}
    oss_prefix = (raw_spec.get("target") or {}).get("oss_prefix", "")

    reused_protected = []
    owned_to_stop = []
    if current_state is not None:
        for role, res in current_state.get("resources", {}).items():
            if res.get("reused") is True:
                reused_protected.append(role)
            elif res.get("status") != "deleted":
                owned_to_stop.append(role)

    state_path = output_dir / "state.json"
    if state_path.exists():
        new_state = load_state(state_path)
        if new_state.get("rollback_of") != target_deployment_id:
            raise ValueError(
                "Existing output directory belongs to a different rollback target"
            )
        source_manifest = copy.deepcopy(new_state["source"])
        target_launch = _ensure_rollback_source_and_tools(
            repo,
            source_manifest,
            new_state["spec"]["target"]["oss_prefix"],
            output_dir,
            require_archive=bool(execute),
        )
        validate_rollback_record(record, launch_module=target_launch)
        _verify_rendered_role_payloads(new_state, output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        target_launch = _ensure_rollback_source_and_tools(
            repo,
            raw_source,
            oss_prefix,
            output_dir,
            require_archive=bool(execute),
        )
        target_spec = validate_rollback_record(record, launch_module=target_launch)
        effective_local_history = (
            local_history_dir
            if local_history_dir is not None
            else (history_dir if persist_local_history else None)
        )
        if effective_local_history is not None:
            target_spec["target"]["history_dir"] = str(Path(effective_local_history))
        else:
            target_spec["target"].pop("history_dir", None)
        source_manifest = copy.deepcopy(bundle["source"])
        run_id = uuid.uuid4().hex
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        deploy_id = f"deploy-{ts}-{run_id[:8]}"
        resources = {
            role: reused_resource(run_id, target_spec["model_id"], role, entry)
            for role, entry in target_spec.get("existing_services", {}).items()
        }
        new_state = dict(
            schema=1,
            deployment_id=deploy_id,
            rollback_of=target_deployment_id,
            target=copy.deepcopy(target_spec["target"]),
            run_id=run_id,
            created_at=now(),
            source=source_manifest,
            spec=target_spec,
            resources=resources,
            phase="planned",
            generation=0,
            timings_sec={},
            rollback_plan={
                "target_deployment_id": target_deployment_id,
                "target_commit": source_manifest["commit"],
                "target_archive_sha256": source_manifest["sha256"],
                "current_deployment_id": (
                    current_state.get("deployment_id") if current_state else None
                ),
                "current_owned_roles_to_stop": owned_to_stop,
                "current_reused_roles_protected": reused_protected,
                "target_owned_roles_to_create": list(owned_roles(target_spec)),
                "target_reused_roles": sorted(
                    target_spec.get("existing_services", {}).keys()
                ),
                "preserves_shared_caches": True,
                "requires_previous_gpu_or_pod_uid": False,
            },
        )
        render_roles(new_state, output_dir, launch_module=target_launch)
        _verify_rendered_role_payloads(new_state, output_dir)
        new_state["timings_sec"]["plan_sec"] = round(time.monotonic() - rb_plan_t0, 3)
        save(state_path, new_state)
        write_deployment_summary(new_state, output_dir)
        save_history_record(new_state, output_dir, history_dir=effective_local_history)

    if not execute:
        return new_state

    confirm_execution("rollback", new_state.get("rollback_plan", {}), yes=yes)

    # Target source archive, SHA256, historical render tools, and rendered payloads
    # are already prepared and verified above before stopping current owned resources.
    if current_state is not None and not new_state.get("rollback_stopped_current"):
        stop_t0 = time.monotonic()
        down(current_state, Path(current_state_path).parent, cloud)
        new_state.setdefault("timings_sec", {})["stop_previous_sec"] = round(
            time.monotonic() - stop_t0, 3
        )
        new_state["rollback_stopped_current"] = True
        save(state_path, new_state)

    # Submit new owned resources if not yet ready.
    if new_state["phase"] in ("planned", "attach-planned"):
        submit(new_state, output_dir, cloud)
    elif new_state["phase"] != "ready":
        wait_ready(new_state, output_dir, cloud)

    write_deployment_summary(new_state, output_dir)
    save_history_record(
        new_state,
        output_dir,
        history_dir=(
            local_history_dir
            if local_history_dir is not None
            else (history_dir if persist_local_history else None)
        ),
    )
    return new_state


def _add_oss_sync_flags(p):
    p.add_argument(
        "--sync-oss",
        dest="sync_oss",
        action="store_true",
        default=True,
        help="Archive deployment credentials and history to fixed environment OSS prefix (enabled by default)",
    )
    p.add_argument(
        "--no-sync-oss",
        dest="sync_oss",
        action="store_false",
        help="Skip archiving deployment history to OSS",
    )


def _add_optional_local_history_flag(p):
    p.add_argument(
        "--local-history-dir",
        "--history-dir",
        dest="history_dir",
        type=Path,
        default=None,
        help="Optional extra local directory to mirror deployment history",
    )


def _add_oss_lookup_flags(p):
    _add_optional_local_history_flag(p)
    p.add_argument(
        "--environment",
        default=None,
        help="Environment tier for fixed OSS history prefix (selftest, test-shanghai-copy, prod-shanghai, prod-thailand)",
    )
    p.add_argument(
        "--region",
        default=None,
        help="Optional region override for selftest OSS bucket selection (cn-shanghai or ap-southeast-7)",
    )
    p.add_argument(
        "--aliyun-profile",
        dest="aliyun_profile",
        default=argparse.SUPPRESS,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--aliyun-profile",
        dest="aliyun_profile",
        default=None,
        help="Aliyun CLI profile name for Eflo, PAI-DLC, and OSS operations",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("probe", "up"):
        p = sub.add_parser(action, allow_abbrev=False)
        p.add_argument("--spec", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument(
            "--aliyun-profile",
            dest="aliyun_profile",
            default=argparse.SUPPRESS,
        )
        if action == "up":
            p.add_argument("--commit", required=True)
            p.add_argument("--execute", action="store_true")
            p.add_argument("--yes", action="store_true")
            _add_optional_local_history_flag(p)
            _add_oss_sync_flags(p)
    for action in ("selftest", "attach", "down", "inspect"):
        p = sub.add_parser(action, allow_abbrev=False)
        p.add_argument("--state", type=Path, required=True)
        p.add_argument(
            "--aliyun-profile",
            dest="aliyun_profile",
            default=argparse.SUPPRESS,
        )
        if action != "inspect":
            p.add_argument("--execute", action="store_true")
            p.add_argument("--yes", action="store_true")
            _add_optional_local_history_flag(p)
            _add_oss_sync_flags(p)
        if action == "attach":
            p.add_argument(
                "--in-place", dest="in_place", action="store_true", default=True
            )
            p.add_argument("--recreate", dest="in_place", action="store_false")
    p_hist = sub.add_parser("history", allow_abbrev=False)
    _add_oss_lookup_flags(p_hist)

    p_show = sub.add_parser("show", allow_abbrev=False)
    _add_oss_lookup_flags(p_show)
    p_show.add_argument("--deployment-id", required=True)

    p_diff = sub.add_parser("diff", allow_abbrev=False)
    _add_oss_lookup_flags(p_diff)
    p_diff.add_argument("--from-id", required=True)
    p_diff.add_argument("--to-id", required=True)

    p_rb = sub.add_parser("rollback", allow_abbrev=False)
    _add_oss_lookup_flags(p_rb)
    p_rb.add_argument("--deployment-id", default=None)
    p_rb.add_argument(
        "--previous",
        action="store_true",
        help="Automatically select the previous deployment before current-state",
    )
    p_rb.add_argument("--output", type=Path, required=True)
    p_rb.add_argument("--current-state", type=Path, default=None)
    p_rb.add_argument("--execute", action="store_true")
    p_rb.add_argument("--yes", action="store_true")
    _add_oss_sync_flags(p_rb)
    return parser


def _maybe_sync_oss_history(
    state, directory, cloud, *, sync_oss=True, history_dir=None
):
    target = state.get("target", {})
    if not sync_oss or target.get("sync_history_oss") is False:
        return
    if history_dir is not None:
        save_history_record(state, directory, history_dir=history_dir)
    store = history_store_dir(state, directory, history_dir)
    if store is None:
        return
    cloud.sync_history_artifacts(store, state["deployment_id"], summary_dir=directory)


@contextlib.contextmanager
def _resolve_cli_history_store(
    args, aliyun_profile, *, current_state=None, deployment_ids=None
):
    if getattr(args, "history_dir", None) is not None:
        yield Path(args.history_dir), True
        return
    base_target = (current_state or {}).get("target") or {}
    env_name = getattr(args, "environment", None) or base_target.get(
        "environment", "test-shanghai-copy"
    )
    eff_region = resolve_environment_region(
        base_target, environment=env_name, region=getattr(args, "region", None)
    )
    lookup_target = {
        **copy.deepcopy(TARGET if env_name == "test-shanghai-copy" else base_target),
        "environment": env_name,
        "region": eff_region,
    }
    cloud = Cloud(target=lookup_target, aliyun_profile=aliyun_profile)
    with tempfile.TemporaryDirectory() as td:
        yield cloud.fetch_oss_history_store(td, deployment_ids=deployment_ids), False


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    aliyun_profile = getattr(args, "aliyun_profile", None)
    if args.action == "history":
        with _resolve_cli_history_store(args, aliyun_profile, deployment_ids=[]) as (
            store_dir,
            _,
        ):
            print(json.dumps(list_history(store_dir), indent=2, sort_keys=True))
        return
    if args.action == "show":
        with _resolve_cli_history_store(
            args, aliyun_profile, deployment_ids=[args.deployment_id]
        ) as (store_dir, _):
            print(
                json.dumps(
                    show_deployment(store_dir, args.deployment_id),
                    indent=2,
                    sort_keys=True,
                )
            )
        return
    if args.action == "diff":
        with _resolve_cli_history_store(
            args, aliyun_profile, deployment_ids=[args.from_id, args.to_id]
        ) as (store_dir, _):
            print(
                json.dumps(
                    diff_deployments(store_dir, args.from_id, args.to_id),
                    indent=2,
                    sort_keys=True,
                )
            )
        return
    if args.action == "rollback":
        if bool(args.deployment_id) == bool(args.previous):
            raise ValueError(
                "rollback requires exactly one of --deployment-id or --previous"
            )
        current_state = (
            load_state(args.current_state) if args.current_state is not None else None
        )
        needed = [args.deployment_id] if args.deployment_id else None
        with _resolve_cli_history_store(
            args,
            aliyun_profile,
            current_state=current_state,
            deployment_ids=needed,
        ) as (store_dir, is_local_store):
            target_dep_id = args.deployment_id
            if args.previous:
                curr_id = (
                    current_state.get("deployment_id")
                    if current_state is not None
                    else None
                )
                target_dep_id = resolve_previous_deployment_id(store_dir, curr_id)
            rec = show_deployment(store_dir, target_dep_id)
            cloud = Cloud(target=rec.get("target"), aliyun_profile=aliyun_profile)
            result = rollback_deployment(
                store_dir,
                target_dep_id,
                args.output,
                cloud,
                execute=args.execute,
                yes=args.yes,
                current_state_path=args.current_state,
                local_history_dir=args.history_dir,
                persist_local_history=is_local_store,
            )
            if args.execute:
                _maybe_sync_oss_history(
                    result,
                    args.output,
                    cloud,
                    sync_oss=args.sync_oss,
                    history_dir=args.history_dir,
                )
            if not args.execute:
                print(
                    json.dumps(
                        {
                            "action": "rollback",
                            "status": "not-executed",
                            "deployment_id": result["deployment_id"],
                            "rollback_of": result["rollback_of"],
                            "plan": result.get("rollback_plan"),
                        },
                        sort_keys=True,
                    )
                )
        return
    if args.action in ("probe", "up"):
        raw = read(args.spec)
        spec = expand_minimal_spec(raw)
        if getattr(args, "history_dir", None) is not None:
            spec["target"]["history_dir"] = str(args.history_dir)
        if getattr(args, "sync_oss", None) is not None:
            spec["target"]["sync_history_oss"] = bool(args.sync_oss)
        cloud = Cloud(target=spec["target"], aliyun_profile=aliyun_profile)
        if args.action == "probe":
            probe(spec, args.output, cloud)
        else:
            state = create_plan(spec, args.output, args.commit)
            if args.execute:
                confirm_execution(
                    "up",
                    {
                        "deployment_id": state["deployment_id"],
                        "environment": spec["target"]["environment"],
                        "owned_roles": list(owned_roles(spec)),
                        "reused_roles": sorted(
                            spec.get("existing_services", {}).keys()
                        ),
                    },
                    yes=args.yes,
                )
                submit(state, args.output, cloud)
                _maybe_sync_oss_history(
                    state,
                    args.output,
                    cloud,
                    sync_oss=args.sync_oss,
                    history_dir=args.history_dir,
                )
    else:
        state = load_state(args.state)
        directory = args.state.parent
        cloud = Cloud(target=state["target"], aliyun_profile=aliyun_profile)
        if args.action == "inspect":
            wait_ready(state, directory, cloud)
        elif args.action == "attach":
            if args.execute:
                confirm_execution(
                    "attach",
                    {
                        "deployment_id": state.get("deployment_id"),
                        "run_id": state["run_id"],
                        "in_place": args.in_place,
                        "owned_roles": list(owned_roles(state["spec"])),
                    },
                    yes=args.yes,
                )
            attach(
                state,
                directory,
                cloud,
                execute=args.execute,
                in_place=args.in_place,
            )
            if args.execute:
                _maybe_sync_oss_history(
                    state,
                    directory,
                    cloud,
                    sync_oss=args.sync_oss,
                    history_dir=args.history_dir,
                )
        elif not args.execute:
            print(
                json.dumps(
                    {
                        "action": args.action,
                        "status": "not-executed",
                        "run_id": state["run_id"],
                    }
                )
            )
        elif args.action == "selftest":
            confirm_execution(
                "selftest",
                {
                    "deployment_id": state.get("deployment_id"),
                    "run_id": state["run_id"],
                },
                yes=args.yes,
            )
            asyncio.run(selftest(state, directory, cloud=cloud))
            _maybe_sync_oss_history(
                state,
                directory,
                cloud,
                sync_oss=args.sync_oss,
                history_dir=args.history_dir,
            )
        else:
            confirm_execution(
                "down",
                {
                    "deployment_id": state.get("deployment_id"),
                    "run_id": state["run_id"],
                    "owned_roles": list(owned_roles(state["spec"])),
                    "reused_roles": sorted(
                        state["spec"].get("existing_services", {}).keys()
                    ),
                },
                yes=args.yes,
            )
            down(state, directory, cloud)
            _maybe_sync_oss_history(
                state,
                directory,
                cloud,
                sync_oss=args.sync_oss,
                history_dir=args.history_dir,
            )


if __name__ == "__main__":
    main()
