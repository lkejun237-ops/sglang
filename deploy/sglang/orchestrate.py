"""SGLang multi-role environment orchestration. Planning is offline; mutations require --execute.

No local kube context, SSH aliases, production targets, or automatic create retries.
State files are private operator evidence, not a source of arbitrary shell commands.
"""

import argparse
import asyncio
import base64
import copy
import datetime
import hashlib
import importlib.util
import ipaddress
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import launch
from source import digest, prepare, validate_manifest

ROLES = ("encoder", "dit", "decoder")
FORBIDDEN_ENVIRONMENTS = {"prod", "production", "prod-shanghai", "prod-thailand"}
FORBIDDEN_IPS = {"10.0.0.145", "10.0.1.31", "10.0.1.33"}
FORBIDDEN_PORTS = {28110, 28210}
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
    temp = path.with_name(path.name + ".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    temp.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check_target(value):
    if not isinstance(value, dict):
        raise ValueError("Target must be a JSON object")
    allowed_keys = set(TARGET) | {"topology_profile", "history_dir"}
    if set(value) - allowed_keys:
        raise ValueError("Unsupported target configuration keys")
    env_name = str(value.get("environment", "")).strip()
    if not env_name:
        raise ValueError("Target environment is required")
    if (
        env_name.lower() in FORBIDDEN_ENVIRONMENTS
        or env_name.lower().startswith("prod")
        or str(value.get("pool_id", "")).lower().startswith("prod")
    ):
        raise ValueError(
            "Production targets are forbidden; only non-production environments are permitted (pinned check)"
        )
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
        for url_key in ("coordinator_url", "room_url", "wtp_url"):
            uri = urlsplit(value[url_key])
            if uri.scheme not in ("http", "https") or not uri.hostname:
                raise ValueError(f"Invalid target {url_key}")
            if uri.hostname in FORBIDDEN_IPS or uri.port in FORBIDDEN_PORTS:
                raise ValueError("Production endpoints are forbidden (pinned check)")
        h100_ip = value.get("h100_ip")
        if h100_ip is not None:
            if not isinstance(h100_ip, str) or h100_ip in FORBIDDEN_IPS:
                raise ValueError("Forbidden or invalid h100_ip (pinned check)")
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
        topology = "b300" if "b300" in spec.get("sites", {}) else "h100_h200"
    if topology not in ("h100_h200", "b300"):
        raise ValueError(f"Unsupported topology '{topology}'")
    default_model_dir = spec.pop("model", f"/assets/{model_id}")
    default_receipt = spec.pop("asset_receipt", "/assets/assets.json")
    default_receipt_sha = spec.pop("asset_receipt_sha256", "a" * 64)
    default_image = spec.pop("image", "test@sha256:" + "c" * 64)
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
                    )
                    if k in r_in
                }
            continue
        prof = r_in.get(
            "profile",
            "b300" if topology == "b300" else ("h200" if role == "dit" else "h100"),
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
        if "service_count" in r_in:
            role_cfg["service_count"] = r_in["service_count"]
        if "ports" in r_in:
            role_cfg["ports"] = r_in["ports"]
        if "secret_files" in r_in:
            role_cfg["secret_files"] = r_in["secret_files"]
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
    if "b300" in spec["sites"] and len(spec["sites"]) == 1:
        return spec["sites"]["b300"]
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
    return dict(
        run_id=run_id,
        generation=0,
        name="reused-" + role + "-" + entry["worker_id"],
        reused=True,
        mode="reuse",
        status="reused",
        service_index=entry.get("service_index", 0),
        service_count=entry.get("service_count", 1),
        pai=bool(entry.get("pai")),
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
            if role == "dit":
                raise ValueError(
                    "existing_services may only reuse encoder and/or decoder"
                )
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
                    )
                    if k in cfg
                }
            roles_dict.pop(role, None)
    if set(existing) - {"encoder", "decoder"}:
        raise ValueError("existing_services may only reuse encoder and/or decoder")
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
            }
            or not isinstance(entry.get("http"), str)
            or not isinstance(entry.get("worker_id"), str)
            or not entry["worker_id"]
        ):
            raise ValueError("Invalid existing service specification")
        count = entry.get("service_count", 1)
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
        if set(allocated_ports_fn(role_for_validation)) & FORBIDDEN_PORTS:
            raise ValueError("Production ports are forbidden")
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
                            raise RuntimeError(
                                "Eflo node command failed or output was truncated"
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
                        raise RuntimeError("Eflo invocation failed: " + status)
            time.sleep(2)
        raise TimeoutError("Eflo invocation is UNKNOWN; reconcile InvokeId " + invoke)

    def upload_source(self, local_path, uri):
        profile = self._require_aliyun_profile()
        region = self.target.get("region", "cn-shanghai")
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
        is_pai = "job_id" in resource or role == "dit"
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
                raise RuntimeError("Expected exactly one DiT Worker Pod")
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
    # The Eflo host runs Python 3.6. Keep syntax and urllib behavior compatible.
    parameters = json.dumps(dict(url=url, payload=payload, headers=headers))
    code = """import json,urllib.request
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,request,fp,code,message,headers,new_url):
  raise ValueError('Cross-environment HTTP redirects are forbidden')
p=json.loads(PARAMETERS)
data=None if p['payload'] is None else json.dumps(p['payload']).encode()
r=urllib.request.Request(p['url'],data=data,headers=p['headers'])
with urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect()).open(r,timeout=60) as response:
 print(json.dumps(json.load(response)))
""".replace("PARAMETERS", repr(parameters))
    return json.loads(cloud.eflo("python3 -c " + shlex.quote(code)))


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
            for j in jobs
            if j.get("ResourceId") == target["quota_id"]
        ],
    )
    for role in owned_roles(spec):
        save(output / (role + ".json"), spec["roles"][role])
    for site in sorted(spec["sites"]):
        save(output / (site + "-site.json"), spec["sites"][site])
    evidence["blockers"] = []
    if len(free) < len(h100_roles):
        evidence["blockers"].append(
            "Fewer than two unclaimed idle H100 GPUs"
            if len(h100_roles) == 2
            else f"Fewer than {len(h100_roles)} unclaimed idle H100 GPUs"
        )
    if inventory["busy_ports"]:
        evidence["blockers"].append("Configured H100 ports are occupied")
    if not all(inventory["models"].values()):
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
        else "h100-inventory-matched; h200-assets-unverified"
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
    )
    if rollback_of:
        state["rollback_of"] = rollback_of
    render_roles(state, output)
    save(output / "state.json", state)
    write_deployment_summary(state, output)
    save_history_record(state, output)
    return state


def render_roles(state, output, launch_module=launch):
    for role in owned_roles(state["spec"]):
        directory = output / (role + "-" + str(state["generation"]))
        site = role_site(state["spec"], role)
        pai = is_pai_role(state["spec"], role)
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
    # Write-ahead marker covers an upload timeout too; no create is repeated on resume.
    state["phase"] = "uploading"
    save(state_path, state)
    if digest(directory / "source/source.tar") != state["source"]["sha256"]:
        raise ValueError("Local source archive changed after planning")
    cloud.upload_source(directory / "source/source.tar", state["source"]["uri"])
    if any(not is_pai_role(state["spec"], role) for role in owned):
        cloud.stage_h100_source(state["source"])
    for role, entry in state["spec"].get("existing_services", {}).items():
        state["resources"].setdefault(
            role,
            reused_resource(state["run_id"], state["spec"]["model_id"], role, entry),
        )
    for role in owned:
        plan = directory / (role + "-" + str(state["generation"]))
        pai = is_pai_role(state["spec"], role)
        resource = dict(
            run_id=state["run_id"],
            generation=state["generation"],
            name=read(plan / "render.json")["name"],
            status="submitting",
        )
        if not pai:
            resource["namespace"] = role_site(state["spec"], role)["namespace"]
        state["resources"][role] = resource
        state["phase"] = "submitting"
        save(state_path, state)
        # A timeout leaves this exact name in state. No speculative rollback or duplicate creation.
        if pai:
            resource["expected_job"] = read(plan / "pai.json")
            save(state_path, state)
            resource["job_id"] = cloud.submit_job(resource["expected_job"])
        else:

            def submitted(invoke):
                resource["invoke_id"] = invoke
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
    state["phase"] = "waiting"
    save(state_path, state)
    wait_ready(state, directory, cloud)


def wait_ready(state, directory, cloud):
    target = state.get("target", TARGET)
    deadline = time.monotonic() + cloud.timeout
    while time.monotonic() < deadline:
        errors = {}

        def check_role(role):
            resource = state["resources"][role]
            if resource.get("reused") is True:
                receipt = copy.deepcopy(resource["receipt"])
                validate_endpoint(
                    receipt["http"],
                    role,
                    pai=bool(resource.get("pai")),
                    target=target,
                )
                health = http(receipt["http"] + "/health", cloud=cloud)
                if health.get("ready") is not True or health.get("role") != role:
                    raise RuntimeError("Reused worker is not ready")
                try:
                    conn = http(
                        receipt["http"] + "/get_connection_status?reserve=false",
                        cloud=cloud,
                    )
                    service = conn.get("service", {})
                    if (
                        service.get("worker_id") != receipt["worker_id"]
                        or service.get("role") != role
                    ):
                        raise ValueError("Reused service worker_id or role mismatch")
                    receipt.update(status="ready", health=health, managed=False)
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
                        if proc_status.get("process", {}).get("state") != "ready":
                            raise RuntimeError(
                                "Reused managed worker is not ready in Coordinator"
                            ) from None
                    receipt.update(
                        status="ready",
                        health=health,
                        managed=True,
                        boot_epoch=identity["boot_epoch"],
                        service=service,
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
        raise ValueError("H100 endpoint is outside the test node")
    if pai or not h100_ip:
        address = ipaddress.ip_address(uri.hostname or "")
        forbidden = set(FORBIDDEN_IPS) | {"10.0.1.34", "10.0.1.37"}
        if h100_ip:
            forbidden.add(h100_ip)
        if (
            address not in ipaddress.ip_network("10.0.0.0/8")
            or str(address) in forbidden
        ):
            raise ValueError("DiT endpoint is outside the test network")
    if uri.port is None or uri.port in FORBIDDEN_PORTS:
        raise ValueError("Production ports are forbidden")


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
    # Revalidate the owned Pod before exec; do not select an arbitrary existing service.
    if cloud.pod(encoder)["metadata"]["uid"] != encoder["uid"]:
        raise ValueError("Encoder Pod UID changed before selftest")
    package = Path(encoder["receipt"]["runtime"]["sglang"]["path"])
    if not package.is_absolute():
        raise ValueError("Encoder receipt has no absolute verified package path")
    root = str(package.parents[2])
    attestation = dict(
        commit=state["source"]["commit"], archive_sha256=state["source"]["sha256"]
    )
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
            execution = "owned-encoder-pod"
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
    state["selftest"] = dict(
        status="passed",
        scope=scope,
        observed_at=now(),
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
    configs = worker_configs(state)
    for role, config in configs.items():
        save(directory / (role + "-role_pool_worker.json"), config)
    if not execute:
        return configs
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
            "verification": {
                "health_observed_at": resource.get("health_observed_at"),
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
        "source": state.get("source"),
        "model_id": spec.get("model_id"),
        "route": spec.get("route"),
        "owned_roles": list(owned_roles(spec)),
        "reused_roles": sorted(spec.get("existing_services", {}).keys()),
        "roles": roles_summary,
        "selftest": state.get("selftest"),
        "attachment": state.get("attachment"),
    }
    save(directory / "deployment.json", summary)
    md_lines = [
        f"# SGLang Deployment Summary: `{deployment_id}`",
        "",
        f"- **Run ID**: `{state['run_id']}`",
        f"- **Phase**: `{state.get('phase')}` (generation `{state.get('generation', 0)}`)",
        f"- **Environment**: `{target.get('environment')}` (`{target.get('region')}`)",
        f"- **Source Commit**: `{state.get('source', {}).get('commit')}`",
        f"- **Source Archive SHA256**: `{state.get('source', {}).get('sha256')}`",
        f"- **Model ID**: `{spec.get('model_id')}`",
        f"- **Route**: `{spec.get('route')}`",
    ]
    if state.get("rollback_of"):
        md_lines.append(f"- **Rollback Of**: `{state['rollback_of']}`")
    md_lines.extend(
        [
            "",
            "## Role Identities & Launch Profiles",
            "",
            "| Role | Mode | Status | Resource | Worker / Service ID | Boot Epoch | GPU UUID | HTTP Endpoints | Services | Profile |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
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
        md_lines.append(
            f"| `{role}` | `{r['mode']}` | `{r['status']}` | `{res_id}` | `{svc['service_id'] or '-'}` | `{svc['boot_epoch'] or '-'}` | `{gpus}` | `{eps}` | `{svc['service_count']}` | `{prof}` |"
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


def history_store_dir(state_or_target=None, directory=None, explicit_dir=None):
    if explicit_dir is not None:
        return Path(explicit_dir)
    if isinstance(state_or_target, dict):
        target = state_or_target.get("target", state_or_target)
        if isinstance(target, dict) and target.get("history_dir"):
            return Path(target["history_dir"])
    if directory is not None:
        return Path(directory) / "history"
    return REPO / ".sglang-deployments/history"


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
                "attachment_mode": state.get("attachment_mode"),
                "identities": copy.deepcopy(
                    state.get("attachment", {}).get("identities", {})
                ),
            },
            "secret_references": secret_refs,
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


def save_history_record(state, directory=None, history_dir=None):
    record = build_history_record(state, directory)
    stores = [history_store_dir(state, directory, history_dir)]
    local_dir_store = Path(directory) / "history" if directory is not None else None
    if local_dir_store is not None and local_dir_store not in stores:
        stores.append(local_dir_store)
    for store in stores:
        store.mkdir(parents=True, exist_ok=True)
        save(store / f"{record['deployment_id']}.json", record)
        index_path = store / "index.json"
        index = (
            read(index_path)
            if index_path.exists()
            else {"schema": 1, "deployments": []}
        )
        entries = [
            e
            for e in index.get("deployments", [])
            if e.get("deployment_id") != record["deployment_id"]
        ]
        entries.append(
            {
                "deployment_id": record["deployment_id"],
                "run_id": record["run_id"],
                "rollback_of": record.get("rollback_of"),
                "created_at": record.get("created_at"),
                "updated_at": record["updated_at"],
                "phase": record["phase"],
                "environment": record["target"].get("environment"),
                "commit": (record["rollback_bundle"].get("source") or {}).get("commit"),
                "model_id": record["rollback_bundle"]["spec"].get("model_id"),
                "owned_roles": record["rollback_bundle"]["ownership"]["owned_roles"],
                "reused_roles": record["rollback_bundle"]["ownership"]["reused_roles"],
            }
        )
        entries.sort(key=lambda x: (x.get("created_at") or "", x["deployment_id"]))
        index["deployments"] = entries
        save(index_path, index)
    return record


def list_history(history_dir):
    store = Path(history_dir)
    index_path = store / "index.json"
    if index_path.exists():
        return read(index_path)["deployments"]
    records = []
    if store.exists():
        for path in sorted(store.glob("deploy-*.json")):
            rec = read(path)
            records.append(
                {
                    "deployment_id": rec["deployment_id"],
                    "run_id": rec["run_id"],
                    "rollback_of": rec.get("rollback_of"),
                    "created_at": rec.get("created_at"),
                    "updated_at": rec.get("updated_at"),
                    "phase": rec.get("phase"),
                    "environment": rec.get("target", {}).get("environment"),
                    "commit": (rec.get("rollback_bundle", {}).get("source") or {}).get(
                        "commit"
                    ),
                }
            )
    return records


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


def rollback_deployment(
    history_dir,
    target_deployment_id,
    output_dir,
    cloud,
    *,
    execute=False,
    yes=False,
    current_state_path=None,
    repo=REPO,
):
    output_dir = Path(output_dir)
    record = show_deployment(history_dir, target_deployment_id)
    if not isinstance(record, dict) or "rollback_bundle" not in record:
        validate_rollback_record(record)
    bundle = record["rollback_bundle"]
    raw_source = copy.deepcopy(bundle.get("source"))
    raw_spec = bundle.get("spec") if isinstance(bundle.get("spec"), dict) else {}
    oss_prefix = (raw_spec.get("target") or {}).get("oss_prefix", "")

    current_state = (
        load_state(current_state_path) if current_state_path is not None else None
    )
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
        target_spec["target"]["history_dir"] = str(Path(history_dir))
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
        save(state_path, new_state)
        write_deployment_summary(new_state, output_dir)
        save_history_record(new_state, output_dir, history_dir=history_dir)

    if not execute:
        return new_state

    confirm_execution("rollback", new_state.get("rollback_plan", {}), yes=yes)

    # Target source archive, SHA256, historical render tools, and rendered payloads
    # are already prepared and verified above before stopping current owned resources.
    if current_state is not None and not new_state.get("rollback_stopped_current"):
        down(current_state, Path(current_state_path).parent, cloud)
        new_state["rollback_stopped_current"] = True
        save(state_path, new_state)

    # Submit new owned resources if not yet ready.
    if new_state["phase"] in ("planned", "attach-planned"):
        submit(new_state, output_dir, cloud)
    elif new_state["phase"] != "ready":
        wait_ready(new_state, output_dir, cloud)

    write_deployment_summary(new_state, output_dir)
    save_history_record(new_state, output_dir, history_dir=history_dir)
    return new_state


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
            p.add_argument("--history-dir", type=Path, default=None)
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
        if action == "attach":
            p.add_argument(
                "--in-place", dest="in_place", action="store_true", default=True
            )
            p.add_argument("--recreate", dest="in_place", action="store_false")
    p_hist = sub.add_parser("history", allow_abbrev=False)
    p_hist.add_argument("--history-dir", type=Path, required=True)

    p_show = sub.add_parser("show", allow_abbrev=False)
    p_show.add_argument("--history-dir", type=Path, required=True)
    p_show.add_argument("--deployment-id", required=True)

    p_diff = sub.add_parser("diff", allow_abbrev=False)
    p_diff.add_argument("--history-dir", type=Path, required=True)
    p_diff.add_argument("--from-id", required=True)
    p_diff.add_argument("--to-id", required=True)

    p_rb = sub.add_parser("rollback", allow_abbrev=False)
    p_rb.add_argument("--history-dir", type=Path, required=True)
    p_rb.add_argument("--deployment-id", required=True)
    p_rb.add_argument("--output", type=Path, required=True)
    p_rb.add_argument("--current-state", type=Path, default=None)
    p_rb.add_argument("--execute", action="store_true")
    p_rb.add_argument("--yes", action="store_true")
    p_rb.add_argument(
        "--aliyun-profile",
        dest="aliyun_profile",
        default=argparse.SUPPRESS,
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    aliyun_profile = getattr(args, "aliyun_profile", None)
    if args.action == "history":
        print(json.dumps(list_history(args.history_dir), indent=2, sort_keys=True))
        return
    if args.action == "show":
        print(
            json.dumps(
                show_deployment(args.history_dir, args.deployment_id),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.action == "diff":
        print(
            json.dumps(
                diff_deployments(args.history_dir, args.from_id, args.to_id),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.action == "rollback":
        rec = show_deployment(args.history_dir, args.deployment_id)
        cloud = Cloud(target=rec.get("target"), aliyun_profile=aliyun_profile)
        result = rollback_deployment(
            args.history_dir,
            args.deployment_id,
            args.output,
            cloud,
            execute=args.execute,
            yes=args.yes,
            current_state_path=args.current_state,
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
        spec = (
            expand_minimal_spec(raw) if args.action == "probe" else validate_spec(raw)
        )
        if getattr(args, "history_dir", None) is not None:
            spec["target"]["history_dir"] = str(args.history_dir)
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


if __name__ == "__main__":
    main()
