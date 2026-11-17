"""Compile versioned regional policy; never read cloud credentials or deploy."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent


def _load_profile_launcher():
    module_name = "_whitebox_zing_h3_profile_launcher"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = (
        ROOT.parents[1]
        / "python/sglang/multimodal_gen/tools/zing_h3_profile_launcher.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


REGIONS = {"shanghai": "cn-shanghai", "thailand": "ap-southeast-7"}
ROLES = ("coordinator", "gateway", "room", "model")
REGIONAL_FIELDS = {
    "region",
    "deployment",
    "gpu.capability",
    "gpu.minimum_memory_gib",
    "model.cache_dir",
    "model.cache_capacity_dir",
    "model.cache_host_gb",
    "model.cache_host_total_gb",
    "model.cache_disk_gb",
    "model.cache_disk_reserve_gb",
    "model.cache_disk_inflight_gb",
    "model.cache_disk_max_writers",
}
# Only site-specific wiring belongs here. Policy cannot be overridden by a secret
# file or inherited environment without changing the versioned regional config.
BINDINGS = {
    "coordinator_url",
    "room_url",
    "owner_id",
    "redis_host",
    "redis_port",
    "redis_prefix",
    "platform_url",
    "gateway_url",
    "oss_bucket",
    "oss_prefix",
    "rtc_public_ip",
    "rtc_public_port",
    "routing_revision",
    "worker_id",
    "availability_zone",
    "advertise_ip",
    "asset_root",
    "state_root",
}
OPTIONAL_BINDINGS = {
    "room_pool_compatibility",
    "cache_config",
    "control_credential_refs",
}

SECRET_KEYS = {
    "coordinator": ["WHITEBOX_INTERNAL_TOKEN", "REDIS_PASSWORD"],
    "gateway": [
        "WHITEBOX_INTERNAL_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ],
    "room": [
        "WHITEBOX_INTERNAL_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "WORLD_TEST_API_TOKEN",
    ],
    "model": ["WHITEBOX_INTERNAL_TOKEN"],
}


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def overlay(base, patch, path="config"):
    if not isinstance(patch, dict):
        raise TypeError(f"{path} must be an object")
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if key not in result:
            raise ValueError(f"Unknown field: {path}.{key}")
        if isinstance(result[key], dict):
            result[key] = overlay(result[key], value, f"{path}.{key}")
        elif type(value) is not type(result[key]):
            raise ValueError(f"Invalid type: {path}.{key}")
        else:
            result[key] = value
    return result


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"Invalid {name}")


def load(region, *, root=ROOT):
    if region not in REGIONS:
        raise ValueError("Select shanghai or thailand")
    patch = json.loads((root / "regions" / f"{region}.json").read_text())

    def check_patch(value, prefix=""):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                check_patch(item, path)
            elif path not in REGIONAL_FIELDS:
                raise ValueError(
                    f"Model and player behavior belongs in common.json: {path}"
                )

    check_patch(patch)
    config = overlay(
        json.loads((root / "common.json").read_text()),
        patch,
    )
    return validate(config)


def validate(config):
    if (
        config["schema_version"] != 1
        or config["release_branch"] != "main"
        or config["region"] not in REGIONS.values()
    ):
        raise ValueError("Unexpected release policy or region")
    model, media = config["model"], config["media"]
    if not re.fullmatch(r"[0-9a-f]{64}", model["checkpoint_sha256"]):
        raise ValueError("Checkpoint identity must be explicit")
    if (
        config["environment"]["coordinator"]["WHITEBOX_SLOTS"] != "1"
        or config["environment"]["room"]["WHITEBOX_MAX_ROOMS"] != "1"
    ):
        raise ValueError("Single-session deployment capacity must remain one")
    integer(config["gpu"]["minimum_memory_gib"], 1, 1024, "GPU memory bound")
    if config["gpu"]["capability"] not in ([9, 0], [10, 3]):
        raise ValueError("Unknown GPU profile")
    if config["gpu"]["count"] != 4 or [
        model[k] for k in ("dit_gpus", "encoder_gpus", "decoder_gpus")
    ] != [[0, 1], [2], [3]]:
        raise ValueError(
            "Only the four-GPU E1+SP2+D1 single-session layout is supported"
        )
    integer(config["play_seconds"], 1, 86400, "play_seconds")
    if (model["width"], model["height"]) not in ((832, 480), (1280, 736)):
        raise ValueError(
            "Model output must use the reviewed 832x480 or 1280x736 profile"
        )
    source = config["environment"]["shared"]
    if (source.get("WHITEBOX_WIDTH"), source.get("WHITEBOX_HEIGHT")) != ("832", "480"):
        raise ValueError("This release profile preserves native 832x480 source capture")
    for steps in (model["denoising_steps"], model["initial_steps"]):
        if (
            not steps
            or any(type(v) is not int or not 0 < v <= 1000 for v in steps)
            or sorted(set(steps), reverse=True) != steps
        ):
            raise ValueError(
                "Denoising steps must be strictly descending positive integers"
            )
    if "sink_chunks" in model and (
        type(model["sink_chunks"]) is not int or model["sink_chunks"] != 0
    ):
        raise ValueError("Whitebox model sessions must start with sink_chunks=0")
    integer(model["initial_blocks"], 0, 64, "initial_blocks")
    integer(model["window_chunks"], 4, 21, "window_chunks")
    integer(model["source_work_quantum"], 0, 1, "source_work_quantum")
    if config["environment"]["model"]["SGLANG_ZING_SOURCE_PROMPT_JOINT"] not in (
        "0",
        "1",
    ):
        raise ValueError("Source prompt joint policy must be 0 or 1")
    fraction = model.get("mem_fraction_static")
    if type(fraction) not in (int, float) or not 0 < fraction < 1:
        raise ValueError(
            "Model static memory fraction must be explicit and between zero and one"
        )
    integer(model["cache_host_gb"], 1, 4096, "cache_host_gb")
    if model.get("cache_backend", "local") not in {"local", "mooncake"}:
        raise ValueError("Unknown Block0 cache backend")
    cache_recording = config["environment"]["room"].get(
        "WHITEBOX_CACHE_RECORDING_ENABLED", "0"
    )
    if cache_recording not in {"0", "1"} or (
        cache_recording == "1" and model.get("cache_backend", "local") != "mooncake"
    ):
        raise ValueError("Cache operation recording requires the Mooncake policy")
    integer(model["cache_disk_gb"], 1, 16384, "cache_disk_gb")
    if "cache_host_total_gb" in model:
        integer(
            model["cache_host_total_gb"],
            model["cache_host_gb"],
            16384,
            "cache_host_total_gb",
        )
        integer(model["cache_disk_reserve_gb"], 1, 16384, "cache_disk_reserve_gb")
        integer(
            model["cache_disk_inflight_gb"],
            1,
            model["cache_disk_gb"],
            "cache_disk_inflight_gb",
        )
        integer(model["cache_disk_max_writers"], 1, 64, "cache_disk_max_writers")
        if (
            not isinstance(model["cache_capacity_dir"], str)
            or not Path(model["cache_capacity_dir"]).is_absolute()
        ):
            raise ValueError("Machine capacity directory must be absolute")
    integer(model["reference_max_images"], 1, 21, "reference_max_images")
    integer(media["cli_fps"], 1, 60, "cli_fps")
    room = config["environment"]["room"]
    sampling = room.get("WHITEBOX_MODEL_SOURCE_SAMPLING", "latest")
    if sampling not in ("latest", "adaptive") or (
        sampling == "adaptive" and room.get("WHITEBOX_RENDER_SEGMENTS") != "1"
    ):
        raise ValueError("Invalid Room model source sampling policy")
    for key, value in media.items():
        if key.endswith("_bps"):
            integer(value, 100000, 100000000, key)
    if (
        media["model_min_bps"] > media["model_max_bps"]
        or media["transport_min_bps"] > media["transport_max_bps"]
    ):
        raise ValueError("Minimum bitrate exceeds maximum")
    if media["transport_min_bps"] < media["model_min_bps"] + 300000:
        raise ValueError("Transport floor cannot cover both encoder floors")
    if media["transport_max_bps"] < media["model_max_bps"] + media["source_max_bps"]:
        raise ValueError("Transport ceiling cannot cover both encoder ceilings")
    if (
        not model["cache_dir"].startswith("/var/cache/sglang/block0/")
        or ".." in Path(model["cache_dir"]).parts
    ):
        raise ValueError("Use a dedicated local Block0 directory")
    if model["adaln_mode"] not in ("resident", "table"):
        raise ValueError("Unknown AdaLN policy")
    if model.get("quantization", "none") not in ("none", "fp8"):
        raise ValueError("Only BF16 or native online FP8 is supported")
    if model.get("quantization_scope", "ffn-attention") not in (
        "ffn",
        "fc1",
        "interior-qkv-fc1",
        "ffn-attention",
    ):
        raise ValueError("Unknown Zing FP8 linear scope")
    for role, values in config["environment"].items():
        for key, value in values.items():
            if (
                not re.fullmatch(r"[A-Z][A-Z0-9_]*", key)
                or not isinstance(value, str)
                or "\n" in value
            ):
                raise ValueError(f"Invalid environment entry for {role}")
            if key in {k for keys in SECRET_KEYS.values() for k in keys} | {
                "WHITEBOX_ROOM_ADMIN_TOKEN",
                "SGLANG_CACHE_ADMIN_TOKEN",
                "WHITEBOX_CACHE_RECORDING_SECRET",
            }:
                raise ValueError(
                    "Credentials must be supplied by the platform secret store"
                )
    return config


def validate_bindings(values):
    if not BINDINGS <= set(values) or set(values) - BINDINGS - OPTIONAL_BINDINGS:
        raise ValueError(
            f"Bindings differ: missing={sorted(BINDINGS - set(values))}, unknown={sorted(set(values) - BINDINGS - OPTIONAL_BINDINGS)}"
        )
    if any(
        not isinstance(v, str) or not v or any(c in v for c in "\n\r\0")
        for v in values.values()
    ):
        raise ValueError("Bindings must be nonempty single-line strings")
    for key in ("coordinator_url", "room_url", "platform_url", "gateway_url"):
        url = urlsplit(values[key])
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(f"Invalid URL binding: {key}")
    for key in ("asset_root", "state_root"):
        if not Path(values[key]).is_absolute() or ".." in Path(values[key]).parts:
            raise ValueError(f"Expected absolute path: {key}")
    for key in ("redis_port", "rtc_public_port"):
        integer(int(values[key]), 1, 65535, key)
    for key in ("owner_id", "redis_prefix", "routing_revision", "worker_id"):
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", values[key]):
            raise ValueError(f"Invalid identity: {key}")


def render(config, bindings):
    validate(config)
    validate_bindings(bindings)
    b, c, m = bindings, config, config["media"]
    digest = fingerprint({"policy": c, "bindings": b})
    envs = {
        role: {
            **c["environment"]["shared"],
            **c["environment"][role],
            "WHITEBOX_RELEASE_CONFIG_SHA256": digest,
            "WHITEBOX_RELEASE_REGION": c["region"],
            "WHITEBOX_ROLE": role,
        }
        for role in ROLES
    }
    if "control_credential_refs" in b:
        from control_bindings import REFS_ENV, parse_manifest

        refs = parse_manifest(b["control_credential_refs"])
        if c["region"] != refs["region"]:
            raise ValueError("Control bindings do not match the release region")
        if (
            urlsplit(b["gateway_url"]).hostname
            != "shcopy.world-test-platform.loopit.com.cn"
        ):
            raise ValueError("Control bindings require the Shanghai test Gateway")
        for role in ("room", "gateway", "coordinator"):
            envs[role][REFS_ENV] = b["control_credential_refs"]
    shared = {
        "WHITEBOX_ROOM_COORDINATOR_URL": b["coordinator_url"],
        "WHITEBOX_ROOM_OWNER_ID": b["owner_id"],
        "WHITEBOX_ROOM_KEY_PREFIX": b["redis_prefix"] + ":rooms:",
        "WHITEBOX_ROUTING_REVISION": b["routing_revision"],
        "WHITEBOX_PLAY_SECONDS": str(c["play_seconds"]),
        "WHITEBOX_RENDER_PLAY_SECONDS": str(c["play_seconds"]),
        "WHITEBOX_PLAYER_SESSION_TTL_SECONDS": str(c["play_seconds"]),
        "WHITEBOX_OSS_BUCKET": b["oss_bucket"],
        "WHITEBOX_OSS_REGION": c["region"],
        "WHITEBOX_OSS_ENDPOINT": f"https://oss-{c['region']}-internal.aliyuncs.com",
        "WHITEBOX_OSS_PREFIX": b["oss_prefix"],
        "WHITEBOX_BROWSER_ORIGIN": b["platform_url"],
    }
    for role in ("room", "gateway"):
        envs[role].update(shared)
        # The native CLI canvas and model output are separate protocol fields.
        # Never upscale raw source RGB to pretend the CLI rendered at model size.
        envs[role]["WHITEBOX_MODEL_WIDTH"] = str(c["model"]["width"])
        envs[role]["WHITEBOX_MODEL_HEIGHT"] = str(c["model"]["height"])
    envs["coordinator"].update(
        {
            "WHITEBOX_PLAYER_SESSION_TTL_SECONDS": str(c["play_seconds"]),
            "REDIS_HOST": b["redis_host"],
            "REDIS_PORT": b["redis_port"],
            "REDIS_USERNAME": "default",
            "WHITEBOX_REDIS_PREFIX": b["redis_prefix"],
            "WHITEBOX_ROOM_OWNERS": json.dumps(
                {b["owner_id"]: b["room_url"]}, sort_keys=True
            ),
        }
    )
    envs["gateway"]["COORDINATOR_URL"] = b["coordinator_url"]
    envs["room"].update(
        {
            "WHITEBOX_CLI_PLATFORM_URL": b["platform_url"],
            "WHITEBOX_CLI_FPS": str(m["cli_fps"]),
            "WHITEBOX_MODEL_MIN_BITRATE": str(m["model_min_bps"]),
            "WHITEBOX_PLAYBACK_MAX_BITRATE": str(m["model_max_bps"]),
            "WHITEBOX_SOURCE_MAX_BITRATE": str(m["source_max_bps"]),
            "WHITEBOX_RTC_MIN_SEND_BITRATE": str(m["transport_min_bps"]),
            "WHITEBOX_RTC_MAX_SEND_BITRATE": str(m["transport_max_bps"]),
            "WHITEBOX_ARCHIVE_SOURCE_BITRATE": str(m["archive_source_bps"]),
            "WHITEBOX_ARCHIVE_PAIRED_SOURCE_BITRATE": str(m["archive_paired_bps"]),
            "WHITEBOX_ARCHIVE_MODEL_BITRATE": str(m["archive_model_bps"]),
            "WHITEBOX_NATIVE_RTC_PUBLIC_IP": b["rtc_public_ip"],
            "WHITEBOX_NATIVE_RTC_PUBLIC_PORT": b["rtc_public_port"],
        }
    )
    envs["model"]["SGLANG_ZING_EXPERIMENTAL_SOURCE_WORK_QUANTUM"] = str(
        c["model"]["source_work_quantum"]
    )
    if c["model"].get("quantization", "none") == "fp8":
        envs["model"]["SGLANG_ZING_QUANT_SCOPE"] = c["model"].get(
            "quantization_scope", "ffn-attention"
        )
    required_secrets = copy.deepcopy(SECRET_KEYS)
    if envs["room"].get("WHITEBOX_CACHE_RECORDING_ENABLED") == "1":
        for role in ("room", "model"):
            required_secrets[role].append("WHITEBOX_CACHE_RECORDING_SECRET")
    if b.get("room_pool_compatibility"):
        for role in ("gateway", "coordinator", "room"):
            envs[role]["WHITEBOX_DYNAMIC_ROOMS"] = "1"
            envs[role]["WHITEBOX_ROOM_POOL_COMPATIBILITY"] = b[
                "room_pool_compatibility"
            ]
        envs["room"]["WHITEBOX_ROOM_ADVERTISE_URL"] = b["room_url"]
        required_secrets["coordinator"].append("WHITEBOX_ROOM_ADMIN_TOKEN")
    profile_meta = _load_profile_launcher().policy_source_metadata()
    hardware_profile = "b300" if c["gpu"]["capability"] == [10, 3] else "h200"
    return {
        "schema_version": 1,
        "region": c["region"],
        "config_sha256": digest,
        "launch_profile": {
            **profile_meta,
            "mode": "integrated",
            "profile": hardware_profile,
            "resolution": str(c["model"]["height"]),
        },
        "environments": envs,
        "required_secret_keys": required_secrets,
        "policy": c,
        "bindings": b,
        "activation": "manual_only",
        "active_session_capacity": 1,
        "restart_scope": {
            "media": "room",
            "play_seconds": "gateway_and_room",
            "model": "all_model_roles",
            "bindings": "affected_services",
        },
    }
