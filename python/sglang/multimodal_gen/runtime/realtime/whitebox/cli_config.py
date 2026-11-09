# SPDX-License-Identifier: Apache-2.0
"""Room deployment policy around the official CLI, not a second SDK owner."""

import re
from pathlib import Path


def seconds(env, key, default, *, maximum=2147483, minimum=0):
    value = str(env.get(key, default))
    if not re.fullmatch(r"[0-9]+", value) or not minimum <= int(value) <= maximum:
        raise ValueError("Invalid CLI timeout setting: " + key)
    return int(value)


def startup_seconds(env):
    return seconds(
        env, "WHITEBOX_CLI_STARTUP_TIMEOUT_SECONDS", 420, minimum=1, maximum=3600
    )


def resource_transport(env):
    value = env.get("WHITEBOX_CLI_RESOURCE_TRANSPORT", "auto")
    if value not in {"auto", "cdn", "oss-public", "oss-internal", "archive"}:
        raise ValueError("Invalid CLI resource transport")
    return value


def game_cache_directory(env):
    value = env.get("WHITEBOX_CLI_GAME_CACHE_DIRECTORY") or env.get(
        "WORLDKIT_GAME_CACHE_DIR"
    )
    if not value and env.get("WHITEBOX_CLI_CACHE_DIRECTORY"):
        # Keep the old playable-object cache intact; the SDK game cache owns its
        # own index, leases, verified runtime/assets and retention policy.
        value = str(Path(env["WHITEBOX_CLI_CACHE_DIRECTORY"]) / "games")
    if value and not Path(value).is_absolute():
        raise ValueError("The CLI game cache directory must be absolute")
    return value


def stream_policy_arguments(env):
    # Existing Room deployments use SERVICE_MODE; the managed CLI adapter also
    # accepts SERVICE. Resolve once so one invocation cannot select two owners.
    deployed_mode = env.get("WHITEBOX_CLI_SERVICE_MODE")
    if deployed_mode is not None and deployed_mode not in {"auto", "required", "off"}:
        raise ValueError("Invalid CLI service mode")
    configured_mode = env.get("WHITEBOX_CLI_SERVICE")
    if (
        deployed_mode is not None
        and configured_mode is not None
        and deployed_mode != configured_mode
    ):
        raise ValueError("Conflicting CLI resident policy")
    mode = (
        deployed_mode
        if deployed_mode is not None
        else env.get("WHITEBOX_CLI_SERVICE", "auto")
    )
    if mode not in {"auto", "required", "off"}:
        raise ValueError("Invalid CLI resident policy")
    args = ["--service", mode]
    if mode != "off":
        args += ["--service-name", env.get("WHITEBOX_CLI_SERVICE_NAME", "whitebox")]
        if "WHITEBOX_ROOM_PIPELINE_CAPACITY" in env:
            # Each admitted source needs its own resident stream worker; otherwise
            # the second source waits behind the first game's whole lifetime.
            capacity = seconds(
                env, "WHITEBOX_ROOM_PIPELINE_CAPACITY", 1, minimum=1, maximum=16
            )
            args += ["--workers", str(capacity)]
    for flag, key, default in (
        ("--idle-timeout", "WHITEBOX_CLI_IDLE_TIMEOUT_SECONDS", 60),
        ("--connection-timeout", "WHITEBOX_CLI_CONNECTION_TIMEOUT_SECONDS", 30),
        ("--max-duration", "WHITEBOX_CLI_MAX_DURATION_SECONDS", 0),
    ):
        args += [flag, str(seconds(env, key, default))]
    return args
