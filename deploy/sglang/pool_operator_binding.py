"""Render Podman launch arguments from one platform-provisioned OOS secret version."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def check_reference(binding, profile, purpose=None):
    # Both consumers use this same deployment-owned file. Never accept a secret
    # reference in an attach request or import secret data from a running role.
    pool, name, parameter_id, version = (
        binding[k] for k in ("pool_id", "parameter_name", "parameter_id", "version")
    )
    if not (
        isinstance(pool, str)
        and pool
        and isinstance(name, str)
        and re.fullmatch(r"[A-Za-z0-9_-]{1,180}", name)
        and isinstance(parameter_id, str)
        and re.fullmatch(r"p-[A-Za-z0-9]+", parameter_id)
        and type(version) is int
        and 1 <= version <= 100
    ):
        raise ValueError(
            "Expected a pool and a pinned OOS parameter name, ID and version"
        )

    result = subprocess.check_output(
        [
            "aliyun",
            "oos",
            "GetSecretParameter",
            "--profile",
            profile,
            "--region",
            "cn-shanghai",
            "--RegionId",
            "cn-shanghai",
            "--Name",
            name,
            "--ParameterVersion",
            str(version),
            "--WithDecryption",
            "false",
        ],
        timeout=20,
    )
    parameter = json.loads(result)["Parameter"]
    tags = parameter.get("Tags") or {}
    pool_scoped = (
        purpose not in {"room_internal", "rtc", "room_admin"} or "role-pool" in tags
    )
    valid_purposes = (
        {"operator", "operator-rotation"}
        if purpose in (None, "operator")
        else {purpose, purpose.replace("_", "-")}
    )
    if purpose is None:
        valid_purposes.add(
            None
        )  # Support existing operator-only unlabelled projections.
    identity = (
        parameter["Type"],
        parameter["Name"],
        parameter["Id"],
        parameter["ParameterVersion"],
    )
    if (
        identity != ("Secret", name, parameter_id, version)
        or tags.get("environment") != "test-shanghai-copy"
        or (pool_scoped and tags.get("role-pool") != pool)
        or tags.get("purpose") not in valid_purposes
        or tags.get("phase") == "staged"
    ):
        raise ValueError(
            "OOS secret is staged or has a different scope, purpose or version"
        )

    return parameter_id, version


def launch_args(binding, profile):
    receipt_env = []
    if "refs" in binding:
        from control_bindings import marker, parse_manifest

        manifest = parse_manifest(json.dumps(binding))
        ref = manifest["refs"]["operator"]
        parameter_id, version = check_reference(
            dict(ref, pool_id=manifest["pool_id"]), profile, "operator"
        )
        receipt_env = ["--env", "ROLE_POOL_OPERATOR_TOKEN_OOS_REF=" + marker(ref)]
    else:
        parameter_id, version = check_reference(binding, profile)
    # The credential manager supplies this versioned secret to each launch host.
    # Inspect only its metadata. Pin the local object ID for the actual launch.
    local_name = f"sglang-operator-{parameter_id}-v{version}"
    result = subprocess.check_output(
        [
            "podman",
            "secret",
            "inspect",
            local_name,
            "--format",
            "{{.ID}} {{.Spec.Name}}",
        ],
        timeout=10,
        text=True,
    )
    object_id, observed_name = result.strip().split()
    if observed_name != local_name or not re.fullmatch(r"[a-zA-Z0-9]+", object_id):
        raise ValueError("Provisioned Podman secret does not match the OOS binding")
    return [
        "--secret",
        f"{object_id},type=env,target=ROLE_POOL_OPERATOR_TOKEN",
        *receipt_env,
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--aliyun-profile", required=True)
    parser.add_argument(
        "--verify-controls",
        action="store_true",
        help="Validate the shared control-plane manifest against OOS metadata",
    )
    args = parser.parse_args()
    if args.verify_controls:
        from control_bindings import manifest_digest, parse_manifest

        manifest = parse_manifest(args.binding.read_text())
        failures = []
        for purpose, ref in manifest["refs"].items():
            try:
                check_reference(
                    dict(ref, pool_id=manifest["pool_id"]), args.aliyun_profile, purpose
                )
            except (ValueError, subprocess.SubprocessError, OSError, KeyError):
                failures.append(purpose)
        if failures:
            parser.exit(
                1, "OOS reference check failed: " + ", ".join(sorted(failures)) + "\n"
            )
        print(
            json.dumps(
                {
                    "control_binding_sha256": manifest_digest(manifest),
                    "checked": sorted(manifest["refs"]),
                }
            )
        )
    else:
        print(
            json.dumps(
                launch_args(json.loads(args.binding.read_text()), args.aliyun_profile)
            )
        )
