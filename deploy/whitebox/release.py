"""Manual main-branch release tools. No command deploys or modifies a cloud service."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from config import load, render
from docker_storage import check_headroom
from image_storage import inspect_local
from output_paths import outside_checkout

REPO = Path(__file__).resolve().parents[2]
RELEASE_ROLES = ("gateway", "coordinator", "room", "model")


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def exact_commit(repo, ref):
    return git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")


def archive_command(repo, sha, *, room_sdk=False):
    if room_sdk:
        script = Path(__file__).with_name("room-sdk-archive.sh")
        return ["sh", str(script), str(repo), sha]
    return ["git", "-C", str(repo), "archive", sha]


def archive_member_filter(member, destination):
    # Retain data_filter's path/link/type/ownership safety. Its directory mode
    # is unset, which would otherwise inherit a private builder's umask 077.
    filtered = tarfile.data_filter(member, destination)
    if filtered is not None and filtered.isdir():
        filtered = filtered.replace(mode=member.mode & 0o755)
    return filtered


def export(repo, sha, destination, *, room_sdk=False):
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as archive:
        subprocess.run(
            archive_command(repo, sha, room_sdk=room_sdk), stdout=archive, check=True
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive) as tar:
            tar.extractall(destination, filter=archive_member_filter)


def release_inputs(args):
    sha = exact_commit(REPO, args.source)
    # A feature commit can be tested locally, but the production builder only
    # accepts commits merged into the fetched production branch.
    production = exact_commit(REPO, "origin/main")
    subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", sha, production],
        check=True,
    )
    pin = git(REPO, "show", f"{sha}:benchmark/whitebox/sdk-revision.txt")
    if not re.fullmatch(r"[0-9a-f]{40}", pin):
        raise ValueError("SDK pin must be an immutable commit")
    exact_commit(args.sdk_checkout, pin)
    sdk_production = exact_commit(args.sdk_checkout, "origin/dev")
    subprocess.run(
        [
            "git",
            "-C",
            str(args.sdk_checkout),
            "merge-base",
            "--is-ancestor",
            pin,
            sdk_production,
        ],
        check=True,
    )
    return sha, pin


def build(args):
    sha, pin = release_inputs(args)
    if args.component == "model" and not re.fullmatch(
        r"[^\s]+@sha256:[0-9a-f]{64}", args.base or ""
    ):
        raise ValueError("Model dependency base must be pinned by digest")
    manifest = {
        "schema_version": 1,
        "source_branch": "main",
        "source_sha": sha,
        "sdk_sha": pin,
    }
    if args.component in {"cpu", "room"}:
        incoming = getattr(args, "incoming_gib", None)
        check_headroom(
            incoming
            if incoming is not None
            else (6 if args.component == "room" else 1),
            getattr(args, "reserve_gib", 15),
        )
    # Fresh git archives prevent local dirty files, runtime overlays, credentials
    # and developer node_modules from leaking into a release context.
    with tempfile.TemporaryDirectory(prefix="whitebox-build-") as scratch:
        context = Path(scratch)
        export(REPO, sha, context)
        export(
            args.sdk_checkout,
            pin,
            context / ".whitebox-sdk",
            room_sdk=args.component == "room",
        )
        dockerfile, target = (
            ("docker/whitebox-model.Dockerfile", "model")
            if args.component == "model"
            else (
                "docker/whitebox.Dockerfile",
                "room" if args.component == "room" else "release",
            )
        )
        command = [
            "docker",
            "build",
            "--file",
            dockerfile,
            "--target",
            target,
            "--tag",
            args.tag,
            "--build-arg",
            "WHITEBOX_SOURCE_SHA=" + sha,
            "--build-arg",
            "WHITEBOX_SOURCE_BRANCH=main",
            "--build-arg",
            "WHITEBOX_SDK_SHA=" + pin,
            "--label",
            "org.opencontainers.image.revision=" + sha,
            "--label",
            "com.seedleap.worldkit.revision=" + pin,
        ]
        if args.component == "model":
            command += ["--build-arg", "MODEL_BASE_IMAGE=" + args.base]
        if args.platform:
            command += ["--platform", args.platform]
        # No --push, registry login, cloud API or deployment hook.
        subprocess.run(command + ["."], cwd=context, check=True)
    storage = (
        inspect_local(args.tag, args.component)
        if args.component in {"cpu", "room"}
        else None
    )
    print(
        json.dumps(
            {
                **manifest,
                "local_image_tag": args.tag,
                "component": args.component,
                "storage": storage,
                "deployed": False,
            },
            indent=2,
        )
    )


def verify(bundle, source, sdk, observations):
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("Expected an exact source commit")
    if bundle["policy"]["release_branch"] != "main":
        raise ValueError("Production verification requires a main release policy")
    for role in RELEASE_ROLES:
        observed = observations.get(role)
        if not isinstance(observed, dict):
            raise ValueError(f"Missing runtime observation: {role}")
        if not re.fullmatch(
            r"[^\s]+@sha256:[0-9a-f]{64}", observed.get("platform_image", "")
        ):
            raise ValueError(f"Missing actual platform image digest for {role}")
        version = observed.get("version")
        if not isinstance(version, dict):
            raise ValueError(f"Missing runtime version: {role}")
        expected = {
            "role": role,
            "source_branch": "main",
            "source_sha": source,
            "sdk_sha": sdk,
            "config_sha256": bundle["config_sha256"],
            "region": bundle["region"],
            "identity_status": "configured",
        }
        if any(version.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Runtime identity mismatch: {role}")
    pins = {observations[role]["version"].get("sdk_sha") for role in RELEASE_ROLES}
    if len(pins) != 1 or not re.fullmatch(r"[0-9a-f]{40}", next(iter(pins)) or ""):
        raise ValueError("SDK identities differ between roles")
    return {
        "identity_match": True,
        "performance_verified": False,
        "region": bundle["region"],
        "source_sha": source,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    plan = sub.add_parser(
        "plan", help="Compile config locally; does not build or deploy"
    )
    plan.add_argument("--region", choices=("shanghai", "thailand"), required=True)
    plan.add_argument("--bindings", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    b = sub.add_parser(
        "build", help="Explicit local image build, with no push or activation"
    )
    b.add_argument(
        "--source", required=True, help="Commit reachable from fetched origin/main"
    )
    b.add_argument("--sdk-checkout", type=Path, required=True)
    b.add_argument("--component", choices=("cpu", "room", "model"), required=True)
    b.add_argument("--tag", required=True)
    b.add_argument("--base", help="Model dependency image pinned by digest")
    b.add_argument("--platform", default="linux/amd64")
    b.add_argument(
        "--incoming-gib",
        type=float,
        help="Conservative unpacked CPU/Room image budget (defaults: 1/6 GiB)",
    )
    b.add_argument(
        "--reserve-gib",
        type=float,
        default=15,
        help="Keep this much free Docker disk after the build allowance",
    )
    a = sub.add_parser(
        "room-sdk-archive", help="Write the pinned Room SDK tar to standard output"
    )
    a.add_argument("--source", required=True)
    a.add_argument("--sdk-checkout", type=Path, required=True)
    v = sub.add_parser(
        "verify", help="Compare captured runtime identities, without network access"
    )
    v.add_argument("--bundle", type=Path, required=True)
    v.add_argument("--source", required=True)
    v.add_argument(
        "--sdk", required=True, help="Expected SDK commit from this source release"
    )
    v.add_argument("--observations", type=Path, required=True)
    args = p.parse_args()
    if args.command == "plan":
        outside_checkout(args.output)
        bundle = render(load(args.region), json.loads(args.bindings.read_text()))
        # Never overwrite an earlier reviewed release plan implicitly.
        with args.output.open("x") as f:
            json.dump(bundle, f, indent=2)
        print(
            json.dumps(
                {
                    "region": bundle["region"],
                    "config_sha256": bundle["config_sha256"],
                    "output": str(args.output),
                    "deployed": False,
                }
            )
        )
    elif args.command == "room-sdk-archive":
        _, pin = release_inputs(args)
        subprocess.run(
            archive_command(args.sdk_checkout, pin, room_sdk=True),
            stdout=sys.stdout.buffer,
            check=True,
        )
    elif args.command == "build":
        build(args)
    else:
        print(
            json.dumps(
                verify(
                    json.loads(args.bundle.read_text()),
                    args.source,
                    args.sdk,
                    json.loads(args.observations.read_text()),
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
