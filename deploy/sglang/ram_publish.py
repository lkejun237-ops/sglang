"""RAM-triggered in-place attachment of one reviewed Shanghai test deployment."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import orchestrate as orch

REGION = "cn-shanghai"
PREFIX = "test-shanghai-copy/attach/"
PRIVATE_ENV = {
    "ROLE_POOL_OPERATOR_TOKEN",
    "ROLE_POOL_SIGNING_KEY",
    "ROLE_POOL_REGISTRATION_TOKEN",
    "WHITEBOX_ROOM_TOKEN",
}


class ReleaseInbox:
    """Only object metadata is consulted. Requesters cannot supply executor inputs."""

    def __init__(self, profile):
        if not profile or not profile.strip():
            raise ValueError("An explicit Aliyun profile is required")
        self.profile = profile
        account = self.cli("sts", "GetCallerIdentity", "--region", REGION)["AccountId"]
        if not re.fullmatch(r"[0-9]{8,32}", account):
            raise ValueError("Invalid RAM account ID")
        self.bucket = "sglang-shcopy-publish-" + account
        info = self.oss("get-bucket-info")["Bucket"]
        if (
            info["Name"] != self.bucket
            or info["Location"] != "oss-cn-shanghai"
            or info["Owner"]["ID"] != account
            or info["AccessControlList"]["Grant"] != "private"
            or str(info.get("BlockPublicAccess")).lower() != "true"
        ):
            raise ValueError(
                "Expected an owner-controlled, private Shanghai publish bucket"
            )

    def cli(self, *args):
        try:
            raw = subprocess.check_output(
                ["aliyun", "--profile", self.profile, *args],
                stderr=subprocess.PIPE,
                timeout=30,
                env={k: v for k, v in os.environ.items() if k not in PRIVATE_ENV},
            )
        except (OSError, subprocess.SubprocessError):
            raise RuntimeError(
                f"Aliyun {args[:3]} failed; check profile, bucket and RAM grant"
            ) from None
        return json.loads(raw) if raw.strip() else {}

    def oss(self, operation, *args):
        return self.cli(
            "ossutil",
            "api",
            operation,
            "--bucket",
            self.bucket,
            "--region",
            REGION,
            "--endpoint",
            "https://oss-cn-shanghai.aliyuncs.com",
            "--output-format",
            "json",
            *args,
        )

    def submit(self, key):
        # All callers write the same empty marker. Repeating a PUT cannot
        # enqueue a second action; the executor claims the approved state once.
        self.oss("put-object", "--key", key, "--body", "")

    def requested(self, key):
        page = self.oss("list-objects-v2", "--prefix", key, "--max-keys", "1")
        count = int(page["KeyCount"])
        if count == 0:
            return False
        entries = page["Contents"]
        if isinstance(entries, dict):
            entries = [entries]
        if count != 1 or len(entries) != 1:
            raise RuntimeError("Unexpected OSS request listing")
        row = entries[0]
        if row["Key"] != key:
            return False
        if int(row["Size"]) != 0:
            raise ValueError("Invalid deployment request marker")
        return True


def request_key(release):
    if not re.fullmatch(r"[a-f0-9]{64}", release):
        raise ValueError("Expected a SHA-256 release ID from the trusted publisher")
    return PREFIX + release


def validate_scope(state):
    for target in (state["target"], state["spec"]["target"]):
        for name in (
            "environment",
            "region",
            "workspace_id",
            "pool_id",
            "coordinator_url",
            "room_url",
            "wtp_url",
            "quota_id",
            "node_id",
            "h100_ip",
        ):
            if target.get(name) != orch.TARGET[name]:
                raise ValueError(
                    f"Release does not target the Shanghai test pool ({name})"
                )


def approved_release(state):
    validate_scope(state)
    if state.get("attachment"):
        raise ValueError(
            "Use native reconciliation for a previously attached deployment"
        )
    tested = state.get("selftest", {})
    if (
        state.get("phase") != "ready"
        or tested.get("status") != "passed"
        or tested.get("source") != state["source"]
        or tested.get("worker_ids")
        != {r: state["resources"][r]["receipt"]["worker_id"] for r in orch.ROLES}
    ):
        raise ValueError(
            "Release requires a successful selftest of these exact workers"
        )
    # The handle binds the entire validated local plan, including run, resources,
    # images, source and test evidence. Never reconstruct a plan from an OSS object.
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def serve(path, release, inbox, cloud, timeout):
    if path.name != "state.json":
        raise ValueError("Use the publisher's canonical state.json")
    directory = path.parent
    key = request_key(release)
    intent = directory / ("ram-attach-" + release + ".intent")
    completed = directory / ("ram-attach-" + release + ".json")
    state = orch.load_state(path)
    validate_scope(state)
    identity = {
        "release": release,
        "deployment_id": state["deployment_id"],
        "run_id": state["run_id"],
        "generation": state["generation"],
    }
    if completed.exists():
        receipt = orch.read(completed)
        if receipt != {**identity, "status": "attached"}:
            raise ValueError("Unrecognized native attachment receipt")
        return receipt
    if intent.exists():
        raise RuntimeError("Previous attachment outcome requires native reconciliation")
    if release != approved_release(state):
        raise ValueError("Release ID no longer matches the authorized test deployment")
    orch.require_attach_credentials(state["target"]["pool_id"])
    deadline = time.monotonic() + timeout
    while not inbox.requested(key):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("No RAM request received for this release")
        time.sleep(min(5, remaining))

    state = orch.load_state(path)
    if approved_release(state) != release:
        raise ValueError("Release state changed while waiting for the RAM request")
    # Verify the actual bearer on the VPC before any cloud or native mutation.
    # A RAM request cannot establish or replace the executor's pool authority.
    scope = {
        "pool_id": state["target"]["pool_id"],
        "control_epoch": state["spec"]["pool"]["control_epoch"],
    }
    verified = orch.http(
        state["target"]["coordinator_url"] + "/v1/role-pool/operator-check",
        {},
        os.environ["ROLE_POOL_OPERATOR_TOKEN"],
    )
    if verified != scope:
        raise ValueError(
            "Coordinator operator scope does not match the authorized deployment"
        )
    # All executors for this state share its protected release directory.
    # On an unknown outcome, the native run must be reconciled before any retry.
    with intent.open("x") as stream:
        stream.write(release + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    orch.attach(state, directory, cloud, execute=True, in_place=True)
    orch._maybe_sync_oss_history(state, directory, cloud)
    receipt = {**identity, "status": "attached"}
    orch.save(completed, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser(
        "plan", help="Identify a reviewed, self-tested local release"
    )
    prepare.add_argument("--state", required=True, type=Path)
    request = actions.add_parser(
        "request", help="Submit a request with the caller's RAM profile"
    )
    runner = actions.add_parser(
        "serve", help="Wait for a request on the trusted release runner"
    )
    runner.add_argument("--state", required=True, type=Path)
    runner.add_argument("--timeout", type=int, default=900)
    for action in (request, runner):
        action.add_argument("--release", required=True)
        action.add_argument("--aliyun-profile", required=True)
        action.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "plan":
        print(json.dumps({"release": approved_release(orch.load_state(args.state))}))
        return
    key = request_key(args.release)
    if args.action == "serve" and args.timeout < 0:
        parser.error("--timeout must be non-negative")
    if not args.execute:
        print(
            json.dumps(
                {
                    "environment": "test-shanghai-copy",
                    "key": key,
                    "status": "not-executed",
                }
            )
        )
        return
    inbox = ReleaseInbox(args.aliyun_profile)
    if args.action == "request":
        inbox.submit(key)
        result = {"release": args.release, "status": "requested"}
    else:
        cloud = orch.Cloud(
            target=orch.TARGET, aliyun_profile=args.aliyun_profile, relay_vpc_http=False
        )
        result = serve(args.state, args.release, inbox, cloud, args.timeout)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
