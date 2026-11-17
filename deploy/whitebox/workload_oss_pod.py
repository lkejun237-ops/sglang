"""Prepare a named, single-GPU test Pod for an existing scoped OSS Secret.

The Secret must contain a standard AWS shared credentials file named
`credentials`. This utility renders the Pod and checks the SDK path. Scheduling
and replacement are separate operator actions.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

MOUNT = "/var/run/whitebox-oss"
LEGACY_MOUNT = "/mnt/.alibabacloud"


def render(pod, *, expected_uid, gpu, name, secret, runtime_directory):
    meta, spec = pod["metadata"], copy.deepcopy(pod["spec"])
    if meta["uid"] != expected_uid or meta.get("ownerReferences"):
        raise ValueError("Standalone Pod identity changed")
    if len(spec["containers"]) != 1 or spec.get("initContainers"):
        raise ValueError("Expected a single Encoder container")
    container = spec["containers"][0]
    env = {v["name"]: v for v in container.get("env", [])}
    if container["name"] != "encoder" or env["NVIDIA_VISIBLE_DEVICES"]["value"] != gpu:
        raise ValueError("Encoder device changed")
    if "@sha256:" not in container["image"] or spec.get("activeDeadlineSeconds"):
        raise ValueError("Expected a digest-pinned Pod without a time limit")
    if not name or not secret or name == meta["name"]:
        raise ValueError("A distinct candidate name and Secret are required")
    if any(k.startswith("AWS_") for k in env):
        raise ValueError("Existing AWS provider configuration requires review")
    mounts = container["volumeMounts"]
    legacy = [m for m in mounts if m["mountPath"] == LEGACY_MOUNT]
    if len(legacy) != 1 or any(m["mountPath"] == MOUNT for m in mounts):
        raise ValueError("Unexpected credentials mounts")
    old_name = legacy[0]["name"]
    volume = next(v for v in spec["volumes"] if v["name"] == old_name)
    if "secret" not in volume:
        raise ValueError("Old credentials are not a Secret volume")
    volume["secret"] = {
        "secretName": secret,
        "defaultMode": 0o400,
        "items": [{"key": "credentials", "path": "credentials"}],
    }
    legacy[0]["mountPath"] = MOUNT
    legacy[0]["readOnly"] = True
    container["env"].extend(
        [
            {"name": "AWS_SHARED_CREDENTIALS_FILE", "value": MOUNT + "/credentials"},
            {"name": "AWS_EC2_METADATA_DISABLED", "value": "true"},
        ]
    )
    runtime = next(v["hostPath"] for v in spec["volumes"] if v["name"] == "runtime")
    old = Path(runtime["path"])
    new = Path(runtime_directory)
    if (
        not new.is_absolute()
        or new.parent != old.parent
        or new == old
        or ".." in new.parts
    ):
        raise ValueError("The new runtime must be a separate sibling directory")
    runtime.update(path=str(new), type="DirectoryOrCreate")

    # Keep the original selector and priority class. Let admission issue a fresh
    # ServiceAccount projection and the scheduler bind the replacement.
    if spec.get("nodeSelector", {}).get("kubernetes.io/hostname") != spec.get(
        "nodeName"
    ):
        raise ValueError("Frozen node selector differs from the scheduled node")
    spec.pop("nodeName")
    spec.pop("priority", None)
    projected = {
        v["name"]
        for v in spec["volumes"]
        if v["name"].startswith("kube-api-access-") and "projected" in v
    }
    spec["volumes"] = [v for v in spec["volumes"] if v["name"] not in projected]
    container["volumeMounts"] = [m for m in mounts if m["name"] not in projected]
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": meta["namespace"],
            "labels": dict(meta.get("labels", {})),
        },
        "spec": spec,
    }


def verify(region, endpoint, bucket, prefix, access_key_hash):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.realtime.cache_archive import S3CacheObjects
    from sglang.multimodal_gen.runtime.realtime.cache_credentials import storage_session

    if (
        not endpoint.startswith("https://")
        or not prefix.endswith("/")
        or not prefix.strip("/")
    ):
        raise ValueError("An HTTPS endpoint and a scoped probe prefix are required")
    session = storage_session()
    cred = session.get_credentials()
    frozen = cred.get_frozen_credentials()
    digest = hashlib.sha256(frozen.access_key.encode()).hexdigest()
    if (
        cred.method != "shared-credentials-file"
        or digest != access_key_hash
        or frozen.token
    ):
        raise ValueError(
            "The deployed application did not select the expected workload key"
        )
    client = S3CacheObjects(
        SimpleNamespace(endpoint=endpoint, bucket=bucket, region=region, deadline_s=30)
    ).client
    key = prefix + "identity-check-" + uuid4().hex
    payload = os.urandom(64)
    written = client.put_object(Bucket=bucket, Key=key, Body=payload)
    version = {"VersionId": written["VersionId"]} if written.get("VersionId") else {}
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
        try:
            if obj["Body"].read() != payload:
                raise ValueError("OSS read-back differs")
        finally:
            obj["Body"].close()
    finally:
        client.delete_object(Bucket=bucket, Key=key, **version)
    from botocore.exceptions import ClientError

    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
            raise
    else:
        raise ValueError("Probe object is still present")
    return {
        "provider": cred.method,
        "access_key_sha256": digest,
        "session_token": False,
        "put_get_delete_verified": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    create = commands.add_parser("render")
    create.add_argument("pod", type=Path)
    for arg in ("expected-uid", "gpu", "name", "secret", "runtime-directory"):
        create.add_argument("--" + arg, required=True)
    probe = commands.add_parser("verify")
    for arg in ("region", "endpoint", "bucket", "prefix", "access-key-hash"):
        probe.add_argument("--" + arg, required=True)
    args = vars(parser.parse_args())
    if args.pop("action") == "render":
        result = render(json.loads(args.pop("pod").read_text()), **args)
    else:
        result = verify(**args)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
