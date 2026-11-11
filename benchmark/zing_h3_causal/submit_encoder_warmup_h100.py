# SPDX-License-Identifier: Apache-2.0
"""Create an isolated Encoder, Decoder or SP2 DiT warmup Pod on the H100 node.

Run on the node through Eflo, from a verified archive of a pushed commit. The
CPU preflight must finish successfully before the GPU phase can be submitted.
Only the GPU selector inspects other Pods, and it emits no workload metadata.
This helper does not delete Pods; collect logs and delete these exact names when
finished. With --registry-instance, also delete the run's NAME-pull Secret after
both phases finish. Python 3.6 syntax is intentional for the node interpreter.
"""

import argparse
import fcntl
import json
import re
import runpy
import subprocess
from pathlib import Path


def read(argv):
    return subprocess.check_output(argv, universal_newlines=True)


def select_gpus(source, count):
    """Select from one snapshot while the caller holds the allocation lock."""
    select = runpy.run_path(str(source / "deploy/sglang/select_gpu.py"))["select"]
    gpus = []
    for row in read(
        ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"]
    ).splitlines():
        device, memory = row.split(",")
        gpus.append({"uuid": device.strip(), "memory": int(memory.strip())})
    processes = read(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"]
    ).splitlines()
    pods = json.loads(
        read(["kubectl", "get", "pods", "--all-namespaces", "-o", "json"])
    )
    selected = []
    for _ in range(count):
        gpu = select(gpus, processes + selected, pods)
        if not re.fullmatch(r"GPU-[0-9a-f-]+", gpu) or gpu in selected:
            raise RuntimeError("GPU selector did not return isolated GPU UUIDs")
        selected.append(gpu)
    return selected


def create_pull_secret(instance, image, name, namespace):
    """Keep temporary registry credentials in memory and kubectl stdin only."""
    diagnostic = {"stage": "GetAuthorizationToken", "returncode": None}
    authorization_metadata = {}

    def safe_metadata(response):
        sensitive = [
            value
            for key in ("TempUsername", "AuthorizationToken")
            for value in (response.get(key),)
            if isinstance(value, str) and value
        ]
        metadata = {}
        for key, pattern in (
            ("Code", r"[A-Za-z][A-Za-z0-9_.-]{0,127}"),
            ("RequestId", r"[0-9A-Fa-f-]{16,64}"),
            ("ExpireTime", r"[0-9][0-9TtZz: .+-]{0,63}"),
            ("Millisecond", r"[0-9]{1,20}"),
        ):
            value = response.get(key)
            if type(value) not in (str, int):
                continue
            text = str(value)
            if re.fullmatch(pattern, text) and not any(s in text for s in sensitive):
                metadata[key] = value
        return metadata

    try:
        authorization = subprocess.run(
            [
                "aliyun",
                "cr",
                "GetAuthorizationToken",
                "--region",
                "cn-shanghai",
                "--endpoint",
                "cr-vpc.cn-shanghai.aliyuncs.com",
                "--version",
                "2018-12-01",
                "--InstanceId",
                instance,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=60,
        )
        diagnostic["returncode"] = authorization.returncode
        credentials = None
        # Error responses can be JSON on either stream; retain only explicit,
        # format-checked metadata, never the response or its message text.
        for stream_name, stream in (
            ("stdout", authorization.stdout),
            ("stderr", authorization.stderr),
        ):
            try:
                response = json.loads(stream)
            except (TypeError, ValueError):
                continue
            if isinstance(response, dict):
                authorization_metadata.update(safe_metadata(response))
                if stream_name == "stdout":
                    credentials = response
        diagnostic.update(authorization_metadata)
        if authorization.returncode != 0:
            diagnostic["reason"] = "command_failed"
            raise RuntimeError("Registry authorization command failed")
        if credentials is None:
            diagnostic["reason"] = "invalid_json_response"
            raise ValueError("Registry authorization response is not a JSON object")
        diagnostic["reason"] = "missing_credentials"
        username = credentials["TempUsername"]
        token = credentials["AuthorizationToken"]
        if not isinstance(username, str) or not username:
            raise ValueError("Registry username is missing")
        if not isinstance(token, str) or not token:
            raise ValueError("Registry token is missing")
        config = {
            "auths": {image.split("/", 1)[0]: {"username": username, "password": token}}
        }
        secret = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": name, "namespace": namespace},
            "type": "kubernetes.io/dockerconfigjson",
            "stringData": {".dockerconfigjson": json.dumps(config)},
        }
        diagnostic = {"stage": "kubectl create", "returncode": None}
        created = subprocess.run(
            ["kubectl", "create", "-f", "-"],
            input=json.dumps(secret),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=60,
        )
        diagnostic["returncode"] = created.returncode
        if created.returncode != 0:
            diagnostic["reason"] = "command_failed"
            raise RuntimeError("Run-specific pull Secret creation failed")
    except Exception as exc:
        # CLI errors and JSON decoding diagnostics may contain response/input
        # data. Do not forward either stream or exception details to the logs.
        diagnostic["error_type"] = type(exc).__name__
        raise RuntimeError(
            "Run-specific pull Secret creation failed: " + json.dumps(diagnostic)
        ) from None
    print(
        json.dumps(
            {
                "pull_secret": name,
                "created": True,
                "authorization": authorization_metadata,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "gpu"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--role", choices=("encoder", "decoder", "dit"), default="encoder"
    )
    parser.add_argument("--decoder-config-path", type=Path)
    parser.add_argument("--pipeline-config-path", type=Path)
    parser.add_argument("--quantization")
    parser.add_argument("--mem-fraction-static", type=float)
    parser.add_argument("--service-count", type=int, default=1)
    parser.add_argument(
        "--registry-instance",
        help="Mint a temporary pull Secret during preflight; reuse it during the GPU phase",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be a full pushed commit SHA")
    if not re.fullmatch(r"[a-z0-9-]{1,50}", args.name):
        parser.error("name must be a lowercase Kubernetes name of at most 50 chars")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", args.image):
        parser.error("image must be pinned by digest")
    if args.service_count < 1:
        parser.error("service-count must be positive")
    if args.role == "encoder" and args.service_count != 1:
        parser.error("Encoder startup validation uses one service")
    if args.role == "decoder" and args.decoder_config_path is None:
        parser.error("Decoder startup validation requires decoder-config-path")
    if args.role == "dit":
        if args.pipeline_config_path is None:
            parser.error("DiT startup validation requires pipeline-config-path")
        if args.service_count != 1:
            parser.error("SP2 DiT startup validation uses one service")
    args.source, args.output = args.source.resolve(), args.output.resolve()
    if not str(args.model).startswith("/mnt/zing-nvme/"):
        parser.error("model must be under the documented H100 model root")
    namespace = "minwm-realtime"
    name = args.name + "-" + args.phase
    existing = read(
        [
            "kubectl",
            "-n",
            namespace,
            "get",
            "pod",
            name,
            "--ignore-not-found",
            "-o",
            "name",
        ]
    ).strip()
    if existing:
        raise RuntimeError(
            "Test Pod already exists; inspect it instead of resubmitting"
        )
    if args.phase == "gpu":
        prep = json.loads(
            read(
                [
                    "kubectl",
                    "-n",
                    namespace,
                    "get",
                    "pod",
                    args.name + "-preflight",
                    "-o",
                    "json",
                ]
            )
        )
        result = json.loads((args.output / "preflight" / "result.json").read_text())
        if prep["status"].get("phase") != "Succeeded" or not result.get("passed"):
            raise RuntimeError("CPU preflight has not passed")
        if result.get("revision") != args.revision:
            raise RuntimeError("CPU preflight revision differs from the GPU test")
    args.output.mkdir(parents=True, exist_ok=True)
    harness = "/source/benchmark/zing_h3_causal/validate_{}_startup_warmup.py".format(
        args.role
    )
    command = [
        "/opt/sglang/bin/python",
        "-u",
        harness,
        "--model",
        str(args.model),
        "--output",
        "/results/" + args.phase,
        "--revision",
        args.revision,
    ]
    if args.phase == "preflight":
        command.append("--preflight")
    if args.role == "decoder":
        command.extend(
            [
                "--decoder-config-path",
                str(args.decoder_config_path),
                "--service-count",
                str(args.service_count),
            ]
        )
    if args.role == "dit":
        command.extend(["--pipeline-config-path", str(args.pipeline_config_path)])
        if args.quantization is not None:
            command.extend(["--quantization", args.quantization])
        if args.mem_fraction_static is not None:
            command.extend(["--mem-fraction-static", str(args.mem_fraction_static)])
    env = {
        "PYTHONPATH": "/source/python",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "4",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
    }
    if args.role == "dit" and args.phase == "gpu":
        # Match the Whitebox CUDA allocator configuration for model loading.
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    volumes, mounts = [], []
    for label, host, mount, readonly in (
        ("source", str(args.source), "/source", True),
        ("results", str(args.output), "/results", False),
        ("models", "/mnt/zing-nvme", "/mnt/zing-nvme", True),
    ):
        volumes.append({"name": label, "hostPath": {"path": host, "type": "Directory"}})
        mounts.append({"name": label, "mountPath": mount, "readOnly": readonly})
    volumes.append(
        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "4Gi"}}
    )
    mounts.append({"name": "shm", "mountPath": "/dev/shm"})
    spec = {
        "nodeName": "world-model-b27mys7b-0000",
        # Use the same runtime handler for both phases. The CPU phase still
        # exposes no GPU: NVIDIA_VISIBLE_DEVICES=void is explicit above.
        "runtimeClassName": "nvidia",
        "restartPolicy": "Never",
        # DiT allows 1800 seconds for startup, plus rank cleanup and Pod startup.
        "activeDeadlineSeconds": (
            (2100 if args.role == "dit" else 1200) if args.phase == "gpu" else 600
        ),
        "terminationGracePeriodSeconds": 45,
        "containers": [
            {
                "name": "test",
                "image": args.image,
                # Kubelet may need to verify the namespace's existing pull
                # credentials even when this pinned image is already cached.
                "imagePullPolicy": "IfNotPresent",
                "command": command,
                "volumeMounts": mounts,
                "resources": {
                    "requests": {"cpu": "2", "memory": "4Gi"},
                    "limits": {"cpu": "8", "memory": "128Gi"},
                },
            }
        ],
        "volumes": volumes,
    }
    if args.registry_instance:
        spec["imagePullSecrets"] = [{"name": args.name + "-pull"}]
    manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {"app": args.role + "-startup-warmup"},
            "annotations": {"sglang/source-revision": args.revision},
        },
        "spec": spec,
    }
    with open("/tmp/sglang-role-allocation.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == "gpu":
            selected = select_gpus(args.source, 2 if args.role == "dit" else 1)
            env["NVIDIA_VISIBLE_DEVICES"] = ",".join(selected)
        spec["containers"][0]["env"] = [{"name": k, "value": v} for k, v in env.items()]
        path = args.output / (args.phase + "-pod.json")
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        subprocess.check_call(
            ["kubectl", "create", "--dry-run=client", "-f", str(path)]
        )
        if args.registry_instance and args.phase == "preflight":
            create_pull_secret(
                args.registry_instance, args.image, args.name + "-pull", namespace
            )
        subprocess.check_call(["kubectl", "create", "-f", str(path)])
    print(
        json.dumps(
            {
                "pod": name,
                "namespace": namespace,
                "revision": args.revision,
                "gpu": env["NVIDIA_VISIBLE_DEVICES"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
