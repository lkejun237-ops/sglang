# SPDX-License-Identifier: Apache-2.0
"""Render an isolated, exact-GPU SM90 product review Pod; never submit it."""

import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "name",
        "image",
        "gpu-uuid",
        "host-root",
        "source-sha",
        "model",
        "lynn",
        "adaln-table",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--namespace", default="minwm-realtime")
    p.add_argument("--image-pull-secret")
    p.add_argument(
        "--frozen-latents",
        action="store_true",
        help="Run sequential Encoder/DiT and identical-latent Decoder comparison",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--node", default="world-model-b27mys7b-0000")
    args = p.parse_args()
    if args.resume and not args.frozen_latents:
        p.error("--resume requires --frozen-latents")
    runner = (
        "run_sm90_frozen_review.py"
        if args.frozen_latents
        else "run_sm90_product_review.py"
    )
    command = [
        "/opt/sglang/bin/python",
        "-u",
        f"/experiment/source/benchmark/zing_h3_causal/{runner}",
        "--model",
        args.model,
        "--lynn",
        args.lynn,
        "--adaln-table",
        args.adaln_table,
        "--media-root",
        "/experiment/media",
        "--output",
        "/experiment/results",
        "--source-sha",
        args.source_sha,
    ]
    if args.resume:
        command.append("--resume")
    env = {
        "NVIDIA_VISIBLE_DEVICES": args.gpu_uuid,
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        "PR693_EXPECTED_GPU": args.gpu_uuid,
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": "/experiment/source/python",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "NCCL_NET": "Socket",
        "NCCL_IB_DISABLE": "1",
        "SGLANG_LIGHTWEIGHT_RUNTIME": "1",
        "SGLANG_ZING_EXPERIMENTAL_ATTENTION": "fa4",
        "SGLANG_DIFFUSION_STAGE_LOGGING": "0",
    }
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": args.name, "namespace": args.namespace},
        "spec": {
            "nodeName": args.node,
            "runtimeClassName": "nvidia",
            "restartPolicy": "Never",
            "activeDeadlineSeconds": 14400,
            "containers": [
                {
                    "name": "review",
                    "image": args.image,
                    "imagePullPolicy": (
                        "IfNotPresent" if args.image_pull_secret else "Never"
                    ),
                    "command": command,
                    "env": [{"name": k, "value": v} for k, v in env.items()],
                    "resources": {
                        "requests": {"cpu": "8", "memory": "64Gi"},
                        "limits": {"cpu": "24", "memory": "192Gi"},
                    },
                    "volumeMounts": [
                        {
                            "name": "nvme",
                            "mountPath": "/mnt/zing-nvme",
                            "readOnly": True,
                        },
                        {"name": "owned", "mountPath": "/experiment"},
                        {"name": "shm", "mountPath": "/dev/shm"},
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "nvme",
                    "hostPath": {"path": "/mnt/zing-nvme", "type": "Directory"},
                },
                {
                    "name": "owned",
                    "hostPath": {"path": args.host_root, "type": "Directory"},
                },
                {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
            ],
        },
    }
    if args.image_pull_secret:
        pod["spec"]["imagePullSecrets"] = [{"name": args.image_pull_secret}]
    print(json.dumps(pod, indent=2))


if __name__ == "__main__":
    main()
