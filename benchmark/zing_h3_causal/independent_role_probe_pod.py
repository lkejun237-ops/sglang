# SPDX-License-Identifier: Apache-2.0
"""Render an isolated H100 load-probe Pod; never select or allocate GPUs here."""

import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "load"))
    for name in (
        "root",
        "image",
        "revision",
        "model",
        "lynn",
        "deps",
        "name",
        "advertise-host",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--encoder-uuid")
    p.add_argument("--decoder-uuid")
    p.add_argument("--encoder-port", type=int, default=41171)
    p.add_argument("--decoder-port", type=int, default=41671)
    p.add_argument("--hold-seconds", type=int, default=300)
    args = p.parse_args()
    if "@sha256:" not in args.image:
        p.error("Use an exact cached image digest")
    if args.mode == "load" and (
        not args.encoder_uuid
        or not args.decoder_uuid
        or args.encoder_uuid == args.decoder_uuid
    ):
        p.error("Load requires two distinct allocated GPU UUIDs")
    source = args.root + "/source"
    command = [
        "/opt/sglang/bin/python",
        "-u",
        source + "/benchmark/zing_h3_causal/validate_independent_role.py",
        args.mode,
        "--root",
        args.root + "/run",
        "--revision",
        args.revision,
        "--advertise-host",
        args.advertise_host,
        "--hold-seconds",
        str(args.hold_seconds),
    ]
    env = {
        "PYTHONPATH": source
        + "/python:"
        + args.deps
        + ":"
        + source
        + "/benchmark/zing_h3_causal",
        "SGLANG_LIGHTWEIGHT_RUNTIME": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "4",
        "NCCL_IB_DISABLE": "1",
        "NCCL_NET": "Socket",
        "NCCL_DEBUG": "WARN",
        "TRITON_CACHE_DIR": args.root + "/triton",
        "TORCHINDUCTOR_CACHE_DIR": args.root + "/inductor",
    }
    containers = []
    for role in ["prepare"] if args.mode == "prepare" else ["encoder", "decoder"]:
        local_env = dict(env)
        local_command = list(command)
        if args.mode == "prepare":
            local_env.update(
                NVIDIA_VISIBLE_DEVICES="void",
                CUDA_VISIBLE_DEVICES="",
                SGLANG_DIFFUSION_PLATFORM_OVERRIDE="cpu",
            )
            local_command += [
                "--model",
                args.model,
                "--lynn",
                args.lynn,
                "--http-port",
                str(args.encoder_port),
            ]
        else:
            local_env.update(
                NVIDIA_VISIBLE_DEVICES=getattr(args, role + "_uuid"),
                CUDA_VISIBLE_DEVICES="0",
                NVIDIA_DRIVER_CAPABILITIES="compute,utility",
            )
            local_command += [
                "--role",
                role,
                "--http-port",
                str(getattr(args, role + "_port")),
            ]
        containers.append(
            dict(
                name=role,
                image=args.image,
                imagePullPolicy="Never",
                command=local_command,
                env=[dict(name=k, value=v) for k, v in local_env.items()],
                resources=dict(
                    requests=dict(cpu="4", memory="24Gi"),
                    limits=dict(cpu="12", memory="160Gi"),
                ),
                volumeMounts=[
                    dict(name="nvme", mountPath="/mnt/zing-nvme", readOnly=True),
                    dict(name="work", mountPath=args.root),
                    dict(name="shm", mountPath="/dev/shm"),
                ],
            )
        )
    pod = dict(
        apiVersion="v1",
        kind="Pod",
        metadata=dict(
            name=args.name,
            namespace="minwm-realtime",
            labels={"codex-task": "pr471-independent-acceptance"},
            annotations={"runtime-sha": args.revision},
        ),
        spec=dict(
            nodeName="world-model-b27mys7b-0000",
            hostNetwork=True,
            dnsPolicy="ClusterFirstWithHostNet",
            runtimeClassName="nvidia",
            restartPolicy="Never",
            activeDeadlineSeconds=1800,
            containers=containers,
            volumes=[
                dict(
                    name="nvme", hostPath=dict(path="/mnt/zing-nvme", type="Directory")
                ),
                dict(name="work", hostPath=dict(path=args.root, type="Directory")),
                dict(name="shm", emptyDir=dict(medium="Memory", sizeLimit="16Gi")),
            ],
        ),
    )
    print(json.dumps(pod, indent=2))


if __name__ == "__main__":
    main()
