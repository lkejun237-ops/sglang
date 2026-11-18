"""Run live-source tests in a private container using an existing Room image.

Credentials stay on the controller host. No shared container or route is changed.
The supplied root contains immutable source/, state.json, cases.json and media/.
"""

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path


def inside(root, prepare_only=False):
    if prepare_only:
        from sglang.multimodal_gen.runtime.realtime.whitebox.cli_room import cli_room
        from sglang.multimodal_gen.runtime.realtime.whitebox.contracts import VideoSpec

        async def export():
            cases = json.loads((root / "cases.json").read_text())
            os.environ["WHITEBOX_CLI_RESOURCE_TRANSPORT"] = "cdn"
            os.environ["WHITEBOX_CLI_CACHE_UPDATE"] = "sync"
            for case in cases["cases"]:
                room = await cli_room(
                    world_id=case["world_id"],
                    spec=VideoSpec(case["video"]["width"], case["video"]["height"], 24),
                    trace_id="live-export-" + case["case_id"],
                )
                try:
                    case["live_game_identity"] = room.descriptor["artifact"]
                    print(
                        "Prepared",
                        case["case_id"],
                        json.dumps(case["live_game_identity"]),
                        flush=True,
                    )
                finally:
                    await room.close()
            (root / "cases.json").write_text(
                json.dumps(cases, ensure_ascii=False, indent=2)
            )

        asyncio.run(export())
        return
    from prepare_isolated_role_replay import prepare
    from run_room_sglang_source_replay import run_source_replay_suite

    state = json.loads((root / "state.json").read_text())
    secret = os.environ.get("ROLE_POOL_SIGNING_SECRET")
    if secret is None:
        secret = Path(os.environ["ROLE_POOL_SIGNING_SECRET_FILE"]).read_text().strip()
        os.environ["ROLE_POOL_SIGNING_SECRET"] = secret
    endpoints = prepare(state, root / "authority", secret)
    result = asyncio.run(
        run_source_replay_suite(
            json.loads((root / "cases.json").read_text()),
            encoder_url=endpoints["encoder"],
            dit_url=endpoints["dit"],
            decoder_url=endpoints["decoder"],
            media_root=root / "media",
            output_dir=root / "results",
            mode="real-sglang",
            detailed_timing=True,
            flow_mode="natural",
            realtime_pacing=True,
            seed_base=117234900,
            preserve_switch_boundaries=True,
            max_frames_after_last_event=120,
        )
    )
    print(json.dumps({"status": result["status"], "cases": len(result["cases"])}))


def controller(args):
    def inspect(name):
        return json.loads(subprocess.check_output(["docker", "inspect", name]))[0]

    room = inspect(args.room)
    env = dict(x.split("=", 1) for x in room["Config"]["Env"])
    secret = ""
    if not args.prepare_only:
        coordinator = inspect(args.coordinator)
        authority = dict(x.split("=", 1) for x in coordinator["Config"]["Env"])
        secret = authority.get("ROLE_POOL_SIGNING_KEY") or authority.get(
            "ROLE_POOL_SIGNING_SECRET"
        )
        if not secret:
            raise ValueError(
                "Controller has no pre-provisioned role-pool signing credential"
            )
    env.update(
        ROLE_POOL_SIGNING_SECRET=secret,
        PYTHONPATH="/test/source/python",
        SGLANG_LIGHTWEIGHT_RUNTIME="1",
        WHITEBOX_CLI_SERVICE_MODE="off",
        WHITEBOX_INFERENCE_STEPS="2",
        WHITEBOX_MODEL_INPUT_FRAMES="17",
        WHITEBOX_DETAILED_TIMING="1",
        WHITEBOX_CLI_GAME_CACHE_DIRECTORY="/test/game-cache",
        WHITEBOX_SPOOL="/test/spool",
        WHITEBOX_ROOM_SPOOL="/test/spool",
    )
    env.pop("WHITEBOX_CLI_SERVICE", None)
    env.pop("WHITEBOX_ROLE_POOL_CONFIG", None)
    env.pop("WHITEBOX_RELEASE_MANIFEST", None)
    root = args.root.resolve()
    private = root / "controller.env"
    with open(private, "x", opener=lambda p, f: os.open(p, f, 0o600)) as file:
        for key, value in env.items():
            if "\n" in value:
                raise ValueError("Multiline container environment is unsupported")
            file.write(key + "=" + value + "\n")
    receipt = dict(
        image=room["Image"],
        room_source=room["Config"].get("Labels", {}),
        model_source=json.loads((root / "state.json").read_text())["source"],
        scope="Live Worldkit CLI -> in-process production SourceSession -> dedicated model roles",
    )
    (root / "controller-receipt.json").write_text(json.dumps(receipt, indent=2))
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--name",
                args.name,
                "--network",
                "host",
                "--shm-size",
                "1g",
                "--env-file",
                str(private),
                "-v",
                str(root) + ":/test",
                "--entrypoint",
                "python",
                room["Image"],
                "/test/source/benchmark/whitebox/run_live_source_controller.py",
                "--inside",
                "--root",
                "/test",
                *(["--prepare-only"] if args.prepare_only else []),
            ],
            check=True,
        )
    finally:
        private.unlink()


def render_pod(args):
    if not args.image or "@sha256:" not in args.image:
        raise ValueError("A pinned Room image digest is required")
    env = {
        "PYTHONPATH": "/test/source/python",
        "SGLANG_LIGHTWEIGHT_RUNTIME": "1",
        "WHITEBOX_CLI_ENTRYPOINT": "/opt/worldkit/packages/cli/bin/worldkit.mjs",
        "WHITEBOX_CLI_SERVICE_MODE": "off",
        "WHITEBOX_CLI_RESOURCE_TRANSPORT": "cdn",
        "WHITEBOX_CLI_PLATFORM_URL": "https://world-test-platform.loopit.com.cn",
        "WHITEBOX_CLI_CACHE_UPDATE": "async",
        "WHITEBOX_CLI_RENDERING": "auto",
        "WHITEBOX_CLI_GAME_CACHE_DIRECTORY": "/test/game-cache",
        "WHITEBOX_INFERENCE_STEPS": "2",
        "WHITEBOX_MODEL_INPUT_FRAMES": "17",
        "WHITEBOX_DETAILED_TIMING": "1",
        "WHITEBOX_MODEL_SUBCLIP_ADMISSION": "1",
        "WHITEBOX_MODEL_SOURCE_SAMPLING": "adaptive",
        "WHITEBOX_MODEL_STARTUP_PREPARE": "1",
        "ROLE_POOL_SIGNING_SECRET_FILE": "/role-auth/ROLE_POOL_SIGNING_KEY",
        "NVIDIA_VISIBLE_DEVICES": "void",
    }
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": args.name, "namespace": "wm1103-sh"},
        "spec": {
            "restartPolicy": "Never",
            "hostNetwork": True,
            "imagePullSecrets": [{"name": "image-reader-1711"}],
            "containers": [
                {
                    "name": "controller",
                    "image": args.image,
                    "command": [
                        "python",
                        "/test/source/benchmark/whitebox/run_live_source_controller.py",
                        "--inside",
                        "--root",
                        "/test",
                    ],
                    "env": [{"name": k, "value": v} for k, v in env.items()],
                    "volumeMounts": [
                        {"name": "test", "mountPath": "/test"},
                        {"name": "auth", "mountPath": "/role-auth", "readOnly": True},
                        {"name": "shm", "mountPath": "/dev/shm"},
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "test",
                    "hostPath": {"path": args.root.as_posix(), "type": "Directory"},
                },
                {"name": "auth", "secret": {"secretName": "model-role-identity"}},
                {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "1Gi"}},
            ],
        },
    }
    print(json.dumps(pod, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--render-pod", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--room", default="wm1103-room")
    parser.add_argument("--coordinator", default="wm1103-coordinator")
    parser.add_argument("--name", default="sglang-live-source-test")
    args = parser.parse_args()
    if args.render_pod:
        render_pod(args)
    elif args.inside:
        inside(args.root, args.prepare_only)
    else:
        controller(args)
