"""Manual, all-role launcher for an existing four-GPU allocation.

This does not allocate GPUs, submit PAI jobs, stop other processes, copy weights,
clear caches, alter admission or activate a regional deployment. The platform
operator owns those actions. Without --run it only prints the command plan.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from config import _load_profile_launcher, fingerprint, render
from output_paths import outside_checkout
from prepare_assets import verify_prepared

HERE = Path(__file__).resolve().parent
_LAUNCHER = _load_profile_launcher()


def commands(bundle, *, python=sys.executable):
    return _LAUNCHER.build_integrated_group_commands(
        bundle,
        python=python,
        fixed_roles_script=HERE / "fixed_roles.py",
    )


def hardware(bundle, expected):
    import torch

    if torch.cuda.device_count() != 4 or len(expected) != 4 or len(set(expected)) != 4:
        raise ValueError("Exactly four explicitly owned GPU UUIDs must be visible")
    facts = []
    for i in range(4):
        p = torch.cuda.get_device_properties(i)
        uid = str(p.uuid)
        uid = uid if uid.startswith("GPU-") else "GPU-" + uid
        if (
            uid != expected[i]
            or [p.major, p.minor] != bundle["policy"]["gpu"]["capability"]
            or p.total_memory < bundle["policy"]["gpu"]["minimum_memory_gib"] * 2**30
        ):
            raise ValueError("Allocated GPUs do not match the reviewed regional plan")
        facts.append(
            {
                "uuid": uid,
                "memory_bytes": p.total_memory,
                "capability": [p.major, p.minor],
            }
        )
    return facts


def stop_children(children):
    _LAUNCHER.stop_children(children, os_module=os)


def wait_http(process, url, headers, timeout=600):
    _LAUNCHER.wait_http(
        process,
        url,
        headers,
        timeout=timeout,
        urllib_request=urllib.request,
        time_module=time,
    )


def run(bundle, expected, *, retries=0):
    if retries != 0:
        raise ValueError(
            "Hard faults require quarantining and manual recovery; automatic restart is disabled"
        )
    state = outside_checkout(bundle["bindings"]["state_root"])
    state.mkdir(parents=True, exist_ok=True)
    with (state / "launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        facts = hardware(bundle, expected)
        if bundle["policy"]["model"].get("cache_backend", "local") == "mooncake":
            from sglang.multimodal_gen.runtime.realtime.block0_rdma import (
                block0_rdma_devices,
            )
            from sglang.multimodal_gen.runtime.realtime.cache_tiers import CacheTiers

            cache_config = json.loads(
                Path(bundle["bindings"]["cache_config"]).read_text()
            )
            block0_rdma_devices(cache_config)
            CacheTiers.parse(cache_config.get("tiers", {}))
        root = Path(bundle["bindings"]["asset_root"])
        verify_prepared(root, bundle)
        pipeline = json.loads((root / "pipeline.json").read_text())
        config = json.loads((root / "model/zing_config.json").read_text())
        if "sink_chunks" in bundle["policy"]["model"]:
            profile = json.loads(
                (root / "model/zing_realtime_profile.json").read_text()
            )
            window = profile.get("source_attention_window", {})
            if (
                type(window.get("window_chunks")) is not int
                or window["window_chunks"] < 4
                or window.get("sink_chunks") != 0
                or pipeline.get("zing_kv_pool_sink_chunks") != 0
                or pipeline.get("zing_kv_pool_window_chunks") != window["window_chunks"]
            ):
                raise ValueError(
                    "Prepare matching S0 profile and KV pool before launching Whitebox"
                )
        if (
            config.get("num_frame_per_block") != [3, 2]
            or config.get("split_first_latent") is not False
        ):
            raise ValueError(
                "Prepared model must preserve [3,2] and split_first_latent=false"
            )
        mode = bundle["policy"]["model"]["adaln_mode"]
        if mode == "resident" and (
            pipeline.get("zing_adaln_mode") != "resident"
            or pipeline.get("zing_adaln_table_path")
        ):
            raise ValueError("Expected prepared resident AdaLN configuration")
        if mode == "table" and not pipeline.get("zing_adaln_table_path"):
            raise ValueError("Expected prepared AdaLN table configuration")
        worker_config = json.loads((root / "worker.secret.json").read_text())
        token = worker_config.get("bearer_token", worker_config.get("token"))
        if not isinstance(token, str) or not token:
            raise ValueError("Conditioning credential is missing")
        if (
            config.get("checkpoint_sha256")
            != bundle["policy"]["model"]["checkpoint_sha256"]
        ):
            raise ValueError("Prepared checkpoint identity differs from release policy")
        plan = commands(bundle)
        plan["decoder"]["argv"] += [
            "--decoder-config",
            json.dumps(pipeline["decoder_config"], sort_keys=True),
        ]
        evidence = {
            "config_sha256": bundle["config_sha256"],
            "launch_profile": bundle.get("launch_profile"),
            "hardware": facts,
            "asset_config_sha256": {
                str(p.relative_to(root)): __import__("hashlib")
                .sha256(p.read_bytes())
                .hexdigest()
                for p in (
                    root / "pipeline.json",
                    root / "model/zing_config.json",
                    root / "model/zing_realtime_profile.json",
                    root / "conditioning-profile.json",
                )
            },
        }
        (state / "launch.json").write_text(json.dumps(evidence, indent=2))
        _LAUNCHER.supervise_integrated_group(
            plan,
            bundle=bundle,
            state=state,
            expected=expected,
            config=config,
            evidence=evidence,
            token=token,
            retries=retries,
            subprocess_module=subprocess,
            os_module=os,
            signal_module=signal,
            time_module=time,
            wait_http_fn=wait_http,
            stop_children_fn=stop_children,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--run", action="store_true")
    p.add_argument(
        "--gpu-uuids", help="Four allocated UUIDs, in the reviewed role order"
    )
    p.add_argument(
        "--retries",
        type=int,
        choices=(0,),
        default=0,
        help="Automatic whole-group recovery is disabled",
    )
    args = p.parse_args()
    supplied = json.loads(args.bundle.read_text())
    bundle = render(supplied["policy"], supplied["bindings"])
    if fingerprint(supplied) != fingerprint(bundle):
        raise ValueError("Bundle changed after compilation")
    if not args.run:
        print(json.dumps(commands(bundle), indent=2))
    else:
        if not args.gpu_uuids:
            p.error("--run requires explicit --gpu-uuids")
        run(bundle, args.gpu_uuids.split(","), retries=args.retries)


if __name__ == "__main__":
    main()
