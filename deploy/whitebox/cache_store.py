"""One Mooncake 0.3.13 Master/Store per host; model slots are pure clients.

Run from a complete image with --config pointing to the same immutable JSON
passed to --zing-block0-mooncake-config. This launcher only owns its children.
"""

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from sglang.multimodal_gen.runtime.realtime.block0_rdma import block0_rdma_devices
from sglang.multimodal_gen.runtime.realtime.cache_tiers import CacheTiers


def settings(config, disk_total=None):
    policy = CacheTiers.parse(config.get("tiers", {}))
    memory = config.get("store_memory_bytes")
    if type(memory) is not int or memory < 64 << 20:
        raise ValueError(
            "Shared Store requires an explicit memory budget of at least 64 MiB"
        )
    host, port = config["master_server_address"].rsplit(":", 1)
    local = config.get("local_hostname") or socket.gethostbyname(socket.gethostname())
    if host not in {"127.0.0.1", "localhost", local}:
        raise ValueError("The cache Master must belong to this host")
    http = urlparse(config["master_http_url"])
    if (
        http.scheme != "http"
        or http.hostname not in {host, "127.0.0.1", "localhost", local}
        or not http.port
    ):
        raise ValueError("The cache metadata endpoint must belong to this host")
    offload = policy.l2 and policy.l1_to_l2
    command = [
        "mooncake_master",
        f"--port={int(port)}",
        f"--metrics_port={http.port}",
        f"--enable_offload={str(offload).lower()}",
        "--enable_disk_eviction=true",
        "--promotion_on_hit=true",
        "--promotion_admission_threshold=1",
        "--promotion_max_per_heartbeat=256",
    ]
    env = {"MC_STORE_MEMCPY": "0", "MC_STORE_PIN_MEMORY_MAX_BYTES": str(memory)}
    if policy.l2:
        # Native capacity/LRU and filesystem watermark eviction, not a second GC.
        if not disk_total or not 0 < policy.min_free_bytes < disk_total:
            raise ValueError(
                "Native L2 requires a measurable positive free-space reserve"
            )
        high = 1 - policy.min_free_bytes / disk_total
        if high <= 0.10:
            raise ValueError(
                "Native L2 free-space reserve leaves no usable disk budget"
            )
        command += [
            f"--root_fs_dir={policy.disk_directory}",
            "--offload_on_evict=false",
        ]
        env.update(
            MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR="bucket_storage_backend",
            MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=policy.disk_directory,
            MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY="lru",
            MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=str(policy.disk_bytes),
            MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES=str(policy.disk_bytes),
            MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=str(policy.disk_bytes),
            MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT="1",
            MOONCAKE_OFFLOAD_BUCKET_SIZE_LIMIT_BYTES=str(policy.max_object_bytes),
            MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=str(
                max(256 << 20, policy.max_object_bytes * 2)
            ),
            MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS="1",
            MOONCAKE_OFFLOAD_CLIENT_BUFFER_GC_TTL_MS="5000",
            MOONCAKE_OFFLOAD_ENABLE_DISK_WATERMARK_EVICTION="true",
            MOONCAKE_OFFLOAD_DISK_EVICTION_HIGH_WATERMARK_RATIO=str(high),
            MOONCAKE_OFFLOAD_DISK_EVICTION_LOW_WATERMARK_RATIO=str(high - 0.05),
        )
    store = dict(
        local_hostname=local,
        metadata_server=config.get("metadata_server", "P2PHANDSHAKE"),
        global_segment_size=memory,
        local_buffer_size=config.get("metadata_buffer_bytes", 64 << 20),
        protocol="rdma",
        rdma_devices=block0_rdma_devices(config),
        master_server_addr=config["master_server_address"],
        enable_ssd_offload=policy.l2,
        ssd_offload_path=policy.disk_directory or "",
    )
    return policy, command, env, store


def run(config_path):
    import importlib.metadata
    import shutil
    import threading

    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

    if importlib.metadata.version("mooncake-transfer-engine") != "0.3.13":
        raise ValueError("This launcher requires the verified Mooncake 0.3.13 contract")
    config = json.loads(config_path.read_text())
    policy = CacheTiers.parse(config.get("tiers", {}))
    root = Path(policy.state_directory)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    # The same host's independent launchers cannot race a second Store into use.
    with (root / "store.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        disk_total = None
        filesystem = None
        if policy.l2:
            disk = Path(policy.disk_directory)
            disk.mkdir(parents=True, exist_ok=True)
            fs = json.loads(
                subprocess.check_output(
                    ["findmnt", "-J", "-T", str(disk), "-o", "FSTYPE,SOURCE"], text=True
                )
            )["filesystems"][0]
            allowed = {"ext4", "xfs", "btrfs"}
            if policy.allow_overlay_l2:
                allowed.add("overlay")
            if fs["fstype"] not in allowed:
                raise ValueError(
                    "Native L2 requires a local block filesystem, not OSS/FUSE/network storage"
                )
            usage = shutil.disk_usage(disk)
            if (
                usage.free < policy.min_free_bytes
                or policy.disk_bytes + policy.min_free_bytes > usage.total
            ):
                raise ValueError(
                    "Native L2 disk budget or free-space reserve is unavailable"
                )
            disk_total = usage.total
            filesystem = {**fs, "total_bytes": usage.total, "free_bytes": usage.free}
        _, command, env, store_config = settings(config, disk_total)
        os.environ.update(env)
        stopped = threading.Event()
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: stopped.set())
        ready = root / "store-ready.json"
        ready.unlink(missing_ok=True)
        child = subprocess.Popen(command)
        store = None
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if child.poll() is not None or stopped.is_set():
                    raise RuntimeError("Cache Master startup ended")
                try:
                    with urllib.request.urlopen(
                        config["master_http_url"] + "/get_all_keys", timeout=1
                    ):
                        break
                except OSError:
                    stopped.wait(0.2)
            else:
                raise TimeoutError("Cache Master readiness deadline exceeded")
            store = MooncakeDistributedStore()
            if store.setup(store_config) != 0:
                raise RuntimeError("Shared cache Store setup failed")
            # Actual bytes must round-trip before model clients use this host.
            key = "store-probe:" + str(os.getpid())
            if (
                store.put(key, b"ready", ReplicateConfig()) != 0
                or store.get(key) != b"ready"
            ):
                raise RuntimeError("Shared cache Store readiness failed")
            ready.write_text(
                json.dumps(
                    dict(
                        pid=os.getpid(),
                        master_pid=child.pid,
                        mooncake_version="0.3.13",
                        memory_bytes=store_config["global_segment_size"],
                        protocol=store_config["protocol"],
                        rdma_devices=store_config["rdma_devices"],
                        l2=policy.l2,
                        l1_to_l2=policy.l1_to_l2,
                        filesystem=filesystem,
                    )
                )
            )
            reap_at = 0
            while not stopped.wait(1):
                if child.poll() is not None:
                    raise RuntimeError("Cache Master exited")
                if time.monotonic() >= reap_at:
                    from sglang.multimodal_gen.runtime.realtime.cache_tiers import (
                        prune_persistence_states,
                    )

                    prune_persistence_states(root)
                    reap_at = time.monotonic() + 60
        finally:
            ready.unlink(missing_ok=True)
            if store is not None:
                store.close()
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
