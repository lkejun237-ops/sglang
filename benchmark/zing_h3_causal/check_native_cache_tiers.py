"""Isolated CPU smoke for the host Store on an RDMA node, no OSS/GPU access."""

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import time
from pathlib import Path


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for(check, timeout, label):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = check()
        if value:
            return value
        time.sleep(0.1)
    raise TimeoutError(label)


def main(root, rdma_devices="", *, allow_overlay_l2=False, runtime_root=None):
    from sglang.multimodal_gen.runtime.realtime.block0_rdma import block0_rdma_devices

    root.mkdir(parents=True, exist_ok=False)
    root = root.resolve()
    repo = (runtime_root or Path(__file__).resolve().parents[2]).resolve()
    config = json.loads(
        (repo / "deploy/whitebox/cache/test-shanghai-copy.example.json").read_text()
    )
    config["rdma_devices"] = rdma_devices
    config["rdma_devices"] = block0_rdma_devices(config)
    config.update(
        store_memory_bytes=64 << 20,
        master_server_address=f"127.0.0.1:{port()}",
        master_http_url=f"http://127.0.0.1:{port()}",
    )
    config["tiers"].update(
        l3=False,
        l2_to_l3=False,
        state_directory=str(root / "state"),
        disk_directory=str(root / "disk"),
        disk_bytes=512 << 20,
        min_free_bytes=128 << 20,
        allow_overlay_l2=allow_overlay_l2,
    )
    path = root / "config.json"
    path.write_text(json.dumps(config))
    process = None
    clients = []
    result = dict(
        passed=False,
        mooncake_version="0.3.13",
        gpu_test=False,
        oss_test=False,
        allow_overlay_l2=allow_overlay_l2,
        runtime_root=str(repo),
        runtime_store_sha256=hashlib.sha256(
            (repo / "deploy/whitebox/cache_store.py").read_bytes()
        ).hexdigest(),
    )
    with (root / "store.log").open("w") as log:
        try:
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig

            from sglang.multimodal_gen.runtime.realtime.cache_tiers import (
                CacheTiers,
                NativeReplicas,
            )

            process = subprocess.Popen(
                [
                    sys.executable,
                    str(repo / "deploy/whitebox/cache_store.py"),
                    "--config",
                    str(path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )

            def ready():
                if process.poll() is not None:
                    raise RuntimeError("Native Store exited; inspect store.log")
                return (root / "state/store-ready.json").is_file()

            wait_for(ready, 90, "Native Store readiness")
            result["store_readiness"] = json.loads(
                (root / "state/store-ready.json").read_text()
            )
            for _ in range(2):
                client = MooncakeDistributedStore()
                rc = client.setup(
                    "127.0.0.1",
                    "P2PHANDSHAKE",
                    0,
                    32 << 20,
                    "rdma",
                    config["rdma_devices"],
                    config["master_server_address"],
                )
                if rc != 0:
                    raise RuntimeError(f"Pure client setup failed: {rc}")
                clients.append(client)
            writer, reader = clients
            policy = CacheTiers.parse(config["tiers"])
            replicas = NativeReplicas(reader, policy)

            def disk_copies(key):
                copies = reader.batch_get_replica_desc([key])[key]
                filesystem = sum(v.is_disk_replica() for v in copies)
                local = [
                    v
                    for v in copies
                    if v.is_local_disk_replica() and str(v.status).endswith("COMPLETE")
                ]
                assert filesystem == 0, "Unexpected duplicate filesystem write"
                assert len(local) <= 1, "More than one native SSD owner"
                return dict(
                    filesystem=filesystem,
                    native_ssd=len(local),
                )

            payload = b"native-cache-test" * (1 << 16)
            assert writer.put("shared-key", payload, ReplicateConfig()) == 0
            assert reader.get("shared-key") == payload
            assert replicas.memory_ready(["shared-key"])
            result["cross_client_l1"] = True
            assert replicas.wait(["shared-key"], "l2", time.monotonic() + 60)
            result["native_l2_persisted"] = True
            result["after_persistence_disk_copies"] = disk_copies("shared-key")
            assert result["after_persistence_disk_copies"]["native_ssd"] == 1
            # Only this isolated Store is pressured; never evict real player data.
            keys = ["shared-key"]
            for index in range(120):
                key = "pressure-" + str(index)
                wait_for(
                    lambda: writer.put(key, payload, ReplicateConfig()) == 0,
                    15,
                    "Native bounded eviction",
                )
                keys.append(key)

            def cold_key():
                return next(
                    (
                        k
                        for k, tiers in replicas.layers(keys).items()
                        if tiers == {"l2"}
                    ),
                    None,
                )

            cold = wait_for(cold_key, 60, "Native L2-only replica")
            started = time.monotonic()
            assert replicas.promote([cold], time.monotonic() + 60)
            assert replicas.memory_ready([cold]) and reader.get(cold) == payload
            result["after_recall_disk_copies"] = disk_copies(cold)
            promotion_ms = (time.monotonic() - started) * 1000
            result.update(native_l2_recall=True, l2_to_l1_ms=promotion_ms)
            wait_for(
                lambda: all(code in (0, -704) for code in writer.batch_remove(keys)),
                20,
                "Retire this probe's pressure keys",
            )
            # Exercise the same registered-buffer API used for real KV and
            # conditioning objects larger than the 32 MiB SDK scratch arena.
            import ctypes

            large = b"cache-large-buffer" * (3 << 20)
            source = ctypes.create_string_buffer(large)
            target = ctypes.create_string_buffer(len(large))
            source_ptr, target_ptr = ctypes.addressof(source), ctypes.addressof(target)
            assert writer.register_buffer(source_ptr, len(large)) == 0
            assert reader.register_buffer(target_ptr, len(large)) == 0
            try:
                config_large = ReplicateConfig()
                config_large.group_ids = ["native-large"]
                wait_for(
                    lambda: (
                        writer.batch_put_from_multi_buffers(
                            ["native-large"],
                            [[source_ptr]],
                            [[len(large)]],
                            config_large,
                        )
                        == [0]
                    ),
                    20,
                    "Large native buffer publication",
                )
                assert reader.batch_get_into_multi_buffers(
                    ["native-large"], [[target_ptr]], [[len(large)]]
                ) == [len(large)]
                assert target.raw == large
                result["large_registered_buffers_bytes"] = len(large)
            finally:
                writer.unregister_buffer(source_ptr)
                reader.unregister_buffer(target_ptr)
            result.update(
                native_l2_recall=True,
                l2_to_l1_ms=promotion_ms,
                passed=True,
            )
        except BaseException as error:
            result["error"] = type(error).__name__ + ": " + str(error)
            raise
        finally:
            for client in clients:
                client.close()
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                result["store_exit_code"] = process.returncode
            (root / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rdma-devices", default="")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        help="Full verified runtime checkout; defaults to this tool's repository.",
    )
    parser.add_argument(
        "--allow-overlay-l2",
        action="store_true",
        help="Explicitly validate local container overlay storage (not NVMe evidence).",
    )
    args = parser.parse_args()
    main(
        args.root,
        args.rdma_devices,
        allow_overlay_l2=args.allow_overlay_l2,
        runtime_root=args.runtime_root,
    )
