"""Node-side H100 selection. Caller holds the allocation lock through Pod creation.

Uses Python 3.6 syntax for the existing node; only selects a physically idle UUID
not claimed by any nonterminal NVIDIA Pod, including Pending Pods.
"""

import json
import subprocess


def select(gpus, processes, pods):
    claimed = set(processes)
    for pod in pods["items"]:
        if pod.get("status", {}).get("phase") in ("Succeeded", "Failed"):
            continue
        if pod.get("spec", {}).get("nodeName") != "world-model-b27mys7b-0000":
            continue
        for container in pod["spec"].get("containers", []) + pod["spec"].get(
            "initContainers", []
        ):
            env = {v["name"]: v.get("value") for v in container.get("env", [])}
            visible = env.get("NVIDIA_VISIBLE_DEVICES")
            if visible in ("void", "none"):
                continue
            if visible in ("all", "") or (
                "NVIDIA_VISIBLE_DEVICES" in env and visible is None
            ):
                raise ValueError("Existing Pod has ambiguous GPU allocation")
            if visible:
                # Numeric indices cannot be reliably compared with UUIDs here.
                if any(not item.startswith("GPU-") for item in visible.split(",")):
                    raise ValueError("Existing Pod uses a non-UUID GPU allocation")
                claimed.update(visible.split(","))
            elif pod["spec"].get("runtimeClassName") == "nvidia":
                raise ValueError("Existing NVIDIA Pod does not declare its allocation")
    for gpu in gpus:
        if gpu["memory"] == 0 and gpu["uuid"] not in claimed:
            return gpu["uuid"]
    raise ValueError("No unclaimed idle GPU is available")


def main():
    def read(argv):
        return subprocess.check_output(argv, universal_newlines=True)

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
    print(select(gpus, processes, pods))


if __name__ == "__main__":
    main()
