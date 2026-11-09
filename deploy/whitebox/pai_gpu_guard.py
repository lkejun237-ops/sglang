"""Read-only PAI admission guard for a user-approved, idle GPU node.

This tool never creates/stops a job or changes quota/node/training state. Its
receipt is an observation, not a reservation: the submitter must query again
immediately before CreateJob and still let PAI allocate devices exclusively.
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone


def check_quota(quota, *, quota_id):
    """Do not modify scheduling policy; reject a changed preemption boundary."""
    config = quota.get("QuotaConfig", {})
    if (
        quota.get("QuotaId") != quota_id
        or quota.get("Status") != "Ready"
        or config.get("EnableSelfQuotaPreemption") is not False
        or quota.get("ParentQuotaId")
        or quota.get("SubQuotas") != []
    ):
        raise ValueError("Quota preemption boundary changed; stop deployment")


def check_node(nodes, *, region, quota_id, approved_node, gpu_count=3):
    if (
        region != "ap-southeast-7"
        or type(gpu_count) is not int
        or gpu_count not in (1, 2, 3, 4)
    ):
        raise ValueError("This rollout supports one to four approved GPUs in Thailand")
    matches = [node for node in nodes if node.get("NodeName") == approved_node]
    if len(matches) != 1:
        raise ValueError("The approved node is absent or ambiguous; stop deployment")
    node = matches[0]
    if quota_id not in {q.get("QuotaId") for q in node.get("BoundQuotas", [])}:
        raise ValueError("The approved node is not in the selected quota")
    if node.get("NodeStatus") != "Ready" or node.get("HealthRate") != 10000:
        raise ValueError("The approved node is not healthy; stop deployment")
    if node.get("GPUType") != "GX9D":
        raise ValueError("The approved GPU resource type changed")
    # PAI's AllocatableGPU is total allocatable, not unused capacity. Never use
    # utilization percentage or low memory usage to steal a training allocation.
    total, allocated, limits = (
        int(node[key]) for key in ("AllocatableGPU", "RequestGPU", "LimitGPU")
    )
    if min(total, allocated, limits) < 0:
        raise ValueError("Invalid GPU accounting")
    free = total - max(allocated, limits)
    if free < gpu_count:
        raise ValueError(
            f"Fewer than {gpu_count} unallocated GPUs remain; wait for free capacity"
        )
    return {
        "region": region,
        "quota_id": quota_id,
        "node": approved_node,
        "total_gpus": total,
        "allocated_gpus": allocated,
        "limit_gpus": limits,
        "free_gpus": free,
        "requested_gpus": gpu_count,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "reservation_created": False,
        "hardware_runtime_check": "required_after_scheduler_allocation",
    }


def check_job(job, *, quota_id, approved_node, gpu_count=3):
    """Validate a prepared PAI request, without copying credentials into receipts."""
    if type(gpu_count) is not int or gpu_count not in (1, 2, 3, 4):
        raise ValueError(
            "An explicit one-, two-, three- or four-GPU allocation is required"
        )
    if job.get("ResourceId") != quota_id or job.get("WorkspaceId") != "384":
        raise ValueError("GPU quota/workspace changed")
    # CreateJob does not document a top-level EnablePriorityPreemption flag.
    # Use its documented lowest priority and the independently checked quota;
    # never assume an unsupported request field disables preemption.
    if (
        job.get("Priority") != 1
        or job.get("EnablePriorityPreemption") is True
        or job.get("Settings", {}).get("EnablePriorityPreemption") is True
    ):
        raise ValueError("The inference job must not preempt training")
    specs = job.get("JobSpecs", [])
    if len(specs) != 1:
        raise ValueError("Exactly one GPU worker pod is allowed")
    spec = specs[0]
    if spec.get("PodCount") != 1 or str(
        spec.get("ResourceConfig", {}).get("GPU")
    ) != str(gpu_count):
        raise ValueError(
            "The inference job must use exactly the approved GPU allocation"
        )
    node = spec.get("AssignNodeSpec", {})
    if (
        node.get("EnableAssignNode") is not True
        or node.get("NodeNames") != approved_node
    ):
        raise ValueError("The job must stay on the exact approved node")
    if spec.get("RestartPolicy") != "Never" or job.get("ElasticSpec"):
        raise ValueError("Automatic restart/scaling is not approved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-node", required=True)
    parser.add_argument("--quota-id", required=True)
    parser.add_argument("--profile", default="ali-ak")
    parser.add_argument("--gpu-count", type=int, choices=(1, 2, 3, 4), default=3)
    args = parser.parse_args()
    quota = subprocess.run(
        [
            "aliyun",
            "paistudio",
            "GetQuota",
            "--profile",
            args.profile,
            "--region",
            "ap-southeast-7",
            "--endpoint",
            "pai.ap-southeast-7.aliyuncs.com",
            "--QuotaId",
            args.quota_id,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    check_quota(json.loads(quota.stdout), quota_id=args.quota_id)
    # Explicit endpoint is essential: old CLI endpoint maps can silently query
    # another region even when --region is supplied.
    result = subprocess.run(
        [
            "aliyun",
            "paistudio",
            "ListNodes",
            "--profile",
            args.profile,
            "--region",
            "ap-southeast-7",
            "--endpoint",
            "pai.ap-southeast-7.aliyuncs.com",
            "--QuotaId",
            args.quota_id,
            "--NodeNames",
            args.approved_node,
            "--Verbose",
            "true",
            "--PageSize",
            "100",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    print(
        json.dumps(
            check_node(
                json.loads(result.stdout)["Nodes"],
                region="ap-southeast-7",
                quota_id=args.quota_id,
                approved_node=args.approved_node,
                gpu_count=args.gpu_count,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
