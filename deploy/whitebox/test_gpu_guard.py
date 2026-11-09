"""Preserved training-allocation guards, using synthetic control-plane records."""

import copy

import pytest
from pai_gpu_guard import check_job, check_node, check_quota


def node(allocated="5", limits="5"):
    return dict(
        NodeName="approved",
        BoundQuotas=[{"QuotaId": "quota"}],
        NodeStatus="Ready",
        HealthRate=10000,
        GPUType="GX9D",
        AllocatableGPU="8",
        RequestGPU=allocated,
        LimitGPU=limits,
    )


def check(nodes):
    return check_node(
        nodes, region="ap-southeast-7", quota_id="quota", approved_node="approved"
    )


def test_free_capacity_is_total_minus_allocations_and_limits():
    result = check([node()])
    assert result["free_gpus"] == 3 and result["reservation_created"] is False
    assert result["requested_gpus"] == 3


@pytest.mark.parametrize(
    "nodes",
    [
        [],
        [node("8", "8")],
        [node("6", "6")],
        [node("5", "7")],
        [node(), node()],
        [{**node(), "HealthRate": 9900}],
        [{**node(), "BoundQuotas": []}],
    ],
)
def test_busy_unhealthy_or_ambiguous_node_fails_closed(nodes):
    with pytest.raises(ValueError):
        check(nodes)


@pytest.mark.parametrize("gpu_count", [1, 2, 3, 4])
def test_manifest_never_expands_or_moves_the_approved_allocation(gpu_count):
    job = {
        "ResourceId": "quota",
        "WorkspaceId": "384",
        "Priority": 1,
        "JobSpecs": [
            {
                "PodCount": 1,
                "ResourceConfig": {"GPU": str(gpu_count)},
                "RestartPolicy": "Never",
                "AssignNodeSpec": {"EnableAssignNode": True, "NodeNames": "approved"},
            }
        ],
    }
    kwargs = dict(quota_id="quota", approved_node="approved", gpu_count=gpu_count)
    check_job(job, **kwargs)
    for field, value in (
        ("PodCount", 2),
        ("ResourceConfig", {"GPU": str(gpu_count + 1)}),
        ("ResourceConfig", {"GPU": str(gpu_count - 1)}),
        ("AssignNodeSpec", {"EnableAssignNode": True, "NodeNames": "another"}),
        ("RestartPolicy", "OnFailure"),
    ):
        changed = copy.deepcopy(job)
        changed["JobSpecs"][0][field] = value
        with pytest.raises(ValueError):
            check_job(changed, **kwargs)
    job["EnablePriorityPreemption"] = True
    with pytest.raises(ValueError):
        check_job(job, **kwargs)


def test_changed_quota_cannot_preempt_training():
    quota = {
        "QuotaId": "quota",
        "Status": "Ready",
        "ParentQuotaId": "",
        "SubQuotas": [],
        "QuotaConfig": {"EnableSelfQuotaPreemption": False},
    }
    check_quota(quota, quota_id="quota")
    for field, value in (
        ("SubQuotas", [{"QuotaId": "child"}]),
        ("ParentQuotaId", "parent"),
        ("QuotaConfig", {"EnableSelfQuotaPreemption": True}),
    ):
        with pytest.raises(ValueError):
            check_quota({**quota, field: value}, quota_id="quota")
