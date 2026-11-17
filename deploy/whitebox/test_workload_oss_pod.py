import copy

import pytest
from workload_oss_pod import render


@pytest.fixture
def pod():
    return {
        "metadata": {"uid": "test-uid", "name": "old", "namespace": "test"},
        "spec": {
            "nodeName": "test-node",
            "nodeSelector": {"kubernetes.io/hostname": "test-node"},
            "priorityClassName": "test-no-preempt",
            "priority": -100,
            "restartPolicy": "Never",
            "containers": [
                {
                    "name": "encoder",
                    "image": "test/encoder@sha256:" + "1" * 64,
                    "env": [{"name": "NVIDIA_VISIBLE_DEVICES", "value": "GPU-test"}],
                    "volumeMounts": [
                        {"name": "sts", "mountPath": "/mnt/.alibabacloud"},
                        {"name": "runtime", "mountPath": "/runtime"},
                        {"name": "kube-api-access-old", "mountPath": "/var/run/kube"},
                    ],
                }
            ],
            "volumes": [
                {"name": "sts", "secret": {"secretName": "old-sts"}},
                {"name": "runtime", "hostPath": {"path": "/test/old"}},
                {"name": "kube-api-access-old", "projected": {}},
            ],
        },
    }


def candidate(pod, directory="/test/new"):
    return render(
        pod,
        expected_uid="test-uid",
        gpu="GPU-test",
        name="new",
        secret="test-oss",
        runtime_directory=directory,
    )


def test_candidate_uses_secret_reference_and_preserves_placement(pod):
    before = copy.deepcopy(pod)
    result = candidate(pod)
    assert pod == before
    spec = result["spec"]
    c = spec["containers"][0]
    assert c["image"] == before["spec"]["containers"][0]["image"]
    assert "nodeName" not in spec and "priority" not in spec
    assert spec["nodeSelector"] == before["spec"]["nodeSelector"]
    assert spec["priorityClassName"] == "test-no-preempt"
    assert {v["name"] for v in spec["volumes"]} == {"runtime", "sts"}
    assert spec["volumes"][0]["secret"]["secretName"] == "test-oss"
    assert c["volumeMounts"][0] == {
        "name": "sts",
        "mountPath": "/var/run/whitebox-oss",
        "readOnly": True,
    }
    assert {v["name"]: v["value"] for v in c["env"]} == {
        "NVIDIA_VISIBLE_DEVICES": "GPU-test",
        "AWS_SHARED_CREDENTIALS_FILE": "/var/run/whitebox-oss/credentials",
        "AWS_EC2_METADATA_DISABLED": "true",
    }


def test_reject_unexpected_provider_or_live_runtime(pod):
    with pytest.raises(ValueError, match="separate sibling directory"):
        candidate(pod, "/test/old")
    pod["spec"]["containers"][0]["env"].append(
        {"name": "AWS_PROFILE", "value": "other"}
    )
    with pytest.raises(ValueError, match="requires review"):
        candidate(pod)
