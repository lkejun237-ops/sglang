"""User-authorized retirement of the exact idle Shanghai-copy cachetiers DiT/Decoder."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy/sglang"))
import orchestrate


def main():
    directory = Path(sys.argv[1])
    state = orchestrate.load_state(directory / "state.json")
    assert state["target"]["environment"] == "test-shanghai-copy"
    cloud = orchestrate.Cloud(target=state["target"], aliyun_profile="shengdong")
    role = sys.argv[2]
    job_id, boot, endpoint = {
        "dit": (
            "dlcy8wtmze7b4tn0",
            "8a8d158bc74943d09674dd006e8568f4",
            "http://10.0.0.200:33000",
        ),
        "decoder": (
            "dlcy8wtmz7g24pgb",
            "9d6ef5d2b5464bc691d82c01bae3e4dc",
            "http://10.0.0.198:36000",
        ),
    }[role]
    process_id = f"cachetiers-shcopy-{role}-3d83-r1"
    job = cloud.get_job(job_id)
    assert str(job["UserId"]) == "205206788242572972"
    assert job["DisplayName"] == process_id
    assert job["Status"] == "Running"
    assert str(job["WorkspaceId"]) == "1453237"
    assert job["ResourceId"] == "quota3l8bnv9a35n"
    private = directory / f"retired-cachetiers-{role}-job.private.json"
    private.write_text(json.dumps(job))
    private.chmod(0o600)
    epoch = state["spec"]["pool"]["control_epoch"]
    authority = orchestrate.pool_contract().Credentials(
        os.environ["ROLE_POOL_SIGNING_KEY"], epoch
    )

    def observe():
        body = dict(control_epoch=epoch, boot_epoch=boot)
        token = authority.sign("observe", body, expires_at=time.time() + 30)
        snapshot = orchestrate.http(
            endpoint + "/managed/observe", body, token, cloud=cloud
        )
        assert snapshot.get("state") == "idle", (
            "Active service must finish before retirement"
        )
        assert snapshot.get("owner") is None, "Service still has an owner"
        assert not snapshot.get("restart_required"), "Unverified cleanup state"
        return snapshot

    before = observe()
    url = state["target"]["coordinator_url"]
    operator = os.environ["ROLE_POOL_OPERATOR_TOKEN"]
    identity = dict(process_id=process_id, boot_epoch=boot)
    orchestrate.http(url + "/v1/role-pool/drain", identity, operator, cloud=cloud)
    after = observe()
    cloud.cli("pai-dlc", "StopJob", JobId=job_id)
    for _ in range(60):
        job = cloud.get_job(job_id)
        if job["Status"] in ("Stopped", "Succeeded", "Failed"):
            break
        time.sleep(2)
    else:
        raise TimeoutError(
            "Exact cachetiers job has not stopped; do not retire registry"
        )
    receipt = f"user-authorized-cachetiers-{role}-reassignment-20261113"
    orchestrate.http(
        url + "/v1/role-pool/confirm-stopped",
        dict(
            **identity,
            stop_receipt=receipt,
            operator_id="codex-authorized-reassignment",
        ),
        operator,
        cloud=cloud,
    )
    orchestrate.http(
        url + "/v1/role-pool/retire-process",
        dict(**identity, stop_receipt=receipt),
        operator,
        cloud=cloud,
    )
    result = dict(
        job_id=job_id,
        process_id=process_id,
        boot_epoch=boot,
        before=before,
        after_drain=after,
        final_status=job["Status"],
        retired=True,
    )
    (directory / f"cachetiers-{role}-retirement.json").write_text(
        json.dumps(result, indent=2)
    )
    print("RETIRED", job_id, job["Status"], flush=True)


if __name__ == "__main__":
    main()
