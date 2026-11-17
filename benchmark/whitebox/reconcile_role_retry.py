"""Reconcile two owned deployment records after a failed PAI role retry.

Only local records change. Both records must subsequently pass the standard
inspect/selftest/attach flow. Ownership stays with the original creating run.
"""

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy/sglang"))
import orchestrate as orch


def reconcile(original_path, retry_path, role, route, compatibility, cloud, execute):
    original_path, retry_path = Path(original_path), Path(retry_path)
    original, retry = orch.load_state(original_path), orch.load_state(retry_path)
    if original_path.resolve() == retry_path.resolve():
        raise ValueError("Two distinct deployment records are required")
    for key in ("target", "source"):
        if original[key] != retry[key]:
            raise ValueError(f"Retry {key} does not match original deployment")
    if original["spec"]["model_id"] != retry["spec"]["model_id"]:
        raise ValueError("Retry model does not match original deployment")
    if original["spec"]["pool"] != retry["spec"]["pool"]:
        raise ValueError("Retry pool contract does not match original deployment")
    if any(s.get("attachment") for s in (original, retry)):
        raise ValueError("Reconciliation requires unattached deployments")
    old, replacement = original["resources"][role], retry["resources"][role]
    if old.get("reused") or replacement.get("reused"):
        raise ValueError("Failed and replacement roles must both be owned")
    if not old.get("job_id") or not replacement.get("job_id"):
        raise ValueError("Only explicit PAI job retries are supported")
    failed_job = cloud.get_job(old["job_id"])
    orch.verify_job(failed_job, old, target=original["target"])
    if failed_job["Status"] != "Failed":
        raise ValueError("Original role is not a terminal failed job")
    live_job = cloud.get_job(replacement["job_id"])
    orch.verify_job(live_job, replacement, target=retry["target"])
    if live_job["Status"] != "Running":
        raise ValueError("Replacement job is not running")
    for other in orch.ROLES:
        if other == role:
            continue
        owned = original["resources"][other]
        reused = retry["resources"][other]
        if owned.get("reused") or not reused.get("reused"):
            raise ValueError("Other roles must retain original ownership")
        # The first up may fail before persisting the surviving roles' receipts.
        # Read their exact cloud jobs; the subsequent inspect still verifies
        # source, config digest, assets and health before any attach is allowed.
        owned_receipt = owned.get("receipt") or cloud.receipt(owned, other)
        if any(
            owned_receipt[k] != reused["receipt"][k]
            for k in ("worker_id", "http", "model_id")
        ) or owned.get("job_id") != reused.get("job_id"):
            raise ValueError("Reused role does not match its owning deployment")
    receipt = replacement["receipt"]
    entry = {
        **{k: receipt[k] for k in ("worker_id", "http", "gpu")},
        "pai": True,
        "job_id": replacement["job_id"],
        "job_name": replacement["name"],
    }
    original.setdefault("replaced_resources", []).append(
        dict(
            role=role,
            resource=copy.deepcopy(old),
            terminal_status="Failed",
            replacement_owner=retry["deployment_id"],
            observed_at=orch.now(),
        )
    )
    original["spec"].setdefault("existing_services", {})[role] = entry
    original["resources"][role] = orch.reused_resource(
        original["run_id"], original["spec"]["model_id"], role, entry
    )
    for state in (original, retry):
        state["spec"]["route"] = route
        state["spec"]["pool"]["compatibility"] = compatibility
        state.pop("selftest", None)
        state["phase"] = "waiting"
        state["reconciliation"] = dict(
            original_deployment=original["deployment_id"],
            retry_deployment=retry["deployment_id"],
            ownership="retained-by-creating-run",
            observed_at=orch.now(),
        )
        orch.validate_spec(copy.deepcopy(state["spec"]), isolated=False)
    if execute:
        # Never overwrite the only recovery copy, including after partial failure.
        for path in (original_path, retry_path):
            backup = path.with_name("state.before-reconcile.json")
            if backup.exists():
                raise ValueError("Recovery backup already exists; inspect before retry")
        for path in (original_path, retry_path):
            orch.save(path.with_name("state.before-reconcile.json"), orch.read(path))
        for path, state in ((original_path, original), (retry_path, retry)):
            orch.save(path, state)
            orch.load_state(path)
            orch.write_deployment_summary(state, path.parent)
            orch.save_history_record(state, path.parent)
    return {
        s["deployment_id"]: list(orch.owned_roles(s["spec"])) for s in (original, retry)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-state", required=True)
    parser.add_argument("--retry-state", required=True)
    parser.add_argument("--role", choices=orch.ROLES, required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--compatibility", required=True)
    parser.add_argument("--aliyun-profile", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    original = orch.load_state(args.original_state)
    print(
        reconcile(
            args.original_state,
            args.retry_state,
            args.role,
            args.route,
            args.compatibility,
            orch.Cloud(target=original["target"], aliyun_profile=args.aliyun_profile),
            args.execute,
        )
    )
