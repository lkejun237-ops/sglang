"""Recover only the two failed roles of the authorized Shanghai test group.

The original Decoder stays running. Failed native owners must match our smoke
allocation; an expired lease alone never authorizes stopping a process.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy/sglang"))
import orchestrate

ALLOCATION = "b4c252087f3946318ee26358afcc681b"
POD = "cachetiers-shcopy-encoder-06d-r2"
UID = "4dfb7a2b-32e8-4eae-81f4-193bfe6cad6f"
JOB = "dlc1r9083vt0f6nn"
OLD_SOURCE = "06d04b805c96ab0d3d5dbc6035659902f9c4aabc"
ROLES = {
    "encoder": (POD, "56a9167ccc8b4876aa4ea023a6eb84f6", "http://10.0.0.121:28510"),
    "dit": (
        "cachetiers-shcopy-dit-06d-r2",
        "df005f77213c43728fc8a1770020bc7a",
        "http://10.0.0.232:33000",
    ),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--role", choices=ROLES, required=True)
    p.add_argument("--retry-failed-job")
    p.add_argument(
        "--action",
        choices=("prepare-encoder", "inspect", "stop", "retire", "start"),
        required=True,
    )
    args = p.parse_args()
    state = orchestrate.load_state(args.state)
    assert state["target"]["environment"] == "test-shanghai-copy"
    cloud = orchestrate.Cloud(target=state["target"], aliyun_profile="shengdong")
    args.evidence.mkdir(parents=True, exist_ok=True)
    process, boot, endpoint = ROLES[args.role]
    identity = dict(process_id=process, boot_epoch=boot)
    credentials = orchestrate.pool_contract().Credentials(
        os.environ["ROLE_POOL_SIGNING_KEY"], state["spec"]["pool"]["control_epoch"]
    )

    def pool(operation, body):
        return orchestrate.http(
            state["target"]["coordinator_url"] + "/v1/role-pool/" + operation,
            body,
            os.environ["ROLE_POOL_OPERATOR_TOKEN"],
            cloud=cloud,
        )

    def observe_failed():
        body = dict(control_epoch=credentials.epoch, boot_epoch=boot)
        token = credentials.sign("observe", body, expires_at=time.time() + 60)
        value = orchestrate.http(
            endpoint + "/managed/observe", body, token, cloud=cloud
        )
        assert value["boot_epoch"] == boot
        assert value["state"] == "quarantined" and value["native_state"] == "failed"
        assert value["restart_required"] is True
        assert value["owner"]["allocation_id"] == ALLOCATION
        assert (
            value["owner"]["user_id"]
            == "ttff-prefix-smoke-10c4e8d887324693a0ac795f76759518"
        )
        return value

    def saved(action):
        return args.evidence / (args.role + "-" + action + ".json")

    if args.action == "prepare-encoder":
        assert args.role == "encoder"
        code = f"""
import json,hashlib,subprocess
from pathlib import Path
old=json.loads(subprocess.check_output(['k3s','kubectl','get','pod','-n','minwm-realtime',{POD!r},'-o','json']))
assert old['metadata']['uid']=={UID!r}
root=Path('/tmp/ttff-shared-recovery-20261114');root.mkdir(exist_ok=True)
(root/'pod-original.private.json').write_text(json.dumps(old))
p=json.loads(json.dumps(old));p.pop('status',None)
p['metadata']={{'name':old['metadata']['name'],'namespace':old['metadata']['namespace'],'labels':old['metadata'].get('labels',{{}})}}
p['spec']['volumes']=[v for v in p['spec']['volumes'] if not v['name'].startswith('kube-api-access-')]
for c in p['spec']['containers']:
 c['volumeMounts']=[v for v in c.get('volumeMounts',[]) if not v['name'].startswith('kube-api-access-')]
v=next(v for v in p['spec']['volumes'] if v['name']=='runtime')
assert v['hostPath']['path']=='/mnt/zing-nvme/cachetiers-h100-integration-06d-r2/runtime-source-prepare-20261114'
v['hostPath']['path']='/mnt/zing-nvme/cachetiers-h100-integration-06d-r2/runtime-dit-repair-20261114'
assert not Path(v['hostPath']['path']).exists()
assert next(e['value'] for e in p['spec']['containers'][0]['env'] if e['name']=='SGLANG_ZING_SOURCE_PREPARE')=='1'
assert p['spec']['containers'][0]['image'].endswith('@sha256:3de46a431fb0267a7bd142d4d69ba87f1f7d300890c4ae5aeb6385450b93ea1e')
path=root/'pod-06d04b805c96.private.json';path.write_text(json.dumps(p))
subprocess.check_call(['k3s','kubectl','create','--dry-run=client','-f',str(path)],stdout=subprocess.DEVNULL)
print(json.dumps(dict(manifest_path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),image=p['spec']['containers'][0]['image'],same_gpu=True)))
"""
        result = json.loads(cloud.eflo("python3 - <<'PY'\n" + code + "\nPY"))
        saved("prepare").write_text(json.dumps(result, indent=2))
    elif args.action == "inspect":
        result = dict(
            native=observe_failed(), coordinator=pool("process-status", identity)
        )
    elif args.action == "stop":
        assert saved("prepare").exists(), (
            "Prepare replacement and rollback before stopping"
        )
        # Exact owner + native failed state authorizes this recovery, not the
        # normal idle drain condition (which failed communication cannot reach).
        observe_failed()
        pool("drain", identity)
        observe_failed()
        if args.role == "encoder":
            code = f"""
import json, subprocess, time
from pathlib import Path
cmd=['k3s','kubectl']; name={POD!r}; namespace='minwm-realtime'
old=json.loads(subprocess.check_output(cmd+['get','pod','-n',namespace,name,'-o','json']))
assert old['metadata']['uid']=={UID!r}
root=Path('/tmp/ttff-shared-recovery-20261114');root.mkdir(exist_ok=True)
(root/'pod-original.private.json').write_text(json.dumps(old))
f=root/'delete-options.json'
f.write_text(json.dumps(dict(apiVersion='v1',kind='DeleteOptions',gracePeriodSeconds=45,preconditions=dict(uid={UID!r}))))
subprocess.check_call(cmd+['delete','--raw','/api/v1/namespaces/'+namespace+'/pods/'+name,'-f',str(f)],stdout=subprocess.DEVNULL)
for _ in range(55):
 value=subprocess.check_output(cmd+['get','pod','-n',namespace,name,'--ignore-not-found','-o','json']).strip()
 if not value:break
 assert json.loads(value)['metadata']['uid']=={UID!r}
 time.sleep(1)
else:raise RuntimeError('Old Pod exit is not confirmed')
print(json.dumps(dict(exit_confirmed=True,pod=name,uid={UID!r})))
"""
            result = json.loads(cloud.eflo("python3 - <<'PY'\n" + code + "\nPY"))
        else:
            job = cloud.get_job(JOB)
            assert job["UserId"] == "205206788242572972"
            assert (
                job["DisplayName"] == process
                and job["ResourceId"] == "quota3l8bnv9a35n"
            )
            assert job["Status"] == "Running"
            cloud.cli("pai-dlc", "StopJob", JobId=JOB)
            for _ in range(40):
                job = cloud.get_job(JOB)
                if job["Status"] == "Stopped":
                    break
                time.sleep(2)
            else:
                raise RuntimeError("Old DiT job exit is not confirmed")
            result = dict(exit_confirmed=True, job_id=JOB, status=job["Status"])
    elif args.action == "retire":
        assert json.loads(saved("stop").read_text())["exit_confirmed"]
        receipt = "shared-test-repair-20261114-" + args.role
        pool(
            "confirm-stopped",
            dict(**identity, stop_receipt=receipt, operator_id="208673589353288819"),
        )
        result = pool("retire-process", dict(**identity, stop_receipt=receipt))
    else:
        assert json.loads(saved("retire").read_text())["state"] == "retired"
        if saved("start").exists():
            previous = json.loads(saved("start").read_text())
            assert args.role == "dit" and args.retry_failed_job == previous["job_id"], (
                "Replacement already submitted; inspect it before retrying"
            )
            assert cloud.get_job(args.retry_failed_job)["Status"] == "Failed"
            # The old retired epoch must still be registered: the failed retry
            # never entered the service pool and has no native owner to release.
            assert pool("process-status", identity)["process"]["state"] == "retired"
            saved("start").rename(
                args.evidence / ("dit-start-" + args.retry_failed_job + ".json")
            )
        prepared = json.loads(saved("prepare").read_text())
        if args.role == "encoder":
            path = prepared["manifest_path"]
            assert re.fullmatch(
                r"/tmp/ttff-shared-recovery-20261114/pod-[a-f0-9]{12}\.private\.json",
                path,
            )
            code = f"""
import json,hashlib,subprocess
from pathlib import Path
p=Path({path!r});assert hashlib.sha256(p.read_bytes()).hexdigest()=={prepared["sha256"]!r}
result=json.loads(subprocess.check_output(['k3s','kubectl','create','-f',str(p),'-o','json']))
assert result['metadata']['uid']!={UID!r}
print(json.dumps(dict(pod=result['metadata']['name'],uid=result['metadata']['uid'])))
"""
            result = json.loads(cloud.eflo("python3 - <<'PY'\n" + code + "\nPY"))
        else:
            body_path = args.evidence / "dit-create.private.json"
            assert (
                hashlib.sha256(body_path.read_bytes()).hexdigest() == prepared["sha256"]
            )
            assert cloud.get_job(JOB)["Status"] == "Stopped"
            body = json.loads(body_path.read_text())
            assert (
                len(body["JobSpecs"]) == 1
                and body["JobSpecs"][0]["ResourceConfig"]["GPU"] == "1"
            )
            result = dict(job_id=cloud.submit_job(body), replaced_job_id=JOB)
    saved(args.action).write_text(json.dumps(result, indent=2))
    print("DONE", args.role, args.action, flush=True)


if __name__ == "__main__":
    main()
