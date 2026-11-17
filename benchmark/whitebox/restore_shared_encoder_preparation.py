"""Explicitly authorized, idle-only preparation fix for the Shanghai test Encoder.

Keep the existing image, model plan and GPU reservation. Preserve the old runtime
directory and use a fresh output directory because the launcher requires one.
Each step writes evidence before proceeding; never infer process exit from TTL.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy/sglang"))
import orchestrate

POD = "cachetiers-shcopy-encoder-06d-r2"
UID = "6c41414a-834f-48ce-abf8-595282ba50cf"
BOOT = "f2e740228e964deca6208f7fcfbbfd69"
ROOT = "/tmp/ttff-encoder-preparation-20261114"
RECEIPT = "authorized-encoder-preparation-20261114"


def main():
    state = orchestrate.load_state(Path(sys.argv[1]) / "state.json")
    evidence = Path(sys.argv[2])
    evidence.mkdir(exist_ok=True)
    action = sys.argv[3]
    assert state["target"]["environment"] == "test-shanghai-copy"
    cloud = orchestrate.Cloud(target=state["target"], aliyun_profile="shengdong")
    identity = dict(process_id=POD, boot_epoch=BOOT)
    operator = os.environ["ROLE_POOL_OPERATOR_TOKEN"]

    def pool(operation, body):
        return orchestrate.http(
            state["target"]["coordinator_url"] + "/v1/role-pool/" + operation,
            body,
            operator,
            cloud=cloud,
        )

    def node(code):
        return cloud.eflo("python3 - <<'PY'\n" + code + "\nPY\n")

    if action == "prepare":
        result = node(f"""
import json, hashlib, subprocess
from pathlib import Path
r=Path({ROOT!r}); p=json.loads((r/'pod-preparation.private.json').read_text())
assert hashlib.sha256((r/'pod-preparation.private.json').read_bytes()).hexdigest() == 'aa1238f6dc29d944746300f02f7dbe247793311adde623958f3bcfa0a895dccd'
old=json.loads(subprocess.check_output(['k3s','kubectl','get','pod','-n','minwm-realtime',{POD!r},'-o','json']))
assert old['metadata']['uid']=={UID!r}
assert p['spec']['containers'][0]['image']==old['spec']['containers'][0]['image']
assert p['spec']['containers'][0]['command']==old['spec']['containers'][0]['command']
v=next(v for v in p['spec']['volumes'] if v['name']=='runtime')
assert v['hostPath']['path']=='/mnt/zing-nvme/cachetiers-h100-integration-06d-r2/runtime'
v['hostPath']['path']='/mnt/zing-nvme/cachetiers-h100-integration-06d-r2/runtime-source-prepare-20261114'
assert not Path(v['hostPath']['path']).exists()
raw=json.dumps(p,indent=2); (r/'pod-final.private.json').write_text(raw)
subprocess.check_call(['k3s','kubectl','create','--dry-run=client','-f',str(r/'pod-final.private.json')],stdout=subprocess.DEVNULL)
print(json.dumps(dict(prepared=True,uid={UID!r},manifest_sha256=hashlib.sha256(raw.encode()).hexdigest(),environment={{'SGLANG_ZING_SOURCE_PREPARE':'1'}},runtime=v['hostPath']['path'])))
""")
    elif action == "drain":
        authority = orchestrate.pool_contract().Credentials(
            os.environ["ROLE_POOL_SIGNING_KEY"], state["spec"]["pool"]["control_epoch"]
        )

        def observe():
            body = dict(control_epoch=authority.epoch, boot_epoch=BOOT)
            token = authority.sign("observe", body, expires_at=time.time() + 30)
            value = orchestrate.http(
                "http://10.0.0.121:28510/managed/observe", body, token, cloud=cloud
            )
            assert value.get("state") == "idle" and value.get("owner") is None
            assert not value.get("restart_required")
            return value

        before = observe()
        pool("drain", identity)
        after = observe()
        status = pool("process-status", identity)
        assert status.get("safe_to_stop") is True
        result = json.dumps(dict(before=before, after=after, status=status))
    elif action == "stop":
        assert (evidence / "drain.json").exists()
        status = pool("process-status", identity)
        assert status.get("safe_to_stop") is True
        result = node(f"""
import json,subprocess,time
from pathlib import Path
r=Path({ROOT!r}); cmd=['k3s','kubectl']; name={POD!r}
p=json.loads(subprocess.check_output(cmd+['get','pod','-n','minwm-realtime',name,'-o','json']))
assert p['metadata']['uid']=={UID!r}
options=dict(apiVersion='v1',kind='DeleteOptions',gracePeriodSeconds=45,preconditions=dict(uid={UID!r}))
f=r/'delete-options.json'; f.write_text(json.dumps(options))
subprocess.check_call(cmd+['delete','--raw','/api/v1/namespaces/minwm-realtime/pods/'+name,'-f',str(f)],stdout=subprocess.DEVNULL)
for _ in range(55):
 value=subprocess.check_output(cmd+['get','pod','-n','minwm-realtime',name,'--ignore-not-found','-o','json']).strip()
 if not value: break
 assert json.loads(value)['metadata']['uid']=={UID!r}
 time.sleep(1)
else: raise RuntimeError('Old Pod deletion is not confirmed')
print(json.dumps(dict(pod=name,deleted_uid={UID!r},exit_confirmed=True)))
""")
    elif action == "retire":
        assert json.loads((evidence / "stop.json").read_text())["exit_confirmed"]
        pool(
            "confirm-stopped",
            dict(**identity, stop_receipt=RECEIPT, operator_id="208673589353288819"),
        )
        result = json.dumps(
            pool("retire-process", dict(**identity, stop_receipt=RECEIPT))
        )
    elif action == "start":
        assert json.loads((evidence / "retire.json").read_text())["state"] == "retired"
        expected = json.loads((evidence / "prepare.json").read_text())[
            "manifest_sha256"
        ]
        result = node(f"""
import json,hashlib,subprocess
from pathlib import Path
p=Path({ROOT!r})/'pod-final.private.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()=={expected!r}
d=json.loads(subprocess.check_output(['k3s','kubectl','create','-f',str(p),'-o','json']))
assert d['metadata']['uid']!={UID!r}
print(json.dumps(dict(pod=d['metadata']['name'],uid=d['metadata']['uid'],created=True)))
""")
    else:
        raise ValueError("Unknown recovery step")
    value = json.loads(result)
    (evidence / (action + ".json")).write_text(json.dumps(value, indent=2))
    print("DONE", action, flush=True)


if __name__ == "__main__":
    main()
