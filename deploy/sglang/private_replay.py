"""Native drain checks for owned replay workers without a shared Coordinator.

The ephemeral replay authority has no process ledger after the suite exits.
Use each worker's signed identity and permanent drain flag instead. Cloud stop
and exact Job/Pod checks remain in orchestrate.down; this module never deletes.
"""

import json
import shlex
from pathlib import PurePosixPath


class PrivateReplayControl:
    def __init__(self, state, cloud):
        self.state, self.cloud = state, cloud
        roles = {"encoder", "dit", "decoder"}
        resources = state["resources"]
        if (
            state.get("private_replay_authority") is not True
            or state["target"]["coordinator_url"] != "http://127.0.0.1:9"
            or state["spec"]["target"]["coordinator_url"] != "http://127.0.0.1:9"
            or set(resources) != roles
            or set(state.get("attachment", {}).get("identities", {})) != roles
            or state["spec"].get("existing_services")
            or any(r.get("reused") for r in resources.values())
            or not resources["encoder"].get("uid")
        ):
            raise ValueError("Private replay stop requires three owned isolated roles")
        for role, resource in resources.items():
            cfg = resource["receipt"].get("worker_config", {})
            identity = state["attachment"]["identities"][role]
            if (
                cfg.get("coordinator_url") != "http://127.0.0.1:9"
                or cfg.get("service_count") != 1
                or identity.get("process_id") != resource["receipt"]["worker_id"]
                or identity.get("service_id") != identity["process_id"] + "/0"
                or not identity.get("boot_epoch")
                or state["attachment"].get("endpoint_intents", {}).get(role)
            ):
                raise ValueError("Private replay worker identity is incomplete")

    def command(self, role, action):
        if action not in {"identity", "observe", "drain"}:
            raise ValueError("Unsupported private replay lifecycle command")
        enc = self.state["resources"]["encoder"]
        resource = self.state["resources"][role]
        body = dict(control_epoch=self.state["spec"]["pool"]["control_epoch"])
        if action != "identity":
            body["boot_epoch"] = self.state["attachment"]["identities"][role][
                "boot_epoch"
            ]
        # The existing controller Pod can read its mounted secret. Neither the
        # secret nor the scoped bearer token is returned to the operator host.
        package = PurePosixPath(enc["receipt"]["runtime"]["sglang"]["path"]).parent
        contract = package / "multimodal_gen/runtime/realtime/role_pool/contracts.py"
        secret = self.state["spec"]["roles"]["encoder"]["secret_files"][
            "ROLE_POOL_SIGNING_KEY"
        ]
        params = dict(
            contract=str(contract),
            secret=secret,
            body=body,
            action=action,
            url=resource["receipt"]["http"] + "/managed/" + action,
        )
        code = """import json,runpy,pathlib,time,urllib.request
p=json.loads(PARAMS)
c=runpy.run_path(p['contract'])['Credentials'](pathlib.Path(p['secret']).read_text().strip(),p['body']['control_epoch'])
until=time.time()+30
if p['action']=='drain': p['body']['executor_until']=until
token=c.sign(p['action'],p['body'],expires_at=until)
r=urllib.request.Request(p['url'],data=json.dumps(p['body']).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+token})
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs): raise ValueError('Redirect forbidden')
with urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect()).open(r,timeout=10) as response:
 print(json.dumps(json.load(response)))
""".replace("PARAMS", repr(json.dumps(params)))
        return json.loads(
            self.cloud.eflo(
                "kubectl -n "
                + shlex.quote(enc["namespace"])
                + " exec "
                + shlex.quote(enc["name"])
                + " -- /opt/sglang/bin/python -c "
                + shlex.quote(code)
            )
        )

    def verify_idle(self, role, *, drain=False):
        resource = self.state["resources"][role]
        expected = self.state["attachment"]["identities"][role]
        identity = self.command(role, "identity")
        cfg = resource["receipt"]["worker_config"]
        service = identity.get("service", {})
        if (
            identity.get("boot_epoch") != expected["boot_epoch"]
            or identity.get("control_epoch") != cfg["control_epoch"]
            or service.get("worker_id") != expected["process_id"]
            or service.get("service_id") != expected["service_id"]
            or service.get("role") != role
            or any(
                identity.get(k) != cfg[k]
                for k in ("capacity", "profiles", "profile_limits")
            )
            or any(
                identity.get("deployment", {}).get(k) != cfg[k]
                for k in ("pool_id", "compatibility", "network_domain", "hardware_ids")
            )
        ):
            raise ValueError("Private replay process identity changed before stop")
        snapshot = self.command(role, "drain" if drain else "observe")
        if (
            snapshot.get("boot_epoch") != expected["boot_epoch"]
            or snapshot.get("service_id") != expected["service_id"]
            or snapshot.get("state") != "idle"
            or snapshot.get("native_state") != "idle"
            or snapshot.get("owner") is not None
            or snapshot.get("restart_required") is not False
            or snapshot.get("children")
            or (drain and snapshot.get("draining") is not True)
        ):
            raise ValueError("Private replay worker is not safely idle")
        return snapshot
