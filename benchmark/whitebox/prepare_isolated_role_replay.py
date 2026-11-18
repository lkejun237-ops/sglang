"""Bind an owned, self-tested deployment to the replay suite's private authority.

No shared Coordinator is changed. The existing source-replay runner supplies its
in-memory Coordinator; native workers retain authentication and single-player
admission. Run on the authorized test controller with its pre-mounted secret.
"""

import argparse
import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deploy/sglang"))
import orchestrate  # noqa: E402


def prepare(state, output, secret):
    if (
        state.get("phase") != "ready"
        or state.get("selftest", {}).get("status") != "passed"
        or state["spec"].get("existing_services")
        or set(orchestrate.owned_roles(state["spec"])) != set(orchestrate.ROLES)
        or state["selftest"].get("source") != state["source"]
        or state["selftest"].get("worker_ids")
        != {
            role: state["resources"][role]["receipt"]["worker_id"]
            for role in orchestrate.ROLES
        }
    ):
        raise ValueError(
            "Replay activation requires three owned, currently self-tested roles"
        )
    state = copy.deepcopy(state)
    # The replay suite owns the authority in memory, not a platform route.
    for target in (state["target"], state["spec"]["target"]):
        target["coordinator_url"] = "http://127.0.0.1:9"
    configs = orchestrate.worker_configs(state)
    authority = orchestrate.pool_contract().Credentials(
        secret, state["spec"]["pool"]["control_epoch"]
    )
    output.mkdir(parents=True, exist_ok=False)
    orchestrate.save(output / "state.json", state)
    orchestrate._preflight_or_reconcile_owned_endpoints(
        state, output, configs, authority, None, in_place=True
    )
    orchestrate._attach_owned_endpoints_in_place(
        state, output, configs, authority, None
    )
    state["private_replay_authority"] = True
    orchestrate.save(output / "state.json", state)
    return {
        role: state["resources"][role]["receipt"]["http"] for role in orchestrate.ROLES
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signing-secret-file", type=Path, required=True)
    args = parser.parse_args()
    addresses = prepare(
        orchestrate.read(args.state),
        args.output,
        args.signing_secret_file.read_text().strip(),
    )
    for role, address in addresses.items():
        print(role, address)
